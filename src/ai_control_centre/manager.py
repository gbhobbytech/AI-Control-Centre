from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from .config import AppConfig
from .docker_service import DockerService
from .domain import DockerServiceConfig, ProcessServiceConfig, ServiceState
from .models import render_model_args
from .process_service import ProcessService
from .runtime import RuntimeStore
from .service_base import ServiceController, ServiceStatus


_ACTIVE_STATES = {
    ServiceState.READY,
    ServiceState.RUNNING_NOT_READY,
    ServiceState.EXTERNAL,
}


class ServiceManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.runtime_store = RuntimeStore(config.settings.runtime_dir)
        self.services: dict[str, ServiceController] = {}

        for service_id, service in config.services.items():
            if isinstance(service, ProcessServiceConfig):
                controller: ServiceController = ProcessService(
                    service, self.runtime_store, config.settings.log_dir
                )
            elif isinstance(service, DockerServiceConfig):
                controller = DockerService(service, self.runtime_store)
            else:  # pragma: no cover
                raise TypeError(f"Unsupported service config for {service_id}: {type(service)!r}")
            self.services[service_id] = controller

    def get(self, service_id: str) -> ServiceController:
        try:
            return self.services[service_id]
        except KeyError as exc:
            raise KeyError(f"Unknown service: {service_id}") from exc

    def running_model_id(self, service_id: str) -> str | None:
        record = self.runtime_store.load_process(service_id)
        return record.model_id if record is not None else None

    def _controller_for_start(
        self,
        service_id: str,
        requested_model: str | None,
    ) -> ServiceController:
        config = self.config.services[service_id]
        if not isinstance(config, ProcessServiceConfig) or not config.uses_model:
            return self.get(service_id)

        model_id = requested_model or config.default_model
        if model_id is None:
            raise RuntimeError(f"{config.display_name} requires a model selection")
        try:
            model = self.config.models[model_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown model: {model_id}") from exc
        if not model.complete:
            raise RuntimeError(f"Model '{model_id}' is incomplete: {model.warning or 'missing files'}")

        try:
            rendered_args = render_model_args(config.args, model, config.model_defaults)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        effective = replace(config, args=rendered_args)
        return ProcessService(
            effective,
            self.runtime_store,
            self.config.settings.log_dir,
            model_id=model_id,
        )

    def _dependency_order(self, requested: Iterable[str]) -> list[str]:
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(service_id: str) -> None:
            if service_id in visited:
                return
            if service_id in visiting:
                raise RuntimeError(f"Dependency cycle involving {service_id}")
            if service_id not in self.config.services:
                raise KeyError(f"Unknown service: {service_id}")
            visiting.add(service_id)
            for dep in self.config.services[service_id].dependencies:
                visit(dep)
            visiting.remove(service_id)
            visited.add(service_id)
            order.append(service_id)

        for service_id in requested:
            visit(service_id)
        return order

    def _dependent_closure(self, roots: Iterable[str]) -> set[str]:
        roots = set(roots)
        closure = set(roots)
        changed = True
        while changed:
            changed = False
            for service_id, config in self.config.services.items():
                if service_id in closure:
                    continue
                if any(dep in closure for dep in config.dependencies):
                    closure.add(service_id)
                    changed = True
        return closure

    def _requested_conflict_groups(self, requested: Iterable[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for service_id in requested:
            for group in self.config.services[service_id].conflict_groups:
                groups.setdefault(group, []).append(service_id)
        return groups

    def _prepare_conflicts(
        self,
        requested: Iterable[str],
        *,
        stop_conflicts: bool,
    ) -> list[tuple[str, ServiceStatus]]:
        requested_order = self._dependency_order(requested)
        requested_set = set(requested_order)
        groups = self._requested_conflict_groups(requested_order)

        internal = {group: ids for group, ids in groups.items() if len(ids) > 1}
        if internal:
            detail = ", ".join(f"{group}: {', '.join(ids)}" for group, ids in sorted(internal.items()))
            raise RuntimeError(f"Requested services contain an internal resource conflict ({detail})")

        blockers: set[str] = set()
        for service_id, config in self.config.services.items():
            if service_id in requested_set:
                continue
            if not set(config.conflict_groups).intersection(groups):
                continue
            status = self.get(service_id).status()
            if status.state in _ACTIVE_STATES:
                blockers.add(service_id)

        if not blockers:
            return []

        transition_set = self._dependent_closure(blockers) - requested_set
        active_transition: list[tuple[str, ServiceStatus]] = []
        for service_id in transition_set:
            status = self.get(service_id).status()
            if status.state in _ACTIVE_STATES:
                active_transition.append((service_id, status))

        external = [service_id for service_id, status in active_transition if status.state == ServiceState.EXTERNAL]
        if external:
            names = ", ".join(sorted(external))
            raise RuntimeError(
                "Resource conflict involves externally started service(s) "
                f"{names}; refusing to stop them automatically"
            )

        if not stop_conflicts:
            names = ", ".join(sorted(service_id for service_id, _ in active_transition))
            raise RuntimeError(
                f"Resource conflict with running launcher-owned service(s): {names}. "
                "Retry with --stop-conflicts to stop them in dependency-safe order."
            )

        results: list[tuple[str, ServiceStatus]] = []
        ordered = self._dependency_order(transition_set)
        for service_id in reversed(ordered):
            status = self.get(service_id).status()
            if status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                results.append((service_id, self.get(service_id).stop()))
            elif status.state == ServiceState.EXTERNAL:
                raise RuntimeError(
                    f"{service_id} became externally owned during conflict resolution; aborting transition"
                )
        return results

    def _prepare_model_switch(
        self,
        requested: Iterable[str],
        requested_model: str | None,
        *,
        replace_model: bool,
    ) -> list[tuple[str, ServiceStatus]]:
        if requested_model is None:
            return []

        requested_order = self._dependency_order(requested)
        model_services = [
            service_id
            for service_id in requested_order
            if isinstance(self.config.services[service_id], ProcessServiceConfig)
            and self.config.services[service_id].uses_model
        ]
        mismatched: list[str] = []

        for service_id in model_services:
            status = self.get(service_id).status()
            if status.state == ServiceState.EXTERNAL:
                raise RuntimeError(
                    f"{self.config.services[service_id].display_name} is externally started; "
                    "cannot safely change its model"
                )
            if status.state not in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                continue
            running_model = self.running_model_id(service_id)
            if running_model != requested_model:
                mismatched.append(service_id)

        if not mismatched:
            return []

        names = ", ".join(
            f"{self.config.services[sid].display_name} ({self.running_model_id(sid) or 'unknown'} -> {requested_model})"
            for sid in mismatched
        )
        if not replace_model:
            raise RuntimeError(
                f"Model change required for {names}. "
                "Retry with --replace-model to stop affected launcher-owned services and reload safely."
            )

        transition_set = self._dependent_closure(mismatched)
        active_transition: list[tuple[str, ServiceStatus]] = []
        for service_id in transition_set:
            status = self.get(service_id).status()
            if status.state in _ACTIVE_STATES:
                active_transition.append((service_id, status))

        external = [service_id for service_id, status in active_transition if status.state == ServiceState.EXTERNAL]
        if external:
            names = ", ".join(sorted(external))
            raise RuntimeError(
                "Model transition depends on externally started service(s) "
                f"{names}; refusing to stop them automatically"
            )

        results: list[tuple[str, ServiceStatus]] = []
        ordered = self._dependency_order(transition_set)
        for service_id in reversed(ordered):
            status = self.get(service_id).status()
            if status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                results.append((service_id, self.get(service_id).stop()))
        return results

    def start_service(
        self,
        service_id: str,
        *,
        stop_conflicts: bool = False,
        replace_model: bool = False,
        model_id: str | None = None,
    ) -> list[tuple[str, ServiceStatus]]:
        requested_order = self._dependency_order([service_id])
        results = self._prepare_conflicts(requested_order, stop_conflicts=stop_conflicts)
        results.extend(
            self._prepare_model_switch(
                requested_order,
                model_id,
                replace_model=replace_model,
            )
        )
        for item in requested_order:
            status = self.get(item).status()
            if status.state == ServiceState.EXTERNAL:
                results.append((item, status))
                continue
            controller = self._controller_for_start(item, model_id)
            result = controller.start()
            results.append((item, result))
            if result.state not in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                raise RuntimeError(f"{item} failed to become usable: {result.detail}")
        return results

    def stop_service(self, service_id: str) -> ServiceStatus:
        return self.get(service_id).stop()

    def start_profile(
        self,
        profile_id: str,
        *,
        open_interface: bool = True,
        stop_conflicts: bool = False,
        replace_model: bool = False,
        model_id: str | None = None,
    ) -> list[tuple[str, ServiceStatus]]:
        try:
            profile = self.config.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile: {profile_id}") from exc

        selected_model = model_id or profile.selected_model
        requested_order = self._dependency_order(profile.services)
        results = self._prepare_conflicts(requested_order, stop_conflicts=stop_conflicts)
        results.extend(
            self._prepare_model_switch(
                requested_order,
                selected_model,
                replace_model=replace_model,
            )
        )

        for service_id in requested_order:
            current = self.get(service_id).status()
            if current.state == ServiceState.EXTERNAL:
                results.append((service_id, current))
                continue
            controller = self._controller_for_start(service_id, selected_model)
            result = controller.start()
            results.append((service_id, result))
            if result.state not in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                raise RuntimeError(
                    f"Profile '{profile.display_name}' stopped because {service_id} failed: {result.detail}"
                )

        if open_interface and profile.open_service:
            self.get(profile.open_service).open()
        return results

    def stop_profile(self, profile_id: str) -> list[tuple[str, ServiceStatus]]:
        try:
            profile = self.config.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile: {profile_id}") from exc

        results: list[tuple[str, ServiceStatus]] = []
        for service_id in reversed(self._dependency_order(profile.services)):
            status = self.get(service_id).status()
            if status.state == ServiceState.EXTERNAL:
                results.append((service_id, status))
                continue
            results.append((service_id, self.get(service_id).stop()))
        return results

    def stop_all(self) -> list[tuple[str, ServiceStatus]]:
        results: list[tuple[str, ServiceStatus]] = []
        for service_id in reversed(self._dependency_order(self.services.keys())):
            service = self.get(service_id)
            status = service.status()
            if status.state == ServiceState.EXTERNAL:
                results.append((service_id, status))
                continue
            results.append((service_id, service.stop()))
        return results
