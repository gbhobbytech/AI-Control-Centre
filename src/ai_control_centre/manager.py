from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace

from .config import AppConfig
from .docker_service import DockerService
from .domain import DockerServiceConfig, ProcessServiceConfig, TerminalServiceConfig, ServiceState
from .models import render_model_args, smallest_complete_model
from .process_service import ProcessService
from .terminal_service import TerminalService
from .runtime import RuntimeStore, command_hash
from .tuning import apply_values, is_llama_service, model_values
from .service_base import ServiceController, ServiceStatus

_ACTIVE_STATES = {
    ServiceState.READY,
    ServiceState.RUNNING_NOT_READY,
    ServiceState.EXTERNAL,
}


@dataclass(frozen=True)
class PlannedStop:
    service_id: str
    display_name: str
    model: str | None = None  # display name of the loaded model, if known

    def describe(self) -> str:
        return f"{self.display_name} (model {self.model})" if self.model else self.display_name


@dataclass(frozen=True)
class SwitchPlan:
    """What starting a task or service would stop first, in plain terms."""

    target: str
    reason: str  # "conflict": shares the GPU or another resource; "model": model must change
    stops: tuple[PlannedStop, ...]  # in the order they would be stopped
    external: tuple[str, ...] = ()  # display names AI Control Centre will not stop

    @property
    def blocked(self) -> bool:
        return bool(self.external)

    def describe(self) -> str:
        if self.blocked:
            names = ", ".join(self.external)
            return (
                f"{self.target} cannot start: {names} "
                f"{'was' if len(self.external) == 1 else 'were'} started outside AI Control Centre "
                f"and {'is' if len(self.external) == 1 else 'are'} using a resource {self.target} needs. "
                "AI Control Centre will not stop programs it did not start. "
                "Close it yourself, then try again."
            )
        if self.reason == "model":
            lead = f"{self.target} needs a different model or launch settings than the one running."
        else:
            lead = f"{self.target} needs the GPU (or another shared resource) that running services are using."
        items = "\n".join(f"  - {stop.describe()}" for stop in self.stops)
        return f"{lead}\n\nThese will be stopped first, in this order:\n{items}\n\nThen {self.target} will start."


class SwitchRequired(RuntimeError):
    """Starting needs other services stopped first. The message keeps the CLI wording."""

    def __init__(self, message: str, plan: SwitchPlan):
        super().__init__(message)
        self.plan = plan


class ConflictError(SwitchRequired):
    pass


class ModelChangeError(SwitchRequired):
    pass


class ServiceStartError(RuntimeError):
    """A service failed to start or become usable. Names the service and its log."""

    def __init__(self, message: str, service_id: str, log_path=None):
        super().__init__(message)
        self.service_id = service_id
        self.log_path = log_path


class TransitionError(RuntimeError):
    """A switch stopped services and then could not complete."""

    def __init__(self, message: str, stopped: tuple[str, ...] = ()):
        super().__init__(message)
        self.stopped = stopped


class ServiceManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.runtime_store = RuntimeStore(config.settings.runtime_dir)
        self.services: dict[str, ServiceController] = {}

        for service_id, service in config.services.items():
            if isinstance(service, TerminalServiceConfig):
                controller: ServiceController = TerminalService(
                    service, self.runtime_store
                )
            elif isinstance(service, ProcessServiceConfig):
                controller = ProcessService(
                    service, self.runtime_store, config.settings.log_dir
                )
            elif isinstance(service, DockerServiceConfig):
                controller = DockerService(service, self.runtime_store)
            else:  # pragma: no cover
                raise TypeError(
                    f"Unsupported service config for {service_id}: {type(service)!r}"
                )
            self.services[service_id] = controller

    def get(self, service_id: str) -> ServiceController:
        try:
            return self.services[service_id]
        except KeyError as exc:
            raise KeyError(f"Unknown service: {service_id}") from exc

    def running_model_id(self, service_id: str) -> str | None:
        record = self.runtime_store.load_process(service_id)
        return record.model_id if record is not None else None

    def running_launch_mode(self, service_id: str) -> str | None:
        record = self.runtime_store.load_process(service_id)
        return record.launch_mode if record is not None else None

    @staticmethod
    def _replace_cli_flag(args: tuple[str, ...], flag: str, value: str) -> tuple[str, ...]:
        items = list(args)
        try:
            index = items.index(flag)
        except ValueError:
            items.extend([flag, value])
        else:
            if index + 1 >= len(items):
                items.append(value)
            else:
                items[index + 1] = value
        return tuple(items)

    def _prompt_helper_model_id(self) -> str | None:
        workshop = self.config.settings.prompt_workshop
        if workshop.model_strategy == "smallest":
            model = smallest_complete_model(self.config.models)
            return model.id if model is not None else None
        if workshop.preferred_model is not None:
            return workshop.preferred_model
        service = self.config.services.get(workshop.service)
        if isinstance(service, ProcessServiceConfig):
            return service.default_model
        return None

    def prompt_helper_processing_mode(self) -> str:
        workshop = self.config.settings.prompt_workshop
        running_mode = self.running_launch_mode(workshop.service)
        if running_mode in {"cpu", "gpu"}:
            return running_mode
        if workshop.processing_mode != "auto":
            return workshop.processing_mode
        for service_id, service in self.config.services.items():
            if service_id == workshop.service or not service.gpu:
                continue
            try:
                status = self.get(service_id).status()
            except Exception:
                continue
            if status.state in _ACTIVE_STATES:
                return "cpu"
        return "gpu"

    def _controller_for_start(
        self,
        service_id: str,
        requested_model: str | None,
    ) -> ServiceController:
        config = self.config.services[service_id]
        if not isinstance(config, ProcessServiceConfig) or not config.uses_model:
            return self.get(service_id)

        model_id = requested_model or config.default_model
        if model_id is None and service_id == self.config.settings.prompt_workshop.service:
            model_id = self._prompt_helper_model_id()
        if model_id is None:
            raise RuntimeError(f"{config.display_name} requires a model selection")
        try:
            model = self.config.models[model_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown model: {model_id}") from exc
        if not model.complete:
            raise RuntimeError(
                f"Model '{model_id}' is incomplete: {model.warning or 'missing files'}"
            )

        main_llama = service_id != self.config.settings.prompt_workshop.service and is_llama_service(config)
        if main_llama and not model.tuning_reviewed:
            raise RuntimeError(
                f"Review launch settings for '{model.display_name}' under Preferences > Settings > Model tuning before starting. "
                "Choose values for this computer, or import them from a running local server."
            )
        render_model = replace(model, **model_values(model)) if main_llama else model
        try:
            rendered_args = render_model_args(config.args, render_model, config.model_defaults)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        health = config.health
        if main_llama:
            rendered_args = apply_values(rendered_args, model)
            if health is not None and model.startup_timeout_seconds is not None:
                health = replace(health, startup_timeout=model.startup_timeout_seconds)

        effective_gpu = config.gpu
        launch_mode: str | None = None
        if service_id == self.config.settings.prompt_workshop.service:
            workshop = self.config.settings.prompt_workshop
            mode = self.prompt_helper_processing_mode()
            launch_mode = mode
            if mode == "cpu":
                layers = 0
            else:
                layers = workshop.gpu_layers
                if layers is None:
                    layers = model.recommended_gpu_layers
                if layers is None:
                    layers = 999
            rendered_args = self._replace_cli_flag(rendered_args, "-ngl", str(layers))
            effective_gpu = layers > 0

        effective = replace(config, args=rendered_args, gpu=effective_gpu, health=health)
        if isinstance(effective, TerminalServiceConfig):
            return TerminalService(
                effective,
                self.runtime_store,
                model_id=model_id,
                launch_mode=launch_mode,
            )
        return ProcessService(
            effective,
            self.runtime_store,
            self.config.settings.log_dir,
            model_id=model_id,
            launch_mode=launch_mode,
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

    def dependency_order(self, requested: Iterable[str]) -> list[str]:
        """Return dependency-safe startup order without exposing orchestration internals."""
        return self._dependency_order(requested)

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

    def _display(self, service_id: str) -> str:
        return self.config.services[service_id].display_name

    def _planned_stop(self, service_id: str) -> PlannedStop:
        model_id = self.running_model_id(service_id)
        model = self.config.models.get(model_id) if model_id else None
        return PlannedStop(service_id, self._display(service_id), model.display_name if model else model_id)

    def _stop_verified(self, service_id: str) -> ServiceStatus:
        """Stop a service and confirm it has really stopped before anything else starts."""
        result = self.get(service_id).stop()
        after = self.get(service_id).status()
        if result.state != ServiceState.STOPPED or after.state in _ACTIVE_STATES:
            raise TransitionError(
                f"{self._display(service_id)} did not stop ({after.state.value}: {after.detail}), "
                "so nothing else was started. Check its logs, then try again.",
            )
        return result

    def _requested_conflict_groups(
        self, requested: Iterable[str]
    ) -> dict[str, list[str]]:
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
        target: str | None = None,
    ) -> list[tuple[str, ServiceStatus]]:
        requested_order = self._dependency_order(requested)
        requested_set = set(requested_order)
        groups = self._requested_conflict_groups(requested_order)

        internal = {group: ids for group, ids in groups.items() if len(ids) > 1}
        if internal:
            detail = ", ".join(
                f"{group}: {', '.join(ids)}" for group, ids in sorted(internal.items())
            )
            raise RuntimeError(
                f"Requested services contain an internal resource conflict ({detail})"
            )

        blockers: set[str] = set()

        # A resident Prompt Helper may have been GPU-offloaded in GPU/Auto mode.
        # Treat it as a potential blocker when a normal GPU workload is requested;
        # explicit CPU mode remains non-conflicting.
        workshop = self.config.settings.prompt_workshop
        helper_id = workshop.service
        requests_gpu = any(self.config.services[sid].gpu for sid in requested_order)
        if (
            requests_gpu
            and helper_id in self.config.services
            and helper_id not in requested_set
            and workshop.processing_mode in {"gpu", "auto"}
        ):
            helper_status = self.get(helper_id).status()
            helper_mode = self.running_launch_mode(helper_id)
            if helper_status.state == ServiceState.EXTERNAL or (
                helper_status.state in _ACTIVE_STATES and helper_mode == "gpu"
            ):
                blockers.add(helper_id)

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

        external = [
            service_id
            for service_id, status in active_transition
            if status.state == ServiceState.EXTERNAL
        ]
        active_ids = {service_id for service_id, _ in active_transition}
        stop_order = [sid for sid in reversed(self._dependency_order(transition_set)) if sid in active_ids]
        plan = SwitchPlan(
            target=target or ", ".join(self._display(sid) for sid in requested_order),
            reason="conflict",
            stops=tuple(self._planned_stop(sid) for sid in stop_order if sid not in external),
            external=tuple(self._display(sid) for sid in sorted(external)),
        )
        if external:
            names = ", ".join(sorted(external))
            raise ConflictError(
                "Resource conflict involves externally started service(s) "
                f"{names}; refusing to stop them automatically",
                plan,
            )

        if not stop_conflicts:
            names = ", ".join(sorted(service_id for service_id, _ in active_transition))
            raise ConflictError(
                f"Resource conflict with running launcher-owned service(s): {names}. "
                "Retry with --stop-conflicts to stop them in dependency-safe order.",
                plan,
            )

        results: list[tuple[str, ServiceStatus]] = []
        for service_id in stop_order:
            status = self.get(service_id).status()
            if status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                results.append((service_id, self._stop_verified(service_id)))
            elif status.state == ServiceState.EXTERNAL:
                raise TransitionError(
                    f"{service_id} became externally owned during conflict resolution; aborting transition",
                    tuple(sid for sid, _ in results),
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
                raise ModelChangeError(
                    f"{self.config.services[service_id].display_name} is externally started; "
                    "cannot safely change its model",
                    SwitchPlan(
                        target=self._display(service_id), reason="model", stops=(),
                        external=(self._display(service_id),),
                    ),
                )
            if status.state not in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                continue
            running_model = self.running_model_id(service_id)
            desired = self._controller_for_start(service_id, requested_model)
            record = self.runtime_store.load_process(service_id)
            changed_args = (isinstance(desired, ProcessService) and record is not None
                            and record.command_sha256 != command_hash(desired.config.command))
            if running_model != requested_model or changed_args:
                mismatched.append(service_id)

        if not mismatched:
            return []

        names = ", ".join(
            f"{self.config.services[sid].display_name} (model or launch settings changed: {requested_model})"
            for sid in mismatched
        )
        transition_set = self._dependent_closure(mismatched)
        active_transition: list[tuple[str, ServiceStatus]] = []
        for service_id in transition_set:
            status = self.get(service_id).status()
            if status.state in _ACTIVE_STATES:
                active_transition.append((service_id, status))

        external = [
            service_id
            for service_id, status in active_transition
            if status.state == ServiceState.EXTERNAL
        ]
        active_ids = {service_id for service_id, _ in active_transition}
        stop_order = [sid for sid in reversed(self._dependency_order(transition_set)) if sid in active_ids]
        requested_model_obj = self.config.models.get(requested_model) if requested_model else None
        plan = SwitchPlan(
            target=f"{', '.join(self._display(sid) for sid in mismatched)} with "
                   f"{requested_model_obj.display_name if requested_model_obj else requested_model}",
            reason="model",
            stops=tuple(self._planned_stop(sid) for sid in stop_order if sid not in external),
            external=tuple(self._display(sid) for sid in sorted(external)),
        )
        if external:
            external_names = ", ".join(sorted(external))
            raise ModelChangeError(
                "Model transition depends on externally started service(s) "
                f"{external_names}; refusing to stop them automatically",
                plan,
            )
        if not replace_model:
            raise ModelChangeError(
                f"Model change required for {names}. "
                "Retry with --replace-model to stop affected launcher-owned services and reload safely.",
                plan,
            )

        results: list[tuple[str, ServiceStatus]] = []
        for service_id in stop_order:
            status = self.get(service_id).status()
            if status.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
                results.append((service_id, self._stop_verified(service_id)))
        return results

    def start_service(
        self,
        service_id: str,
        *,
        stop_conflicts: bool = False,
        replace_model: bool = False,
        model_id: str | None = None,
        after_stops: Callable[[list[tuple[str, ServiceStatus]]], None] | None = None,
    ) -> list[tuple[str, ServiceStatus]]:
        requested_order = self._dependency_order([service_id])
        target = self.config.services[service_id]
        # A harness/service without its own model should reuse a running model
        # dependency. Starting a harness must not silently swap the active LLM
        # because of an unrelated model default.
        if model_id is None and not (
            isinstance(target, ProcessServiceConfig) and target.uses_model
        ):
            active_models = {
                self.running_model_id(sid)
                for sid in requested_order
                if sid != service_id
                and isinstance(self.config.services[sid], ProcessServiceConfig)
                and self.config.services[sid].uses_model
                and self.get(sid).status().state in {
                    ServiceState.READY, ServiceState.RUNNING_NOT_READY
                }
            }
            active_models.discard(None)
            if len(active_models) > 1:
                raise RuntimeError(
                    "Dependencies use different running models; select the intended model explicitly."
                )
            if active_models:
                model_id = next(iter(active_models))
            else:
                # When no dependency model is running, use the selected Agent
                # profile's model for its harness rather than the LLM service's
                # possibly unrelated default model.
                agent = self.config.profiles.get("agent")
                if agent is not None and agent.harness is not None:
                    harness = self.config.harnesses.get(agent.harness)
                    if harness is not None and service_id in harness.services:
                        model_id = agent.selected_model
        # Validate model selection/settings before stopping any conflicting workload.
        for sid in requested_order:
            self._controller_for_start(sid, model_id)
        results = self._prepare_conflicts(
            requested_order, stop_conflicts=stop_conflicts, target=self._display(service_id)
        )
        results.extend(
            self._prepare_model_switch(
                requested_order,
                model_id,
                replace_model=replace_model,
            )
        )
        self._after_stops(results, after_stops)
        with self._reporting_switch(results):
            for item in requested_order:
                status = self.get(item).status()
                if status.state == ServiceState.EXTERNAL:
                    results.append((item, status))
                    continue
                controller = self._controller_for_start(item, model_id)
                result = self._start_controller(item, controller)
                results.append((item, result))
                if result.state != ServiceState.READY and not (
                    result.state == ServiceState.RUNNING_NOT_READY and controller.config.health is None
                ):
                    raise ServiceStartError(
                        f"{item} failed to become usable: {result.detail}", item, getattr(controller, "log_path", None)
                    )
        return results

    @staticmethod
    def _start_controller(service_id: str, controller) -> ServiceStatus:
        """Start one service; failures name the service (same message text as before)."""
        try:
            return controller.start()
        except (ServiceStartError, SwitchRequired, TransitionError):
            raise
        except (OSError, RuntimeError) as exc:
            raise ServiceStartError(str(exc), service_id, getattr(controller, "log_path", None)) from exc

    @staticmethod
    def _after_stops(results, callback) -> None:
        if callback is not None and any(status.state == ServiceState.STOPPED for _, status in results):
            callback(list(results))

    @contextmanager
    def _reporting_switch(self, results):
        """Make a failure after stopping services say what was already stopped."""
        stopped = tuple(
            self._display(sid) for sid, status in results if status.state == ServiceState.STOPPED
        )
        try:
            yield
        except TransitionError:
            raise
        except Exception as exc:
            if not stopped:
                raise
            raise TransitionError(
                f"{exc}\n\nAlready stopped for this switch: {', '.join(stopped)}. "
                "It was not restarted automatically.",
                stopped,
            ) from exc

    def stop_service(self, service_id: str) -> ServiceStatus:
        return self.get(service_id).stop()

    def _profile_services(self, profile) -> tuple[str, ...]:
        services = list(profile.services)
        if profile.harness:
            harness = self.config.harnesses.get(profile.harness)
            if harness is None:
                raise RuntimeError(f"Profile '{profile.display_name}' refers to unknown harness '{profile.harness}'")
            for service_id in harness.services:
                if service_id not in services:
                    services.append(service_id)
        return tuple(services)

    def _profile_open_service(self, profile) -> str | None:
        if profile.harness:
            harness = self.config.harnesses.get(profile.harness)
            if harness and harness.open_service:
                return harness.open_service
        return profile.open_service

    def start_profile(
        self,
        profile_id: str,
        *,
        open_interface: bool = True,
        stop_conflicts: bool = False,
        replace_model: bool = False,
        model_id: str | None = None,
        after_stops: Callable[[list[tuple[str, ServiceStatus]]], None] | None = None,
    ) -> list[tuple[str, ServiceStatus]]:
        try:
            profile = self.config.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile: {profile_id}") from exc

        selected_model = model_id or profile.selected_model
        requested_order = self._dependency_order(self._profile_services(profile))
        for sid in requested_order:
            self._controller_for_start(sid, selected_model)
        results = self._prepare_conflicts(
            requested_order, stop_conflicts=stop_conflicts, target=profile.display_name
        )
        results.extend(
            self._prepare_model_switch(
                requested_order,
                selected_model,
                replace_model=replace_model,
            )
        )
        self._after_stops(results, after_stops)

        with self._reporting_switch(results):
            for service_id in requested_order:
                current = self.get(service_id).status()
                if current.state == ServiceState.EXTERNAL:
                    results.append((service_id, current))
                    continue
                controller = self._controller_for_start(service_id, selected_model)
                result = self._start_controller(service_id, controller)
                results.append((service_id, result))
                if result.state != ServiceState.READY and not (
                    result.state == ServiceState.RUNNING_NOT_READY and controller.config.health is None
                ):
                    raise ServiceStartError(
                        f"Profile '{profile.display_name}' stopped because {service_id} failed: {result.detail}",
                        service_id,
                        getattr(controller, "log_path", None),
                    )

        open_service = self._profile_open_service(profile)
        if open_interface and open_service:
            self.get(open_service).open()
        return results

    def stop_profile(self, profile_id: str) -> list[tuple[str, ServiceStatus]]:
        try:
            profile = self.config.profiles[profile_id]
        except KeyError as exc:
            raise KeyError(f"Unknown profile: {profile_id}") from exc

        results: list[tuple[str, ServiceStatus]] = []
        for service_id in reversed(self._dependency_order(self._profile_services(profile))):
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
