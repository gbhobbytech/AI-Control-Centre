from __future__ import annotations

import json
import subprocess
import time
import webbrowser
from dataclasses import dataclass

from .domain import DockerServiceConfig, ServiceState
from .health import HealthResult, check_http, wait_until_ready
from .runtime import DockerRecord, RuntimeStore
from .service_base import ServiceStatus


@dataclass(frozen=True)
class ContainerState:
    container_id: str
    name: str
    running: bool
    status: str
    started_at: str


class DockerService:
    def __init__(self, config: DockerServiceConfig, runtime_store: RuntimeStore):
        self.config = config
        self.runtime_store = runtime_store

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def display_name(self) -> str:
        return self.config.display_name

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self.config.docker_executable, *args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=check,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Docker executable not found: {self.config.docker_executable}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"docker {' '.join(args)} failed: {detail}") from exc

    def _inspect(self) -> ContainerState:
        result = self._run("inspect", self.config.container_name)
        try:
            raw = json.loads(result.stdout)
            item = raw[0]
            state = item["State"]
            return ContainerState(
                container_id=str(item["Id"]),
                name=str(item["Name"]).lstrip("/"),
                running=bool(state["Running"]),
                status=str(state["Status"]),
                started_at=str(state.get("StartedAt", "")),
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Could not parse docker inspect output for {self.config.container_name}"
            ) from exc

    def _health(self) -> HealthResult | None:
        if self.config.health is None:
            return None
        return check_http(self.config.health)

    def _owned_record(self, state: ContainerState | None = None) -> DockerRecord | None:
        record = self.runtime_store.load_docker(self.config.id)
        if record is None:
            return None
        if state is None:
            state = self._inspect()
        if (
            state.running
            and record.container_name == state.name
            and record.container_id == state.container_id
            and record.started_at == state.started_at
        ):
            return record
        self.runtime_store.delete(self.config.id)
        return None

    def status(self) -> ServiceStatus:
        state = self._inspect()
        record = self._owned_record(state)
        health = self._health() if state.running else None

        if state.running and record is not None:
            if health is None:
                return ServiceStatus(ServiceState.RUNNING_NOT_READY, f"container running ({state.status})")
            if health.ok:
                return ServiceStatus(ServiceState.READY, health.detail)
            return ServiceStatus(ServiceState.RUNNING_NOT_READY, health.detail)

        if state.running:
            if health is not None and health.ok:
                return ServiceStatus(
                    ServiceState.EXTERNAL,
                    f"container running but not launcher-owned: {health.detail}",
                )
            return ServiceStatus(
                ServiceState.EXTERNAL,
                f"container running but not launcher-owned ({state.status})",
            )

        return ServiceStatus(ServiceState.STOPPED, f"container {state.status}")

    def start(self) -> ServiceStatus:
        current = self.status()
        if current.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
            return current
        if current.state == ServiceState.EXTERNAL:
            raise RuntimeError(
                f"{self.config.display_name} is already running externally; refusing to claim ownership"
            )

        self._run("start", self.config.container_name)
        state = self._inspect()
        if not state.running:
            raise RuntimeError(
                f"Docker reported start success but {self.config.container_name} is not running"
            )

        record = DockerRecord(
            service_id=self.config.id,
            container_name=state.name,
            container_id=state.container_id,
            started_at=state.started_at,
        )
        self.runtime_store.save_docker(record)

        if self.config.health is None:
            return ServiceStatus(ServiceState.RUNNING_NOT_READY, "container started; no health check configured")

        result = wait_until_ready(self.config.health, lambda: self._inspect().running)
        if result.ok:
            return ServiceStatus(ServiceState.READY, result.detail)
        return ServiceStatus(ServiceState.ERROR, result.detail)

    def stop(self) -> ServiceStatus:
        state = self._inspect()
        record = self.runtime_store.load_docker(self.config.id)

        if not state.running:
            self.runtime_store.delete(self.config.id)
            return ServiceStatus(ServiceState.STOPPED, f"container {state.status}")

        if record is None or self._owned_record(state) is None:
            raise RuntimeError(
                f"{self.config.display_name} is externally started; refusing to stop it automatically"
            )

        timeout = max(1, int(round(self.config.stop_timeout)))
        self._run("stop", "--time", str(timeout), self.config.container_name)

        deadline = time.monotonic() + self.config.stop_timeout + 3.0
        while time.monotonic() < deadline:
            state = self._inspect()
            if not state.running:
                self.runtime_store.delete(self.config.id)
                return ServiceStatus(ServiceState.STOPPED, f"container {state.status}")
            time.sleep(0.2)

        raise RuntimeError(f"Failed to stop Docker container {self.config.container_name}")

    def logs(self, lines: int = 40) -> list[str]:
        count = max(lines, 0)
        result = self._run("logs", "--tail", str(count), self.config.container_name, check=False)
        merged = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        return merged.splitlines() if merged else []

    def open(self) -> bool:
        if not self.config.open_url:
            return False
        return bool(webbrowser.open(self.config.open_url))
