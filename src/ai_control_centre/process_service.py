from __future__ import annotations

import os
import signal
import subprocess
import time
import webbrowser
from pathlib import Path

from .domain import ProcessServiceConfig, ServiceState
from .health import HealthResult, check_http, wait_until_ready
from .runtime import (
    ProcessRecord,
    RuntimeStore,
    command_hash,
    proc_executable,
    proc_start_ticks,
    record_matches_process,
)
from .service_base import ServiceStatus


class ProcessService:
    def __init__(
        self,
        config: ProcessServiceConfig,
        runtime_store: RuntimeStore,
        log_dir: Path,
        model_id: str | None = None,
        launch_mode: str | None = None,
    ):
        self.config = config
        self.runtime_store = runtime_store
        self.log_dir = log_dir
        self.model_id = model_id
        self.launch_mode = launch_mode
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def display_name(self) -> str:
        return self.config.display_name

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.config.id}.log"

    def _owned_record(self) -> ProcessRecord | None:
        record = self.runtime_store.load_process(self.config.id)
        if record is None:
            return None
        if record_matches_process(record):
            return record
        self.runtime_store.delete(self.config.id)
        return None

    def _health(self) -> HealthResult | None:
        if self.config.health is None:
            return None
        return check_http(self.config.health)

    def status(self) -> ServiceStatus:
        record = self._owned_record()
        health = self._health()

        if record is not None:
            if health is None:
                return ServiceStatus(ServiceState.RUNNING_NOT_READY, "owned process is running", record.pid)
            if health.ok:
                return ServiceStatus(ServiceState.READY, health.detail, record.pid)
            return ServiceStatus(ServiceState.RUNNING_NOT_READY, health.detail, record.pid)

        if health is not None and health.ok:
            return ServiceStatus(
                ServiceState.EXTERNAL,
                f"health check passed but process is not launcher-owned: {health.detail}",
            )

        return ServiceStatus(ServiceState.STOPPED, health.detail if health else "no launcher-owned process")

    def _rotate_log_if_needed(self) -> None:
        path = self.log_path
        if not path.exists() or path.stat().st_size <= self.config.log_max_bytes:
            return
        rotated = path.with_suffix(path.suffix + ".1")
        try:
            rotated.unlink()
        except FileNotFoundError:
            pass
        path.replace(rotated)

    def start(self) -> ServiceStatus:
        current = self.status()
        if current.state in {ServiceState.READY, ServiceState.RUNNING_NOT_READY}:
            if self.model_id is not None:
                record = self._owned_record()
                running_model = record.model_id if record is not None else None
                if running_model != self.model_id:
                    raise RuntimeError(
                        f"{self.config.display_name} is already running with model "
                        f"{running_model or 'unknown'}; requested {self.model_id}. "
                        "Stop the service before changing models."
                    )
            record = self._owned_record()
            if record is not None and record.command_sha256 != command_hash(self.config.command):
                raise RuntimeError('Launch settings changed; stop this service before applying them.')
            if current.state == ServiceState.RUNNING_NOT_READY and self.config.health is not None:
                result = wait_until_ready(self.config.health, lambda: self._owned_record() is not None)
                return ServiceStatus(ServiceState.READY if result.ok else ServiceState.ERROR, result.detail, current.pid)
            return current
        if current.state == ServiceState.EXTERNAL:
            raise RuntimeError(
                f"{self.config.display_name} appears to be running externally; refusing to claim ownership"
            )

        if not self.config.executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.config.executable}")
        if not self.config.cwd.is_dir():
            raise FileNotFoundError(f"Working directory not found: {self.config.cwd}")

        self._rotate_log_if_needed()
        env = os.environ.copy()
        env.update(self.config.environment)

        log_handle = self.log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                list(self.config.command),
                cwd=self.config.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            log_handle.close()

        deadline = time.monotonic() + 2.0
        record: ProcessRecord | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{self.config.display_name} exited immediately with code {process.returncode}; see {self.log_path}"
                )
            try:
                record = ProcessRecord(
                    service_id=self.config.id,
                    pid=process.pid,
                    pgid=os.getpgid(process.pid),
                    sid=os.getsid(process.pid),
                    proc_start_ticks=proc_start_ticks(process.pid),
                    executable=proc_executable(process.pid),
                    command_sha256=command_hash(self.config.command),
                    model_id=self.model_id,
                    launch_mode=self.launch_mode,
                )
                break
            except (FileNotFoundError, ProcessLookupError, OSError):
                time.sleep(0.02)

        if record is None:
            process.terminate()
            raise RuntimeError(f"Could not establish safe process identity for {self.config.display_name}")

        if not (record.pid == record.pgid == record.sid):
            try:
                process.terminate()
            finally:
                raise RuntimeError(
                    f"Unsafe process ownership for {self.config.display_name}: "
                    f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
                )

        self.runtime_store.save_process(record)

        if self.config.health is None:
            return ServiceStatus(
                ServiceState.RUNNING_NOT_READY,
                "process started; no health check configured",
                record.pid,
            )

        result = wait_until_ready(self.config.health, lambda: process.poll() is None)
        if result.ok:
            return ServiceStatus(ServiceState.READY, result.detail, record.pid)

        code = process.poll()
        detail = result.detail
        if code is not None:
            self.runtime_store.delete(self.config.id)
            detail += f" (exit code {code})"
        detail += f"; see {self.log_path}"
        return ServiceStatus(ServiceState.ERROR, detail, record.pid)

    def stop(self) -> ServiceStatus:
        record = self.runtime_store.load_process(self.config.id)
        if record is None:
            status = self.status()
            if status.state == ServiceState.EXTERNAL:
                raise RuntimeError(
                    f"{self.config.display_name} is externally started; refusing to stop it automatically"
                )
            return ServiceStatus(ServiceState.STOPPED, "no launcher-owned process")

        if not record_matches_process(record):
            self.runtime_store.delete(self.config.id)
            status = self.status()
            if status.state == ServiceState.EXTERNAL:
                raise RuntimeError(
                    "Saved ownership record is stale and a service is still externally reachable; not stopping it"
                )
            return ServiceStatus(ServiceState.STOPPED, "stale ownership record removed")

        if not (record.pid == record.pgid == record.sid):
            raise RuntimeError(
                f"Refusing group stop because ownership is unsafe: "
                f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
            )

        os.killpg(record.pgid, signal.SIGTERM)
        deadline = time.monotonic() + self.config.stop_timeout
        while time.monotonic() < deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                return ServiceStatus(ServiceState.STOPPED, "stopped gracefully")
            time.sleep(0.1)

        os.killpg(record.pgid, signal.SIGKILL)
        hard_deadline = time.monotonic() + 2.0
        while time.monotonic() < hard_deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                return ServiceStatus(ServiceState.STOPPED, "forced stop after graceful timeout")
            time.sleep(0.1)

        raise RuntimeError(f"Failed to stop {self.config.display_name}; process group {record.pgid} remains")

    def logs(self, lines: int = 40) -> list[str]:
        if not self.log_path.exists():
            return [f"No log exists yet: {self.log_path}"]
        content = self.log_path.read_text(errors="replace").splitlines()
        return content[-max(lines, 0) :]

    def open(self) -> bool:
        if not self.config.open_url:
            return False
        return bool(webbrowser.open(self.config.open_url))
