from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

from .domain import ServiceState, TerminalServiceConfig
from .runtime import (
    ProcessRecord,
    RuntimeStore,
    command_hash,
    proc_executable,
    proc_start_ticks,
    record_matches_process,
)
from .service_base import ServiceStatus


class TerminalService:
    """Launch an interactive CLI inside the user's desktop terminal."""

    def __init__(
        self,
        config: TerminalServiceConfig,
        runtime_store: RuntimeStore,
        model_id: str | None = None,
        launch_mode: str | None = None,
    ):
        self.config = config
        self.runtime_store = runtime_store
        self.model_id = model_id
        self.launch_mode = launch_mode

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def display_name(self) -> str:
        return self.config.display_name

    def _owned_record(self) -> ProcessRecord | None:
        record = self.runtime_store.load_process(self.config.id)
        if record is None:
            return None
        if record_matches_process(record):
            return record
        self.runtime_store.delete(self.config.id)
        return None

    def status(self) -> ServiceStatus:
        record = self._owned_record()
        if record is not None:
            return ServiceStatus(
                ServiceState.READY,
                "interactive terminal is running",
                record.pid,
            )
        return ServiceStatus(ServiceState.STOPPED, "no launcher-owned terminal")

    def start(self) -> ServiceStatus:
        current = self.status()
        if current.state == ServiceState.READY:
            record = self._owned_record()
            if record is not None and record.command_sha256 != command_hash(self.config.command):
                raise RuntimeError(
                    "Launch settings changed; stop this service before applying them."
                )
            return current

        if shutil.which(self.config.terminal_executable) is None:
            raise FileNotFoundError(
                f"Terminal executable not found: {self.config.terminal_executable}"
            )
        if not self.config.executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.config.executable}")
        if not self.config.cwd.is_dir():
            raise FileNotFoundError(f"Working directory not found: {self.config.cwd}")

        env = os.environ.copy()
        env.update(self.config.environment)

        command = list(self.config.command)
        process = subprocess.Popen(
            command,
            cwd=self.config.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

        if self.config.startup_grace:
            deadline = time.monotonic() + self.config.startup_grace
            while time.monotonic() < deadline:
                code = process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"{self.config.display_name} terminal exited during startup with code {code}"
                    )
                time.sleep(0.05)

        try:
            record = ProcessRecord(
                service_id=self.config.id,
                pid=process.pid,
                pgid=os.getpgid(process.pid),
                sid=os.getsid(process.pid),
                proc_start_ticks=proc_start_ticks(process.pid),
                executable=proc_executable(process.pid),
                command_sha256=command_hash(command),
                model_id=self.model_id,
                launch_mode=self.launch_mode,
            )
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            raise RuntimeError(
                f"Could not establish safe terminal identity for {self.config.display_name}"
            ) from exc

        if not (record.pid == record.pgid == record.sid):
            try:
                process.terminate()
            finally:
                raise RuntimeError(
                    f"Unsafe terminal ownership for {self.config.display_name}: "
                    f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
                )

        self.runtime_store.save_process(record)
        return ServiceStatus(
            ServiceState.READY,
            "interactive terminal launched",
            record.pid,
        )

    def stop(self) -> ServiceStatus:
        record = self.runtime_store.load_process(self.config.id)
        if record is None:
            return ServiceStatus(ServiceState.STOPPED, "no launcher-owned terminal")

        if not record_matches_process(record):
            self.runtime_store.delete(self.config.id)
            return ServiceStatus(ServiceState.STOPPED, "stale terminal record removed")

        if not (record.pid == record.pgid == record.sid):
            raise RuntimeError(
                "Refusing group stop because terminal ownership is unsafe: "
                f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
            )

        os.killpg(record.pgid, signal.SIGTERM)
        deadline = time.monotonic() + self.config.stop_timeout
        while time.monotonic() < deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                return ServiceStatus(ServiceState.STOPPED, "terminal stopped gracefully")
            time.sleep(0.1)

        os.killpg(record.pgid, signal.SIGKILL)
        hard_deadline = time.monotonic() + 2.0
        while time.monotonic() < hard_deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                return ServiceStatus(ServiceState.STOPPED, "terminal forced closed")
            time.sleep(0.1)

        raise RuntimeError(
            f"Failed to stop {self.config.display_name}; process group {record.pgid} remains"
        )

    def logs(self, lines: int = 40) -> list[str]:
        return [
            "Interactive terminal service: output is displayed in the terminal window."
        ]

    def open(self) -> bool:
        return self.status().state == ServiceState.READY
