from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .domain import ServiceState, TerminalServiceConfig
from .runtime import (
    ProcessRecord,
    RuntimeStore,
    command_hash,
    proc_cmdline,
    proc_executable,
    proc_start_ticks,
    record_matches_process,
)
from .service_base import ServiceStatus


class TerminalService:
    """Launch an interactive CLI in a desktop terminal and track the CLI process."""

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

    @property
    def _runner_state_path(self) -> Path:
        return self.runtime_store.directory / f"{self.config.id}.terminal.json"

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
                "interactive process is running",
                record.pid,
            )
        return ServiceStatus(ServiceState.STOPPED, "not running")

    def _read_runner_identity(self) -> tuple[int, int, int] | None:
        try:
            raw = json.loads(self._runner_state_path.read_text(encoding="utf-8"))
            return int(raw["pid"]), int(raw["pgid"]), int(raw["sid"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def start(self) -> ServiceStatus:
        current = self.status()
        if current.state == ServiceState.READY:
            return current

        if shutil.which(self.config.terminal_executable) is None:
            raise FileNotFoundError(
                f"Terminal executable not found: {self.config.terminal_executable}"
            )
        if not self.config.executable.exists():
            raise FileNotFoundError(f"Executable not found: {self.config.executable}")
        if not self.config.cwd.is_dir():
            raise FileNotFoundError(f"Working directory not found: {self.config.cwd}")

        try:
            self._runner_state_path.unlink()
        except FileNotFoundError:
            pass

        env = os.environ.copy()
        env.update(self.config.environment)

        target_command = [str(self.config.executable), *self.config.args]
        runner_command = [
            sys.executable,
            "-m",
            "ai_control_centre.terminal_runner",
            "--state-file",
            str(self._runner_state_path),
            "--",
            *target_command,
        ]
        terminal_command = [
            self.config.terminal_executable,
            *self.config.terminal_args,
            *runner_command,
        ]

        launcher = subprocess.Popen(
            terminal_command,
            cwd=self.config.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

        deadline = time.monotonic() + max(self.config.startup_grace, 0.25) + 4.0
        identity: tuple[int, int, int] | None = None
        while time.monotonic() < deadline:
            identity = self._read_runner_identity()
            if identity is not None:
                break
            code = launcher.poll()
            if code is not None and code != 0:
                raise RuntimeError(
                    f"{self.config.display_name} terminal launcher exited with code {code}"
                )
            time.sleep(0.05)

        if identity is None:
            code = launcher.poll()
            detail = (
                f"terminal launcher exited with code {code}"
                if code is not None
                else "terminal opened but interactive process identity was not reported"
            )
            raise RuntimeError(f"{self.config.display_name} failed to start: {detail}")

        pid, pgid, sid = identity

        exec_deadline = time.monotonic() + 2.0
        actual_command: list[str] | None = None
        while time.monotonic() < exec_deadline:
            try:
                candidate = proc_cmdline(pid)
            except (FileNotFoundError, ProcessLookupError, OSError):
                candidate = []
            if candidate and "ai_control_centre.terminal_runner" not in " ".join(candidate):
                actual_command = candidate
                break
            time.sleep(0.02)

        if actual_command is None:
            raise RuntimeError(
                f"{self.config.display_name} did not enter its interactive process"
            )

        try:
            record = ProcessRecord(
                service_id=self.config.id,
                pid=pid,
                pgid=pgid,
                sid=sid,
                proc_start_ticks=proc_start_ticks(pid),
                executable=proc_executable(pid),
                command_sha256=command_hash(actual_command),
                model_id=self.model_id,
                launch_mode=self.launch_mode,
            )
        except (FileNotFoundError, ProcessLookupError, OSError) as exc:
            raise RuntimeError(
                f"{self.config.display_name} exited before Control Centre could track it"
            ) from exc

        if not (record.pid == record.pgid == record.sid):
            try:
                os.killpg(record.pgid, signal.SIGTERM)
            except OSError:
                pass
            raise RuntimeError(
                f"Unsafe interactive process ownership for {self.config.display_name}: "
                f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
            )

        self.runtime_store.save_process(record)
        return ServiceStatus(
            ServiceState.READY,
            "interactive terminal ready",
            record.pid,
        )

    def stop(self) -> ServiceStatus:
        record = self.runtime_store.load_process(self.config.id)
        if record is None:
            return ServiceStatus(ServiceState.STOPPED, "not running")

        if not record_matches_process(record):
            self.runtime_store.delete(self.config.id)
            return ServiceStatus(ServiceState.STOPPED, "stale process record removed")

        if not (record.pid == record.pgid == record.sid):
            raise RuntimeError(
                "Refusing group stop because interactive process ownership is unsafe: "
                f"pid={record.pid} pgid={record.pgid} sid={record.sid}"
            )

        os.killpg(record.pgid, signal.SIGTERM)
        deadline = time.monotonic() + self.config.stop_timeout
        while time.monotonic() < deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                try:
                    self._runner_state_path.unlink()
                except FileNotFoundError:
                    pass
                return ServiceStatus(ServiceState.STOPPED, "stopped gracefully")
            time.sleep(0.1)

        os.killpg(record.pgid, signal.SIGKILL)
        hard_deadline = time.monotonic() + 2.0
        while time.monotonic() < hard_deadline:
            if not record_matches_process(record):
                self.runtime_store.delete(self.config.id)
                try:
                    self._runner_state_path.unlink()
                except FileNotFoundError:
                    pass
                return ServiceStatus(ServiceState.STOPPED, "forced stop")
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
