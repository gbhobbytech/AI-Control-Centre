from __future__ import annotations

import sys
from pathlib import Path

from ai_control_centre.config import load_services
from ai_control_centre.domain import ServiceState, TerminalServiceConfig
from ai_control_centre.runtime import RuntimeStore
from ai_control_centre.terminal_service import TerminalService


def test_load_terminal_service(tmp_path: Path):
    services_path = tmp_path / "services.toml"
    services_path.write_text(
        f"""
[services.aider]
display_name = "Aider"
type = "terminal"
executable = "{sys.executable}"
cwd = "{tmp_path}"
args = ["-c", "print('hello')"]
terminal_executable = "{sys.executable}"
terminal_args = ["-c", "import time; time.sleep(5)"]
startup_grace = 0.1
dependencies = []
""".strip(),
        encoding="utf-8",
    )

    services = load_services(services_path)
    service = services["aider"]

    assert isinstance(service, TerminalServiceConfig)
    assert service.terminal_executable == sys.executable
    assert service.startup_grace == 0.1
    assert service.command[0] == sys.executable


def test_terminal_service_reports_ready_and_stops(tmp_path: Path):
    config = TerminalServiceConfig(
        id="terminal",
        display_name="Terminal",
        executable=Path(sys.executable),
        args=(),
        cwd=tmp_path,
        terminal_executable=sys.executable,
        terminal_args=("-c", "import time; time.sleep(30)"),
        startup_grace=0.1,
        stop_timeout=1.0,
    )
    runtime = RuntimeStore(tmp_path / "runtime")
    service = TerminalService(config, runtime)

    try:
        started = service.start()
        assert started.state == ServiceState.READY

        status = service.status()
        assert status.state == ServiceState.READY
        assert status.pid == started.pid

        stopped = service.stop()
        assert stopped.state == ServiceState.STOPPED
    finally:
        record = runtime.load_process("terminal")
        if record is not None:
            try:
                service.stop()
            except Exception:
                pass
