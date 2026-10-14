from __future__ import annotations

import socket
import sys
from pathlib import Path

from ai_control_centre.domain import HealthCheckConfig, ProcessServiceConfig, ServiceState
from ai_control_centre.process_service import ProcessService
from ai_control_centre.runtime import RuntimeStore


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_process_service_start_ready_stop(tmp_path: Path):
    port = _free_port()
    config = ProcessServiceConfig(
        id="dummy",
        display_name="Dummy HTTP",
        executable=Path(sys.executable),
        args=("-m", "http.server", str(port), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{port}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        stop_timeout=2,
    )
    service = ProcessService(config, RuntimeStore(tmp_path / "runtime"), tmp_path / "logs")

    assert service.status().state == ServiceState.STOPPED
    started = service.start()
    assert started.state == ServiceState.READY
    assert started.pid is not None
    assert service.status().state == ServiceState.READY

    stopped = service.stop()
    assert stopped.state == ServiceState.STOPPED
    assert service.status().state == ServiceState.STOPPED


def test_script_launcher_retains_ownership_after_restart(tmp_path: Path):
    """An executable script is interpreted by Python, changing its live argv."""
    port = _free_port()
    wrapper = tmp_path / "web-launcher"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import http.server, sys\n"
        "server = http.server.ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), http.server.SimpleHTTPRequestHandler)\n"
        "server.serve_forever()\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    config = ProcessServiceConfig(
        id="script_agent",
        display_name="Script Agent",
        executable=wrapper,
        args=(str(port),),
        cwd=tmp_path,
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{port}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        stop_timeout=2,
    )
    store = RuntimeStore(tmp_path / "runtime")
    service = ProcessService(config, store, tmp_path / "logs")
    try:
        started = service.start()
        assert started.state == ServiceState.READY
        record = store.load_process("script_agent")
        assert record is not None
        assert record.process_command_sha256 is not None
        assert record.process_command_sha256 != record.command_sha256
        reopened = ProcessService(config, store, tmp_path / "logs")
        assert reopened.status().state == ServiceState.READY
        assert reopened.stop().state == ServiceState.STOPPED
        assert reopened.status().state == ServiceState.STOPPED
    finally:
        import os
        import signal
        from ai_control_centre.runtime import record_matches_process
        record = store.load_process("script_agent")
        if record is not None and record_matches_process(record):
            os.killpg(record.pgid, signal.SIGTERM)
