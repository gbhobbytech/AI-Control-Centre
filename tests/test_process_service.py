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
