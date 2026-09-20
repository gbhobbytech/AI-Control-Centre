from __future__ import annotations

import socket
import sys
from pathlib import Path

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import HealthCheckConfig, ProcessServiceConfig, ProfileConfig, ServiceState
from ai_control_centre.manager import ServiceManager


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_profile_starts_services_in_declared_dependency_order(tmp_path: Path):
    port_a = _free_port()
    port_b = _free_port()
    first = ProcessServiceConfig(
        id="first",
        display_name="First",
        executable=Path(sys.executable),
        args=("-m", "http.server", str(port_a), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{port_a}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        stop_timeout=2,
    )
    second = ProcessServiceConfig(
        id="second",
        display_name="Second",
        executable=Path(sys.executable),
        args=("-m", "http.server", str(port_b), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{port_b}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        dependencies=("first",),
        stop_timeout=2,
    )
    config = AppConfig(
        settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
        services={"first": first, "second": second},
        profiles={
            "agent": ProfileConfig(
                id="agent",
                display_name="Agent",
                services=("second",),
            )
        },
    )
    manager = ServiceManager(config)
    try:
        results = manager.start_profile("agent", open_interface=False)
        assert [service_id for service_id, _ in results] == ["first", "second"]
        assert all(status.state == ServiceState.READY for _, status in results)
    finally:
        manager.stop_all()
