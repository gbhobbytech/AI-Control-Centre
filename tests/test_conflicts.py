from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import HealthCheckConfig, ProcessServiceConfig, ProfileConfig, ServiceState
from ai_control_centre.manager import ServiceManager


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_service(
    tmp_path: Path,
    service_id: str,
    port: int,
    *,
    dependencies: tuple[str, ...] = (),
    conflict_groups: tuple[str, ...] = (),
) -> ProcessServiceConfig:
    return ProcessServiceConfig(
        id=service_id,
        display_name=service_id,
        executable=Path(sys.executable),
        args=("-m", "http.server", str(port), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{port}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        dependencies=dependencies,
        conflict_groups=conflict_groups,
        stop_timeout=2,
    )


def _manager(tmp_path: Path) -> ServiceManager:
    llama = _http_service(
        tmp_path,
        "llama",
        _free_port(),
        conflict_groups=("large-gpu-workload",),
    )
    dmz = _http_service(
        tmp_path,
        "computer_dmz",
        _free_port(),
        dependencies=("llama",),
    )
    comfy = _http_service(
        tmp_path,
        "comfyui",
        _free_port(),
        conflict_groups=("large-gpu-workload",),
    )
    return ServiceManager(
        AppConfig(
            settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
            services={"llama": llama, "computer_dmz": dmz, "comfyui": comfy},
            profiles={
                "agent": ProfileConfig(
                    id="agent",
                    display_name="Agent",
                    services=("computer_dmz",),
                ),
                "image": ProfileConfig(
                    id="image",
                    display_name="Image",
                    services=("comfyui",),
                ),
            },
        )
    )


def test_conflict_refuses_without_explicit_stop(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("agent", open_interface=False)
        with pytest.raises(RuntimeError, match="--stop-conflicts"):
            manager.start_profile("image", open_interface=False)
        assert manager.get("llama").status().state == ServiceState.READY
        assert manager.get("computer_dmz").status().state == ServiceState.READY
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
    finally:
        manager.stop_all()


def test_agent_to_image_stops_dependent_before_conflicting_llama(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("agent", open_interface=False)
        results = manager.start_profile("image", open_interface=False, stop_conflicts=True)
        assert [service_id for service_id, _ in results] == ["computer_dmz", "llama", "comfyui"]
        assert results[0][1].state == ServiceState.STOPPED
        assert results[1][1].state == ServiceState.STOPPED
        assert results[2][1].state == ServiceState.READY
        assert manager.get("computer_dmz").status().state == ServiceState.STOPPED
        assert manager.get("llama").status().state == ServiceState.STOPPED
        assert manager.get("comfyui").status().state == ServiceState.READY
    finally:
        manager.stop_all()


def test_image_to_agent_stops_comfyui_then_starts_dependencies(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("image", open_interface=False)
        results = manager.start_profile("agent", open_interface=False, stop_conflicts=True)
        assert [service_id for service_id, _ in results] == ["comfyui", "llama", "computer_dmz"]
        assert results[0][1].state == ServiceState.STOPPED
        assert results[1][1].state == ServiceState.READY
        assert results[2][1].state == ServiceState.READY
    finally:
        manager.stop_all()


def test_external_conflict_is_never_stopped_automatically(tmp_path: Path):
    import subprocess

    manager = _manager(tmp_path)
    comfy_cfg = manager.config.services["comfyui"]
    external = subprocess.Popen(
        list(comfy_cfg.command),
        cwd=comfy_cfg.cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        # Wait for the independently started HTTP service to become visible as External.
        import time
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if manager.get("comfyui").status().state == ServiceState.EXTERNAL:
                break
            time.sleep(0.05)
        assert manager.get("comfyui").status().state == ServiceState.EXTERNAL

        with pytest.raises(RuntimeError, match="externally started"):
            manager.start_profile("agent", open_interface=False, stop_conflicts=True)

        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=5)
        manager.stop_all()
