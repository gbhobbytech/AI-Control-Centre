from __future__ import annotations

import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import HealthCheckConfig, ModelConfig, ProcessServiceConfig, ProfileConfig, ServiceState
from ai_control_centre.manager import ConflictError, ServiceManager, TransitionError
from ai_control_centre.service_base import ServiceStatus


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


# --- v1.3.0 Phase 4: clear, confirmed, verified switching -----------------------


def _chat_image_manager(tmp_path: Path) -> ServiceManager:
    """Chat on llama.cpp and Image on ComfyUI, sharing one GPU conflict group."""
    manager = _manager(tmp_path)
    config = manager.config
    profiles = dict(config.profiles)
    profiles["chat"] = ProfileConfig(id="chat", display_name="Chat", services=("llama",))
    config = AppConfig(
        settings=config.settings,
        services={
            **config.services,
            "llama": _replace(config.services["llama"], display_name="llama.cpp"),
            "comfyui": _replace(config.services["comfyui"], display_name="ComfyUI"),
        },
        profiles=profiles,
    )
    return ServiceManager(config)


def _replace(service, **changes):
    return replace(service, **changes)


def test_chat_to_image_identifies_the_gpu_conflict_in_plain_terms(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        with pytest.raises(ConflictError) as info:
            manager.start_profile("image", open_interface=False)
        plan = info.value.plan
        assert plan.target == "Image" and plan.reason == "conflict" and not plan.blocked
        assert [stop.display_name for stop in plan.stops] == ["llama.cpp"]
        text = plan.describe()
        assert "Image needs the GPU" in text
        assert "  - llama.cpp" in text and "Then Image will start." in text
        assert "--stop-conflicts" in str(info.value)  # CLI wording unchanged
        # Cancelling: nothing was changed.
        assert manager.get("llama").status().state == ServiceState.READY
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
    finally:
        manager.stop_all()


def test_plan_names_the_loaded_model(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    manager.config.models["qwen"] = ModelConfig(id="qwen", display_name="Qwen3.5-4B", path=tmp_path / "q.gguf")
    try:
        manager.start_profile("chat", open_interface=False)
        record = manager.runtime_store.load_process("llama")
        manager.runtime_store.save_process(_replace(record, model_id="qwen"))
        with pytest.raises(ConflictError) as info:
            manager.start_profile("image", open_interface=False)
        assert info.value.plan.stops[0].describe() == "llama.cpp (model Qwen3.5-4B)"
    finally:
        manager.stop_all()


def test_confirmed_switch_stops_llama_before_starting_comfyui_and_back(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        to_image = manager.start_profile("image", open_interface=False, stop_conflicts=True)
        assert [(sid, status.state) for sid, status in to_image] == [
            ("llama", ServiceState.STOPPED), ("comfyui", ServiceState.READY),
        ]
        to_chat = manager.start_profile("chat", open_interface=False, stop_conflicts=True)
        assert [(sid, status.state) for sid, status in to_chat] == [
            ("comfyui", ServiceState.STOPPED), ("llama", ServiceState.READY),
        ]
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
    finally:
        manager.stop_all()


def test_after_stops_runs_between_shutdown_and_startup(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    seen = []

    def after_stops(results):
        seen.append((
            [(sid, status.state) for sid, status in results],
            manager.get("comfyui").status().state,
        ))

    try:
        manager.start_profile("chat", open_interface=False, after_stops=after_stops)
        assert seen == []  # nothing was stopped
        manager.start_profile("image", open_interface=False, stop_conflicts=True, after_stops=after_stops)
        assert seen == [([("llama", ServiceState.STOPPED)], ServiceState.STOPPED)]
    finally:
        manager.stop_all()


def test_external_blocker_is_explained_and_never_stopped(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    comfy = manager.config.services["comfyui"]
    external = subprocess.Popen(
        list(comfy.command), cwd=comfy.cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while manager.get("comfyui").status().state != ServiceState.EXTERNAL:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        with pytest.raises(ConflictError) as info:
            manager.start_profile("chat", open_interface=False, stop_conflicts=True)
        plan = info.value.plan
        assert plan.blocked and plan.external == ("ComfyUI",)
        assert "will not stop programs it did not start" in plan.describe()
        assert external.poll() is None
        assert manager.get("llama").status().state == ServiceState.STOPPED
    finally:
        external.terminate()
        external.wait(timeout=5)
        manager.stop_all()


def test_failed_shutdown_blocks_the_next_start(tmp_path: Path, monkeypatch):
    manager = _chat_image_manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        llama = manager.get("llama")
        monkeypatch.setattr(
            llama, "stop", lambda: ServiceStatus(ServiceState.ERROR, "still shutting down", None)
        )
        with pytest.raises(TransitionError, match="llama.cpp did not stop"):
            manager.start_profile("image", open_interface=False, stop_conflicts=True)
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
        assert manager.get("llama").status().state == ServiceState.READY  # state still accurate
    finally:
        monkeypatch.undo()
        manager.stop_all()


def test_failed_start_after_switch_reports_what_was_stopped(tmp_path: Path, monkeypatch):
    manager = _chat_image_manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        comfy = manager.get("comfyui")
        monkeypatch.setattr(
            comfy, "start",
            lambda: (_ for _ in ()).throw(RuntimeError("ComfyUI exited immediately; see comfyui.log")),
        )
        with pytest.raises(TransitionError) as info:
            manager.start_profile("image", open_interface=False, stop_conflicts=True)
        message = str(info.value)
        assert "ComfyUI exited immediately; see comfyui.log" in message
        assert "Already stopped for this switch: llama.cpp. It was not restarted automatically." in message
        assert info.value.stopped == ("llama.cpp",)
        assert manager.get("llama").status().state == ServiceState.STOPPED
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
    finally:
        monkeypatch.undo()
        manager.stop_all()


def test_stop_all_leaves_externally_started_services_running(tmp_path: Path):
    manager = _chat_image_manager(tmp_path)
    comfy = manager.config.services["comfyui"]
    manager.start_profile("agent", open_interface=False)
    try:
        # ComfyUI is outside the GPU group check here: started externally after Agent.
        external = subprocess.Popen(
            list(comfy.command), cwd=comfy.cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        deadline = time.monotonic() + 5
        while manager.get("comfyui").status().state != ServiceState.EXTERNAL:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        results = dict(manager.stop_all())
        assert results["comfyui"].state == ServiceState.EXTERNAL
        assert results["llama"].state == ServiceState.STOPPED
        assert results["computer_dmz"].state == ServiceState.STOPPED
        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=5)
        manager.stop_all()
