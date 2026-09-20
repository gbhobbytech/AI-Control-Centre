from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import HealthCheckConfig, ModelConfig, ProcessServiceConfig, ProfileConfig, ServiceState
from ai_control_centre.manager import ServiceManager


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _manager(tmp_path: Path) -> ServiceManager:
    llama_port = _free_port()
    dmz_port = _free_port()
    model_a_dir = tmp_path / "model-a"
    model_b_dir = tmp_path / "model-b"
    model_a_dir.mkdir()
    model_b_dir.mkdir()

    llama = ProcessServiceConfig(
        id="llama",
        display_name="llama.cpp",
        executable=Path(sys.executable),
        args=(
            "-m", "http.server", str(llama_port), "--bind", "127.0.0.1",
            "--directory", "{model.path}",
        ),
        cwd=tmp_path,
        default_model="chat_model",
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{llama_port}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        stop_timeout=2,
    )
    dmz = ProcessServiceConfig(
        id="computer_dmz",
        display_name="Computer DMZ",
        executable=Path(sys.executable),
        args=("-m", "http.server", str(dmz_port), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        dependencies=("llama",),
        health=HealthCheckConfig(
            url=f"http://127.0.0.1:{dmz_port}/",
            startup_timeout=5,
            retry_interval=0.05,
            request_timeout=0.2,
        ),
        stop_timeout=2,
    )
    return ServiceManager(
        AppConfig(
            settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
            services={"llama": llama, "computer_dmz": dmz},
            profiles={
                "chat": ProfileConfig(
                    id="chat", display_name="Chat", services=("llama",), selected_model="chat_model"
                ),
                "agent": ProfileConfig(
                    id="agent", display_name="Agent", services=("computer_dmz",), selected_model="chat_model"
                ),
                "coding": ProfileConfig(
                    id="coding", display_name="Coding", services=("llama",), selected_model="coder_model"
                ),
            },
            models={
                "chat_model": ModelConfig(id="chat_model", display_name="Chat", path=model_a_dir),
                "coder_model": ModelConfig(id="coder_model", display_name="Coder", path=model_b_dir),
            },
        )
    )


def test_model_change_requires_explicit_permission(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        assert manager.running_model_id("llama") == "chat_model"
        with pytest.raises(RuntimeError, match="--replace-model"):
            manager.start_profile("coding", open_interface=False)
        assert manager.running_model_id("llama") == "chat_model"
    finally:
        manager.stop_all()


def test_model_change_stops_and_reloads_llama(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        results = manager.start_profile("coding", open_interface=False, replace_model=True)
        assert [service_id for service_id, _ in results] == ["llama", "llama"]
        assert results[0][1].state == ServiceState.STOPPED
        assert results[1][1].state == ServiceState.READY
        assert manager.running_model_id("llama") == "coder_model"
    finally:
        manager.stop_all()


def test_chat_to_agent_reuses_same_llama_model(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("chat", open_interface=False)
        first_pid = manager.get("llama").status().pid
        results = manager.start_profile("agent", open_interface=False)
        assert [service_id for service_id, _ in results] == ["llama", "computer_dmz"]
        assert manager.get("llama").status().pid == first_pid
        assert manager.running_model_id("llama") == "chat_model"
        assert manager.get("computer_dmz").status().state == ServiceState.READY
    finally:
        manager.stop_all()


def test_agent_to_coding_stops_dependent_then_reloads_model(tmp_path: Path):
    manager = _manager(tmp_path)
    try:
        manager.start_profile("agent", open_interface=False)
        results = manager.start_profile("coding", open_interface=False, replace_model=True)
        assert [service_id for service_id, _ in results] == ["computer_dmz", "llama", "llama"]
        assert results[0][1].state == ServiceState.STOPPED
        assert results[1][1].state == ServiceState.STOPPED
        assert results[2][1].state == ServiceState.READY
        assert manager.running_model_id("llama") == "coder_model"
    finally:
        manager.stop_all()
