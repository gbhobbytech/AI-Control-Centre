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


def test_agent_profile_adds_selected_harness_services(tmp_path: Path):
    from ai_control_centre.domain import HarnessConfig

    port_a = _free_port()
    port_b = _free_port()
    llama = ProcessServiceConfig(
        id="llama", display_name="LLM", executable=Path(sys.executable),
        args=("-m", "http.server", str(port_a), "--bind", "127.0.0.1"), cwd=tmp_path,
        health=HealthCheckConfig(url=f"http://127.0.0.1:{port_a}/", startup_timeout=5, retry_interval=0.05, request_timeout=0.2),
        stop_timeout=2,
    )
    harness_service = ProcessServiceConfig(
        id="harness", display_name="Harness", executable=Path(sys.executable),
        args=("-m", "http.server", str(port_b), "--bind", "127.0.0.1"), cwd=tmp_path,
        health=HealthCheckConfig(url=f"http://127.0.0.1:{port_b}/", startup_timeout=5, retry_interval=0.05, request_timeout=0.2),
        dependencies=("llama",), stop_timeout=2,
    )
    config = AppConfig(
        settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
        services={"llama": llama, "harness": harness_service},
        profiles={"agent": ProfileConfig(id="agent", display_name="Agent", services=("llama",), harness="test")},
        harnesses={"test": HarnessConfig(id="test", display_name="Test Harness", services=("harness",))},
    )
    manager = ServiceManager(config)
    try:
        results = manager.start_profile("agent", open_interface=False)
        assert [service_id for service_id, _ in results] == ["llama", "harness"]
    finally:
        manager.stop_all()


def test_start_harness_reuses_running_dependency_model(tmp_path: Path, monkeypatch):
    """Starting a model-free harness must not request a different default model."""
    from ai_control_centre.service_base import ServiceStatus

    llama = ProcessServiceConfig(
        id="llama", display_name="LLM", executable=Path(sys.executable),
        args=("-m", "{model.path}"), cwd=tmp_path,
        default_model="heretic",
    )
    harness = ProcessServiceConfig(
        id="deepseek", display_name="DeepSeek", executable=Path(sys.executable),
        cwd=tmp_path, dependencies=("llama",),
    )
    config = AppConfig(
        settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
        services={"llama": llama, "deepseek": harness},
        profiles={},
    )
    manager = ServiceManager(config)

    class FakeController:
        def __init__(self, state):
            self.state = state
            self.config = None

        def status(self):
            return ServiceStatus(self.state, "test")

        def start(self):
            self.state = ServiceState.READY
            return self.status()

    controllers = {
        "llama": FakeController(ServiceState.READY),
        "deepseek": FakeController(ServiceState.STOPPED),
    }
    selected = []
    monkeypatch.setattr(manager, "get", lambda sid: controllers[sid])
    monkeypatch.setattr(manager, "running_model_id", lambda sid: "qwen_coder" if sid == "llama" else None)
    monkeypatch.setattr(manager, "_controller_for_start", lambda sid, model: (
        selected.append((sid, model)) or controllers[sid]
    ))
    monkeypatch.setattr(manager, "_prepare_conflicts", lambda *args, **kwargs: [])
    monkeypatch.setattr(manager, "_prepare_model_switch", lambda *args, **kwargs: [])

    results = manager.start_service("deepseek")
    assert [sid for sid, _ in results] == ["llama", "deepseek"]
    assert selected == [
        ("llama", "qwen_coder"), ("deepseek", "qwen_coder"),
        ("llama", "qwen_coder"), ("deepseek", "qwen_coder"),
    ]
    assert all(status.state == ServiceState.READY for _, status in results)


def test_start_harness_uses_agent_model_when_dependency_stopped(tmp_path: Path, monkeypatch):
    """A cold harness start uses the Agent model, not llama's unrelated default."""
    from ai_control_centre.domain import HarnessConfig
    from ai_control_centre.service_base import ServiceStatus

    llama = ProcessServiceConfig(
        id="llama", display_name="LLM", executable=Path(sys.executable),
        args=("-m", "{model.path}"), cwd=tmp_path, default_model="heretic",
    )
    harness = ProcessServiceConfig(
        id="deepseek", display_name="DeepSeek", executable=Path(sys.executable),
        cwd=tmp_path, dependencies=("llama",),
    )
    config = AppConfig(
        settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
        services={"llama": llama, "deepseek": harness},
        profiles={"agent": ProfileConfig(
            id="agent", display_name="Agent", services=("llama",),
            harness="deepseek_harness", selected_model="qwen_coder",
        )},
        harnesses={"deepseek_harness": HarnessConfig(
            id="deepseek_harness", display_name="DeepSeek", services=("deepseek",),
        )},
    )
    manager = ServiceManager(config)

    class FakeController:
        config = None

        def __init__(self):
            self.state = ServiceState.STOPPED

        def status(self):
            return ServiceStatus(self.state, "test")

        def start(self):
            self.state = ServiceState.READY
            return self.status()

    controllers = {"llama": FakeController(), "deepseek": FakeController()}
    selected = []
    monkeypatch.setattr(manager, "get", lambda sid: controllers[sid])
    monkeypatch.setattr(manager, "_controller_for_start", lambda sid, model: (
        selected.append((sid, model)) or controllers[sid]
    ))
    monkeypatch.setattr(manager, "_prepare_conflicts", lambda *args, **kwargs: [])
    monkeypatch.setattr(manager, "_prepare_model_switch", lambda *args, **kwargs: [])

    results = manager.start_service("deepseek")
    assert [sid for sid, _ in results] == ["llama", "deepseek"]
    assert selected == [
        ("llama", "qwen_coder"), ("deepseek", "qwen_coder"),
        ("llama", "qwen_coder"), ("deepseek", "qwen_coder"),
    ]
    assert all(status.state == ServiceState.READY for _, status in results)
