"""Phase 5: explanations for unavailable tasks and failed starts (no Tk needed)."""

from __future__ import annotations

import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import (
    HealthCheckConfig,
    ModelConfig,
    ProcessServiceConfig,
    ProfileConfig,
    ServiceState,
    TASK_IDS,
)
from ai_control_centre.environments import evaluate_profile
from ai_control_centre.feedback import (
    EXTERNAL_DETAIL,
    STATE_LABELS,
    attention_items,
    attention_text,
    failure_advice,
    task_tooltip,
)
from ai_control_centre.manager import ServiceManager, ServiceStartError, TransitionError


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(tmp_path: Path, *, environments=None, models=None, comfy_exe=None, comfy_args=None, health=None) -> AppConfig:
    model_file = tmp_path / "qwen.gguf"
    model_file.write_bytes(b"GGUF")
    port = _free_port()
    services = {
        "llama": ProcessServiceConfig(
            id="llama", display_name="llama.cpp", executable=Path(sys.executable),
            args=("-m", "http.server", str(_free_port()), "--bind", "127.0.0.1", "--directory", "{model.path}"),
            cwd=tmp_path, gpu=True, conflict_groups=("large-gpu-workload",),
        ),
        "comfyui": ProcessServiceConfig(
            id="comfyui", display_name="ComfyUI", executable=comfy_exe or Path(sys.executable),
            args=comfy_args or ("-m", "http.server", str(port), "--bind", "127.0.0.1"),
            cwd=tmp_path, gpu=True, conflict_groups=("large-gpu-workload",),
            health=health or HealthCheckConfig(url=f"http://127.0.0.1:{port}/", startup_timeout=5.0, retry_interval=0.05, request_timeout=0.3),
        ),
    }
    profiles = {
        "chat": ProfileConfig("chat", "Chat", ("llama",), "llama", "qwen"),
        "image": ProfileConfig("image", "Image", ("comfyui",), "comfyui"),
    }
    settings = Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime")
    if environments is None:
        settings = replace(settings, environments={t: t in profiles for t in TASK_IDS})
    else:
        settings = replace(settings, environments=environments, environments_explicit=True)
    if models is None:
        models = {"qwen": ModelConfig("qwen", "Qwen3.5-4B", model_file, tuning_reviewed=True)}
    return AppConfig(settings=settings, services=services, profiles=profiles, harnesses={}, models=models)


# --- Unavailable tasks -------------------------------------------------------------


def test_missing_gguf_model_produces_clear_explanation(tmp_path: Path):
    config = _config(tmp_path, models={})
    [item] = attention_items(config)
    assert item.task_id == "chat" and item.setup_page == "models"
    assert item.message == "Chat: Chat model 'qwen' was not found. Rescan models or choose another model in Settings."
    tooltip = task_tooltip(evaluate_profile(config, "chat"))
    assert tooltip.endswith("Use Fix in Setup to correct it.")


def test_missing_comfyui_installation_produces_clear_explanation(tmp_path: Path):
    config = _config(tmp_path, comfy_exe=tmp_path / "ComfyUI/venv/bin/python")
    [item] = attention_items(config)
    assert item.task_id == "image" and item.setup_page == "software"
    assert item.message == f"Image: ComfyUI not found at {tmp_path / 'ComfyUI/venv/bin/python'}."


def test_unconfigured_tasks_direct_users_to_setup(tmp_path: Path):
    config = _config(tmp_path, environments={"coding": True, "chat": True, "agent": False, "image": True})
    items = {item.task_id: item for item in attention_items(config)}
    assert items["coding"].setup_page == "environments"  # enabled but no Coding profile
    assert items["coding"].message == "Coding: Coding is not configured."
    assert "chat" not in items and "image" not in items  # these are ready


def test_environments_turned_off_are_not_flagged(tmp_path: Path):
    config = _config(
        tmp_path, models={},
        environments={"coding": False, "chat": False, "agent": False, "image": True},
    )
    assert attention_items(config) == []
    status = evaluate_profile(config, "chat")
    assert task_tooltip(status) == "Chat is turned off. Enable it in Setup to use it."


def test_v12_configuration_without_a_profile_is_not_flagged(tmp_path: Path):
    config = _config(tmp_path)  # inferred environments; no Coding or Agent profile
    assert attention_items(config) == []


def test_attention_text_for_one_or_several_items(tmp_path: Path):
    config = _config(tmp_path, models={}, comfy_exe=tmp_path / "missing/python")
    items = attention_items(config)
    assert [item.task_id for item in items] == ["chat", "image"]
    text = attention_text(items)
    assert text.startswith("Needs attention - Chat: ") and "Image: ComfyUI not found" in text
    assert attention_text(items[:1]) == items[0].message
    assert attention_text([]) == ""


def test_ready_task_tooltip(tmp_path: Path):
    assert task_tooltip(evaluate_profile(_config(tmp_path), "image")) == "Start Image."


# --- States ------------------------------------------------------------------------


def test_users_can_distinguish_service_states():
    shown = {STATE_LABELS[state] for state in (
        ServiceState.STARTING, ServiceState.READY, ServiceState.STOPPED, ServiceState.ERROR, ServiceState.EXTERNAL,
    )}
    assert shown == {"Starting", "Ready", "Stopped", "Error", "External"}
    assert STATE_LABELS[ServiceState.RUNNING_NOT_READY] == "Starting"
    assert "will not be stopped" in EXTERNAL_DETAIL


# --- Failed starts -------------------------------------------------------------------


def _failing_start(config: AppConfig, profile: str = "image") -> Exception:
    manager = ServiceManager(config)
    try:
        manager.start_profile(profile, open_interface=False, stop_conflicts=True)
    except Exception as exc:  # noqa: BLE001
        return exc
    finally:
        manager.stop_all()
    raise AssertionError("start did not fail")


def test_failure_identifies_missing_program_and_points_to_setup(tmp_path: Path):
    manager_config = _config(tmp_path)
    services = dict(manager_config.services)
    services["comfyui"] = replace(services["comfyui"], executable=tmp_path / "gone/python")
    exc = _failing_start(replace(manager_config, services=services))
    assert isinstance(exc, ServiceStartError) and exc.service_id == "comfyui"
    advice = failure_advice(exc, manager_config)
    assert advice.title == "ComfyUI could not start"
    assert advice.setup_page == "software" and advice.log_path is None
    assert "Choose it again in Setup" in advice.as_text()


def test_failure_when_process_exits_offers_its_log(tmp_path: Path):
    port = _free_port()
    config = _config(
        tmp_path, comfy_args=("-c", "raise SystemExit(3)"),
        health=HealthCheckConfig(url=f"http://127.0.0.1:{port}/", startup_timeout=3.0, retry_interval=0.05, request_timeout=0.2),
    )
    advice = failure_advice(_failing_start(config), config)
    assert advice.title == "ComfyUI could not start"
    assert advice.log_path is not None and advice.log_path.name.startswith("comfyui")
    assert "Its log usually says why" in advice.hint


def test_failure_on_startup_timeout_explains_it(tmp_path: Path):
    port = _free_port()
    config = _config(
        tmp_path, comfy_args=("-c", "import time; time.sleep(30)"),
        health=HealthCheckConfig(url=f"http://127.0.0.1:{port}/", startup_timeout=0.5, retry_interval=0.05, request_timeout=0.1),
    )
    advice = failure_advice(_failing_start(config), config)
    assert "still not ready when the startup time ran out" in advice.hint
    assert advice.log_path is not None


def test_failure_on_busy_port_explains_it(tmp_path: Path):
    config = _config(tmp_path)
    url = config.services["comfyui"].health.url
    port = int(url.rsplit(":", 1)[1].strip("/"))
    with socket.socket() as other:
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        other.bind(("127.0.0.1", port))
        other.listen()
        advice = failure_advice(_failing_start(config), config)
    assert advice.title == "ComfyUI could not start"
    assert "Another program is listening on that port" in advice.hint


def test_failure_after_a_switch_still_names_the_service(tmp_path: Path):
    port = _free_port()
    config = _config(
        tmp_path, comfy_args=("-c", "raise SystemExit(3)"),
        health=HealthCheckConfig(url=f"http://127.0.0.1:{port}/", startup_timeout=3.0, retry_interval=0.05, request_timeout=0.2),
    )
    manager = ServiceManager(config)
    try:
        manager.start_profile("chat", open_interface=False)
        with pytest.raises(TransitionError) as info:
            manager.start_profile("image", open_interface=False, stop_conflicts=True)
    finally:
        manager.stop_all()
    advice = failure_advice(info.value, config)
    assert advice.title == "ComfyUI could not start"
    assert "Already stopped for this switch: llama.cpp" in advice.message
