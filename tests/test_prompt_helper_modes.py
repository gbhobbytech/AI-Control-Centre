from __future__ import annotations

from pathlib import Path

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import ModelConfig, ProcessServiceConfig, PromptWorkshopConfig, ServiceState
from ai_control_centre.manager import ServiceManager
from ai_control_centre.service_base import ServiceStatus


class _FakeController:
    def __init__(self, state: ServiceState):
        self._state = state
    def status(self):
        return ServiceStatus(self._state, "test")


def _manager(tmp_path: Path, mode: str = "auto") -> ServiceManager:
    model_path = tmp_path / "small.gguf"
    model_path.write_bytes(b"model")
    helper = ProcessServiceConfig(
        id="prompt_helper",
        display_name="Prompt Helper",
        executable=Path("/bin/echo"),
        args=("-m", "{model.path}", "-ngl", "0"),
        cwd=tmp_path,
    )
    comfy = ProcessServiceConfig(
        id="comfyui",
        display_name="ComfyUI",
        executable=Path("/bin/echo"),
        cwd=tmp_path,
        gpu=True,
    )
    settings = Settings(
        log_dir=tmp_path / "logs",
        runtime_dir=tmp_path / "runtime",
        prompt_workshop=PromptWorkshopConfig(
            service="prompt_helper",
            startup_profile="prompt_helper",
            model_strategy="manual",
            preferred_model="small",
            processing_mode=mode,
            gpu_layers=12,
        ),
    )
    config = AppConfig(
        settings=settings,
        services={"prompt_helper": helper, "comfyui": comfy},
        profiles={},
        models={"small": ModelConfig(id="small", display_name="Small", path=model_path)},
    )
    return ServiceManager(config)


def test_auto_prompt_helper_uses_gpu_when_no_gpu_workload_active(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.services["comfyui"] = _FakeController(ServiceState.STOPPED)
    assert manager.prompt_helper_processing_mode() == "gpu"
    controller = manager._controller_for_start("prompt_helper", None)
    idx = controller.config.args.index("-ngl")
    assert controller.config.args[idx + 1] == "12"


def test_auto_prompt_helper_uses_cpu_when_gpu_workload_active(tmp_path: Path):
    manager = _manager(tmp_path)
    manager.services["comfyui"] = _FakeController(ServiceState.READY)
    assert manager.prompt_helper_processing_mode() == "cpu"
    controller = manager._controller_for_start("prompt_helper", None)
    idx = controller.config.args.index("-ngl")
    assert controller.config.args[idx + 1] == "0"
