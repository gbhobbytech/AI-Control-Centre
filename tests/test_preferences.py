from __future__ import annotations

from pathlib import Path

from ai_control_centre.config import load_app_config, load_settings
from ai_control_centre.preferences import save_preferences, write_toml_atomic


def _make_config(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    model_root = tmp_path / "models"
    model_root.mkdir()
    (model_root / "small.gguf").write_bytes(b"x" * 32)
    (model_root / "large.gguf").write_bytes(b"x" * 64)

    write_toml_atomic(
        config / "settings.toml",
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "model_roots": {"llm": [str(model_root)]},
            "monitoring": {"gpu_backend": "auto", "interval_seconds": 2.0},
            "prompt_workshop": {
                "service": "prompt_helper",
                "startup_profile": "prompt_helper",
                "agent_profile": "agent",
                "image_profile": "image",
                "model_strategy": "smallest",
            },
        },
    )
    write_toml_atomic(
        config / "services.toml",
        {
            "services": {
                "llama": {
                    "display_name": "llama",
                    "type": "process",
                    "executable": "/bin/echo",
                    "cwd": str(tmp_path),
                    "args": ["-m", "{model.path}"],
                },
                "prompt_helper": {
                    "display_name": "Prompt Helper",
                    "type": "process",
                    "executable": "/bin/echo",
                    "cwd": str(tmp_path),
                    "args": ["-m", "{model.path}", "-ngl", "0", "--ctx-size", "4096", "--port", "8081"],
                    "health": {"url": "http://127.0.0.1:8081/health"},
                },
                "computer_dmz": {"display_name": "DMZ", "type": "docker", "container_name": "dummy", "dependencies": ["llama"]},
                "comfyui": {"display_name": "ComfyUI", "type": "process", "executable": "/bin/echo", "cwd": str(tmp_path), "args": []},
            }
        },
    )
    write_toml_atomic(
        config / "profiles.toml",
        {
            "profiles": {
                "coding": {"display_name": "Coding", "services": ["llama"], "default_model": "small"},
                "chat": {"display_name": "Chat", "services": ["llama"], "default_model": "large"},
                "agent": {"display_name": "Agent", "services": ["llama", "computer_dmz"], "default_model": "large"},
                "image": {"display_name": "Image", "services": ["comfyui"]},
                "prompt_helper": {"display_name": "Prompt Helper", "services": ["prompt_helper"]},
            }
        },
    )
    write_toml_atomic(
        config / "models.toml",
        {
            "models": {
                "small": {"display_name": "Small", "path": str(model_root / "small.gguf")},
                "large": {"display_name": "Large", "path": str(model_root / "large.gguf")},
            }
        },
    )
    return config


def test_save_preferences_round_trips_task_and_prompt_settings(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    save_preferences(
        config_dir,
        model_roots=(tmp_path / "models",),
        task_models={"coding": "large", "chat": "small", "agent": "small"},
        prompt_model="small",
        prompt_auto_lightest=False,
        prompt_processing_mode="gpu",
        prompt_keep_loaded=False,
        prompt_gpu_layers=7,
        prompt_context_length=8192,
        prompt_port=8099,
        prompt_startup_timeout=45,
        prompt_cache_type_k="q4_0",
        prompt_cache_type_v="f16",
    )

    config = load_app_config(config_dir)
    assert config.settings.setup_completed is True
    assert config.profiles["coding"].default_model == "large"
    assert config.profiles["chat"].default_model == "small"
    assert config.settings.prompt_workshop.model_strategy == "manual"
    assert config.settings.prompt_workshop.preferred_model == "small"
    assert config.settings.prompt_workshop.processing_mode == "gpu"
    assert config.settings.prompt_workshop.keep_loaded is False
    assert config.settings.prompt_workshop.gpu_layers == 7
    assert config.settings.prompt_workshop.context_length == 8192
    assert config.settings.prompt_workshop.port == 8099
    assert config.settings.prompt_workshop.cache_type_k == "q4_0"
    assert config.settings.prompt_workshop.cache_type_v == "f16"
    helper = config.services["prompt_helper"]
    assert "8192" in helper.args
    assert "8099" in helper.args
    assert helper.health is not None
    assert helper.health.url.endswith(":8099/health")
    assert helper.health.startup_timeout == 45


def test_settings_defaults_setup_to_incomplete(tmp_path: Path):
    settings = tmp_path / "settings.toml"
    settings.write_text(
        '[paths]\nlog_dir = "~/logs"\nruntime_dir = "~/run"\n',
        encoding="utf-8",
    )
    loaded = load_settings(settings)
    assert loaded.setup_completed is False
    assert loaded.prompt_workshop.processing_mode == "auto"
