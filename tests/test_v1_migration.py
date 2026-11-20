from pathlib import Path

from ai_control_centre.config import CURRENT_SETUP_SCHEMA, load_app_config, load_models, load_settings
from ai_control_centre.domain import HarnessConfig, ModelConfig
from ai_control_centre.preferences import save_model_tuning, write_toml_atomic
from ai_control_centre.tuning import BASELINE, CURRENT_TUNING_SCHEMA


def test_pre_v1_completed_setup_is_forced_through_v1_setup(tmp_path: Path):
    settings = tmp_path / "settings.toml"
    write_toml_atomic(
        settings,
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "setup": {"completed": True},
        },
    )
    loaded = load_settings(settings)
    assert loaded.setup_schema_version == 0
    assert loaded.setup_completed is False

    write_toml_atomic(
        settings,
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "setup": {"completed": True, "schema_version": CURRENT_SETUP_SCHEMA},
        },
    )
    assert load_settings(settings).setup_completed is True


def test_pre_v1_tuning_is_historical_until_resaved(tmp_path: Path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"x")
    write_toml_atomic(
        tmp_path / "models.toml",
        {
            "models": {
                "m": {
                    "display_name": "M",
                    "path": str(model_file),
                    "recommended_gpu_layers": 22,
                    "recommended_context_length": 65536,
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "flash_attention": "auto",
                    "max_output_tokens": 2048,
                    "startup_timeout_seconds": 180,
                    "tuning_reviewed": True,
                    # No V1 tuning_schema_version: historical only.
                }
            }
        },
    )
    loaded = load_models(tmp_path / "models.toml", (tmp_path,))
    assert loaded["m"].tuning_reviewed is False
    assert loaded["m"].tuning_schema_version == 0

    save_model_tuning(tmp_path, loaded["m"], BASELINE, "v1 test")
    reloaded = load_models(tmp_path / "models.toml", (tmp_path,))
    assert reloaded["m"].tuning_reviewed is True
    assert reloaded["m"].tuning_schema_version == CURRENT_TUNING_SCHEMA


def test_legacy_agent_profile_is_exposed_as_migration_harness(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_file = model_root / "m.gguf"
    model_file.write_bytes(b"x")

    write_toml_atomic(
        config / "settings.toml",
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "model_roots": {"llm": [str(model_root)]},
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
                    "args": ["-m", "{model.path}"],
                },
                "computer_dmz": {
                    "display_name": "Computer DMZ",
                    "type": "docker",
                    "container_name": "dummy",
                    "dependencies": ["llama"],
                },
                "comfyui": {
                    "display_name": "ComfyUI",
                    "type": "process",
                    "executable": "/bin/echo",
                    "cwd": str(tmp_path),
                    "args": [],
                },
            }
        },
    )
    write_toml_atomic(
        config / "profiles.toml",
        {
            "profiles": {
                "coding": {"display_name": "Coding", "services": ["llama"]},
                "chat": {"display_name": "Chat", "services": ["llama"]},
                "agent": {
                    "display_name": "Agent",
                    "services": ["llama", "computer_dmz"],
                    "open_service": "computer_dmz",
                },
                "image": {"display_name": "Image", "services": ["comfyui"]},
                "prompt_helper": {"display_name": "Prompt Helper", "services": ["prompt_helper"]},
            }
        },
    )
    write_toml_atomic(config / "models.toml", {"models": {"m": {"path": str(model_file)}}})

    loaded = load_app_config(config)
    assert "legacy_agent" in loaded.harnesses
    assert loaded.harnesses["legacy_agent"].services == ("computer_dmz",)
    assert loaded.harnesses["legacy_agent"].open_service == "computer_dmz"


def test_v1_agent_harness_ignores_stale_profile_open_service(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_file = model_root / "m.gguf"
    model_file.write_bytes(b"x")

    write_toml_atomic(
        config / "settings.toml",
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "model_roots": {"llm": [str(model_root)]},
            "prompt_workshop": {"enabled": False},
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
                "computer_dmz": {
                    "display_name": "Computer DMZ",
                    "type": "docker",
                    "container_name": "dummy",
                    "dependencies": ["llama"],
                },
            }
        },
    )
    write_toml_atomic(
        config / "profiles.toml",
        {
            "harnesses": {
                "computer_dmz": {
                    "display_name": "Computer DMZ",
                    "services": ["computer_dmz"],
                    "open_service": "computer_dmz",
                }
            },
            "profiles": {
                "agent": {
                    "display_name": "Agent",
                    "services": ["llama"],
                    "harness": "computer_dmz",
                    # V0.x migration residue. This must not invalidate V1.
                    "open_service": "computer_dmz",
                }
            },
        },
    )
    write_toml_atomic(config / "models.toml", {"models": {"m": {"path": str(model_file)}}})

    loaded = load_app_config(config)
    assert loaded.profiles["agent"].harness == "computer_dmz"
    assert loaded.profiles["agent"].open_service is None
    assert loaded.harnesses["computer_dmz"].open_service == "computer_dmz"


# --- v1.2 -> v1.3 upgrade -------------------------------------------------------

import hashlib
import shutil

from ai_control_centre.domain import SetupState


def _v12_completed_config(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "examples" / "reference"
    target = tmp_path / "config"
    shutil.copytree(source, target)
    settings_path = target / "settings.toml"
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8").replace(
            "completed = false\nschema_version = 2", "completed = true\nschema_version = 2"
        ),
        encoding="utf-8",
    )
    return target


def _digests(config_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(config_dir.glob("*.toml"))
    }


def test_v12_completed_setup_stays_usable_with_review_pending(tmp_path: Path):
    target = _v12_completed_config(tmp_path)
    config = load_app_config(target)
    settings = config.settings
    assert settings.setup_schema_version == 2
    assert settings.setup_completed is True
    assert settings.setup_state == SetupState.UPGRADE_REVIEW
    assert settings.setup_review_pending is True


def test_v12_environments_are_inferred_from_existing_profiles(tmp_path: Path):
    target = _v12_completed_config(tmp_path)
    config = load_app_config(target)
    assert config.settings.environments_explicit is False
    assert dict(config.settings.environments) == {
        "coding": True, "chat": True, "agent": True, "image": True,
    }

    profiles = target / "profiles.toml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8").split("[profiles.image]")[0], encoding="utf-8"
    )
    assert load_app_config(target).settings.environments["image"] is False


def test_v12_profiles_retain_models_and_load_never_rewrites_files(tmp_path: Path):
    target = _v12_completed_config(tmp_path)
    before = _digests(target)
    for _ in range(2):  # repeatable without changes
        config = load_app_config(target)
        assert config.profiles["coding"].default_model == "qwen3_coder_30b_heretic"
        assert config.profiles["chat"].default_model == "qwen_qwen3_30b_a3b_instruct_2507_q4_k_m"
        assert config.profiles["agent"].harness == "computer_dmz"
    assert _digests(target) == before


def test_profile_with_missing_model_keeps_selection_and_still_loads(tmp_path: Path):
    target = _v12_completed_config(tmp_path)
    profiles = target / "profiles.toml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8").replace(
            'default_model = "qwen3_coder_30b_heretic"', 'default_model = "moved_model"'
        ),
        encoding="utf-8",
    )
    config = load_app_config(target)
    assert config.profiles["coding"].default_model == "moved_model"
    assert "moved_model" not in config.models


def test_unrecognised_newer_setup_schema_requires_setup(tmp_path: Path):
    settings = tmp_path / "settings.toml"
    write_toml_atomic(
        settings,
        {
            "paths": {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")},
            "setup": {"completed": True, "schema_version": CURRENT_SETUP_SCHEMA + 1},
        },
    )
    loaded = load_settings(settings)
    assert loaded.setup_state == SetupState.INCOMPLETE
    assert loaded.setup_completed is False
