from __future__ import annotations

import hashlib
import shutil
import tomllib
from pathlib import Path

import pytest

from ai_control_centre.config import CURRENT_SETUP_SCHEMA, load_app_config, load_services, load_settings
from ai_control_centre.discovery import validate_comfyui, validate_llama_executable
from ai_control_centre.domain import SetupState
from ai_control_centre.environments import evaluate_profile
from ai_control_centre.preferences import (
    configure_image_environment,
    keep_current_setup,
    save_preferences,
    update_service_paths,
    write_toml_atomic,
)


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
        agent_harness=None,
        agent_base_services=("llama",),
        harnesses={},
        prompt_model="small",
        prompt_enabled=True,
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
    assert config.settings.setup_schema_version == CURRENT_SETUP_SCHEMA
    assert config.settings.prompt_workshop.model_strategy == "manual"
    assert config.settings.prompt_workshop.preferred_model == "small"
    assert config.settings.prompt_workshop.enabled is True
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


def test_saving_agent_as_unconfigured_removes_legacy_runtime_services(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    save_preferences(
        config_dir,
        model_roots=(tmp_path / "models",),
        task_models={"coding": "small", "chat": "large", "agent": None},
        agent_harness=None,
        agent_base_services=("llama",),
        harnesses={},
        prompt_model=None,
        prompt_enabled=False,
        prompt_auto_lightest=True,
        prompt_processing_mode="auto",
        prompt_keep_loaded=True,
        prompt_gpu_layers=None,
        prompt_context_length=4096,
        prompt_port=8081,
        prompt_startup_timeout=60,
        prompt_cache_type_k="q8_0",
        prompt_cache_type_v="q8_0",
    )
    loaded = load_app_config(config_dir)
    assert loaded.profiles["agent"].services == ("llama",)
    assert loaded.profiles["agent"].harness is None
    assert loaded.profiles["agent"].default_model is None
    assert loaded.harnesses == {}
    assert loaded.settings.prompt_workshop.enabled is False


# --- v1.3.0 environment persistence --------------------------------------------


def _save(config_dir: Path, tmp_path: Path, **extra) -> None:
    save_preferences(
        config_dir,
        model_roots=(tmp_path / "models",),
        task_models={"coding": "large", "chat": "small", "agent": "small"},
        agent_harness=None,
        agent_base_services=("llama",),
        harnesses={},
        prompt_model=None,
        prompt_enabled=False,
        prompt_auto_lightest=True,
        prompt_processing_mode="cpu",
        prompt_keep_loaded=False,
        prompt_gpu_layers=None,
        prompt_context_length=4096,
        prompt_port=8081,
        prompt_startup_timeout=60,
        prompt_cache_type_k="q8_0",
        prompt_cache_type_v="q8_0",
        **extra,
    )


def _settings_raw(config_dir: Path) -> dict:
    with (config_dir / "settings.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_environment_selections_round_trip(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    selection = {"coding": False, "chat": True, "agent": False, "image": True}
    _save(config_dir, tmp_path, environments=selection)
    settings = load_app_config(config_dir).settings
    assert settings.environments_explicit is True
    assert dict(settings.environments) == selection
    assert settings.setup_state == SetupState.COMPLETE


def test_saving_without_environments_leaves_existing_selection(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    _save(config_dir, tmp_path, environments={"chat": True, "image": True})
    raw = _settings_raw(config_dir)
    raw["environments"]["future_task"] = True
    write_toml_atomic(config_dir / "settings.toml", raw)

    _save(config_dir, tmp_path)
    environments = _settings_raw(config_dir)["environments"]
    assert environments == {
        "coding": False, "chat": True, "agent": False, "image": True, "future_task": True,
    }


def test_invalid_environment_selection_is_rejected_before_writing(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    before = (config_dir / "settings.toml").read_bytes()
    with pytest.raises(ValueError, match="Unknown environment"):
        _save(config_dir, tmp_path, environments={"audio": True})
    assert (config_dir / "settings.toml").read_bytes() == before


def _mark_v12_completed(config_dir: Path) -> None:
    raw = _settings_raw(config_dir)
    raw["setup"] = {"completed": True, "schema_version": 2}
    write_toml_atomic(config_dir / "settings.toml", raw)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_keep_current_setup_changes_only_environments_and_schema(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    _mark_v12_completed(config_dir)
    before_settings = _settings_raw(config_dir)
    other_files = {
        name: _digest(config_dir / name) for name in ("services.toml", "profiles.toml", "models.toml")
    }
    before = load_app_config(config_dir)
    assert before.settings.setup_state == SetupState.UPGRADE_REVIEW

    kept = keep_current_setup(config_dir)

    assert kept == {"coding": True, "chat": True, "agent": True, "image": True}
    after_settings = _settings_raw(config_dir)
    assert after_settings.pop("environments") == kept
    assert after_settings.pop("setup") == {"completed": True, "schema_version": CURRENT_SETUP_SCHEMA}
    before_settings.pop("setup")
    assert after_settings == before_settings
    assert {name: _digest(config_dir / name) for name in other_files} == other_files

    after = load_app_config(config_dir)
    assert after.settings.setup_state == SetupState.COMPLETE
    assert after.profiles == before.profiles
    assert after.models == before.models
    assert after.services == before.services


def test_keep_current_setup_is_repeatable(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    _mark_v12_completed(config_dir)
    keep_current_setup(config_dir)
    once = _digest(config_dir / "settings.toml")
    keep_current_setup(config_dir)
    assert _digest(config_dir / "settings.toml") == once


def test_keep_current_setup_requires_a_completed_setup(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    before = _digest(config_dir / "settings.toml")
    with pytest.raises(ValueError, match="run the Setup Wizard"):
        keep_current_setup(config_dir)
    assert _digest(config_dir / "settings.toml") == before


# --- v1.3.0 Phase 2: service paths and Image environment -----------------------

_REPO = Path(__file__).resolve().parents[1]


def _starter_config(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(_REPO / "packaging" / "default-config", target)
    return target


def _fake_comfyui(root: Path) -> Path:
    (root / "comfy").mkdir(parents=True)
    (root / "main.py").write_text("# ComfyUI\n", encoding="utf-8")
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    (root / "venv" / "lib" / "python3.12" / "site-packages" / "torch").mkdir(parents=True)
    return root


def test_update_service_paths_changes_only_those_lines(tmp_path: Path):
    config_dir = _starter_config(tmp_path)
    services = config_dir / "services.toml"
    before = services.read_text(encoding="utf-8").splitlines()

    update_service_paths(
        config_dir, "llama",
        executable="~/opt/llama.cpp/build/bin/llama-server", cwd="~/opt/llama.cpp",
    )

    after = services.read_text(encoding="utf-8").splitlines()
    changed = [(old, new) for old, new in zip(before, after) if old != new]
    assert len(before) == len(after)
    assert changed == [
        ('executable = "~/ai/apps/llama.cpp/build/bin/llama-server"',
         'executable = "~/opt/llama.cpp/build/bin/llama-server"'),
        ('cwd = "~/ai/apps/llama.cpp"', 'cwd = "~/opt/llama.cpp"'),
    ]
    assert after[0].startswith("#")  # comments kept
    llama = load_app_config(config_dir).services["llama"]
    assert llama.executable == Path("~/opt/llama.cpp/build/bin/llama-server").expanduser()


def test_update_service_paths_handles_quoted_headers_and_missing_keys(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    services = config_dir / "services.toml"
    services.write_text(
        '# hand written\n["services"."web"]\n"display_name" = "Web"\ntype = "process"\n'
        'executable = "/bin/true"\n\n[services.web.health]\nurl = "http://127.0.0.1:1/"\n',
        encoding="utf-8",
    )
    update_service_paths(config_dir, "web", cwd="/srv/web")
    text = services.read_text(encoding="utf-8")
    assert '["services"."web"]\ncwd = "/srv/web"\n' in text
    assert load_services(services)["web"].cwd == Path("/srv/web")


def test_update_service_paths_rejects_unknown_or_non_process_services(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    services = config_dir / "services.toml"
    services.write_text(
        '[services.box]\ndisplay_name = "Box"\ntype = "docker"\ncontainer_name = "box"\n', encoding="utf-8"
    )
    before = services.read_bytes()
    with pytest.raises(ValueError, match=r"no \[services\.missing\] table"):
        update_service_paths(config_dir, "missing", executable="/bin/true")
    with pytest.raises(ValueError, match="would make services.toml invalid"):
        update_service_paths(config_dir, "box", executable="/bin/true")
    assert services.read_bytes() == before


def test_configure_image_environment_on_fresh_install(tmp_path: Path):
    config_dir = _starter_config(tmp_path)
    comfy = validate_comfyui(_fake_comfyui(tmp_path / "apps" / "ComfyUI"))

    result = configure_image_environment(config_dir, comfy, home=tmp_path / "elsewhere")

    assert result.service_added and result.profile_added and result.environment_enabled
    assert result.service_differences == ()
    config = load_app_config(config_dir)
    service = config.services["comfyui"]
    assert service.executable == tmp_path / "apps/ComfyUI/venv/bin/python"
    assert service.cwd == tmp_path / "apps/ComfyUI"
    assert config.profiles["image"].services == ("comfyui",)
    assert config.profiles["image"].open_service == "comfyui"
    assert config.settings.environment_enabled("image")
    # Other environments keep what they were before (inferred from profiles).
    assert config.settings.environments["chat"] is True
    assert evaluate_profile(config, "image").available


def test_configure_image_environment_is_repeatable(tmp_path: Path):
    config_dir = _starter_config(tmp_path)
    comfy = validate_comfyui(_fake_comfyui(tmp_path / "apps" / "ComfyUI"))
    configure_image_environment(config_dir, comfy, home=tmp_path / "elsewhere")
    digests = {p.name: p.read_bytes() for p in config_dir.glob("*.toml")}

    again = configure_image_environment(config_dir, comfy, home=tmp_path / "elsewhere")

    assert not (again.service_added or again.profile_added or again.environment_enabled)
    assert {p.name: p.read_bytes() for p in config_dir.glob("*.toml")} == digests


def test_configure_image_environment_keeps_existing_comfyui_and_reports_difference(tmp_path: Path):
    config_dir = tmp_path / "config"
    shutil.copytree(_REPO / "examples" / "reference", config_dir)
    services_before = (config_dir / "services.toml").read_bytes()
    comfy = validate_comfyui(_fake_comfyui(tmp_path / "ComfyUI"))

    result = configure_image_environment(config_dir, comfy, home=tmp_path / "elsewhere")

    assert not result.service_added and not result.profile_added
    assert set(result.service_differences) == {"executable", "cwd"}
    assert (config_dir / "services.toml").read_bytes() == services_before

    # After the user confirms the new installation:
    update_service_paths(config_dir, result.service_id, executable=comfy.executable, cwd=comfy.cwd)
    service = load_app_config(config_dir).services["comfyui"]
    assert (service.executable, service.cwd) == (comfy.executable, comfy.cwd)


def test_configure_image_environment_rejects_unusable_installation(tmp_path: Path):
    config_dir = _starter_config(tmp_path)
    before = {p.name: p.read_bytes() for p in config_dir.glob("*.toml")}
    with pytest.raises(ValueError, match="cannot be used"):
        configure_image_environment(config_dir, validate_comfyui(tmp_path / "missing"))
    with pytest.raises(ValueError, match="needs a ComfyUI installation"):
        configure_image_environment(config_dir, validate_llama_executable(tmp_path / "x"))
    assert {p.name: p.read_bytes() for p in config_dir.glob("*.toml")} == before


def test_save_preferences_never_creates_empty_profiles(tmp_path: Path):
    config_dir = _make_config(tmp_path)
    raw = tomllib.loads((config_dir / "profiles.toml").read_text(encoding="utf-8"))
    for profile_id in ("chat", "agent"):
        raw["profiles"].pop(profile_id)
    write_toml_atomic(config_dir / "profiles.toml", raw)

    _save(config_dir, tmp_path)

    profiles = load_app_config(config_dir).profiles
    assert "chat" not in profiles and "agent" not in profiles
    assert profiles["coding"].default_model == "large"


def test_add_service_conflict_group_keeps_existing_groups_and_comments(tmp_path: Path):
    from ai_control_centre.preferences import add_service_conflict_group

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    services = config_dir / "services.toml"
    services.write_text(
        "# mine\n[services.web]\ndisplay_name = \"Web\"\ntype = \"process\"\nexecutable = \"/bin/true\"\ncwd = \"/tmp\"\n"
        "conflict_groups = [\"ports\"]  # keep me\n",
        encoding="utf-8",
    )
    assert add_service_conflict_group(config_dir, "web", "large-gpu-workload") is True
    text = services.read_text(encoding="utf-8")
    assert text.startswith("# mine\n")
    assert load_services(services)["web"].conflict_groups == ("ports", "large-gpu-workload")
    before = services.read_bytes()
    assert add_service_conflict_group(config_dir, "web", "large-gpu-workload") is False
    assert services.read_bytes() == before


def test_add_service_conflict_group_refuses_multiline_arrays(tmp_path: Path):
    from ai_control_centre.preferences import add_service_conflict_group

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    services = config_dir / "services.toml"
    services.write_text(
        "[services.web]\ndisplay_name = \"Web\"\ntype = \"process\"\nexecutable = \"/bin/true\"\ncwd = \"/tmp\"\n"
        "conflict_groups = [\n  \"ports\",\n]\n",
        encoding="utf-8",
    )
    before = services.read_bytes()
    with pytest.raises(ValueError, match="spans several lines; edit it by hand"):
        add_service_conflict_group(config_dir, "web", "large-gpu-workload")
    assert services.read_bytes() == before
