"""Setup Wizard behaviour (Phase 3), tested through the Tk-free setup_flow."""

from __future__ import annotations

import hashlib
import shutil
import tomllib
from pathlib import Path

import pytest

from ai_control_centre.config import load_app_config
from ai_control_centre.domain import SetupState
from ai_control_centre.environments import evaluate_profile
from ai_control_centre.preferences import save_model_tuning, write_toml_atomic
from ai_control_centre.setup_flow import (
    STEP_ENVIRONMENTS,
    STEP_LAUNCH,
    STEP_MODELS,
    STEP_PROMPT,
    STEP_SOFTWARE,
    SetupError,
    SetupSession,
)
from ai_control_centre.tuning import model_values

ROOT = Path(__file__).resolve().parents[1]


# --- Fixtures ----------------------------------------------------------------


def _program(path: Path, header: bytes = b"\x7fELF\x02\x01\x01") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"\0" * 32)
    path.chmod(0o755)
    return path


def _install_llama(home: Path) -> Path:
    return _program(home / "ai/apps/llama.cpp/build/bin/llama-server")


def _install_comfyui(home: Path) -> Path:
    root = home / "ComfyUI"
    (root / "comfy").mkdir(parents=True)
    (root / "main.py").write_text("# ComfyUI\n", encoding="utf-8")
    _program(root / "venv/bin/python", b"#!/bin/sh\n")
    (root / "venv/lib/python3.12/site-packages/torch").mkdir(parents=True)
    return root


def _add_model(models_dir: Path, name: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.gguf").write_bytes(b"GGUF" + b"\0" * 64)


def _fresh_install(tmp_path: Path, *, models: tuple[str, ...] = ()) -> tuple[Path, Path, Path]:
    """Starter configuration as seeded by the installer, with runtime paths in tmp."""
    home = tmp_path / "home"
    home.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in models:
        _add_model(models_dir, name)
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "packaging" / "default-config", config_dir)
    with (config_dir / "settings.toml").open("rb") as fh:
        settings = tomllib.load(fh)
    settings["paths"] = {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")}
    settings["model_roots"] = {"llm": [str(models_dir)]}
    write_toml_atomic(config_dir / "settings.toml", settings)
    return config_dir, home, models_dir


def _session(config_dir: Path, home: Path) -> SetupSession:
    return SetupSession(config_dir, home=home, path_env="")


def _tune(session: SetupSession, model_id: str) -> None:
    """What the tuning window's Save does (saved immediately, as in v1.2)."""
    model = session.models[model_id]
    save_model_tuning(session.config_dir, model, model_values(model))
    session.refresh_models()


def _digests(config_dir: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(config_dir.glob("*.toml"))}


def _only(session: SetupSession, *tasks: str) -> None:
    session.draft.environments = {task: task in tasks for task in ("coding", "chat", "agent", "image")}


# --- Fresh installs ------------------------------------------------------------


def test_fresh_install_supports_image_only_setup(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)  # no GGUF models at all
    _install_comfyui(home)
    session = _session(config_dir, home)
    assert session.setup_state == SetupState.INCOMPLETE
    assert session.draft.environments["image"] is True  # ComfyUI was discovered
    _only(session, "image")

    assert session.issues() == []
    result = session.apply()

    assert result.environments == {"coding": False, "chat": False, "agent": False, "image": True}
    config = load_app_config(config_dir)
    assert config.settings.setup_state == SetupState.COMPLETE
    assert config.services["comfyui"].cwd == home / "ComfyUI"
    assert evaluate_profile(config, "image").available
    assert not evaluate_profile(config, "chat").available  # turned off, not broken


def test_fresh_install_supports_chat_only_setup(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path, models=("qwen4b",))
    _install_llama(home)
    session = _session(config_dir, home)
    _only(session, "chat")
    session.scan_models(session.draft.model_roots)
    session.draft.task_models["chat"] = "qwen4b"

    assert [i.step for i in session.issues()] == [STEP_LAUNCH]
    _tune(session, "qwen4b")
    assert session.issues() == []
    session.apply()

    config = load_app_config(config_dir)
    assert config.settings.environments == {"coding": False, "chat": True, "agent": False, "image": False}
    assert config.profiles["chat"].default_model == "qwen4b"
    assert config.services["llama"].executable == home / "ai/apps/llama.cpp/build/bin/llama-server"
    assert evaluate_profile(config, "chat").available


def test_chat_and_coding_may_share_one_model(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path, models=("qwen4b",))
    _install_llama(home)
    session = _session(config_dir, home)
    _only(session, "coding", "chat")
    session.scan_models(session.draft.model_roots)
    session.draft.task_models.update(coding="qwen4b", chat="qwen4b")
    assert session.selected_models() == ["qwen4b"]  # tuned once
    _tune(session, "qwen4b")
    session.apply()

    config = load_app_config(config_dir)
    assert config.profiles["coding"].default_model == config.profiles["chat"].default_model == "qwen4b"
    assert evaluate_profile(config, "coding").available and evaluate_profile(config, "chat").available


def test_agent_remains_optional(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path, models=("qwen4b",))
    _install_llama(home)
    session = _session(config_dir, home)
    assert session.draft.environments["agent"] is False  # no harness on a fresh install
    _only(session, "chat")
    session.scan_models(session.draft.model_roots)
    session.draft.task_models["chat"] = "qwen4b"
    _tune(session, "qwen4b")
    session.apply()
    assert load_app_config(config_dir).settings.environments["agent"] is False


def test_unselected_environments_do_not_block_completion(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)  # no llama.cpp, no models
    _install_comfyui(home)
    session = _session(config_dir, home)
    _only(session, "image")
    # Coding/Chat/Agent are unselected, so missing llama.cpp and models are not problems.
    assert session.issues() == []
    session.apply()


def test_selected_environment_with_problems_is_saved_switched_off(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)  # no llama.cpp, no models
    _install_comfyui(home)
    session = _session(config_dir, home)
    _only(session, "chat", "image")

    review = session.review()
    assert "chat" in review.switched_off
    assert any("will be saved switched off" in line for _, lines in review.sections for line in lines)

    result = session.apply()
    assert result.environments["chat"] is False and result.environments["image"] is True
    # The starter config's llama.cpp path does not exist on this machine.
    assert "llama.cpp:" in result.switched_off["chat"][0]
    assert "does not exist" in result.switched_off["chat"][0]


def test_nothing_is_saved_when_no_selected_environment_can_work(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)
    before = _digests(config_dir)
    session = _session(config_dir, home)
    _only(session, "chat")
    with pytest.raises(SetupError, match="None of the selected environments can be set up yet"):
        session.apply()
    _only(session)
    assert session.issues(STEP_ENVIRONMENTS)[0].message == "Choose at least one environment to set up."
    assert _digests(config_dir) == before


# --- Identifying problems ------------------------------------------------------


def test_wizard_identifies_missing_executables_and_models(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)
    session = _session(config_dir, home)
    _only(session, "chat", "image")
    software = [i.message for i in session.issues(STEP_SOFTWARE)]
    assert any(m.startswith("llama.cpp:") and "does not exist" in m for m in software)
    assert any("ComfyUI was not found" in m for m in software)
    session.draft.llama = None  # nothing configured or discovered at all
    assert "llama.cpp (llama-server) was not found" in session.issues(STEP_SOFTWARE)[0].message
    assert [i.message for i in session.issues(STEP_MODELS)] == ["Choose a model for Chat."]

    session.choose_llama_path(home / "nope/llama-server")
    assert "does not exist" in session.issues(STEP_SOFTWARE)[0].message
    session.draft.task_models["chat"] = "gone"
    assert "was not found" in session.issues(STEP_MODELS)[0].message


def test_comfyui_port_conflicts_are_reported(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path)
    _install_comfyui(home)
    session = _session(config_dir, home)
    _only(session, "image")
    session.draft.comfyui_port = 8080  # llama.cpp's port in the starter config
    assert "already used by another configured service" in session.issues(STEP_SOFTWARE)[0].message
    session.draft.comfyui_port = 70000
    assert "must be from 1 to 65535" in session.issues(STEP_SOFTWARE)[0].message


def test_prompt_helper_without_its_service_is_explained(tmp_path: Path):
    config_dir, home, _ = _fresh_install(tmp_path, models=("qwen4b",))
    _install_llama(home)
    session = _session(config_dir, home)
    _only(session, "chat")
    session.draft.prompt.enabled = True
    assert "Leave Prompt Helper off for now" in session.issues(STEP_PROMPT)[0].message
    with pytest.raises(SetupError, match="Prompt Helper needs"):
        session.apply()


# --- Existing configurations -------------------------------------------------


def _existing_setup(tmp_path: Path) -> tuple[Path, Path]:
    """A completed v1.3 setup with Coding, Chat and Image."""
    config_dir, home, _ = _fresh_install(tmp_path, models=("qwen4b", "qwen14b"))
    _install_llama(home)
    _install_comfyui(home)
    session = _session(config_dir, home)
    _only(session, "coding", "chat", "image")
    session.scan_models(session.draft.model_roots)
    session.draft.task_models.update(coding="qwen14b", chat="qwen4b")
    _tune(session, "qwen4b")
    _tune(session, "qwen14b")
    session.apply()
    return config_dir, home


def test_existing_settings_are_prepopulated_when_reopened(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    session = _session(config_dir, home)
    draft = session.draft
    assert draft.environments == {"coding": True, "chat": True, "agent": False, "image": True}
    assert draft.task_models["coding"] == "qwen14b" and draft.task_models["chat"] == "qwen4b"
    assert draft.llama.path == home / "ai/apps/llama.cpp/build/bin/llama-server"
    assert draft.comfyui.path == home / "ComfyUI"
    assert draft.comfyui_port == 8188 and not session.comfyui_port_editable
    assert session.issues() == [] and session.path_changes() == []


def test_cancelling_setup_does_not_alter_saved_configuration(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    before = _digests(config_dir)
    session = _session(config_dir, home)
    _only(session, "image")
    session.draft.task_models["chat"] = "qwen14b"
    session.scan_models([tmp_path / "elsewhere"])
    session.review()
    del session  # the window was closed without Save & Finish
    assert _digests(config_dir) == before


def test_finishing_setup_writes_valid_loadable_toml(tmp_path: Path):
    config_dir, _ = _existing_setup(tmp_path)
    for path in config_dir.glob("*.toml"):
        tomllib.loads(path.read_text(encoding="utf-8"))
    config = load_app_config(config_dir)
    assert config.settings.setup_state == SetupState.COMPLETE
    assert not list(config_dir.parent.glob(".acc-setup-*"))  # staging cleaned up
    assert not list(config_dir.glob("*.tmp"))


def test_existing_configuration_can_be_edited_without_starting_over(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    services_before = (config_dir / "services.toml").read_bytes()
    session = _session(config_dir, home)
    session.draft.task_models["chat"] = "qwen14b"  # change one thing only
    result = session.apply()

    config = load_app_config(config_dir)
    assert config.profiles["chat"].default_model == "qwen14b"
    assert config.profiles["coding"].default_model == "qwen14b"
    assert config.settings.environment_enabled("image")
    assert (config_dir / "services.toml").read_bytes() == services_before
    assert "services.toml" not in result.files_changed


def test_turning_an_environment_off_keeps_its_model_choice(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    session = _session(config_dir, home)
    session.draft.environments["coding"] = False
    session.draft.task_models["coding"] = None
    session.apply()
    config = load_app_config(config_dir)
    assert config.settings.environments["coding"] is False
    assert config.profiles["coding"].default_model == "qwen14b"


def test_confirmed_new_installation_path_is_written_and_comments_kept(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    moved = _program(home / "llama.cpp/build/bin/llama-server")
    session = _session(config_dir, home)
    session.choose_llama_path(moved)
    [exe, cwd] = session.path_changes()
    assert (exe.field, exe.new, cwd.new) == ("executable", moved, home / "llama.cpp")
    session.apply()

    text = (config_dir / "services.toml").read_text(encoding="utf-8")
    assert text.startswith("# Starter service configuration.")
    assert load_app_config(config_dir).services["llama"].executable == moved


def test_changes_made_elsewhere_are_never_overwritten(tmp_path: Path):
    config_dir, home = _existing_setup(tmp_path)
    session = _session(config_dir, home)
    profiles = config_dir / "profiles.toml"
    profiles.write_text(profiles.read_text(encoding="utf-8") + "\n# edited by hand\n", encoding="utf-8")
    edited = _digests(config_dir)
    with pytest.raises(SetupError, match="profiles.toml was changed outside the Setup Wizard"):
        session.apply()
    assert _digests(config_dir) == edited


# --- Upgrade from v1.2 -----------------------------------------------------------


def _v12_setup(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "examples" / "reference", config_dir)
    settings = config_dir / "settings.toml"
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            "completed = false\nschema_version = 2", "completed = true\nschema_version = 2"
        ),
        encoding="utf-8",
    )
    return config_dir


def test_upgrade_welcome_keep_current_setup(tmp_path: Path):
    config_dir = _v12_setup(tmp_path)
    session = _session(config_dir, tmp_path / "home")
    assert session.upgrade_review_pending
    other = {name: digest for name, digest in _digests(config_dir).items() if name != "settings.toml"}

    kept = session.keep_current()

    assert kept == {"coding": True, "chat": True, "agent": True, "image": True}
    assert load_app_config(config_dir).settings.setup_state == SetupState.COMPLETE
    assert {k: v for k, v in _digests(config_dir).items() if k != "settings.toml"} == other


def test_upgrade_review_prefills_existing_v12_setup(tmp_path: Path):
    config_dir = _v12_setup(tmp_path)
    session = _session(config_dir, tmp_path / "home")
    draft = session.draft
    assert draft.environments == {"coding": True, "chat": True, "agent": True, "image": True}
    assert draft.task_models["coding"] == "qwen3_coder_30b_heretic"
    assert draft.agent_harness == "computer_dmz"
    assert session.comfyui_service_id() == "comfyui"
