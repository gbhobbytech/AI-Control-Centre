from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings
from ai_control_centre.domain import (
    DockerServiceConfig,
    HarnessConfig,
    ModelConfig,
    ProcessServiceConfig,
    ProfileConfig,
    TASK_IDS,
)
from ai_control_centre.environments import (
    Availability,
    evaluate_profile,
    evaluate_tasks,
    profile_requires_model,
)


def _model(model_id: str, *, complete: bool = True, tuned: bool = True) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        display_name=model_id.upper(),
        path=Path(f"/models/{model_id}.gguf"),
        complete=complete,
        tuning_reviewed=tuned,
    )


def _config(
    *,
    models: dict[str, ModelConfig] | None = None,
    profiles: dict[str, ProfileConfig] | None = None,
    environments: dict[str, bool] | None = None,
    harnesses: dict[str, HarnessConfig] | None = None,
) -> AppConfig:
    services = {
        "llama": ProcessServiceConfig(
            id="llama", display_name="llama.cpp", executable=Path(sys.executable),
            args=("-m", "{model.path}"), cwd=Path("/"),
        ),
        "comfyui": ProcessServiceConfig(
            id="comfyui", display_name="ComfyUI", executable=Path(sys.executable),
            args=("main.py",), cwd=Path("/"),
        ),
        "dmz": DockerServiceConfig(
            id="dmz", display_name="DMZ", container_name="dmz", dependencies=("llama",),
        ),
    }
    if profiles is None:
        profiles = {
            "coding": ProfileConfig("coding", "Coding", ("llama",), "llama", "m"),
            "chat": ProfileConfig("chat", "Chat", ("llama",), "llama", "m"),
            "agent": ProfileConfig("agent", "Agent", ("llama",), None, "m", "h"),
            "image": ProfileConfig("image", "Image", ("comfyui",), "comfyui"),
        }
    if harnesses is None:
        harnesses = {"h": HarnessConfig("h", "Harness", ("dmz",), "dmz")}
    settings = Settings(log_dir=Path("/tmp/logs"), runtime_dir=Path("/tmp/run"))
    if environments is None:
        settings = replace(settings, environments={t: t in profiles for t in TASK_IDS})
    else:
        settings = replace(settings, environments=environments, environments_explicit=True)
    return AppConfig(
        settings=settings,
        services=services,
        profiles=profiles,
        harnesses=harnesses,
        models={"m": _model("m")} if models is None else models,
    )


def test_all_tasks_ready_when_complete():
    statuses = evaluate_tasks(_config())
    assert list(statuses) == list(TASK_IDS)
    assert all(status.available for status in statuses.values())


def test_image_needs_no_gguf_model():
    config = _config(models={})
    image = evaluate_profile(config, "image")
    assert image.available
    assert profile_requires_model(config, config.profiles["image"]) is False
    assert profile_requires_model(config, config.profiles["chat"]) is True
    assert evaluate_profile(config, "chat").availability == Availability.MODEL_NOT_FOUND


def test_disabled_environment_is_distinct_from_incomplete():
    config = _config(environments={"coding": True, "chat": False, "agent": False, "image": True})
    chat = evaluate_profile(config, "chat")
    assert chat.enabled is False
    assert chat.availability == Availability.DISABLED
    assert "turned off" in chat.reason
    assert evaluate_profile(config, "coding").available


def test_enabled_environment_without_profile_is_not_configured():
    profiles = {"chat": ProfileConfig("chat", "Chat", ("llama",), "llama", "m")}
    config = _config(
        profiles=profiles,
        environments={"coding": False, "chat": True, "agent": False, "image": True},
    )
    image = evaluate_profile(config, "image")
    assert image.enabled is True
    assert image.availability == Availability.NOT_CONFIGURED


@pytest.mark.parametrize(
    ("models", "expected", "reason"),
    [
        ({}, Availability.MODEL_NOT_FOUND, "Chat model 'm' was not found."),
        ({"m": _model("m", complete=False)}, Availability.MODEL_INCOMPLETE,
         "Chat needs a complete default model."),
        ({"m": _model("m", tuned=False)}, Availability.NEEDS_TUNING,
         "M needs launch configuration."),
    ],
)
def test_model_problems_are_distinguished(models, expected, reason):
    status = evaluate_profile(_config(models=models), "chat")
    assert status.availability == expected
    assert status.reason.startswith(reason)


def test_prompt_helper_profile_is_not_a_task_environment():
    profiles = {"prompt_helper": ProfileConfig("prompt_helper", "Prompt Helper", ("llama",))}
    config = _config(
        profiles=profiles,
        environments={"coding": False, "chat": False, "agent": False, "image": False},
    )
    assert evaluate_profile(config, "prompt_helper").availability != Availability.DISABLED


def _v12_task_capability(config: AppConfig, profile_id: str) -> tuple[bool, str]:
    """Copy of the v1.2.0 ControlCentreWindow._task_capability logic."""
    profile = config.profiles.get(profile_id)
    if profile is None:
        return False, f"{profile_id.title()} is not configured."
    for service_id in profile.services:
        if service_id not in config.services:
            return False, f"Required service '{service_id}' is not configured."
    if profile_id in {"coding", "chat", "agent"}:
        model_id = profile.default_model
        model = config.models.get(model_id) if model_id else None
        if model is None or not model.complete:
            return False, f"{profile.display_name} needs a complete default model."
        if not model.tuning_reviewed:
            return False, f"{model.display_name} needs launch configuration."
    if profile_id == "agent":
        if not profile.harness:
            return False, "Agent unavailable - no Agent harness is configured."
        harness = config.harnesses.get(profile.harness)
        if harness is None:
            return False, "Agent unavailable - the selected harness is missing."
        missing = [sid for sid in harness.services if sid not in config.services]
        if missing:
            return False, "Agent harness has missing service(s): " + ", ".join(missing)
    return True, "Ready"


_V12_SCENARIOS = {
    "complete": {},
    "untuned": {"models": {"m": _model("m", tuned=False)}},
    "incomplete": {"models": {"m": _model("m", complete=False)}},
    "no agent harness": {"profiles": {
        "agent": ProfileConfig("agent", "Agent", ("llama",), None, "m", None)}},
    "harness missing": {"harnesses": {}},
    "harness service missing": {"harnesses": {"h": HarnessConfig("h", "H", ("gone",))}},
    "profile service missing": {"profiles": {
        "chat": ProfileConfig("chat", "Chat", ("gone",), None, "m")}},
    "no profiles": {"profiles": {}},
    "no model selected": {"profiles": {
        "coding": ProfileConfig("coding", "Coding", ("llama",), "llama", None)}},
}


@pytest.mark.parametrize("scenario", sorted(_V12_SCENARIOS))
def test_inferred_environments_match_v12_capability_messages(scenario):
    config = _config(**_V12_SCENARIOS[scenario])
    for task_id in TASK_IDS:
        status = evaluate_profile(config, task_id)
        assert (status.available, status.reason) == _v12_task_capability(config, task_id), task_id


def test_missing_software_is_reported_before_model_problems(tmp_path):
    config = _config(models={})
    gone = tmp_path / "llama.cpp/build/bin/llama-server"
    services = dict(config.services)
    services["llama"] = replace(services["llama"], executable=gone)
    config = replace(config, services=services)
    chat = evaluate_profile(config, "chat")
    assert chat.availability == Availability.MISSING_SOFTWARE
    assert chat.reason == f"llama.cpp not found at {gone}."
    assert chat.setup_page == "software"
    assert evaluate_profile(config, "image").available  # ComfyUI is unaffected


def test_missing_working_folder_is_reported(tmp_path):
    config = _config()
    services = dict(config.services)
    services["comfyui"] = replace(services["comfyui"], cwd=tmp_path / "ComfyUI")
    image = evaluate_profile(replace(config, services=services), "image")
    assert image.availability == Availability.MISSING_SOFTWARE
    assert image.reason == f"ComfyUI folder not found: {tmp_path / 'ComfyUI'}."


def test_every_problem_points_to_a_setup_page():
    from ai_control_centre.environments import SETUP_PAGE

    assert set(SETUP_PAGE) == set(Availability) - {Availability.READY}
    assert SETUP_PAGE[Availability.NEEDS_TUNING] == "launch"
    assert SETUP_PAGE[Availability.DISABLED] == "environments"
