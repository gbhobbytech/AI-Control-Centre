"""Task-environment availability, independent of the GUI toolkit.

An environment can be *enabled* (the user intends to run it) without being
*available* (its configuration is complete enough to launch). This module
explains which, so the main window and later the Setup Wizard can show the
same reason and tests can run without Tk.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import AppConfig, environment_services
from .domain import ProcessServiceConfig, ProfileConfig, TASK_IDS


class Availability(str, Enum):
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    DISABLED = "disabled"
    MISSING_SERVICE = "missing_service"
    MISSING_SOFTWARE = "missing_software"
    NO_MODEL_SELECTED = "no_model_selected"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_INCOMPLETE = "model_incomplete"
    NEEDS_TUNING = "needs_tuning"
    MISSING_HARNESS = "missing_harness"


# Setup Wizard page that fixes each kind of problem.
SETUP_PAGE = {
    Availability.NOT_CONFIGURED: "environments",
    Availability.DISABLED: "environments",
    Availability.MISSING_SERVICE: "software",
    Availability.MISSING_SOFTWARE: "software",
    Availability.NO_MODEL_SELECTED: "models",
    Availability.MODEL_NOT_FOUND: "models",
    Availability.MODEL_INCOMPLETE: "models",
    Availability.NEEDS_TUNING: "launch",
    Availability.MISSING_HARNESS: "models",
}


@dataclass(frozen=True)
class EnvironmentStatus:
    profile_id: str
    enabled: bool
    availability: Availability
    reason: str

    @property
    def available(self) -> bool:
        return self.availability == Availability.READY

    @property
    def setup_page(self) -> str | None:
        return SETUP_PAGE.get(self.availability)


def profile_requires_model(config: AppConfig, profile: ProfileConfig) -> bool:
    """True when a service the profile starts takes a GGUF model.

    Derived from the services' {model.*} placeholders rather than the task
    name, so an Image profile backed by ComfyUI never needs a model.
    """
    prompt_service = config.settings.prompt_workshop.service
    for service_id in environment_services(profile, config.services, config.harnesses):
        service = config.services[service_id]
        if (
            isinstance(service, ProcessServiceConfig)
            and service.uses_model
            and service_id != prompt_service
        ):
            return True
    return False


def missing_software(config: AppConfig, profile: ProfileConfig) -> str | None:
    """Describe the first program or folder a profile needs that no longer exists."""
    for service_id in sorted(environment_services(profile, config.services, config.harnesses)):
        service = config.services[service_id]
        if not isinstance(service, ProcessServiceConfig):
            continue
        if not service.executable.is_file():
            return f"{service.display_name} not found at {_home_relative(service.executable)}."
        if not service.cwd.is_dir():
            return f"{service.display_name} folder not found: {_home_relative(service.cwd)}."
    return None


def _home_relative(path) -> str:
    from .discovery import portable_path

    return portable_path(path)


def evaluate_profile(config: AppConfig, profile_id: str) -> EnvironmentStatus:
    is_task = profile_id in TASK_IDS
    enabled = config.settings.environment_enabled(profile_id) if is_task else True

    def status(availability: Availability, reason: str) -> EnvironmentStatus:
        return EnvironmentStatus(profile_id, enabled, availability, reason)

    profile = config.profiles.get(profile_id)
    if profile is None:
        return status(Availability.NOT_CONFIGURED, f"{profile_id.title()} is not configured.")
    if is_task and not enabled:
        return status(
            Availability.DISABLED,
            f"{profile.display_name} is turned off. Enable it in Setup to use it.",
        )
    for service_id in profile.services:
        if service_id not in config.services:
            return status(
                Availability.MISSING_SERVICE,
                f"Required service '{service_id}' is not configured.",
            )

    missing = missing_software(config, profile)
    if missing is not None:
        return status(Availability.MISSING_SOFTWARE, missing)

    if profile_requires_model(config, profile):
        model_id = profile.default_model
        if model_id is None:
            return status(
                Availability.NO_MODEL_SELECTED,
                f"{profile.display_name} needs a complete default model.",
            )
        model = config.models.get(model_id)
        if model is None:
            return status(
                Availability.MODEL_NOT_FOUND,
                f"{profile.display_name} model '{model_id}' was not found. "
                "Rescan models or choose another model in Settings.",
            )
        if not model.complete:
            return status(
                Availability.MODEL_INCOMPLETE,
                f"{profile.display_name} needs a complete default model.",
            )
        if not model.tuning_reviewed:
            return status(
                Availability.NEEDS_TUNING,
                f"{model.display_name} needs launch configuration.",
            )

    if profile_id == "agent":
        if not profile.harness:
            return status(
                Availability.MISSING_HARNESS,
                "Agent unavailable - no Agent harness is configured.",
            )
        harness = config.harnesses.get(profile.harness)
        if harness is None:
            return status(
                Availability.MISSING_HARNESS,
                "Agent unavailable - the selected harness is missing.",
            )
        missing = [sid for sid in harness.services if sid not in config.services]
        if missing:
            return status(
                Availability.MISSING_SERVICE,
                "Agent harness has missing service(s): " + ", ".join(missing),
            )
    return status(Availability.READY, "Ready")


def evaluate_tasks(config: AppConfig) -> dict[str, EnvironmentStatus]:
    return {task_id: evaluate_profile(config, task_id) for task_id in TASK_IDS}
