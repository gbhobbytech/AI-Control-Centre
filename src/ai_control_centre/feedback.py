"""User-facing explanations for unavailable tasks and failed starts.

Independent of the GUI toolkit so the wording can be tested. The main window
shows these as button tooltips, a short "needs attention" line, and error
dialogs with a practical next step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .domain import ServiceState, TASK_IDS
from .environments import Availability, EnvironmentStatus, evaluate_profile

TASK_NAMES = {"coding": "Coding", "chat": "Chat", "agent": "Agent", "image": "Image"}

# What the Services panel shows for each state.
STATE_LABELS = {
    ServiceState.STOPPED: "Stopped",
    ServiceState.STARTING: "Starting",
    ServiceState.READY: "Ready",
    ServiceState.RUNNING_NOT_READY: "Starting",
    ServiceState.EXTERNAL: "External",
    ServiceState.ERROR: "Error",
    ServiceState.UNKNOWN: "Unknown",
}
EXTERNAL_DETAIL = "Started outside AI Control Centre; it will not be stopped from here."


@dataclass(frozen=True)
class Attention:
    task_id: str
    message: str
    setup_page: str | None


def task_tooltip(status: EnvironmentStatus) -> str:
    """Text shown when hovering over a task button."""
    if status.available:
        return f"Start {TASK_NAMES.get(status.profile_id, status.profile_id)}."
    if status.availability == Availability.DISABLED:
        return status.reason
    if status.setup_page:
        return f"{status.reason} Use Fix in Setup to correct it."
    return status.reason


def attention_items(config: AppConfig) -> list[Attention]:
    """Enabled environments that cannot run. Environments turned off are not listed."""
    items: list[Attention] = []
    for task_id in TASK_IDS:
        status = evaluate_profile(config, task_id)
        if status.available or not status.enabled or status.availability == Availability.DISABLED:
            continue
        if status.availability == Availability.NOT_CONFIGURED and not config.settings.environments_explicit:
            # A v1.2 configuration without this profile simply does not offer the task.
            continue
        items.append(Attention(task_id, f"{TASK_NAMES[task_id]}: {status.reason}", status.setup_page))
    return items


def attention_text(items: list[Attention]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0].message
    return "Needs attention - " + "  ".join(item.message for item in items)


@dataclass(frozen=True)
class FailureAdvice:
    title: str
    message: str
    hint: str
    log_path: Path | None
    setup_page: str | None
    service_id: str | None = None

    def as_text(self) -> str:
        return f"{self.message}\n\n{self.hint}" if self.hint else self.message


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def failure_advice(exc: Exception, config: AppConfig) -> FailureAdvice:
    """Name the affected service and suggest one practical next step."""
    from .manager import ServiceStartError

    start_error = next((item for item in _chain(exc) if isinstance(item, ServiceStartError)), None)
    service_id = start_error.service_id if start_error else None
    service = config.services.get(service_id) if service_id else None
    name = service.display_name if service is not None else None
    log_path = start_error.log_path if start_error else None
    text = " ".join(str(item) for item in _chain(exc)).lower()
    missing = any(isinstance(item, FileNotFoundError) for item in _chain(exc))

    setup_page = None
    if missing or "executable not found" in text or "working directory not found" in text:
        hint = "The program or its folder has moved or was removed. Choose it again in Setup (Software)."
        setup_page = "software"
        log_path = None
    elif "already in use" in text:
        hint = (
            "Another program is listening on that port. Close it, or change the port for this "
            "service in services.toml."
        )
    elif "exited immediately" in text or "process exited before readiness" in text or "exit code" in text:
        hint = (
            "It stopped while starting. Its log usually says why, for example a missing package, "
            "a bad launch setting or not enough GPU memory."
        )
    elif "startup timeout" in text:
        hint = (
            "It was still not ready when the startup time ran out. Large models can take longer to load; "
            "its log shows how far it got."
        )
    elif log_path is not None:
        hint = "Its log may show more detail."
    else:
        hint = ""

    title = f"{name} could not start" if name else "AI Control Centre"
    return FailureAdvice(
        title=title, message=str(exc), hint=hint, log_path=log_path, setup_page=setup_page, service_id=service_id
    )
