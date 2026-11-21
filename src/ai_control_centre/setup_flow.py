"""Setup Wizard logic, independent of the GUI toolkit.

The wizard window (settings_ui.SetupWizard) shows and edits a SetupDraft;
everything that decides what is valid, what the review says and how the
result is written lives here so it can be tested without a display.

Nothing is written until apply(). The one exception is a model's launch
settings, which the tuning window saves immediately when the user presses
its Save button (as in v1.2). apply() builds the new configuration in a
temporary copy, checks that it loads, and only then replaces the changed
files. If the configuration was changed elsewhere while the wizard was
open, apply() stops instead of overwriting it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig, ConfigError, load_app_config, load_models
from .discovery import (
    COMFYUI,
    LLAMA_CPP,
    DiscoveryReport,
    Installation,
    discover_installations,
    portable_path,
    validate_comfyui,
    validate_llama_executable,
)
from .domain import HarnessConfig, ModelConfig, ProcessServiceConfig, SetupState, TASK_IDS
from .harnesses import discover_agent_harnesses
from .health import url_endpoint
from .preferences import (
    append_toml_table,
    configure_image_environment,
    detect_setup_environment,
    keep_current_setup,
    save_preferences,
    update_service_paths,
)
from .presets import apply_preset, load_preset
from .tuning import is_llama_service

LLM_ENVIRONMENTS: tuple[str, ...] = ("coding", "chat", "agent")
ENVIRONMENT_NAMES = {"coding": "Coding", "chat": "Chat", "agent": "Agent", "image": "Image"}
ENVIRONMENT_DESCRIPTIONS = {
    "coding": "Coding assistant using a local GGUF model through llama.cpp.",
    "chat": "General chat using a local GGUF model through llama.cpp.",
    "agent": "Agent harness (for example a Docker workspace) driven by a local GGUF model.",
    "image": "Image generation with ComfyUI in your web browser. No GGUF model needed.",
}

STEP_ENVIRONMENTS = "environments"
STEP_SOFTWARE = "software"
STEP_MODELS = "models"
STEP_LAUNCH = "launch"
STEP_PROMPT = "prompt"

DEFAULT_COMFYUI_PORT = 8188
_GUARDED_FILES = ("settings.toml", "services.toml", "profiles.toml")
# Applied in this order so settings.toml, which marks setup complete, is last.
_COMMIT_ORDER = ("services.toml", "profiles.toml", "settings.toml")


class SetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Issue:
    step: str
    message: str
    # Environments this problem stops from working; empty for general problems.
    environments: tuple[str, ...] = ()


@dataclass
class PromptChoices:
    enabled: bool
    model: str | None
    auto_lightest: bool
    processing_mode: str
    keep_loaded: bool
    gpu_layers: int | None
    context_length: int
    port: int
    startup_timeout: float
    cache_type_k: str
    cache_type_v: str


@dataclass
class SetupDraft:
    environments: dict[str, bool]
    llama: Installation | None
    comfyui: Installation | None
    comfyui_port: int
    model_roots: list[Path]
    task_models: dict[str, str | None]
    agent_harness: str | None
    harnesses: dict[str, HarnessConfig]
    agent_base_services: tuple[str, ...]
    prompt: PromptChoices

    @property
    def needs_llama(self) -> bool:
        return any(self.environments.get(task, False) for task in LLM_ENVIRONMENTS)

    @property
    def needs_comfyui(self) -> bool:
        return self.environments.get("image", False)


@dataclass(frozen=True)
class PathChange:
    service_id: str
    field: str
    old: Path
    new: Path


@dataclass(frozen=True)
class Review:
    sections: tuple[tuple[str, tuple[str, ...]], ...]
    issues: tuple[Issue, ...]
    # Selected environments that will be saved switched off, with the reasons.
    switched_off: Mapping[str, tuple[str, ...]]

    def as_text(self) -> str:
        blocks = []
        for title, lines in self.sections:
            blocks.append(title + ":\n" + "\n".join(f"  {line}" for line in lines))
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class ApplyResult:
    environments: Mapping[str, bool]
    switched_off: Mapping[str, tuple[str, ...]]
    files_changed: tuple[str, ...]


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _same_path(a: Path | None, b: Path | None) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return a.expanduser().resolve() == b.expanduser().resolve()
    except OSError:
        return a.expanduser() == b.expanduser()


def _service_port(service: ProcessServiceConfig) -> int | None:
    if service.health is None:
        return None
    endpoint = url_endpoint(service.health.url)
    return endpoint[1] if endpoint else None


class SetupSession:
    """One run of the Setup Wizard over a configuration folder."""

    def __init__(
        self,
        config_dir: Path,
        *,
        home: Path | None = None,
        path_env: str | None = None,
        gpu_status: Callable[[], object] | None = None,
    ):
        self.config_dir = config_dir
        self.home = home
        self.path_env = path_env
        self.gpu_status = gpu_status
        self.config: AppConfig = load_app_config(config_dir)
        self._opened = {name: _digest(config_dir / name) for name in _GUARDED_FILES}
        self.report: DiscoveryReport = discover_installations(
            self.config.services, home=home, path_env=path_env
        )
        self.models: dict[str, ModelConfig] = dict(self.config.models)
        self.draft = self._initial_draft()

    # --- Initial state -------------------------------------------------------

    @property
    def setup_state(self) -> SetupState:
        return self.config.settings.setup_state

    @property
    def upgrade_review_pending(self) -> bool:
        return self.config.settings.setup_review_pending

    def llama_service_id(self) -> str | None:
        """The main llama.cpp service (not Prompt Helper's)."""
        prompt_service = self.config.settings.prompt_workshop.service
        candidates = [
            sid
            for sid, service in self.config.services.items()
            if sid != prompt_service
            and isinstance(service, ProcessServiceConfig)
            and is_llama_service(service)
        ]
        if "llama" in candidates:
            return "llama"
        return candidates[0] if candidates else None

    def comfyui_service_id(self) -> str | None:
        image = self.config.profiles.get("image")
        if image is not None:
            for sid in image.services:
                if isinstance(self.config.services.get(sid), ProcessServiceConfig):
                    return sid
        return "comfyui" if "comfyui" in self.config.services else None

    @property
    def comfyui_port_editable(self) -> bool:
        """A port can be chosen only when the ComfyUI service will be created."""
        return self.comfyui_service_id() is None

    def _initial_draft(self) -> SetupDraft:
        config = self.config
        settings = config.settings
        llama = self.report.recommended(LLAMA_CPP) or next(iter(self.report.llama_cpp), None)
        comfyui = self.report.recommended(COMFYUI) or next(iter(self.report.comfyui), None)

        if settings.setup_state == SetupState.INCOMPLETE and not settings.environments_explicit:
            agent = config.profiles.get("agent")
            environments = {
                "coding": "coding" in config.profiles,
                "chat": "chat" in config.profiles,
                "agent": bool(agent and agent.harness and agent.harness in config.harnesses),
                "image": bool(comfyui and comfyui.usable),
            }
        else:
            environments = {task: settings.environment_enabled(task) for task in TASK_IDS}

        comfy_port = DEFAULT_COMFYUI_PORT
        comfy_sid = self.comfyui_service_id()
        if comfy_sid is not None:
            comfy_port = _service_port(config.services[comfy_sid]) or DEFAULT_COMFYUI_PORT

        agent_profile = config.profiles.get("agent")
        harnesses = dict(config.harnesses) or discover_agent_harnesses(config.services, {})
        workshop = settings.prompt_workshop
        return SetupDraft(
            environments=environments,
            llama=llama,
            comfyui=comfyui,
            comfyui_port=comfy_port,
            model_roots=list(settings.llm_model_roots),
            task_models={
                task: config.profiles[task].default_model if task in config.profiles else None
                for task in LLM_ENVIRONMENTS
            },
            agent_harness=agent_profile.harness if agent_profile else None,
            harnesses=harnesses,
            agent_base_services=tuple(
                sid
                for sid in (agent_profile.services if agent_profile else ())
                if sid in config.services and getattr(config.services[sid], "uses_model", False)
            ),
            prompt=PromptChoices(
                enabled=workshop.enabled,
                model=workshop.preferred_model,
                auto_lightest=workshop.model_strategy == "smallest",
                processing_mode=workshop.processing_mode,
                keep_loaded=workshop.keep_loaded,
                gpu_layers=workshop.gpu_layers,
                context_length=workshop.context_length,
                port=workshop.port,
                startup_timeout=workshop.startup_timeout_seconds,
                cache_type_k=workshop.cache_type_k,
                cache_type_v=workshop.cache_type_v,
            ),
        )

    # --- Editing ---------------------------------------------------------------

    def rediscover(self) -> DiscoveryReport:
        self.report = discover_installations(self.config.services, home=self.home, path_env=self.path_env)
        return self.report

    def choose_llama_path(self, path: Path | str) -> Installation:
        self.draft.llama = validate_llama_executable(path, home=self.home)
        return self.draft.llama

    def choose_comfyui_path(self, folder: Path | str, python: Path | str | None = None) -> Installation:
        self.draft.comfyui = validate_comfyui(folder, python, home=self.home)
        return self.draft.comfyui

    def scan_models(self, roots: list[Path] | tuple[Path, ...]) -> dict[str, ModelConfig]:
        """Look for GGUF models in roots. Reads only; nothing is saved until apply()."""
        self.draft.model_roots = [Path(root).expanduser() for root in roots]
        self.models = load_models(self.config_dir / "models.toml", tuple(self.draft.model_roots))
        return self.models

    def refresh_models(self) -> dict[str, ModelConfig]:
        """Pick up launch settings the tuning window has just saved."""
        return self.scan_models(self.draft.model_roots)

    def selected_models(self) -> list[str]:
        """Models chosen for enabled LLM environments, without duplicates."""
        chosen: list[str] = []
        for task in LLM_ENVIRONMENTS:
            model_id = self.draft.task_models.get(task)
            if self.draft.environments.get(task) and model_id and model_id not in chosen:
                chosen.append(model_id)
        return chosen

    def path_changes(self) -> list[PathChange]:
        """Configured service paths that the chosen installations would change."""
        changes: list[PathChange] = []
        llama_sid = self.llama_service_id()
        if self.draft.needs_llama and llama_sid and self.draft.llama and self.draft.llama.usable:
            service = self.config.services[llama_sid]
            if not _same_path(service.executable, self.draft.llama.executable):
                changes.append(PathChange(llama_sid, "executable", service.executable, self.draft.llama.executable))
            if not _same_path(service.cwd, self.draft.llama.cwd):
                changes.append(PathChange(llama_sid, "cwd", service.cwd, self.draft.llama.cwd))
        comfy_sid = self.comfyui_service_id()
        if self.draft.needs_comfyui and comfy_sid and self.draft.comfyui and self.draft.comfyui.usable:
            service = self.config.services[comfy_sid]
            if not _same_path(service.executable, self.draft.comfyui.executable):
                changes.append(PathChange(comfy_sid, "executable", service.executable, self.draft.comfyui.executable))
            if not _same_path(service.cwd, self.draft.comfyui.cwd):
                changes.append(PathChange(comfy_sid, "cwd", service.cwd, self.draft.comfyui.cwd))
        return changes

    # --- Checks ----------------------------------------------------------------

    def issues(self, step: str | None = None) -> list[Issue]:
        found = (
            self._environment_issues()
            + self._software_issues()
            + self._model_issues()
            + self._prompt_issues()
        )
        return [issue for issue in found if step is None or issue.step == step]

    def _environment_issues(self) -> list[Issue]:
        if not any(self.draft.environments.values()):
            return [Issue(STEP_ENVIRONMENTS, "Choose at least one environment to set up.")]
        return []

    def _llm_enabled(self) -> tuple[str, ...]:
        return tuple(task for task in LLM_ENVIRONMENTS if self.draft.environments.get(task))

    def _software_issues(self) -> list[Issue]:
        issues: list[Issue] = []
        draft = self.draft
        llm = self._llm_enabled()
        if llm:
            if draft.llama is None:
                issues.append(Issue(
                    STEP_SOFTWARE,
                    "llama.cpp (llama-server) was not found. Choose the llama-server program, "
                    "or turn off Coding, Chat and Agent.",
                    llm,
                ))
            elif not draft.llama.usable:
                issues.append(Issue(STEP_SOFTWARE, "llama.cpp: " + "; ".join(draft.llama.problems), llm))
        if draft.needs_comfyui:
            if draft.comfyui is None:
                issues.append(Issue(
                    STEP_SOFTWARE,
                    "ComfyUI was not found. Choose your ComfyUI folder, or turn off Image.",
                    ("image",),
                ))
            elif not draft.comfyui.usable:
                issues.append(Issue(STEP_SOFTWARE, "ComfyUI: " + "; ".join(draft.comfyui.problems), ("image",)))
            if self.comfyui_port_editable:
                port = draft.comfyui_port
                taken = {
                    _service_port(service)
                    for sid, service in self.config.services.items()
                    if isinstance(service, ProcessServiceConfig)
                }
                if draft.prompt.enabled:
                    taken.add(draft.prompt.port)
                if not 1 <= port <= 65535:
                    issues.append(Issue(STEP_SOFTWARE, f"ComfyUI port {port} must be from 1 to 65535.", ("image",)))
                elif port in taken:
                    issues.append(Issue(
                        STEP_SOFTWARE,
                        f"ComfyUI port {port} is already used by another configured service; choose another port.",
                        ("image",),
                    ))
        return issues

    def _model_issues(self) -> list[Issue]:
        issues: list[Issue] = []
        draft = self.draft
        llm = self._llm_enabled()
        if llm and not draft.model_roots:
            issues.append(Issue(STEP_MODELS, "Add at least one folder that contains GGUF models.", llm))
        for task in llm:
            name = ENVIRONMENT_NAMES[task]
            model_id = draft.task_models.get(task)
            if model_id is None:
                issues.append(Issue(STEP_MODELS, f"Choose a model for {name}.", (task,)))
                continue
            model = self.models.get(model_id)
            if model is None:
                issues.append(Issue(STEP_MODELS, f"{name} model '{model_id}' was not found. Rescan or choose another.", (task,)))
            elif not model.complete:
                issues.append(Issue(STEP_MODELS, f"{name} model {model.display_name} is incomplete or missing.", (task,)))
            elif not model.tuning_reviewed:
                issues.append(Issue(STEP_LAUNCH, f"{model.display_name} needs launch settings ({name}).", (task,)))
        if draft.environments.get("agent"):
            harness_id = draft.agent_harness
            if harness_id is None:
                issues.append(Issue(STEP_MODELS, "Choose an Agent harness, scan for one, add a custom harness, or turn off Agent.", ("agent",)))
            elif harness_id not in draft.harnesses:
                issues.append(Issue(STEP_MODELS, f"Agent harness '{harness_id}' is not configured.", ("agent",)))
            else:
                missing = [sid for sid in draft.harnesses[harness_id].services if sid not in self.config.services]
                if missing:
                    issues.append(Issue(STEP_MODELS, "Agent harness has missing service(s): " + ", ".join(missing), ("agent",)))
        return issues

    def prompt_available(self) -> bool:
        return bool(self._llm_enabled())

    def _prompt_issues(self) -> list[Issue]:
        prompt = self.draft.prompt
        if not (prompt.enabled and self.prompt_available()):
            return []
        issues: list[Issue] = []
        service = self.config.settings.prompt_workshop.service
        if service not in self.config.services:
            issues.append(Issue(
                STEP_PROMPT,
                f"Prompt Helper needs a '{service}' service, which this configuration does not have. "
                "Leave Prompt Helper off for now.",
            ))
        if not prompt.auto_lightest:
            if prompt.model is None:
                issues.append(Issue(STEP_PROMPT, "Choose a Prompt Helper model, or use the lightest suitable model."))
            elif prompt.model not in self.models:
                issues.append(Issue(STEP_PROMPT, f"Prompt Helper model '{prompt.model}' was not found."))
        return issues

    def environment_problems(self) -> dict[str, tuple[str, ...]]:
        problems: dict[str, list[str]] = {}
        for issue in self.issues():
            for task in issue.environments:
                if self.draft.environments.get(task):
                    problems.setdefault(task, []).append(issue.message)
        return {task: tuple(messages) for task, messages in problems.items()}

    def final_environments(self) -> dict[str, bool]:
        problems = self.environment_problems()
        return {
            task: bool(self.draft.environments.get(task)) and task not in problems
            for task in TASK_IDS
        }

    # --- Review ----------------------------------------------------------------

    def review(self) -> Review:
        draft = self.draft
        problems = self.environment_problems()
        final = self.final_environments()
        sections: list[tuple[str, tuple[str, ...]]] = []

        host = detect_setup_environment()
        machine = [f"Operating system: {host.os_name}", f"Python: {host.python_version}"]
        gpu = self._gpu_line()
        if gpu:
            machine.append(gpu)
        machine.append(f"Docker: {'available' if host.docker_available else 'not found'}")
        sections.append(("This computer", tuple(machine)))

        env_lines = []
        for task in TASK_IDS:
            name = ENVIRONMENT_NAMES[task]
            if not draft.environments.get(task):
                env_lines.append(f"{name}: off")
            elif final[task]:
                env_lines.append(f"{name}: on")
            else:
                env_lines.append(f"{name}: will be saved switched off - {problems[task][0]}")
        sections.append(("Environments", tuple(env_lines)))

        software: list[str] = []
        changes = {(change.service_id, change.field): change for change in self.path_changes()}
        if draft.needs_llama:
            if draft.llama is not None:
                software.append(f"llama.cpp: {portable_path(draft.llama.path)}")
                llama_sid = self.llama_service_id()
                change = changes.get((llama_sid, "executable")) if llama_sid else None
                if change:
                    software.append(f"  (changes from {portable_path(change.old)})")
                llama_service = self.config.services.get(llama_sid) if llama_sid else None
                if isinstance(llama_service, ProcessServiceConfig) and llama_service.health:
                    software.append(f"  Port {_service_port(llama_service)}, ready when {llama_service.health.url} answers")
            else:
                software.append("llama.cpp: not found")
        if draft.needs_comfyui:
            if draft.comfyui is not None:
                software.append(f"ComfyUI: {portable_path(draft.comfyui.path)}")
                if draft.comfyui.executable is not None:
                    software.append(f"  Python: {portable_path(draft.comfyui.executable)}")
                software.append(
                    f"  Port {draft.comfyui_port}, ready when "
                    f"http://127.0.0.1:{draft.comfyui_port}/system_stats answers, then opens in your browser"
                )
            else:
                software.append("ComfyUI: not found")
        if software:
            sections.append(("Software", tuple(software)))

        if draft.needs_llama:
            model_lines = []
            for task in self._llm_enabled():
                model_id = draft.task_models.get(task)
                model = self.models.get(model_id) if model_id else None
                model_lines.append(f"{ENVIRONMENT_NAMES[task]}: {model.display_name if model else model_id or 'not selected'}")
            if draft.environments.get("agent"):
                harness = draft.harnesses.get(draft.agent_harness) if draft.agent_harness else None
                model_lines.append(f"Agent harness: {harness.display_name if harness else 'not selected'}")
            model_lines.append("Model folders: " + (", ".join(str(root) for root in draft.model_roots) or "none"))
            sections.append(("Models", tuple(model_lines)))
            prompt = draft.prompt
            if prompt.enabled:
                helper = "lightest suitable model" if prompt.auto_lightest else (prompt.model or "not selected")
                sections.append(("Prompt Helper", (f"On - {helper}, {prompt.processing_mode.upper()} processing",)))
            else:
                sections.append(("Prompt Helper", ("Off",)))

        issues = tuple(self.issues())
        if issues:
            sections.append(("Unresolved issues", tuple(issue.message for issue in issues)))
        return Review(sections=tuple(sections), issues=issues, switched_off=problems)

    def _gpu_line(self) -> str | None:
        if self.gpu_status is None:
            return None
        try:
            status = self.gpu_status()
        except Exception:  # noqa: BLE001 - monitoring is informational only
            return None
        if not getattr(status, "available", False):
            return f"GPU: not detected ({getattr(status, 'detail', '')})".rstrip(" ()")
        total = getattr(status, "memory_total_mib", None)
        memory = f", {total} MiB VRAM" if total else ""
        return f"GPU: {getattr(status, 'name', 'unknown')}{memory}"

    # --- Saving ----------------------------------------------------------------

    def keep_current(self) -> dict[str, bool]:
        """Upgrade welcome: accept the existing setup unchanged."""
        return keep_current_setup(self.config_dir)

    def apply(self) -> ApplyResult:
        """Validate and save the setup. Selected environments with problems are saved off."""
        blocking = [issue for issue in self.issues() if not issue.environments]
        if blocking:
            raise SetupError("\n".join(issue.message for issue in blocking))
        final = self.final_environments()
        problems = self.environment_problems()
        if not any(final.values()):
            reasons = "\n".join(message for messages in problems.values() for message in messages)
            raise SetupError("None of the selected environments can be set up yet:\n" + reasons)
        for name, digest in self._opened.items():
            if _digest(self.config_dir / name) != digest:
                raise SetupError(
                    f"{name} was changed outside the Setup Wizard while it was open. "
                    "Close the wizard and open it again so nothing is overwritten."
                )

        staging = Path(tempfile.mkdtemp(prefix=".acc-setup-", dir=self.config_dir.parent))
        try:
            for path in self.config_dir.glob("*.toml"):
                shutil.copy2(path, staging / path.name)
            self._apply_to(staging, final)
            try:
                load_app_config(staging)
            except ConfigError as exc:
                raise SetupError(f"The new configuration did not pass validation, so nothing was saved: {exc}") from exc
            changed = self._commit(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return ApplyResult(environments=final, switched_off=problems, files_changed=changed)

    @staticmethod
    def _portable(path: Path) -> str:
        # Collapse against the real home folder: that is what ~ expands to when
        # the configuration is loaded. self.home only steers discovery.
        return portable_path(path)

    def _apply_to(self, staging: Path, final: Mapping[str, bool]) -> None:
        draft = self.draft
        config = self.config
        llm_on = [task for task in LLM_ENVIRONMENTS if final[task]]

        llama_sid = self.llama_service_id()
        if llm_on:
            llama = draft.llama
            assert llama is not None and llama.usable and llama.executable is not None
            if llama_sid is None:
                plan = apply_preset(staging, load_preset("llama-cpp"), llama.preset_values())
                llama_sid = plan.service_id
            else:
                old = config.services[llama_sid].executable
                if not _same_path(old, llama.executable) or not _same_path(config.services[llama_sid].cwd, llama.cwd):
                    # Prompt Helper normally runs the same llama-server; keep it in step.
                    for sid, service in config.services.items():
                        if isinstance(service, ProcessServiceConfig) and _same_path(service.executable, old):
                            update_service_paths(
                                staging, sid,
                                executable=self._portable(llama.executable),
                                cwd=self._portable(llama.cwd),
                            )

        if final["image"]:
            comfy = draft.comfyui
            assert comfy is not None and comfy.usable and comfy.executable is not None
            result = configure_image_environment(
                staging,
                comfy,
                port=draft.comfyui_port if self.comfyui_port_editable else None,
                service_id=self.comfyui_service_id(),
            )
            if result.service_differences:
                update_service_paths(
                    staging, result.service_id,
                    executable=self._portable(comfy.executable),
                    cwd=self._portable(comfy.cwd),
                )

        for task in llm_on:
            if task not in config.profiles:
                table: dict[str, object] = {"display_name": ENVIRONMENT_NAMES[task], "services": [llama_sid]}
                if task != "agent":
                    table["open_service"] = llama_sid
                append_toml_table(
                    staging / "profiles.toml", ("profiles", task), table,
                    comment=f"Added by AI Control Centre for the {ENVIRONMENT_NAMES[task]} environment.",
                )

        existing = {
            task: config.profiles[task].default_model if task in config.profiles else None
            for task in LLM_ENVIRONMENTS
        }
        task_models = {task: draft.task_models.get(task) if final[task] else existing[task] for task in LLM_ENVIRONMENTS}
        agent_profile = config.profiles.get("agent")
        if final["agent"]:
            agent_harness = draft.agent_harness
            agent_base = draft.agent_base_services or ((llama_sid,) if llama_sid else ())
        else:
            agent_harness = agent_profile.harness if agent_profile else None
            agent_base = draft.agent_base_services
        prompt = draft.prompt
        save_preferences(
            staging,
            model_roots=tuple(draft.model_roots) if llm_on else tuple(config.settings.llm_model_roots),
            task_models=task_models,
            agent_harness=agent_harness,
            agent_base_services=agent_base,
            harnesses=draft.harnesses,
            prompt_model=prompt.model if prompt.enabled else None,
            prompt_enabled=prompt.enabled and bool(llm_on),
            prompt_auto_lightest=prompt.auto_lightest,
            prompt_processing_mode=prompt.processing_mode,
            prompt_keep_loaded=prompt.keep_loaded,
            prompt_gpu_layers=prompt.gpu_layers,
            prompt_context_length=prompt.context_length,
            prompt_port=prompt.port,
            prompt_startup_timeout=prompt.startup_timeout,
            prompt_cache_type_k=prompt.cache_type_k,
            prompt_cache_type_v=prompt.cache_type_v,
            setup_completed=True,
            environments=final,
        )

    def _commit(self, staging: Path) -> tuple[str, ...]:
        changed: list[str] = []
        for name in _COMMIT_ORDER:
            new = staging / name
            if not new.exists():
                continue
            target = self.config_dir / name
            data = new.read_bytes()
            if target.exists() and target.read_bytes() == data:
                continue
            fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=self.config_dir)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
            finally:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
            changed.append(name)
        return tuple(changed)
