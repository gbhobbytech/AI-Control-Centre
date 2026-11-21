from __future__ import annotations

import os
import platform
import re
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import CURRENT_SETUP_SCHEMA, ConfigError, load_app_config, load_models, load_profiles, load_services
from .domain import HarnessConfig, ModelConfig, SetupState, TASK_IDS


@dataclass(frozen=True)
class SetupDetection:
    os_name: str
    python_version: str
    docker_available: bool
    nvidia_smi_available: bool


def detect_setup_environment() -> SetupDetection:
    os_name = platform.system()
    try:
        release = platform.freedesktop_os_release()
        os_name = release.get("PRETTY_NAME", os_name)
    except OSError:
        pass
    return SetupDetection(
        os_name=os_name,
        python_version=platform.python_version(),
        docker_available=shutil.which("docker") is not None,
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
    )


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        value = tomllib.load(fh)
    return dict(value)


def _quote(value: str) -> str:
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    value = value.replace("\n", "\\n")
    return f'"{value}"'


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(part: str, bare: bool) -> str:
    return part if bare and _BARE_KEY.match(part) else _quote(part)


def _emit_table(
    lines: list[str],
    prefix: tuple[str, ...],
    table: Mapping[str, Any],
    bare_keys: bool = False,
) -> None:
    scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if prefix:
        lines.append(f"[{'.'.join(_key(part, bare_keys) for part in prefix)}]")
    for key, value in scalar_items:
        if value is None:
            continue
        lines.append(f"{_key(key, bare_keys)} = {_scalar(value)}")
    if prefix and (scalar_items or child_items):
        lines.append("")
    for key, child in child_items:
        _emit_table(lines, (*prefix, key), child, bare_keys)


def _write_text_atomic(
    path: Path, text: str, validate: Callable[[Path], object] | None = None
) -> None:
    """Write text via a temporary file; validate it before it replaces path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if validate is not None:
            validate(Path(name))
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def write_toml_atomic(path: Path, data: Mapping[str, Any]) -> None:
    lines: list[str] = []
    _emit_table(lines, (), data)
    text = "\n".join(lines).rstrip() + "\n"
    _write_text_atomic(path, text)


def toml_table_text(prefix: tuple[str, ...], table: Mapping[str, Any]) -> str:
    """Hand-editable TOML text for one table (bare keys where valid)."""
    lines: list[str] = []
    _emit_table(lines, prefix, table, bare_keys=True)
    return "\n".join(lines).rstrip() + "\n"


def append_toml_table(
    path: Path,
    prefix: tuple[str, ...],
    table: Mapping[str, Any],
    *,
    comment: str | None = None,
    validate: Callable[[Path], object] | None = None,
) -> None:
    """Add a new table to the end of a TOML file, keeping existing text as-is.

    Unlike write_toml_atomic this preserves the user's comments and layout.
    The table must not already exist. validate, if given, receives the path of
    the complete candidate file and may raise to abort the change.
    """
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    existing: Any = tomllib.loads(original) if original else {}
    for part in prefix:
        if not isinstance(existing, dict):
            break
        existing = existing.get(part)
    else:
        if existing is not None:
            raise ValueError(f"{path.name} already contains [{'.'.join(prefix)}]")

    block = toml_table_text(prefix, table)
    if comment:
        block = "".join(f"# {line}".rstrip() + "\n" for line in comment.splitlines()) + block
    if not original:
        separator = ""
    elif original.endswith("\n\n"):
        separator = ""
    elif original.endswith("\n"):
        separator = "\n"
    else:
        separator = "\n\n"
    text = original + separator + block
    tomllib.loads(text)
    _write_text_atomic(path, text, validate)


def rescan_models(config_dir: Path, roots: tuple[Path, ...]) -> dict[str, ModelConfig]:
    return load_models(config_dir / "models.toml", roots)


def save_preferences(
    config_dir: Path,
    *,
    model_roots: tuple[Path, ...],
    task_models: Mapping[str, str | None],
    agent_harness: str | None,
    agent_base_services: tuple[str, ...],
    harnesses: Mapping[str, HarnessConfig],
    prompt_model: str | None,
    prompt_enabled: bool,
    prompt_auto_lightest: bool,
    prompt_processing_mode: str,
    prompt_keep_loaded: bool,
    prompt_gpu_layers: int | None,
    prompt_context_length: int,
    prompt_port: int,
    prompt_startup_timeout: float,
    prompt_cache_type_k: str,
    prompt_cache_type_v: str,
    setup_completed: bool = True,
    environments: Mapping[str, bool] | None = None,
) -> None:
    settings_path = config_dir / "settings.toml"
    profiles_path = config_dir / "profiles.toml"
    settings = _load_toml(settings_path)
    profiles = _load_toml(profiles_path)

    settings.setdefault("paths", {})
    settings.setdefault("monitoring", {})
    settings["model_roots"] = {"llm": [str(path.expanduser()) for path in model_roots]}
    settings["setup"] = {
        "completed": bool(setup_completed),
        "schema_version": CURRENT_SETUP_SCHEMA,
    }
    if environments is not None:
        _store_environments(settings, environments)

    prompt = settings.setdefault("prompt_workshop", {})
    prompt["enabled"] = bool(prompt_enabled)
    prompt["model_strategy"] = "smallest" if prompt_auto_lightest else "manual"
    if prompt_model:
        prompt["preferred_model"] = prompt_model
    else:
        prompt.pop("preferred_model", None)
    prompt["processing_mode"] = prompt_processing_mode
    prompt["keep_loaded"] = bool(prompt_keep_loaded)
    if prompt_gpu_layers is None:
        prompt.pop("gpu_layers", None)
    else:
        prompt["gpu_layers"] = int(prompt_gpu_layers)
    prompt["context_length"] = int(prompt_context_length)
    prompt["port"] = int(prompt_port)
    prompt["startup_timeout_seconds"] = float(prompt_startup_timeout)
    prompt["cache_type_k"] = prompt_cache_type_k
    prompt["cache_type_v"] = prompt_cache_type_v
    prompt["endpoint"] = f"http://127.0.0.1:{int(prompt_port)}/v1/chat/completions"

    harness_table = {}
    profiles["harnesses"] = harness_table
    for harness_id, harness in harnesses.items():
        raw_harness = {}
        harness_table[harness_id] = raw_harness
        raw_harness["display_name"] = harness.display_name
        raw_harness["services"] = list(harness.services)
        if harness.open_service:
            raw_harness["open_service"] = harness.open_service

    profile_table = profiles.setdefault("profiles", {})
    # Only profiles that already exist are updated. Creating an empty table
    # here would leave a profile with no services, which does not load.
    for profile_id, model_id in task_models.items():
        raw = profile_table.get(profile_id)
        if not isinstance(raw, dict):
            continue
        if model_id:
            raw["default_model"] = model_id
        else:
            raw.pop("default_model", None)
    agent = profile_table.get("agent")
    if isinstance(agent, dict):
        # V1 harnesses own the interface/runtime to open for Agent. Remove any
        # profile-level V0.x value so future loads cannot resurrect the legacy
        # relationship after migration.
        agent.pop("open_service", None)
        if agent_base_services:
            agent["services"] = list(agent_base_services)
        elif "services" in agent:
            agent["services"] = []
        if agent_harness:
            agent["harness"] = agent_harness
        else:
            agent.pop("harness", None)

    write_toml_atomic(settings_path, settings)
    write_toml_atomic(profiles_path, profiles)


def _store_environments(settings: dict[str, Any], environments: Mapping[str, bool]) -> None:
    """Write every task's selection; keep keys this version does not know."""
    unknown = sorted(set(environments) - set(TASK_IDS))
    if unknown:
        raise ValueError(f"Unknown environment(s): {', '.join(unknown)}")
    for task_id, value in environments.items():
        if not isinstance(value, bool):
            raise ValueError(f"Environment '{task_id}' must be True or False")
    table = settings.get("environments")
    table = dict(table) if isinstance(table, dict) else {}
    for task_id in TASK_IDS:
        table[task_id] = bool(environments.get(task_id, False))
    settings["environments"] = table


def keep_current_setup(config_dir: Path) -> dict[str, bool]:
    """Accept an earlier completed setup as-is under the current setup schema.

    Records the environments the user already has (as inferred from their
    profiles) and the current schema version in settings.toml. Nothing else
    in any configuration file is changed. Calling it again changes nothing.
    """
    config = load_app_config(config_dir)
    current = config.settings
    if current.setup_state == SetupState.INCOMPLETE:
        raise ValueError("Setup has not been completed yet; run the Setup Wizard instead.")
    environments = dict(current.environments)
    if current.setup_state == SetupState.COMPLETE and current.environments_explicit:
        return environments

    path = config_dir / "settings.toml"
    settings = _load_toml(path)
    if not current.environments_explicit:
        _store_environments(settings, environments)
    setup = settings.get("setup")
    setup = dict(setup) if isinstance(setup, dict) else {}
    setup["completed"] = True
    setup["schema_version"] = CURRENT_SETUP_SCHEMA
    settings["setup"] = setup
    write_toml_atomic(path, settings)
    return environments


def save_model_tuning(config_dir: Path, model: ModelConfig, values: Mapping, source: str = "manual") -> None:
    from .tuning import CURRENT_TUNING_SCHEMA, validate_values
    clean = validate_values(values)
    path = config_dir / "models.toml"
    data = _load_toml(path)
    table = data.setdefault("models", {}).setdefault(model.id, {})
    table.setdefault("display_name", model.display_name)
    table.setdefault("path", str(model.path))
    table.update(clean)
    table["tuning_reviewed"] = True
    table["tuning_source"] = source
    table["tuning_schema_version"] = CURRENT_TUNING_SCHEMA
    write_toml_atomic(path, data)


def save_appearance(config_dir: Path, values: Mapping) -> None:
    from .theme import validate_appearance
    clean = validate_appearance(dict(values))
    path = config_dir / "settings.toml"
    data = _load_toml(path)
    data.setdefault("appearance", {}).update(clean)
    write_toml_atomic(path, data)


def save_window_size(config_dir: Path, width: int, height: int) -> None:
    """Persist the main-window size without changing other user settings."""
    path = config_dir / "settings.toml"
    data = _load_toml(path)
    ui = data.setdefault("ui", {})
    ui["window_width"] = max(1, int(width))
    ui["window_height"] = max(1, int(height))
    write_toml_atomic(path, data)


# --- Targeted service edits (comment-preserving) -------------------------------

_TABLE_HEADER = re.compile(r"^\s*\[(?!\[)\s*(?P<inner>[^\]]+?)\s*\]\s*(?:#.*)?$")
_ANY_HEADER = re.compile(r"^\s*\[")


def _header_parts(inner: str) -> tuple[str, ...]:
    return tuple(part.strip().strip('"').strip("'") for part in inner.split("."))


def update_service_paths(
    config_dir: Path,
    service_id: str,
    *,
    executable: Path | str | None = None,
    cwd: Path | str | None = None,
) -> None:
    """Change where one process service is installed, after the user confirms it.

    Only the executable and cwd lines of [services.<id>] are rewritten; the
    rest of services.toml, including comments, stays exactly as it was. The
    result is validated before it replaces the file.
    """
    updates = {key: str(value) for key, value in (("executable", executable), ("cwd", cwd)) if value is not None}
    if not updates:
        return
    path = config_dir / "services.toml"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    header_index = None
    for index, line in enumerate(lines):
        match = _TABLE_HEADER.match(line)
        if match and _header_parts(match.group("inner")) == ("services", service_id):
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"services.toml has no [services.{service_id}] table to update")
    end = next(
        (index for index in range(header_index + 1, len(lines)) if _ANY_HEADER.match(lines[index])),
        len(lines),
    )

    insert_at = header_index + 1
    for key, value in updates.items():
        pattern = re.compile(rf"^(?P<indent>\s*)(?P<q>[\"']?){key}(?P=q)\s*=(?P<rest>.*)$")
        new_line = f"{key} = {_quote(value)}\n"
        for index in range(header_index + 1, end):
            match = pattern.match(lines[index].rstrip("\n"))
            if match:
                rest = match.group("rest").strip()
                if rest.startswith(('"""', "'''")):
                    raise ValueError(f"[services.{service_id}] {key} uses a multi-line string; edit it by hand")
                lines[index] = match.group("indent") + new_line
                break
        else:
            lines.insert(insert_at, new_line)
            insert_at += 1
            end += 1

    def validate(candidate: Path) -> None:
        services = load_services(candidate)
        if service_id not in services or not hasattr(services[service_id], "executable"):
            raise ConfigError(f"[services.{service_id}] is not a process service")

    try:
        _write_text_atomic(path, "".join(lines), validate)
    except (ConfigError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Updating [services.{service_id}] would make services.toml invalid: {exc}") from exc


# --- Image environment ---------------------------------------------------------


@dataclass(frozen=True)
class ImageSetupResult:
    service_id: str
    service_added: bool
    # Keys where an existing service differs from the selected installation.
    # The service is not changed; update_service_paths() applies a confirmed choice.
    service_differences: tuple[str, ...]
    profile_added: bool
    environment_enabled: bool


def configure_image_environment(
    config_dir: Path,
    installation: Any,
    *,
    port: int | None = None,
    service_id: str | None = None,
    home: Path | None = None,
) -> ImageSetupResult:
    """Make the Image environment run the given ComfyUI installation.

    Adds the ComfyUI service from its preset if it does not exist, adds an
    Image profile if there is none, and turns the Image environment on.
    Existing services and profiles are never modified.
    """
    from .discovery import COMFYUI
    from .presets import apply_preset, load_preset

    if getattr(installation, "kind", None) != COMFYUI:
        raise ValueError("configure_image_environment needs a ComfyUI installation")
    values = installation.preset_values(home)
    if port is not None:
        values["port"] = str(port)
    plan = apply_preset(config_dir, load_preset("comfyui"), values, service_id)

    config = load_app_config(config_dir)
    profile_added = False
    if "image" not in config.profiles:
        def validate(candidate: Path) -> None:
            load_profiles(candidate, config.services, config.models, config.harnesses)

        append_toml_table(
            config_dir / "profiles.toml",
            ("profiles", "image"),
            {"display_name": "Image", "services": [plan.service_id], "open_service": plan.service_id},
            comment="Added by AI Control Centre for the Image environment.",
            validate=validate,
        )
        profile_added = True

    settings = config.settings
    environment_enabled = False
    if not (settings.environments_explicit and settings.environment_enabled("image")):
        environments = dict(settings.environments)
        environments["image"] = True
        raw = _load_toml(config_dir / "settings.toml")
        _store_environments(raw, environments)
        write_toml_atomic(config_dir / "settings.toml", raw)
        environment_enabled = True

    differences = tuple(
        key for key in plan.customised_keys if key in {"executable", "cwd"}
    )
    return ImageSetupResult(
        service_id=plan.service_id,
        service_added=plan.changes_configuration,
        service_differences=differences,
        profile_added=profile_added,
        environment_enabled=environment_enabled,
    )
