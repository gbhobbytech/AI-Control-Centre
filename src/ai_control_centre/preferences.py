from __future__ import annotations

import os
import platform
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import CURRENT_SETUP_SCHEMA, load_models
from .domain import HarnessConfig, ModelConfig


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


def _emit_table(lines: list[str], prefix: tuple[str, ...], table: Mapping[str, Any]) -> None:
    scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if prefix:
        lines.append(f"[{'.'.join(_quote(part) for part in prefix)}]")
    for key, value in scalar_items:
        if value is None:
            continue
        lines.append(f"{_quote(key)} = {_scalar(value)}")
    if prefix and (scalar_items or child_items):
        lines.append("")
    for key, child in child_items:
        _emit_table(lines, (*prefix, key), child)


def write_toml_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    _emit_table(lines, (), data)
    text = "\n".join(lines).rstrip() + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


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
    for profile_id, model_id in task_models.items():
        raw = profile_table.setdefault(profile_id, {})
        if model_id:
            raw["default_model"] = model_id
        else:
            raw.pop("default_model", None)
    agent = profile_table.setdefault("agent", {})
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
