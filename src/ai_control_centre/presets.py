"""Reusable service presets.

A preset is a service definition from services.toml with {preset.NAME}
placeholders for machine-specific values such as installation paths and
ports. Presets ship with the application under ``presets/``.

Merge rule: a preset never changes a service the user already has. Applying
a preset either adds a service that does not exist yet (appended to
services.toml so existing comments and layout survive), or leaves the
existing service untouched and reports how it differs from the preset.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, load_services
from .preferences import append_toml_table

_PLACEHOLDER = re.compile(r"\{preset\.([A-Za-z_][A-Za-z0-9_]*)\}")
_VARIABLE_KINDS = {"text", "path", "port"}


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class PresetVariable:
    name: str
    description: str
    kind: str = "text"
    default: str | None = None

    @property
    def required(self) -> bool:
        return self.default is None


@dataclass(frozen=True)
class ServicePreset:
    id: str
    display_name: str
    version: int
    service_id: str
    description: str = ""
    variables: Mapping[str, PresetVariable] = field(default_factory=dict)
    service: Mapping[str, Any] = field(default_factory=dict)

    def resolve_values(self, values: Mapping[str, str]) -> dict[str, str]:
        unknown = sorted(set(values) - set(self.variables))
        if unknown:
            raise PresetError(f"Preset '{self.id}': unknown value(s): {', '.join(unknown)}")
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for name, variable in self.variables.items():
            value = values.get(name, variable.default)
            if value is None:
                missing.append(f"{name} ({variable.description})")
                continue
            if not isinstance(value, str) or not value.strip() or "\n" in value:
                raise PresetError(
                    f"Preset '{self.id}': {name} must be a non-empty, single-line value"
                )
            if variable.kind == "port":
                if not value.isdigit() or not 1 <= int(value) <= 65535:
                    raise PresetError(
                        f"Preset '{self.id}': {name} must be a port number from 1 to 65535, not {value!r}"
                    )
            resolved[name] = value
        if missing:
            raise PresetError(f"Preset '{self.id}' needs: {'; '.join(missing)}")
        return resolved

    def render(self, values: Mapping[str, str]) -> dict[str, Any]:
        """The service table with every {preset.NAME} filled in."""
        resolved = self.resolve_values(values)
        rendered = _render(self.service, resolved)
        rendered["preset"] = self.id
        rendered["preset_version"] = self.version
        return rendered


def _render(value: Any, values: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda match: values[match.group(1)], value)
    if isinstance(value, list):
        return [_render(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, values) for key, item in value.items()}
    return value


def _placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_PLACEHOLDER.findall(value))
    if isinstance(value, list):
        return set().union(*(_placeholders(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(_placeholders(item) for item in value.values())) if value else set()
    return set()


def parse_preset(data: Mapping[str, Any], source: str) -> ServicePreset:
    meta = data.get("preset")
    service = data.get("service")
    if not isinstance(meta, dict):
        raise PresetError(f"{source}: missing [preset] table")
    if not isinstance(service, dict) or not service.get("type"):
        raise PresetError(f"{source}: missing [service] table with a type")
    for key in ("id", "display_name", "service_id"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            raise PresetError(f"{source}: [preset] '{key}' must be a non-empty string")
    version = meta.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PresetError(f"{source}: [preset] 'version' must be a positive integer")

    variables: dict[str, PresetVariable] = {}
    raw_variables = meta.get("variables", {})
    if not isinstance(raw_variables, dict):
        raise PresetError(f"{source}: [preset.variables] must be a table")
    for name, raw in raw_variables.items():
        if not isinstance(raw, dict):
            raise PresetError(f"{source}: variable '{name}' must be a table")
        kind = raw.get("kind", "text")
        if kind not in _VARIABLE_KINDS:
            raise PresetError(f"{source}: variable '{name}' has unknown kind {kind!r}")
        default = raw.get("default")
        if default is not None and not isinstance(default, str):
            raise PresetError(f"{source}: variable '{name}' default must be a string")
        variables[name] = PresetVariable(
            name=name,
            description=str(raw.get("description", name)),
            kind=kind,
            default=default,
        )

    undefined = sorted(_placeholders(service) - set(variables))
    if undefined:
        raise PresetError(f"{source}: undefined placeholder(s): {', '.join(undefined)}")

    return ServicePreset(
        id=meta["id"],
        display_name=meta["display_name"],
        version=version,
        service_id=meta["service_id"],
        description=str(meta.get("description", "")),
        variables=variables,
        service=service,
    )


def list_presets() -> dict[str, ServicePreset]:
    """Presets bundled with the application, keyed by preset id."""
    presets: dict[str, ServicePreset] = {}
    directory = resources.files("ai_control_centre").joinpath("presets")
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".toml"):
            continue
        try:
            data = tomllib.loads(entry.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise PresetError(f"{entry.name} is not valid TOML: {exc}") from exc
        preset = parse_preset(data, entry.name)
        presets[preset.id] = preset
    return presets


def load_preset(preset_id: str) -> ServicePreset:
    presets = list_presets()
    try:
        return presets[preset_id]
    except KeyError:
        raise PresetError(
            f"Unknown service preset '{preset_id}'. Available: {', '.join(sorted(presets))}"
        ) from None


def _flatten(table: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            items.update(_flatten(value, dotted + "."))
        else:
            items[dotted] = value
    return items


@dataclass(frozen=True)
class PresetMergePlan:
    preset_id: str
    service_id: str
    action: str  # "add": service is new; "keep": existing service is left unchanged
    service: Mapping[str, Any]
    missing_keys: tuple[str, ...] = ()
    customised_keys: tuple[str, ...] = ()

    @property
    def changes_configuration(self) -> bool:
        return self.action == "add"


_PRESET_MARKERS = {"preset", "preset_version"}


def plan_service_merge(
    existing_services: Mapping[str, Any],
    preset: ServicePreset,
    values: Mapping[str, str],
    service_id: str | None = None,
) -> PresetMergePlan:
    """Work out what applying a preset would do, without changing anything."""
    service_id = service_id or preset.service_id
    rendered = preset.render(values)
    current = existing_services.get(service_id)
    if current is None:
        return PresetMergePlan(preset.id, service_id, "add", rendered)
    if not isinstance(current, dict):
        raise PresetError(f"service '{service_id}' in services.toml must be a table")

    wanted = {k: v for k, v in _flatten(rendered).items() if k not in _PRESET_MARKERS}
    have = _flatten(current)
    missing = tuple(sorted(key for key in wanted if key not in have))
    customised = tuple(sorted(key for key in wanted if key in have and have[key] != wanted[key]))
    return PresetMergePlan(
        preset.id, service_id, "keep", rendered, missing_keys=missing, customised_keys=customised
    )


def apply_preset(
    config_dir: Path,
    preset: ServicePreset,
    values: Mapping[str, str],
    service_id: str | None = None,
) -> PresetMergePlan:
    """Add the preset's service to services.toml if it is not already there.

    An existing service with the same id is never modified; the returned plan
    lists how it differs so the user can decide what to change.
    """
    path = config_dir / "services.toml"
    data: dict[str, Any] = {}
    if path.exists():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    services = data.get("services", {})
    if not isinstance(services, dict):
        raise PresetError("services.toml: [services] must be a table")

    plan = plan_service_merge(services, preset, values, service_id)
    if not plan.changes_configuration:
        return plan

    def validate(candidate: Path) -> None:
        try:
            load_services(candidate)
        except ConfigError as exc:
            raise PresetError(f"Preset '{preset.id}' would make services.toml invalid: {exc}") from exc

    append_toml_table(
        path,
        ("services", plan.service_id),
        plan.service,
        comment=f"Added by AI Control Centre from the {preset.display_name} preset (version {preset.version}).",
        validate=validate,
    )
    return plan
