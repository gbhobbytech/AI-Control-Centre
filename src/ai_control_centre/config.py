from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .domain import (
    DockerServiceConfig,
    HealthCheckConfig,
    ModelConfig,
    ProcessServiceConfig,
    ProfileConfig,
    ServiceConfig,
)
from .models import discover_models


class ConfigError(ValueError):
    pass


def _expand(text: str) -> str:
    return os.path.expanduser(os.path.expandvars(text))


def _require_table(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Expected [{key}] table")
    return value


def _require_str(parent: dict[str, Any], key: str, context: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}: '{key}' must be a non-empty string")
    return value


def _optional_str(parent: dict[str, Any], key: str, context: str) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}: '{key}' must be a non-empty string when supplied")
    return value


def _string_list(parent: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = parent.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{context}: '{key}' must be an array of non-empty strings")
    return tuple(value)


def _optional_int(parent: dict[str, Any], key: str, context: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context}: '{key}' must be an integer when supplied")
    return value


@dataclass(frozen=True)
class Settings:
    log_dir: Path
    runtime_dir: Path
    llm_model_roots: tuple[Path, ...] = ()
    gpu_backend: str = "auto"
    monitoring_interval: float = 2.0


@dataclass(frozen=True)
class AppConfig:
    settings: Settings
    services: dict[str, ServiceConfig]
    profiles: dict[str, ProfileConfig]
    models: dict[str, ModelConfig] = field(default_factory=dict)


def load_settings(path: Path) -> Settings:
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    paths = _require_table(data, "paths")
    log_dir = Path(_expand(_require_str(paths, "log_dir", "[paths]")))
    runtime_dir = Path(_expand(_require_str(paths, "runtime_dir", "[paths]")))

    model_roots_raw = data.get("model_roots", {})
    if model_roots_raw is None:
        model_roots_raw = {}
    if not isinstance(model_roots_raw, dict):
        raise ConfigError("[model_roots] must be a table")
    llm_raw = model_roots_raw.get("llm", [])
    if not isinstance(llm_raw, list) or not all(isinstance(item, str) and item for item in llm_raw):
        raise ConfigError("[model_roots]: 'llm' must be an array of non-empty strings")
    llm_model_roots = tuple(Path(_expand(item)) for item in llm_raw)

    monitoring_raw = data.get("monitoring", {})
    if monitoring_raw is None:
        monitoring_raw = {}
    if not isinstance(monitoring_raw, dict):
        raise ConfigError("[monitoring] must be a table")
    gpu_backend = str(monitoring_raw.get("gpu_backend", "auto"))
    monitoring_interval = float(monitoring_raw.get("interval_seconds", 2.0))
    if monitoring_interval < 1.0:
        raise ConfigError("[monitoring]: interval_seconds must be at least 1.0")

    return Settings(
        log_dir=log_dir,
        runtime_dir=runtime_dir,
        llm_model_roots=llm_model_roots,
        gpu_backend=gpu_backend,
        monitoring_interval=monitoring_interval,
    )


def _parse_health(service_id: str, raw: Any) -> HealthCheckConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"service '{service_id}': health must be a table")

    url = _require_str(raw, "url", f"service '{service_id}' health")
    body_contains = _optional_str(raw, "body_contains", f"service '{service_id}' health")

    return HealthCheckConfig(
        url=url,
        expected_status=int(raw.get("expected_status", 200)),
        body_contains=body_contains,
        retry_interval=float(raw.get("retry_interval", 1.0)),
        startup_timeout=float(raw.get("startup_timeout", 120.0)),
        request_timeout=float(raw.get("request_timeout", 2.0)),
    )


def _common_kwargs(service_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    context = f"service '{service_id}'"
    return {
        "id": service_id,
        "display_name": _require_str(raw, "display_name", context),
        "health": _parse_health(service_id, raw.get("health")),
        "open_url": _optional_str(raw, "open_url", context),
        "stop_timeout": float(raw.get("stop_timeout", 10.0)),
        "gpu": bool(raw.get("gpu", False)),
        "conflict_groups": _string_list(raw, "conflict_groups", context),
        "dependencies": _string_list(raw, "dependencies", context),
    }


def load_services(path: Path) -> dict[str, ServiceConfig]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    services_raw = _require_table(data, "services")
    services: dict[str, ServiceConfig] = {}

    for service_id, raw in services_raw.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"service '{service_id}' must be a table")

        service_type = raw.get("type")
        common = _common_kwargs(service_id, raw)

        if service_type == "process":
            executable = Path(_expand(_require_str(raw, "executable", f"service '{service_id}'")))
            cwd = Path(_expand(_require_str(raw, "cwd", f"service '{service_id}'")))

            args_raw = raw.get("args", [])
            if not isinstance(args_raw, list) or not all(isinstance(item, str) for item in args_raw):
                raise ConfigError(f"service '{service_id}': args must be an array of strings")
            args = tuple(_expand(item) for item in args_raw)

            env_raw = raw.get("environment", {})
            if not isinstance(env_raw, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
            ):
                raise ConfigError(f"service '{service_id}': environment must be a string table")
            environment = {key: _expand(value) for key, value in env_raw.items()}

            defaults_raw = raw.get("model_defaults", {})
            if not isinstance(defaults_raw, dict):
                raise ConfigError(f"service '{service_id}': model_defaults must be a table")
            model_defaults: dict[str, str] = {}
            for key, value in defaults_raw.items():
                if not isinstance(key, str) or isinstance(value, (dict, list)):
                    raise ConfigError(
                        f"service '{service_id}': model_defaults values must be scalar values"
                    )
                model_defaults[key] = str(value)

            services[service_id] = ProcessServiceConfig(
                **common,
                executable=executable,
                args=args,
                cwd=cwd,
                environment=environment,
                log_max_bytes=int(raw.get("log_max_bytes", 2_000_000)),
                default_model=_optional_str(raw, "default_model", f"service '{service_id}'"),
                model_defaults=model_defaults,
            )
        elif service_type == "docker":
            services[service_id] = DockerServiceConfig(
                **common,
                container_name=_require_str(raw, "container_name", f"service '{service_id}'"),
                docker_executable=_expand(str(raw.get("docker_executable", "docker"))),
            )
        else:
            raise ConfigError(
                f"service '{service_id}': supported types are 'process' and 'docker'; got {service_type!r}"
            )

    for service_id, service in services.items():
        for dependency in service.dependencies:
            if dependency not in services:
                raise ConfigError(f"service '{service_id}': unknown dependency '{dependency}'")
            if dependency == service_id:
                raise ConfigError(f"service '{service_id}': service cannot depend on itself")

    return services


def _normal_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return path.expanduser().absolute()


def load_models(path: Path, roots: tuple[Path, ...]) -> dict[str, ModelConfig]:
    models = discover_models(roots)
    if not path.exists():
        return models

    with path.open("rb") as fh:
        data = tomllib.load(fh)
    models_raw = data.get("models", {})
    if not isinstance(models_raw, dict):
        raise ConfigError("Expected [models] table")

    by_path = {_normal_path(model.path): model for model in models.values()}

    for model_id, raw in models_raw.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"model '{model_id}' must be a table")
        context = f"model '{model_id}'"
        configured_path = Path(_expand(_require_str(raw, "path", context)))
        base = by_path.get(_normal_path(configured_path))
        if base is None:
            base = ModelConfig(
                id=model_id,
                display_name=configured_path.stem,
                path=configured_path,
                shard_paths=(configured_path,),
                expected_shards=1,
                complete=configured_path.is_file(),
                warning=None if configured_path.is_file() else "configured model file was not found",
                discovered=False,
            )
        else:
            models.pop(base.id, None)

        projector_raw = _optional_str(raw, "projector_path", context)
        projector = Path(_expand(projector_raw)) if projector_raw else base.projector_path

        display_name = raw.get("display_name", base.display_name)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ConfigError(f"{context}: 'display_name' must be a non-empty string")

        models[model_id] = replace(
            base,
            id=model_id,
            display_name=display_name,
            path=configured_path if base.path != configured_path and not base.discovered else base.path,
            family=_optional_str(raw, "family", context) or base.family,
            quant=_optional_str(raw, "quant", context) or base.quant,
            architecture=_optional_str(raw, "architecture", context) or base.architecture,
            recommended_gpu_layers=_optional_int(raw, "recommended_gpu_layers", context)
            if "recommended_gpu_layers" in raw
            else base.recommended_gpu_layers,
            recommended_context_length=_optional_int(raw, "recommended_context_length", context)
            if "recommended_context_length" in raw
            else base.recommended_context_length,
            cache_type_k=_optional_str(raw, "cache_type_k", context) or base.cache_type_k,
            cache_type_v=_optional_str(raw, "cache_type_v", context) or base.cache_type_v,
            vision=bool(raw.get("vision", base.vision)),
            projector_path=projector,
            estimated_vram_mib=_optional_int(raw, "estimated_vram_mib", context)
            if "estimated_vram_mib" in raw
            else base.estimated_vram_mib,
            notes=_optional_str(raw, "notes", context) or base.notes,
        )

    return dict(sorted(models.items()))


def load_profiles(
    path: Path,
    services: dict[str, ServiceConfig],
    models: dict[str, ModelConfig],
) -> dict[str, ProfileConfig]:
    if not path.exists():
        return {}

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    profiles_raw = _require_table(data, "profiles")
    profiles: dict[str, ProfileConfig] = {}

    for profile_id, raw in profiles_raw.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"profile '{profile_id}' must be a table")
        context = f"profile '{profile_id}'"
        profile_services = _string_list(raw, "services", context)
        if not profile_services:
            raise ConfigError(f"{context}: services must not be empty")
        for service_id in profile_services:
            if service_id not in services:
                raise ConfigError(f"{context}: unknown service '{service_id}'")

        open_service = _optional_str(raw, "open_service", context)
        if open_service is not None and open_service not in profile_services:
            raise ConfigError(f"{context}: open_service must be one of the profile services")

        selected_model = _optional_str(raw, "default_model", context)
        if selected_model is None:
            selected_model = _optional_str(raw, "selected_model", context)
        if selected_model is not None and selected_model not in models:
            raise ConfigError(f"{context}: unknown default_model '{selected_model}'")

        profiles[profile_id] = ProfileConfig(
            id=profile_id,
            display_name=_require_str(raw, "display_name", context),
            services=profile_services,
            open_service=open_service,
            selected_model=selected_model,
        )

    return profiles


def load_app_config(config_dir: Path) -> AppConfig:
    settings = load_settings(config_dir / "settings.toml")
    services = load_services(config_dir / "services.toml")
    models = load_models(config_dir / "models.toml", settings.llm_model_roots)

    for service_id, service in services.items():
        if isinstance(service, ProcessServiceConfig) and service.default_model is not None:
            if service.default_model not in models:
                raise ConfigError(
                    f"service '{service_id}': unknown default_model '{service.default_model}'"
                )
        if isinstance(service, ProcessServiceConfig) and service.uses_model:
            if service.default_model is None and not models:
                raise ConfigError(
                    f"service '{service_id}' uses model placeholders but no models are configured or discovered"
                )

    profiles = load_profiles(config_dir / "profiles.toml", services, models)
    return AppConfig(settings=settings, services=services, profiles=profiles, models=models)
