from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .domain import (
    DockerServiceConfig,
    HealthCheckConfig,
    ModelConfig,
    ProcessServiceConfig,
    ProfileConfig,
    PromptWorkshopConfig,
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
        raise ConfigError(
            f"{context}: '{key}' must be a non-empty string when supplied"
        )
    return value


def _string_list(parent: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = parent.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
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
    setup_completed: bool = False
    prompt_workshop: PromptWorkshopConfig = field(default_factory=PromptWorkshopConfig)
    appearance: dict = field(default_factory=dict)


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
    if not isinstance(llm_raw, list) or not all(
        isinstance(item, str) and item for item in llm_raw
    ):
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

    setup_raw = data.get("setup", {})
    if setup_raw is None:
        setup_raw = {}
    if not isinstance(setup_raw, dict):
        raise ConfigError("[setup] must be a table")
    setup_completed = bool(setup_raw.get("completed", False))

    prompt_raw = data.get("prompt_workshop", {})
    if prompt_raw is None:
        prompt_raw = {}
    if not isinstance(prompt_raw, dict):
        raise ConfigError("[prompt_workshop] must be a table")
    defaults = PromptWorkshopConfig()
    context = "[prompt_workshop]"

    port = int(prompt_raw.get("port", defaults.port))
    if not 1 <= port <= 65535:
        raise ConfigError("[prompt_workshop]: port must be between 1 and 65535")
    endpoint_default = f"http://127.0.0.1:{port}/v1/chat/completions"
    endpoint = str(prompt_raw.get("endpoint", endpoint_default)).strip()
    parsed_endpoint = urlparse(endpoint)
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
        raise ConfigError("[prompt_workshop]: endpoint must be an http or https URL")
    request_timeout = float(
        prompt_raw.get("request_timeout_seconds", defaults.request_timeout_seconds)
    )
    if request_timeout <= 0:
        raise ConfigError("[prompt_workshop]: request_timeout_seconds must be positive")
    temperature = float(prompt_raw.get("temperature", defaults.temperature))
    if not 0.0 <= temperature <= 2.0:
        raise ConfigError("[prompt_workshop]: temperature must be between 0.0 and 2.0")
    model_strategy = str(prompt_raw.get("model_strategy", defaults.model_strategy)).strip()
    if model_strategy not in {"service_default", "smallest", "manual"}:
        raise ConfigError(
            "[prompt_workshop]: model_strategy must be 'service_default', 'smallest' or 'manual'"
        )
    preferred_model = _optional_str(prompt_raw, "preferred_model", context)
    processing_mode = str(prompt_raw.get("processing_mode", defaults.processing_mode)).strip().lower()
    if processing_mode not in {"cpu", "gpu", "auto"}:
        raise ConfigError("[prompt_workshop]: processing_mode must be cpu, gpu or auto")
    gpu_layers = _optional_int(prompt_raw, "gpu_layers", context)
    if gpu_layers is not None and gpu_layers < 0:
        raise ConfigError("[prompt_workshop]: gpu_layers must be zero or greater")
    context_length = int(prompt_raw.get("context_length", defaults.context_length))
    if context_length < 512:
        raise ConfigError("[prompt_workshop]: context_length must be at least 512")
    helper_startup_timeout = float(
        prompt_raw.get("startup_timeout_seconds", defaults.startup_timeout_seconds)
    )
    if helper_startup_timeout <= 0:
        raise ConfigError("[prompt_workshop]: startup_timeout_seconds must be positive")
    cache_type_k = str(prompt_raw.get("cache_type_k", defaults.cache_type_k)).strip()
    cache_type_v = str(prompt_raw.get("cache_type_v", defaults.cache_type_v)).strip()
    if not cache_type_k or not cache_type_v:
        raise ConfigError("[prompt_workshop]: cache types must not be empty")

    prompt_workshop = PromptWorkshopConfig(
        service=_require_str(prompt_raw, "service", context)
        if "service" in prompt_raw
        else defaults.service,
        startup_profile=_require_str(prompt_raw, "startup_profile", context)
        if "startup_profile" in prompt_raw
        else defaults.startup_profile,
        endpoint=endpoint,
        request_timeout_seconds=request_timeout,
        temperature=temperature,
        agent_profile=_require_str(prompt_raw, "agent_profile", context)
        if "agent_profile" in prompt_raw
        else defaults.agent_profile,
        image_profile=_require_str(prompt_raw, "image_profile", context)
        if "image_profile" in prompt_raw
        else defaults.image_profile,
        model_strategy=model_strategy,
        preferred_model=preferred_model,
        processing_mode=processing_mode,
        keep_loaded=bool(prompt_raw.get("keep_loaded", defaults.keep_loaded)),
        gpu_layers=gpu_layers,
        context_length=context_length,
        port=port,
        startup_timeout_seconds=helper_startup_timeout,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        system_instruction=_require_str(prompt_raw, "system_instruction", context)
        if "system_instruction" in prompt_raw
        else defaults.system_instruction,
    )

    return Settings(
        log_dir=log_dir,
        runtime_dir=runtime_dir,
        llm_model_roots=llm_model_roots,
        gpu_backend=gpu_backend,
        monitoring_interval=monitoring_interval,
        setup_completed=setup_completed,
        prompt_workshop=prompt_workshop,
        appearance=_load_appearance(data.get("appearance", {})),
    )


def _load_appearance(raw) -> dict:
    from .theme import validate_appearance
    try:
        return validate_appearance(raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_health(service_id: str, raw: Any) -> HealthCheckConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"service '{service_id}': health must be a table")

    url = _require_str(raw, "url", f"service '{service_id}' health")
    body_contains = _optional_str(
        raw, "body_contains", f"service '{service_id}' health"
    )

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
            executable = Path(
                _expand(_require_str(raw, "executable", f"service '{service_id}'"))
            )
            cwd = Path(_expand(_require_str(raw, "cwd", f"service '{service_id}'")))

            args_raw = raw.get("args", [])
            if not isinstance(args_raw, list) or not all(
                isinstance(item, str) for item in args_raw
            ):
                raise ConfigError(
                    f"service '{service_id}': args must be an array of strings"
                )
            args = tuple(_expand(item) for item in args_raw)

            env_raw = raw.get("environment", {})
            if not isinstance(env_raw, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
            ):
                raise ConfigError(
                    f"service '{service_id}': environment must be a string table"
                )
            environment = {key: _expand(value) for key, value in env_raw.items()}

            defaults_raw = raw.get("model_defaults", {})
            if not isinstance(defaults_raw, dict):
                raise ConfigError(
                    f"service '{service_id}': model_defaults must be a table"
                )
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
                default_model=_optional_str(
                    raw, "default_model", f"service '{service_id}'"
                ),
                model_defaults=model_defaults,
            )
        elif service_type == "docker":
            services[service_id] = DockerServiceConfig(
                **common,
                container_name=_require_str(
                    raw, "container_name", f"service '{service_id}'"
                ),
                docker_executable=_expand(str(raw.get("docker_executable", "docker"))),
            )
        else:
            raise ConfigError(
                f"service '{service_id}': supported types are 'process' and 'docker'; got {service_type!r}"
            )

    for service_id, service in services.items():
        for dependency in service.dependencies:
            if dependency not in services:
                raise ConfigError(
                    f"service '{service_id}': unknown dependency '{dependency}'"
                )
            if dependency == service_id:
                raise ConfigError(
                    f"service '{service_id}': service cannot depend on itself"
                )

    return services



def _replace_cli_flag(args: tuple[str, ...], flag: str, value: str) -> tuple[str, ...]:
    items = list(args)
    try:
        index = items.index(flag)
    except ValueError:
        items.extend([flag, value])
    else:
        if index + 1 >= len(items):
            items.append(value)
        else:
            items[index + 1] = value
    return tuple(items)


def _apply_prompt_helper_settings(
    services: dict[str, ServiceConfig],
    workshop: PromptWorkshopConfig,
) -> dict[str, ServiceConfig]:
    service = services.get(workshop.service)
    if not isinstance(service, ProcessServiceConfig):
        return services

    args = service.args
    args = _replace_cli_flag(args, "--ctx-size", str(workshop.context_length))
    args = _replace_cli_flag(args, "--cache-type-k", workshop.cache_type_k)
    args = _replace_cli_flag(args, "--cache-type-v", workshop.cache_type_v)
    args = _replace_cli_flag(args, "--port", str(workshop.port))

    health = service.health
    if health is not None:
        parsed = urlparse(health.url)
        host = parsed.hostname or "127.0.0.1"
        scheme = parsed.scheme or "http"
        path = parsed.path or "/health"
        health = replace(
            health,
            url=f"{scheme}://{host}:{workshop.port}{path}",
            startup_timeout=workshop.startup_timeout_seconds,
        )

    updated = dict(services)
    updated[workshop.service] = replace(
        service,
        args=args,
        health=health,
        gpu=workshop.processing_mode == "gpu",
    )
    return updated

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
                warning=None
                if configured_path.is_file()
                else "configured model file was not found",
                discovered=False,
            )
        else:
            models.pop(base.id, None)

        projector_raw = _optional_str(raw, "projector_path", context)
        projector = (
            Path(_expand(projector_raw)) if projector_raw else base.projector_path
        )

        display_name = raw.get("display_name", base.display_name)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ConfigError(f"{context}: 'display_name' must be a non-empty string")

        models[model_id] = replace(
            base,
            id=model_id,
            display_name=display_name,
            path=configured_path
            if base.path != configured_path and not base.discovered
            else base.path,
            family=_optional_str(raw, "family", context) or base.family,
            quant=_optional_str(raw, "quant", context) or base.quant,
            architecture=_optional_str(raw, "architecture", context)
            or base.architecture,
            recommended_gpu_layers=_optional_int(raw, "recommended_gpu_layers", context)
            if "recommended_gpu_layers" in raw
            else base.recommended_gpu_layers,
            recommended_context_length=_optional_int(
                raw, "recommended_context_length", context
            )
            if "recommended_context_length" in raw
            else base.recommended_context_length,
            cache_type_k=_optional_str(raw, "cache_type_k", context)
            or base.cache_type_k,
            cache_type_v=_optional_str(raw, "cache_type_v", context)
            or base.cache_type_v,
            vision=bool(raw.get("vision", base.vision)),
            projector_path=projector,
            estimated_vram_mib=_optional_int(raw, "estimated_vram_mib", context)
            if "estimated_vram_mib" in raw
            else base.estimated_vram_mib,
            notes=_optional_str(raw, "notes", context) or base.notes,
            tuning_reviewed=raw.get("tuning_reviewed", False) is True,
            tuning_source=str(raw.get("tuning_source", "unreviewed")),
            flash_attention=_optional_str(raw, "flash_attention", context),
            max_output_tokens=_optional_int(raw, "max_output_tokens", context),
            startup_timeout_seconds=_optional_int(raw, "startup_timeout_seconds", context),
        )

    from .tuning import model_values, validate_values
    for model in models.values():
        if model.tuning_reviewed:
            try:
                validate_values(model_values(model))
            except ValueError as exc:
                raise ConfigError(f"Model '{model.id}': {exc}") from exc
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
            raise ConfigError(
                f"{context}: open_service must be one of the profile services"
            )

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
    services = _apply_prompt_helper_settings(services, settings.prompt_workshop)
    models = load_models(config_dir / "models.toml", settings.llm_model_roots)

    for service_id, service in services.items():
        if (
            isinstance(service, ProcessServiceConfig)
            and service.default_model is not None
            and service.default_model not in models
        ):
            raise ConfigError(
                f"service '{service_id}': unknown default_model '{service.default_model}'"
            )
        if (
            isinstance(service, ProcessServiceConfig)
            and service.uses_model
            and service.default_model is None
            and not models
        ):
            raise ConfigError(
                f"service '{service_id}' uses model placeholders but no models are configured or discovered"
            )

    profiles = load_profiles(config_dir / "profiles.toml", services, models)
    workshop = settings.prompt_workshop
    if workshop.model_strategy == "manual":
        if workshop.preferred_model is None:
            raise ConfigError("[prompt_workshop]: manual model strategy requires preferred_model")
        if workshop.preferred_model not in models:
            raise ConfigError(
                f"[prompt_workshop]: unknown preferred_model '{workshop.preferred_model}'"
            )
    if workshop.service not in services:
        raise ConfigError(f"[prompt_workshop]: unknown service '{workshop.service}'")
    for key, profile_id in (
        ("startup_profile", workshop.startup_profile),
        ("agent_profile", workshop.agent_profile),
        ("image_profile", workshop.image_profile),
    ):
        if profile_id not in profiles:
            raise ConfigError(
                f"[prompt_workshop]: {key} refers to unknown profile '{profile_id}'"
            )
    return AppConfig(
        settings=settings, services=services, profiles=profiles, models=models
    )
