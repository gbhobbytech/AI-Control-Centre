from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

DEFAULT_PROMPT_INSTRUCTION = (
    "Rewrite the user's request into a clear, effective prompt. Preserve intent. "
    "Improve specificity, clarity, subject description, style, composition and useful "
    "constraints. Do not add unnecessary complexity. Return only the refined prompt."
)


class ServiceState(str, Enum):
    STOPPED = "Stopped"
    STARTING = "Starting"
    READY = "Ready"
    RUNNING_NOT_READY = "Running, not ready"
    EXTERNAL = "External"
    ERROR = "Error"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class HealthCheckConfig:
    url: str
    expected_status: int = 200
    body_contains: str | None = None
    retry_interval: float = 1.0
    startup_timeout: float = 120.0
    request_timeout: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    path: Path
    shard_paths: tuple[Path, ...] = ()
    expected_shards: int = 1
    complete: bool = True
    warning: str | None = None
    discovered: bool = True
    family: str | None = None
    quant: str | None = None
    architecture: str | None = None
    recommended_gpu_layers: int | None = None
    recommended_context_length: int | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    vision: bool = False
    projector_path: Path | None = None
    estimated_vram_mib: int | None = None
    notes: str | None = None
    tuning_reviewed: bool = False
    tuning_source: str = "unreviewed"
    tuning_schema_version: int = 0
    flash_attention: str | None = None
    max_output_tokens: int | None = None
    startup_timeout_seconds: int | None = None

    @property
    def split(self) -> bool:
        return self.expected_shards > 1


@dataclass(frozen=True)
class CommonServiceConfig:
    id: str
    display_name: str
    health: HealthCheckConfig | None = None
    open_url: str | None = None
    stop_timeout: float = 10.0
    gpu: bool = False
    conflict_groups: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessServiceConfig(CommonServiceConfig):
    executable: Path = Path()
    args: tuple[str, ...] = ()
    cwd: Path = Path()
    environment: Mapping[str, str] = field(default_factory=dict)
    log_max_bytes: int = 2_000_000
    default_model: str | None = None
    model_defaults: Mapping[str, str] = field(default_factory=dict)

    @property
    def command(self) -> Sequence[str]:
        return (str(self.executable), *self.args)

    @property
    def uses_model(self) -> bool:
        return any("{model." in arg for arg in self.args)


@dataclass(frozen=True)
class DockerServiceConfig(CommonServiceConfig):
    container_name: str = ""
    docker_executable: str = "docker"


ServiceConfig = ProcessServiceConfig | DockerServiceConfig


@dataclass(frozen=True)
class HarnessConfig:
    id: str
    display_name: str
    services: tuple[str, ...]
    open_service: str | None = None


@dataclass(frozen=True)
class ProfileConfig:
    id: str
    display_name: str
    services: tuple[str, ...]
    open_service: str | None = None
    selected_model: str | None = None
    harness: str | None = None

    @property
    def default_model(self) -> str | None:
        """Preferred task model. selected_model is retained for config compatibility."""
        return self.selected_model


@dataclass(frozen=True)
class PromptWorkshopConfig:
    enabled: bool = True
    service: str = "prompt_helper"
    startup_profile: str = "prompt_helper"
    endpoint: str = "http://127.0.0.1:8081/v1/chat/completions"
    request_timeout_seconds: float = 120.0
    temperature: float = 0.2
    agent_profile: str = "agent"
    image_profile: str = "image"
    model_strategy: str = "service_default"
    preferred_model: str | None = None
    processing_mode: str = "auto"
    keep_loaded: bool = True
    gpu_layers: int | None = None
    context_length: int = 4096
    port: int = 8081
    startup_timeout_seconds: float = 60.0
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    system_instruction: str = DEFAULT_PROMPT_INSTRUCTION
