from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


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
class ProfileConfig:
    id: str
    display_name: str
    services: tuple[str, ...]
    open_service: str | None = None
    selected_model: str | None = None

    @property
    def default_model(self) -> str | None:
        """Preferred task model. selected_model is retained for config compatibility."""
        return self.selected_model
