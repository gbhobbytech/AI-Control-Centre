from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .domain import ServiceState


@dataclass(frozen=True)
class ServiceStatus:
    state: ServiceState
    detail: str
    pid: int | None = None


class ServiceController(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    def status(self) -> ServiceStatus: ...

    def start(self) -> ServiceStatus: ...

    def stop(self) -> ServiceStatus: ...

    def logs(self, lines: int = 40) -> list[str]: ...

    def open(self) -> bool: ...
