from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ProcessRecord:
    service_id: str
    pid: int
    pgid: int
    sid: int
    proc_start_ticks: int
    executable: str
    command_sha256: str
    model_id: str | None = None
    launch_mode: str | None = None


@dataclass
class DockerRecord:
    service_id: str
    container_name: str
    container_id: str
    started_at: str


def command_hash(command: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in command:
        digest.update(item.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def proc_start_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    tail = text[text.rfind(")") + 2 :].split()
    return int(tail[19])


def proc_cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]


def proc_executable(pid: int) -> str:
    return str(Path(f"/proc/{pid}/exe").resolve())


class RuntimeStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, service_id: str) -> Path:
        return self.directory / f"{service_id}.json"

    def _save_dict(self, service_id: str, payload: dict[str, Any]) -> None:
        path = self._path(service_id)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, path)

    def save_process(self, record: ProcessRecord) -> None:
        self._save_dict(record.service_id, {"kind": "process", **asdict(record)})

    def save_docker(self, record: DockerRecord) -> None:
        self._save_dict(record.service_id, {"kind": "docker", **asdict(record)})

    def save(self, record: ProcessRecord) -> None:
        self.save_process(record)

    def _load_dict(self, service_id: str) -> dict[str, Any] | None:
        path = self._path(service_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def load_process(self, service_id: str) -> ProcessRecord | None:
        data = self._load_dict(service_id)
        if data is None:
            return None
        data = dict(data)
        kind = data.pop("kind", "process")
        if kind != "process":
            return None
        try:
            return ProcessRecord(**data)
        except TypeError:
            return None

    def load_docker(self, service_id: str) -> DockerRecord | None:
        data = self._load_dict(service_id)
        if data is None:
            return None
        data = dict(data)
        if data.pop("kind", None) != "docker":
            return None
        try:
            return DockerRecord(**data)
        except TypeError:
            return None

    def load(self, service_id: str) -> ProcessRecord | None:
        return self.load_process(service_id)

    def delete(self, service_id: str) -> None:
        try:
            self._path(service_id).unlink()
        except FileNotFoundError:
            pass


def record_matches_process(record: ProcessRecord) -> bool:
    try:
        if proc_start_ticks(record.pid) != record.proc_start_ticks:
            return False
        if os.getpgid(record.pid) != record.pgid:
            return False
        if os.getsid(record.pid) != record.sid:
            return False
        if proc_executable(record.pid) != record.executable:
            return False
        if command_hash(proc_cmdline(record.pid)) != record.command_sha256:
            return False
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return False
    return True
