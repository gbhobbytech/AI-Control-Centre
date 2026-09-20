from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    backend: str
    name: str | None = None
    memory_total_mib: int | None = None
    memory_used_mib: int | None = None
    utilization_percent: int | None = None
    temperature_c: int | None = None
    detail: str = ""

    @property
    def memory_free_mib(self) -> int | None:
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        return max(0, self.memory_total_mib - self.memory_used_mib)


@dataclass(frozen=True)
class CpuCounters:
    idle: int
    total: int


@dataclass(frozen=True)
class HostStatus:
    memory_total_mib: float | None = None
    memory_used_mib: float | None = None
    cpu_utilization_percent: float | None = None
    detail: str = ""

    @property
    def memory_percent(self) -> float | None:
        if self.memory_total_mib is None or self.memory_used_mib is None:
            return None
        if self.memory_total_mib <= 0:
            return None
        return _clamp_percent(self.memory_used_mib / self.memory_total_mib * 100.0)


def _clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def parse_proc_meminfo(text: str) -> tuple[float, float]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            values[key] = int(fields[0])
        except ValueError:
            continue

    total_kib = values.get("MemTotal")
    available_kib = values.get("MemAvailable")
    if available_kib is None:
        fallback = ("MemFree", "Buffers", "Cached")
        if all(key in values for key in fallback):
            available_kib = sum(values[key] for key in fallback)
    if total_kib is None or available_kib is None or total_kib <= 0:
        raise ValueError("Could not read total and available memory from /proc/meminfo")

    used_kib = max(0, min(total_kib, total_kib - available_kib))
    return total_kib / 1024.0, used_kib / 1024.0


def parse_proc_stat(text: str) -> CpuCounters:
    line = next((line for line in text.splitlines() if line.startswith("cpu ")), "")
    fields = line.split()
    if len(fields) < 5:
        raise ValueError("Could not read aggregate CPU counters from /proc/stat")
    try:
        values = [int(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError("Invalid aggregate CPU counters in /proc/stat") from exc
    # Linux reports guest time inside user/nice as well as separate guest fields.
    # Sum only user through steal to avoid counting guest time twice.
    accounted = values[:8]
    idle = accounted[3] + (accounted[4] if len(accounted) > 4 else 0)
    return CpuCounters(idle=idle, total=sum(accounted))


def calculate_cpu_percent(previous: CpuCounters, current: CpuCounters) -> float | None:
    total_delta = current.total - previous.total
    idle_delta = current.idle - previous.idle
    if total_delta <= 0 or idle_delta < 0:
        return None
    return _clamp_percent((total_delta - idle_delta) / total_delta * 100.0)


class LinuxHostMonitor:
    def __init__(
        self,
        meminfo_path: Path = Path("/proc/meminfo"),
        stat_path: Path = Path("/proc/stat"),
    ):
        self.meminfo_path = meminfo_path
        self.stat_path = stat_path
        self._previous_cpu: CpuCounters | None = None
        self._lock = threading.Lock()

    def status(self) -> HostStatus:
        with self._lock:
            errors: list[str] = []
            memory_total: float | None = None
            memory_used: float | None = None
            cpu_percent: float | None = None

            try:
                memory_total, memory_used = parse_proc_meminfo(
                    self.meminfo_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                errors.append(str(exc))

            try:
                current_cpu = parse_proc_stat(
                    self.stat_path.read_text(encoding="utf-8")
                )
                if self._previous_cpu is not None:
                    cpu_percent = calculate_cpu_percent(self._previous_cpu, current_cpu)
                self._previous_cpu = current_cpu
            except (OSError, ValueError) as exc:
                errors.append(str(exc))

            return HostStatus(
                memory_total_mib=memory_total,
                memory_used_mib=memory_used,
                cpu_utilization_percent=cpu_percent,
                detail="; ".join(errors) if errors else "OK",
            )


class NvidiaSmiMonitor:
    backend = "nvidia-smi"

    def __init__(self, executable: str = "nvidia-smi"):
        self.executable = executable

    def status(self) -> GpuStatus:
        executable = shutil.which(self.executable)
        if executable is None:
            return GpuStatus(False, self.backend, detail=f"{self.executable} not found")

        query = "name,memory.total,memory.used,utilization.gpu,temperature.gpu"
        try:
            result = subprocess.run(
                [
                    executable,
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return GpuStatus(False, self.backend, detail=f"GPU query failed: {exc}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
            return GpuStatus(False, self.backend, detail=detail)

        line = next(
            (item.strip() for item in result.stdout.splitlines() if item.strip()), ""
        )
        try:
            return parse_nvidia_smi_line(line)
        except ValueError as exc:
            return GpuStatus(False, self.backend, detail=str(exc))


def parse_nvidia_smi_line(line: str) -> GpuStatus:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        raise ValueError(f"Unexpected nvidia-smi output: {line!r}")

    name, total, used, util, temp = parts
    try:
        return GpuStatus(
            available=True,
            backend="nvidia-smi",
            name=name,
            memory_total_mib=int(total),
            memory_used_mib=int(used),
            utilization_percent=int(util),
            temperature_c=int(temp),
            detail="OK",
        )
    except ValueError as exc:
        raise ValueError(f"Could not parse nvidia-smi output: {line!r}") from exc


def create_gpu_monitor(backend: str):
    value = (backend or "auto").strip().lower()
    if value in {"auto", "nvidia", "nvidia-smi"}:
        return NvidiaSmiMonitor()
    if value in {"none", "off", "unavailable"}:
        return None
    raise ValueError(f"Unsupported GPU monitoring backend: {backend}")
