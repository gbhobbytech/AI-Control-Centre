from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return GpuStatus(False, self.backend, detail=f"GPU query failed: {exc}")

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
            return GpuStatus(False, self.backend, detail=detail)

        line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
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
