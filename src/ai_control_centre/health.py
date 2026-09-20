from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .domain import HealthCheckConfig


@dataclass(frozen=True)
class HealthResult:
    ok: bool
    detail: str
    status_code: int | None = None


def check_http(config: HealthCheckConfig) -> HealthResult:
    request = urllib.request.Request(config.url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout) as response:
            status = response.getcode()
            body = response.read(64 * 1024).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return HealthResult(False, f"HTTP {exc.code}", exc.code)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return HealthResult(False, f"connection failed: {exc}")

    if status != config.expected_status:
        return HealthResult(False, f"expected HTTP {config.expected_status}, got {status}", status)

    if config.body_contains and config.body_contains not in body:
        return HealthResult(False, f"response did not contain {config.body_contains!r}", status)

    return HealthResult(True, f"HTTP {status}", status)


def wait_until_ready(config: HealthCheckConfig, process_is_alive) -> HealthResult:
    deadline = time.monotonic() + config.startup_timeout
    last = HealthResult(False, "not checked")

    while time.monotonic() < deadline:
        if not process_is_alive():
            return HealthResult(False, "process exited before readiness")

        last = check_http(config)
        if last.ok:
            return last
        time.sleep(config.retry_interval)

    return HealthResult(False, f"startup timeout; last health result: {last.detail}")
