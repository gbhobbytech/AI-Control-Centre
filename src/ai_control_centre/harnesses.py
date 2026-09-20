from __future__ import annotations

import re
from collections.abc import Mapping

from .domain import HarnessConfig, ProcessServiceConfig, ServiceConfig

_AGENT_HINTS = (
    "agent",
    "computer",
    "cptr",
    "openhands",
    "open hands",
    "openwebui",
    "open webui",
    "browser-use",
    "browser use",
    "autogen",
)


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "agent_harness"


def _service_is_main_llm(service: ServiceConfig) -> bool:
    return isinstance(service, ProcessServiceConfig) and service.uses_model


def looks_like_agent_service(service_id: str, service: ServiceConfig) -> bool:
    """Conservative signature check for already configured agent runtimes.

    Harness discovery does not treat every Docker container or process as an
    agent. It only proposes configured services whose identifiers/display names
    look like recognised agent runtimes. Users can always add a custom harness.
    """
    if _service_is_main_llm(service):
        return False
    text = " ".join(
        part for part in (
            service_id,
            service.display_name,
            getattr(service, "container_name", ""),
            service.open_url or "",
        ) if part
    ).lower()
    return any(hint in text for hint in _AGENT_HINTS)


def discover_agent_harnesses(
    services: Mapping[str, ServiceConfig],
    existing: Mapping[str, HarnessConfig] | None = None,
) -> dict[str, HarnessConfig]:
    """Return usable harness suggestions from configured services.

    Existing harnesses are retained. New suggestions are one-service harnesses;
    multi-service/custom combinations are created explicitly in the UI.
    """
    result = dict(existing or {})
    already_covered = {sid for harness in result.values() for sid in harness.services}
    for service_id, service in services.items():
        if service_id in already_covered or not looks_like_agent_service(service_id, service):
            continue
        base = _slug(service.display_name)
        harness_id = base
        suffix = 2
        while harness_id in result:
            harness_id = f"{base}_{suffix}"
            suffix += 1
        result[harness_id] = HarnessConfig(
            id=harness_id,
            display_name=service.display_name,
            services=(service_id,),
            open_service=service_id if service.open_url else None,
        )
    return result


def available_custom_harness_services(
    services: Mapping[str, ServiceConfig],
    *,
    prompt_helper_service: str | None = None,
) -> dict[str, ServiceConfig]:
    """Services that can sensibly be grouped into a custom agent harness."""
    result: dict[str, ServiceConfig] = {}
    for service_id, service in services.items():
        if service_id == prompt_helper_service or _service_is_main_llm(service):
            continue
        result[service_id] = service
    return result
