from pathlib import Path

from ai_control_centre.domain import DockerServiceConfig, HarnessConfig, ProcessServiceConfig
from ai_control_centre.harnesses import available_custom_harness_services, discover_agent_harnesses


def test_discovers_configured_computer_service_as_harness():
    services = {
        "llama": ProcessServiceConfig(
            id="llama", display_name="llama.cpp", executable=Path("/bin/true"),
            args=("-m", "{model.path}"), cwd=Path("/tmp"),
        ),
        "computer_dmz": DockerServiceConfig(
            id="computer_dmz", display_name="Computer DMZ", container_name="cptr-dmz",
            open_url="http://127.0.0.1:8001",
        ),
    }
    found = discover_agent_harnesses(services)
    assert len(found) == 1
    harness = next(iter(found.values()))
    assert harness.services == ("computer_dmz",)
    assert harness.open_service == "computer_dmz"


def test_discovery_keeps_existing_and_does_not_duplicate_covered_service():
    services = {
        "computer_dmz": DockerServiceConfig(
            id="computer_dmz", display_name="Computer DMZ", container_name="cptr-dmz"
        )
    }
    existing = {
        "custom": HarnessConfig(
            id="custom", display_name="My Harness", services=("computer_dmz",)
        )
    }
    found = discover_agent_harnesses(services, existing)
    assert found == existing


def test_custom_harness_services_exclude_model_and_prompt_helper():
    services = {
        "llama": ProcessServiceConfig(
            id="llama", display_name="llama", executable=Path("/bin/true"),
            args=("-m", "{model.path}"), cwd=Path("/tmp"),
        ),
        "prompt_helper": ProcessServiceConfig(
            id="prompt_helper", display_name="Prompt Helper", executable=Path("/bin/true"), cwd=Path("/tmp")
        ),
        "computer": DockerServiceConfig(
            id="computer", display_name="Computer", container_name="computer"
        ),
    }
    result = available_custom_harness_services(services, prompt_helper_service="prompt_helper")
    assert set(result) == {"computer"}
