import shutil
from pathlib import Path

import pytest

from ai_control_centre.config import ConfigError, load_app_config, load_settings
from ai_control_centre.domain import DockerServiceConfig, ProcessServiceConfig


def test_load_orson_example():
    root = Path(__file__).resolve().parents[1]
    config = load_app_config(root / "examples" / "orson")
    assert set(config.services) == {"llama", "prompt_helper", "computer_dmz", "comfyui"}
    assert set(config.profiles) == {"coding", "chat", "prompt_helper", "agent", "image"}
    assert config.settings.gpu_backend == "nvidia-smi"
    assert config.settings.monitoring_interval == 2.0
    assert config.settings.prompt_workshop.service == "prompt_helper"
    assert config.settings.prompt_workshop.startup_profile == "prompt_helper"
    assert config.settings.prompt_workshop.model_strategy == "smallest"
    assert config.settings.prompt_workshop.agent_profile == "agent"
    assert config.settings.prompt_workshop.image_profile == "image"
    assert config.settings.prompt_workshop.endpoint == "http://127.0.0.1:8081/v1/chat/completions"

    llama = config.services["llama"]
    assert isinstance(llama, ProcessServiceConfig)
    assert llama.health is not None
    assert llama.health.url.endswith(":8080/health")
    assert "--ctx-size" in llama.args
    assert llama.default_model == "qwen3_coder_30b_heretic"
    assert "{model.path}" in llama.args

    dmz = config.services["computer_dmz"]
    assert isinstance(dmz, DockerServiceConfig)
    assert dmz.container_name == "cptr-dmz"
    assert dmz.dependencies == ("llama",)

    helper = config.services["prompt_helper"]
    assert isinstance(helper, ProcessServiceConfig)
    assert helper.health is not None
    assert helper.health.url.endswith(":8081/health")
    assert "0" in helper.args
    assert helper.gpu is False

    assert "qwen3_coder_30b_heretic" in config.models

    coding = config.profiles["coding"]
    assert coding.open_service == "llama"
    assert coding.default_model == "qwen3_coder_30b_heretic"

    chat = config.profiles["chat"]
    assert chat.services == ("llama",)
    assert chat.open_service == "llama"
    assert chat.default_model == "qwen_qwen3_30b_a3b_instruct_2507_q4_k_m"

    agent = config.profiles["agent"]
    assert agent.services == ("llama", "computer_dmz")
    assert agent.open_service == "computer_dmz"
    assert agent.default_model == "qwen_qwen3_30b_a3b_instruct_2507_q4_k_m"


def test_prompt_workshop_rejects_non_http_endpoint(tmp_path: Path):
    settings = tmp_path / "settings.toml"
    settings.write_text(
        """
[paths]
log_dir = "~/logs"
runtime_dir = "~/runtime"

[prompt_workshop]
endpoint = "file:///tmp/not-an-api"
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="http or https"):
        load_settings(settings)


def test_prompt_workshop_rejects_unknown_profile(tmp_path: Path):
    source = Path(__file__).resolve().parents[1] / "examples" / "orson"
    target = tmp_path / "config"
    shutil.copytree(source, target)
    settings = target / "settings.toml"
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            'agent_profile = "agent"',
            'agent_profile = "missing"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown profile 'missing'"):
        load_app_config(target)
