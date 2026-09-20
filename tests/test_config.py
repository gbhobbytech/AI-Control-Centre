from pathlib import Path

from ai_control_centre.config import load_app_config
from ai_control_centre.domain import DockerServiceConfig, ProcessServiceConfig


def test_load_orson_example():
    root = Path(__file__).resolve().parents[1]
    config = load_app_config(root / "examples" / "orson")
    assert set(config.services) == {"llama", "computer_dmz", "comfyui"}
    assert set(config.profiles) == {"coding", "chat", "agent", "image"}
    assert config.settings.gpu_backend == "nvidia-smi"
    assert config.settings.monitoring_interval == 2.0

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
