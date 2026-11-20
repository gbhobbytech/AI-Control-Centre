from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from ai_control_centre.config import load_app_config, load_services
from ai_control_centre.domain import ProcessServiceConfig
from ai_control_centre.presets import (
    PresetError,
    apply_preset,
    list_presets,
    load_preset,
    parse_preset,
    plan_service_merge,
)

ROOT = Path(__file__).resolve().parents[1]

COMFY_VALUES = {"install_dir": "/opt/ComfyUI", "python": "/opt/ComfyUI/venv/bin/python"}
LLAMA_VALUES = {"executable": "/opt/llama.cpp/build/bin/llama-server", "cwd": "/opt/llama.cpp"}


def _load_rendered(tmp_path: Path, service_id: str, table: dict) -> ProcessServiceConfig:
    path = tmp_path / "services.toml"
    path.unlink(missing_ok=True)
    from ai_control_centre.preferences import write_toml_atomic

    write_toml_atomic(path, {"services": {service_id: table}})
    service = load_services(path)[service_id]
    assert isinstance(service, ProcessServiceConfig)
    return service


def test_bundled_presets_are_available():
    presets = list_presets()
    assert {"llama-cpp", "comfyui"} <= set(presets)
    assert presets["llama-cpp"].service_id == "llama"
    assert presets["comfyui"].service_id == "comfyui"


def test_comfyui_preset_renders_a_loadable_service(tmp_path: Path):
    preset = load_preset("comfyui")
    service = _load_rendered(tmp_path, "comfyui", preset.render(COMFY_VALUES))
    assert service.executable == Path("/opt/ComfyUI/venv/bin/python")
    assert service.cwd == Path("/opt/ComfyUI")
    assert service.args == ("main.py", "--listen", "127.0.0.1", "--port", "8188")
    assert service.health is not None and service.health.url == "http://127.0.0.1:8188/system_stats"
    assert service.open_url == "http://127.0.0.1:8188"
    assert service.uses_model is False


def test_llama_preset_keeps_model_placeholders_and_binds_locally(tmp_path: Path):
    preset = load_preset("llama-cpp")
    service = _load_rendered(tmp_path, "llama", preset.render({**LLAMA_VALUES, "port": "9090"}))
    assert service.uses_model is True
    assert "{model.path}" in service.args
    host = service.args[service.args.index("--host") + 1]
    port = service.args[service.args.index("--port") + 1]
    assert (host, port) == ("127.0.0.1", "9090")
    assert service.health is not None and service.health.url == "http://127.0.0.1:9090/health"


def test_gpu_presets_share_a_conflict_group():
    llama = load_preset("llama-cpp").render(LLAMA_VALUES)
    comfy = load_preset("comfyui").render(COMFY_VALUES)
    assert set(llama["conflict_groups"]) & set(comfy["conflict_groups"])
    # Matches the group used by existing v1.2 configurations.
    with (ROOT / "examples" / "reference" / "services.toml").open("rb") as fh:
        reference = tomllib.load(fh)["services"]
    assert reference["llama"]["conflict_groups"] == llama["conflict_groups"]
    assert reference["comfyui"]["conflict_groups"] == comfy["conflict_groups"]


def test_missing_required_value_is_explained():
    with pytest.raises(PresetError, match=r"needs: python \(Python interpreter"):
        load_preset("comfyui").render({"install_dir": "/opt/ComfyUI"})


@pytest.mark.parametrize("port", ["0", "70000", "http", "-1"])
def test_invalid_port_is_rejected(port):
    with pytest.raises(PresetError, match="port number"):
        load_preset("comfyui").render({**COMFY_VALUES, "port": port})


def test_unknown_value_and_preset_are_rejected():
    with pytest.raises(PresetError, match="unknown value"):
        load_preset("comfyui").render({**COMFY_VALUES, "colour": "blue"})
    with pytest.raises(PresetError, match="Unknown service preset 'ollama'"):
        load_preset("ollama")


def test_invalid_preset_definitions_are_rejected():
    with pytest.raises(PresetError, match="missing \\[service\\]"):
        parse_preset({"preset": {"id": "x", "display_name": "X", "service_id": "x", "version": 1}}, "x.toml")
    with pytest.raises(PresetError, match="undefined placeholder"):
        parse_preset(
            {
                "preset": {"id": "x", "display_name": "X", "service_id": "x", "version": 1},
                "service": {"type": "process", "executable": "{preset.nope}"},
            },
            "x.toml",
        )


def _config_without_comfyui(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(ROOT / "examples" / "reference", target)
    services = target / "services.toml"
    text = services.read_text(encoding="utf-8").split("[services.comfyui]")[0]
    services.write_text("# My hand-written services\n" + text, encoding="utf-8")
    profiles = target / "profiles.toml"
    profiles.write_text(
        profiles.read_text(encoding="utf-8").split("[profiles.image]")[0], encoding="utf-8"
    )
    return target


def test_applying_preset_adds_new_service_and_preserves_existing_text(tmp_path: Path):
    target = _config_without_comfyui(tmp_path)
    services_path = target / "services.toml"
    before = services_path.read_text(encoding="utf-8")

    plan = apply_preset(target, load_preset("comfyui"), COMFY_VALUES)

    after = services_path.read_text(encoding="utf-8")
    assert plan.action == "add" and plan.service_id == "comfyui"
    assert after.startswith(before)  # comments and layout untouched
    assert "# Added by AI Control Centre from the ComfyUI preset" in after
    config = load_app_config(target)
    assert config.services["comfyui"].health.url.endswith(":8188/system_stats")
    assert config.services["llama"] == load_app_config(_config_without_comfyui(tmp_path / "b")).services["llama"]


def test_preset_never_overwrites_user_customisations(tmp_path: Path):
    target = tmp_path / "config"
    shutil.copytree(ROOT / "examples" / "reference", target)
    services_path = target / "services.toml"
    before = services_path.read_bytes()

    plan = apply_preset(target, load_preset("comfyui"), COMFY_VALUES)

    assert plan.action == "keep"
    assert plan.changes_configuration is False
    assert services_path.read_bytes() == before
    # The user's ComfyUI differs from the preset; that is reported, not changed.
    assert {"executable", "cwd", "args"} <= set(plan.customised_keys)
    assert "health.url" not in plan.customised_keys


def test_applying_a_preset_twice_is_repeatable(tmp_path: Path):
    target = _config_without_comfyui(tmp_path)
    apply_preset(target, load_preset("comfyui"), COMFY_VALUES)
    once = (target / "services.toml").read_bytes()
    second = apply_preset(target, load_preset("comfyui"), COMFY_VALUES)
    assert second.action == "keep"
    assert second.missing_keys == () and second.customised_keys == ()
    assert (target / "services.toml").read_bytes() == once


def test_merge_plan_reports_missing_keys():
    preset = load_preset("comfyui")
    rendered = preset.render(COMFY_VALUES)
    partial = {key: value for key, value in rendered.items() if key != "health"}
    plan = plan_service_merge({"comfyui": partial}, preset, COMFY_VALUES)
    assert plan.action == "keep"
    assert "health.url" in plan.missing_keys


def test_preset_can_add_service_under_another_id(tmp_path: Path):
    target = tmp_path / "config"
    shutil.copytree(ROOT / "examples" / "reference", target)
    plan = apply_preset(target, load_preset("comfyui"), {**COMFY_VALUES, "port": "8189"}, "comfyui_alt")
    assert plan.action == "add"
    assert load_app_config(target).services["comfyui_alt"].open_url == "http://127.0.0.1:8189"


def test_invalid_result_leaves_services_file_unchanged(tmp_path: Path):
    target = _config_without_comfyui(tmp_path)
    services_path = target / "services.toml"
    before = services_path.read_bytes()
    preset = load_preset("comfyui")
    broken = type(preset)(**{**preset.__dict__, "service": {**preset.service, "display_name": ""}})
    with pytest.raises(PresetError, match="would make services.toml invalid"):
        apply_preset(target, broken, COMFY_VALUES)
    assert services_path.read_bytes() == before
    assert not list(target.glob("services.toml.*.tmp"))
