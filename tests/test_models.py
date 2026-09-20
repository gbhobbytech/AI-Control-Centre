from pathlib import Path

import pytest

from ai_control_centre.config import AppConfig, Settings, load_models
from ai_control_centre.domain import ModelConfig, ProcessServiceConfig
from ai_control_centre.manager import ServiceManager
from ai_control_centre.models import (
    discover_models,
    model_total_size_bytes,
    render_model_args,
    smallest_complete_model,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF-test")
    return path


def test_discovery_groups_split_gguf_and_uses_first_shard(tmp_path: Path):
    root = tmp_path / "models"
    _touch(root / "Single-Q8_0.gguf")
    first = _touch(root / "Qwen3.5-35B-Q4_K_M-00001-of-00002.gguf")
    second = _touch(root / "Qwen3.5-35B-Q4_K_M-00002-of-00002.gguf")

    models = discover_models([root])
    assert len(models) == 2

    split = next(model for model in models.values() if model.split)
    assert split.complete
    assert split.path == first
    assert split.shard_paths == (first, second)
    assert split.expected_shards == 2


def test_discovery_marks_missing_split_shards_incomplete(tmp_path: Path):
    root = tmp_path / "models"
    _touch(root / "BigModel-Q4_K_M-00001-of-00003.gguf")
    _touch(root / "BigModel-Q4_K_M-00003-of-00003.gguf")

    models = discover_models([root])
    model = next(iter(models.values()))
    assert not model.complete
    assert model.expected_shards == 3
    assert "00002" in (model.warning or "")


def test_models_toml_overrides_discovered_identity_and_metadata(tmp_path: Path):
    root = tmp_path / "models"
    path = _touch(root / "Coder-Q4_K_M.gguf")
    config = tmp_path / "models.toml"
    config.write_text(
        f'''[models.my_coder]\ndisplay_name = "My Coder"\npath = "{path}"\nfamily = "Qwen"\nrecommended_gpu_layers = 22\n'''
    )

    models = load_models(config, (root,))
    assert "my_coder" in models
    model = models["my_coder"]
    assert model.display_name == "My Coder"
    assert model.family == "Qwen"
    assert model.recommended_gpu_layers == 22
    assert model.complete


def test_render_model_args_uses_metadata_then_service_defaults(tmp_path: Path):
    model = ModelConfig(
        id="test",
        display_name="Test",
        path=tmp_path / "test.gguf",
        shard_paths=(tmp_path / "test.gguf",),
        recommended_gpu_layers=18,
    )
    args = (
        "-m",
        "{model.path}",
        "-ngl",
        "{model.recommended_gpu_layers}",
        "--ctx-size",
        "{model.recommended_context_length}",
    )
    rendered = render_model_args(args, model, {"recommended_context_length": "32768"})
    assert rendered == (
        "-m",
        str(model.path),
        "-ngl",
        "18",
        "--ctx-size",
        "32768",
    )


def test_incomplete_model_cannot_render(tmp_path: Path):
    model = ModelConfig(
        id="broken",
        display_name="Broken",
        path=tmp_path / "first.gguf",
        complete=False,
        warning="missing shard(s): 00002",
        expected_shards=2,
    )
    with pytest.raises(ValueError, match="incomplete"):
        render_model_args(("-m", "{model.path}"), model, {})


def test_manager_renders_requested_model_without_mutating_base_service(tmp_path: Path):
    a = ModelConfig(id="a", display_name="A", path=tmp_path / "a.gguf")
    b = ModelConfig(id="b", display_name="B", path=tmp_path / "b.gguf")
    service = ProcessServiceConfig(
        id="llama",
        display_name="llama",
        executable=Path("/bin/echo"),
        args=("-m", "{model.path}"),
        cwd=tmp_path,
        default_model="a",
    )
    manager = ServiceManager(
        AppConfig(
            settings=Settings(log_dir=tmp_path / "logs", runtime_dir=tmp_path / "runtime"),
            services={"llama": service},
            profiles={},
            models={"a": a, "b": b},
        )
    )
    controller = manager._controller_for_start("llama", "b")
    assert controller.config.args == ("-m", str(b.path))
    assert service.args == ("-m", "{model.path}")


def test_smallest_complete_model_uses_total_split_size(tmp_path: Path):
    small = _touch(tmp_path / "small.gguf")
    small.write_bytes(b"x" * 10)
    split_a = _touch(tmp_path / "split-00001-of-00002.gguf")
    split_b = _touch(tmp_path / "split-00002-of-00002.gguf")
    split_a.write_bytes(b"x" * 8)
    split_b.write_bytes(b"x" * 8)

    small_model = ModelConfig(
        id="small",
        display_name="Small",
        path=small,
        shard_paths=(small,),
    )
    split_model = ModelConfig(
        id="split",
        display_name="Split",
        path=split_a,
        shard_paths=(split_a, split_b),
        expected_shards=2,
    )

    assert model_total_size_bytes(split_model) == 16
    assert smallest_complete_model({"split": split_model, "small": small_model}) == small_model


def test_smallest_complete_model_ignores_incomplete_models(tmp_path: Path):
    tiny = _touch(tmp_path / "tiny.gguf")
    large = _touch(tmp_path / "large.gguf")
    tiny.write_bytes(b"x")
    large.write_bytes(b"x" * 20)
    incomplete = ModelConfig(
        id="tiny", display_name="Tiny", path=tiny, shard_paths=(tiny,), complete=False
    )
    complete = ModelConfig(
        id="large", display_name="Large", path=large, shard_paths=(large,), complete=True
    )
    assert smallest_complete_model({"tiny": incomplete, "large": complete}) == complete
