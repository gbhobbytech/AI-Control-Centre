from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ai_control_centre.config import load_app_config
from ai_control_centre.discovery import (
    COMFYUI,
    LLAMA_CPP,
    SOURCE_COMMON,
    SOURCE_CONFIGURED,
    SOURCE_PATH,
    discover_installations,
    find_comfyui,
    find_llama_cpp,
    portable_path,
    validate_comfyui,
    validate_llama_executable,
)

ROOT = Path(__file__).resolve().parents[1]


def _program(path: Path, header: bytes = b"\x7fELF\x02\x01\x01") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"\0" * 32)
    path.chmod(0o755)
    return path


def _venv(path: Path, *, torch: bool) -> Path:
    python = _program(path / "bin" / "python", b"#!/bin/sh\n")
    site = path / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    if torch:
        (site / "torch").mkdir()
    return python


def _comfyui(path: Path) -> Path:
    (path / "comfy").mkdir(parents=True)
    (path / "main.py").write_text("# ComfyUI\n", encoding="utf-8")
    return path


def _orson_mini(home: Path) -> None:
    """Layout reported on Orson Mini (8 GB RTX 2000 Ada)."""
    _program(home / "ai/apps/llama.cpp/build/bin/llama-server")
    _venv(_comfyui(home / "ComfyUI") / "venv", torch=True)
    _venv(home / ".venv", torch=False)  # unrelated home-folder environment


def _orson(home: Path) -> None:
    """Layout described for Orson (12 GB RTX A2000)."""
    _program(home / "ai/apps/llama.cpp/build/bin/llama-server")
    _venv(_comfyui(home / "ai/ComfyUI") / "venv", torch=True)


# --- llama.cpp -----------------------------------------------------------------


def test_finds_llama_server_six_folders_deep(tmp_path: Path):
    _orson_mini(tmp_path)
    [found] = find_llama_cpp(home=tmp_path, path_env="")
    assert found.usable and found.recommended
    assert found.source == SOURCE_COMMON
    assert found.executable == tmp_path / "ai/apps/llama.cpp/build/bin/llama-server"
    assert found.cwd == tmp_path / "ai/apps/llama.cpp"
    assert found.preset_values(tmp_path) == {
        "executable": "~/ai/apps/llama.cpp/build/bin/llama-server",
        "cwd": "~/ai/apps/llama.cpp",
    }


def test_llama_server_on_path_and_configured_come_first(tmp_path: Path):
    _orson_mini(tmp_path)
    bin_dir = tmp_path / "usr-local-bin"
    _program(bin_dir / "llama-server")
    configured = tmp_path / "custom/llama-server"
    _program(configured)

    found = find_llama_cpp(home=tmp_path, configured=[configured], path_env=str(bin_dir))
    assert [item.source for item in found] == [SOURCE_CONFIGURED, SOURCE_PATH, SOURCE_COMMON]
    assert [item.recommended for item in found] == [True, False, False]


def test_missing_configured_llama_is_reported_and_next_usable_is_recommended(tmp_path: Path):
    _orson_mini(tmp_path)
    found = find_llama_cpp(home=tmp_path, configured=[tmp_path / "gone/llama-server"], path_env="")
    assert found[0].source == SOURCE_CONFIGURED and not found[0].usable
    assert "does not exist" in found[0].problems[0]
    assert found[1].recommended and found[1].usable


def test_same_llama_reached_two_ways_is_listed_once(tmp_path: Path):
    _orson_mini(tmp_path)
    executable = tmp_path / "ai/apps/llama.cpp/build/bin/llama-server"
    found = find_llama_cpp(home=tmp_path, configured=[executable], path_env=str(executable.parent))
    assert len(found) == 1 and found[0].source == SOURCE_CONFIGURED


def test_no_llama_cpp_found(tmp_path: Path):
    assert find_llama_cpp(home=tmp_path, path_env="") == []


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda p: None, "does not exist"),
        (lambda p: p.mkdir(parents=True), "is a folder; choose the llama-server program"),
        (lambda p: (p.parent.mkdir(parents=True), p.write_bytes(b"\x7fELF")), "is not executable"),
        (lambda p: _program(p, b"just text"), "is not a program"),
    ],
)
def test_invalid_llama_paths_are_explained(tmp_path: Path, setup, message):
    target = tmp_path / "llama.cpp/build/bin/llama-server"
    setup(target)
    result = validate_llama_executable(target)
    assert not result.usable
    assert message in result.problems[0]
    with pytest.raises(ValueError, match="cannot be used"):
        result.preset_values(tmp_path)


def test_unexpected_executable_name_is_a_note_not_a_failure(tmp_path: Path):
    result = validate_llama_executable(_program(tmp_path / "bin/server"))
    assert result.usable
    assert "not llama-server" in result.notes[0]


def test_user_path_with_tilde_is_expanded(tmp_path: Path):
    _orson_mini(tmp_path)
    result = validate_llama_executable("~/ai/apps/llama.cpp/build/bin/llama-server", home=tmp_path)
    assert result.usable


# --- ComfyUI -------------------------------------------------------------------


def test_detects_comfyui_on_orson_mini_and_ignores_home_venv(tmp_path: Path):
    _orson_mini(tmp_path)
    [found] = find_comfyui(home=tmp_path)
    assert found.usable and found.recommended
    assert found.path == tmp_path / "ComfyUI"
    assert found.executable == tmp_path / "ComfyUI/venv/bin/python"
    assert found.preset_values(tmp_path) == {
        "install_dir": "~/ComfyUI",
        "python": "~/ComfyUI/venv/bin/python",
    }


def test_detects_comfyui_on_orson(tmp_path: Path):
    _orson(tmp_path)
    [found] = find_comfyui(home=tmp_path)
    assert found.path == tmp_path / "ai/ComfyUI"
    assert found.executable == tmp_path / "ai/ComfyUI/venv/bin/python"


def test_dot_venv_and_parent_venv_layouts(tmp_path: Path):
    comfy = _comfyui(tmp_path / "ComfyUI")
    _venv(comfy / ".venv", torch=True)
    assert validate_comfyui(comfy).executable == comfy / ".venv/bin/python"

    shared = _comfyui(tmp_path / "comfy/ComfyUI")
    _venv(tmp_path / "comfy/.venv", torch=True)  # comfy-cli style
    assert validate_comfyui(shared).executable == tmp_path / "comfy/.venv/bin/python"


def test_environment_without_torch_is_skipped_for_one_with_torch(tmp_path: Path):
    comfy = _comfyui(tmp_path / "ComfyUI")
    _venv(comfy / "venv", torch=False)
    _venv(comfy / ".venv", torch=True)
    result = validate_comfyui(comfy)
    assert result.executable == comfy / ".venv/bin/python"
    assert "without torch" in result.notes[0]


def test_comfyui_without_usable_python_is_explained(tmp_path: Path):
    comfy = _comfyui(tmp_path / "ComfyUI")
    assert "No Python environment was found" in validate_comfyui(comfy).problems[0]
    _venv(comfy / "venv", torch=False)
    assert "No Python environment with torch" in validate_comfyui(comfy).problems[0]


def test_folder_with_main_py_but_not_comfyui_is_rejected(tmp_path: Path):
    other = tmp_path / "ComfyUI"
    other.mkdir()
    (other / "main.py").write_text("print('hello')\n", encoding="utf-8")
    assert find_comfyui(home=tmp_path) == []
    assert "is not a ComfyUI folder" in validate_comfyui(other).problems[0]


def test_user_selected_python_without_visible_torch_is_allowed_with_note(tmp_path: Path):
    comfy = _comfyui(tmp_path / "ComfyUI")
    conda_python = _program(tmp_path / "miniconda3/envs/comfy/bin/python", b"#!/bin/sh\n")
    result = validate_comfyui(comfy, conda_python)
    assert result.usable
    assert "Could not confirm that torch" in result.notes[0]


def test_invalid_user_selected_python_is_rejected(tmp_path: Path):
    comfy = _comfyui(tmp_path / "ComfyUI")
    assert "does not exist" in validate_comfyui(comfy, tmp_path / "nope/python").problems[0]
    assert "not an executable Python" in validate_comfyui(comfy, comfy).problems[0]


def test_multiple_comfyui_installations_are_all_offered(tmp_path: Path):
    _orson_mini(tmp_path)
    _venv(_comfyui(tmp_path / "ai/ComfyUI") / "venv", torch=True)
    found = find_comfyui(home=tmp_path)
    assert [item.path for item in found] == [tmp_path / "ComfyUI", tmp_path / "ai/ComfyUI"]
    assert [item.recommended for item in found] == [True, False]


def test_configured_comfyui_is_preferred_and_not_duplicated(tmp_path: Path):
    _orson_mini(tmp_path)
    _venv(_comfyui(tmp_path / "ai/ComfyUI") / "venv", torch=True)
    configured = [(tmp_path / "ai/ComfyUI", tmp_path / "ai/ComfyUI/venv/bin/python")]
    found = find_comfyui(home=tmp_path, configured=configured)
    assert found[0].source == SOURCE_CONFIGURED and found[0].recommended
    assert [item.path for item in found] == [tmp_path / "ai/ComfyUI", tmp_path / "ComfyUI"]


# --- From configuration ------------------------------------------------------


def test_discovery_uses_paths_from_existing_configuration(tmp_path: Path):
    home = tmp_path / "home"
    _orson(home)
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "examples" / "reference", config_dir)
    services = load_app_config(config_dir).services

    report = discover_installations(services, home=home, path_env="")
    # The reference config points at folders that do not exist on this "machine".
    assert report.llama_cpp[0].source == SOURCE_CONFIGURED and not report.llama_cpp[0].usable
    assert report.recommended(LLAMA_CPP).path == home / "ai/apps/llama.cpp/build/bin/llama-server"
    assert report.comfyui[0].source == SOURCE_CONFIGURED
    assert report.recommended(COMFYUI).path == home / "ai/ComfyUI"


def test_portable_path(tmp_path: Path):
    assert portable_path(tmp_path / "ComfyUI", tmp_path) == "~/ComfyUI"
    assert portable_path(tmp_path, tmp_path) == "~"
    assert portable_path(Path("/opt/ComfyUI"), tmp_path) == "/opt/ComfyUI"


def test_discover_command_output(tmp_path: Path):
    from ai_control_centre.cli import format_discovery

    _orson_mini(tmp_path)
    (tmp_path / "ComfyUI" / "venv" / "lib" / "python3.12" / "site-packages" / "torch").rmdir()
    report = discover_installations(home=tmp_path, path_env="")
    text = "\n".join(format_discovery(report, tmp_path))
    assert " * ~/ai/apps/llama.cpp/build/bin/llama-server  [OK, common location]" in text
    assert "   ~/ComfyUI  [PROBLEM, common location]" in text
    assert "No Python environment with torch was found" in text
    assert "nothing was started or changed" in text


def test_discover_command_runs_without_configuration(tmp_path: Path, capsys):
    from ai_control_centre.cli import main

    assert main(["--config-dir", str(tmp_path / "missing"), "discover"]) == 0
    out = capsys.readouterr().out
    assert "current configuration not loaded" in out
    assert "llama.cpp (llama-server)" in out and "ComfyUI" in out


def test_broken_configured_python_does_not_hide_working_environment(tmp_path: Path):
    """Seen on Orson Mini: config pointed at venv/bin/pythonX for ~/ComfyUI."""
    _orson_mini(tmp_path)
    configured = [(tmp_path / "ComfyUI", tmp_path / "ComfyUI/venv/bin/pythonX")]
    found = find_comfyui(home=tmp_path, configured=configured)
    assert [(item.source, item.usable) for item in found] == [(SOURCE_CONFIGURED, False), (SOURCE_COMMON, True)]
    assert found[1].recommended and found[1].executable == tmp_path / "ComfyUI/venv/bin/python"
