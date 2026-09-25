from __future__ import annotations

import subprocess
from pathlib import Path

from ai_control_centre.config import load_app_config


ROOT = Path(__file__).resolve().parents[1]
SEED_SCRIPT = ROOT / "installer" / "seed-config.sh"
DEFAULT_CONFIG = ROOT / "packaging" / "default-config"
CONFIG_FILES = ("settings.toml", "services.toml", "profiles.toml", "models.toml")


def _seed(config_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SEED_SCRIPT), str(DEFAULT_CONFIG), str(config_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_seed_config_creates_loadable_fresh_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    config_dir = tmp_path / "config"

    result = _seed(config_dir)

    for name in CONFIG_FILES:
        assert (config_dir / name).is_file()
    assert "Created 4 missing user configuration file(s)" in result.stdout

    config = load_app_config(config_dir)
    assert config.settings.setup_completed is False
    assert config.models == {}
    assert config.profiles["coding"].selected_model is None
    assert config.profiles["chat"].selected_model is None


def test_seed_config_preserves_existing_user_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    custom = config_dir / "settings.toml"
    custom.write_text("# user edited\n", encoding="utf-8")

    result = _seed(config_dir)

    assert custom.read_text(encoding="utf-8") == "# user edited\n"
    for name in CONFIG_FILES[1:]:
        assert (config_dir / name).is_file()
    assert f"Preserved {custom}" in result.stdout


def test_seed_config_reinstall_does_not_change_seeded_files(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _seed(config_dir)

    before = {name: (config_dir / name).read_bytes() for name in CONFIG_FILES}
    result = _seed(config_dir)
    after = {name: (config_dir / name).read_bytes() for name in CONFIG_FILES}

    assert after == before
    assert "Existing user configuration preserved." in result.stdout
