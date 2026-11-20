"""End-to-end ComfyUI tests: discovery -> preset -> Image profile -> start/stop.

A small stand-in for ComfyUI's main.py answers /system_stats the way
ComfyUI 0.46 does, so the real service code paths are exercised.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_control_centre.config import load_app_config
from ai_control_centre.discovery import find_comfyui
from ai_control_centre.domain import ServiceState
from ai_control_centre.health import check_http
from ai_control_centre.manager import ServiceManager
from ai_control_centre.preferences import configure_image_environment, write_toml_atomic

ROOT = Path(__file__).resolve().parents[1]

FAKE_MAIN = '''
import argparse, http.server, json
parser = argparse.ArgumentParser()
parser.add_argument("--listen", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8188)
args, _ = parser.parse_known_args()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/system_stats":
            body = json.dumps({"system": {"os": "linux", "comfyui_version": "0.46.0"}}).encode()
            self.send_response(200)
        else:
            body = b"ComfyUI"
            self.send_response(200 if self.path == "/" else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

http.server.ThreadingHTTPServer((args.listen, args.port), Handler).serve_forever()
'''


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _install_fake_comfyui(home: Path) -> Path:
    root = home / "ComfyUI"
    (root / "comfy").mkdir(parents=True)
    (root / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "lib" / "python3.12" / "site-packages" / "torch").mkdir(parents=True)
    return root


@pytest.fixture
def image_setup(tmp_path: Path):
    """A fresh starter configuration with the Image environment configured by discovery."""
    home = tmp_path / "home"
    _install_fake_comfyui(home)
    config_dir = tmp_path / "config"
    shutil.copytree(ROOT / "packaging" / "default-config", config_dir)
    settings = config_dir / "settings.toml"
    with settings.open("rb") as fh:
        data = tomllib.load(fh)
    data["paths"] = {"log_dir": str(tmp_path / "logs"), "runtime_dir": str(tmp_path / "runtime")}
    write_toml_atomic(settings, data)

    [installation] = find_comfyui(home=home)
    port = _free_port()
    # home=None keeps absolute paths, as the test home is not the real ~.
    configure_image_environment(config_dir, installation, port=port)
    config = load_app_config(config_dir)
    manager = ServiceManager(config)
    yield config, manager, port
    try:
        manager.stop_profile("image")
    except RuntimeError:
        pass


def test_image_profile_starts_comfyui_on_configured_port(image_setup):
    config, manager, port = image_setup
    service = config.services["comfyui"]
    assert service.args == ("main.py", "--listen", "127.0.0.1", "--port", str(port))

    with patch("ai_control_centre.process_service.webbrowser.open", return_value=True):
        results = manager.start_profile("image")

    assert results == [("comfyui", results[0][1])]
    assert results[0][1].state == ServiceState.READY
    assert check_http(service.health).ok
    assert manager.runtime_store.load_process("comfyui") is not None


def test_browser_opens_only_after_comfyui_is_ready(image_setup):
    config, manager, port = image_setup
    seen: list[tuple[str, bool]] = []

    def fake_open(url: str) -> bool:
        seen.append((url, check_http(config.services["comfyui"].health).ok))
        return True

    with patch("ai_control_centre.process_service.webbrowser.open", side_effect=fake_open):
        manager.start_profile("image")

    assert seen == [(f"http://127.0.0.1:{port}", True)]


def test_launcher_owned_comfyui_stops_cleanly(image_setup):
    config, manager, _ = image_setup
    with patch("ai_control_centre.process_service.webbrowser.open", return_value=True):
        manager.start_profile("image")
    pid = manager.runtime_store.load_process("comfyui").pid

    [(service_id, status)] = manager.stop_profile("image")

    assert (service_id, status.state) == ("comfyui", ServiceState.STOPPED)
    assert manager.get("comfyui").status().state == ServiceState.STOPPED
    assert manager.runtime_store.load_process("comfyui") is None
    assert not Path(f"/proc/{pid}").exists() or _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
    except OSError:
        return True


def test_externally_started_comfyui_is_reused_and_never_stopped(image_setup):
    config, manager, _ = image_setup
    service = config.services["comfyui"]
    external = subprocess.Popen(
        [str(service.executable), *service.args],
        cwd=service.cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not check_http(service.health).ok:
            assert time.monotonic() < deadline, "fake ComfyUI did not start"
            time.sleep(0.05)
        assert manager.get("comfyui").status().state == ServiceState.EXTERNAL

        with patch("ai_control_centre.process_service.webbrowser.open", return_value=True) as opened:
            [(_, started)] = manager.start_profile("image")
        assert started.state == ServiceState.EXTERNAL
        assert opened.called  # the existing interface is still offered
        assert manager.runtime_store.load_process("comfyui") is None

        [(_, stopped)] = manager.stop_profile("image")
        assert stopped.state == ServiceState.EXTERNAL
        manager.stop_all()
        with pytest.raises(RuntimeError, match="externally started; refusing to stop"):
            manager.get("comfyui").stop()
        assert external.poll() is None
        assert check_http(service.health).ok
    finally:
        external.terminate()
        external.wait(timeout=10)


def test_port_taken_by_another_program_does_not_start_comfyui(image_setup):
    config, manager, port = image_setup
    with socket.socket() as other:
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        other.bind(("127.0.0.1", port))
        other.listen()
        with patch("ai_control_centre.process_service.webbrowser.open", return_value=True) as opened:
            with pytest.raises(RuntimeError, match=rf"Port {port} on 127\.0\.0\.1 is already in use, so ComfyUI was not started"):
                manager.start_profile("image")
        assert not opened.called
        assert manager.runtime_store.load_process("comfyui") is None
        assert manager.get("comfyui").status().state == ServiceState.STOPPED
