from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_control_centre.docker_service import DockerService
from ai_control_centre.domain import DockerServiceConfig, ServiceState
from ai_control_centre.runtime import RuntimeStore


def _fake_docker(tmp_path: Path, *, running: bool = False) -> Path:
    state_path = tmp_path / "docker-state.json"
    state_path.write_text(
        json.dumps(
            {
                "running": running,
                "status": "running" if running else "exited",
                "started_at": "2026-09-20T05:00:00Z" if running else "0001-01-01T00:00:00Z",
            }
        )
    )
    script = tmp_path / "docker"
    script.write_text(
        f'''#!/usr/bin/env python3
import json, sys
from pathlib import Path
p = Path({str(state_path)!r})
s = json.loads(p.read_text())
args = sys.argv[1:]
if args[0] == "inspect":
    print(json.dumps([{{
        "Id": "container-123",
        "Name": "/cptr-dmz",
        "State": {{
            "Running": s["running"],
            "Status": s["status"],
            "StartedAt": s["started_at"],
        }}
    }}]))
elif args[0] == "start":
    s["running"] = True
    s["status"] = "running"
    s["started_at"] = "2026-09-20T05:30:00Z"
    p.write_text(json.dumps(s))
    print("cptr-dmz")
elif args[0] == "stop":
    s["running"] = False
    s["status"] = "exited"
    p.write_text(json.dumps(s))
    print("cptr-dmz")
elif args[0] == "logs":
    print("fake docker log")
else:
    print("unsupported", file=sys.stderr)
    sys.exit(2)
'''
    )
    script.chmod(0o755)
    return script


def _service(tmp_path: Path, *, running: bool = False) -> DockerService:
    docker = _fake_docker(tmp_path, running=running)
    config = DockerServiceConfig(
        id="computer_dmz",
        display_name="Computer DMZ",
        container_name="cptr-dmz",
        docker_executable=str(docker),
        stop_timeout=1,
    )
    return DockerService(config, RuntimeStore(tmp_path / "runtime"))


def test_docker_start_persists_ownership_and_stop(tmp_path: Path):
    service = _service(tmp_path)
    assert service.status().state == ServiceState.STOPPED

    started = service.start()
    assert started.state == ServiceState.RUNNING_NOT_READY
    assert service.status().state == ServiceState.RUNNING_NOT_READY

    record = service.runtime_store.load_docker("computer_dmz")
    assert record is not None
    assert record.container_id == "container-123"
    assert record.started_at == "2026-09-20T05:30:00Z"

    stopped = service.stop()
    assert stopped.state == ServiceState.STOPPED
    assert service.runtime_store.load_docker("computer_dmz") is None


def test_running_unowned_container_is_external_and_not_stopped(tmp_path: Path):
    service = _service(tmp_path, running=True)
    assert service.status().state == ServiceState.EXTERNAL
    with pytest.raises(RuntimeError, match="externally started"):
        service.stop()
