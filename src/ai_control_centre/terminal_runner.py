from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an interactive command and publish its process identity."
    )
    parser.add_argument("--state-file", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)

    command = list(ns.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    state_file = Path(ns.state_file)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_file.unlink()
    except FileNotFoundError:
        pass

    child = subprocess.Popen(
        command,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
        close_fds=True,
    )

    payload = {
        "pid": child.pid,
        "pgid": os.getpgid(child.pid),
        "sid": os.getsid(child.pid),
    }
    temp = state_file.with_suffix(state_file.suffix + ".tmp")
    temp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temp, state_file)

    try:
        return int(child.wait())
    finally:
        try:
            state_file.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
