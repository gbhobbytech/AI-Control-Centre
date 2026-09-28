from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish terminal process identity, then exec an interactive command."
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

    payload = {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
    }
    temp = state_file.with_suffix(state_file.suffix + ".tmp")
    temp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temp, state_file)

    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
