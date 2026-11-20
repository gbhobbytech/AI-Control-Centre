"""Shared pytest setup.

Some tests launch child Python processes (for example the terminal runner)
with a working directory outside the repository. A relative PYTHONPATH such
as ``src`` does not resolve from there, so expose an absolute source path to
both this process and any children it starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_entries = [_SRC]
for _part in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if _part:
        _absolute = str(Path(_part).resolve())
        if _absolute not in _entries:
            _entries.append(_absolute)
os.environ["PYTHONPATH"] = os.pathsep.join(_entries)
