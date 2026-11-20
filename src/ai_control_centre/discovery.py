"""Find llama.cpp and ComfyUI installations on this machine.

Discovery only inspects files; it never runs the programs it finds. It
checks the paths already in the user's configuration, then ``PATH`` (for
llama.cpp) and a short list of conventional folders under the home folder.
Inside each folder it looks for known layouts rather than searching to an
arbitrary depth, so it is fast and predictable.

This finds *installations*. It does not look at models: GGUF models are
found by models.discover_models(), and ComfyUI manages its own models.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .domain import ProcessServiceConfig, ServiceConfig

LLAMA_CPP = "llama-cpp"
COMFYUI = "comfyui"

SOURCE_CONFIGURED = "current configuration"
SOURCE_PATH = "PATH"
SOURCE_COMMON = "common location"
SOURCE_USER = "selected by you"

# Conventional folders, relative to the home folder unless absolute.
LLAMA_CPP_ROOTS: tuple[str, ...] = (
    "llama.cpp",
    "ai/llama.cpp",
    "ai/apps/llama.cpp",
    "apps/llama.cpp",
    "src/llama.cpp",
    "git/llama.cpp",
    "projects/llama.cpp",
    "/opt/llama.cpp",
)
# Where llama-server sits inside a llama.cpp folder.
LLAMA_SERVER_LAYOUTS: tuple[str, ...] = (
    "build/bin/llama-server",
    "bin/llama-server",
    "llama-server",
)

COMFYUI_ROOTS: tuple[str, ...] = (
    "ComfyUI",
    "comfyui",
    "ai/ComfyUI",
    "ai/comfyui",
    "ai/apps/ComfyUI",
    "apps/ComfyUI",
    "comfy/ComfyUI",
    "Documents/ComfyUI",
    "/opt/ComfyUI",
)
# Virtual environment folders, relative to the ComfyUI folder, in preference order.
COMFYUI_VENV_LAYOUTS: tuple[str, ...] = ("venv", ".venv", "../venv", "../.venv")


@dataclass(frozen=True)
class Installation:
    """One candidate installation and whether it can be used."""

    kind: str
    # llama.cpp: the llama-server executable. ComfyUI: the installation folder.
    path: Path
    # The program the service runs: llama-server, or ComfyUI's Python.
    executable: Path | None
    cwd: Path
    source: str
    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    recommended: bool = False

    @property
    def usable(self) -> bool:
        return not self.problems

    def preset_values(self, home: Path | None = None) -> dict[str, str]:
        """Values for the matching service preset, with ~ for the home folder."""
        if not self.usable or self.executable is None:
            raise ValueError(f"{self.path} cannot be used: {'; '.join(self.problems)}")
        if self.kind == LLAMA_CPP:
            return {
                "executable": portable_path(self.executable, home),
                "cwd": portable_path(self.cwd, home),
            }
        return {
            "install_dir": portable_path(self.path, home),
            "python": portable_path(self.executable, home),
        }


def _home(home: Path | None) -> Path:
    return home if home is not None else Path.home()


def portable_path(path: Path, home: Path | None = None) -> str:
    """Write a path under the home folder as ~/..., so configs move between machines."""
    home_dir = _home(home)
    try:
        relative = path.relative_to(home_dir)
    except ValueError:
        return str(path)
    return "~" if str(relative) == "." else f"~/{relative}"


def _expand(path: Path | str, home: Path | None) -> Path:
    text = str(path)
    if text == "~" or text.startswith("~/"):
        return _home(home) / text[2:] if text != "~" else _home(home)
    return Path(text)


def _candidate_root(entry: str, home: Path | None) -> Path:
    return Path(entry) if entry.startswith("/") else _home(home) / entry


def _is_program(path: Path) -> bool:
    """An ELF binary or a script with a #! line."""
    try:
        with path.open("rb") as fh:
            header = fh.read(4)
    except OSError:
        return False
    return header == b"\x7fELF" or header.startswith(b"#!")


def _same_place(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _dedupe(installations: Iterable[Installation]) -> list[Installation]:
    unique: list[Installation] = []
    for item in installations:
        if any(_same_place(item.path, seen.path) for seen in unique):
            continue
        unique.append(item)
    return unique


def _mark_recommended(installations: list[Installation]) -> list[Installation]:
    for index, item in enumerate(installations):
        if item.usable:
            installations[index] = Installation(**{**item.__dict__, "recommended": True})
            break
    return installations


# --- llama.cpp -----------------------------------------------------------------


def validate_llama_executable(path: Path | str, source: str = SOURCE_USER, home: Path | None = None) -> Installation:
    """Check a llama-server path and explain anything that would stop it working."""
    executable = _expand(path, home)
    problems: list[str] = []
    notes: list[str] = []
    if not executable.exists():
        problems.append(f"{executable} does not exist")
    elif executable.is_dir():
        problems.append(
            f"{executable} is a folder; choose the llama-server program inside it "
            "(usually build/bin/llama-server)"
        )
    elif not os.access(executable, os.X_OK):
        problems.append(f"{executable} is not executable (check its permissions)")
    elif not _is_program(executable):
        problems.append(f"{executable} is not a program")
    elif not executable.name.startswith("llama-server"):
        notes.append(
            f"The file is called {executable.name}, not llama-server; check this is the llama.cpp server"
        )

    cwd = executable.parent
    if cwd.name == "bin" and cwd.parent.name == "build":
        cwd = cwd.parent.parent
    return Installation(
        kind=LLAMA_CPP,
        path=executable,
        executable=executable,
        cwd=cwd,
        source=source,
        problems=tuple(problems),
        notes=tuple(notes),
    )


def find_llama_cpp(
    *,
    home: Path | None = None,
    configured: Iterable[Path] = (),
    path_env: str | None = None,
) -> list[Installation]:
    """llama-server candidates: configured paths, then PATH, then common folders.

    Missing configured paths are reported (so the user can see why their
    current setup fails); other candidates are only listed if they exist.
    """
    found: list[Installation] = [
        validate_llama_executable(path, SOURCE_CONFIGURED, home) for path in configured
    ]

    search_path = os.environ.get("PATH", "") if path_env is None else path_env
    for folder in search_path.split(os.pathsep):
        if folder:
            candidate = Path(folder) / "llama-server"
            if candidate.is_file():
                found.append(validate_llama_executable(candidate, SOURCE_PATH, home))

    for entry in LLAMA_CPP_ROOTS:
        root = _candidate_root(entry, home)
        if not root.is_dir():
            continue
        for layout in LLAMA_SERVER_LAYOUTS:
            candidate = root / layout
            if candidate.is_file():
                found.append(validate_llama_executable(candidate, SOURCE_COMMON, home))
                break

    return _mark_recommended(_dedupe(found))


# --- ComfyUI -------------------------------------------------------------------


def _env_root(python: Path) -> Path:
    """The environment folder for a Python inside .../bin/."""
    return python.parent.parent if python.parent.name == "bin" else python.parent


def environment_has_torch(python: Path) -> bool:
    """Whether the Python's environment contains the torch package (file check only)."""
    root = _env_root(python)
    patterns = (
        "lib*/python3*/site-packages/torch",
        "lib*/python3*/dist-packages/torch",
    )
    return any(
        Path(match).is_dir()
        for pattern in patterns
        for match in glob.glob(str(root / pattern))
    )


def _venv_python(venv: Path) -> Path | None:
    for name in ("python", "python3"):
        candidate = venv / "bin" / name
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_comfyui_folder(path: Path) -> bool:
    return (path / "main.py").is_file() and (path / "comfy").is_dir()


def validate_comfyui(
    path: Path | str,
    python: Path | str | None = None,
    source: str = SOURCE_USER,
    home: Path | None = None,
) -> Installation:
    """Check a ComfyUI folder and choose (or check) the Python that runs it."""
    folder = _expand(path, home)
    problems: list[str] = []
    notes: list[str] = []
    chosen: Path | None = None

    if not folder.exists():
        problems.append(f"{folder} does not exist")
    elif not folder.is_dir():
        problems.append(f"{folder} is not a folder; choose the ComfyUI folder that contains main.py")
    elif not is_comfyui_folder(folder):
        problems.append(f"{folder} is not a ComfyUI folder (main.py and the comfy folder were not found)")

    if python is not None:
        chosen = _expand(python, home)
        if not chosen.exists():
            problems.append(f"Python {chosen} does not exist")
        elif chosen.is_dir() or not os.access(chosen, os.X_OK):
            problems.append(f"{chosen} is not an executable Python interpreter")
        elif not environment_has_torch(chosen):
            # Not fatal for a Python the user chose: some layouts (conda,
            # user-site installs) keep packages where this check can't see.
            notes.append(f"Could not confirm that torch is installed for {chosen}")
    elif not problems:
        without_torch: list[Path] = []
        for layout in COMFYUI_VENV_LAYOUTS:
            # normpath, not resolve(): a venv's bin/python is usually a symlink
            # that must stay as written.
            candidate = _venv_python(Path(os.path.normpath(folder / layout)))
            if candidate is None:
                continue
            if environment_has_torch(candidate):
                chosen = candidate
                break
            without_torch.append(candidate)
        if chosen is None:
            if without_torch:
                checked = ", ".join(str(item) for item in without_torch)
                problems.append(
                    f"No Python environment with torch was found for ComfyUI (checked {checked}); "
                    "choose the Python that runs ComfyUI"
                )
            else:
                problems.append(
                    "No Python environment was found in venv or .venv; choose the Python that runs ComfyUI"
                )
        elif without_torch:
            notes.append(
                "Ignored environment(s) without torch: " + ", ".join(str(item) for item in without_torch)
            )

    return Installation(
        kind=COMFYUI,
        path=folder,
        executable=chosen,
        cwd=folder,
        source=source,
        problems=tuple(problems),
        notes=tuple(notes),
    )


def find_comfyui(
    *,
    home: Path | None = None,
    configured: Iterable[tuple[Path, Path | None]] = (),
) -> list[Installation]:
    """ComfyUI candidates: configured (folder, python) pairs, then common folders."""
    found: list[Installation] = [
        validate_comfyui(folder, python, SOURCE_CONFIGURED, home) for folder, python in configured
    ]
    for entry in COMFYUI_ROOTS:
        root = _candidate_root(entry, home)
        if is_comfyui_folder(root):
            found.append(validate_comfyui(root, source=SOURCE_COMMON, home=home))
    return _mark_recommended(_dedupe(found))


# --- From the current configuration ------------------------------------------


def configured_llama_executables(services: Mapping[str, ServiceConfig]) -> list[Path]:
    return [
        service.executable
        for service in services.values()
        if isinstance(service, ProcessServiceConfig) and service.executable.name.startswith("llama-server")
    ]


def configured_comfyui(services: Mapping[str, ServiceConfig]) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for service in services.values():
        if isinstance(service, ProcessServiceConfig) and any(
            arg == "main.py" or arg.endswith("/main.py") for arg in service.args
        ):
            pairs.append((service.cwd, service.executable))
    return pairs


@dataclass(frozen=True)
class DiscoveryReport:
    llama_cpp: tuple[Installation, ...]
    comfyui: tuple[Installation, ...]

    def recommended(self, kind: str) -> Installation | None:
        items = self.llama_cpp if kind == LLAMA_CPP else self.comfyui
        return next((item for item in items if item.recommended), None)


def discover_installations(
    services: Mapping[str, ServiceConfig] | None = None,
    *,
    home: Path | None = None,
    path_env: str | None = None,
) -> DiscoveryReport:
    services = services or {}
    return DiscoveryReport(
        llama_cpp=tuple(
            find_llama_cpp(home=home, configured=configured_llama_executables(services), path_env=path_env)
        ),
        comfyui=tuple(find_comfyui(home=home, configured=configured_comfyui(services))),
    )
