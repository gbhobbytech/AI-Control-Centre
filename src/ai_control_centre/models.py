from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Mapping

from .domain import ModelConfig

_SPLIT_RE = re.compile(
    r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)
_QUANT_RE = re.compile(r"(?i)(?:^|[-_.])((?:Q|IQ)\d(?:_[A-Z0-9]+)+|Q\d_[A-Z0-9]+|F16|BF16)(?:$|[-_.])")
_MODEL_TOKEN_RE = re.compile(r"\{model\.([a-zA-Z0-9_]+)\}")


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return value or "model"


def _unique_id(base: str, path: Path, used: set[str]) -> str:
    candidate = _slug(base)
    if candidate not in used:
        used.add(candidate)
        return candidate
    suffix = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    candidate = f"{candidate}_{suffix}"
    used.add(candidate)
    return candidate


def _infer_quant(name: str) -> str | None:
    match = _QUANT_RE.search(name)
    return match.group(1).upper() if match else None


def discover_models(roots: Iterable[Path]) -> dict[str, ModelConfig]:
    singles: list[Path] = []
    split_groups: dict[tuple[Path, str, int], dict[int, Path]] = {}

    for root in roots:
        root = root.expanduser()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.gguf")):
            if not path.is_file():
                continue
            match = _SPLIT_RE.match(path.name)
            if match:
                total = int(match.group("total"))
                index = int(match.group("index"))
                key = (path.parent, match.group("base"), total)
                split_groups.setdefault(key, {})[index] = path
            else:
                singles.append(path)

    models: dict[str, ModelConfig] = {}
    used: set[str] = set()

    for path in sorted(singles):
        model_id = _unique_id(path.stem, path, used)
        models[model_id] = ModelConfig(
            id=model_id,
            display_name=path.stem,
            path=path,
            shard_paths=(path,),
            expected_shards=1,
            complete=True,
            quant=_infer_quant(path.stem),
        )

    for (parent, base, total), indexed in sorted(
        split_groups.items(), key=lambda item: str(item[0][0] / item[0][1])
    ):
        expected = set(range(1, total + 1))
        present = set(indexed)
        missing = sorted(expected - present)
        shard_paths = tuple(indexed[index] for index in sorted(indexed))
        launch_path = indexed.get(1, shard_paths[0])
        complete = not missing and 1 in indexed
        warning = None
        if missing:
            preview = ", ".join(f"{index:05d}" for index in missing[:8])
            if len(missing) > 8:
                preview += ", ..."
            warning = f"missing shard(s): {preview} of {total:05d}"
        elif 1 not in indexed:
            warning = "first shard is missing"

        model_id = _unique_id(base, launch_path, used)
        models[model_id] = ModelConfig(
            id=model_id,
            display_name=base,
            path=launch_path,
            shard_paths=shard_paths,
            expected_shards=total,
            complete=complete,
            warning=warning,
            quant=_infer_quant(base),
        )

    return models


def model_field_value(model: ModelConfig, field: str, defaults: Mapping[str, str]) -> str:
    if field == "path":
        return str(model.path)
    if field == "id":
        return model.id
    if field == "display_name":
        return model.display_name

    if not hasattr(model, field):
        raise ValueError(f"Unknown model template field: {field}")

    value = getattr(model, field)
    if value is None:
        if field in defaults:
            return str(defaults[field])
        raise ValueError(
            f"Model '{model.id}' has no value for '{field}' and the service has no model default"
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_model_args(
    args: tuple[str, ...],
    model: ModelConfig,
    defaults: Mapping[str, str],
) -> tuple[str, ...]:
    if not model.complete:
        raise ValueError(f"Model '{model.id}' is incomplete: {model.warning or 'missing files'}")

    def replace_token(match: re.Match[str]) -> str:
        return model_field_value(model, match.group(1), defaults)

    return tuple(_MODEL_TOKEN_RE.sub(replace_token, arg) for arg in args)


def apply_model_overrides(base: ModelConfig, **overrides) -> ModelConfig:
    values = {key: value for key, value in overrides.items() if value is not None}
    return replace(base, **values)
