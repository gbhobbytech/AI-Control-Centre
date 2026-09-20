"""Per-model llama.cpp settings and read-only import of local launch arguments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .domain import ModelConfig, ProcessServiceConfig

# A starting point, not a claim that every model fits system RAM.
BASELINE = {
    'recommended_gpu_layers': 0,
    'recommended_context_length': 4096,
    'cache_type_k': 'f16',
    'cache_type_v': 'f16',
    'flash_attention': 'auto',
    'max_output_tokens': 2048,
    'startup_timeout_seconds': 180,
}
CURRENT_TUNING_SCHEMA = 1

CACHE_TYPES = ('f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl', 'bf16', 'f32')
FLAGS = {
    'recommended_gpu_layers': ('-ngl', '--gpu-layers', '--n-gpu-layers'),
    'recommended_context_length': ('--ctx-size', '-c'),
    'cache_type_k': ('--cache-type-k', '-ctk'),
    'cache_type_v': ('--cache-type-v', '-ctv'),
    'flash_attention': ('--flash-attn', '-fa'),
    'max_output_tokens': ('--n-predict', '--predict', '-n'),
}

@dataclass(frozen=True)
class TuningPreset:
    id: str
    display_name: str
    values: dict[str, object]
    summary: str


def _approx_gpu_layers(model_size_mib: float | None, gpu_total_mib: int | None, *, reserve_mib: int) -> int:
    """Return a cautious layer-count starting point from size/VRAM only.

    GGUF files do not expose a universal layer-count relationship here, so this is
    deliberately a rough starting heuristic. 999 is used only when the model file
    itself comfortably fits the usable VRAM budget.
    """
    if not model_size_mib or model_size_mib <= 0 or not gpu_total_mib or gpu_total_mib <= reserve_mib:
        return 0
    usable = max(0.0, float(gpu_total_mib - reserve_mib))
    fraction = min(1.0, usable / model_size_mib)
    if fraction >= 0.95:
        return 999
    # Forty is a deliberately conservative reference span rather than a claim
    # about the selected architecture's real transformer-layer count.
    return max(0, min(64, int(40 * fraction)))


def suggested_tuning_presets(
    *,
    model_size_bytes: int | None,
    gpu_total_mib: int | None,
    host_total_mib: float | None,
) -> dict[str, TuningPreset]:
    """Build practical, hardware-aware starting presets.

    These values are intentionally framed as starting points. They use only model
    file size and detected memory, not architecture-specific benchmarking.
    """
    model_size_mib = model_size_bytes / (1024 ** 2) if model_size_bytes else None
    balanced_layers = _approx_gpu_layers(model_size_mib, gpu_total_mib, reserve_mib=3072)
    max_layers = _approx_gpu_layers(model_size_mib, gpu_total_mib, reserve_mib=1536)
    if max_layers != 999 and balanced_layers != 999:
        max_layers = max(max_layers, balanced_layers)

    timeout = 180
    if model_size_mib and model_size_mib >= 16384:
        timeout = 240
    if model_size_mib and model_size_mib >= 32768:
        timeout = 300

    conservative = {
        **BASELINE,
        'startup_timeout_seconds': timeout,
    }
    balanced = {
        **BASELINE,
        'recommended_gpu_layers': balanced_layers,
        'recommended_context_length': 8192,
        'max_output_tokens': 2048,
        'cache_type_k': 'q8_0',
        'cache_type_v': 'q8_0',
        'flash_attention': 'auto',
        'startup_timeout_seconds': timeout,
    }
    maximum = {
        **BASELINE,
        'recommended_gpu_layers': max_layers,
        'recommended_context_length': 8192,
        'max_output_tokens': 4096,
        'cache_type_k': 'q8_0',
        'cache_type_v': 'q8_0',
        'flash_attention': 'auto',
        'startup_timeout_seconds': timeout,
    }

    if not gpu_total_mib:
        gpu_note = 'No supported GPU memory reading was detected, so this keeps model layers on the CPU.'
        balanced['recommended_gpu_layers'] = 0
        maximum['recommended_gpu_layers'] = 0
    elif model_size_mib and model_size_mib <= max(0, gpu_total_mib - 3072):
        gpu_note = 'The GGUF file fits comfortably inside the detected VRAM budget; full GPU offload is a reasonable starting test.'
    else:
        gpu_note = 'GPU-layer counts are approximate because architectures differ. Start here, confirm the model loads, then tune upward if desired.'

    ram_note = ''
    if model_size_mib and host_total_mib and model_size_mib > host_total_mib * 0.80:
        ram_note = ' The model file is close to the detected system-RAM capacity, so even CPU-first loading may fail.'

    return {
        'conservative': TuningPreset(
            'conservative', 'Conservative', conservative,
            'Best chance of a first successful launch. CPU-first, 4K context and compatibility-oriented caches.' + ram_note,
        ),
        'balanced': TuningPreset(
            'balanced', 'Balanced', balanced,
            'Uses detected VRAM for a cautious amount of GPU offload while keeping useful headroom. ' + gpu_note + ram_note,
        ),
        'maximum': TuningPreset(
            'maximum', 'Maximum GPU', maximum,
            'Prioritises GPU acceleration while retaining some VRAM headroom. It is a test starting point, not a fit guarantee. ' + gpu_note + ram_note,
        ),
    }



def is_llama_service(service: ProcessServiceConfig) -> bool:
    return 'llama-server' in service.executable.name or any(
        '{model.recommended_gpu_layers}' in arg or '{model.cache_type_k}' in arg
        for arg in service.args
    )


def model_values(model: ModelConfig) -> dict:
    return {key: getattr(model, key, None) if getattr(model, key, None) is not None else default
            for key, default in BASELINE.items()}


def validate_values(values: Mapping) -> dict:
    result = dict(values)
    labels = {'recommended_gpu_layers': 'GPU layers', 'recommended_context_length': 'Context tokens', 'max_output_tokens': 'Maximum reply tokens', 'startup_timeout_seconds': 'Startup timeout'}
    for key, minimum, maximum in (
        ('recommended_gpu_layers', 0, 999),
        ('recommended_context_length', 512, 2097152),
        ('max_output_tokens', 1, 2097152),
        ('startup_timeout_seconds', 1, 3600),
    ):
        value = result.get(key)
        if isinstance(value, bool):
            raise ValueError(f'{labels[key]} must be a whole number')
        try:
            number = int(str(value))
        except (ValueError, TypeError):
            raise ValueError(f'{labels[key]} must be a whole number') from None
        if not minimum <= number <= maximum:
            raise ValueError(f'{labels[key]} must be between {minimum} and {maximum}')
        result[key] = number
    for key in ('cache_type_k', 'cache_type_v'):
        if result.get(key) not in CACHE_TYPES:
            raise ValueError(f'Unsupported {key}: {result.get(key)}')
    if result.get('flash_attention') not in ('auto', 'on', 'off'):
        raise ValueError('Flash attention must be auto, on or off')
    if result['max_output_tokens'] > result['recommended_context_length']:
        raise ValueError('Output limit must not exceed the context length; leave room for your prompt.')
    if result['flash_attention'] == 'off' and result['cache_type_v'] not in ('f16', 'f32', 'bf16'):
        raise ValueError('Quantised V cache requires flash attention. Use auto/on or an unquantised V cache.')
    return {key: result[key] for key in BASELINE}


def replace_flag(args: Sequence[str], aliases: Sequence[str], value: str) -> tuple[str, ...]:
    """Remove all aliases and duplicates, including --flag=value; preserve unrelated args."""
    result = []
    index = 0
    while index < len(args):
        token = args[index]
        if token.split('=', 1)[0] in aliases:
            if '=' not in token and index + 1 < len(args):
                # Flash attention can be a bare flag in older launch commands.
                if not args[index + 1].startswith('-') or args[index + 1].lstrip('-').isdigit():
                    index += 1
        else:
            result.append(token)
        index += 1
    result.extend((aliases[0], str(value)))
    return tuple(result)


def apply_values(args: Sequence[str], model: ModelConfig) -> tuple[str, ...]:
    values = validate_values(model_values(model))
    result = tuple(args)
    for field, aliases in FLAGS.items():
        result = replace_flag(result, aliases, str(values[field]))
    return result


def parse_launch(args: Sequence[str]) -> tuple[str | None, dict[str, object]]:
    """Import only recognised fields; do not expose the full command or environment."""
    lookup = {alias: key for key, aliases in FLAGS.items() for alias in aliases}
    lookup.update({'-m': 'path', '--model': 'path'})
    values = {}
    index = 0
    while index < len(args):
        token = args[index]
        flag, equals, inline = token.partition('=')
        key = lookup.get(flag)
        if key:
            if equals:
                value = inline
            elif index + 1 < len(args) and (not args[index + 1].startswith('-') or args[index + 1].lstrip('-').isdigit()):
                index += 1
                value = args[index]
            elif key == 'flash_attention':
                value = 'on'
            else:
                index += 1
                continue
            values[key] = value
        index += 1
    path = values.pop('path', None)
    return path, values


@dataclass(frozen=True)
class RunningSettings:
    pid: int
    values: dict


def running_settings(model: ModelConfig, executable: Path) -> list[RunningSettings]:
    """Read same-user, exact-executable, exact-model matches. Never claim ownership."""
    from .runtime import proc_cmdline, proc_executable, proc_start_ticks
    candidates = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            pid = int(entry.name)
            ticks = proc_start_ticks(pid)
            if Path(proc_executable(pid)).resolve() != executable.resolve():
                continue
            path, values = parse_launch(proc_cmdline(pid)[1:])
            if not path:
                continue
            model_path = Path(path).expanduser()
            if not model_path.is_absolute():
                model_path = (entry / 'cwd').resolve() / model_path
            if model_path.resolve() != model.path.resolve() or proc_start_ticks(pid) != ticks:
                continue
            candidates.append(RunningSettings(pid, values))
        except (OSError, ValueError):
            continue
    return candidates
