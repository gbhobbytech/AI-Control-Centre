from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from ai_control_centre.config import AppConfig, Settings, load_models, load_settings
from ai_control_centre.domain import HealthCheckConfig, ModelConfig, ProcessServiceConfig, ProfileConfig, ServiceState
from ai_control_centre.manager import ServiceManager
from ai_control_centre.preferences import save_model_tuning, save_appearance, write_toml_atomic
from ai_control_centre.runtime import ProcessRecord, command_hash
from ai_control_centre.service_base import ServiceStatus
from ai_control_centre.theme import DEFAULT_APPEARANCE, contrast_text
from ai_control_centre.tuning import BASELINE, apply_values, model_values, parse_launch, validate_values


def test_new_model_has_hardware_neutral_starting_values():
    model = ModelConfig(id='new', display_name='New', path=Path('/models/new.gguf'))
    assert model_values(model) == BASELINE
    assert not model.tuning_reviewed


def test_saved_tuning_preserves_metadata_and_unrelated_models(tmp_path):
    model_file = tmp_path / 'model.gguf'
    model_file.touch()
    model = ModelConfig(id='model.with.dots', display_name='Model', path=model_file)
    write_toml_atomic(tmp_path / 'models.toml', {'models': {model.id: {'path': str(model_file), 'notes': 'keep me'}, 'other': {'path': '/other.gguf'}}})
    values = {**BASELINE, 'recommended_gpu_layers': 14, 'recommended_context_length': 8192}
    save_model_tuning(tmp_path, model, values, 'manual')
    loaded = load_models(tmp_path / 'models.toml', (tmp_path,))
    assert loaded[model.id].recommended_gpu_layers == 14
    assert loaded[model.id].notes == 'keep me'
    assert loaded[model.id].tuning_reviewed
    assert 'other' in loaded
    assert model_values(loaded[model.id]) == values


@pytest.mark.parametrize('changes', [
    {'recommended_gpu_layers': -1}, {'recommended_gpu_layers': 'auto'},
    {'recommended_context_length': 0}, {'startup_timeout_seconds': 'nan'},
    {'startup_timeout_seconds': 0}, {'cache_type_k': 'made_up'},
    {'max_output_tokens': 8192}, {'recommended_gpu_layers': True},
    {'flash_attention': 'off', 'cache_type_v': 'q8_0'},
])
def test_invalid_tuning_is_rejected(changes):
    with pytest.raises(ValueError):
        validate_values({**BASELINE, **changes})


def test_override_removes_all_aliases_but_preserves_unrelated_options():
    model = ModelConfig(id='m', display_name='M', path=Path('m.gguf'), **BASELINE)
    args = ('--gpu-layers=22', '-ngl', '30', '-c', '65536', '-fa', '--port', '8080', '--api-key', 'secret')
    rendered = apply_values(args, model)
    assert '--gpu-layers=22' not in rendered
    assert rendered.count('-ngl') == 1
    assert rendered[rendered.index('-ngl')+1] == '0'
    assert rendered[rendered.index('--ctx-size')+1] == '4096'
    assert ('--port', '8080', '--api-key', 'secret') == rendered[:4]


def test_import_only_supported_explicit_values():
    path, values = parse_launch(('--model=/models/a b.gguf', '--gpu-layers', '14', '-c', '8192', '-fa', '--api-key', 'do-not-import', '--port=8080'))
    assert path == '/models/a b.gguf'
    assert values == {'recommended_gpu_layers': '14', 'recommended_context_length': '8192', 'flash_attention': 'on'}
    assert 'do-not-import' not in str(values)


def _manager(tmp_path, model):
    service = ProcessServiceConfig(id='llama', display_name='llama', executable=Path('/tmp/llama-server'),
                                   cwd=tmp_path, default_model=model.id,
                                   args=('-m', '{model.path}', '-ngl', '{model.recommended_gpu_layers}', '--ctx-size', '65536'),
                                   model_defaults={'recommended_gpu_layers': '22'},
                                   health=HealthCheckConfig(url='http://127.0.0.1:9/health'))
    return ServiceManager(AppConfig(settings=Settings(log_dir=tmp_path/'logs', runtime_dir=tmp_path/'runtime'),
                                   services={'llama': service}, profiles={'chat': ProfileConfig(id='chat', display_name='Chat', services=('llama',), selected_model=model.id)},
                                   models={model.id: model}))


def test_unreviewed_model_rejected_before_conflicts_are_stopped(tmp_path, monkeypatch):
    manager = _manager(tmp_path, ModelConfig(id='m', display_name='M', path=tmp_path/'m.gguf'))
    monkeypatch.setattr(manager, '_prepare_conflicts', lambda *a, **kw: pytest.fail('conflict mutation happened before review'))
    with pytest.raises(RuntimeError, match='Review launch settings'):
        manager.start_profile('chat', stop_conflicts=True)


def test_reviewed_model_overrides_service_defaults_and_timeout(tmp_path):
    model = ModelConfig(id='m', display_name='M', path=tmp_path/'m.gguf', tuning_reviewed=True, **BASELINE)
    manager = _manager(tmp_path, model)
    launch = manager._controller_for_start('llama', model.id)
    assert launch.config.args[launch.config.args.index('-ngl')+1] == '0'
    assert launch.config.args[launch.config.args.index('--ctx-size')+1] == '4096'
    assert launch.config.health.startup_timeout == 180
    assert manager.config.services['llama'].args[-1] == '65536'


def test_same_model_changed_settings_requires_restart(tmp_path, monkeypatch):
    model = ModelConfig(id='m', display_name='M', path=tmp_path/'m.gguf', tuning_reviewed=True, **BASELINE)
    manager = _manager(tmp_path, model)
    launch = manager._controller_for_start('llama', model.id)
    record = ProcessRecord('llama', 123, 123, 123, 1, '/tmp/llama-server', command_hash(launch.config.command), model_id=model.id)
    monkeypatch.setattr(manager.runtime_store, 'load_process', lambda sid: record)
    monkeypatch.setattr(manager.get('llama'), 'status', lambda: ServiceStatus(ServiceState.READY, 'ready', 123))
    assert manager._prepare_model_switch(['llama'], 'm', replace_model=False) == []
    manager.config.models['m'] = replace(model, recommended_gpu_layers=8)
    with pytest.raises(RuntimeError, match='--replace-model'):
        manager._prepare_model_switch(['llama'], 'm', replace_model=False)


def test_appearance_round_trip_preserves_other_settings(tmp_path):
    write_toml_atomic(tmp_path/'settings.toml', {'paths': {'log_dir': '/logs', 'runtime_dir': '/runtime'}, 'custom': {'leave': 'alone'}})
    values = {**DEFAULT_APPEARANCE, 'mode': 'dark', 'accent': '#AA6644', 'font_size': 12}
    save_appearance(tmp_path, values)
    assert load_settings(tmp_path/'settings.toml').appearance == values
    assert tomllib.loads((tmp_path/'settings.toml').read_text())['custom']['leave'] == 'alone'
    before = (tmp_path/'settings.toml').read_bytes()
    with pytest.raises(ValueError):
        save_appearance(tmp_path, {**values, 'primary': 'bad colour'})
    assert (tmp_path/'settings.toml').read_bytes() == before
    assert contrast_text('#FFFFFF') == '#111111'
    assert contrast_text('#000000') == '#FFFFFF'


def test_ready_reckoner_small_model_can_use_full_gpu_offload():
    from ai_control_centre.tuning import suggested_tuning_presets

    presets = suggested_tuning_presets(
        model_size_bytes=3 * 1024**3,
        gpu_total_mib=12288,
        host_total_mib=64 * 1024,
    )
    assert presets['conservative'].values['recommended_gpu_layers'] == 0
    assert presets['balanced'].values['recommended_gpu_layers'] == 999
    assert presets['maximum'].values['recommended_gpu_layers'] == 999
    for preset in presets.values():
        validate_values(preset.values)


def test_ready_reckoner_large_model_uses_cautious_partial_offload():
    from ai_control_centre.tuning import suggested_tuning_presets

    presets = suggested_tuning_presets(
        model_size_bytes=18 * 1024**3,
        gpu_total_mib=12288,
        host_total_mib=64 * 1024,
    )
    balanced = presets['balanced'].values['recommended_gpu_layers']
    maximum = presets['maximum'].values['recommended_gpu_layers']
    assert 0 < balanced < 999
    assert balanced <= maximum < 999
    assert presets['balanced'].values['cache_type_k'] == 'q8_0'
    assert presets['balanced'].values['recommended_context_length'] == 8192


def test_ready_reckoner_without_gpu_stays_cpu_first():
    from ai_control_centre.tuning import suggested_tuning_presets

    presets = suggested_tuning_presets(
        model_size_bytes=8 * 1024**3,
        gpu_total_mib=None,
        host_total_mib=32 * 1024,
    )
    assert presets['balanced'].values['recommended_gpu_layers'] == 0
    assert presets['maximum'].values['recommended_gpu_layers'] == 0
    assert 'No supported GPU' in presets['balanced'].summary


def test_ready_reckoner_warns_when_model_is_close_to_system_ram():
    from ai_control_centre.tuning import suggested_tuning_presets

    presets = suggested_tuning_presets(
        model_size_bytes=14 * 1024**3,
        gpu_total_mib=8192,
        host_total_mib=16 * 1024,
    )
    assert 'close to the detected system-RAM capacity' in presets['conservative'].summary
