import pytest

from ai_control_centre.monitoring import parse_nvidia_smi_line


def test_parse_nvidia_smi_line():
    status = parse_nvidia_smi_line("NVIDIA RTX A2000 12GB, 12282, 1234, 24, 60")
    assert status.available is True
    assert status.name == "NVIDIA RTX A2000 12GB"
    assert status.memory_total_mib == 12282
    assert status.memory_used_mib == 1234
    assert status.memory_free_mib == 11048
    assert status.utilization_percent == 24
    assert status.temperature_c == 60


def test_parse_nvidia_smi_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_nvidia_smi_line("bad output")
