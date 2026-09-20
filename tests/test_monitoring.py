import pytest

from ai_control_centre.monitoring import (
    CpuCounters,
    calculate_cpu_percent,
    parse_nvidia_smi_line,
    parse_proc_meminfo,
    parse_proc_stat,
)


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


def test_parse_proc_meminfo_uses_available_memory():
    total, used = parse_proc_meminfo(
        "MemTotal:       65536 kB\nMemAvailable:   16384 kB\n"
    )
    assert total == 64
    assert used == 48


def test_parse_proc_meminfo_rejects_missing_values():
    with pytest.raises(ValueError):
        parse_proc_meminfo("MemFree: 123 kB\n")


def test_parse_proc_stat_and_calculate_cpu_percent():
    previous = parse_proc_stat("cpu  100 0 100 700 100 0 0 0\n")
    current = parse_proc_stat("cpu  150 0 150 750 100 0 0 0\n")
    assert previous == CpuCounters(idle=800, total=1000)
    assert calculate_cpu_percent(previous, current) == pytest.approx(100 * 100 / 150)


def test_parse_proc_stat_does_not_double_count_guest_time():
    counters = parse_proc_stat("cpu  100 10 20 700 30 5 5 10 40 2\n")
    assert counters.total == 880


def test_cpu_percent_rejects_non_forward_sample():
    sample = CpuCounters(idle=10, total=20)
    assert calculate_cpu_percent(sample, sample) is None
