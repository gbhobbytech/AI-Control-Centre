import os
import sys
import time
import subprocess

from ai_control_centre.runtime import (
    ProcessRecord,
    command_hash,
    proc_cmdline,
    proc_executable,
    proc_start_ticks,
    record_matches_process,
)


def test_process_identity_matches_live_process():
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        time.sleep(0.05)
        actual_cmdline = proc_cmdline(process.pid)
        record = ProcessRecord(
            service_id="test",
            pid=process.pid,
            pgid=os.getpgid(process.pid),
            sid=os.getsid(process.pid),
            proc_start_ticks=proc_start_ticks(process.pid),
            executable=proc_executable(process.pid),
            command_sha256=command_hash(actual_cmdline),
        )
        assert record.pid == record.pgid == record.sid
        assert record_matches_process(record)
    finally:
        process.terminate()
        process.wait(timeout=2)
