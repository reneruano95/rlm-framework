# tests/test_winjob.py
import subprocess
import sys
import time

import pytest

from rlm.sandbox.winjob import Job
from rlm.sandbox.winproc import AppContainer, spawn

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

# NEVER assign a job/pid test around sys.executable under a uv-managed venv:
# it is a 262 KB trampoline that re-execs the real interpreter as a CHILD
# process (measured here: Popen.pid != child's own os.getpid()), exactly the
# hazard Recipes §bridge documents. Assigning the trampoline's pid to a
# Job with active_process_limit=1 then races the trampoline's own re-exec
# against ACTIVE_PROCESS_LIMIT before the real interpreter -- the one
# actually running the test's code -- ever starts. Use the real interpreter
# path directly, same fix the probes used.
PY = getattr(sys, "_base_executable", sys.executable)


def test_kill_on_job_close_reaps_the_tree(tmp_path):
    job = Job(memory_limit_mb=512, active_process_limit=1)
    proc = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
    job.assign_pid(proc.pid)
    job.close()  # closing the sole handle must kill it (D5)
    assert proc.wait(timeout=10) is not None


def test_memory_limit_notifies_rather_than_killing_silently(tmp_path):
    """D3: the allocation fails; the pump is what turns that into a kill."""
    seen = []
    job = Job(memory_limit_mb=128, active_process_limit=1)
    job.watch(lambda msg, ts: seen.append(msg))
    code = "b = bytearray()\nwhile True: b += bytearray(8*1024*1024)"
    proc = subprocess.Popen([PY, "-c", code],
                            stderr=subprocess.DEVNULL)
    job.assign_pid(proc.pid)
    deadline = time.time() + 30
    while time.time() < deadline and not seen:
        time.sleep(0.2)
    job.terminate(0xB0DE)
    proc.wait(timeout=10)
    assert any("MEMORY" in m for m in seen), f"no memory notification: {seen}"


def test_active_process_limit_blocks_helper_processes():
    job = Job(memory_limit_mb=512, active_process_limit=1)
    code = ("import subprocess, sys\n"
            "try:\n"
            "    subprocess.Popen([sys.executable, '-c', 'pass'])\n"
            "    print('SPAWNED')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n")
    proc = subprocess.Popen([PY, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    job.assign_pid(proc.pid)
    out, _ = proc.communicate(timeout=30)
    job.close()
    assert "SPAWNED" not in out


def test_appcontainer_profile_lifecycle():
    ac = AppContainer()
    sid = ac.create("rlm-test-probe")
    assert sid
    ac.delete()


def test_spawn_rejects_duplicate_handle_values():
    """D2: duplicates make CreateProcessW fail with ERROR_INVALID_PARAMETER."""
    from rlm.sandbox.winproc import dedupe_handles
    assert dedupe_handles([5, 7, 5, 9, 7]) == [5, 7, 9]
