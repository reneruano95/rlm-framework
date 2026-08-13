# tests/test_winjob.py
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

import pytest

from rlm.sandbox.winjob import Job
from rlm.sandbox.winproc import AppContainer, Stdio, kernel32, spawn

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


class _ProbeJob:
    """Wraps a real Job; `.assign()` is what spawn() calls BEFORE
    ResumeThread, so recording whether the child's marker file exists at
    that exact moment proves CREATE_SUSPENDED genuinely held until then --
    without this, a bug that resumed before (or without) assigning to the
    job would not be caught by any test (fix-round-1 finding: nothing
    called spawn() at all)."""

    def __init__(self, real_job: Job, marker) -> None:
        self._real = real_job
        self._marker = marker
        self.marker_existed_at_assign: bool | None = None

    def assign(self, handle: int) -> None:
        self.marker_existed_at_assign = self._marker.exists()
        self._real.assign(handle)


def test_spawn_holds_suspend_until_resume_and_quotes_args_correctly(tmp_path):
    """Fix-round-1 smoke test for winproc.spawn() (previously exercised by
    zero tests). Covers three things at once:
      - suspend -> job.assign -> resume ordering (Cross-check Conflict 1):
        the child cannot have run before job.assign() is called;
      - the child actually resumes and runs to completion (exit code 0);
      - argv quoting: a space-containing argument must arrive as ONE argv
        entry in the child, not silently split (fix-round-1 bug 1).
    """
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, sys\n"
        "marker, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
        "marker.write_text('ran', encoding='utf-8')\n"
        "out.write_text('\\x1f'.join(sys.argv[3:]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    marker = tmp_path / "marker.txt"
    out = tmp_path / "out.txt"

    import msvcrt
    nul_fd = os.open(os.devnull, os.O_RDWR)
    os.set_inheritable(nul_fd, True)
    nul_handle = msvcrt.get_osfhandle(nul_fd)
    stdio = Stdio(stdin=nul_handle, stdout=nul_handle, stderr=nul_handle)

    real_job = Job()  # active_process_limit defaults to 1 (fix-round-1 #6)
    probe = _ProbeJob(real_job, marker)
    try:
        result = spawn(
            PY, [str(child), str(marker), str(out), "hello world", "plain-arg"],
            [], None, probe, None, stdio,
        )

        assert probe.marker_existed_at_assign is False

        deadline = time.time() + 10
        while time.time() < deadline and not out.exists():
            time.sleep(0.05)
        assert marker.exists(), "child never ran: resume/job-assign is broken"
        assert out.exists()
        assert out.read_text(encoding="utf-8").split("\x1f") == ["hello world", "plain-arg"]

        WAIT_OBJECT_0 = 0
        assert kernel32.WaitForSingleObject(result.hprocess, 10_000) == WAIT_OBJECT_0
        code = wintypes.DWORD()
        assert kernel32.GetExitCodeProcess(result.hprocess, ctypes.byref(code))
        assert code.value == 0
    finally:
        real_job.close()
        os.close(nul_fd)
