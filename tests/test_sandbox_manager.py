import sys
import time

import pytest

from rlm.errors import SandboxError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


async def test_handshake_pid_must_match_the_spawned_pid(manager, cfg):
    async with manager.session("ep-1", cfg) as s:
        assert s.pid > 0
        out = await s.exec_cell("import os; print(os.getpid())")
        assert out.stdout.strip() == str(s.pid)  # D23


async def test_kill_terminates_via_the_job_and_reports_the_reason(manager, cfg):
    async with manager.session("ep-2", cfg) as s:
        await s.exec_cell("x = 1")
        await s.kill("wall_clock", 0xC5)
        assert s.kill_reason == "wall_clock"
        with pytest.raises(SandboxError):
            await s.exec_cell("print(x)")


async def test_setvar_injects_a_32mb_payload_through_the_bridge(manager, cfg):
    """Deferred gap: measure this through an AppContainer bridge, not a plain pipe."""
    payload = "x" * (32 * 1024 * 1024)
    async with manager.session("ep-3", cfg) as s:
        t0 = time.perf_counter()
        await s.setvar("context", payload)
        elapsed = time.perf_counter() - t0
        out = await s.exec_cell("print(len(context))")
        assert out.stdout.strip() == str(len(payload))
        assert elapsed < 10.0, f"32 MB setvar took {elapsed:.2f}s"


async def test_sandbox_cannot_read_the_repo(manager, cfg):
    """D7: config.yaml and prompts/ must be denied by default."""
    async with manager.session("ep-4", cfg) as s:
        out = await s.exec_cell(
            "try:\n"
            "    open(r'D:\\PROJECTS\\rlm-halo-framework\\config.yaml').read()\n"
            "    print('READABLE')\n"
            "except OSError as e:\n"
            "    print('DENIED')\n")
        assert out.stdout.strip() == "DENIED"


async def test_episodes_do_not_share_state(manager, cfg):
    async with manager.session("ep-5", cfg) as s:
        await s.exec_cell("marker = 'first-episode'")
    async with manager.session("ep-6", cfg) as s:
        out = await s.exec_cell("print('marker' in dir())")
        assert out.stdout.strip() == "False"
