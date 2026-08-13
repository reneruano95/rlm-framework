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


async def test_bridge_desync_is_classified_not_an_unattributed_death(manager, cfg):
    """A cell CAN reach the protocol fd and corrupt the stream. §6's status
    ENUM cannot express that on its own, so C1 attributes it: the sandbox is
    still alive, therefore `bridge_desync`, not `sandbox_death`."""
    from rlm import trace

    async with manager.session("ep-desync", cfg) as s:
        with pytest.raises(SandboxError) as excinfo:
            await s.exec_cell("import os\nos.write(101, b'garbage\\n')")
        assert trace.BRIDGE_DESYNC_REASON in str(excinfo.value)
        assert s.kill_reason == trace.BRIDGE_DESYNC_REASON == "bridge_desync"


async def test_frames_before_the_handshake_are_refused():
    """D23 is about ORDER as much as identity: nothing may be dispatched in an
    episode's name before that episode has been admitted."""
    import asyncio

    from rlm.sandbox.manager import _FrameGate

    served = []

    async def serve(kind, payload):
        served.append(kind)
        return "served"

    fut = asyncio.get_running_loop().create_future()
    gate = _FrameGate(fut, serve)

    with pytest.raises(SandboxError, match="before the handshake"):
        await gate("llm_query", {"prompt": "sneak"})
    with pytest.raises(SandboxError, match="before the handshake"):
        await gate("final_answer", {"value": "sneak"})
    assert served == []

    assert await gate("handshake", {"pid": 1234}) is None
    assert fut.result() == 1234
    assert await gate("llm_query", {"prompt": "ok"}) == "served"

    with pytest.raises(SandboxError, match="duplicate handshake"):
        await gate("handshake", {"pid": 4321})


async def test_episodes_do_not_share_state(manager, cfg):
    async with manager.session("ep-5", cfg) as s:
        await s.exec_cell("marker = 'first-episode'")
    async with manager.session("ep-6", cfg) as s:
        out = await s.exec_cell("print('marker' in dir())")
        assert out.stdout.strip() == "False"
