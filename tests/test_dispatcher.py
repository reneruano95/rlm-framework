# tests/test_dispatcher.py
import hashlib

import pytest

from rlm.dispatcher import MockDispatcher
from rlm.errors import DispatchError, StepStatus


async def test_mock_dispatcher_is_keyed_by_role_and_prompt_hash(tmp_path):
    fixtures = {f"leaf:{hashlib.sha256(b'q').hexdigest()}": "canned"}
    d = MockDispatcher(fixtures)
    assert await d.query("q", role="leaf", call_id="c1") == "canned"


async def test_preflight_rejects_oversize_prompts_without_dispatching(mock_server):
    d = mock_server.dispatcher(slot_capacity_tokens=100)
    with pytest.raises(DispatchError):
        await d.query("x " * 5000, role="leaf", call_id="c1")
    assert mock_server.dispatch_count == 0
    assert d.last_step["status"] == StepStatus.REJECTED


async def test_retries_share_a_call_id_and_increment_retry_idx(mock_server):
    mock_server.fail_times(2)
    d = mock_server.dispatcher()
    await d.query("q", role="leaf", call_id="c1")
    statuses = [s["status"] for s in d.steps]
    assert len(d.steps) == 3
    assert {s["call_id"] for s in d.steps} == {"c1"}
    assert [s["retry_idx"] for s in d.steps] == [0, 1, 2]
    assert statuses[-1] == StepStatus.OK


async def test_semaphore_never_exceeds_leaf_parallel(mock_server):
    d = mock_server.dispatcher(parallel=8)
    import asyncio
    await asyncio.gather(*[d.query(f"q{i}", role="leaf", call_id=f"c{i}")
                           for i in range(32)])
    assert mock_server.max_concurrent <= 8


async def test_server_death_produces_error_status_not_a_restart(mock_server):
    mock_server.kill()
    d = mock_server.dispatcher()
    with pytest.raises(DispatchError):
        await d.query("q", role="leaf", call_id="c1")
    assert d.steps[-1]["status"] == StepStatus.ERROR
    assert mock_server.restart_count == 0  # the scaffold never restarts servers


async def test_cancellation_aborts_the_stream(mock_server):
    import asyncio
    d = mock_server.dispatcher()
    task = asyncio.create_task(d.query("slow", role="leaf", call_id="c1"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mock_server.last_request_disconnected
