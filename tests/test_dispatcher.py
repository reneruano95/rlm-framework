# tests/test_dispatcher.py
import copy
import hashlib

import pytest

from rlm.config import Config
from rlm.dispatcher import LLMDispatcher, MockDispatcher
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


async def test_backoff_sleep_does_not_hold_the_semaphore(mock_server):
    """rlm/dispatcher.py holds the semaphore only around a single completion
    attempt, not around the backoff sleep between attempts ("Held only
    around THIS attempt" -- a failed attempt pins no id_slot, so holding the
    semaphore across a 1-4s backoff would starve every other queued leaf
    call for no benefit). parallel=1 makes this observable: force one call
    into a real 1s backoff after its first attempt fails, then start a
    second, healthy call on the same (single-slot) dispatcher -- it must
    complete well inside that 1s window, not after it, proving the slot was
    actually free during the sleep rather than merely idle-but-held."""
    import asyncio
    import time

    mock_server.fail_times(1)  # only the very next /completion fails
    d = mock_server.dispatcher(parallel=1, backoff_s=[1.0, 4.0])

    slow_task = asyncio.create_task(d.query("q-slow", role="leaf", call_id="slow"))
    # Give the slow call's first (failing) attempt time to land, release the
    # semaphore, and enter its 1s backoff sleep -- comfortably short of that
    # 1s window, same margin test_cancellation_aborts_the_stream uses.
    await asyncio.sleep(0.2)

    start = time.monotonic()
    await d.query("q-fast", role="leaf", call_id="fast")
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, (
        f"second call took {elapsed:.2f}s on a parallel=1 dispatcher -- it "
        "waited out the first call's backoff instead of running during it"
    )

    await slow_task  # let the retried (now-succeeding) attempt finish cleanly


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


async def test_retry_exhaustion_raises_after_max_attempts_all_logged(mock_server):
    """Distinct from test_server_death_produces_error_status_not_a_restart,
    which never enters the retry loop at all (it fails at the pre-flight
    /tokenize stage against a dead server). Here the server stays fully
    reachable -- /tokenize succeeds -- and only /completion fails, 3
    consecutive times, so this actually exercises retry exhaustion."""
    mock_server.fail_times(3)
    d = mock_server.dispatcher()
    with pytest.raises(DispatchError):
        await d.query("q", role="leaf", call_id="c1")
    assert len(d.steps) == 3
    assert {s["call_id"] for s in d.steps} == {"c1"}
    assert [s["retry_idx"] for s in d.steps] == [0, 1, 2]
    assert all(s["status"] == StepStatus.ERROR for s in d.steps)
    assert mock_server.restart_count == 0  # the scaffold never restarts servers


async def test_leaf_sampling_params_reach_the_server(mock_server, minimal_cfg_dict):
    """cfg.scaffold.sampling.leaf must reach the server on every /completion
    call -- real, non-defaulted config, not ServerClient.completion's
    (deliberately absent) greedy defaults."""
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["servers"]["leaf"]["port"] = mock_server.port
    cfg = Config.model_validate(raw)
    d = LLMDispatcher.from_config(cfg)
    try:
        await d.query("q", role="leaf", call_id="c1")
    finally:
        await d.aclose()
    body = mock_server.last_completion_body
    expected = cfg.scaffold.sampling.leaf
    assert body["temperature"] == expected.temperature
    assert body["top_p"] == expected.top_p
    assert body["seed"] == expected.seed


async def test_root_role_is_not_a_valid_dispatch_target_from_config(minimal_cfg_dict, mock_server):
    """Root traffic never goes through LLMDispatcher (rootclient talks to a
    raw ServerClient directly), so from_config() must not build a "root"
    target that could silently apply the leaf-sized semaphore to root
    calls if ever queried by mistake."""
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["servers"]["leaf"]["port"] = mock_server.port
    cfg = Config.model_validate(raw)
    d = LLMDispatcher.from_config(cfg)
    try:
        with pytest.raises(DispatchError):
            await d.query("q", role="root", call_id="c1")
    finally:
        await d.aclose()
