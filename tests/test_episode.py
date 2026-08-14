# tests/test_episode.py
import asyncio
import sys

import pytest

from rlm.errors import Outcome

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


class _RetryingDispatcher:
    """A C4 stand-in whose first attempts fail AFTER burning real tokens.

    The shipped `LLMDispatcher` fills token counts only on the OK path (a
    stream that never yields a final event reports nothing), so no existing
    fixture can exercise "every attempt's tokens count". A server that times
    out after prefill genuinely does spend them, which is the case §5 C4's
    rule is written for.
    """

    def __init__(self, *, fail_attempts: int = 2, tokens_in: int = 100,
                 tokens_out: int = 50, parallel: int = 8) -> None:
        self.semaphore = asyncio.Semaphore(parallel)
        self.steps: list[dict] = []
        self._fail = fail_attempts
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        return (len(text) + 3) // 4

    def set_corpus(self, chunks) -> None:
        """Part of C4's interface since R13: the episode runner hands every
        dispatcher the corpus so the foreign-string detector can run. This
        double answers from a canned table and never produces a leaf answer to
        check, so it accepts and ignores it -- deliberately not defended
        against in `run_episode`, where a dispatcher that cannot be given the
        corpus should fail loudly rather than silently skip detection."""

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None) -> str:
        async with self.semaphore:
            for attempt in range(self._fail + 1):
                if attempt == self._fail:
                    status = "ok"
                else:
                    status = "error" if attempt == 0 else "timeout"
                self.steps.append({
                    "call_id": call_id, "retry_idx": attempt, "status": status,
                    "tokens_in": self._tokens_in, "tokens_out": self._tokens_out,
                })
            return f"LEAF:{prompt}"


async def test_happy_path_emits_final_and_logs_every_step(episode_env):
    """Mock dispatcher, real sandbox, real C1/C2/C3/C5/C6 (spec §5 dry-run)."""
    env = episode_env(root_script=[
        "```repl\nprint(len(chunks))\n```",
        "```repl\nfinal_answer('42')\n```",
    ])
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert res.final_answer == "42"
    kinds = [s["action_type"] for s in env.steps()]
    assert kinds[-1] == "final"
    assert all(s["episode_id"] == res.episode_id for s in env.steps())


async def test_final_answer_is_the_only_terminal_channel(episode_env):
    """I2: prose is never parsed as an answer — it would smuggle context past C3."""
    env = episode_env(root_script=[
        "The answer is 42.",              # prose only, no cell
        "```repl\nfinal_answer('42')\n```",
    ])
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    no_cell = [s for s in env.steps() if s.get("error_detail") == "no_cell_extracted"]
    assert len(no_cell) == 1
    assert no_cell[0]["status"] == "rejected"


async def test_root_never_receives_untruncated_output(episode_env):
    env = episode_env(root_script=[
        "```repl\nprint(context)\n```",
        "```repl\nfinal_answer('done')\n```",
    ], context="X" * 500_000, truncation_cap=2000)
    await env.run()
    exec_step = [s for s in env.steps() if s["action_type"] == "repl_exec"][0]
    assert len(exec_step["observation_view"]) <= 2000
    assert "[truncated: showing" in exec_step["observation_view"]
    assert env.blob(exec_step["observation_full_ref"]).__len__() > 400_000  # I2


async def test_wall_clock_breach_is_budget_kill_and_persists_the_trace(episode_env):
    env = episode_env(root_script=["```repl\nwhile True:\n    pass\n```"],
                      max_wall_clock_s=5)
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert res.reason == "wall_clock"
    assert env.episode_row()["ended_at"] is not None
    assert env.steps(), "partial trace must survive the kill"


async def test_runaway_subcalls_terminate_deterministically(episode_env):
    env = episode_env(root_script=[
        "```repl\nimport asyncio\n"
        "await asyncio.gather(*[llm_query(f'q{i}') for i in range(100)])\n```",
    ], max_subcalls=8)
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert len(calls) <= 8
    # `<= 8` alone passes vacuously at zero calls — which is what a broken
    # dispatch path looks like. The cap is only demonstrated if calls actually
    # got through, and only attributable if the breach names the right budget.
    assert len(calls) > 0, "no sub-call was dispatched; the cap proves nothing"
    assert len({s["call_id"] for s in calls}) == len(calls) <= 8
    assert res.reason == "max_subcalls"


async def test_dry_run_episodes_are_flagged(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('x')\n```"])
    res = await env.run()
    assert env.episode_row()["dry_run"] is True
    assert env.episode_row()["config_snapshot"]["scaffold"]["dispatcher"] == "mock"


async def test_root_view_hash_is_recorded_for_every_root_turn(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('x')\n```"])
    await env.run()
    root_steps = [s for s in env.steps() if s["actor"] == "root"]
    assert all(s["root_view_hash"] for s in root_steps)
    assert all(s["root_request_ref"] for s in root_steps)  # D17 offline replay


# --------------------------------------------------------------------------- #
# Added beyond the brief: the three contracts the brief's own tests state in
# prose but never assert — I2's message-array half, parent_step_idx causality,
# and the `no_final_emitted` arm of §6's outcome semantics.
# --------------------------------------------------------------------------- #


async def test_context_never_enters_a_message_array(episode_env):
    """I2's other half. The brief's truncation test proves the root never SEES
    the full observation; this proves the full CONTEXT never reaches the
    rendered request either — it is setvar'd into the sandbox and referenced
    by name, so the only place it exists scaffold-side is the blob store."""
    needle = "SENTINEL-" + "Q" * 200
    env = episode_env(root_script=[
        "```repl\nprint(len(context))\n```",
        "```repl\nfinal_answer('ok')\n```",
    ], context=needle + ("z" * 50_000))
    await env.run()
    refs = {s["root_request_ref"] for s in env.steps() if s["root_request_ref"]}
    assert refs
    for ref in refs:
        assert needle.encode() not in env.blob(ref)


async def test_leaf_calls_hang_off_the_repl_exec_that_spawned_them(episode_env):
    """Causality lives in parent_step_idx/call_id, never in step_idx adjacency."""
    env = episode_env(root_script=[
        "```repl\nprint(await llm_query('one'))\n```",
        "```repl\nfinal_answer('done')\n```",
    ])
    await env.run()
    steps = env.steps()
    execs = {s["step_idx"] for s in steps if s["action_type"] == "repl_exec"}
    calls = [s for s in steps if s["action_type"] == "llm_call"]
    assert len(calls) == 1
    assert calls[0]["parent_step_idx"] in execs
    assert calls[0]["actor"] == "leaf" and calls[0]["depth"] == 1
    assert calls[0]["call_id"] is not None


async def test_a_chunk_kwarg_is_composed_scaffold_side_before_dispatch(episode_env):
    """§4's `[prefix][chunk][question]` layout is ENFORCED, not hoped for: the
    model hands the bridge two fields and the scaffold composes the user
    message, exactly as it already composes the system prefix (I1). The logged
    `action_payload` is the composed string, because that is what was sent."""
    env = episode_env(root_script=[
        "```repl\nprint(await llm_query('Q?', chunk='CHUNK'))\n```",
        "```repl\nfinal_answer('done')\n```",
    ])
    await env.run()

    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert len(calls) == 1
    assert calls[0]["action_payload"] == "CHUNK\n\nQ?"
    assert env.dispatcher.last_step["layout"] == "chunk_question"


async def test_the_single_string_llm_query_still_dispatches(episode_env):
    """`chunk=None` is today's behaviour unchanged -- the S1 prompts and the
    paper harness's own call site both use it."""
    env = episode_env(root_script=[
        "```repl\nprint(await llm_query('CHUNK\\n\\nQ?'))\n```",
        "```repl\nfinal_answer('done')\n```",
    ])
    await env.run()

    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert calls[0]["action_payload"] == "CHUNK\n\nQ?"
    assert env.dispatcher.last_step["layout"] == "question_only"


async def test_a_non_string_chunk_is_refused_scaffold_side(episode_env):
    """The bridge carries whatever JSON the model wrote. Composition happens
    scaffold-side, so a non-string chunk must be refused there with a legible
    message rather than raising a TypeError out of the middle of C4."""
    env = episode_env(root_script=[
        "```repl\ntry:\n    await llm_query('Q?', chunk=[1, 2])\n"
        "except Exception as exc:\n    print(type(exc).__name__, exc)\n```",
        "```repl\nfinal_answer('done')\n```",
    ])
    await env.run()
    execs = [s for s in env.steps() if s["action_type"] == "repl_exec"]
    assert "chunk" in execs[0]["observation_view"]
    assert not [s for s in env.steps() if s["action_type"] == "llm_call"]


async def test_a_leaf_step_stores_its_rendered_request_like_a_root_turn(
        episode_env, mock_server):
    """§6's state-rule instrument, applied to the leaf. A gate-(a) failure is
    otherwise uninvestigable: prefix drift and slot eviction produce an
    identical symptom, and telling them apart needs the exact bytes that were
    sent and the prefix's token length next to the `tokens_cached` being
    judged. The root path already stores both; the leaf was the odd one out.

    Runs the REAL LLMDispatcher (against the loopback fixture server) rather
    than the canned one, because the render is what is under test."""
    import hashlib
    import json

    from rlm.trace import unpack_blob

    d = mock_server.dispatcher()
    env = episode_env(root_script=[
        "```repl\nprint(await llm_query('Q?', chunk='CHUNK'))\n```",
        "```repl\nfinal_answer('done')\n```",
    ], dispatcher=d)
    await env.run()

    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert len(calls) == 1
    call = calls[0]
    assert call["root_request_ref"], "the leaf request was never stored"

    blob = unpack_blob(env.blob(call["root_request_ref"]))
    rendered = mock_server.rendered_prompts[0]
    assert blob["rendered"].decode("utf-8") == rendered
    assert hashlib.sha256(blob["rendered"]).hexdigest() == call["root_view_hash"]

    meta = json.loads(blob["meta"])
    assert meta["layout"] == "chunk_question"
    assert meta["prefix_tokens"] == d.prefix_tokens("leaf") > 0


async def test_root_that_never_finalises_is_a_fail_with_its_own_reason(episode_env):
    """§6: fail = final emitted and checker fails OR the root ended without
    emitting one; outcome_reason is what distinguishes them."""
    env = episode_env(root_script=["```repl\nx = 1\n```", "```repl\nx = 2\n```"],
                      max_turns=2)
    res = await env.run()
    assert res.outcome == Outcome.FAIL
    assert res.reason == "no_final_emitted"
    assert env.episode_row()["outcome_reason"] == "no_final_emitted"


@pytest.mark.timeout(20)
async def test_operator_abort_records_its_outcome_and_never_dies_mid_write(episode_env):
    """Spec §5 C5: Ctrl-C routes through the same path, with
    outcome_reason=operator_abort. The row must be CLOSED — an episode left
    with a NULL outcome tombstones as `orphaned_at_recovery` instead, which
    would misattribute a deliberate abort as a crash.

    The 20 s bound is load-bearing, not hygiene. Killing AFTER unwinding the
    session (rather than inside it) makes `close()` take its graceful path
    against a sandbox wedged in `while True: pass`, spending the 15 s shutdown
    grace twice — 33 s measured. Without a bound that regression reads as a
    slow test rather than as broken kill ordering. Observed here: ~3.0 s, all
    of it the 3 s delay below.
    """
    import asyncio

    env = episode_env(root_script=["```repl\nwhile True:\n    pass\n```"],
                      max_wall_clock_s=120)
    task = asyncio.current_task()
    asyncio.get_running_loop().call_later(3.0, task.cancel)
    with pytest.raises(asyncio.CancelledError):
        await env.run()
    env._load()
    assert env.episode_row()["outcome"] == "budget_kill"
    assert env.episode_row()["outcome_reason"] == "operator_abort"
    assert env.episode_row()["ended_at"] is not None


async def test_checker_failure_is_a_fail_not_an_error(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('wrong')\n```"],
                      answer="right")
    res = await env.run()
    assert res.outcome == Outcome.FAIL
    assert res.reason == "checker_failed"
    assert res.final_answer == "wrong"


def test_every_attempt_of_a_retried_call_is_charged():
    """§5 C4: a retried call counts ONCE against max_subcalls, but EVERY
    attempt's tokens count against max_total_tokens. Charging only the
    successful attempt would make retries free, which is the exact thing that
    asymmetry exists to prevent."""
    from rlm.episode import settled_tokens

    attempts = [
        {"retry_idx": 0, "status": "error", "tokens_in": 100, "tokens_out": 50},
        {"retry_idx": 1, "status": "timeout", "tokens_in": 100, "tokens_out": 50},
        {"retry_idx": 2, "status": "ok", "tokens_in": 100, "tokens_out": 50},
    ]
    assert settled_tokens(attempts) == (300, 150)          # not (100, 50)
    # Attempts that reported no usage contribute nothing, so the same
    # summation is safe on the cancellation path.
    assert settled_tokens([{"status": "cancelled"}]) == (0, 0)
    assert settled_tokens([]) == (0, 0)


async def test_retried_attempt_tokens_reach_the_budget_and_the_trace(episode_env):
    """The end-to-end half: two failing attempts that each burned real tokens
    must push the SECOND call past max_total_tokens. Charging only the
    successful attempt leaves room and the episode wrongly succeeds."""
    env = episode_env(
        root_script=[
            "```repl\nprint(await llm_query('q'))\nprint(await llm_query('r'))\n```",
            "```repl\nfinal_answer('x')\n```",
        ],
        # call 1 settles 3 x (100+50) = 450; the call-2 reservation is
        # prompt(1) + max_predict.leaf(512) = 513. 450+513 = 963 > 800 breaches,
        # while charging only the ok attempt gives 150+513 = 663 and does not.
        dispatcher=_RetryingDispatcher(fail_attempts=2, tokens_in=100, tokens_out=50),
        max_total_tokens=800)
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert res.reason == "max_total_tokens"
    # C4's other half: every attempt is its own logged step, sharing one
    # call_id with an incrementing retry_idx.
    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert len({s["call_id"] for s in calls}) == 1
    assert sorted(s["retry_idx"] for s in calls) == [0, 1, 2]


@pytest.mark.timeout(60)
async def test_a_final_that_arrived_first_survives_a_breach_on_the_same_turn(episode_env):
    """§6 outcome ordering, and the reason it is ordering and not preference.

    The cell submits a valid answer, waits long enough for the frame to cross
    the bridge and be ACCEPTED, then wedges itself so the wall clock expires on
    that same turn. Both facts are now true of one episode. The answer was
    accepted before the breach — `_on_final_answer` refuses once a breach is
    set, so `_final_emitted` implies it arrived first — and the episode is a
    success. Checking the breach first would erase an episode that genuinely
    finished, on nothing but a timer.
    """
    env = episode_env(root_script=[
        "```repl\n"
        "import asyncio, time\n"
        "final_answer('42')\n"
        "await asyncio.sleep(0.5)\n"   # the frame goes out and is accepted here
        "time.sleep(30)\n"             # then wedge: the wall clock expires
        "```",
    ], max_wall_clock_s=3, answer="42")
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert res.final_answer == "42"
    assert env.episode_row()["outcome"] == "success"
    assert [s["action_type"] for s in env.steps()][-1] == "final"
    assert env.episode_row()["final_answer_ref"], "the answer must be stored"


async def test_the_corpus_reaches_c4_so_every_leaf_answer_is_leak_checked(
        episode_env, mock_server):
    """R13 (§5 C4): the detector is free only because the scaffold already
    holds every chunk -- so C2's chunks must actually reach C4, once, at
    episode start. Without that the leaf steps record NULL ("not checked"),
    which is honest but useless; with it they record a real verdict.

    A False here is evidence, not a certificate: 138 clean calls give a 95%
    upper bound of 2.2%, so an 848-call episode may still carry ~19
    contaminated answers."""
    d = mock_server.dispatcher()
    env = episode_env(context="alpha beta gamma delta", root_script=[
        "```repl\nprint(await llm_query('Q?', chunk=chunks[0]))\n```",
        "```repl\nfinal_answer('done')\n```",
    ], dispatcher=d)
    await env.run()

    assert d.chunk_index is not None, "C2's chunks never reached C4"
    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert len(calls) == 1
    assert calls[0]["leak_detected"] is False   # checked, no foreign identifier
    assert calls[0]["leak_detail"] is None


# --------------------------------------------------------------------------- #
# Slot-pool ROTATION (spec §5 C4, v0.2.6).
#
# R13's mitigation spends one slot per window, so a pool of `--parallel` slots
# dies at window `--parallel` while a 200K corpus needs 261 -- the mitigation is
# inert without a rotation. §5 C4 permits exactly one shape of it: on pool
# exhaustion, never on an error; quiesce, replace the process, RE-RUN §4's
# /props handshake, resume; logged as a lifecycle event and stamped on the step
# that triggered it; and its wall-clock counted inside the episode's.
#
# The runner drives it; the process is owned by an injected manager
# (`rlm.serverproc.ProcessManager`), so these tests need no llama-server -- and
# C4 keeps having no code path that restarts anything.
# --------------------------------------------------------------------------- #


class FakeLeafProcess:
    """A `ProcessManager` that replaces nothing and records everything."""

    def __init__(self, *, delay: float = 0.0, fail: Exception | None = None,
                 on_restart=None) -> None:
        self.restarts = 0
        self._delay = delay
        self._fail = fail
        self._on_restart = on_restart

    async def restart(self) -> None:
        self.restarts += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._on_restart is not None:
            self._on_restart()
        if self._fail is not None:
            raise self._fail


class DeadLeafDispatcher:
    """C4 against a server that is gone: every call exhausts its retries."""

    def __init__(self) -> None:
        self.semaphore = asyncio.Semaphore(8)
        self.steps: list[dict] = []

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        return (len(text) + 3) // 4

    def set_corpus(self, chunks) -> None:
        pass

    async def query(self, prompt: str, *, role: str, call_id: str,
                    chunk: str | None = None) -> str:
        from rlm.errors import DispatchError

        self.steps.append({"call_id": call_id, "retry_idx": 0, "status": "error",
                           "error_detail": "connection refused"})
        raise DispatchError("dispatch failed after 3 attempts: connection refused")


class RestartingLeafProcess:
    """A `ProcessManager` that GENUINELY takes the server away.

    `FakeLeafProcess` replaces nothing -- the mock server answers throughout a
    "restart" -- so every rotation test written against it passes whether or
    not the scaffold quiesced the traffic whose process it was about to pull
    out. This one stops listening for `down_s`, so anything on the wire during
    that window fails exactly as it would against a killed process, and then
    rebinds the same port the way a relaunched llama-server does.
    """

    def __init__(self, server, *, down_s: float = 0.2) -> None:
        self.server = server
        self.restarts = 0
        self._down_s = down_s

    async def restart(self) -> None:
        self.restarts += 1
        self.server.take_down()
        await asyncio.sleep(self._down_s)
        self.server.bring_up()


FANOUT = ("```repl\n"
          "for i in range(3):\n"
          "    print(await llm_query('Q?', chunk=f'window {i}'))\n"
          "```")
#: The shape the pinned aggregation template actually prescribes: "map once
#: over all of `chunks`. One `asyncio.gather`, the same question for every
#: chunk, no exceptions and no early stopping." `return_exceptions=True` so the
#: cell survives to report per-window outcomes instead of the first failure
#: taking the whole map down -- which is also what makes a silently partial map
#: visible to the assertions rather than to nobody.
#:
#: The stagger is not decoration and it is not a synthetic race. A real map is
#: over HUNDREDS of windows against a concurrency of 8, so calls do not arrive
#: in one burst: they start continuously for the whole episode, and some
#: therefore start while a rotation is in progress. A single simultaneous burst
#: is the one arrival pattern that CANNOT expose this -- every call is already
#: past its pre-flight before the first pool exhaustion, so nothing is on the
#: wire when the process is replaced.
CONCURRENT_FANOUT = (
    "```repl\n"
    "import asyncio\n"
    "async def one(i):\n"
    "    await asyncio.sleep(i * 0.08)\n"
    "    return await llm_query('Q?', chunk=f'window {i}')\n"
    "outs = await asyncio.gather(*[one(i) for i in range(6)],\n"
    "                            return_exceptions=True)\n"
    "print([type(o).__name__ if isinstance(o, BaseException) else 'ok'\n"
    "       for o in outs])\n"
    "```")
FINAL = "```repl\nfinal_answer('42')\n```"


async def test_pool_exhaustion_rotates_the_leaf_and_the_episode_continues(
        episode_env, mock_server):
    """Without this the R13 mitigation is inert: window `--parallel` + 1 ends
    the episode. Two windows fit the pool, the third rotates."""
    pm = FakeLeafProcess()
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert pm.restarts == 1
    assert mock_server.requested_slots() == [0, 1, 0]     # virgin again after it
    ok = [s for s in env.steps()
          if s["action_type"] == "llm_call" and s["status"] == "ok"]
    assert len(ok) == 3                                    # all three answered


async def test_a_concurrent_fanout_loses_no_window_to_a_rotation(
        episode_env, mock_server):
    """THE case every other rotation test misses, in the shape the benchmark
    actually runs.

    Two things had to be true at once for the sequential tests to pass a
    broken rotation: they dispatch one call at a time, and `FakeLeafProcess`
    replaces nothing. Here six windows go out under one `asyncio.gather` -- the
    pinned aggregation template's own full-coverage MAP -- against a manager
    that genuinely stops the listener. A call that has passed its pre-flight
    but not yet taken a slot, or one re-dispatching after another call's
    rotation, is then talking to a process that is being replaced; before the
    gate moved ahead of the pre-flight, quiesce() returned while those round
    trips were still on the wire and they died with `status=error` and no
    retry.

    Why this is worse than an error rate: the aggregation category exists to
    force coverage and punish sampling, the template maps concurrently over
    every chunk, and a partial map still prints an answer. The episode reports
    SUCCESS with windows silently missing -- which is precisely the failure §8
    says the category must catch.
    """
    pm = RestartingLeafProcess(mock_server, down_s=0.2)
    env = episode_env(root_script=[CONCURRENT_FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()

    assert res.outcome == Outcome.SUCCESS
    assert pm.restarts >= 2, "six windows on a pool of two must rotate twice"

    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    answered = [s for s in calls if s["status"] == "ok"]
    assert len(answered) == 6, (
        f"{len(answered)}/6 windows answered -- the rest were lost to a "
        f"rotation: {[(s['status'], s['error_detail']) for s in calls]}")

    # Every window, once each: coverage is the property, not merely the count.
    windows = {env.blob(s["observation_full_ref"]).decode("utf-8")
               for s in answered}
    assert len(windows) == 6

    # The only admissible error is the pool refusal that TRIGGERS a rotation.
    # Anything else is a leaf round trip that died against a replaced process.
    stray = [s for s in calls if s["status"] == "error"
             and "slot pool exhausted" not in (s["error_detail"] or "")]
    assert stray == [], f"leaf traffic died across a rotation: {stray}"


async def test_the_rotation_is_a_lifecycle_event_and_a_stamp_on_its_trigger(
        episode_env, mock_server):
    """Both channels, because each covers the other's blind spot: the S3 gate
    deletes the lifecycle log, and the log is the only place a rotation that
    never reached a step could appear."""
    pm = FakeLeafProcess()
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    await env.run()

    rotating = [e for e in env.lifecycle_events()
                if e["kind"] == "server_health" and e.get("state") == "rotating"]
    assert len(rotating) == 1
    assert rotating[0]["reason"] == "slot_pool_exhausted"
    assert rotating[0]["role"] == "leaf"
    assert [e for e in env.lifecycle_events()
            if e["kind"] == "server_health" and e.get("state") == "rotated"]

    stamped = [s for s in env.steps() if s["server_rotation"] is not None]
    assert len(stamped) == 1
    assert stamped[0]["server_rotation"] == 1
    assert stamped[0]["status"] == "error"
    assert "slot pool exhausted" in stamped[0]["error_detail"]


async def test_the_props_handshake_re_runs_before_the_episode_resumes(
        episode_env, mock_server):
    pm = FakeLeafProcess()
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    before = mock_server.props_count
    await env.run()
    assert mock_server.props_count > before


async def test_a_leaf_that_comes_back_with_different_flags_stops_the_episode(
        episode_env, mock_server):
    """The failure the handshake exists to catch (§4): a rotation that silently
    returns a server configured differently. Continuing would measure one
    topology while reporting another, and at `total_slots` it would hand out
    slot ids the server silently reassigns onto used slots (R13)."""
    pm = FakeLeafProcess(
        on_restart=lambda: setattr(mock_server, "total_slots", 4))
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()
    assert pm.restarts == 1
    assert res.outcome == Outcome.ERROR
    assert res.reason == "rotation_failed"
    assert mock_server.requested_slots() == [0, 1]   # nothing dispatched after


async def test_a_failed_server_is_never_restarted(episode_env):
    """§5 C4's original rule, intact: rotation fires on POOL EXHAUSTION only.
    Relaunching a server that just failed would mask the fault the trace exists
    to record."""
    pm = FakeLeafProcess()
    env = episode_env(
        root_script=["```repl\nprint(await llm_query('Q?', chunk='w'))\n```", FINAL],
        answer="42", process_manager=pm, dispatcher=DeadLeafDispatcher())
    await env.run()
    assert pm.restarts == 0
    assert any(s["status"] == "error" for s in env.steps())


#: Same fan-out as FANOUT, but the cell survives a failing window so the pool
#: can actually be drained by errors rather than the first one ending the cell.
ERROR_TOLERANT_FANOUT = ("```repl\n"
                         "for i in range(3):\n"
                         "    try:\n"
                         "        print(await llm_query('Q?', chunk=f'window {i}'))\n"
                         "    except Exception as exc:\n"
                         "        print('ERR', type(exc).__name__)\n"
                         "```")


async def test_a_pool_drained_by_errors_is_never_rotated(episode_env, mock_server):
    """§5 C4's original rule, which the rotation quietly broke: the scaffold
    never restarts a FAILED server.

    `SlotPoolExhausted` says nothing about WHY the slots went. A pool drained
    by three consecutive dispatch failures raises exactly the same exception
    as a pool drained by three answered windows, so the runner relaunched a
    failing leaf and the lifecycle log recorded it as a planned rotation --
    `restarts 1`, every step `error`, and `[('rotating',
    'slot_pool_exhausted'), ('rotated', None)]` in the log. That masks the
    fault the trace exists to record, which is the entire reason rotation was
    only ever permissible for a HEALTHY server.
    """
    mock_server.fail_times(10_000)          # every /completion attempt fails
    pm = FakeLeafProcess()
    env = episode_env(root_script=[ERROR_TOLERANT_FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()

    assert pm.restarts == 0, "a failing leaf was relaunched"
    assert res.outcome == Outcome.ERROR
    assert res.reason == "slot_pool_error_drained"

    health = [(e.get("state"), e.get("reason")) for e in env.lifecycle_events()
              if e["kind"] == "server_health"]
    assert ("rotating", "slot_pool_exhausted") not in health, health
    assert ("rotation_refused", "slot_pool_error_drained") in health, health
    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert calls and all(s["status"] == "error" for s in calls)


async def test_a_pool_that_answered_a_window_still_rotates(episode_env, mock_server):
    """The refusal is scoped to a generation that served NOTHING. A server
    that answered a window on this pool has demonstrated it can serve, so a
    single transient failure alongside it must not turn into a dead episode --
    that would be the opposite over-correction, and it would make one flaky
    call cost the whole coverage pass."""
    mock_server.fail_times(3)               # exactly window 0's three attempts
    pm = FakeLeafProcess()
    env = episode_env(root_script=[ERROR_TOLERANT_FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()

    assert res.outcome == Outcome.SUCCESS
    assert pm.restarts == 1
    assert any(e.get("state") == "rotating" for e in env.lifecycle_events())


async def test_rotation_time_is_inside_the_episodes_measured_wall_clock(
        episode_env, mock_server):
    """§5 C4: "its wall-clock is included in the episode's measured time" --
    2 rotations ~ 13.4 s per 200K corpus, and §8 excludes between-ARM relaunch
    time, never this."""
    pm = FakeLeafProcess(delay=1.0)
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    await env.run()
    row = env.episode_row()
    assert (row["ended_at"] - row["started_at"]).total_seconds() >= 1.0


async def test_a_rotation_that_outlives_the_wall_clock_is_killed_by_c5(
        episode_env, mock_server):
    """The other half of "counted": the clock keeps running THROUGH a rotation,
    so a rotation that overruns the budget ends the episode like any other
    overrun instead of quietly extending it."""
    pm = FakeLeafProcess(delay=30.0)
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port, process_manager=pm,
                      max_wall_clock_s=5,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert res.reason == "wall_clock"


async def test_without_a_process_manager_exhaustion_still_refuses_to_reuse(
        episode_env, mock_server):
    """No manager injected (today's `rlm run` against externally launched
    servers): the refusal reaches the cell unchanged. What must NOT happen is a
    silent wrap-around onto a used slot."""
    env = episode_env(root_script=[FANOUT, FINAL], answer="42",
                      leaf_port=mock_server.port,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    await env.run()
    assert mock_server.requested_slots() == [0, 1]
    assert any(e.get("state") == "rotation_unavailable"
               for e in env.lifecycle_events())


async def test_both_questions_about_a_window_precede_its_retirement(
        episode_env, mock_server):
    """A rotation discards every warm slot, so both questions about a window
    have to be asked before that window's slot is retired. What guarantees it
    is intra-window grouping: `window_key` is the chunk's bytes, so a second
    question takes the window's own slot and consumes no virgin one. A pool of
    N therefore serves N windows x 2 questions with NO rotation -- and a change
    that keys windows by call_id instead would rotate at N/2 windows, which is
    what this test fails on."""
    pm = FakeLeafProcess()
    env = episode_env(root_script=[
        "```repl\n"
        "for i in range(2):\n"
        "    w = f'window {i}'\n"
        "    print(await llm_query('first?', chunk=w))\n"
        "    print(await llm_query('second?', chunk=w))\n"
        "```",
        FINAL,
    ], answer="42", leaf_port=mock_server.port, process_manager=pm,
        dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2))
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert pm.restarts == 0
    assert mock_server.requested_slots() == [0, 0, 1, 1]
