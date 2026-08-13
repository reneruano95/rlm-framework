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
