# tests/test_episode.py
import sys

import pytest

from rlm.errors import Outcome

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


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
    assert len([s for s in env.steps() if s["action_type"] == "llm_call"]) <= 8


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


async def test_root_that_never_finalises_is_a_fail_with_its_own_reason(episode_env):
    """§6: fail = final emitted and checker fails OR the root ended without
    emitting one; outcome_reason is what distinguishes them."""
    env = episode_env(root_script=["```repl\nx = 1\n```", "```repl\nx = 2\n```"],
                      max_turns=2)
    res = await env.run()
    assert res.outcome == Outcome.FAIL
    assert res.reason == "no_final_emitted"
    assert env.episode_row()["outcome_reason"] == "no_final_emitted"


async def test_checker_failure_is_a_fail_not_an_error(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('wrong')\n```"],
                      answer="right")
    res = await env.run()
    assert res.outcome == Outcome.FAIL
    assert res.reason == "checker_failed"
    assert res.final_answer == "wrong"
