"""§8's baseline arms (`src/rlm/measure/arms.py`): shared plumbing + B1 single-shot.

Every test here runs the REAL arm code against a canned dispatcher -- no
servers. The dispatcher double subclasses `conftest.CannedDispatcher` so the
steps the arm logs are the ones C4 actually records (leak columns included),
rather than a second, prettier implementation of them.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import socket
from pathlib import Path

import duckdb
import pytest
from conftest import CannedDispatcher, _decode_episode_row, _rows

from rlm import episode as episodemod
from rlm.measure.arms import (
    ARM_ERROR,
    CHECKER_FAILED,
    NO_ANSWER,
    NO_SUMMARY,
    ROTATION_FAILED,
    SERVER_UNREACHABLE,
    SLOT_POOL_EXHAUSTED,
    ArmEpisode,
    ArmResult,
    b2_summary_n_predict,
    bench_slot_capacity,
    bm25_select,
    outcome_for_error,
    run_b1,
    run_b2,
    run_b3,
    truncate_head_tail,
)
from rlm.config import Config, resolve_prompt_path
from rlm.episode import Task
from rlm.errors import (
    BudgetBreach,
    ConfigError,
    DispatchError,
    Outcome,
    ServerRotationError,
    SlotPoolExhausted,
    StepStatus,
)
from rlm.trace import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


class _ScriptedDispatcher(CannedDispatcher):
    """A `CannedDispatcher` that answers every prompt with one fixed reply.

    The arm assembles its own prompt (head + truncated corpus + question), so
    a test cannot key a fixture by hash up front; this seeds the fixture for
    whatever prompt actually arrives and keeps every other piece of
    `MockDispatcher` -- the step record, the leak columns, the semaphore -- real.
    """

    def __init__(self, reply: str = "ANSWER", *, delay: float = 0.0,
                 fail: bool = False, count_penalty: int = 0,
                 penalise=None) -> None:
        super().__init__()
        self.reply = reply
        self.delay = delay
        self.fail = fail
        self.count_penalty = count_penalty
        self._penalise = penalise
        self.prompts: list[str] = []
        self.counted: list[str] = []
        self.slot_ids: list[int | None] = []
        self.seeds: list[int | None] = []
        self.n_predicts: list[int | None] = []

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        self.counted.append(text)
        base = await super().count_tokens(text, role=role)
        if self.count_penalty and self._penalise is not None and self._penalise(text):
            # A real /tokenize counts the chat template's markup, which a raw
            # text count cannot see -- so the assembled prompt is bigger than
            # the sum of its parts. That is the case the fit verify loop exists
            # for, and this is how it is provoked deterministically.
            return base + self.count_penalty
        return base

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, seed: int | None = None,
                     n_predict: int | None = None) -> str:
        self.seeds.append(seed)
        self.n_predicts.append(n_predict)
        from rlm.serve.dispatcher import compose_leaf_user

        self.prompts.append(prompt)
        composed = compose_leaf_user(prompt, chunk)
        key = f"{role}:{hashlib.sha256(composed.encode('utf-8')).hexdigest()}"
        if self.fail:
            self._inner._fixtures.pop(key, None)
        else:
            self._inner._fixtures[key] = self.reply
        if self.delay:
            await asyncio.sleep(self.delay)
        return await self._inner.query(prompt, role=role, call_id=call_id,
                                        chunk=chunk)


class _SlotAwareDispatcher(_ScriptedDispatcher):
    """A dispatcher whose `query` DECLARES `slot_id` -- the Task-10 bench
    profile's shape. The arm must pass the pin through to it."""

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, slot_id: int | None = None,
                     seed: int | None = None,
                     n_predict: int | None = None) -> str:
        self.slot_ids.append(slot_id)
        return await super().query(prompt, role=role, call_id=call_id,
                                    chunk=chunk, seed=seed, n_predict=n_predict)


@pytest.fixture
def bench_cfg(minimal_cfg_dict: dict, tmp_path: Path):
    """Factory for a bench-profile Config on a tmp trace store.

    Prompt paths are absolutized (as `_episode_cfg_dict` does) so the
    registry's sha256 pins resolve regardless of the working directory.
    """

    def build(*, bench_ctx: int | None = None, bench_parallel: int | None = None,
              drop_bench_leaf: bool = False, max_wall_clock_s: int | None = None,
              max_subcalls: int | None = None, chunk: dict | None = None,
              leaf_seed: int | None = None) -> Config:
        raw = copy.deepcopy(minimal_cfg_dict)
        prompts = raw["scaffold"]["prompts"]
        prompts["root"]["path"] = str(resolve_prompt_path(Path(prompts["root"]["path"])))
        prompts["leaf_prefix"]["path"] = str(resolve_prompt_path(Path(prompts["leaf_prefix"]["path"])))
        if prompts.get("leaf_envelope"):
            prompts["leaf_envelope"]["path"] = str(
                resolve_prompt_path(Path(prompts["leaf_envelope"]["path"])))
        for ref in prompts["strategy_templates"].values():
            ref["path"] = str(REPO_ROOT / ref["path"])
        for ref in (prompts.get("baselines") or {}).values():
            ref["path"] = str(REPO_ROOT / ref["path"])
        raw["trace"]["db_path"] = str(tmp_path / "rlm.duckdb")
        raw["trace"]["blob_root"] = str(tmp_path / "blobs")
        raw["scaffold"]["dispatcher"] = "mock"
        if drop_bench_leaf:
            raw["servers"].pop("bench_leaf", None)
        else:
            if bench_ctx is not None:
                raw["servers"]["bench_leaf"]["ctx"] = bench_ctx
            if bench_parallel is not None:
                raw["servers"]["bench_leaf"]["parallel"] = bench_parallel
        if max_wall_clock_s is not None:
            raw["scaffold"]["budgets"]["max_wall_clock_s"] = max_wall_clock_s
        if max_subcalls is not None:
            raw["scaffold"]["budgets"]["max_subcalls"] = max_subcalls
        if chunk is not None:
            raw["scaffold"]["chunk"].update(chunk)
        if leaf_seed is not None:
            # §8's replicate identity. `rlm.measure.bench.seeded_config` patches this
            # per attempt on the RAW dict, exactly as this does.
            raw["scaffold"]["sampling"]["leaf"]["seed"] = leaf_seed
        return Config.model_validate(raw)

    return build


class _ArmEnv:
    """Runs one arm against a real TraceLogger and reads the rows back."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.episode: dict | None = None
        self.steps: list[dict] = []

    async def run(self, factory):
        tl = TraceLogger(self.cfg.trace.db_path, self.cfg.trace.blob_root)
        await tl.start()
        try:
            result = await factory(tl)
        finally:
            await tl.drain()
            await tl.aclose()
        self._load()
        return result

    def _load(self) -> None:
        con = duckdb.connect(str(self.cfg.trace.db_path), read_only=True)
        try:
            eps = _rows(con, "SELECT * FROM episodes ORDER BY started_at")
            self.episode = _decode_episode_row(eps[-1]) if eps else None
            self.steps = _rows(con, "SELECT * FROM steps ORDER BY step_idx")
        finally:
            con.close()

    def blob(self, rel: str) -> bytes:
        return (Path(self.cfg.trace.blob_root) / rel).read_bytes()


BENCH_EXTRA = {"run_id": "r-1", "seed": 1, "block": 0}


def _task(**over) -> Task:
    kw = {"task_id": "b1-t1", "text": "What is the key?",
          "context": "the key is ANSWER", "answer": "ANSWER", "checker": "exact"}
    kw.update(over)
    return Task(**kw)


async def _run_b1(cfg: Config, task: Task, dispatcher, **kw):
    env = _ArmEnv(cfg)
    result = await env.run(
        lambda tl: run_b1(task, cfg, dispatcher=dispatcher, trace=tl,
                           registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA,
                           **kw))
    return result, env


async def _run_b3(cfg: Config, task: Task, dispatcher, **kw):
    env = _ArmEnv(cfg)
    result = await env.run(
        lambda tl: run_b3(task, cfg, dispatcher=dispatcher, trace=tl,
                           registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA,
                           **kw))
    return result, env


#: A UUID that appears in the corpus and nowhere in the question -- R13's
#: detector only fires on identifier-SHAPED tokens (uuids, ENT codes, hex runs,
#: long mixed alphanumerics), so a corpus of ordinary prose is checked-and-clean
#: by construction and cannot exercise a hit.
LEAKED_KEY = "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"


# --------------------------------------------------------------------------- #
# truncate_head_tail
# --------------------------------------------------------------------------- #


def test_truncate_head_tail_is_5050_and_recorded():
    corpus = "A" * 1000 + "B" * 1000
    text, rec = truncate_head_tail(corpus, corpus_tokens=500, fit_tokens=250)
    assert rec["truncated"] is True
    assert abs(rec["kept_head_tokens"] - rec["kept_tail_tokens"]) <= 1
    assert rec["kept_head_tokens"] + rec["kept_tail_tokens"] == 250
    assert rec["corpus_tokens"] == 500
    assert text.startswith("A") and text.endswith("B")
    # char-proportional: 2000 chars / 500 tokens = 4 chars per token
    assert text.count("A") == 500 and text.count("B") == 500

    t2, rec2 = truncate_head_tail(corpus, corpus_tokens=200, fit_tokens=250)
    assert t2 == corpus and rec2["truncated"] is False
    assert rec2["kept_head_tokens"] == 200 and rec2["kept_tail_tokens"] == 0


def test_truncate_head_tail_is_deterministic():
    corpus = "".join(chr(97 + i % 26) for i in range(4000))
    a = truncate_head_tail(corpus, corpus_tokens=1000, fit_tokens=333)
    b = truncate_head_tail(corpus, corpus_tokens=1000, fit_tokens=333)
    assert a == b


def test_truncate_head_tail_never_duplicates_or_overruns():
    """Rounding must not let head+tail overlap -- a duplicated span would put
    text in the prompt twice and make the kept-token record a lie."""
    corpus = "x" * 101
    text, rec = truncate_head_tail(corpus, corpus_tokens=25, fit_tokens=24)
    assert len(text) <= len(corpus)
    assert rec["truncated"] is True


def test_truncate_head_tail_handles_a_zero_or_negative_fit():
    corpus = "A" * 100
    text, rec = truncate_head_tail(corpus, corpus_tokens=25, fit_tokens=0)
    assert text == "" and rec["truncated"] is True
    assert rec["kept_head_tokens"] == 0 and rec["kept_tail_tokens"] == 0
    text, rec = truncate_head_tail(corpus, corpus_tokens=25, fit_tokens=-10)
    assert text == "" and rec["truncated"] is True


def test_truncate_head_tail_tolerates_an_empty_corpus():
    text, rec = truncate_head_tail("", corpus_tokens=0, fit_tokens=-1)
    assert text == "" and rec["corpus_tokens"] == 0


# --------------------------------------------------------------------------- #
# outcome mapping (shared by B1/B2/B3)
# --------------------------------------------------------------------------- #


def test_reason_constants_match_the_episode_runner():
    """§6's outcome_reason vocabulary is one vocabulary. `arms` cannot import
    `episode` (it would drag C4 in), so the strings are pinned here instead."""
    assert CHECKER_FAILED == episodemod.CHECKER_FAILED
    assert SERVER_UNREACHABLE == episodemod.SERVER_UNREACHABLE


def test_outcome_for_error_maps_the_three_failure_classes():
    from rlm.errors import BudgetBreach

    assert outcome_for_error(BudgetBreach(Outcome.BUDGET_KILL, "wall_clock")) == (
        Outcome.BUDGET_KILL, "wall_clock")
    assert outcome_for_error(DispatchError("leaf is gone")) == (
        Outcome.ERROR, SERVER_UNREACHABLE)
    assert outcome_for_error(RuntimeError("bug")) == (Outcome.ERROR, ARM_ERROR)


def test_arm_snapshot_keys_do_not_shadow_config_fields(bench_cfg):
    """`config_snapshot` merges `extra` over the config dump, so a key that
    collides silently replaces a whole config section."""
    from rlm.measure.arms import SNAPSHOT_KEYS

    cfg = bench_cfg()
    assert not (set(SNAPSHOT_KEYS) & set(type(cfg).model_fields))


# --------------------------------------------------------------------------- #
# B1
# --------------------------------------------------------------------------- #


async def test_b1_success_logs_episode_and_step(bench_cfg):
    cfg = bench_cfg()
    task = _task()
    disp = _ScriptedDispatcher("ANSWER")
    res, env = await _run_b1(cfg, task, disp)

    assert isinstance(res, ArmResult)
    assert res.outcome == Outcome.SUCCESS and res.reason is None
    assert res.answer == "ANSWER"

    row = env.episode
    assert row is not None
    assert row["episode_id"] == res.episode_id
    assert row["outcome"] == "success"
    assert row["task_id"] == "b1-t1" and row["task_hash"] == task.task_hash
    assert row["dry_run"] is True          # scaffold.dispatcher == "mock"
    assert row["sandbox_pid"] is None      # a baseline arm runs no sandbox
    assert row["benchmark_version"] == cfg.benchmark.version

    snap = row["config_snapshot"]
    assert snap["bench"]["arm"] == "b1"
    assert snap["bench"]["run_id"] == "r-1" and snap["bench"]["seed"] == 1
    assert snap["bench"]["block"] == 0
    assert snap["bench"]["slot_id"] == 0
    assert snap["bench"]["slot_id_applied"] is False   # MockDispatcher cannot pin
    assert snap["bench"]["b1_truncation"]["truncated"] is False
    # the pinned bytes each arm actually ran (§8's no-overfit audit)
    assert "baselines.b1_single_shot.file" in snap["prompt_hashes"]
    assert snap["task"]["text"] == task.text
    assert snap["servers"]["bench_leaf"]["ctx"] == 524288   # the config is there too

    assert len(env.steps) == 1
    step = env.steps[0]
    assert step["actor"] == "leaf" and step["action_type"] == "llm_call"
    assert step["status"] == "ok" and step["depth"] == 1
    assert step["step_idx"] == 0 and step["episode_id"] == res.episode_id
    assert step["call_id"] is not None

    # the prompt is head + truncated corpus + question, in that order
    prompt = disp.prompts[0]
    head = cfg.prompt_registry().render_baseline("b1_single_shot")
    assert prompt == f"{head}\n\n{task.context}\n\n{task.text}"
    assert step["action_payload"] == prompt

    # the answer is stored, and the episode's final_answer_ref names that blob
    assert env.blob(step["observation_full_ref"]) == b"ANSWER"
    assert row["final_answer_ref"] == step["observation_full_ref"]


async def test_b1_records_the_scaffold_provenance_columns(bench_cfg):
    """§6's provenance columns. `rlm run` fills both for the RLM arm
    (`cli.py:529`); a baseline row carrying '' would make the bench grid's two
    halves unattributable to the same process and tree."""
    cfg = bench_cfg()
    res, env = await _run_b1(cfg, _task(), _ScriptedDispatcher(),
                              scaffold_instance_id="4242",
                              scaffold_git_sha="deadbeef")
    assert env.episode["scaffold_instance_id"] == "4242"
    assert env.episode["scaffold_git_sha"] == "deadbeef"


async def test_b1_runs_r13s_detector_on_the_leaf_call(bench_cfg):
    """§8 (v0.2.6): "R13's foreign-string detector runs on every leaf call
    during S4 and its hit count is reported per arm in the verdict." The column
    is TRI-STATE -- NULL means NOT CHECKED, and a NULL column has no
    denominator, so the verdict could not be written from it at all."""
    cfg = bench_cfg()
    res, env = await _run_b1(cfg, _task(), _ScriptedDispatcher("ANSWER"))
    assert res.outcome == Outcome.SUCCESS
    step = env.steps[0]
    assert step["leak_detected"] is False      # checked, and clean -- not NULL
    assert step["leak_detail"] is None


async def test_b1_detects_an_answer_from_the_truncated_away_middle(bench_cfg):
    """The index is built from the FULL corpus while the TRUNCATED document is
    what gets sent, so an identifier the model produced from the dropped middle
    is a hit -- text it was never shown. Indexing the truncated text instead
    would make the check vacuous (every indexed token would also be in `sent`)."""
    cfg = bench_cfg(bench_ctx=4096, bench_parallel=2)
    corpus = "A" * 4000 + f" {LEAKED_KEY} " + "B" * 4000
    task = _task(context=corpus, answer=LEAKED_KEY, checker="uuid_exact")
    res, env = await _run_b1(cfg, task, _ScriptedDispatcher(LEAKED_KEY))

    prompt = env.steps[0]["action_payload"]
    assert LEAKED_KEY not in prompt            # the middle really was dropped
    assert env.steps[0]["leak_detected"] is True
    assert LEAKED_KEY in env.steps[0]["leak_detail"]


async def test_b1_records_tokenized_task_len(bench_cfg):
    cfg = bench_cfg()
    task = _task(context="x" * 400)
    res, env = await _run_b1(cfg, task, _ScriptedDispatcher())
    assert env.episode["tokenized_task_len"] == 100   # (400 + 3) // 4


async def test_b1_checker_fail_is_fail_checker_failed(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b1(cfg, _task(), _ScriptedDispatcher("NOT THE ANSWER"))
    assert res.outcome == Outcome.FAIL and res.reason == CHECKER_FAILED
    assert res.answer == "NOT THE ANSWER"
    assert env.episode["outcome"] == "fail"
    assert env.episode["outcome_reason"] == "checker_failed"
    assert len(env.steps) == 1 and env.steps[0]["status"] == "ok"


async def test_b1_wall_clock_breach_is_budget_kill(bench_cfg):
    cfg = bench_cfg(max_wall_clock_s=1)
    disp = _ScriptedDispatcher("ANSWER", delay=1.2)
    res, env = await _run_b1(cfg, _task(), disp)
    assert res.outcome == Outcome.BUDGET_KILL and res.reason == "wall_clock"
    assert env.episode["outcome"] == "budget_kill"
    assert env.episode["outcome_reason"] == "wall_clock"
    # the call that overran is still ON the trace: the kill granularity is one
    # model call, not a watchdog tick, so the call completes and is recorded.
    assert len(env.steps) == 1 and env.steps[0]["status"] == "ok"


async def test_b1_dispatcher_failure_is_error_server_unreachable(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b1(cfg, _task(), _ScriptedDispatcher(fail=True))
    assert res.outcome == Outcome.ERROR and res.reason == SERVER_UNREACHABLE
    assert res.answer is None
    assert env.episode["outcome"] == "error"
    # C4's failed attempt is logged, and the episode carries no answer blob
    assert len(env.steps) == 1 and env.steps[0]["status"] == "error"
    assert env.steps[0]["observation_full_ref"] is None
    assert env.episode["final_answer_ref"] is None


async def test_b1_truncates_an_over_window_corpus_and_records_it(bench_cfg):
    # capacity = 4096 // 2 = 2048 tokens per slot; the corpus alone is 2000.
    cfg = bench_cfg(bench_ctx=4096, bench_parallel=2)
    corpus = "A" * 4000 + "B" * 4000
    task = _task(context=corpus, answer="ANSWER")
    res, env = await _run_b1(cfg, task, _ScriptedDispatcher("ANSWER"))

    assert res.outcome == Outcome.SUCCESS
    rec = env.episode["config_snapshot"]["bench"]["b1_truncation"]
    assert rec["truncated"] is True
    assert rec["corpus_tokens"] == 2000
    assert abs(rec["kept_head_tokens"] - rec["kept_tail_tokens"]) <= 1
    assert rec["kept_head_tokens"] + rec["kept_tail_tokens"] == rec["fit_tokens"]
    assert rec["slot_capacity_tokens"] == 2048
    # the head and the tail both survived, and nothing in between did
    body = env.steps[0]["action_payload"]
    assert "A" in body and "B" in body
    assert body.count("A") + body.count("B") < 8000
    assert rec["prompt_tokens"] <= 2048 - cfg.scaffold.budgets.max_predict.leaf


async def test_b1_verify_loop_shrinks_a_prompt_the_raw_count_understated(bench_cfg):
    cfg = bench_cfg(bench_ctx=4096, bench_parallel=2)
    corpus = "A" * 4000 + "B" * 4000
    task = _task(context=corpus, answer="ANSWER")
    head = cfg.prompt_registry().render_baseline("b1_single_shot")
    disp = _ScriptedDispatcher(
        "ANSWER", count_penalty=200,
        # only the ASSEMBLED prompt is penalised: it alone starts with the
        # baseline head AND carries corpus text.
        penalise=lambda t: t.startswith(head) and "AAAAAAAAAA" in t)
    res, env = await _run_b1(cfg, task, disp)

    assert res.outcome == Outcome.SUCCESS
    rec = env.episode["config_snapshot"]["bench"]["b1_truncation"]
    assert rec["verify_rounds"] >= 1
    assert rec["prompt_tokens"] <= 2048 - cfg.scaffold.budgets.max_predict.leaf


async def test_b1_refuses_a_config_with_no_bench_leaf_profile(bench_cfg):
    cfg = bench_cfg(drop_bench_leaf=True)
    tl = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root)
    await tl.start()
    try:
        with pytest.raises(ConfigError, match="bench_leaf"):
            await run_b1(_task(), cfg, dispatcher=_ScriptedDispatcher(), trace=tl,
                          registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA)
    finally:
        await tl.aclose()


async def test_b1_opens_no_episode_when_the_prompt_cannot_be_built(bench_cfg):
    """Preparation runs BEFORE `open_episode`, so a refusal never leaves a
    NULL-outcome row for crash recovery to tombstone."""
    cfg = bench_cfg(drop_bench_leaf=True)
    env = _ArmEnv(cfg)
    with pytest.raises(ConfigError):
        await env.run(lambda tl: run_b1(
            _task(), cfg, dispatcher=_ScriptedDispatcher(), trace=tl,
            registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA))
    assert env.episode is None


async def test_b1_passes_the_slot_pin_through_when_the_dispatcher_takes_it(bench_cfg):
    cfg = bench_cfg()
    disp = _SlotAwareDispatcher("ANSWER")
    res, env = await _run_b1(cfg, _task(), disp, slot_id=0)
    assert res.outcome == Outcome.SUCCESS
    assert disp.slot_ids == [0]
    assert env.episode["config_snapshot"]["bench"]["slot_id_applied"] is True


async def test_b1_records_the_slot_pin_as_UNAPPLIED_when_it_is_dropped(bench_cfg):
    """A pin the dispatcher cannot take is a NO-OP, and recording it as if it
    had been applied would silently void §8's v0.2.6 obligation that B1 and B3
    each take their own slot. Two facts, both recorded.

    `steps.slot_id` is left to C4 either way: that column means the slot the
    server actually served on (an out-of-range request is silently reassigned
    with HTTP 200, which is why `SlotMismatch` exists)."""
    cfg = bench_cfg()
    res, env = await _run_b1(cfg, _task(), _ScriptedDispatcher(), slot_id=7)
    assert env.steps[0]["slot_id"] is None
    bench = env.episode["config_snapshot"]["bench"]
    assert bench["slot_id"] == 7 and bench["slot_id_applied"] is False


async def test_bench_slot_capacity_is_ctx_over_parallel(bench_cfg):
    cfg = bench_cfg()
    assert bench_slot_capacity(cfg) == 524288 // 2
    with pytest.raises(ConfigError, match="bench_leaf"):
        bench_slot_capacity(bench_cfg(drop_bench_leaf=True))


# --------------------------------------------------------------------------- #
# bm25_select -- §8's pre-registered B3 selection rule
# --------------------------------------------------------------------------- #


def test_bm25_select_ranks_the_rare_term_chunk_first():
    """A chunk sharing no term with the question scores NULL in DuckDB's FTS
    and is excluded from `ranked` entirely -- absent, not merely last."""
    chunks = ["the quick brown fox jumps over the lazy dog",
              "zylophonic quasar emits an unusual signature",
              "another ordinary sentence with common words"]
    question = "what does the zylophonic quasar do"
    selected, record = bm25_select(chunks, question, budget_tokens=1000,
                                    token_counts=[10, 10, 10])
    assert record["ranked"] == [1]
    assert selected == [1]
    assert record["selected"] == [1]
    assert record["budget_tokens"] == 1000
    assert record["fts"] is True


def test_bm25_select_restores_original_order_even_when_rank_order_differs():
    """Chunk 2 outranks chunk 0 (more occurrences of the matched term), but
    both fit the budget, so the returned selection must come back in ORIGINAL
    corpus order -- not BM25 rank order -- for the prompt to stay coherent."""
    chunks = ["needle appears once here",
              "no relevant terms in this filler sentence",
              "needle needle needle needle"]
    selected, record = bm25_select(chunks, "needle", budget_tokens=1000,
                                    token_counts=[5, 5, 5])
    assert record["ranked"] == [2, 0]          # rank order: 2 scores higher
    assert selected == [0, 2]                  # return order: original
    assert record["selected"] == [0, 2]


def test_bm25_select_stops_at_the_first_chunk_that_does_not_fit():
    """The pre-registered rule is STOP, not skip-and-continue: chunk 1 is
    ranked second and does not fit, so chunk 2 -- ranked third but small
    enough to fit -- must NOT be picked up after it."""
    chunks = ["needle needle needle", "needle needle", "needle"]
    selected, record = bm25_select(chunks, "needle", budget_tokens=10,
                                    token_counts=[5, 100, 5])
    assert record["ranked"] == [0, 1, 2]
    assert selected == [0]                     # NOT [0, 2]
    assert record["selected"] == [0]


def test_bm25_select_is_deterministic_across_two_calls():
    chunks = ["needle appears once here",
              "no relevant terms in this filler sentence",
              "needle needle needle needle"]
    a = bm25_select(chunks, "needle", budget_tokens=1000, token_counts=[5, 5, 5])
    b = bm25_select(chunks, "needle", budget_tokens=1000, token_counts=[5, 5, 5])
    assert a == b


def test_bm25_select_refuses_mismatched_chunks_and_token_counts():
    with pytest.raises(ValueError, match="token_counts"):
        bm25_select(["a", "b"], "q", budget_tokens=10, token_counts=[1])


# --------------------------------------------------------------------------- #
# B3
# --------------------------------------------------------------------------- #


async def test_b3_success_logs_episode_and_step(bench_cfg):
    cfg = bench_cfg()
    task = _task()
    disp = _ScriptedDispatcher("ANSWER")
    res, env = await _run_b3(cfg, task, disp)

    assert isinstance(res, ArmResult)
    assert res.outcome == Outcome.SUCCESS and res.reason is None
    assert res.answer == "ANSWER"

    row = env.episode
    assert row is not None
    assert row["episode_id"] == res.episode_id
    assert row["outcome"] == "success"
    assert row["task_id"] == "b1-t1" and row["task_hash"] == task.task_hash
    assert row["dry_run"] is True          # scaffold.dispatcher == "mock"
    assert row["sandbox_pid"] is None      # a baseline arm runs no sandbox
    assert row["benchmark_version"] == cfg.benchmark.version

    snap = row["config_snapshot"]
    assert snap["bench"]["arm"] == "b3"
    assert snap["bench"]["run_id"] == "r-1" and snap["bench"]["seed"] == 1
    assert snap["bench"]["block"] == 0
    assert snap["bench"]["slot_id"] == 1                # B3's pre-registered pin
    assert snap["bench"]["slot_id_applied"] is False     # MockDispatcher cannot pin
    # bm25_select's record, written verbatim into the bench snapshot
    assert snap["bench"]["fts"] is True
    assert snap["bench"]["ranked"] == [0]
    assert snap["bench"]["selected"] == [0]
    assert isinstance(snap["bench"]["budget_tokens"], int)
    assert snap["bench"]["budget_tokens"] > 0
    # the pinned bytes each arm actually ran (§8's no-overfit audit)
    assert "baselines.b3_single_shot.file" in snap["prompt_hashes"]
    assert snap["task"]["text"] == task.text

    assert len(env.steps) == 1
    step = env.steps[0]
    assert step["actor"] == "leaf" and step["action_type"] == "llm_call"
    assert step["status"] == "ok" and step["depth"] == 1
    assert step["step_idx"] == 0 and step["episode_id"] == res.episode_id
    assert step["call_id"] is not None
    assert step["leak_detected"] is False      # R13 checked, and clean -- not NULL

    # the prompt is head + selected chunk(s) (original order) + question
    prompt = disp.prompts[0]
    head = cfg.prompt_registry().render_baseline("b3_single_shot")
    assert prompt == f"{head}\n\n{task.context}\n\n{task.text}"
    assert step["action_payload"] == prompt

    # the answer is stored, and the episode's final_answer_ref names that blob
    assert env.blob(step["observation_full_ref"]) == b"ANSWER"
    assert row["final_answer_ref"] == step["observation_full_ref"]


async def test_b3_records_the_scaffold_provenance_columns(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b3(cfg, _task(), _ScriptedDispatcher(),
                              scaffold_instance_id="4242",
                              scaffold_git_sha="deadbeef")
    assert env.episode["scaffold_instance_id"] == "4242"
    assert env.episode["scaffold_git_sha"] == "deadbeef"


async def test_b3_runs_r13s_detector_on_an_unselected_chunk(bench_cfg):
    """§8 (v0.2.6): R13's detector runs on every leaf call, indexed against
    EVERY chunk C2 produced -- not just the one(s) `bm25_select` kept. Chunk
    A (below) shares no term with the question and is therefore never sent;
    chunk B is on-topic and is the whole prompt. An answer that reproduces
    chunk A's identifier could only have come from cross-slot leakage -- the
    same story `run_b1`'s truncated-middle test tells for B1's overflow
    policy, told here for B3's selection rule instead.

    `size_tokens=50` (`4*50-3 = 197` chars) is chosen so C2's char-count
    binary search (content-agnostic under `MockDispatcher`'s `(len+3)//4`)
    cuts EXACTLY at the end of chunk A, regardless of chunk B's content --
    verified directly against `rlm.context.chunker.split` before being relied on here.
    """
    cfg = bench_cfg(chunk={"size_tokens": 50, "stride_tokens": 50,
                            "snap_to_boundary": False})
    chunk_a = (f"archive record about old relics references item {LEAKED_KEY} "
               "filed long ago").ljust(197, "q")
    chunk_b = ("the target phrase asked about right now is TARGETPHRASE and "
               "that is the whole point of this second passage")
    corpus = chunk_a + chunk_b
    task = _task(context=corpus, text="What is the target phrase?",
                 answer=LEAKED_KEY, checker="uuid_exact")
    res, env = await _run_b3(cfg, task, _ScriptedDispatcher(LEAKED_KEY))

    assert res.outcome == Outcome.SUCCESS
    step = env.steps[0]
    prompt = step["action_payload"]
    assert LEAKED_KEY not in prompt            # chunk A really was never sent
    assert "TARGETPHRASE" in prompt            # chunk B is what got sent
    assert step["leak_detected"] is True
    assert LEAKED_KEY in step["leak_detail"]

    bench = env.episode["config_snapshot"]["bench"]
    assert bench["ranked"] == [1]              # chunk A never even matched
    assert bench["selected"] == [1]


async def test_b3_checker_fail_is_fail_checker_failed(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b3(cfg, _task(), _ScriptedDispatcher("NOT THE ANSWER"))
    assert res.outcome == Outcome.FAIL and res.reason == CHECKER_FAILED
    assert res.answer == "NOT THE ANSWER"
    assert env.episode["outcome"] == "fail"
    assert env.episode["outcome_reason"] == "checker_failed"


async def test_b3_dispatcher_failure_is_error_server_unreachable(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b3(cfg, _task(), _ScriptedDispatcher(fail=True))
    assert res.outcome == Outcome.ERROR and res.reason == SERVER_UNREACHABLE
    assert res.answer is None
    assert env.episode["outcome"] == "error"
    assert env.episode["final_answer_ref"] is None


async def test_b3_refuses_a_config_with_no_bench_leaf_profile(bench_cfg):
    cfg = bench_cfg(drop_bench_leaf=True)
    tl = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root)
    await tl.start()
    try:
        with pytest.raises(ConfigError, match="bench_leaf"):
            await run_b3(_task(), cfg, dispatcher=_ScriptedDispatcher(), trace=tl,
                          registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA)
    finally:
        await tl.aclose()


async def test_b3_opens_no_episode_when_the_prompt_cannot_be_built(bench_cfg):
    cfg = bench_cfg(drop_bench_leaf=True)
    env = _ArmEnv(cfg)
    with pytest.raises(ConfigError):
        await env.run(lambda tl: run_b3(
            _task(), cfg, dispatcher=_ScriptedDispatcher(), trace=tl,
            registry=cfg.prompt_registry(), bench_extra=BENCH_EXTRA))
    assert env.episode is None


async def test_b3_passes_the_slot_pin_through_when_the_dispatcher_takes_it(bench_cfg):
    cfg = bench_cfg()
    disp = _SlotAwareDispatcher("ANSWER")
    res, env = await _run_b3(cfg, _task(), disp)
    assert res.outcome == Outcome.SUCCESS
    assert disp.slot_ids == [1]                # B3's pre-registered pin (slot 1)
    assert env.episode["config_snapshot"]["bench"]["slot_id_applied"] is True


async def test_b3_records_the_slot_pin_as_UNAPPLIED_when_it_is_dropped(bench_cfg):
    cfg = bench_cfg()
    res, env = await _run_b3(cfg, _task(), _ScriptedDispatcher(), slot_id=7)
    assert env.steps[0]["slot_id"] is None
    bench = env.episode["config_snapshot"]["bench"]
    assert bench["slot_id"] == 7 and bench["slot_id_applied"] is False


# --------------------------------------------------------------------------- #
# ArmEpisode -- the plumbing B2/B3 reuse
# --------------------------------------------------------------------------- #


async def test_arm_episode_ids_are_real_uuids(bench_cfg):
    """The DuckDB column is UUID-typed and the writer loop swallows conversion
    failures, so a malformed id silently loses every write for that episode."""
    import uuid

    cfg = bench_cfg()
    ep = ArmEpisode(_task(), cfg, dispatcher=_ScriptedDispatcher(), trace=None,
                     registry=cfg.prompt_registry(), arm="b1", bench_extra={})
    assert uuid.UUID(ep.episode_id).version == 4


async def test_arm_episode_outcome_for_answer(bench_cfg):
    cfg = bench_cfg()
    ep = ArmEpisode(_task(), cfg, dispatcher=None, trace=None,
                     registry=cfg.prompt_registry(), arm="b1", bench_extra={})
    assert ep.outcome_for_answer("ANSWER") == (Outcome.SUCCESS, None)
    assert ep.outcome_for_answer("nope") == (Outcome.FAIL, CHECKER_FAILED)
    assert ep.outcome_for_answer(None) == (Outcome.FAIL, NO_ANSWER)


async def test_arm_episode_step_helper_is_idempotent_and_allocates_in_order(bench_cfg):
    """`log_call` is called on both the success and the failure path, so it
    must never double-write an attempt (C4's contract, `episode.py:720-757`)."""
    cfg = bench_cfg()
    disp = _ScriptedDispatcher("ANSWER")
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b1",
                         bench_extra=BENCH_EXTRA)
        ep.start_clock()
        ep.open_episode()
        answer = await ep.call_leaf("q1")
        ep.log_call(disp.steps[-1]["call_id"], "q1", answer=answer)   # replay
        answer2 = await ep.call_leaf("q2")
        return ep.close(*ep.outcome_for_answer(answer2), answer=answer2)

    res = await env.run(factory)
    assert res.outcome == Outcome.SUCCESS
    assert [s["step_idx"] for s in env.steps] == [0, 1]
    assert [s["action_payload"] for s in env.steps] == ["q1", "q2"]
    assert all(s["actor"] == "leaf" and s["action_type"] == "llm_call"
               for s in env.steps)


async def test_arm_episode_close_is_idempotent(bench_cfg):
    cfg = bench_cfg()
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=_ScriptedDispatcher(), trace=tl,
                         registry=cfg.prompt_registry(), arm="b1", bench_extra={})
        ep.open_episode()
        first = ep.close(Outcome.FAIL, CHECKER_FAILED)
        second = ep.close(Outcome.SUCCESS, None)
        return first, second

    first, second = await env.run(factory)
    assert first is second
    assert env.episode["outcome"] == "fail"


# --------------------------------------------------------------------------- #
# ArmEpisode.call_leaf -- the admission plumbing this task adds
# --------------------------------------------------------------------------- #


async def test_call_leaf_without_admit_tokens_does_not_touch_the_enforcer(bench_cfg):
    """B1/B3 never pass `admit_tokens` -- their existing behaviour (no
    admission at all) must be unchanged by this task's addition."""
    cfg = bench_cfg()
    env = _ArmEnv(cfg)
    holder: dict = {}

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=_ScriptedDispatcher("ANSWER"),
                         trace=tl, registry=cfg.prompt_registry(), arm="b1",
                         bench_extra=BENCH_EXTRA)
        holder["ep"] = ep
        ep.start_clock()
        ep.open_episode()
        answer = await ep.call_leaf("q1")
        return ep.finish(answer)

    res = await env.run(factory)
    assert res.outcome == Outcome.SUCCESS
    assert holder["ep"].enforcer.subcalls_used == 0
    assert holder["ep"].enforcer.tokens_used == 0


async def test_call_leaf_admits_and_settles_against_the_enforcer(bench_cfg):
    cfg = bench_cfg()
    env = _ArmEnv(cfg)
    holder: dict = {}

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=_ScriptedDispatcher("ANSWER"),
                         trace=tl, registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA)
        holder["ep"] = ep
        ep.start_clock()
        ep.open_episode()
        answer = await ep.call_leaf("q1", admit_tokens=10)
        return ep.finish(answer)

    res = await env.run(factory)
    assert res.outcome == Outcome.SUCCESS
    ep = holder["ep"]
    assert ep.enforcer.subcalls_used == 1
    # the mock dispatcher records no token usage -- settle with zeros is fine
    assert ep.enforcer.tokens_used == 0
    assert ep.enforcer.reserved_total == 0     # released by settle


async def test_call_leaf_admit_breach_raises_before_dispatch_and_logs_nothing(bench_cfg):
    cfg = bench_cfg(max_subcalls=1)
    env = _ArmEnv(cfg)
    disp = _ScriptedDispatcher("ANSWER")

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA)
        ep.start_clock()
        ep.open_episode()
        await ep.call_leaf("q1", admit_tokens=10)     # spends the only subcall
        try:
            await ep.call_leaf("q2", admit_tokens=10)
        except BudgetBreach as exc:
            return ep.close(*outcome_for_error(exc))
        raise AssertionError("expected a BudgetBreach on the second call")

    res = await env.run(factory)
    assert res.outcome == Outcome.BUDGET_KILL and res.reason == "max_subcalls"
    # the breached call never dispatched, so only the FIRST call's step exists
    assert len(env.steps) == 1
    assert env.steps[0]["action_payload"] == "q1"


def test_settled_tokens_matches_the_episode_runners_implementation():
    """`arms._settled_tokens` is `src/rlm/episode.py::settled_tokens`, duplicated
    (the dependency rule -- see the module docstring). Pinned equal so the
    duplication cannot drift in silence."""
    from rlm.measure.arms import _settled_tokens

    attempts = [{"tokens_in": 10, "tokens_out": 5},
                {"tokens_in": None, "tokens_out": None},
                {"tokens_in": 3, "tokens_out": 2}]
    assert _settled_tokens(attempts) == episodemod.settled_tokens(attempts)


def test_strip_reasoning_matches_rootclients_implementation():
    """`arms._strip_reasoning` is `rlm.serve.rootclient.strip_reasoning`, duplicated
    for the same import-rule reason (`arms.py` may not import `rlm.serve.rootclient`
    -- `checks/test_import_rules.py`). Pinned equal here."""
    from rlm.measure.arms import _strip_reasoning
    from rlm.serve.rootclient import strip_reasoning

    samples = [
        "<think>\nreasoning\n</think>\nFINAL",
        "no think block here",
        "<think></think>FINAL",
        "FINAL<think>reopened but never closed",
        "  <think>a</think>  <think>b</think>  tail",
    ]
    for s in samples:
        assert _strip_reasoning(s) == strip_reasoning(s)


# --------------------------------------------------------------------------- #
# B2 -- deterministic map-reduce
# --------------------------------------------------------------------------- #


@pytest.fixture
async def b2_root_client(fake_root_server):
    """A real `ServerClient` pointed at `FakeRootServer`, injected as B2's
    `root_client` -- exactly how a real bench run injects a `ServerClient`
    built against `servers.root.port`. `arms.py` never constructs its own
    (the dependency rule: it may not import `rlm.serve.dispatcher`)."""
    from rlm.serve.dispatcher import ServerClient

    client = ServerClient(fake_root_server.base_url, timeout=5.0)
    yield client
    await client.aclose()


class _PerChunkDispatcher(CannedDispatcher):
    """Answers each B2 leaf summary call with a reply DERIVED from the
    prompt (via `reply_for`), or an empty string when `empty=True`, so a test
    can verify the reduce step's ordering/content from the dispatcher's own
    record instead of trusting it. Also spies on `set_corpus`."""

    def __init__(self, reply_for=None, *, empty: bool = False) -> None:
        super().__init__()
        self.reply_for = reply_for
        self.empty = empty
        self.prompts: list[str] = []
        self.set_corpus_calls: list[Any] = []
        self.seeds: list[int | None] = []
        self.n_predicts: list[int | None] = []

    def set_corpus(self, chunks) -> None:
        self.set_corpus_calls.append(chunks)
        self._inner.set_corpus(chunks)

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, seed: int | None = None,
                     n_predict: int | None = None) -> str:
        self.seeds.append(seed)
        self.n_predicts.append(n_predict)
        from rlm.serve.dispatcher import compose_leaf_user

        self.prompts.append(prompt)
        composed = compose_leaf_user(prompt, chunk)
        key = f"{role}:{hashlib.sha256(composed.encode('utf-8')).hexdigest()}"
        reply = "" if self.empty else (self.reply_for(prompt) if self.reply_for
                                        else "SUMMARY")
        self._inner._fixtures[key] = reply
        return await self._inner.query(prompt, role=role, call_id=call_id,
                                        chunk=chunk)


class _SlotAwarePerChunkDispatcher(_PerChunkDispatcher):
    """A dispatcher whose `query` DECLARES `slot_id` -- B2 must never pass
    it (unlike B1/B3, it runs on C4's never-reuse slot discipline and lets
    C4 assign)."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.slot_ids: list[int | None] = []

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, slot_id: int | None = None,
                     seed: int | None = None,
                     n_predict: int | None = None) -> str:
        self.slot_ids.append(slot_id)
        return await super().query(prompt, role=role, call_id=call_id,
                                    chunk=chunk, seed=seed, n_predict=n_predict)


def _reply_by_marker(prompt: str) -> str:
    if "AAAA" in prompt:
        return "SUMMARY-A"
    if "BBBB" in prompt:
        return "SUMMARY-B"
    if "CCCC" in prompt:
        return "SUMMARY-C"
    return "SUMMARY-?"


#: C2's partition chunker (size_tokens=10, snap off) cuts this 94-char corpus
#: into EXACTLY three windows -- [37, 37, 20] chars -- verified directly
#: against `rlm.context.chunker.split` before being relied on here, the same
#: discipline `test_b3_runs_r13s_detector_on_an_unselected_chunk` documents.
_B2_CORPUS = "A" * 37 + "B" * 37 + "C" * 20
_B2_CHUNKS = ["A" * 37, "B" * 37, "C" * 20]


def _b2_task(**over) -> Task:
    kw = {"task_id": "b2-t1", "text": "What is the answer?",
          "context": _B2_CORPUS, "answer": "FINAL", "checker": "exact"}
    kw.update(over)
    return Task(**kw)


def _b2_cfg(bench_cfg, **over):
    return bench_cfg(chunk={"size_tokens": 10, "stride_tokens": 10,
                             "snap_to_boundary": False}, **over)


async def _run_b2(cfg: Config, task: Task, dispatcher, root_client, **kw):
    env = _ArmEnv(cfg)
    result = await env.run(
        lambda tl: run_b2(task, cfg, dispatcher=dispatcher, root_client=root_client,
                           trace=tl, registry=cfg.prompt_registry(),
                           bench_extra=BENCH_EXTRA, **kw))
    return result, env


# -- summary budget formula (pure, no dispatcher needed) --------------------- #


def test_b2_summary_n_predict_is_sized_to_fit_the_root_window(bench_cfg):
    cfg = bench_cfg()
    assert cfg.scaffold.root.window_tokens == 32768
    assert b2_summary_n_predict(cfg, 302) == 86


def test_b2_summary_n_predict_floors_at_16(bench_cfg):
    cfg = bench_cfg()
    assert b2_summary_n_predict(cfg, 100_000) == 16


def test_b2_summary_n_predict_caps_at_leaf_max_predict(bench_cfg):
    cfg = bench_cfg()
    assert cfg.scaffold.budgets.max_predict.leaf == 512
    assert b2_summary_n_predict(cfg, 1) == 512


# -- serial map -> reduce ----------------------------------------------------- #


async def test_b2_serial_map_then_reduce_logs_three_leaf_steps_then_a_root_step(
        bench_cfg, fake_root_server, b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    task = _b2_task()
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, task, disp, b2_root_client)

    assert isinstance(res, ArmResult)
    assert res.outcome == Outcome.SUCCESS and res.reason is None
    assert res.answer == "FINAL"

    assert len(env.steps) == 4
    assert [s["step_idx"] for s in env.steps] == [0, 1, 2, 3]
    leaf_steps, root_step = env.steps[:3], env.steps[3]
    assert all(s["actor"] == "leaf" and s["action_type"] == "llm_call"
               and s["status"] == "ok" for s in leaf_steps)
    assert root_step["actor"] == "root" and root_step["action_type"] == "llm_call"
    assert root_step["status"] == "ok"

    head = cfg.prompt_registry().render_baseline("b2_leaf_summary")
    assert disp.prompts == [f"{head}\n\n{c}" for c in _B2_CHUNKS]

    root_prompt = fake_root_server.last_completion_prompt
    assert "1. SUMMARY-A" in root_prompt
    assert "2. SUMMARY-B" in root_prompt
    assert "3. SUMMARY-C" in root_prompt
    assert (root_prompt.index("SUMMARY-A") < root_prompt.index("SUMMARY-B")
            < root_prompt.index("SUMMARY-C"))
    assert task.text in root_prompt

    snap = env.episode["config_snapshot"]
    assert snap["bench"]["arm"] == "b2"
    assert snap["bench"]["run_id"] == "r-1" and snap["bench"]["seed"] == 1
    assert snap["bench"]["n_chunks"] == 3
    assert isinstance(snap["bench"]["summary_n_predict"], int)
    assert "baselines.b2_leaf_summary.file" in snap["prompt_hashes"]
    assert "baselines.b2_root_final.file" in snap["prompt_hashes"]

    # the root's answer blob is the episode's final answer
    assert env.blob(root_step["observation_full_ref"]) == b"FINAL"
    assert env.episode["final_answer_ref"] == root_step["observation_full_ref"]


async def test_b2_calls_set_corpus_with_the_chunk_list(bench_cfg, fake_root_server,
                                                          b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    await _run_b2(cfg, _b2_task(), disp, b2_root_client)

    assert disp.set_corpus_calls == [_B2_CHUNKS]


async def test_b2_never_pins_a_slot(bench_cfg, fake_root_server, b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _SlotAwarePerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    res, _env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)
    assert res.outcome == Outcome.SUCCESS
    assert disp.slot_ids == [None, None, None]


async def test_b2_empty_summary_becomes_the_no_summary_literal(
        bench_cfg, fake_root_server, b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(empty=True)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)

    assert res.outcome == Outcome.SUCCESS
    root_prompt = fake_root_server.last_completion_prompt
    assert root_prompt.count(NO_SUMMARY) == 3
    # the arm never crashes on an empty summary -- every leaf step still ok
    assert all(s["status"] == "ok" for s in env.steps[:3])


async def test_b2_root_call_strips_reasoning_before_scoring(
        bench_cfg, fake_root_server, b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["<think>\nlet me think\n</think>\nFINAL"]

    res, _env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)

    assert res.outcome == Outcome.SUCCESS
    assert res.answer == "FINAL"


# -- budget / outcome mapping ------------------------------------------------- #


async def test_b2_max_subcalls_breach_is_budget_kill(bench_cfg, fake_root_server,
                                                        b2_root_client):
    cfg = _b2_cfg(bench_cfg, max_subcalls=2)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)

    assert res.outcome == Outcome.BUDGET_KILL and res.reason == "max_subcalls"
    assert env.episode["outcome"] == "budget_kill"
    assert env.episode["outcome_reason"] == "max_subcalls"
    # two chunks were admitted and dispatched before the third breached; the
    # map never finished, so the root call never ran
    assert len(env.steps) == 2
    assert all(s["actor"] == "leaf" for s in env.steps)


async def test_b2_checker_pass_is_success(bench_cfg, fake_root_server, b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(answer="FINAL"), disp, b2_root_client)
    assert res.outcome == Outcome.SUCCESS and res.reason is None
    assert env.episode["outcome"] == "success"


async def test_b2_checker_fail_is_fail_checker_failed(bench_cfg, fake_root_server,
                                                         b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["WRONG"]

    res, env = await _run_b2(cfg, _b2_task(answer="FINAL"), disp, b2_root_client)
    assert res.outcome == Outcome.FAIL and res.reason == CHECKER_FAILED
    assert res.answer == "WRONG"
    assert env.episode["outcome"] == "fail"
    assert env.episode["outcome_reason"] == "checker_failed"


async def test_b2_records_the_scaffold_provenance_columns(bench_cfg, fake_root_server,
                                                             b2_root_client):
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    _res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client,
                               scaffold_instance_id="4242", scaffold_git_sha="deadbeef")
    assert env.episode["scaffold_instance_id"] == "4242"
    assert env.episode["scaffold_git_sha"] == "deadbeef"


# --------------------------------------------------------------------------- #
# Rotation (spec §5 C4, this task's second addition) -- B2's map can drain the
# leaf's never-reuse slot pool (`--parallel` 128 against ~300 aggregation
# windows), and without a rotation every such task would end `error` for a
# reason that has nothing to do with the task -- a manufactured §8
# contamination-class loss. B1/B3 make one call each and never exercise this
# path (their `process_manager` stays unset).
# --------------------------------------------------------------------------- #


def test_rotation_vocab_matches_the_episode_runner():
    """DUPLICATED FROM `src/rlm/episode.py` ON PURPOSE (the module docstring's
    "THIS MODULE NEVER IMPORTS C4") -- pinned equal so the vocabulary cannot
    drift in silence, the same discipline `test_reason_constants_match_the_
    episode_runner` already applies to `CHECKER_FAILED`/`SERVER_UNREACHABLE`."""
    assert SLOT_POOL_EXHAUSTED == episodemod.SLOT_POOL_EXHAUSTED
    assert ROTATION_FAILED == episodemod.ROTATION_FAILED


def test_outcome_for_error_maps_slot_pool_exhausted_and_rotation_failed():
    assert outcome_for_error(SlotPoolExhausted("x")) == (
        Outcome.ERROR, SLOT_POOL_EXHAUSTED)
    assert outcome_for_error(ServerRotationError("x")) == (
        Outcome.ERROR, ROTATION_FAILED)


class _RotatingDispatcher(_PerChunkDispatcher):
    """A `_PerChunkDispatcher` whose GLOBAL `call_count`-th `.query()`
    invocation raises `SlotPoolExhausted` (`exhaust_at`) or a plain
    `DispatchError` (`error_at`) instead of dispatching, and whose rotation
    surface (`rotating()`, `rotate_pool()`, `restart_required`,
    `pool_error_drained`) is close enough to `LLMDispatcher`'s to drive
    `ArmEpisode._rotate_leaf` for real -- retry_idx continuation included:
    `_new_step`'s contract is that a re-dispatched call_id gets the NEXT
    retry_idx, not 0 again (`src/rlm/serve/dispatcher.py:_retry_base`), and
    `MockDispatcher.query` always writes 0, so this patches the successful
    retry's step the way the real dispatcher's `_retry_base` would have
    produced it -- getting this wrong would make `log_call`'s
    `(call_id, retry_idx)` idempotency key silently drop the answering
    attempt as a "duplicate" of the exhausted one.
    """

    def __init__(self, reply_for=None, *, exhaust_at: int | None = None,
                 error_at: int | None = None, error_drained: bool = False) -> None:
        super().__init__(reply_for)
        self.exhaust_at = exhaust_at
        self.error_at = error_at
        self.error_drained_flag = error_drained
        self.call_count = 0
        self.rotate_pool_calls = 0
        self.rotating_calls = 0
        self.restart_required = False
        self.pool_error_drained = False

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, seed: int | None = None,
                     n_predict: int | None = None) -> str:
        from rlm.serve.dispatcher import _new_step

        self.call_count += 1
        retry_idx = len([s for s in self._inner.steps if s.get("call_id") == call_id])
        if self.error_at is not None and self.call_count == self.error_at:
            step = _new_step(call_id, retry_idx, role)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = "server fault (test)"
            self._inner.steps.append(step)
            raise DispatchError("server fault (test)")
        if self.exhaust_at is not None and self.call_count == self.exhaust_at:
            step = _new_step(call_id, retry_idx, role)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = "leaf slot pool exhausted (test)"
            self._inner.steps.append(step)
            self.restart_required = True
            self.pool_error_drained = self.error_drained_flag
            raise SlotPoolExhausted("leaf slot pool exhausted (test)")
        answer = await super().query(prompt, role=role, call_id=call_id, chunk=chunk)
        self._inner.steps[-1]["retry_idx"] = retry_idx
        self.restart_required = False
        return answer

    def rotating(self):
        self.rotating_calls += 1
        return self._rotating_cm()

    @contextlib.asynccontextmanager
    async def _rotating_cm(self):
        yield

    def rotate_pool(self) -> None:
        self.rotate_pool_calls += 1
        self.restart_required = False


class _FakeProcessManager:
    """The `rlm.serve.serverproc.ProcessManager` duck type: one method,
    `.restart()`. `fail=True` makes it raise `ServerRotationError`, the same
    exception a real `LlamaServerProcess.restart()` raises when the
    replacement never comes up."""

    def __init__(self, *, fail: bool = False) -> None:
        self.restart_calls = 0
        self.fail = fail

    async def restart(self) -> None:
        self.restart_calls += 1
        if self.fail:
            raise ServerRotationError("could not restart (test)")


# -- ArmEpisode.call_leaf: the rotation mechanics, isolated from run_b2 ------ #


async def test_call_leaf_rotates_once_on_slot_pool_exhausted_with_a_process_manager(
        bench_cfg):
    cfg = bench_cfg()
    disp = _RotatingDispatcher(exhaust_at=1)
    pm = _FakeProcessManager()
    env = _ArmEnv(cfg)
    holder: dict = {}

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA, process_manager=pm)
        holder["ep"] = ep
        ep.start_clock()
        ep.open_episode()
        answer = await ep.call_leaf("q1")
        return ep.close(Outcome.SUCCESS, None, answer=answer)

    res = await env.run(factory)
    assert res.outcome == Outcome.SUCCESS and res.answer == "SUMMARY"
    ep = holder["ep"]
    assert ep.rotations == 1
    assert pm.restart_calls == 1
    assert disp.rotate_pool_calls == 1 and disp.rotating_calls == 1

    assert len(env.steps) == 2                # the exhausted attempt + the retry
    assert env.steps[0]["status"] == "error"
    assert env.steps[0]["server_rotation"] == 1   # stamped on the TRIGGERING attempt
    assert env.steps[1]["status"] == "ok"
    assert env.steps[1]["server_rotation"] is None
    assert env.steps[0]["call_id"] == env.steps[1]["call_id"]      # same logical call


async def test_call_leaf_without_a_process_manager_slot_pool_exhaustion_is_clean(
        bench_cfg):
    cfg = bench_cfg()
    disp = _RotatingDispatcher(exhaust_at=1)
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA)     # no process_manager
        ep.start_clock()
        ep.open_episode()
        try:
            await ep.call_leaf("q1")
        except SlotPoolExhausted as exc:
            return ep.close(*outcome_for_error(exc))
        raise AssertionError("expected SlotPoolExhausted")

    res = await env.run(factory)
    assert res.outcome == Outcome.ERROR and res.reason == SLOT_POOL_EXHAUSTED
    assert disp.rotating_calls == 0 and disp.rotate_pool_calls == 0
    # the exhausted attempt is still on the trace -- a refusal still happened
    assert len(env.steps) == 1 and env.steps[0]["status"] == "error"


async def test_call_leaf_rotation_refused_when_the_pool_is_error_drained(bench_cfg):
    """§5 C4: a generation that answered NOTHING is a FAILED server wearing
    pool exhaustion's exception, not a healthy one that ran out of windows --
    never restarted, regardless of an injected `process_manager`."""
    cfg = bench_cfg()
    disp = _RotatingDispatcher(exhaust_at=1, error_drained=True)
    pm = _FakeProcessManager()
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA, process_manager=pm)
        ep.start_clock()
        ep.open_episode()
        try:
            await ep.call_leaf("q1")
        except SlotPoolExhausted as exc:
            return ep.close(*outcome_for_error(exc))
        raise AssertionError("expected SlotPoolExhausted")

    res = await env.run(factory)
    assert res.outcome == Outcome.ERROR and res.reason == SLOT_POOL_EXHAUSTED
    assert pm.restart_calls == 0        # never attempted -- the pool is error-drained


async def test_call_leaf_never_rotates_on_a_non_exhaustion_dispatch_error(bench_cfg):
    """§5's rule, checked directly: rotation fires ONLY on `SlotPoolExhausted`,
    never on a plain `DispatchError` -- a FAILED server is never restarted."""
    cfg = bench_cfg()
    disp = _RotatingDispatcher(error_at=1)
    pm = _FakeProcessManager()
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA, process_manager=pm)
        ep.start_clock()
        ep.open_episode()
        try:
            await ep.call_leaf("q1")
        except DispatchError as exc:
            return ep.close(*outcome_for_error(exc))
        raise AssertionError("expected a plain DispatchError")

    res = await env.run(factory)
    assert res.outcome == Outcome.ERROR and res.reason == SERVER_UNREACHABLE
    assert pm.restart_calls == 0
    assert disp.rotating_calls == 0 and disp.rotate_pool_calls == 0


async def test_call_leaf_rotation_failure_is_a_dedicated_outcome_reason(bench_cfg):
    cfg = bench_cfg()
    disp = _RotatingDispatcher(exhaust_at=1)
    pm = _FakeProcessManager(fail=True)
    env = _ArmEnv(cfg)

    async def factory(tl):
        ep = ArmEpisode(_task(), cfg, dispatcher=disp, trace=tl,
                         registry=cfg.prompt_registry(), arm="b2",
                         bench_extra=BENCH_EXTRA, process_manager=pm)
        ep.start_clock()
        ep.open_episode()
        try:
            await ep.call_leaf("q1")
        except ServerRotationError as exc:
            return ep.close(*outcome_for_error(exc))
        raise AssertionError("expected ServerRotationError")

    res = await env.run(factory)
    assert res.outcome == Outcome.ERROR and res.reason == ROTATION_FAILED
    assert pm.restart_calls == 1
    assert disp.rotating_calls == 1     # the gate WAS closed
    assert disp.rotate_pool_calls == 0  # never reached -- restart failed first


# -- run_b2: the end-to-end aggregation-corpus scenario this task exists for  #


async def test_b2_completes_an_episode_that_needed_a_rotation(
        bench_cfg, fake_root_server, b2_root_client):
    """The controller's repro, at B2's own (3-chunk) scale: the SECOND leaf
    dispatch exhausts a pool that can only ever hold one call at a time, a
    `process_manager` is injected, and the map must still finish -- all three
    chunks summarized, the reduce still runs, the rotation still recorded."""
    cfg = _b2_cfg(bench_cfg)
    disp = _RotatingDispatcher(_reply_by_marker, exhaust_at=2)
    pm = _FakeProcessManager()
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client,
                              process_manager=pm)

    assert res.outcome == Outcome.SUCCESS and res.answer == "FINAL"
    assert pm.restart_calls == 1
    assert disp.rotating_calls == 1 and disp.rotate_pool_calls == 1

    root_prompt = fake_root_server.last_completion_prompt
    assert "SUMMARY-A" in root_prompt
    assert "SUMMARY-B" in root_prompt      # the chunk whose call got rotated
    assert "SUMMARY-C" in root_prompt

    leaf_steps = [s for s in env.steps if s["actor"] == "leaf"]
    root_steps = [s for s in env.steps if s["actor"] == "root"]
    assert len(leaf_steps) == 4            # 3 chunks, one retried once
    assert len(root_steps) == 1
    statuses = [s["status"] for s in leaf_steps]
    assert statuses.count("error") == 1 and statuses.count("ok") == 3
    rotated = [s for s in leaf_steps if s["server_rotation"] is not None]
    assert len(rotated) == 1
    assert rotated[0]["server_rotation"] == 1 and rotated[0]["status"] == "error"

    snap = env.episode["config_snapshot"]
    assert snap["bench"]["arm"] == "b2"
    assert snap["bench"]["n_chunks"] == 3


async def test_b2_without_a_process_manager_closes_as_error_on_slot_pool_exhaustion(
        bench_cfg, fake_root_server, b2_root_client):
    """The honest degraded mode: no `process_manager` injected, so the same
    exhaustion that test (a) rotates through here ends the episode cleanly."""
    cfg = _b2_cfg(bench_cfg)
    disp = _RotatingDispatcher(_reply_by_marker, exhaust_at=2)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)

    assert res.outcome == Outcome.ERROR and res.reason == SLOT_POOL_EXHAUSTED
    assert env.episode["outcome"] == "error"
    assert env.episode["outcome_reason"] == "slot_pool_exhausted"
    # chunk A answered before the exhaustion; the reduce step never ran
    assert all(s["actor"] == "leaf" for s in env.steps)
    assert [s["status"] for s in env.steps] == ["ok", "error"]


# --------------------------------------------------------------------------- #
# The seed travels with the CALL, not with the dispatcher (S4 Task 12 fix).
#
# §8's three replicates are three seeds of the WHOLE system, and a bench run
# holds ONE leaf dispatcher across all of them while re-seeding the CONFIG per
# attempt (`rlm.measure.bench.seeded_config`). An arm that let C4's construction-time
# seed stand would send the same leaf seed for all three while
# `config_snapshot` recorded three different ones -- three draws of one leaf,
# reported as three seeds, and every §8 margin computed over them.
# --------------------------------------------------------------------------- #


async def test_b1_sends_its_own_configs_leaf_seed(bench_cfg):
    cfg = bench_cfg(leaf_seed=4)
    disp = _ScriptedDispatcher("ANSWER")
    res, _env = await _run_b1(cfg, _task(), disp)
    assert res.outcome == Outcome.SUCCESS
    assert disp.seeds == [4]


async def test_b3_sends_its_own_configs_leaf_seed(bench_cfg):
    cfg = bench_cfg(leaf_seed=5)
    disp = _ScriptedDispatcher("ANSWER")
    res, _env = await _run_b3(cfg, _task(), disp)
    assert res.outcome == Outcome.SUCCESS
    assert disp.seeds == [5]


async def test_the_leaf_seed_is_the_configs_and_never_the_dispatchers(bench_cfg):
    """The bug's exact shape: two attempts of the SAME task under two seeded
    configs, one dispatcher. Both calls must differ on the wire."""
    disp = _ScriptedDispatcher("ANSWER")
    for seed in (1, 2, 3):
        res, _env = await _run_b1(bench_cfg(leaf_seed=seed), _task(), disp)
        assert res.outcome == Outcome.SUCCESS
    assert disp.seeds == [1, 2, 3]


# --------------------------------------------------------------------------- #
# B2's summary budget is ENFORCED, and B2's root call cannot end the grid
# (S4 Task 12, fix wave 2).
# --------------------------------------------------------------------------- #


class _FailingRootClient:
    """A `root_client` whose first call raises. Structural, not a mock library:
    B2 only ever calls `apply_template` and `completion` on it."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def apply_template(self, *_a, **_kw):
        raise self._exc

    async def completion(self, *_a, **_kw):     # pragma: no cover - never reached
        raise self._exc


async def test_b2_caps_every_summary_at_the_pre_registered_budget(
        bench_cfg, fake_root_server, b2_root_client):
    """§8's formula sizes the summaries so ALL of them fit 80% of the root
    window. Recorded but unenforced, that claim was false for exactly the
    corpora it exists for: the leaf decodes to its own `max_predict` and the
    reduce prompt overflows the root by construction."""
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    fake_root_server.script = ["FINAL"]

    res, env = await _run_b2(cfg, _b2_task(), disp, b2_root_client)
    assert res.outcome == Outcome.SUCCESS

    n_chunks = env.episode["config_snapshot"]["bench"]["n_chunks"]
    want = b2_summary_n_predict(cfg, n_chunks)
    # The RECORDED number is the one that was SENT, on every summary call.
    assert env.episode["config_snapshot"]["bench"]["summary_n_predict"] == want
    assert disp.n_predicts == [want] * n_chunks
    assert len(disp.n_predicts) == 3


async def test_the_summary_budget_actually_shrinks_at_scale(bench_cfg):
    """The formula is only worth enforcing if it BINDS. At §8's aggregation
    scale it is 6x below the leaf's own ceiling, which is the whole gap
    'recorded but not enforced' left open."""
    cfg = bench_cfg()
    assert b2_summary_n_predict(cfg, 299) == 87
    assert cfg.scaffold.budgets.max_predict.leaf == 512


async def test_b2_maps_a_root_transport_failure_to_server_unreachable(
        bench_cfg, fake_root_server):
    """The root server is a server too. C4 wraps every LEAF transport failure
    in `DispatchError`, so a leaf that dies is `error/server_unreachable` and
    §8's rerun-once applies to it. B2's reduce step talks to the root DIRECTLY,
    and an httpx error there used to propagate out of `run_b2`, past
    `rlm.measure.bench._run_cell` (which contains only `ConfigError`) and out of the
    whole grid: one root hiccup at hour 12 ends a 39-hour run."""
    import httpx

    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    root = _FailingRootClient(httpx.ConnectError("connection refused"))

    res, env = await _run_b2(cfg, _b2_task(), disp, root)
    assert res.outcome == Outcome.ERROR
    assert res.reason == SERVER_UNREACHABLE
    # The row was CLOSED, not orphaned -- §6, and what makes the rerun scoreable.
    assert env.episode["outcome"] == "error"
    assert env.episode["outcome_reason"] == SERVER_UNREACHABLE


async def test_a_bug_in_the_root_call_is_not_dressed_as_a_server_failure(
        bench_cfg, fake_root_server):
    """The other half of the same guard: ONLY the transport family is
    converted. A `TypeError` is a bug in the scaffold, and §6 requires it stay
    loud rather than being scored as an ordinary `error` episode."""
    cfg = _b2_cfg(bench_cfg)
    disp = _PerChunkDispatcher(_reply_by_marker)
    root = _FailingRootClient(TypeError("apply_template() got a surprise"))

    with pytest.raises(TypeError):
        await _run_b2(cfg, _b2_task(), disp, root)


def test_the_transport_family_is_recognised_without_importing_httpx():
    """`arms.py` may not import httpx (the dependency rule bars every HTTP
    library from this side), so the family is recognised by the module its type
    is defined in. Pinned against the real classes, so an upstream rename
    cannot quietly empty the set."""
    import json as jsonmod

    import httpx

    from rlm.measure.arms import is_transport_error

    assert is_transport_error(httpx.ConnectError("x"))
    assert is_transport_error(httpx.ReadTimeout("x"))
    assert is_transport_error(OSError("socket died"))
    assert is_transport_error(jsonmod.JSONDecodeError("bad", "{", 0))
    assert not is_transport_error(TypeError("a bug"))
    assert not is_transport_error(ValueError("a bug"))


# --------------------------------------------------------------------------- #
# The S4 smoke crash, at the level it actually happened (2026-08-16, run
# 0f798a78, cell 16/16 = codeqa-01/b3): B3 assembles its prompt by chunking the
# corpus through C2, whose boundary binary search is ~12,000 `/tokenize` round
# trips, and ONE of them died in transport. `_b3_prompt` runs before the episode
# row is opened and `rlm.measure.bench._run_cell` contains only `ConfigError`, so the
# exception went all the way out of `run_bench` -- and because `httpx` types are
# not `RlmError`, `rlm.cli.cmd_bench`'s taxonomy (which turns named failures
# into `refused: ...` + exit 2 + a resume hint, and deliberately lets unnamed
# ones out as tracebacks) filed a transport hiccup as an unnameable bug.
# --------------------------------------------------------------------------- #


async def test_b3_chunk_counting_contains_a_dead_transport_as_an_rlm_error(
        bench_cfg):
    """A REAL C4 over a REAL client pointed at a port nothing listens on --
    not a fake that raises a chosen exception, since the whole question is what
    the HTTP library does and whether it can escape."""
    import httpx

    from rlm.config import Retries
    from rlm.serve.dispatcher import DispatchTarget, LLMDispatcher, ServerClient, SlotPool
    from rlm.errors import RlmError, TransportError

    cfg = bench_cfg()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    client = ServerClient(f"http://127.0.0.1:{port}", timeout=5.0)
    disp = LLMDispatcher(
        targets={"leaf": DispatchTarget(
            client=client, max_predict=64, slot_capacity_tokens=32768,
            temperature=0.3, top_p=0.9, seed=1, system_prefix="s")},
        parallel=2,
        retries=Retries(max_attempts=2, backoff_s=[0.01], per_call_timeout_s=5),
        slots=SlotPool(2))
    try:
        with pytest.raises(RlmError) as caught:
            await _run_b3(cfg, _task(context="a" * 4000), disp)
    finally:
        await disp.aclose()
    assert isinstance(caught.value, TransportError)
    assert not isinstance(caught.value, httpx.HTTPError)
    # The retry happened, and it happened where the storm is (the counter), not
    # around the one /completion B3 never got to make.
    assert disp.transport_retries == 2
