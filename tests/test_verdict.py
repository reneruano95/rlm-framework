"""Task 11: `rlm/verdict.py` -- §8's scoring, inference and report layer.

Everything here runs against a SYNTHETIC store written by the REAL
`TraceLogger` (tests/test_trace.py + tests/test_bench.py patterns): the module
under test reads a closed DuckDB file with the same SQL it will run against a
39-hour benchmark, so a fixture that hand-built rows with a raw INSERT would
test a different query than the one that ships.

No servers, no arms, no dispatcher -- a verdict is computed from the record and
nothing else, which is the property that makes S4 re-scoreable offline.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest
from bench.manifest import BenchmarkManifest, TaskEntry

from rlm.errors import ActionType, Actor, Outcome, StepStatus
from rlm.trace import TraceLogger
from rlm.measure.verdict import (
    MARGIN_GATE,
    NARRATIVE_MARKER,
    VerdictError,
    cost_scorecard,
    decide,
    leak_report,
    load_grid,
    pareto_svg,
    regenerate,
    render_report,
    write_report,
)

BASE_TS = dt.datetime(2026, 8, 16, 9, 0, 0)

#: One character per seed, so a whole mini-grid fits on a line.
OUTCOMES = {
    "T": Outcome.SUCCESS,
    "F": Outcome.FAIL,
    "K": Outcome.BUDGET_KILL,        # §8:346 -- a failure, for every arm
    "X": Outcome.CONTEXT_EXHAUSTED,  # §8:346 -- a failure, for every arm
    "E": Outcome.ERROR,              # terminal once its one rerun is spent
}


# --------------------------------------------------------------------------- #
# synthetic store
# --------------------------------------------------------------------------- #


def _snapshot(run_id: str, arm: str, seed: int, *, chunk: int = 32768,
              block: int = 0) -> dict:
    """The two `config_snapshot` sub-trees the verdict reads: `bench` (the
    identity `rlm.measure.bench.bench_extra` writes) and the chunk size §8's chunk-size
    lock puts on the verdict line."""
    return {"scaffold": {"chunk": {"size_tokens": chunk}},
            "bench": {"run_id": run_id, "arm": arm, "seed": seed, "block": block}}


def leaf_call(*, tokens_in: int = 0, tokens_out: int = 0, leak=None,
              retry: int = 0, status=StepStatus.OK) -> dict:
    return {"actor": Actor.LEAF, "action_type": ActionType.LLM_CALL,
            "status": status, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "leak_detected": leak, "retry_idx": retry}


def root_call(*, tokens_in: int = 0, tokens_out: int = 0) -> dict:
    return {"actor": Actor.ROOT, "action_type": ActionType.LLM_CALL,
            "status": StepStatus.OK, "tokens_in": tokens_in,
            "tokens_out": tokens_out}


class StoreBuilder:
    """A closed §8 trace store, one episode at a time."""

    def __init__(self, tmp_path: Path, run_id: str = "run-1") -> None:
        self.db = Path(tmp_path) / "rlm.duckdb"
        self.blobs = Path(tmp_path) / "blobs"
        self.run_id = run_id
        self.tl = TraceLogger(self.db, self.blobs)
        self._walls: dict[str, float] = {}

    async def start(self) -> "StoreBuilder":
        await self.tl.start()
        return self

    def episode(self, task_id: str, arm: str, seed: int, outcome, *,
                dry_run: bool = False, chunk: int = 32768, run_id: str | None = None,
                steps=(), wall_s: float | None = None, energy_j: float | None = None,
                avg_power_w: float | None = None, close: bool = True) -> str:
        ep = str(uuid.uuid4())
        self.tl.open_episode({
            "episode_id": ep, "task_id": task_id, "task_hash": "h",
            "started_at": BASE_TS, "dry_run": dry_run,
            "config_snapshot": _snapshot(run_id or self.run_id, arm, seed,
                                          chunk=chunk)})
        for step in steps:
            self.tl.put_step({"episode_id": ep, **step})
        if energy_j is not None or avg_power_w is not None:
            self.tl.update_episode_metrics(ep, energy_j=energy_j,
                                            avg_power_w=avg_power_w)
        if close:
            self.tl.close_episode(ep, outcome, None, None)
        if wall_s is not None:
            self._walls[ep] = wall_s
        return ep

    def supersede(self, old_id: str, new_id: str) -> None:
        self.tl.mark_superseded(old_id, new_id)

    async def close(self) -> Path:
        await self.tl.drain()
        await self.tl.aclose()
        if self._walls:
            # `ended_at` is stamped by the writer at close time; a deterministic
            # wall-clock has to be written after the fact, on the closed file.
            con = duckdb.connect(str(self.db))
            try:
                for ep, wall in self._walls.items():
                    con.execute(
                        "UPDATE episodes SET ended_at = started_at + "
                        "to_milliseconds(?) WHERE episode_id = ?",
                        [int(round(wall * 1000)), ep])
                con.execute("CHECKPOINT")
            finally:
                con.close()
        return self.db


async def build_store(tmp_path: Path, spec: dict[str, dict[str, str]], *,
                      seeds=(1, 2, 3), run_id: str = "run-1", **kw) -> Path:
    """`{"t1": {"rlm": "TTT", "b1": "TFF"}}` -> a closed store."""
    b = await StoreBuilder(tmp_path, run_id).start()
    for task_id, arms in spec.items():
        for arm, pattern in arms.items():
            for seed, ch in zip(seeds, pattern):
                b.episode(task_id, arm, seed, OUTCOMES[ch], **kw)
    return await b.close()


def manifest_for(categories: dict[str, str]) -> BenchmarkManifest:
    """A manifest carrying only what `decide` reads: task_id -> category."""
    return BenchmarkManifest(
        benchmark_version="test", built_at="2026-08-16",
        token_counter="approx-offline", assumed_training_cutoff="2025-06",
        tasks=[TaskEntry(task_id=tid, category=cat,
                         task_file=f"bench/tasks/{tid}.json",
                         corpus_path=f"bench/corpora/{tid}.txt",
                         corpus_sha256="0" * 64, corpus_tokens=10,
                         corpus_date="2026-08-15", checker="int_exact",
                         question_sha256="1" * 64)
               for tid, cat in categories.items()])


ALL_ARMS = ("rlm", "b1", "b2", "b3")


def clean_pass_spec() -> dict:
    """RLM 6/6, every baseline 2/6 -> margin +4 against all three."""
    spec: dict[str, dict[str, str]] = {}
    for i in range(1, 7):
        tid = f"t{i}"
        spec[tid] = {"rlm": "TTT"}
        for b in ("b1", "b2", "b3"):
            spec[tid][b] = "TTT" if i <= 2 else "FFF"
    return spec


SIX_CATEGORIES = {"t1": "needle", "t2": "needle", "t3": "needle",
                  "t4": "aggregation", "t5": "aggregation", "t6": "aggregation"}


async def escalated_run(tmp_path) -> tuple:
    """§8:343 end to end, in ONE store under ONE run_id — the production
    sequence Task 12 executes.

    Base grid (seeds 1–3): RLM passes t1–t5 (t2 and t5 at 2/3), every baseline
    passes t1–t2 → margin +3, inside the {+1,+2,+3} band, discordant t3/t4/t5.
    Escalation then runs seeds {4, 5} on those three tasks: RLM takes t3 and t4
    (5/5) and MISSES t5 (2/5, which is below ≥3/5), so t5 flips from a pass to
    a failure and the margin drops +3 → +2. `t2` is the control: the identical
    2/3 base pattern, never escalated, and it still passes at thirds.

    Returns `(db, pre, post)`.
    """
    spec: dict[str, dict[str, str]] = {}
    for i in range(1, 7):
        tid = f"t{i}"
        rlm = {"t2": "TTF", "t5": "TTF", "t6": "FFF"}.get(tid, "TTT")
        base = "TTT" if i <= 2 else "FFF"
        spec[tid] = {"rlm": rlm, "b1": base, "b2": base, "b3": base}
    db = await build_store(tmp_path, spec, wall_s=1.0)
    pre = decide(load_grid(db, "run-1"), manifest_for(SIX_CATEGORIES))

    b = await StoreBuilder(tmp_path).start()          # same db, same run_id
    for tid in ("t3", "t4", "t5"):
        for arm in ALL_ARMS:
            for seed in (4, 5):
                won = arm == "rlm" and tid != "t5"
                b.episode(tid, arm, seed,
                          Outcome.SUCCESS if won else Outcome.FAIL, wall_s=1.0)
    await b.close()
    post = decide(load_grid(db, "run-1"), manifest_for(SIX_CATEGORIES))
    return db, pre, post

NINE_CATEGORIES = {**{f"t{i}": "needle" for i in (1, 2, 3)},
                   **{f"t{i}": "aggregation" for i in range(4, 10)}}


# --------------------------------------------------------------------------- #
# load_grid
# --------------------------------------------------------------------------- #


async def test_the_grid_reads_every_cell_in_seed_order(tmp_path):
    db = await build_store(tmp_path, {"t1": {"rlm": "TFT", "b1": "FFF"}})
    grid = load_grid(db, "run-1")
    assert grid.task_ids == ("t1",)
    assert set(grid.arms) == {"rlm", "b1"}
    assert grid.seeds == (1, 2, 3)
    assert grid.cell("t1", "rlm") == [True, False, True]
    assert grid.cell("t1", "b1") == [False, False, False]


async def test_budget_kill_and_context_exhausted_are_failures_for_every_arm(tmp_path):
    """§8:346 -- they are results, not retries, and they score as failures."""
    db = await build_store(tmp_path, {"t1": {"rlm": "KXE", "b1": "TTT"}})
    grid = load_grid(db, "run-1")
    assert grid.cell("t1", "rlm") == [False, False, False]


async def test_a_terminal_error_scores_false(tmp_path):
    """The scheduler's refusal rows land as `outcome='error'` and an error that
    spent its rerun stands. Either way it is that arm's failure, never a hole."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.ERROR)
        b.episode("t1", "b1", seed, Outcome.SUCCESS)
    db = await b.close()
    assert load_grid(db, "run-1").cell("t1", "rlm") == [False, False, False]


async def test_the_grid_is_scoped_to_its_run_id(tmp_path):
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS)
        b.episode("t1", "rlm", seed, Outcome.FAIL, run_id="some-other-run")
    db = await b.close()
    grid = load_grid(db, "run-1")
    assert grid.cell("t1", "rlm") == [True, True, True]
    assert load_grid(db, "some-other-run").cell("t1", "rlm") == [False] * 3


async def test_dry_run_episodes_are_excluded(tmp_path):
    """A dry-run row beside a real one would otherwise read as a duplicate --
    which is exactly the refusal that must NOT fire here."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS)
        b.episode("t1", "rlm", seed, Outcome.FAIL, dry_run=True)
    db = await b.close()
    assert load_grid(db, "run-1").cell("t1", "rlm") == [True, True, True]


async def test_a_cell_that_only_ever_dry_ran_is_a_missing_cell(tmp_path):
    """A dry run measured nothing, so a cell covered only by one is a HOLE --
    the arm ran for the other task, which is what makes the hole visible."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS)
        b.episode("t1", "b1", seed, Outcome.SUCCESS)
        b.episode("t2", "rlm", seed, Outcome.SUCCESS)
        b.episode("t2", "b1", seed, Outcome.SUCCESS, dry_run=True)
    db = await b.close()
    with pytest.raises(VerdictError, match="missing"):
        load_grid(db, "run-1")


async def test_superseded_rows_are_excluded_and_the_rerun_is_the_result(tmp_path):
    """§8's rerun-once rule (§6 `superseded_by`): the first attempt errored,
    the rerun succeeded, and only the rerun scores."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        old = b.episode("t1", "rlm", seed, Outcome.ERROR)
        new = b.episode("t1", "rlm", seed, Outcome.SUCCESS)
        b.supersede(old, new)
        b.episode("t1", "b1", seed, Outcome.FAIL)
    db = await b.close()
    assert load_grid(db, "run-1").cell("t1", "rlm") == [True, True, True]


async def test_a_missing_cell_is_refused_not_imputed(tmp_path):
    db = await build_store(tmp_path, {"t1": {"rlm": "TT", "b1": "TTT"}})
    with pytest.raises(VerdictError) as exc:
        load_grid(db, "run-1")
    assert "missing" in str(exc.value)
    assert "t1" in str(exc.value) and "rlm" in str(exc.value)


async def test_a_whole_arm_missing_for_one_task_is_a_missing_cell(tmp_path):
    db = await build_store(tmp_path, {"t1": {"rlm": "TTT", "b1": "TTT"},
                                       "t2": {"rlm": "TTT"}})
    with pytest.raises(VerdictError, match="missing"):
        load_grid(db, "run-1")


async def test_a_duplicate_cell_is_refused(tmp_path):
    """Two live rows for one (task, seed, arm) means the rerun link was never
    made -- scoring either one silently would pick a winner by row order."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS)
    b.episode("t1", "rlm", 2, Outcome.FAIL)
    db = await b.close()
    with pytest.raises(VerdictError) as exc:
        load_grid(db, "run-1")
    assert "duplicate" in str(exc.value)


async def test_escalation_seeds_may_cover_a_subset_of_tasks(tmp_path):
    """§8:343 runs seeds {4,5} on the DISCORDANT tasks only, for one pair --
    so the post-escalation grid is legitimately ragged and must not refuse."""
    b = await StoreBuilder(tmp_path).start()
    for tid in ("t1", "t2"):
        for arm in ("rlm", "b2"):
            for seed in (1, 2, 3):
                b.episode(tid, arm, seed, Outcome.SUCCESS)
    for arm in ("rlm", "b2"):
        for seed in (4, 5):
            b.episode("t1", arm, seed, Outcome.SUCCESS)
    db = await b.close()
    grid = load_grid(db, "run-1")
    assert grid.cell("t1", "rlm") == [True] * 5
    assert grid.cell("t2", "rlm") == [True] * 3
    assert grid.escalated_tasks == ("t1",)


async def test_a_half_written_escalation_is_refused(tmp_path):
    """§8:343 runs seeds {4,5} together and re-decides at ≥3/5. A cell holding
    only seed 4 would be a 4-long cell, which `stats.task_passes` reads as a
    5-seed one and scores at a denominator §8 never pre-registered."""
    b = await StoreBuilder(tmp_path).start()
    for tid in ("t1", "t2"):
        for arm in ("rlm", "b1"):
            for seed in (1, 2, 3):
                b.episode(tid, arm, seed, Outcome.SUCCESS)
    for arm in ("rlm", "b1"):
        b.episode("t1", arm, 4, Outcome.SUCCESS)       # seed 5 never ran
    db = await b.close()
    with pytest.raises(VerdictError) as exc:
        load_grid(db, "run-1")
    assert "half-escalated" in str(exc.value)
    assert "t1/rlm" in str(exc.value) and "[5]" in str(exc.value)


async def test_a_seed_named_as_a_base_seed_is_never_read_as_an_escalation_one(
        tmp_path):
    """The completeness check must not fire on a run that legitimately uses 4
    as a base seed — `seeds=` takes it out of the escalation set."""
    db = await build_store(tmp_path, {"t1": {"rlm": "TTT"}}, seeds=(1, 2, 4))
    assert load_grid(db, "run-1", seeds=(1, 2, 4)).cell("t1", "rlm") == [True] * 3


async def test_a_run_of_escalation_seeds_alone_still_checks_for_holes(tmp_path):
    """Subtracting {4,5} from a grid that has nothing else would leave no
    required seeds at all -- and a missing-cell check over an empty seed set
    passes everything."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (4, 5):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS)
    b.episode("t1", "b1", 4, Outcome.SUCCESS)
    db = await b.close()
    with pytest.raises(VerdictError, match="missing"):
        load_grid(db, "run-1")


async def test_explicit_base_seeds_override_the_inference(tmp_path):
    db = await build_store(tmp_path, {"t1": {"rlm": "TTT"}}, seeds=(1, 2, 3))
    assert load_grid(db, "run-1", seeds=(1, 2)).cell("t1", "rlm") == [True] * 3
    with pytest.raises(VerdictError, match="missing"):
        load_grid(db, "run-1", seeds=(1, 2, 3, 9))


async def test_the_grid_records_the_chunk_size(tmp_path):
    db = await build_store(tmp_path, {"t1": {"rlm": "TTT"}}, chunk=32768)
    assert load_grid(db, "run-1").chunk_size == 32768


async def test_a_missing_store_is_refused_with_its_path(tmp_path):
    with pytest.raises(VerdictError, match="no trace store"):
        load_grid(tmp_path / "nope.duckdb", "run-1")


async def test_a_live_store_is_refused_rather_than_half_scored(tmp_path):
    """A verdict is computed from a CLOSED store. On Windows a second reader
    cannot open a file a writer holds (rlm/trace.py:302-307) -- so the refusal
    is free, and it must arrive as an explanation, not a DuckDB IOException."""
    b = await StoreBuilder(tmp_path).start()
    b.episode("t1", "rlm", 1, Outcome.SUCCESS)
    await b.tl.drain()
    try:
        with pytest.raises(VerdictError, match="CLOSED store"):
            load_grid(b.db, "run-1")
    finally:
        await b.close()


async def test_a_run_whose_chunk_size_moved_is_named_not_averaged(tmp_path):
    """§8's chunk-size lock: S4 runs the chunked arms at ONE untouched default.
    A grid carrying two is reported as a finding, never silently summarised."""
    bl = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        bl.episode("t1", "rlm", seed, Outcome.SUCCESS,
                   chunk=32768 if seed < 3 else 16384)
    db = await bl.close()
    grid = load_grid(db, "run-1")
    assert grid.chunk_sizes == (16384, 32768)
    assert grid.chunk_size is None
    v = decide(grid, manifest_for(SIX_CATEGORIES))
    assert any(f.kind == "chunk_size_lock" for f in v.findings)
    assert "chunk_size=MIXED" in render_report(v, None, {})


async def test_an_episode_with_no_bench_identity_is_refused(tmp_path):
    """A row that carries the run_id but no arm cannot be placed in the grid,
    and dropping it silently would lose a cell rather than report one."""
    b = await StoreBuilder(tmp_path).start()
    ep = str(uuid.uuid4())
    b.tl.open_episode({"episode_id": ep, "task_id": "t1", "task_hash": "h",
                        "started_at": BASE_TS,
                        "config_snapshot": {"bench": {"run_id": "run-1"}}})
    b.tl.close_episode(ep, Outcome.SUCCESS, None, None)
    db = await b.close()
    with pytest.raises(VerdictError, match="arm"):
        load_grid(db, "run-1")


# --------------------------------------------------------------------------- #
# decide
# --------------------------------------------------------------------------- #


async def _verdict(tmp_path, spec, categories=None, run_id="run-1"):
    db = await build_store(tmp_path, spec, run_id=run_id, wall_s=1.0)
    return decide(load_grid(db, run_id), manifest_for(categories or SIX_CATEGORIES))


async def test_the_gate_passes_at_the_pre_registered_margin_against_all_three(tmp_path):
    v = await _verdict(tmp_path, clean_pass_spec())
    assert v.gate_pass is True
    assert v.clean_pass is True
    assert v.success_rate["rlm"] == 1.0
    for b in ("b1", "b2", "b3"):
        assert v.pairs[b].margin == 4
        assert v.pairs[b].beats is True


async def test_a_tie_with_any_baseline_fails_the_gate(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:
        spec[tid]["b2"] = spec[tid]["rlm"]          # B2 matches RLM exactly
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b2"].margin == 0
    assert v.gate_pass is False


async def test_a_tie_or_loss_to_b2_records_the_pivot_to_b2_finding(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:
        spec[tid]["b2"] = spec[tid]["rlm"]
    v = await _verdict(tmp_path, spec)
    assert any(f.kind == "pivot_to_b2" for f in v.findings)


async def test_a_tie_or_loss_to_b3_records_the_pivot_to_rag_finding(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:
        spec[tid]["b3"] = spec[tid]["rlm"]
    v = await _verdict(tmp_path, spec)
    assert any(f.kind == "pivot_to_rag" for f in v.findings)
    assert not any(f.kind == "pivot_to_b2" for f in v.findings)


async def test_a_margin_of_exactly_plus_three_beats_the_baseline(tmp_path):
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"                        # RLM 5, baselines 2
    v = await _verdict(tmp_path, spec)
    assert all(v.pairs[b].margin == MARGIN_GATE for b in ("b1", "b2", "b3"))
    assert v.gate_pass is True


async def test_a_plus_three_pass_still_plans_escalation(tmp_path):
    """§8:343's band is {+1,+2,+3} and the gate is >=+3 -- they OVERLAP, so a
    passing run can still owe seeds {4,5}. Collapsing the two would quietly
    drop the de-noising exactly where the decision is tightest."""
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"
    v = await _verdict(tmp_path, spec)
    assert v.gate_pass is True
    assert set(v.escalation_plan) == {"b1", "b2", "b3"}
    assert v.escalation_plan["b1"] == ("t3", "t4", "t5")


async def test_escalation_lists_only_that_pairs_discordant_tasks(tmp_path):
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"                        # margin +3 vs b1/b3
    spec["t3"]["b2"] = "TTT"                         # b2 wins t3 back: margin +2
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b2"].margin == 2
    assert v.escalation_plan["b2"] == ("t4", "t5")
    assert v.escalation_plan["b1"] == ("t3", "t4", "t5")


async def test_a_margin_outside_the_band_plans_no_escalation(tmp_path):
    v = await _verdict(tmp_path, clean_pass_spec())   # +4 everywhere
    assert v.escalation_plan == {}


async def test_a_loss_is_never_escalated_into_a_win(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:                                  # B2 sweeps, RLM 2/6
        spec[tid]["rlm"] = "TTT" if tid in ("t1", "t2") else "FFF"
        spec[tid]["b2"] = "TTT"
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b2"].margin == -4
    assert "b2" not in v.escalation_plan
    assert v.gate_pass is False


async def test_every_pair_carries_its_sign_test_p_and_bootstrap_ci(tmp_path):
    v = await _verdict(tmp_path, clean_pass_spec())
    pair = v.pairs["b1"]
    assert pair.wins == 4 and pair.losses == 0
    assert pair.p == pytest.approx(2 * 0.5 ** 4)      # exact, hand-computable
    lo, hi = pair.ci
    assert lo <= pair.mean_delta <= hi


async def test_the_ci_is_n_a_rather_than_a_crash_on_an_empty_grid(tmp_path):
    """`paired_bootstrap_ci` divides by len(deltas); a zero-task grid must
    report "no CI", never raise ZeroDivisionError out of the report path."""
    b = await StoreBuilder(tmp_path).start()
    db = await b.close()
    v = decide(load_grid(db, "run-1"), manifest_for(SIX_CATEGORIES))
    assert v.n_tasks == 0
    for pair in v.pairs.values():
        assert pair.ci is None
    text = render_report(v, None, {})
    assert "n/a" in text


async def test_a_task_absent_from_the_manifest_is_refused(tmp_path):
    db = await build_store(tmp_path, {"ghost": {"rlm": "TTT", "b1": "TTT"}})
    with pytest.raises(VerdictError, match="ghost"):
        decide(load_grid(db, "run-1"), manifest_for(SIX_CATEGORIES))


async def test_a_partial_grid_is_reported_rather_than_scored_silently(tmp_path):
    spec = {t: {a: "TTT" for a in ALL_ARMS} for t in ("t1", "t2")}
    v = await _verdict(tmp_path, spec)
    assert v.n_tasks == 2
    assert any(f.kind == "partial_grid" for f in v.findings)


async def test_a_missing_baseline_cannot_pass_the_gate(tmp_path):
    spec = {t: {a: ("TTT" if a == "rlm" else "FFF") for a in ("rlm", "b1", "b2")}
            for t in SIX_CATEGORIES}
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b1"].margin == 6
    assert v.gate_pass is False
    assert any(f.kind == "missing_arm" for f in v.findings)


# --------------------------------------------------------------------------- #
# per-category table + the zero-floor tripwire
# --------------------------------------------------------------------------- #


async def test_the_per_category_table_counts_passes_per_arm(tmp_path):
    v = await _verdict(tmp_path, clean_pass_spec())
    rows = {r.category: r for r in v.categories}
    assert rows["needle"].n_tasks == 3 and rows["aggregation"].n_tasks == 3
    assert rows["needle"].passes["rlm"] == 3
    assert rows["aggregation"].passes["b1"] == 0


async def test_the_zero_floor_tripwire_fires_and_blocks_a_clean_pass(tmp_path):
    """RLM scores 0 in `needle` while B1 takes all 3 -- the aggregate gate may
    still pass, but the verdict carries a named category-regression finding."""
    spec = clean_pass_spec()
    for tid in ("t1", "t2", "t3"):
        spec[tid]["rlm"] = "FFF"
        spec[tid]["b1"] = "TTT"
        spec[tid]["b2"] = "FFF"
        spec[tid]["b3"] = "FFF"
    for tid in ("t4", "t5", "t6"):
        spec[tid]["rlm"] = "TTT"
        for b in ("b1", "b2", "b3"):
            spec[tid][b] = "FFF"
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b2"].margin == 3 and v.pairs["b3"].margin == 3
    regressions = [f for f in v.findings if f.kind == "category_regression"]
    assert len(regressions) == 1
    assert "needle" in regressions[0].text
    assert v.clean_pass is False


async def test_a_category_regression_does_not_gate_the_aggregate(tmp_path):
    """§8:350 refuses per-category margin gates outright: the table is
    REPORTED, the aggregate alone decides, and the regression's whole effect is
    to block the WORDING "clean pass"."""
    spec: dict[str, dict[str, str]] = {}
    for tid in NINE_CATEGORIES:
        needle = NINE_CATEGORIES[tid] == "needle"
        spec[tid] = {"rlm": "FFF" if needle else "TTT",
                     **{b: ("TTT" if needle else "FFF") for b in ("b1", "b2", "b3")}}
    v = await _verdict(tmp_path, spec, NINE_CATEGORIES)
    assert all(v.pairs[b].margin == 3 for b in ("b1", "b2", "b3"))
    assert v.gate_pass is True                        # the aggregate still passes
    assert v.clean_pass is False                      # but it is not "clean"
    assert any(f.kind == "category_regression" for f in v.findings)
    assert "NOT a clean pass" in render_report(v, None, {})


async def test_the_tripwire_needs_a_baseline_at_three_tasks(tmp_path):
    """The floor is "any baseline scores >=3 of its tasks" -- 2 is not 3, and
    a tripwire that fired at 2 would fire in every 3-task category by noise."""
    spec = clean_pass_spec()
    for tid in ("t1", "t2", "t3"):
        spec[tid]["rlm"] = "FFF"
    spec["t1"]["b1"] = spec["t2"]["b1"] = "TTT"
    spec["t3"]["b1"] = "FFF"
    v = await _verdict(tmp_path, spec)
    assert not any(f.kind == "category_regression" for f in v.findings)


# --------------------------------------------------------------------------- #
# cost scorecard
# --------------------------------------------------------------------------- #


async def _cost_store(tmp_path) -> Path:
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS, wall_s=30.0,
                  energy_j=1200.0, avg_power_w=40.0,
                  steps=[leaf_call(tokens_in=100, tokens_out=10),
                         leaf_call(tokens_in=100, tokens_out=10, retry=1),
                         root_call(tokens_in=50, tokens_out=5)])
        b.episode("t1", "b1", seed, Outcome.SUCCESS, wall_s=10.0,
                  steps=[leaf_call(tokens_in=40, tokens_out=5)])
    return await b.close()


async def test_task_tokens_sum_every_step_including_retries(tmp_path):
    db = await _cost_store(tmp_path)
    sc = cost_scorecard(db, "run-1")
    assert sc.tasks[("rlm", "t1")].tokens == 275      # 110 + 110 + 55
    assert sc.tasks[("b1", "t1")].tokens == 45


async def test_null_token_columns_count_as_zero_not_as_a_hole(tmp_path):
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS, wall_s=1.0,
                  steps=[leaf_call(), leaf_call(tokens_in=7, tokens_out=None)])
    db = await b.close()
    assert cost_scorecard(db, "run-1").tasks[("rlm", "t1")].tokens == 7


async def test_wall_clock_comes_from_the_episode_timestamps(tmp_path):
    db = await _cost_store(tmp_path)
    sc = cost_scorecard(db, "run-1")
    assert sc.tasks[("rlm", "t1")].wall_s == pytest.approx(30.0)
    assert sc.arms["b1"].median_wall_s == pytest.approx(10.0)


async def test_energy_stays_null_when_nothing_sampled_it(tmp_path):
    """On this host `energy_j` is genuinely NULL; the scorecard must say so
    rather than print a 0.0 that reads like a measurement."""
    db = await _cost_store(tmp_path)
    sc = cost_scorecard(db, "run-1")
    assert sc.arms["rlm"].median_energy_j == pytest.approx(1200.0)
    assert sc.arms["b1"].median_energy_j is None
    assert sc.arms["b1"].median_power_w is None


async def test_cost_multiples_are_computed_against_each_baseline(tmp_path):
    db = await _cost_store(tmp_path)
    sc = cost_scorecard(db, "run-1")
    assert sc.multiple("rlm", "b1", "wall") == pytest.approx(3.0)
    assert sc.multiple("rlm", "b1", "tokens") == pytest.approx(275 / 45)
    assert sc.multiple("rlm", "b2", "wall") is None   # arm never ran


async def test_an_episode_that_never_closed_has_no_wall_clock(tmp_path):
    """`ended_at` is NULL until the writer closes the row. A crashed episode
    must report no wall-clock, never a zero that reads as a fast one."""
    b = await StoreBuilder(tmp_path).start()
    for seed in (1, 2, 3):
        b.episode("t1", "rlm", seed, Outcome.SUCCESS, close=False)
    db = await b.close()
    sc = cost_scorecard(db, "run-1")
    assert sc.tasks[("rlm", "t1")].wall_s is None
    assert sc.arms["rlm"].median_wall_s is None
    assert sc.multiple("rlm", "b1", "wall") is None


# --------------------------------------------------------------------------- #
# R13 leak report
# --------------------------------------------------------------------------- #


async def test_the_leak_report_keeps_all_three_buckets_apart(tmp_path):
    b = await StoreBuilder(tmp_path).start()
    b.episode("t1", "rlm", 1, Outcome.SUCCESS,
              steps=[leaf_call(leak=True), leaf_call(leak=False),
                     leaf_call(leak=False), leaf_call(leak=None)])
    b.episode("t1", "b1", 1, Outcome.SUCCESS, steps=[leaf_call(leak=None)])
    db = await b.close()
    rep = leak_report(db, "run-1")
    assert rep["rlm"] == {"hits": 1, "checked_clean": 2, "not_checked": 1,
                          "leaf_attempts": 4}
    assert rep["b1"] == {"hits": 0, "checked_clean": 0, "not_checked": 1,
                         "leaf_attempts": 1}


async def test_the_leak_report_counts_only_leaf_llm_calls(tmp_path):
    b = await StoreBuilder(tmp_path).start()
    b.episode("t1", "rlm", 1, Outcome.SUCCESS,
              steps=[leaf_call(leak=False), root_call(),
                     {"actor": Actor.ROOT, "action_type": ActionType.REPL_EXEC,
                      "status": StepStatus.OK}])
    db = await b.close()
    assert leak_report(db, "run-1")["rlm"]["leaf_attempts"] == 1


async def test_an_arm_with_no_leaf_calls_still_appears(tmp_path):
    """B1 is single-shot; its row must read 0/0/0 rather than vanish, or the
    table would silently describe three arms and be read as four."""
    db = await build_store(tmp_path, {"t1": {"rlm": "TTT", "b1": "TTT"}})
    rep = leak_report(db, "run-1")
    assert set(rep) == {"rlm", "b1"}
    assert rep["b1"]["leaf_attempts"] == 0


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


@pytest.fixture
async def rendered(tmp_path):
    b = await StoreBuilder(tmp_path).start()
    spec = clean_pass_spec()
    for tid, arms in spec.items():
        for arm, pattern in arms.items():
            for seed, ch in zip((1, 2, 3), pattern):
                b.episode(tid, arm, seed, OUTCOMES[ch],
                          wall_s=40.0 if arm == "rlm" else 10.0,
                          steps=([leaf_call(tokens_in=200, tokens_out=20,
                                             leak=False)] * 4
                                  if arm in ("rlm", "b2")
                                  else [leaf_call(tokens_in=50, tokens_out=5)]))
    db = await b.close()
    grid = load_grid(db, "run-1")
    verdict = decide(grid, manifest_for(SIX_CATEGORIES))
    return verdict, cost_scorecard(db, "run-1"), leak_report(db, "run-1")


async def test_the_rule_is_stated_before_any_number(rendered):
    text = render_report(*rendered)
    assert text.index("pre-registered") < text.index("## S4 GATE")


async def test_the_gate_heading_is_machine_readable(rendered):
    assert "## S4 GATE: PASS" in render_report(*rendered)


async def test_a_failing_gate_says_fail(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:
        spec[tid]["b2"] = spec[tid]["rlm"]
    v = await _verdict(tmp_path, spec)
    assert "## S4 GATE: FAIL" in render_report(v, None, {})


async def test_the_verdict_line_records_the_chunk_size(rendered):
    assert "chunk_size=32768" in render_report(*rendered)


async def test_every_margin_carries_its_p_and_ci(rendered):
    """§8: "the S4 verdict states the p-value and CI next to the margin" -- so
    every line that shows a margin shows both, table row and prose alike."""
    verdict, sc, leaks = rendered
    text = render_report(verdict, sc, leaks)
    for baseline in ("B1", "B2", "B3"):
        lines = [l for l in text.splitlines() if f"vs {baseline}" in l and "+4" in l]
        assert len(lines) >= 2, f"{baseline}: {lines}"   # table row + win claim
        for line in lines:
            assert "p=" in line and "CI=[" in line, line


async def test_every_win_claim_states_its_cost_multiple(rendered):
    text = render_report(*rendered)
    for baseline in ("B1", "B2", "B3"):
        claim = [l for l in text.splitlines()
                 if l.startswith(f"RLM beats {baseline}")]
        assert claim, f"no win claim for {baseline}"
        assert "median wall-clock" in claim[0] and "tokens" in claim[0]
        assert "x " in claim[0]


async def test_a_margin_below_the_threshold_leads_but_never_beats(tmp_path):
    """§8 DEFINES "beats" as a margin of +3 at N=30. Prose that calls +2 a win
    while the gate calls it a failure is a report arguing with its verdict."""
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"
    spec["t3"]["b1"] = "TTT"                          # +2 vs B1, +3 vs B2/B3
    v = await _verdict(tmp_path, spec)
    assert v.pairs["b1"].margin == 2 and v.pairs["b1"].beats is False
    text = render_report(v, None, {})
    assert "RLM leads B1 by +2 tasks" in text
    assert "RLM beats B1" not in text
    assert f"below the +{MARGIN_GATE} threshold" in text
    assert "RLM beats B2 by +3 tasks" in text         # +3 still beats


async def test_the_r13_table_shows_all_three_buckets_and_never_claims_clean(rendered):
    text = render_report(*rendered)
    assert "not_checked" in text and "checked_clean" in text and "hits" in text
    assert "leak-free" not in text.lower()


async def test_the_per_category_table_is_rendered_but_never_gated(rendered):
    text = render_report(*rendered)
    assert "needle" in text and "aggregation" in text
    assert "per-category margin gates" in text.lower()


async def test_the_pareto_is_a_dependency_free_inline_svg(rendered):
    verdict, sc, _ = rendered
    svg = pareto_svg(verdict, sc)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "rlm" in svg and "b1" in svg
    assert "http://www.w3.org/2000/svg" in svg     # the only URL: the namespace
    assert "script" not in svg


async def test_the_pareto_survives_a_scorecard_with_no_wall_clock(rendered):
    verdict, _, _ = rendered
    assert pareto_svg(verdict, None).startswith("<svg")


async def test_the_report_is_written_with_its_pareto_beside_it(tmp_path, rendered):
    verdict, sc, leaks = rendered
    out = tmp_path / "s4" / "RESULTS.md"
    write_report(out, verdict, sc, leaks)
    assert out.exists()
    svg = out.with_suffix(".pareto.svg")
    assert svg.exists() and svg.read_text(encoding="utf-8").startswith("<svg")
    assert svg.name in out.read_text(encoding="utf-8")


async def test_regeneration_preserves_the_hand_written_findings(tmp_path, rendered):
    verdict, sc, leaks = rendered
    out = tmp_path / "RESULTS.md"
    write_report(out, verdict, sc, leaks)
    with out.open("a", encoding="utf-8") as fh:
        fh.write("\n## What actually happened\n\nThe leaf never read chunk 7.\n")
    write_report(out, verdict, sc, leaks)
    text = out.read_text(encoding="utf-8")
    assert "The leaf never read chunk 7." in text
    assert text.count(NARRATIVE_MARKER) == 1
    assert text.index(NARRATIVE_MARKER) < text.index("The leaf never read")


async def test_regenerate_is_a_no_op_when_there_is_nothing_to_preserve(tmp_path):
    body = f"# head\n\n{NARRATIVE_MARKER}\n"
    assert regenerate(tmp_path / "absent.md", body).count(NARRATIVE_MARKER) == 1


async def test_the_report_names_the_escalation_plan_when_one_is_owed(tmp_path):
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"
    v = await _verdict(tmp_path, spec)
    text = render_report(v, None, {})
    assert "Escalation" in text
    assert "t3" in text and "t4" in text and "t5" in text
    assert "4, 5" in text or "{4, 5}" in text


async def test_the_report_carries_pre_and_post_escalation_inference(tmp_path):
    """§8:343: recompute ONCE on the final grid, and report BOTH."""
    _db, pre, post = await escalated_run(tmp_path)
    assert pre.pairs["b1"].margin == 3 and post.pairs["b1"].margin == 2
    text = render_report(pre, None, {}, escalated=post)
    assert "pre-escalation" in text.lower() and "post-escalation" in text.lower()
    assert "+3" in text and "+2" in text


async def test_an_escalated_verdict_from_another_run_is_refused(tmp_path):
    """The title and the pre-escalation figures come from one verdict and the
    GATE from the other; nothing downstream could detect the swap."""
    _db, pre, post = await escalated_run(tmp_path)
    stranger = await _verdict(tmp_path / "elsewhere", clean_pass_spec(),
                              run_id="run-2")
    with pytest.raises(VerdictError, match="run-2"):
        render_report(pre, None, {}, escalated=stranger)
    render_report(pre, None, {}, escalated=post)          # the same run is fine


async def test_a_pre_escalation_grid_may_not_pose_as_the_recomputation(tmp_path):
    """Passing a second pre-escalation verdict as `escalated` would claim §8's
    de-noising step ran when it did not."""
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"
    v = await _verdict(tmp_path, spec)
    assert v.escalated is False
    with pytest.raises(VerdictError, match="no escalation seeds"):
        render_report(v, None, {}, escalated=v)


async def test_escalated_and_unescalated_tasks_score_at_their_own_denominators(
        tmp_path):
    """§8:343's composition, in ONE `decide`: an escalated task is re-decided
    at ≥3/5 while an untouched one stays at ≥2/3 — and both feed the same
    bootstrap. `t2` and `t5` carry the IDENTICAL base pattern (2/3) and end
    with opposite verdicts, which is the whole point of the rule."""
    db, _pre, post = await escalated_run(tmp_path)
    grid = load_grid(db, "run-1")

    assert grid.cell("t2", "rlm") == [True, True, False]              # 2/3
    assert grid.cell("t5", "rlm") == [True, True, False, False, False]  # 2/5
    assert grid.escalated_tasks == ("t3", "t4", "t5")

    assert "t2" in post.passes["rlm"]        # 2/3 passes
    assert "t5" not in post.passes["rlm"]    # the same 2 successes, now 2/5
    assert post.scores[("rlm", "t2")] == pytest.approx(2 / 3)
    assert post.scores[("rlm", "t5")] == pytest.approx(2 / 5)

    pair = post.pairs["b1"]
    assert pair.margin == 2 and pair.discordant == ("t3", "t4")
    lo, hi = pair.ci                         # one bootstrap over mixed fractions
    assert lo <= pair.mean_delta <= hi
    assert post.escalation_plan == {}        # §8 escalates once, never twice
    assert pair.escalates is True            # ...though the margin IS in the band
    text = render_report(post, None, {})
    assert "escalates **once**" in text and "PROVISIONAL" not in text


async def test_a_pre_escalation_gate_is_labelled_provisional(tmp_path):
    """A gate decided on a grid that still owes seeds {4,5} is not §8's final
    decision, and the report must not present it as one."""
    spec = clean_pass_spec()
    spec["t6"]["rlm"] = "FFF"                        # +3: passes AND escalates
    v = await _verdict(tmp_path, spec)
    text = render_report(v, None, {})
    assert "## S4 GATE: PASS" in text
    assert "PROVISIONAL: escalation owed" in text


async def test_the_escalated_grid_is_the_gate_not_the_pre_escalation_one(tmp_path):
    """§8 recomputes ONCE on the final grid and reports both. Here escalation
    REVERSES the gate: +3 (a pass) before, +2 (a failure) after. If the
    headline came from the pre-escalation verdict the two could be chosen
    between, which is exactly what "recomputed once" forbids."""
    _db, pre, post = await escalated_run(tmp_path)
    assert pre.gate_pass is True and post.gate_pass is False
    text = render_report(pre, None, {}, escalated=post)
    assert "## S4 GATE: FAIL" in text
    assert "PROVISIONAL" not in text
    assert "Post-escalation gate: FAIL" in text


async def test_the_findings_section_names_every_finding(tmp_path):
    spec = clean_pass_spec()
    for tid in spec:
        spec[tid]["b2"] = spec[tid]["rlm"]
    v = await _verdict(tmp_path, spec)
    text = render_report(v, None, {})
    assert "pivot" in text.lower()
    assert "B2" in text
