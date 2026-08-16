"""Task 9: `rlm/bench.py` -- §8's blocked (task, seed) scheduler.

SCHEDULING is what is under test here, so everything model-facing is a double:
no servers, no real arms, no `run_episode`. The arms arrive as recording
callables injected through `BenchCtx` -- which is not a test affordance but the
production wiring: `rlm/bench.py` sits on the isolated side of the dependency
rule (`tests/test_import_rules.py`), so the only way it can reach C4 at all is
through a callable someone else built.
"""
from __future__ import annotations

import ast
import copy
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from bench.manifest import BenchmarkManifest, TaskEntry

from rlm.bench import (
    ARM_ORDER,
    BENCH_PROFILE,
    CONFIG_REFUSED,
    RESIDENT_PROFILE,
    BenchCtx,
    BenchLedger,
    Block,
    assert_manifest_pinned,
    bench_extra,
    build_blocks,
    run_bench,
    run_block,
    seeded_config,
)
from rlm.config import Config
from rlm.episode import Task
from rlm.errors import ConfigError, Outcome
from rlm.power import PowerReading
from rlm.trace import TraceLogger, utc_now

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# doubles
# --------------------------------------------------------------------------- #


def _uuid() -> str:
    """A REAL uuid: `TraceLogger` silently swallows a malformed episode id, so
    a double that minted "ep-1" would make every supersede assertion vacuous
    against the real writer."""
    return str(uuid.uuid4())


@dataclass
class FakeResult:
    """`ArmResult`/`EpisodeResult`'s four facts -- the shape §8's grid is built
    from, and all the scheduler ever reads."""

    episode_id: str
    outcome: Outcome
    reason: str | None = None
    answer: str | None = "answer"


class FakeArms:
    """One recording runner per arm.

    `script` maps an arm to the sequence of things its calls produce: an
    `Outcome` (returned as a `FakeResult`) or an exception instance (raised).
    The last entry repeats, so a test can say "always errors" with one item.
    """

    def __init__(self, script: dict[str, list] | None = None) -> None:
        self.calls: list[dict] = []
        self._script = {k: list(v) for k, v in (script or {}).items()}

    def runner(self, arm: str):
        async def run(task, cfg, *, bench_extra):
            self.calls.append({
                "arm": arm,
                "task_id": task.task_id,
                "cfg_root_seed": cfg.scaffold.sampling.root.seed,
                "cfg_leaf_seed": cfg.scaffold.sampling.leaf.seed,
                "bench_extra": dict(bench_extra),
            })
            step = self._next(arm)
            if isinstance(step, BaseException):
                raise step
            return FakeResult(episode_id=_uuid(), outcome=step,
                              reason=None if step is Outcome.SUCCESS else "because")

        return run

    def runners(self, arms=ARM_ORDER) -> dict:
        return {a: self.runner(a) for a in arms}

    def _next(self, arm: str):
        seq = self._script.get(arm)
        if not seq:
            return Outcome.SUCCESS
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def order(self) -> list[str]:
        return [c["arm"] for c in self.calls]


class FakeTrace:
    def __init__(self) -> None:
        self.superseded: list[tuple] = []
        self.metrics: list[tuple] = []

    def mark_superseded(self, old_episode_id, new_episode_id) -> None:
        self.superseded.append((old_episode_id, new_episode_id))

    def update_episode_metrics(self, episode_id, **kw) -> None:
        self.metrics.append((episode_id, kw))


class FakeClock:
    """Monotonic-shaped: every read advances by `step`, so a call bracket
    measures exactly `step` unless something between the reads advanced it."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        now = self.t
        self.t += self.step
        return now

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeSampler:
    def __init__(self, readings: list[PowerReading], *, alive: bool = True) -> None:
        self._readings = list(readings)
        self._alive = alive

    def alive(self) -> bool:
        return self._alive

    def reading(self):
        if not self._readings:
            return None
        return self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]


class Hooks:
    """The three orchestration hooks Task 10 fills, as recorders."""

    def __init__(self, clock: FakeClock | None = None, *, swap_cost: float = 0.0,
                 handshake_error: Exception | None = None) -> None:
        self.events: list[tuple] = []
        self._clock = clock
        self._swap_cost = swap_cost
        self._handshake_error = handshake_error

    async def swap(self, profile: str) -> None:
        self.events.append(("swap", profile))
        if self._clock is not None:
            self._clock.advance(self._swap_cost)

    async def quiesce(self, profile: str):
        self.events.append(("quiesce", profile))
        if self._clock is not None:
            self._clock.advance(self._swap_cost)
        return {"root": True, "leaf": True}

    async def handshake(self, profile: str):
        self.events.append(("handshake", profile))
        if self._clock is not None:
            self._clock.advance(self._swap_cost)
        if self._handshake_error is not None:
            raise self._handshake_error
        return {"root": {}, "leaf": {}}

    def kinds(self, kind: str) -> list[str]:
        return [p for k, p in self.events if k == kind]


def _entry(task_id: str, *, category: str = "needle") -> TaskEntry:
    return TaskEntry(task_id=task_id, category=category,
                     task_file=f"bench/tasks/{task_id}.json",
                     corpus_path=f"bench/corpora/{task_id}.txt",
                     corpus_sha256="0" * 64, corpus_tokens=10,
                     corpus_date="2026-08-15", checker="int_exact",
                     question_sha256="1" * 64)


def _manifest(task_ids: list[str]) -> BenchmarkManifest:
    return BenchmarkManifest(benchmark_version="test", built_at="2026-08-16",
                             token_counter="approx-offline",
                             assumed_training_cutoff="2025-06",
                             tasks=[_entry(t) for t in task_ids])


def _task_loader(path):
    return Task(task_id=Path(path).stem, text="q", context="")


def _ctx(tmp_path: Path, cfg_dict: dict, *, arms: FakeArms, manifest=None,
         trace=None, clock=None, hooks=None, sampler=None, temp_fn=None,
         run_id: str = "run-1", store=None, ledger=None) -> BenchCtx:
    hooks = hooks or Hooks()
    return BenchCtx(
        raw_cfg=cfg_dict,
        cfg=Config.model_validate(copy.deepcopy(cfg_dict)),
        run_id=run_id,
        manifest=manifest if manifest is not None else _manifest(["t1"]),
        ledger=ledger if ledger is not None else BenchLedger(tmp_path / "ledger.jsonl"),
        trace=trace if trace is not None else FakeTrace(),
        store=store,
        arm_runners=arms.runners(),
        load_task_fn=_task_loader,
        quiesce_fn=hooks.quiesce,
        handshake_fn=hooks.handshake,
        swap_servers_fn=hooks.swap,
        sampler=sampler,
        temp_fn=temp_fn,
        clock=clock or FakeClock(),
        repo_root=tmp_path,
    )


@pytest.fixture
def bench_cfg_dict(minimal_cfg_dict: dict, tmp_path: Path) -> dict:
    raw = copy.deepcopy(minimal_cfg_dict)
    raw["trace"]["db_path"] = str(tmp_path / "rlm.duckdb")
    raw["trace"]["blob_root"] = str(tmp_path / "blobs")
    return raw


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


def test_build_blocks_is_task_major_and_seed_minor():
    """§8: '(task, seed) blocks adjacent in time across all arms' -- the seeds
    of one task run back to back so R9 thermal drift cancels inside the paired
    comparison, which task-minor ordering would spread across the whole run."""
    blocks = build_blocks(_manifest(["a", "b"]), [1, 2, 3])
    assert [(b.task_entry.task_id, b.seed) for b in blocks] == [
        ("a", 1), ("a", 2), ("a", 3), ("b", 1), ("b", 2), ("b", 3)]


def test_build_blocks_numbers_blocks_in_execution_order():
    blocks = build_blocks(_manifest(["a", "b"]), [1, 2])
    assert [b.idx for b in blocks] == [0, 1, 2, 3]
    assert all(isinstance(b, Block) for b in blocks)


def test_build_blocks_preserves_manifest_order():
    ids = ["z", "a", "m"]
    assert [b.task_entry.task_id for b in build_blocks(_manifest(ids), [1])] == ids


# --------------------------------------------------------------------------- #
# per-seed config synthesis
# --------------------------------------------------------------------------- #


def test_seeded_config_sets_both_root_and_leaf_seeds(bench_cfg_dict):
    cfg = seeded_config(bench_cfg_dict, 7)
    assert cfg.scaffold.sampling.root.seed == 7
    assert cfg.scaffold.sampling.leaf.seed == 7


def test_seeded_config_disables_per_episode_bundle_export(bench_cfg_dict):
    """Ledgered: a bundle export per episode is O(n^2) over a 360-episode run."""
    assert bench_cfg_dict["trace"]["export_every_episode"] is True
    assert seeded_config(bench_cfg_dict, 1).trace.export_every_episode is False


def test_seeded_config_does_not_mutate_the_caller_dict(bench_cfg_dict):
    before = copy.deepcopy(bench_cfg_dict)
    seeded_config(bench_cfg_dict, 9)
    assert bench_cfg_dict == before


def test_seeded_config_revalidates_the_whole_config(bench_cfg_dict):
    """Patch-the-raw-dict-and-revalidate (s1's `variant_config`): every
    cross-field rule still runs, so a bad synthesis refuses instead of
    running a config nobody validated."""
    raw = copy.deepcopy(bench_cfg_dict)
    raw["scaffold"]["dispatcher"] = "not-a-dispatcher"
    with pytest.raises(ConfigError):
        seeded_config(raw, 1)


# --------------------------------------------------------------------------- #
# the freeze, verified at runtime for the first time
# --------------------------------------------------------------------------- #


def test_assert_manifest_pinned_accepts_the_frozen_manifest():
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = Config.model_validate(raw)
    manifest = BenchmarkManifest.load(REPO_ROOT / "bench" / "manifest.json")
    assert_manifest_pinned(manifest, cfg)          # validates §8 too, no raise


def test_assert_manifest_pinned_refuses_a_manifest_that_moved(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    manifest = BenchmarkManifest.load(REPO_ROOT / "bench" / "manifest.json")
    manifest.tasks[0].corpus_sha256 = "f" * 64
    with pytest.raises(ConfigError, match="manifest_sha256"):
        assert_manifest_pinned(manifest, cfg)


def test_assert_manifest_pinned_refuses_an_unpinned_config(bench_cfg_dict):
    raw = copy.deepcopy(bench_cfg_dict)
    raw["benchmark"]["manifest_sha256"] = None
    cfg = Config.model_validate(raw)
    manifest = BenchmarkManifest.load(REPO_ROOT / "bench" / "manifest.json")
    with pytest.raises(ConfigError, match="manifest_sha256"):
        assert_manifest_pinned(manifest, cfg)


def test_assert_manifest_pinned_refuses_a_manifest_that_violates_section_8(bench_cfg_dict):
    """The pin and §8's rules are separate gates: a hand-built manifest whose
    sha matches its own bytes still has to be a legal benchmark."""
    manifest = _manifest(["a"])
    raw = copy.deepcopy(bench_cfg_dict)
    raw["benchmark"]["manifest_sha256"] = manifest.sha256
    with pytest.raises(ConfigError, match="§8"):
        assert_manifest_pinned(manifest, Config.model_validate(raw))


# --------------------------------------------------------------------------- #
# within-block order, profiles, quiesce/handshake
# --------------------------------------------------------------------------- #


async def test_within_block_arm_order_is_pre_registered(tmp_path, bench_cfg_dict):
    """§8: 'RLM then B2 on the resident topology, then one leaf relaunch serves
    B1 and B3 back-to-back'."""
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["b3", "b1", "rlm", "b2"], ctx)
    assert arms.order() == ["rlm", "b2", "b1", "b3"]
    assert ARM_ORDER == ("rlm", "b2", "b1", "b3")


async def test_requested_arms_are_filtered_and_keep_the_registered_order(
        tmp_path, bench_cfg_dict):
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["b3", "rlm"], ctx)
    assert arms.order() == ["rlm", "b3"]


async def test_a_full_block_relaunches_the_leaf_at_most_twice(tmp_path, bench_cfg_dict):
    """§8 bounds server relaunches at two per block. The single-shot arms run
    on the `bench_leaf` profile; RLM and B2 on the resident one.

    The swap back is LAZY -- it happens when the next block's RLM arm asks for
    the resident topology, not at the block boundary -- so N blocks cost 2N-1
    relaunches, inside §8's bound rather than at it. What the bound actually
    protects is that no arm ever runs on the other arm's profile, and that is
    asserted per arm below.
    """
    arms = FakeArms()
    hooks = Hooks()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, hooks=hooks)
    blocks = build_blocks(ctx.manifest, [1, 2])
    for block in blocks:
        await run_block(block, list(ARM_ORDER), ctx)
    swaps = hooks.kinds("swap")
    assert swaps == [BENCH_PROFILE, RESIDENT_PROFILE, BENCH_PROFILE]
    assert len(swaps) <= 2 * len(blocks)
    # …and every arm ran under its own profile, both blocks.
    assert hooks.kinds("handshake") == [RESIDENT_PROFILE, RESIDENT_PROFILE,
                                        BENCH_PROFILE, BENCH_PROFILE] * 2


async def test_a_resident_only_run_never_relaunches(tmp_path, bench_cfg_dict):
    hooks = Hooks()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), hooks=hooks)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)
    assert hooks.kinds("swap") == []


async def test_quiesce_and_handshake_precede_every_episode(tmp_path, bench_cfg_dict):
    """§4: the /props assertion re-runs per episode at the C5 quiesce point."""
    hooks = Hooks()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), hooks=hooks)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b1"], ctx)
    assert [k for k, _ in hooks.events] == [
        "quiesce", "handshake", "swap", "quiesce", "handshake"]


async def test_wall_clock_excludes_the_relaunch_and_the_handshake(tmp_path, bench_cfg_dict):
    """§8: 'relaunch time is excluded from per-task wall-clock'."""
    clock = FakeClock(step=1.0)
    hooks = Hooks(clock, swap_cost=10.0)
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), hooks=hooks, clock=clock)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b1"], ctx)
    assert [r["wall_s"] for r in records] == [1.0, 1.0]


async def test_a_handshake_refusal_aborts_the_run(tmp_path, bench_cfg_dict):
    """A server that relaunched with different flags must stop the run, not be
    recorded as one task's bad luck (§4)."""
    arms = FakeArms()
    hooks = Hooks(handshake_error=ConfigError("leaf n_parallel mismatch"))
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, hooks=hooks)
    with pytest.raises(ConfigError):
        await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)
    assert arms.calls == []


# --------------------------------------------------------------------------- #
# episode identity
# --------------------------------------------------------------------------- #


def test_only_the_rlm_arm_carries_its_own_arm_key():
    """`ArmEpisode.snapshot` writes `{"arm": self.arm, **bench_extra}` itself,
    so a baseline that also carried one would be asserting a fact it does not
    own. `run_episode` has no arm concept at all, so the RLM arm's must come
    from here."""
    assert bench_extra("r", 3, 2, "b1") == {"run_id": "r", "block": 3, "seed": 2}
    assert bench_extra("r", 3, 2, "rlm") == {"run_id": "r", "block": 3, "seed": 2,
                                             "arm": "rlm"}


async def test_each_episode_runs_under_its_own_seeded_config(tmp_path, bench_cfg_dict):
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    for block in build_blocks(ctx.manifest, [1, 2]):
        await run_block(block, ["rlm"], ctx)
    assert [c["cfg_root_seed"] for c in arms.calls] == [1, 2]
    assert [c["cfg_leaf_seed"] for c in arms.calls] == [1, 2]
    assert [c["bench_extra"]["seed"] for c in arms.calls] == [1, 2]
    assert [c["bench_extra"]["block"] for c in arms.calls] == [0, 1]


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


async def test_the_ledger_records_one_row_per_episode(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "results" / "ledger.jsonl")
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), ledger=ledger)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)
    rows = ledger.read()
    assert len(rows) == 2
    assert set(rows[0]) == {"run_id", "block", "task_id", "seed", "arm", "episode_id",
                            "outcome", "reason", "wall_s", "superseded_by", "ts"}
    assert [r["arm"] for r in rows] == ["rlm", "b2"]
    assert rows[0]["run_id"] == "run-1" and rows[0]["task_id"] == "t1"
    assert rows[0]["outcome"] == "success" and rows[0]["superseded_by"] is None


async def test_completed_reads_the_ledger_back_keyed_by_tuple(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), ledger=ledger)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)
    done = ledger.completed("run-1")
    assert set(done) == {("t1", 1, "rlm"), ("t1", 1, "b2")}
    assert done[("t1", 1, "rlm")]["outcome"] == "success"
    assert ledger.completed("another-run") == {}


async def test_completed_merges_the_store_and_the_store_wins(tmp_path, bench_cfg_dict):
    """The ledger is a crash-resilient mirror, not the record of truth: the
    store is what the verdict is computed from, so where they disagree the
    store's row is the one that counts."""
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    ep = _uuid()
    ledger.append({"run_id": "r", "block": 0, "task_id": "t1", "seed": 1, "arm": "rlm",
                   "episode_id": ep, "outcome": "error", "reason": "arm_error",
                   "wall_s": 3.0, "superseded_by": None})

    trace = TraceLogger(tmp_path / "rlm.duckdb", tmp_path / "blobs")
    await trace.start()
    try:
        trace.open_episode({"episode_id": ep, "task_id": "t1", "task_hash": "h",
                            "started_at": utc_now(),
                            "config_snapshot": {"bench": {"run_id": "r", "seed": 1,
                                                          "arm": "rlm", "block": 0}}})
        trace.close_episode(ep, Outcome.SUCCESS, None)
        await trace.drain()
        done = ledger.completed("r", store=trace.monitor())
    finally:
        await trace.aclose()
    assert done[("t1", 1, "rlm")]["outcome"] == "success"
    assert done[("t1", 1, "rlm")]["wall_s"] == 3.0        # ledger-only column survives


def test_completed_withholds_a_lone_error_until_its_rerun(tmp_path):
    """§8 owes every `error` episode exactly one rerun. A run that died between
    the two must resume into the rerun, not skip the cell as decided."""
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    first = {"run_id": "r", "block": 0, "task_id": "t1", "seed": 1, "arm": "b1",
             "episode_id": _uuid(), "outcome": "error", "reason": "arm_error",
             "wall_s": 1.0, "superseded_by": None}
    ledger.append(first)
    assert ledger.completed("r") == {}
    assert set(ledger.open_errors("r")) == {("t1", 1, "b1")}

    ledger.mark_superseded(first, "0" * 8)                 # not a uuid: ledger-side only
    ledger.append({**first, "episode_id": _uuid(), "superseded_by": None})
    assert set(ledger.completed("r")) == {("t1", 1, "b1")}
    assert ledger.open_errors("r") == {}


def test_a_refusal_does_not_count_as_a_spent_rerun(tmp_path):
    """A cell that refused, was fixed, then errored once has had ONE draw --
    counting the refusal row as the second would retire the rerun §8 owes it."""
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    base = {"run_id": "r", "block": 0, "task_id": "t1", "seed": 1, "arm": "b1",
            "outcome": "error", "wall_s": 0.0, "superseded_by": None}
    ledger.append({**base, "episode_id": None, "reason": CONFIG_REFUSED})
    ep = _uuid()
    ledger.append({**base, "episode_id": ep, "reason": "arm_error"})
    assert ledger.completed("r") == {}
    assert ledger.open_errors("r")[("t1", 1, "b1")]["episode_id"] == ep


def test_completed_ignores_superseded_rows(tmp_path):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    old = {"run_id": "r", "block": 0, "task_id": "t1", "seed": 1, "arm": "b1",
           "episode_id": _uuid(), "outcome": "error", "reason": None,
           "wall_s": 1.0, "superseded_by": None}
    new_id = _uuid()
    ledger.append(old)
    ledger.mark_superseded(old, new_id)
    ledger.append({**old, "episode_id": new_id, "outcome": "success",
                   "superseded_by": None})
    done = ledger.completed("r")
    assert done[("t1", 1, "b1")]["episode_id"] == new_id


# --------------------------------------------------------------------------- #
# rerun-once
# --------------------------------------------------------------------------- #


async def test_an_error_is_rerun_once_and_linked(tmp_path, bench_cfg_dict):
    arms = FakeArms({"b1": [Outcome.ERROR, Outcome.SUCCESS]})
    trace = FakeTrace()
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace, ledger=ledger)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["b1"], ctx)

    assert arms.order() == ["b1", "b1"]
    assert [r["outcome"] for r in records] == ["error", "success"]
    assert len(trace.superseded) == 1
    old, new = trace.superseded[0]
    assert (old, new) == (records[0]["episode_id"], records[1]["episode_id"])
    uuid.UUID(old), uuid.UUID(new)                          # real ids, both ways
    assert ledger.completed("run-1")[("t1", 1, "b1")]["outcome"] == "success"


async def test_a_second_error_stands(tmp_path, bench_cfg_dict):
    arms = FakeArms({"b1": [Outcome.ERROR]})
    trace = FakeTrace()
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace, ledger=ledger)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["b1", "b3"], ctx)

    assert arms.order() == ["b1", "b1", "b3"]              # exactly one rerun
    assert len(trace.superseded) == 1
    assert [r["outcome"] for r in records] == ["error", "error", "success"]
    assert ledger.completed("run-1")[("t1", 1, "b1")]["outcome"] == "error"


async def test_a_failed_episode_is_never_rerun(tmp_path, bench_cfg_dict):
    """§8: budget_kill/context_exhausted/fail count as failures for every arm.
    Only `error` buys a second draw."""
    arms = FakeArms({"rlm": [Outcome.BUDGET_KILL], "b2": [Outcome.FAIL],
                     "b1": [Outcome.CONTEXT_EXHAUSTED]})
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2", "b1"], ctx)
    assert arms.order() == ["rlm", "b2", "b1"]
    assert trace.superseded == []


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


async def test_resume_skips_tuples_already_decided(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    first = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=first, ledger=ledger,
               manifest=_manifest(["t1", "t2"]))
    await run_bench(ctx, arms=["rlm", "b2"], seeds=[1])

    second = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=second, ledger=ledger,
                manifest=_manifest(["t1", "t2"]))
    await run_bench(ctx2, arms=["rlm", "b2"], seeds=[1])
    assert second.calls == []
    assert len(first.calls) == 4

    third = FakeArms()
    ctx3 = _ctx(tmp_path, bench_cfg_dict, arms=third, ledger=ledger,
                manifest=_manifest(["t1", "t2"]))
    await run_bench(ctx3, arms=list(ARM_ORDER), seeds=[1])
    assert third.order() == ["b1", "b3", "b1", "b3"]        # only the new arms


async def test_resume_runs_the_rerun_a_crashed_run_still_owed(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    lone = {"run_id": "run-1", "block": 0, "task_id": "t1", "seed": 1, "arm": "rlm",
            "episode_id": _uuid(), "outcome": "error", "reason": "arm_error",
            "wall_s": 2.0, "superseded_by": None}
    ledger.append(lone)

    arms = FakeArms()
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace, ledger=ledger)
    records = await run_bench(ctx, arms=["rlm"], seeds=[1])

    assert arms.order() == ["rlm"]                          # the rerun, and only it
    assert trace.superseded == [(lone["episode_id"], records[0]["episode_id"])]
    assert ledger.completed("run-1")[("t1", 1, "rlm")]["outcome"] == "success"


async def test_a_different_run_id_shares_no_resume_state(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), ledger=ledger)
    await run_bench(ctx, arms=["rlm"], seeds=[1])
    other = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=other, ledger=ledger, run_id="run-2")
    await run_bench(ctx2, arms=["rlm"], seeds=[1])
    assert other.order() == ["rlm"]


# --------------------------------------------------------------------------- #
# refusals: contained, but never scored as a measurement
# --------------------------------------------------------------------------- #


async def test_a_config_refusal_is_contained_and_the_grid_continues(
        tmp_path, bench_cfg_dict):
    """`run_b1`/`run_b3` refuse before opening a row (missing `bench_leaf`, an
    unreadable corpus). One task's refusal must not abort a 39-hour grid."""
    arms = FakeArms({"b1": [ConfigError("servers.bench_leaf is not configured")]})
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["b1", "b3"], ctx)

    assert arms.order() == ["b1", "b3"]                     # no rerun, no abort
    assert records[0]["outcome"] == "error"
    assert records[0]["reason"] == CONFIG_REFUSED
    assert records[0]["episode_id"] is None
    assert trace.superseded == [] and trace.metrics == []


async def test_a_refusal_leaves_the_cell_open_for_a_resumed_run(tmp_path, bench_cfg_dict):
    """A refusal is a run-configuration fault, not an arm outcome: there is no
    episode row to score, so the cell stays unfilled until the fault is fixed
    and the run resumed."""
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    broken = FakeArms({"b1": [ConfigError("no bench_leaf")]})
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=broken, ledger=ledger)
    await run_bench(ctx, arms=["b1"], seeds=[1])
    assert ledger.completed("run-1") == {}

    fixed = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=fixed, ledger=ledger)
    await run_bench(ctx2, arms=["b1"], seeds=[1])
    assert fixed.order() == ["b1"]
    assert ledger.completed("run-1")[("t1", 1, "b1")]["outcome"] == "success"


async def test_an_unreadable_task_file_refuses_every_arm_in_the_block(
        tmp_path, bench_cfg_dict):
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)

    def boom(path):
        raise ConfigError(f"cannot read task file {path}")

    ctx.load_task_fn = boom
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)
    assert arms.calls == []
    assert [r["arm"] for r in records] == ["rlm", "b2"]
    assert {r["reason"] for r in records} == {CONFIG_REFUSED}


async def test_an_unexpected_exception_aborts_the_run(tmp_path, bench_cfg_dict):
    """`run_b1` re-raises after closing its row on purpose: a bug in an arm must
    not be scoreable as an ordinary `error` episode."""
    arms = FakeArms({"rlm": [RuntimeError("bug")]})
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    with pytest.raises(RuntimeError, match="bug"):
        await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)


async def test_a_config_that_will_not_synthesise_aborts_the_run(tmp_path, bench_cfg_dict):
    """The raw dict is the same for all 90 blocks: recording it as one cell's
    refusal, ninety times, would bury a run-level fault in the grid."""
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    ctx.raw_cfg = copy.deepcopy(bench_cfg_dict)
    ctx.raw_cfg["scaffold"]["dispatcher"] = "not-a-dispatcher"
    with pytest.raises(ConfigError):
        await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)
    assert ctx.ledger.read() == [] and arms.calls == []


async def test_an_unknown_arm_is_refused_before_the_grid_starts(tmp_path, bench_cfg_dict):
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms())
    with pytest.raises(ConfigError, match="b4"):
        await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b4"], ctx)


async def test_an_arm_with_no_runner_is_refused(tmp_path, bench_cfg_dict):
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms)
    ctx.arm_runners = {"rlm": arms.runner("rlm")}
    with pytest.raises(ConfigError, match="b2"):
        await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b2"], ctx)


# --------------------------------------------------------------------------- #
# power / temperature bracketing
# --------------------------------------------------------------------------- #


async def test_power_is_stamped_from_the_energy_delta(tmp_path, bench_cfg_dict):
    """`power_mw` reads garbage on this box: `avg_power_w` is DERIVED from the
    energy delta over the measured duration, never taken from the sampler."""
    sampler = FakeSampler([
        PowerReading(ts=0.0, energy_pwh=0, power_mw=999_999.0),
        PowerReading(ts=2.0, energy_pwh=1_000_000_000, power_mw=999_999.0),
    ])
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace,
               sampler=sampler, clock=FakeClock(step=2.0),
               temp_fn=lambda: 51.5)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)

    assert len(trace.metrics) == 1
    episode_id, cols = trace.metrics[0]
    assert episode_id == records[0]["episode_id"]
    assert cols["energy_j"] == pytest.approx(3.6)
    assert cols["avg_power_w"] == pytest.approx(3.6 / 2.0)
    assert cols["pkg_temp_c_start"] == 51.5 and cols["pkg_temp_c_end"] == 51.5


async def test_a_dead_sampler_records_nulls(tmp_path, bench_cfg_dict):
    """The sampler's known failure mode is dying silently at launch. NULLs are
    the honest record; a fabricated number is not."""
    sampler = FakeSampler([PowerReading(ts=0.0, energy_pwh=5, power_mw=42.0)],
                          alive=False)
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace, sampler=sampler)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)
    assert trace.metrics == []


async def test_temperature_is_stamped_without_a_power_sampler(tmp_path, bench_cfg_dict):
    """R9's per-episode temperature is not gated on the power-overhead check."""
    trace = FakeTrace()
    temps = iter([40.0, 44.0])
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace,
               temp_fn=lambda: next(temps))
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)
    assert trace.metrics[0][1] == {"pkg_temp_c_start": 40.0, "pkg_temp_c_end": 44.0,
                                   "avg_power_w": None, "energy_j": None}


# --------------------------------------------------------------------------- #
# layering
# --------------------------------------------------------------------------- #


def test_bench_imports_neither_c4_nor_the_benchmark_package():
    """Two rules in one assertion. `rlm.episode` reaches C4, so importing it
    would put an HTTP client behind the scheduler (`rlm/arms.py`'s rule, and
    `tests/test_import_rules.py` lists bench.py as isolated). And `bench/` is
    not in the shipped wheel (`pyproject.toml` packages = ["rlm"]), so a
    runtime import of the manifest module would break the installed package."""
    tree = ast.parse((REPO_ROOT / "rlm" / "bench.py").read_text(encoding="utf-8"))
    runtime: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                runtime.update(a.name for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
                runtime.add(sub.module)
    assert "rlm.episode" not in runtime
    assert not any(m == "bench" or m.startswith("bench.") for m in runtime)


def test_the_ledger_default_path_is_the_pre_registered_one():
    from rlm.bench import LEDGER_PATH

    assert LEDGER_PATH.parts[-3:] == ("s4", "results", "ledger.jsonl")
