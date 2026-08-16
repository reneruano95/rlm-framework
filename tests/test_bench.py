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
from rlm.episode import Task, handshake
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


def frozen_manifest() -> BenchmarkManifest:
    """The REAL frozen manifest. `run_bench` verifies the pin itself, so every
    test that goes through it runs against the artifact S4 will schedule --
    a hand-built manifest is refused there, by design."""
    return BenchmarkManifest.load(REPO_ROOT / "bench" / "manifest.json")


def _two_blocks() -> list[Block]:
    """The first two blocks of the frozen grid: enough to see task-major
    ordering and resume, without running 30 tasks of doubles."""
    return build_blocks(frozen_manifest(), [1])[:2]


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
    blocks = _two_blocks()
    first = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=first, ledger=ledger,
               manifest=frozen_manifest())
    await run_bench(ctx, arms=["rlm", "b2"], blocks=blocks)

    second = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=second, ledger=ledger,
                manifest=frozen_manifest())
    await run_bench(ctx2, arms=["rlm", "b2"], blocks=blocks)
    assert second.calls == []
    assert len(first.calls) == 4

    third = FakeArms()
    ctx3 = _ctx(tmp_path, bench_cfg_dict, arms=third, ledger=ledger,
                manifest=frozen_manifest())
    await run_bench(ctx3, arms=list(ARM_ORDER), blocks=blocks)
    assert third.order() == ["b1", "b3", "b1", "b3"]        # only the new arms


async def test_resume_runs_the_rerun_a_crashed_run_still_owed(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    block = _two_blocks()[0]
    lone = {"run_id": "run-1", "block": 0, "task_id": block.task_entry.task_id,
            "seed": 1, "arm": "rlm", "episode_id": _uuid(), "outcome": "error",
            "reason": "arm_error", "wall_s": 2.0, "superseded_by": None}
    ledger.append(lone)

    arms = FakeArms()
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, trace=trace, ledger=ledger,
               manifest=frozen_manifest())
    records = await run_bench(ctx, arms=["rlm"], blocks=[block])

    assert arms.order() == ["rlm"]                          # the rerun, and only it
    assert trace.superseded == [(lone["episode_id"], records[0]["episode_id"])]
    cell = (block.task_entry.task_id, 1, "rlm")
    assert ledger.completed("run-1")[cell]["outcome"] == "success"


async def test_a_different_run_id_shares_no_resume_state(tmp_path, bench_cfg_dict):
    ledger = BenchLedger(tmp_path / "ledger.jsonl")
    blocks = _two_blocks()[:1]
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), ledger=ledger,
               manifest=frozen_manifest())
    await run_bench(ctx, arms=["rlm"], blocks=blocks)
    other = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=other, ledger=ledger, run_id="run-2",
                manifest=frozen_manifest())
    await run_bench(ctx2, arms=["rlm"], blocks=blocks)
    assert other.order() == ["rlm"]


async def test_run_bench_refuses_a_manifest_that_is_not_the_frozen_one(
        tmp_path, bench_cfg_dict):
    """The startup assertion lives where the episodes are: a caller that
    forgot `assert_manifest_pinned` must not be able to score a 39-hour run
    against a manifest nobody verified."""
    arms = FakeArms()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=arms, manifest=_manifest(["t1"]))
    with pytest.raises(ConfigError, match="manifest_sha256"):
        await run_bench(ctx, arms=["rlm"], seeds=[1])
    assert arms.calls == [] and ctx.ledger.read() == []


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
    blocks = _two_blocks()[:1]
    broken = FakeArms({"b1": [ConfigError("no bench_leaf")]})
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=broken, ledger=ledger,
               manifest=frozen_manifest())
    await run_bench(ctx, arms=["b1"], blocks=blocks)
    assert ledger.completed("run-1") == {}

    fixed = FakeArms()
    ctx2 = _ctx(tmp_path, bench_cfg_dict, arms=fixed, ledger=ledger,
                manifest=frozen_manifest())
    await run_bench(ctx2, arms=["b1"], blocks=blocks)
    assert fixed.order() == ["b1"]
    cell = (blocks[0].task_entry.task_id, 1, "b1")
    assert ledger.completed("run-1")[cell]["outcome"] == "success"


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
    energy delta, never taken from the sampler.

    The denominator is the READINGS' interval (3 s here), not `wall_s` (2 s):
    the sampler publishes at 1 Hz, so the bracket reads sit inside a window
    offset from the episode's at both ends, and dividing by `wall_s` would
    report watts nothing measured.
    """
    sampler = FakeSampler([
        PowerReading(ts=10.0, energy_pwh=0, power_mw=999_999.0),
        PowerReading(ts=13.0, energy_pwh=1_000_000_000, power_mw=999_999.0),
    ])
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace,
               sampler=sampler, clock=FakeClock(step=2.0),
               temp_fn=lambda: 51.5)
    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)

    assert records[0]["wall_s"] == 2.0
    assert len(trace.metrics) == 1
    episode_id, cols = trace.metrics[0]
    assert episode_id == records[0]["episode_id"]
    assert cols["energy_j"] == pytest.approx(3.6)
    assert cols["avg_power_w"] == pytest.approx(3.6 / 3.0)      # NOT 3.6 / wall_s
    assert cols["pkg_temp_c_start"] == 51.5 and cols["pkg_temp_c_end"] == 51.5


async def test_an_unchanged_reading_stamps_no_power(tmp_path, bench_cfg_dict):
    """1 Hz sampling against an episode shorter than the sample interval: both
    bracket reads return the SAME cached reading. A zero delta over a real
    episode is not a measurement of zero — stamping 0.0 W would be
    indistinguishable from one at scoring time, which is the fabrication §8
    forbids. The temperature, which IS a fresh read, still lands."""
    cached = PowerReading(ts=7.0, energy_pwh=5_000, power_mw=118_000.0)
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace,
               sampler=FakeSampler([cached]), temp_fn=lambda: 42.0)
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)

    assert trace.metrics[0][1] == {"pkg_temp_c_start": 42.0, "pkg_temp_c_end": 42.0,
                                   "avg_power_w": None, "energy_j": None}


async def test_a_stale_reading_stamps_nothing_at_all(tmp_path, bench_cfg_dict):
    """…and with no temperature source either, the call is skipped entirely
    (`update_episode_metrics` writes only non-None columns, so the two are the
    same NULLs — this pins that no zero is ever manufactured)."""
    cached = PowerReading(ts=7.0, energy_pwh=5_000, power_mw=118_000.0)
    trace = FakeTrace()
    ctx = _ctx(tmp_path, bench_cfg_dict, arms=FakeArms(), trace=trace,
               sampler=FakeSampler([cached]))
    await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm"], ctx)
    assert trace.metrics == []


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


# --------------------------------------------------------------------------- #
# ServerOrchestra (Task 10): FakeProcess/fake-client doubles, no real servers.
#
# `ServerOrchestra` fills bench.py's `quiesce_fn`/`handshake_fn`/
# `swap_servers_fn` hooks -- but the lint (`tests/test_import_rules.py`)
# forces the CLASS ITSELF to live in `rlm/cli.py`, not a new module: it needs
# `ServerClient` for the §4 handshake, and `FORBIDDEN_RLM` bans `rlm.dispatcher`
# from any module that would have to join `ISOLATED` (see the module docstring
# on `rlm.cli.ServerOrchestra`). These tests import it from there and exercise
# it exactly as `rlm.bench` will: through fakes, no servers, no network.
# --------------------------------------------------------------------------- #

from rlm.cli import (  # noqa: E402 -- grouped with the rest of this section
    HandshakingProcessManager,
    ServerOrchestra,
    bench_dispatcher,
    bench_leaf_config,
)
from rlm.episode import assert_props
from rlm.errors import ServerRotationError


def _fake_props(server_cfg) -> dict:
    """A `/props` body a real llama-server launched with `server_cfg` would
    answer with, as far as `rlm.episode.assert_props` looks: model_path,
    total_slots, per-slot n_ctx, build_info."""
    return {
        "model_path": str(server_cfg.model),
        "total_slots": server_cfg.parallel,
        "default_generation_settings": {"n_ctx": server_cfg.ctx // server_cfg.parallel},
        # Matches the sample `-lv 4` build line used below (D27 cache-type
        # tests): `log_is_current` requires BOTH the commit and the number
        # parsed from the log to appear in this string.
        "build_info": "10375 (ba360efe1)",
    }


class FakeWorld:
    """Shared state across one test's `FakeProcess`/`FakeClient` doubles:
    which `ServerConfig` (if any) is answering on each port right now. Leaf
    and bench_leaf SHARE port 8081, so a `/props` probe through the fake
    client must reflect whichever one most recently started -- the same
    coupling `assert_props` exists to catch for real."""

    def __init__(self) -> None:
        self.log: list[tuple] = []              # (event, port, parallel), in order
        self.live: dict[int, Any] = {}           # port -> ServerConfig live there
        self.force_kill_calls: list[int] = []


class FakeProcess:
    """`LlamaServerProcess`'s shape (`start`/`stop`/`restart`, `.owned`),
    enough for `ServerOrchestra` to drive with no real server anywhere."""

    def __init__(self, server_cfg: Any, world: FakeWorld, **_kw: Any) -> None:
        self.server_cfg = server_cfg
        self._world = world
        self.owned = False

    async def start(self) -> None:
        self._world.log.append(("start", self.server_cfg.port, self.server_cfg.parallel))
        self._world.live[self.server_cfg.port] = self.server_cfg
        self.owned = True

    async def stop(self) -> None:
        self._world.log.append(("stop", self.server_cfg.port, self.server_cfg.parallel))
        if self._world.live.get(self.server_cfg.port) is self.server_cfg:
            del self._world.live[self.server_cfg.port]
        self.owned = False

    async def restart(self) -> None:
        self._world.log.append(("restart", self.server_cfg.port, self.server_cfg.parallel))
        self._world.live[self.server_cfg.port] = self.server_cfg
        self.owned = True


class FakeClient:
    """`ServerClient`'s shape, enough for the REAL `handshake`/`assert_props`
    logic to run against synthesized `/props` bodies -- so these tests prove
    the WIRING (which config a probe checks against), not merely that a mock
    was called."""

    def __init__(self, base_url: str, world: FakeWorld, **_kw: Any) -> None:
        self.base_url = base_url
        self._world = world
        self._port = int(base_url.rsplit(":", 1)[1])
        self.closed = False

    async def props(self) -> dict:
        cfg = self._world.live.get(self._port)
        if cfg is None:
            raise ConnectionError(f"nothing live on port {self._port}")
        return _fake_props(cfg)

    async def health(self) -> bool:
        return self._world.live.get(self._port) is not None

    async def aclose(self) -> None:
        self.closed = True


async def _fake_force_kill(port: int, world: FakeWorld) -> None:
    world.force_kill_calls.append(port)
    world.live.pop(port, None)


def _orchestra(cfg: Config, world: FakeWorld, *, launch: bool = True,
               lifecycle: Any = None, handshake_fn=None, cache_check_fn=None) -> ServerOrchestra:
    kwargs: dict[str, Any] = dict(
        launch=launch, lifecycle=lifecycle,
        process_factory=lambda server_cfg, **kw: FakeProcess(server_cfg, world),
        client_factory=lambda url, **kw: FakeClient(url, world),
        force_kill_fn=lambda port: _fake_force_kill(port, world),
        cache_check_fn=cache_check_fn or (lambda *a, **k: True),
    )
    if handshake_fn is not None:
        kwargs["handshake_fn"] = handshake_fn
    return ServerOrchestra(cfg, **kwargs)


# -- start/stop order, port-conflict prevention ------------------------------ #


async def test_start_resident_starts_root_then_leaf_in_order(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    assert world.log == [("start", 8080, 1), ("start", 8081, 128)]
    assert orch.current_profile == RESIDENT_PROFILE
    assert orch.root_proc is not None and orch.leaf_proc is not None


async def test_to_bench_leaf_stops_the_resident_leaf_before_starting_bench_leaf(
        bench_cfg_dict):
    """§8: bench_leaf SHARES port 8081 with the RLM leaf, so starting it
    while the resident leaf is still up would either bind-fail or leave two
    servers answering. The stop must be ordered strictly before the start."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    world.log.clear()

    await orch.to_bench_leaf()
    assert world.log == [("stop", 8081, 128), ("start", 8081, 2)]
    assert orch.current_profile == BENCH_PROFILE
    assert world.live[8081] is cfg.servers.bench_leaf


async def test_to_resident_leaf_reverses_the_swap(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    await orch.to_bench_leaf()
    world.log.clear()

    await orch.to_resident_leaf()
    assert world.log == [("stop", 8081, 2), ("start", 8081, 128)]
    assert orch.current_profile == RESIDENT_PROFILE


async def test_swap_to_is_the_swap_servers_fn_shape_bench_wires_up(bench_cfg_dict):
    """`swap_to(profile) -> Awaitable[Any]` matches `BenchCtx.swap_servers_fn`
    exactly -- Task 12 wires `swap_servers_fn=orchestra.swap_to` directly."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()

    await orch.swap_to(BENCH_PROFILE)
    assert orch.current_profile == BENCH_PROFILE
    await orch.swap_to(RESIDENT_PROFILE)
    assert orch.current_profile == RESIDENT_PROFILE
    with pytest.raises(ConfigError, match="unknown server profile"):
        await orch.swap_to("nonsense")


async def test_bring_up_leaf_is_a_no_op_already_on_the_target_profile(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    world.log.clear()

    relaunch_s = await orch.to_resident_leaf()
    assert relaunch_s == 0.0
    assert world.log == []          # nothing stopped, nothing started again


async def test_stop_all_stops_leaf_then_root(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    world.log.clear()

    await orch.stop_all()
    assert world.log == [("stop", 8081, 128), ("stop", 8080, 1)]
    assert orch.leaf_proc is None and orch.root_proc is None
    assert orch.current_profile is None


# -- handshake against the ARM-SPECIFIC ServerConfig ------------------------- #


async def test_handshake_after_a_swap_checks_the_arm_specific_config(bench_cfg_dict):
    """The ruling this guards: passing `cfg.servers.leaf` (128 slots) for a
    `bench_leaf` process (2 slots) fails `total_slots` and would wrongly
    refuse every B1/B3 episode. `to_bench_leaf` must succeed -- proving it
    handshaked against `servers.bench_leaf`, not `servers.leaf`."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()

    await orch.to_bench_leaf()          # would raise ConfigError if mischecked

    # And the contrast that proves it: the SAME live props, checked against
    # the WRONG config, really do fail -- so a correct implementation had to
    # choose `servers.bench_leaf` deliberately, not by accident.
    bench_props = _fake_props(cfg.servers.bench_leaf)
    with pytest.raises(ConfigError, match="total_slots"):
        assert_props(bench_props, cfg.servers.leaf, "leaf")


async def test_handshake_profile_probes_root_and_the_named_leaf(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()

    result = await orch.handshake_profile(RESIDENT_PROFILE)
    assert set(result) == {"root", "leaf"}

    await orch.to_bench_leaf()
    result = await orch.handshake_profile(BENCH_PROFILE)
    assert set(result) == {"root", "bench_leaf"}


# -- D27 gap: bench_leaf's cache types are never checked by `rlm validate` --- #


async def test_a_bench_leaf_cache_type_mismatch_refuses_the_swap(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world, cache_check_fn=lambda *a, **k: False)
    await orch.start_resident()

    with pytest.raises(ConfigError, match="cache type"):
        await orch.to_bench_leaf()
    # The process really did come up (the mismatch is in the LOG, not the
    # process) -- current_profile reflects what is actually live.
    assert orch.current_profile == BENCH_PROFILE


async def test_the_cache_check_receives_exactly_the_bench_leaf_role(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    seen: list[tuple] = []

    def spy(cfg_, probed, out, err, *, probe_ran, roles=("root", "leaf")):
        seen.append((tuple(probed), roles, probe_ran))
        return True

    orch = _orchestra(cfg, world, cache_check_fn=spy)
    await orch.start_resident()
    await orch.to_bench_leaf()
    assert seen == [(("bench_leaf",), ("bench_leaf",), True)]


async def test_a_real_launch_log_satisfies_the_bench_leaf_cache_check(
        tmp_path, bench_cfg_dict):
    """End to end with the REAL `_check_cache_types`/`parse_launch_log` (D27's
    actual verification, reused rather than reinvented): a `-lv 4` log
    matching `bench_leaf`'s configured cache_type/flash_attn is accepted."""
    from rlm.cli import _check_cache_types

    raw = copy.deepcopy(bench_cfg_dict)
    log_path = tmp_path / "leaf-server-bench.log"
    raw["servers"]["bench_leaf"]["log_path"] = str(log_path)
    cfg = Config.model_validate(raw)
    bl = cfg.servers.bench_leaf
    log_path.write_text(
        "build: 10375 (ba360efe1) with clang for x86_64-pc-windows-msvc\n"
        f"llama_context: flash_attn = {'enabled' if bl.flash_attn == 'on' else 'disabled'}\n"
        f"llama_kv_cache: size =  1088.00 MiB (  32768 cells,  16 layers,  "
        f"1/1 seqs), K ({bl.cache_type}):  544.00 MiB, V ({bl.cache_type}):  544.00 MiB\n",
        encoding="utf-8")

    world = FakeWorld()
    orch = _orchestra(cfg, world, cache_check_fn=_check_cache_types)
    await orch.start_resident()
    await orch.to_bench_leaf()          # does not raise
    assert orch.current_profile == BENCH_PROFILE


# -- the ProcessManager run_b2/run_episode rotate mid-episode ---------------- #


async def test_resident_process_manager_rehandshakes_after_restart(bench_cfg_dict):
    """Ledgered ruling: `run_b2`/`run_episode`'s injected `process_manager`
    must WRAP `restart()` to also re-run the leaf handshake before returning
    -- `arms.py` cannot itself speak HTTP (the dependency rule), so this is
    the ONLY place §5's full rotation contract can be restored for `run_b2`.
    """
    handshake_calls: list[str] = []

    async def counting_handshake(client, server_cfg, role, lifecycle):
        handshake_calls.append(role)
        return await handshake(client, server_cfg, role, lifecycle)

    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world, handshake_fn=counting_handshake)
    await orch.start_resident()
    handshake_calls.clear()
    world.log.clear()

    pm = orch.resident_process_manager()
    assert isinstance(pm, HandshakingProcessManager)
    await pm.restart()

    assert world.log == [("restart", 8081, 128)]
    assert handshake_calls == ["leaf"]     # re-handshake ran AFTER the restart


async def test_resident_process_manager_refuses_when_nothing_is_owned(bench_cfg_dict):
    """No `start_resident()` ever ran -- there is no process to rotate, and
    `HandshakingProcessManager` must refuse rather than silently doing
    nothing (which `rlm.serverproc.ProcessManager`'s own contract forbids)."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    pm = orch.resident_process_manager()
    with pytest.raises(ServerRotationError, match="nothing to rotate"):
        await pm.restart()


# -- resume reconciliation: stop/replace, never assume ------------------------ #


async def test_resume_adopts_a_matching_unowned_resident_leaf(bench_cfg_dict):
    """A crash may have died with the RIGHT leaf profile still up. A fresh
    `ServerOrchestra` (a resumed run) owns nothing, but must not relaunch a
    perfectly good server just because it never spawned it."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    world.live[8081] = cfg.servers.leaf          # survivor: correct profile

    orch = _orchestra(cfg, world)
    await orch.start_resident()

    assert world.force_kill_calls == []
    assert [e for e in world.log if e[1] == 8081] == []   # never (re)started
    assert orch.current_profile == RESIDENT_PROFILE
    assert orch.leaf_proc is None            # adopted, not owned


async def test_resume_force_kills_a_mismatched_survivor(bench_cfg_dict):
    """The scenario the ruling names explicitly: a crash died with
    `bench_leaf` up. `start_resident()` must detect the mismatch (total_slots
    2 != 128) and reclaim the port before the resident leaf is spawned --
    never assume the survivor is fine."""
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    world.live[8081] = cfg.servers.bench_leaf     # survivor: WRONG profile

    orch = _orchestra(cfg, world)
    await orch.start_resident()

    assert world.force_kill_calls == [8081]
    assert ("start", 8081, 128) in world.log      # a fresh resident leaf was spawned
    assert orch.current_profile == RESIDENT_PROFILE
    assert orch.leaf_proc is not None             # this time, OWNED


async def test_resume_with_nothing_live_starts_normally(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    await orch.start_resident()
    assert world.force_kill_calls == []
    assert ("start", 8081, 128) in world.log


# -- --no-launch-servers: assert-only ----------------------------------------- #


async def test_no_launch_servers_handshakes_but_never_spawns(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    world.live[8080] = cfg.servers.root
    world.live[8081] = cfg.servers.leaf
    orch = _orchestra(cfg, world, launch=False)

    await orch.start_resident()
    assert world.log == []            # never touched a process
    assert orch.current_profile == RESIDENT_PROFILE


async def test_no_launch_servers_refuses_a_swap_with_a_clear_error(bench_cfg_dict):
    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    world.live[8080] = cfg.servers.root
    world.live[8081] = cfg.servers.leaf
    orch = _orchestra(cfg, world, launch=False)
    await orch.start_resident()

    with pytest.raises(ConfigError, match="no-launch-servers"):
        await orch.to_bench_leaf()
    assert world.log == []
    assert world.force_kill_calls == []


# -- dispatcher lifecycle: RLM/B2 vs the B1/B3 bench_leaf topology ----------- #


def test_bench_leaf_config_swaps_servers_leaf_for_bench_leafs_fields(bench_cfg_dict):
    cfg2 = bench_leaf_config(bench_cfg_dict)
    bl = Config.model_validate(bench_cfg_dict).servers.bench_leaf
    assert cfg2.servers.leaf.parallel == bl.parallel == 2
    assert cfg2.servers.leaf.ctx == bl.ctx == 524288
    assert cfg2.servers.leaf.port == bl.port == 8081
    # the ORIGINAL config is untouched (never mutate a built Config).
    original = Config.model_validate(bench_cfg_dict)
    assert original.servers.leaf.parallel == 128


def test_bench_leaf_config_refuses_without_a_bench_leaf_profile(bench_cfg_dict):
    raw = copy.deepcopy(bench_cfg_dict)
    raw["servers"]["bench_leaf"] = None
    with pytest.raises(ConfigError, match="bench_leaf"):
        bench_leaf_config(raw)


def test_bench_dispatcher_sees_the_true_two_slot_topology(bench_cfg_dict):
    from rlm.dispatcher import LLMDispatcher

    d = bench_dispatcher(bench_cfg_dict)
    assert isinstance(d, LLMDispatcher)
    assert d.slots.size == 2
    assert d._targets["leaf"].slot_capacity_tokens == 262144


# -- composed with rlm.bench.run_block: relaunch strictly between episodes --- #


async def test_orchestra_wired_into_run_block_keeps_relaunch_out_of_wall_s(
        tmp_path, bench_cfg_dict):
    """The ledgered contract restated with the REAL orchestration, not just
    `Hooks` fakes (Task 9 already proves the scheduler-side half of this in
    `test_wall_clock_excludes_the_relaunch_and_the_handshake`): swaps happen
    strictly inside `_prepare`, entirely before `_run_cell` reads its own
    `t0`, so driving `ServerOrchestra` as the real hooks must not leak any
    extra clock reads into the timed bracket."""
    async def _always_idle(*_a: Any, **_k: Any) -> bool:
        return True

    cfg = Config.model_validate(bench_cfg_dict)
    world = FakeWorld()
    orch = _orchestra(cfg, world)
    orch._slots_idle_fn = _always_idle   # quiesce: no real /slots probe
    await orch.start_resident()
    world.log.clear()

    clock = FakeClock(step=1.0)
    arms = FakeArms()
    ctx = BenchCtx(
        raw_cfg=bench_cfg_dict, cfg=cfg, run_id="run-1",
        manifest=_manifest(["t1"]), ledger=BenchLedger(tmp_path / "ledger.jsonl"),
        trace=FakeTrace(), arm_runners=arms.runners(), load_task_fn=_task_loader,
        quiesce_fn=orch.quiesce, handshake_fn=orch.handshake_profile,
        swap_servers_fn=orch.swap_to, clock=clock, repo_root=tmp_path)

    records = await run_block(build_blocks(ctx.manifest, [1])[0], ["rlm", "b1"], ctx)
    assert [r["wall_s"] for r in records] == [1.0, 1.0]
    # ...and the swap genuinely happened (real FakeProcess start/stop), not
    # merely a no-op hook that would make this assertion vacuous.
    assert ("stop", 8081, 128) in world.log and ("start", 8081, 2) in world.log
