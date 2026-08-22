"""§8's benchmark scheduler: the blocked (task, seed) grid, its ledger, and the
two rules that make a 39-hour run survivable — resume and rerun-once.

WHAT THIS MODULE IS ALLOWED TO TOUCH. It is the shape of a composition root
(it sequences the arms across two server profiles), but it is NOT one: the
dependency rule (spec §5, linted by `tests/test_import_rules.py`) exempts
exactly two modules — `rlm/episode.py` and `rlm/cli.py` — and widening that
list is the drift `rlm/episode.py`'s docstring forbids. So `rlm/bench.py` is
listed as ISOLATED instead, and everything that reaches a model server or a
process arrives as an INJECTED callable on `BenchCtx`:

  * the arms (`arm_runners`) — Task 10 builds them from `run_episode`
    and `run_b1/b2/b3` with the dispatcher, root client, registry and process
    manager already closed over;
  * `load_task_fn` — `Task.from_file` lives in `rlm/episode.py`, which reaches
    C4, so even the task loader is passed in (`Task` is imported here for
    typing only, exactly as `rlm/arms.py` does);
  * `quiesce_fn` / `handshake_fn` / `swap_servers_fn` — the §5 quiesce point,
    §4's per-episode `/props` re-assertion, and the leaf relaunch.

The second layering rule is smaller but just as load-bearing: `bench/` (the
benchmark artifact and its manifest) is NOT in the shipped wheel
(`pyproject.toml` packages = ["rlm"]), and nothing under `rlm/` imports it. So
the manifest arrives as an object too, typed under TYPE_CHECKING.

WHAT IS PRE-REGISTERED HERE, and must not be tuned after runs exist (§8):

  * blocks are (task, seed) adjacent in time across all arms, task-major in
    manifest order and seed-minor, so R9 thermal drift cancels INSIDE each
    paired comparison rather than across the run;
  * within a block the arm order is `rlm → b2 → b1 → b3`: RLM and B2 on the
    resident topology, then ONE leaf relaunch serves B1 and B3 back to back,
    which is what bounds relaunches at two per block;
  * an `error` episode is rerun once with the same seed, the rows linked by
    `superseded_by`; a second error stands as that arm's result;
  * `budget_kill` / `context_exhausted` / `fail` are results, never retries.
"""
from __future__ import annotations

import copy
import json
import time
import uuid as uuidmod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from rlm.config import Config
from rlm.errors import ConfigError, Outcome
from rlm.power import energy_j_between

if TYPE_CHECKING:                       # pragma: no cover - typing only
    # `rlm.episode` reaches C4 and `bench.manifest` is not in the wheel; both
    # are structural here (a task is `.task_id` + what the arm does with it,
    # a manifest is `.tasks`/`.sha256`/`.validate`), so the types are all
    # that is needed. `tests/test_bench.py` pins both out of the runtime
    # import set.
    from bench.manifest import BenchmarkManifest, TaskEntry

    from rlm.episode import Task

REPO_ROOT = Path(__file__).resolve().parents[3]

#: §8's pre-registered within-block order. RLM and B2 run on the resident
#: topology; B1 and B3 share the one `bench_leaf` relaunch, on their own slots
#: (the v0.2.6 correction: two documents on one slot is R13's smallest repro).
#: `rlm-restricted` sits next to `rlm` deliberately: it runs on the RESIDENT
#: profile too, so adding it costs no extra relaunch and §8's two-per-block
#: bound is unchanged.
ARM_ORDER: tuple[str, ...] = ("rlm", "rlm-restricted", "b2", "b1", "b3")

#: The two server profiles a block moves between. Names, not configs: which
#: flags each one launches with belongs to whoever owns the process (Task 10),
#: and this module only ever says WHICH one an arm needs.
RESIDENT_PROFILE = "resident"
BENCH_PROFILE = "bench"
ARM_PROFILE: dict[str, str] = {"rlm": RESIDENT_PROFILE,
                               "rlm-restricted": RESIDENT_PROFILE,
                               "b2": RESIDENT_PROFILE,
                               "b1": BENCH_PROFILE, "b3": BENCH_PROFILE}

#: The §6 outcome_reason for an arm that REFUSED before opening an episode row
#: (`servers.bench_leaf` missing, an unreadable corpus, an unreadable task
#: file). It is not an episode's reason — there is no episode — it is the
#: ledger's note that this cell was never measured. Shares its spelling with
#: the `config_refused` lifecycle kind on purpose: same event, two channels.
CONFIG_REFUSED = "config_refused"

#: Every outcome §6 can close an episode with. `error` is terminal too, but
#: only once its one rerun has been spent (see `BenchLedger.completed`).
TERMINAL_OUTCOMES = frozenset(str(o) for o in Outcome)

#: One record per episode, and exactly these keys. The ledger is a
#: crash-resilient MIRROR of the store, not a second source of truth (I4) —
#: it exists because a run that dies mid-block must be resumable without
#: opening the DuckDB file, which on Windows a second process cannot do at all.
#:
#: `relaunch_s` is the ONE column the store cannot hold: §8 excludes relaunch
#: time from per-task wall-clock (`wall_s` starts after `_prepare` returns), so
#: without a column of its own the ~10 s a leaf swap costs would be spent
#: `2N-1` times over N blocks (179 on the full 30x3 grid) and recorded nowhere.
#: It is per CELL, not per block: the cell whose `_prepare` paid for the swap is
#: the one that carries it, and every other cell in the block records 0.0.
LEDGER_FIELDS = ("run_id", "block", "task_id", "seed", "arm", "episode_id",
                 "outcome", "reason", "wall_s", "relaunch_s", "superseded_by",
                 "ts")

#: Where a bench run writes by default. OUTSIDE `milestones/`, deliberately:
#: `milestones/` is the evidence archive for gates that have already been
#: taken, and a product whose default write target is last run's finished
#: ledger will append a new grid into a closed result. S4's own ledger stays
#: at `milestones/s4/results/ledger.jsonl` as evidence; resume it explicitly
#: with `--ledger` if that is what you mean.
LEDGER_PATH = REPO_ROOT / "runs" / "ledger.jsonl"

ArmRunner = Callable[..., Awaitable[Any]]


# --------------------------------------------------------------------------- #
# blocks and per-seed configs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Block:
    """One (task, seed) cell of §8's grid, run across every arm back to back."""

    task_entry: "TaskEntry"
    seed: int
    idx: int


def build_blocks(manifest: "BenchmarkManifest", seeds: list[int]) -> list[Block]:
    """§8's schedule: task-major (manifest order), seed-minor.

    Seed-minor is the load-bearing half. The paired comparison is per (task,
    seed) across arms; running a task's three seeds back to back keeps the
    whole comparison inside one thermal neighbourhood, where interleaving by
    seed would spread each task's draws across hours of drift (R9).
    """
    return [Block(task_entry=entry, seed=seed, idx=i)
            for i, (entry, seed) in enumerate(
                (entry, seed) for entry in manifest.tasks for seed in seeds)]


def seeded_config(raw_cfg_dict: dict, seed: int) -> Config:
    """The shipped config with this attempt's seed, re-validated.

    Patch the RAW dict and re-validate (`milestones/s1/run_s1.py:variant_config`), never
    mutate a built `Config`: every cross-field rule in `rlm.config` runs again,
    and the prompt sha256 pins are re-derived rather than carried over.

    BOTH seeds move together — `sampling.root.seed` and `sampling.leaf.seed`.
    §8 says "3 seeds" of the WHOLE system; a run that varied only the root
    would hold the leaf's draws fixed across seeds and report three replicates
    of one leaf.

    `trace.export_every_episode` is forced OFF here (ledgered ruling): the
    per-episode bundle export rewrites every episode of the run each time, so
    over 360 episodes it is O(n^2) work for an artifact nobody reads until the
    run ends. Export is a once-at-the-end step for a bench run.
    """
    raw = copy.deepcopy(raw_cfg_dict)
    raw["scaffold"]["sampling"]["root"]["seed"] = seed
    raw["scaffold"]["sampling"]["leaf"]["seed"] = seed
    raw.setdefault("trace", {})["export_every_episode"] = False
    return Config.model_validate(raw)


def assert_manifest_pinned(manifest: "BenchmarkManifest", cfg: Config, *,
                            require_closed_book: bool = True) -> "BenchmarkManifest":
    """The freeze, verified at RUNTIME for the first time.

    `bench/manifest.json`'s sha256 is pinned in `config.yaml`
    (`benchmark.manifest_sha256`). Until now that pin was a claim in a config
    file; a bench run is where it becomes a precondition. Two separate gates,
    both refusing rather than warning: the manifest must be the frozen one,
    and it must still satisfy §8 (`validate`, closed-book probe included).
    """
    pin = cfg.benchmark.manifest_sha256
    if not pin:
        raise ConfigError(
            "benchmark.manifest_sha256 is not set; a benchmark run must name "
            "the freeze it is scoring against (§8)")
    actual = manifest.sha256
    if actual != pin:
        raise ConfigError(
            f"benchmark manifest_sha256 mismatch: config pins {pin}, the "
            f"manifest hashes to {actual} -- the frozen benchmark moved, so "
            f"this run would score a different task set than it reports")
    try:
        manifest.validate(require_closed_book=require_closed_book)
    except AssertionError as exc:
        # `Config.model_validate`'s precedent: callers only ever catch
        # ConfigError, so a startup refusal never arrives as a bare assertion.
        raise ConfigError(str(exc)) from exc
    return manifest


def bench_extra(run_id: str, block: int, seed: int, arm: str) -> dict[str, Any]:
    """The identity this episode is scored by, as it lands in
    `config_snapshot["bench"]`.

    `arm` is added for the RLM arm ONLY. `ArmEpisode.snapshot` writes
    `{"arm": self.arm, **bench_extra}` itself, so a baseline that also carried
    one would be asserting a fact the arm already owns; `run_episode` has no
    arm concept at all, so the RLM arm's has to come from here.
    """
    extra: dict[str, Any] = {"run_id": run_id, "block": block, "seed": seed}
    if arm in ("rlm", "rlm-restricted"):
        extra["arm"] = arm
    return extra


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _key(record: dict) -> tuple[str, int, str]:
    return (record["task_id"], record["seed"], record["arm"])


def _identity(record: dict) -> tuple:
    """What makes two rows THE SAME row across the ledger and the store.

    An episode is its `episode_id`. A refusal has none — no row was ever
    opened — so it is identified by its cell, which is also why a cell can
    only ever carry one refusal.
    """
    ep = record.get("episode_id")
    return ("episode", str(ep)) if ep else ("refusal", *_key(record))


class BenchLedger:
    """Append-only JSONL beside the trace store.

    WHY IT EXISTS AT ALL, given I4 says the store is the record of truth: on
    Windows DuckDB holds an exclusive file lock, so a second process cannot
    read the live store — not read-only, not ATTACH, not a copy. A resumed run
    is a second process. Without this file, `--resume` would have to open the
    store the dead run may still be holding.

    Append-only means an amendment is another line: `mark_superseded` re-writes
    the row with `superseded_by` filled, and the merge takes the LAST line for
    each identity. A half-written trailing line (the crash this file exists
    for) is skipped rather than fatal.
    """

    def __init__(self, path: str | Path = LEDGER_PATH) -> None:
        self.path = Path(path)

    # -- write ------------------------------------------------------------- #

    def append(self, record: dict) -> dict:
        row = {k: record.get(k) for k in LEDGER_FIELDS}
        if row["ts"] is None:
            row["ts"] = _utc_ts()
        if row["outcome"] is not None:
            row["outcome"] = str(row["outcome"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def mark_superseded(self, record: dict, new_episode_id: str) -> dict:
        return self.append({**record, "superseded_by": str(new_episode_id),
                            "ts": None})

    # -- read -------------------------------------------------------------- #

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue            # a torn trailing line: the crash case
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def records(self, run_id: str, *, store: Any = None) -> list[dict]:
        """Every known row for `run_id`, ledger merged with the store.

        THE STORE WINS on every column it owns: it is what the verdict is
        computed from, and `mark_superseded`/`close_episode` land there first.
        A NULL from the store never clobbers a known ledger value — a run
        still in flight has open rows, and "not yet" is not "no".

        `wall_s` is deliberately NOT re-derived from the store: this column is
        the SCHEDULER's bracket around the arm call, which excludes the
        relaunch and the handshake (§8 excludes relaunch time from per-task
        wall-clock). `started_at`/`ended_at` measure the episode row's own
        lifetime, which is a different quantity.
        """
        merged: dict[tuple, dict] = {}
        for row in self.read():
            if row.get("run_id") != run_id:
                continue
            ident = _identity(row)
            merged[ident] = {**merged.get(ident, {}), **row}
        for row in _store_rows(store, run_id):
            ident = _identity(row)
            merged[ident] = {**merged.get(ident, {}), **row}
        return list(merged.values())

    def _by_cell(self, run_id: str, store: Any) -> dict[tuple, list[dict]]:
        """Rows grouped by cell, EPISODES ONLY.

        Refusal rows are dropped here rather than filtered at each use: they
        record that a cell was skipped for a run-configuration fault, and
        every question below ("was this decided", "is a rerun owed") is a
        question about episodes that ran. Counting a refusal as a row would
        make a cell that refused once and then errored look like a cell that
        had already spent its rerun.
        """
        cells: dict[tuple, list[dict]] = {}
        for row in self.records(run_id, store=store):
            if not row.get("episode_id"):
                continue
            if row.get("task_id") is None or row.get("seed") is None or not row.get("arm"):
                continue                    # not a bench row we can place
            cells.setdefault(_key(row), []).append(row)
        return cells

    def completed(self, run_id: str, *, store: Any = None) -> dict[tuple, dict]:
        """The (task_id, seed, arm) cells this run has DECIDED — what a resume
        skips.

        Three things are deliberately not "decided":
          * a superseded row (its rerun is the result);
          * a lone `error` with no rerun beside it — §8 owes that cell one more
            draw, and a run that died between the two must resume into it, not
            record the crash as the arm's answer;
          * a refusal (no `episode_id`): nothing ran, so there is nothing to
            score. Fix the config, resume, and the cell fills.
        """
        done: dict[tuple, dict] = {}
        for cell, rows in self._by_cell(run_id, store).items():
            live = [r for r in rows if not r.get("superseded_by")]
            if not live:
                continue
            final = live[-1]
            if str(final.get("outcome")) not in TERMINAL_OUTCOMES:
                continue
            if str(final.get("outcome")) == str(Outcome.ERROR) and len(rows) < 2:
                continue
            done[cell] = final
        return done

    def open_errors(self, run_id: str, *, store: Any = None) -> dict[tuple, dict]:
        """Cells holding an `error` episode whose one rerun was never run.

        A resumed run reruns exactly these and links them — otherwise the
        rerun §8 promises would either be skipped (if the cell counted as
        decided) or unlinked (if the new row forgot its predecessor).
        """
        open_: dict[tuple, dict] = {}
        for cell, rows in self._by_cell(run_id, store).items():
            if len(rows) != 1 or rows[0].get("superseded_by"):
                continue
            if str(rows[0].get("outcome")) == str(Outcome.ERROR):
                open_[cell] = rows[0]
        return open_


_STORE_SQL = """
SELECT CAST(episode_id AS VARCHAR)                                  AS episode_id,
       task_id                                                      AS task_id,
       json_extract_string(config_snapshot, '$.bench.arm')          AS arm,
       CAST(json_extract_string(config_snapshot, '$.bench.seed')
            AS BIGINT)                                              AS seed,
       CAST(json_extract_string(config_snapshot, '$.bench.block')
            AS BIGINT)                                              AS block,
       CAST(outcome AS VARCHAR)                                     AS outcome,
       outcome_reason                                               AS reason,
       CAST(superseded_by AS VARCHAR)                               AS superseded_by
FROM episodes
WHERE json_extract_string(config_snapshot, '$.bench.run_id') = ?
"""


def _store_rows(store: Any, run_id: str) -> list[dict]:
    """Bench episodes the store holds for this run.

    `store` is a DuckDB connection-shaped object — in production
    `TraceLogger.monitor()`, which is a cursor on the writer's own connection
    (a second process cannot open the file at all on Windows).
    """
    if store is None:
        return []
    cur = store.execute(_STORE_SQL, [run_id])
    cols = [d[0] for d in cur.description]
    rows: list[dict] = []
    for raw in cur.fetchall():
        row = {c: v for c, v in zip(cols, raw) if v is not None}
        if not row.get("arm") or row.get("seed") is None:
            continue            # not a bench episode; nothing to merge
        row["run_id"] = run_id
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# the run context: everything model-facing, injected
# --------------------------------------------------------------------------- #


async def _no_hook(profile: str) -> None:
    """Default quiesce/handshake/swap: do nothing.

    A default that REFUSED would be safer in the abstract and wrong here: this
    module is exercised (and dry-run) with no servers at all, and the wiring
    that must not be forgotten is checked where it can be — Task 10 passes all
    three, and `rlm/cli.py` is where a missing one is a startup bug."""
    return None


def _no_task_loader(path: Any) -> Any:
    raise ConfigError(
        "BenchCtx.load_task_fn was not injected; the benchmark's task files "
        "are read with `rlm.episode.Task.from_file`, which this module may "
        "not import (the dependency rule)")


@dataclass
class BenchCtx:
    """One bench run's long-lived state plus every injected seam.

    THE HOOK SURFACE (Task 10 fills all of it; tests pass doubles):

      arm_runners[arm](task, cfg, *, bench_extra) -> ArmResult-like
          One per arm in ARM_ORDER. `rlm` wraps `run_episode` with
          `snapshot_extra={"bench": bench_extra}`; `b1`/`b2`/`b3` wrap
          `run_b1`/`run_b2`/`run_b3` with `bench_extra=bench_extra` and the
          dispatcher, root client, registry and process manager closed over.
          The result needs `.episode_id`, `.outcome`, `.reason`, `.answer`.
      load_task_fn(path) -> Task                     (`Task.from_file`)
      quiesce_fn(profile) -> awaitable               (the `_slots_idle` pattern)
      handshake_fn(profile) -> awaitable             (§4 `/props` re-assertion)
      swap_servers_fn(profile) -> awaitable          (the leaf relaunch)
      temp_fn() -> float | None                      (`read_pkg_temp_c`)
      clock() -> float                               (`time.monotonic`)
      sampler                                        (`PowerSampler` or None)
      store                                          (`TraceLogger.monitor()`)
    """

    raw_cfg: dict
    cfg: Config
    run_id: str
    manifest: "BenchmarkManifest"
    ledger: BenchLedger
    trace: Any = None
    lifecycle: Any = None
    store: Any = None
    sampler: Any = None
    arm_runners: dict[str, ArmRunner] = field(default_factory=dict)
    load_task_fn: Callable[[Any], "Task"] = _no_task_loader
    quiesce_fn: Callable[[str], Awaitable[Any]] = _no_hook
    handshake_fn: Callable[[str], Awaitable[Any]] = _no_hook
    swap_servers_fn: Callable[[str], Awaitable[Any]] = _no_hook
    temp_fn: Callable[[], float | None] | None = None
    clock: Callable[[], float] = time.monotonic
    repo_root: Path = REPO_ROOT
    #: Which leaf profile is live. Starts resident (the RLM topology is what a
    #: bench run launches into) and is only ever changed through
    #: `swap_servers_fn`, so the count of swaps is the count of relaunches.
    current_profile: str = RESIDENT_PROFILE
    #: What THIS cell's `_prepare` spent relaunching the leaf, as reported by
    #: `swap_servers_fn` (`ServerOrchestra.swap_to` returns its own
    #: `last_relaunch_s`). Read straight off the hook's return value rather
    #: than off the orchestra: this module may not import the thing that owns
    #: the process, and a hook that reports nothing is a hook that cost 0.0.
    last_relaunch_s: float = 0.0


# --------------------------------------------------------------------------- #
# running a block
# --------------------------------------------------------------------------- #


def _resolve_arms(arms: list[str] | tuple[str, ...], ctx: BenchCtx) -> list[str]:
    unknown = [a for a in arms if a not in ARM_ORDER]
    if unknown:
        raise ConfigError(f"unknown arm(s) {unknown}; §8's arms are "
                          f"{list(ARM_ORDER)}")
    missing = [a for a in arms if a not in ctx.arm_runners]
    if missing:
        raise ConfigError(f"no runner injected for arm(s) {missing}")
    requested = set(arms)
    return [a for a in ARM_ORDER if a in requested]


async def _prepare(ctx: BenchCtx, arm: str) -> None:
    """Everything that must be true before the clock starts.

    Order matters and is §8's: relaunch (only when the profile actually
    changes — that is what bounds it at two per block), then quiesce, then the
    `/props` re-assertion. All three are OUTSIDE the timed bracket: §8 excludes
    relaunch time from per-task wall-clock, and a mismatch found here refuses
    the run rather than scoring a task.

    Excluded from `wall_s` is not the same as unrecorded: what the swap cost is
    stashed on `ctx.last_relaunch_s` and ledgered in its own column, so the
    ~10 s a relaunch takes stays attributable to the cell that paid it.
    """
    profile = ARM_PROFILE[arm]
    ctx.last_relaunch_s = 0.0
    if profile != ctx.current_profile:
        cost = await ctx.swap_servers_fn(profile)
        ctx.current_profile = profile
        # A hook is allowed to report nothing (the default `_no_hook` returns
        # None, and so does any double that does not model the cost); "no
        # number" is 0.0 here rather than a TypeError inside the scheduler.
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            ctx.last_relaunch_s = float(cost)
    await ctx.quiesce_fn(profile)
    await ctx.handshake_fn(profile)


def _reading(sampler: Any):
    if sampler is None or not sampler.alive():
        return None
    return sampler.reading()


def _stamp_metrics(ctx: BenchCtx, episode_id: str, *, start, end,
                    temp_start: float | None, temp_end: float | None) -> None:
    """The cost-scorecard columns, or nothing at all.

    `avg_power_w` is DERIVED — energy delta over the interval that delta was
    measured across — and never read from the sampler's `power_mw`, which
    reads garbage on this box.

    THE DENOMINATOR IS THE READINGS' OWN INTERVAL, NOT `wall_s`. The sampler
    publishes at 1 Hz and each bracket read takes whatever sample happened to
    be latest, so `[start.ts, end.ts]` is offset from `[t0, t1]` at both ends;
    dividing a numerator measured over one window by the length of a different
    window reports watts nothing measured. (`wall_s` is a third quantity again
    — it excludes the relaunch — which is exactly the mixing this module's
    docstring forbids.)

    NOTHING is stamped unless a real delta exists: the sampler dies silently
    at launch (so `alive()` is checked at both ends), and on a sub-second
    episode both reads return the SAME cached reading — a delta of zero over a
    real episode is not a measurement of zero, it is no measurement, and a
    fabricated 0.0 W would be indistinguishable from one at scoring time.
    """
    energy_j = avg_power_w = None
    if start is not None and end is not None and end.ts > start.ts:
        energy_j = energy_j_between(start, end)
        avg_power_w = energy_j / (end.ts - start.ts)
    if (energy_j is None and avg_power_w is None
            and temp_start is None and temp_end is None):
        return
    if ctx.trace is None:
        return
    ctx.trace.update_episode_metrics(episode_id, pkg_temp_c_start=temp_start,
                                      pkg_temp_c_end=temp_end,
                                      avg_power_w=avg_power_w, energy_j=energy_j)


def _refusal(ctx: BenchCtx, block: Block, arm: str, exc: Exception,
              wall_s: float = 0.0) -> dict:
    if ctx.lifecycle is not None:
        ctx.lifecycle.event("config_refused", run_id=ctx.run_id, block=block.idx,
                             task_id=block.task_entry.task_id, seed=block.seed,
                             arm=arm, error=repr(exc))
    return ctx.ledger.append({
        "run_id": ctx.run_id, "block": block.idx,
        "task_id": block.task_entry.task_id, "seed": block.seed, "arm": arm,
        "episode_id": None, "outcome": str(Outcome.ERROR), "reason": CONFIG_REFUSED,
        "wall_s": round(wall_s, 3), "relaunch_s": round(ctx.last_relaunch_s, 3),
        "superseded_by": None, "ts": None})


async def _run_cell(ctx: BenchCtx, block: Block, arm: str, task: "Task",
                     cfg: Config) -> dict:
    """One episode: prepare, bracket, run, record.

    A `ConfigError` from the ARM is contained here and only here. `run_b1` and
    `run_b3` refuse before opening a row (`servers.bench_leaf` missing, an
    unreadable corpus) and one task's refusal must not abort a 39-hour grid.
    Every other exception propagates: `run_b1` re-raises after closing its row
    on purpose, because a bug in an arm must not be scoreable as an ordinary
    `error` episode. A `ConfigError` from the QUIESCE/HANDSHAKE path also
    propagates — a server that relaunched with different flags poisons every
    remaining task, not this one (§4).
    """
    await _prepare(ctx, arm)
    temp_start = ctx.temp_fn() if ctx.temp_fn is not None else None
    start = _reading(ctx.sampler)
    t0 = ctx.clock()
    try:
        result = await ctx.arm_runners[arm](
            task, cfg, bench_extra=bench_extra(ctx.run_id, block.idx, block.seed, arm))
    except ConfigError as exc:
        return _refusal(ctx, block, arm, exc, wall_s=ctx.clock() - t0)
    wall_s = ctx.clock() - t0
    end = _reading(ctx.sampler)
    temp_end = ctx.temp_fn() if ctx.temp_fn is not None else None
    _stamp_metrics(ctx, result.episode_id, start=start, end=end,
                    temp_start=temp_start, temp_end=temp_end)
    return ctx.ledger.append({
        "run_id": ctx.run_id, "block": block.idx,
        "task_id": block.task_entry.task_id, "seed": block.seed, "arm": arm,
        "episode_id": str(result.episode_id), "outcome": str(result.outcome),
        "reason": result.reason, "wall_s": round(wall_s, 3),
        "relaunch_s": round(ctx.last_relaunch_s, 3),
        "superseded_by": None, "ts": None})


def _link(ctx: BenchCtx, old: dict, new: dict) -> None:
    """§8's rerun link, in both channels: the store (what the verdict reads)
    and the ledger (what a resume reads)."""
    old_id, new_id = old.get("episode_id"), new.get("episode_id")
    if not old_id or not new_id:
        return
    # Parsed, not just passed: the writer's UPDATE matches by episode_id, so a
    # malformed id would silently link nothing and the rerun rule would be
    # unenforced in exactly the runs that needed it. Fail loudly instead.
    old_id, new_id = str(uuidmod.UUID(str(old_id))), str(uuidmod.UUID(str(new_id)))
    if ctx.trace is not None:
        ctx.trace.mark_superseded(old_id, new_id)
    ctx.ledger.mark_superseded(old, new_id)


async def run_block(block: Block, arms: list[str] | tuple[str, ...], ctx: BenchCtx,
                     *, skip: dict[tuple, dict] | None = None,
                     prior_errors: dict[tuple, dict] | None = None) -> list[dict]:
    """Run one (task, seed) block across `arms`, in §8's pre-registered order.

    Returns the ledger records it wrote, oldest first — the rerun of an
    errored cell is a second record, not a replacement.
    """
    ordered = _resolve_arms(arms, ctx)
    skip = skip or {}
    prior_errors = prior_errors or {}
    entry = block.task_entry
    records: list[dict] = []

    pending = [a for a in ordered if (entry.task_id, block.seed, a) not in skip]
    if not pending:
        return records

    # A config that will not synthesise is a RUN-level fault -- the raw dict is
    # the same for every block, so it would refuse all 90 of them -- and it
    # aborts rather than being recorded 90 times as one cell's bad luck.
    cfg = seeded_config(ctx.raw_cfg, block.seed)
    # The task file, by contrast, is this block's own: read ONCE, before any
    # relaunch, because an unreadable task refuses every arm in the cell and
    # there is no point relaunching a server for it. No swap has happened yet,
    # so the refusal rows below must not inherit the PREVIOUS cell's relaunch.
    ctx.last_relaunch_s = 0.0
    try:
        task = ctx.load_task_fn(ctx.repo_root / entry.task_file)
    except ConfigError as exc:
        return [_refusal(ctx, block, arm, exc) for arm in pending]

    for arm in pending:
        cell = (entry.task_id, block.seed, arm)
        prior = prior_errors.get(cell)
        record = await _run_cell(ctx, block, arm, task, cfg)
        records.append(record)
        if prior is not None:
            # The rerun a crashed run still owed. It is spent whatever this
            # attempt returned, so the link is made and no second one follows.
            _link(ctx, prior, record)
            continue
        if str(record["outcome"]) != str(Outcome.ERROR) or not record["episode_id"]:
            continue
        rerun = await _run_cell(ctx, block, arm, task, cfg)
        records.append(rerun)
        _link(ctx, record, rerun)
    return records


async def run_bench(ctx: BenchCtx, *, arms: list[str] | tuple[str, ...] = ARM_ORDER,
                     seeds: list[int] | None = None,
                     blocks: list[Block] | None = None) -> list[dict]:
    """The whole grid for one `run_id`, resumable.

    The freeze is verified HERE, before any block runs, and not only in
    `rlm/cli.py`: `assert_manifest_pinned` is also exported so a CLI can
    refuse early with a better error surface, but a forgotten call there must
    not be able to score 39 hours of episodes against a manifest nobody
    checked. The startup assertion belongs where the episodes are, so it
    cannot be bypassed by wiring.

    Resume state is read ONCE, up front: the cells this run already decided
    are skipped, and the cells still owed a rerun are run and linked. Reading
    it per block would let this run's own rows re-enter the calculation.
    """
    assert_manifest_pinned(ctx.manifest, ctx.cfg)
    seeds = seeds if seeds is not None else list(ctx.cfg.benchmark.seeds)
    blocks = blocks if blocks is not None else build_blocks(ctx.manifest, seeds)
    done = ctx.ledger.completed(ctx.run_id, store=ctx.store)
    owed = ctx.ledger.open_errors(ctx.run_id, store=ctx.store)
    records: list[dict] = []
    for block in blocks:
        records.extend(await run_block(block, arms, ctx, skip=done, prior_errors=owed))
    return records


__all__ = [
    "ARM_ORDER",
    "ARM_PROFILE",
    "BENCH_PROFILE",
    "CONFIG_REFUSED",
    "LEDGER_FIELDS",
    "LEDGER_PATH",
    "RESIDENT_PROFILE",
    "BenchCtx",
    "BenchLedger",
    "Block",
    "assert_manifest_pinned",
    "bench_extra",
    "build_blocks",
    "run_bench",
    "run_block",
    "seeded_config",
]
