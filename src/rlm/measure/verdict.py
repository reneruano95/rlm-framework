"""§8's scoring, inference and report layer: grid -> decision -> report.

WHAT THIS MODULE IS. Everything §8 pre-registered about how S4 is scored, in
one place that reads a CLOSED trace store and nothing else. It is the last
component of the benchmark slice and the only one whose output is a claim:
`## S4 GATE: PASS|FAIL`.

THE RULES BELOW ARE PRE-REGISTERED (ARCHITECTURE.md §8, "changing any of this
after runs exist is p-hacking"). They are transcribed here, not invented here,
and a post-hoc "improvement" to any of them -- a softer margin, a friendlier
tie, a second bootstrap, a per-category gate -- is the failure mode the
pre-registration exists to prevent:

  * a task passes for an arm iff >=2/3 seeds pass (>=3/5 after escalation);
  * "beats" is a margin of +3 tasks at N=30, against EACH baseline;
  * a tie with any baseline FAILS the gate;
  * a tie-or-loss to B2 triggers the pivot-to-B2 rule, and to B3 the
    pivot-to-RAG finding, of the same standing;
  * `budget_kill` and `context_exhausted` are failures for every arm;
  * a margin in {+1,+2,+3} owes seeds {4,5} on that PAIR's discordant tasks,
    re-decided at >=3/5, with the sign test and bootstrap recomputed ONCE and
    both pre- and post-escalation figures reported;
  * per-category results are TABLED and never gated (§8 refuses per-category
    margin gates: "at 4-6 tasks per category any margin is noise"), except for
    the zero-floor tripwire, which blocks the WORDING "clean pass" and not the
    gate;
  * every win claim states its cost multiple beside the margin.

WHY IT IS ISOLATED (checks/test_import_rules.py). A verdict must be
recomputable from the record with no server reachable -- that is what makes
S4 re-scoreable offline and what stops a model-graded step from creeping into
scoring. So this module imports duckdb, `rlm.measure.stats` and the standard library,
takes the manifest as an OBJECT (`bench/` is not in the shipped wheel, exactly
as `src/rlm/measure/bench.py` documents), and reaches nothing that speaks HTTP.

REFUSAL, NOT REPAIR. A grid with a missing cell or two live rows for one
(task, seed, arm) is not scoreable: imputing the hole or picking a row by
order would decide the gate by accident. Both raise `VerdictError`.
"""
from __future__ import annotations

import pathlib
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import duckdb

from rlm.measure.stats import (fractional_score, needs_escalation, paired_bootstrap_ci,
                        sign_test_p, task_passes)

if TYPE_CHECKING:                       # pragma: no cover - typing only
    # `bench/` is not in the wheel (pyproject packages = ["rlm"]) and nothing
    # under `rlm/` may import it; a manifest is structurally `.tasks` of
    # `.task_id`/`.category`, which is all `decide` reads. Same solution as
    # `src/rlm/measure/bench.py`.
    from bench.manifest import BenchmarkManifest

#: §8's arms. RLM is the subject; the other three are the controls it must
#: beat by +3 tasks EACH.
RLM_ARM = "rlm"
BASELINES: tuple[str, ...] = ("b1", "b2", "b3")
ARMS: tuple[str, ...] = (RLM_ARM, *BASELINES)

#: The pre-registered margin at N=30 (§8: "+2 tasks (N=20) or +3 tasks (N=30)").
MARGIN_GATE = 3

#: §8:343. Disjoint from the base seeds by construction (`rlm.config`), which
#: is what lets a post-escalation grid be told apart from a ragged one.
ESCALATION_SEEDS: tuple[int, ...] = (4, 5)

#: milestones/s1/run_s1.py:521's contract, verbatim: everything below this line is
#: hand-written and survives regeneration.
NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"

#: Stated BEFORE the numbers in every report, so the rule cannot be read as
#: having been chosen to fit them.
DECISION_RULE = """\
Pre-registered before any episode ran (ARCHITECTURE.md §8, "changing any of
this after runs exist is p-hacking"):

1. A task **passes** for an arm iff **≥2/3 seeds** pass — **≥3/5** for tasks
   re-run under escalation. `budget_kill` and `context_exhausted` count as
   failures for every arm; an `error` is re-run once and a second one stands.
2. **Success rate** = tasks passed / N. **Margin** = RLM's passed tasks minus
   the baseline's.
3. The **S4 gate passes iff the margin is ≥ +3 against ALL of B1, B2 and B3**.
   **A tie fails.** A tie-or-loss to **B2** additionally triggers the
   **pivot-to-B2** rule; a tie-or-loss to **B3** records a **pivot-to-RAG**
   finding of the same standing.
4. Every margin is reported beside an exact two-sided **sign (McNemar) test**
   over the discordant tasks and a **10,000-resample paired bootstrap CI** on
   the mean per-task fractional delta.
5. A margin landing in **{+1, +2, +3}** owes seeds **{4, 5}** on that pair's
   **discordant tasks only**; those tasks are re-decided at ≥3/5, the sign
   test and bootstrap are recomputed **once**, and both pre- and
   post-escalation figures are reported. No other recomputation is permitted.
6. Per-category results are **tabled, never gated** — §8 refuses per-category
   margin gates outright. The one per-category rule is the **zero-floor
   tripwire**: a category where RLM scores 0 while any baseline scores ≥3 of
   its tasks blocks the wording "clean pass" and names a mandatory
   category-regression finding, without touching the aggregate gate.
7. Any win claim states its **cost multiple** next to the margin.
"""


class VerdictError(RuntimeError):
    """The grid is not scoreable, or the record contradicts itself."""


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #

#: The run filter, one string so the grid, the scorecard and the leak report
#: cannot drift apart. `superseded_by IS NULL` implements §8's rerun-once rule
#: (the rerun is the result) and `NOT dry_run` keeps a mock-dispatcher episode
#: out of a real verdict.
_RUN_FILTER = """\
json_extract_string(config_snapshot, '$.bench.run_id') = ?
  AND NOT dry_run
  AND superseded_by IS NULL"""

_EPISODES_SQL = f"""\
SELECT CAST(episode_id AS VARCHAR)                                  AS episode_id,
       task_id                                                      AS task_id,
       json_extract_string(config_snapshot, '$.bench.arm')          AS arm,
       CAST(json_extract_string(config_snapshot, '$.bench.seed')
            AS BIGINT)                                              AS seed,
       CAST(outcome AS VARCHAR)                                     AS outcome,
       json_extract_string(config_snapshot,
                            '$.scaffold.chunk.size_tokens')         AS chunk_size
FROM episodes
WHERE {_RUN_FILTER}
"""


def _connect(db_path: str | pathlib.Path) -> duckdb.DuckDBPyConnection:
    """Read-only, and only ever against a CLOSED store.

    On Windows DuckDB holds an exclusive lock, so this cannot open a live run's
    file at all -- which is the correct behaviour: a verdict computed from a
    half-written grid would be a verdict about a different run.
    """
    p = pathlib.Path(db_path)
    if not p.exists():
        raise VerdictError(f"no trace store at {p}: a verdict is computed from "
                            f"the record, and there is no record here")
    try:
        return duckdb.connect(str(p), read_only=True)
    except duckdb.Error as exc:
        raise VerdictError(
            f"cannot open {p} read-only ({exc}). A verdict is computed from a "
            f"CLOSED store -- on Windows DuckDB excludes every other process "
            f"from a file a writer holds open, so this means the run is still "
            f"live. Let it finish (or close the TraceLogger) and score then; "
            f"scoring a half-written grid would score a different run") from exc


def _rows(db_path: str | pathlib.Path, sql: str, params: list) -> list[dict]:
    con = _connect(db_path)
    try:
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, raw)) for raw in cur.fetchall()]
    finally:
        con.close()


@dataclass(frozen=True)
class Grid:
    """§8's primary artifact: the per-task × per-arm × per-seed outcome grid.

    "not two aggregate rates" (§8) -- everything else in this module is derived
    from here, so the grid is what a re-scoring starts from.
    """

    run_id: str
    task_ids: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    required_seeds: tuple[int, ...]
    chunk_sizes: tuple[int, ...]
    cells: Mapping[tuple[str, str], tuple[bool, ...]]
    cell_seeds: Mapping[tuple[str, str], tuple[int, ...]]
    episode_ids: Mapping[tuple[str, str, int], str]

    def cell(self, task_id: str, arm: str) -> list[bool]:
        """This cell's per-seed pass/fail, seed ASCENDING (so an escalated
        cell's seeds 4 and 5 land last, where `stats.task_passes` expects
        them)."""
        try:
            return list(self.cells[(task_id, arm)])
        except KeyError:
            raise VerdictError(
                f"no cell for task {task_id!r} arm {arm!r} in run "
                f"{self.run_id!r}") from None

    def seeds_for(self, task_id: str, arm: str) -> tuple[int, ...]:
        return self.cell_seeds.get((task_id, arm), ())

    @property
    def chunk_size(self) -> int | None:
        """The one chunk size §8's chunk-size lock demands, or None if the run
        carried more than one (which `decide` reports as a finding)."""
        return self.chunk_sizes[0] if len(self.chunk_sizes) == 1 else None

    @property
    def escalated_tasks(self) -> tuple[str, ...]:
        """Tasks that carry seeds beyond the base set for at least one arm."""
        extra = {t for (t, _a), seeds in self.cell_seeds.items()
                 if len(seeds) > len(self.required_seeds)}
        return tuple(t for t in self.task_ids if t in extra)


def load_grid(db_path: str | pathlib.Path, run_id: str, *,
              seeds: Sequence[int] | None = None,
              arms: Sequence[str] | None = None,
              escalation_seeds: Sequence[int] = ESCALATION_SEEDS) -> Grid:
    """Build §8's grid from the store, refusing anything unscoreable.

    `outcome == 'success'` is the ONLY True. `fail`, `budget_kill`,
    `context_exhausted` and a terminal `error` are all that arm's failure on
    that seed (§8: budget_kill/context_exhausted "count as failures for every
    arm"; a second `error` "scores as a failure for that arm"). The scheduler's
    refusal rows are ledgered as `error` too and, where one opened an episode,
    they arrive here and score the same way -- a cell that was never measured
    is a cell that failed to produce an answer, not a cell to be dropped.

    `seeds` is the pre-registered base seed set. When it is not given it is
    inferred as every seed observed MINUS `escalation_seeds`: §8:343 runs seeds
    {4,5} on the discordant tasks of ONE pair, so the post-escalation grid is
    legitimately ragged and must not be refused for it. Base seeds are required
    in every cell; escalation seeds are optional per cell and simply extend it.
    """
    rows = _rows(db_path, _EPISODES_SQL, [run_id])

    by_cell: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        arm, seed = row["arm"], row["seed"]
        if not arm or seed is None:
            raise VerdictError(
                f"episode {row['episode_id']} is in run {run_id!r} but carries "
                f"no bench arm/seed (config_snapshot.bench.arm="
                f"{arm!r}, .seed={seed!r}); it cannot be placed in the grid")
        key = (row["task_id"], int(seed), arm)
        prior = by_cell.get(key)
        if prior is not None:
            raise VerdictError(
                f"duplicate cell {key}: episodes {prior['episode_id']} and "
                f"{row['episode_id']} are both live for it. One of them owes a "
                f"`superseded_by` link (§8's rerun-once rule); scoring either "
                f"by row order would decide the gate by accident")
        by_cell[key] = row

    task_ids = tuple(sorted({k[0] for k in by_cell}))
    observed_seeds = tuple(sorted({k[1] for k in by_cell}))
    observed_arms = {k[2] for k in by_cell}
    arm_order = tuple([a for a in ARMS if a in observed_arms]
                      + sorted(observed_arms - set(ARMS)))
    grid_arms = tuple(arms) if arms is not None else arm_order

    esc = set(escalation_seeds)
    if seeds is not None:
        required = tuple(sorted(seeds))
    else:
        required = tuple(s for s in observed_seeds if s not in esc)
        # A run whose ONLY seeds are the escalation ones (a `--seeds 4` smoke
        # run) would otherwise leave `required` empty and disable the
        # missing-cell check entirely -- a silent hole where there should be a
        # refusal. With no base seeds to subtract, every observed seed is one.
        if not required:
            required = observed_seeds
    # A seed cannot be both a base seed and an escalation seed: an explicit
    # `seeds=` that names one takes it out of the escalation set rather than
    # making every cell fail the completeness check below.
    esc -= set(required)

    missing = [(t, a, s) for t in task_ids for a in grid_arms
               for s in required if (t, s, a) not in by_cell]
    if missing:
        shown = ", ".join(f"{t}/{a}/seed {s}" for t, a, s in missing[:8])
        raise VerdictError(
            f"{len(missing)} missing cell(s) in run {run_id!r}: {shown}"
            f"{' ...' if len(missing) > 8 else ''}. §8's grid is the primary "
            f"artifact; a hole in it is not a zero")

    # §8:343 re-decides an escalated task on >=3/5. A cell holding SOME of the
    # escalation seeds would be scored at a denominator §8 never registered --
    # `stats.task_passes` reads a 4-long cell as a 5-seed one and demands 3 --
    # so a half-written escalation is refused on the same grounds as a hole in
    # the base grid: refusal, not repair.
    partial: list[tuple[str, str, list[int]]] = []
    seen: set[tuple[str, str]] = set()
    for (task_id, seed, arm) in sorted(by_cell):
        if seed not in esc or (task_id, arm) in seen:
            continue
        seen.add((task_id, arm))
        absent = [s for s in sorted(esc) if (task_id, s, arm) not in by_cell]
        if absent:
            partial.append((task_id, arm, absent))
    if partial:
        shown = ", ".join(f"{t}/{a} (missing seed(s) {ab})" for t, a, ab in partial[:8])
        raise VerdictError(
            f"{len(partial)} half-escalated cell(s) in run {run_id!r}: {shown}"
            f"{' ...' if len(partial) > 8 else ''}. §8 runs seeds "
            f"{sorted(esc)} together and re-decides at >=3/5; a cell holding "
            f"only some of them would be scored at a denominator §8 never "
            f"pre-registered")

    cells: dict[tuple[str, str], tuple[bool, ...]] = {}
    cell_seeds: dict[tuple[str, str], tuple[int, ...]] = {}
    episode_ids: dict[tuple[str, str, int], str] = {}
    for task_id in task_ids:
        for arm in grid_arms:
            present = sorted(s for (t, s, a) in by_cell
                             if t == task_id and a == arm)
            if not present:
                continue
            cell_seeds[(task_id, arm)] = tuple(present)
            cells[(task_id, arm)] = tuple(
                by_cell[(task_id, s, arm)]["outcome"] == "success"
                for s in present)
            for s in present:
                episode_ids[(task_id, arm, s)] = by_cell[(task_id, s, arm)]["episode_id"]

    chunk_sizes = tuple(sorted({int(r["chunk_size"]) for r in rows
                                if r["chunk_size"] is not None}))
    return Grid(run_id=run_id, task_ids=task_ids, arms=grid_arms,
                seeds=observed_seeds, required_seeds=required,
                chunk_sizes=chunk_sizes, cells=cells, cell_seeds=cell_seeds,
                episode_ids=episode_ids)


# --------------------------------------------------------------------------- #
# the decision + the inference layer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairResult:
    """One RLM-vs-baseline comparison, with the inference §8 makes mandatory."""

    baseline: str
    present: bool
    rlm_passes: int
    baseline_passes: int
    margin: int | None
    wins: int
    losses: int
    discordant: tuple[str, ...]
    p: float | None
    ci: tuple[float, float] | None
    mean_delta: float | None
    escalates: bool
    beats: bool


@dataclass(frozen=True)
class CategoryRow:
    category: str
    n_tasks: int
    passes: Mapping[str, int]


@dataclass(frozen=True)
class Finding:
    """A named, pre-registered consequence. `kind` is what code branches on;
    `text` is what the report prints."""

    kind: str
    text: str


@dataclass(frozen=True)
class Verdict:
    run_id: str
    n_tasks: int
    n_manifest_tasks: int
    task_ids: tuple[str, ...]
    arms: tuple[str, ...]
    passes: Mapping[str, tuple[str, ...]]
    success_rate: Mapping[str, float]
    scores: Mapping[tuple[str, str], float]
    pairs: Mapping[str, PairResult]
    categories: tuple[CategoryRow, ...]
    findings: tuple[Finding, ...]
    escalation_plan: Mapping[str, tuple[str, ...]]
    gate_pass: bool
    clean_pass: bool
    chunk_sizes: tuple[int, ...]
    escalated: bool

    @property
    def chunk_size(self) -> int | None:
        return self.chunk_sizes[0] if len(self.chunk_sizes) == 1 else None


def _ci(deltas: Sequence[float]) -> tuple[float, float] | None:
    """`stats.paired_bootstrap_ci` divides by `len(deltas)`; an empty grid must
    report "no CI" rather than raise ZeroDivisionError out of the report path.
    Guarded HERE, at the call site, per the Task 1 ruling."""
    if not deltas:
        return None
    return paired_bootstrap_ci(deltas)


def decide(grid: Grid, manifest: "BenchmarkManifest") -> Verdict:
    """Apply §8's pre-registered decision rule and inference layer to a grid.

    The manifest supplies exactly one thing: each task's category, for the
    per-category table and the zero-floor tripwire. A task in the grid that the
    manifest does not know is refused -- it is not in the frozen benchmark, so
    there is no category to score it under and no pre-registration covering it.
    """
    categories: dict[str, str] = {t.task_id: t.category for t in manifest.tasks}
    order = [t.task_id for t in manifest.tasks]

    unknown = [t for t in grid.task_ids if t not in categories]
    if unknown:
        raise VerdictError(
            f"task(s) {unknown} are in the grid but not in benchmark manifest "
            f"{getattr(manifest, 'benchmark_version', '?')!r}: they are outside "
            f"the freeze, so §8 pre-registers nothing about them")

    in_grid = set(grid.task_ids)
    tasks = tuple(t for t in order if t in in_grid)

    passes: dict[str, tuple[str, ...]] = {}
    scores: dict[tuple[str, str], float] = {}
    for arm in grid.arms:
        passed = []
        for t in tasks:
            cell = grid.cell(t, arm)
            scores[(arm, t)] = fractional_score(cell)
            if task_passes(cell):
                passed.append(t)
        passes[arm] = tuple(passed)

    n = len(tasks)
    success_rate = {arm: (len(p) / n if n else 0.0) for arm, p in passes.items()}

    findings: list[Finding] = []
    pairs: dict[str, PairResult] = {}
    plan: dict[str, tuple[str, ...]] = {}

    rlm_present = RLM_ARM in grid.arms
    if not rlm_present:
        findings.append(Finding(
            "missing_arm", "the RLM arm did not run in this grid — there is no "
            "subject to decide, and the gate cannot pass"))

    rlm_set = set(passes.get(RLM_ARM, ()))
    for baseline in BASELINES:
        if baseline not in grid.arms:
            findings.append(Finding(
                "missing_arm",
                f"baseline {baseline.upper()} did not run in this grid; §8's "
                f"gate is a margin against ALL THREE baselines, so it cannot "
                f"pass without it"))
            pairs[baseline] = PairResult(
                baseline=baseline, present=False, rlm_passes=len(rlm_set),
                baseline_passes=0, margin=None, wins=0, losses=0,
                discordant=(), p=None, ci=None, mean_delta=None,
                escalates=False, beats=False)
            continue

        base_set = set(passes[baseline])
        wins = tuple(t for t in tasks if t in rlm_set and t not in base_set)
        losses = tuple(t for t in tasks if t in base_set and t not in rlm_set)
        discordant = tuple(t for t in tasks if (t in rlm_set) != (t in base_set))
        margin = len(rlm_set) - len(base_set)
        deltas = [scores[(RLM_ARM, t)] - scores[(baseline, t)] for t in tasks] \
            if rlm_present else []
        escalates = needs_escalation(margin)
        pairs[baseline] = PairResult(
            baseline=baseline, present=True, rlm_passes=len(rlm_set),
            baseline_passes=len(base_set), margin=margin,
            wins=len(wins), losses=len(losses), discordant=discordant,
            p=sign_test_p(len(wins), len(losses)), ci=_ci(deltas),
            mean_delta=(statistics.fmean(deltas) if deltas else None),
            escalates=escalates, beats=margin >= MARGIN_GATE)
        # §8 escalates ONCE ("the sign test and bootstrap are recomputed once
        # on the final grid ... No other recomputation is permitted"), so a
        # grid that already carries seeds {4,5} plans nothing, however its
        # recomputed margin lands. `escalates` still records the band, because
        # that is a fact about the margin and is reported as one.
        if escalates and discordant and not grid.escalated_tasks:
            plan[baseline] = discordant
        if margin <= 0:
            kind = {"b2": "pivot_to_b2", "b3": "pivot_to_rag"}.get(baseline)
            if kind == "pivot_to_b2":
                findings.append(Finding(
                    kind, f"tie-or-loss to B2 (margin {margin:+d}): §8's "
                    f"**pivot-to-B2 rule** is triggered — fixed map-reduce "
                    f"matched the root's agency, so the agency is not earning "
                    f"its complexity"))
            elif kind == "pivot_to_rag":
                findings.append(Finding(
                    kind, f"tie-or-loss to B3 (margin {margin:+d}): a "
                    f"**pivot-to-RAG finding** of the same standing — a "
                    f"deterministic BM25 pipeline a practitioner would deploy "
                    f"matched the scaffold"))

    # -- per-category table + the zero-floor tripwire -------------------------
    cat_order: list[str] = []
    for t in tasks:
        if categories[t] not in cat_order:
            cat_order.append(categories[t])
    rows: list[CategoryRow] = []
    for cat in cat_order:
        members = [t for t in tasks if categories[t] == cat]
        per_arm = {arm: sum(1 for t in members if t in set(passes[arm]))
                   for arm in grid.arms}
        rows.append(CategoryRow(category=cat, n_tasks=len(members),
                                 passes=per_arm))
        rlm_score = per_arm.get(RLM_ARM, 0)
        floored = [b for b in BASELINES if per_arm.get(b, 0) >= 3]
        if rlm_present and rlm_score == 0 and floored:
            findings.append(Finding(
                "category_regression",
                f"**zero-floor tripwire: `{cat}`** — RLM passed 0 of "
                f"{len(members)} while "
                + ", ".join(f"{b.upper()} passed {per_arm[b]}" for b in floored)
                + f". §8 makes this a mandatory named category-regression "
                  f"finding and the first post-S4 investigation; the aggregate "
                  f"gate is untouched, but this run is not a clean pass"))

    if len(tasks) != len(manifest.tasks):
        findings.append(Finding(
            "partial_grid",
            f"this grid scores {len(tasks)} of the manifest's "
            f"{len(manifest.tasks)} tasks; the +{MARGIN_GATE}-task threshold is "
            f"pre-registered at N=30, so a margin over a smaller N is not the "
            f"pre-registered decision"))

    if len(grid.chunk_sizes) != 1:
        findings.append(Finding(
            "chunk_size_lock",
            f"§8's chunk-size lock is violated: this run carries chunk sizes "
            f"{list(grid.chunk_sizes)}. S4 runs RLM, B2 and B3 at ONE untouched "
            f"config default, and a mixed grid compares arms tuned differently"))

    gate = bool(rlm_present and all(pairs[b].beats for b in BASELINES))
    clean = gate and not any(f.kind == "category_regression" for f in findings)

    return Verdict(
        run_id=grid.run_id, n_tasks=n, n_manifest_tasks=len(manifest.tasks),
        task_ids=tasks, arms=grid.arms, passes=passes,
        success_rate=success_rate, scores=scores, pairs=pairs,
        categories=tuple(rows), findings=tuple(findings),
        escalation_plan=plan, gate_pass=gate, clean_pass=clean,
        chunk_sizes=grid.chunk_sizes,
        escalated=bool(grid.escalated_tasks))


# --------------------------------------------------------------------------- #
# the cost scorecard
# --------------------------------------------------------------------------- #

_COST_SQL = f"""\
SELECT e.task_id                                       AS task_id,
       e.arm                                           AS arm,
       e.seed                                          AS seed,
       CAST(e.episode_id AS VARCHAR)                   AS episode_id,
       e.wall_s                                        AS wall_s,
       e.energy_j                                      AS energy_j,
       e.avg_power_w                                   AS avg_power_w,
       COALESCE(SUM(COALESCE(s.tokens_in, 0)
                    + COALESCE(s.tokens_out, 0)), 0)   AS tokens,
       COUNT(s.step_idx)                               AS n_steps
FROM (
    SELECT episode_id,
           task_id,
           json_extract_string(config_snapshot, '$.bench.arm')   AS arm,
           CAST(json_extract_string(config_snapshot, '$.bench.seed')
                AS BIGINT)                                       AS seed,
           CASE WHEN ended_at IS NULL THEN NULL
                ELSE date_diff('millisecond', started_at, ended_at) / 1000.0
           END                                                   AS wall_s,
           energy_j, avg_power_w
    FROM episodes
    WHERE {_RUN_FILTER}
) e
LEFT JOIN steps s USING (episode_id)
GROUP BY e.task_id, e.arm, e.seed, e.episode_id, e.wall_s, e.energy_j,
         e.avg_power_w
ORDER BY e.task_id, e.arm, e.seed
"""

_METRICS = ("tokens", "wall", "energy", "power")


@dataclass(frozen=True)
class EpisodeCost:
    task_id: str
    arm: str
    seed: int
    episode_id: str
    tokens: int
    n_steps: int
    wall_s: float | None
    energy_j: float | None
    avg_power_w: float | None


@dataclass(frozen=True)
class TaskCost:
    """One (arm, task) cell of the scorecard: the MEDIAN across its seeds.

    Median rather than mean at both levels, deliberately: an aggregation task
    costs two orders of magnitude more than a needle one, so a mean per arm is
    a report about the aggregation category wearing an arm's name.
    """

    arm: str
    task_id: str
    seeds: tuple[int, ...]
    tokens: int
    tokens_total: int
    wall_s: float | None
    energy_j: float | None
    avg_power_w: float | None


@dataclass(frozen=True)
class ArmCost:
    arm: str
    n_tasks: int
    n_episodes: int
    median_tokens: float | None
    median_wall_s: float | None
    median_energy_j: float | None
    median_power_w: float | None
    total_tokens: int


@dataclass(frozen=True)
class Scorecard:
    run_id: str
    episodes: tuple[EpisodeCost, ...]
    tasks: Mapping[tuple[str, str], TaskCost]
    arms: Mapping[str, ArmCost]

    def multiple(self, arm: str, baseline: str, metric: str = "wall") -> float | None:
        """`arm`'s cost as a multiple of `baseline`'s, or None when either side
        was never measured. §8: any win claim states this next to the margin."""
        if metric not in _METRICS:
            raise ValueError(f"unknown cost metric {metric!r}; §8 scores "
                             f"{list(_METRICS)}")
        attr = {"tokens": "median_tokens", "wall": "median_wall_s",
                "energy": "median_energy_j", "power": "median_power_w"}[metric]
        a, b = self.arms.get(arm), self.arms.get(baseline)
        if a is None or b is None:
            return None
        num, den = getattr(a, attr), getattr(b, attr)
        if num is None or den is None or den == 0:
            return None
        return num / den


def _median(values: Iterable[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def cost_scorecard(db_path: str | pathlib.Path, run_id: str) -> Scorecard:
    """§8's mandatory cost scorecard, from the trace alone.

    Per-arm, per-task total tokens (`tokens_in + tokens_out` over ALL steps,
    step-level retries included, NULLs counted as 0) and wall-clock
    (`ended_at - started_at` on the episode row), plus energy where the sampler
    produced any. `energy_j` / `avg_power_w` stay NULL on a host where nothing
    validated a collector, and the report prints "null" rather than a 0.0 that
    would read as a measurement of zero.

    Scoped by the SAME run filter as the grid: the episode whose outcome is
    scored is the episode whose cost is attributed. A superseded attempt is not
    added back in -- its result was discarded, and charging the arm for it
    would price a row §8 already ruled out of the record.
    """
    rows = _rows(db_path, _COST_SQL, [run_id])
    episodes = tuple(
        EpisodeCost(task_id=r["task_id"], arm=r["arm"], seed=int(r["seed"]),
                    episode_id=r["episode_id"], tokens=int(r["tokens"]),
                    n_steps=int(r["n_steps"]),
                    wall_s=r["wall_s"], energy_j=r["energy_j"],
                    avg_power_w=r["avg_power_w"])
        for r in rows if r["arm"] and r["seed"] is not None)

    by_task: dict[tuple[str, str], list[EpisodeCost]] = {}
    for ep in episodes:
        by_task.setdefault((ep.arm, ep.task_id), []).append(ep)

    tasks: dict[tuple[str, str], TaskCost] = {}
    for (arm, task_id), eps in by_task.items():
        tasks[(arm, task_id)] = TaskCost(
            arm=arm, task_id=task_id,
            seeds=tuple(sorted(e.seed for e in eps)),
            tokens=int(_median(e.tokens for e in eps) or 0),
            tokens_total=sum(e.tokens for e in eps),
            wall_s=_median(e.wall_s for e in eps),
            energy_j=_median(e.energy_j for e in eps),
            avg_power_w=_median(e.avg_power_w for e in eps))

    by_arm: dict[str, list[TaskCost]] = {}
    for (arm, _t), tc in tasks.items():
        by_arm.setdefault(arm, []).append(tc)

    arms: dict[str, ArmCost] = {}
    for arm, tcs in by_arm.items():
        arms[arm] = ArmCost(
            arm=arm, n_tasks=len(tcs),
            n_episodes=sum(1 for e in episodes if e.arm == arm),
            median_tokens=_median(t.tokens for t in tcs),
            median_wall_s=_median(t.wall_s for t in tcs),
            median_energy_j=_median(t.energy_j for t in tcs),
            median_power_w=_median(t.avg_power_w for t in tcs),
            total_tokens=sum(t.tokens_total for t in tcs))

    return Scorecard(run_id=run_id, episodes=episodes, tasks=tasks, arms=arms)


# --------------------------------------------------------------------------- #
# R13's foreign-string detector, per arm
# --------------------------------------------------------------------------- #

_LEAK_SQL = f"""\
SELECT e.arm                                                        AS arm,
       COUNT(s.step_idx)                                            AS leaf_attempts,
       COUNT(s.step_idx) FILTER (WHERE s.leak_detected)             AS hits,
       COUNT(s.step_idx) FILTER (WHERE s.leak_detected = FALSE)     AS checked_clean,
       COUNT(s.step_idx) FILTER (WHERE s.leak_detected IS NULL)     AS not_checked
FROM (
    SELECT episode_id,
           json_extract_string(config_snapshot, '$.bench.arm') AS arm
    FROM episodes
    WHERE {_RUN_FILTER}
) e
LEFT JOIN steps s
       ON s.episode_id = e.episode_id
      AND s.actor = 'leaf'
      AND s.action_type = 'llm_call'
GROUP BY e.arm
"""


def leak_report(db_path: str | pathlib.Path, run_id: str) -> dict[str, dict[str, int]]:
    """R13's per-arm hit count, as §8 now requires ("its hit count is reported
    per arm in the verdict").

    THE TRI-STATE IS LOAD-BEARING AND IS NEVER COLLAPSED. `leak_detected` is
    TRUE (a foreign identifier was found), FALSE (checked, none found) or NULL
    (never checked — no corpus index, or the step produced no answer). Folding
    NULL into FALSE would turn "we did not look" into "we looked and it was
    clean", and FALSE is itself only evidence, not a certificate: 138 clean
    calls bound the rate at 2.2%, they do not zero it. The phrase "leak-free"
    is not available to any reader of this function.
    """
    rows = _rows(db_path, _LEAK_SQL, [run_id])
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if not r["arm"]:
            continue
        out[r["arm"]] = {"hits": int(r["hits"]),
                         "checked_clean": int(r["checked_clean"]),
                         "not_checked": int(r["not_checked"]),
                         "leaf_attempts": int(r["leaf_attempts"])}
    return out


# --------------------------------------------------------------------------- #
# the Pareto chart: hand-rolled SVG, no dependencies
# --------------------------------------------------------------------------- #


def pareto_svg(verdict: Verdict, scorecard: Scorecard | None, *,
               width: int = 560, height: int = 340) -> str:
    """§8's mandatory success-vs-cost Pareto: one point per arm, success rate
    against median wall-clock.

    Hand-rolled markup on purpose. A plotting dependency would put a chart in
    the verdict that cannot be regenerated from the record by a reader who has
    only this repo, which is the same standard `src/rlm/measure/stats.py` holds the
    p-values to.
    """
    pad_l, pad_r, pad_t, pad_b = 62, 22, 26, 46
    x0, x1 = pad_l, width - pad_r
    y0, y1 = height - pad_b, pad_t
    pts = []
    for arm in verdict.arms:
        cost = scorecard.arms.get(arm) if scorecard is not None else None
        wall = cost.median_wall_s if cost is not None else None
        if wall is not None:
            pts.append((arm, float(wall), verdict.success_rate.get(arm, 0.0)))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
           f'height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="sans-serif" role="img" '
           f'aria-label="success rate versus median wall-clock, one point per arm">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    if not pts:
        out.append(f'<text x="{width // 2}" y="{height // 2}" '
                   f'text-anchor="middle" font-size="13" fill="#444">no '
                   f'wall-clock in the record — nothing to plot</text>')
        out.append("</svg>")
        return "\n".join(out)

    xmax = max(p[1] for p in pts) * 1.15 or 1.0
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>')
    for frac in (0.0, 0.5, 1.0):
        y = y0 + (y1 - y0) * frac
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                   f'stroke="#e5e5e5"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="#333">{frac:.1f}</text>')
    for frac in (0.0, 0.5, 1.0):
        x = x0 + (x1 - x0) * frac
        out.append(f'<text x="{x:.1f}" y="{y0 + 16}" text-anchor="middle" '
                   f'font-size="11" fill="#333">{xmax * frac:.0f}</text>')
    for arm, wall, rate in pts:
        cx = x0 + (x1 - x0) * (wall / xmax)
        cy = y0 + (y1 - y0) * rate
        fill = "#1a5fb4" if arm == RLM_ARM else "#a51d2d"
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{fill}"/>')
        out.append(f'<text x="{cx + 9:.1f}" y="{cy + 4:.1f}" font-size="12" '
                   f'fill="#111">{arm} ({rate:.2f}, {wall:.0f}s)</text>')
    out.append(f'<text x="{(x0 + x1) // 2}" y="{height - 8}" '
               f'text-anchor="middle" font-size="12" fill="#111">median '
               f'wall-clock per task (s)</text>')
    out.append(f'<text x="14" y="{(y0 + y1) // 2}" font-size="12" fill="#111" '
               f'transform="rotate(-90 14 {(y0 + y1) // 2})" '
               f'text-anchor="middle">success rate</text>')
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def _fmt_margin(margin: int | None) -> str:
    return "n/a" if margin is None else f"{margin:+d}"


def _fmt_p(p: float | None) -> str:
    return "p=n/a" if p is None else f"p={p:.4f}"


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    return "CI=n/a" if ci is None else f"CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _fmt_num(v: float | None, *, digits: int = 1, unit: str = "") -> str:
    return "null" if v is None else f"{v:,.{digits}f}{unit}"


def _fmt_mult(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}x"


def _final(verdict: Verdict, escalated: Verdict | None) -> Verdict:
    """Which verdict decides — and the provenance check that makes the answer
    safe to trust.

    An `escalated` verdict is the DECISION (§8 recomputes once on the final
    grid), while the report's title, run id and pre-escalation figures come
    from `verdict`. So a mismatched pair would render one run's gate under
    another run's name, and a pre-escalation grid passed as `escalated` would
    report a recomputation that never happened. Both are refused rather than
    rendered: this is the one place in the module where two independently
    computed verdicts meet, and nothing downstream could detect the swap.
    """
    if escalated is None:
        return verdict
    if escalated.run_id != verdict.run_id:
        raise VerdictError(
            f"escalated verdict is for run {escalated.run_id!r} but the report "
            f"is for run {verdict.run_id!r}: §8's escalation re-runs seeds "
            f"{list(ESCALATION_SEEDS)} inside the SAME run, and rendering one "
            f"run's gate under another run's name is not a reporting error, it "
            f"is a different result")
    if not escalated.escalated:
        raise VerdictError(
            f"the verdict passed as `escalated` for run {verdict.run_id!r} was "
            f"computed on a grid carrying no escalation seeds; it is a second "
            f"pre-escalation verdict, and reporting it as the post-escalation "
            f"recomputation would claim §8's de-noising step ran when it did not")
    return escalated


def _cost_clause(scorecard: Scorecard | None, baseline: str) -> str:
    """The cost multiple §8 requires beside every win claim."""
    if scorecard is None:
        return "cost multiples unavailable (no scorecard supplied)"
    wall = scorecard.multiple(RLM_ARM, baseline, "wall")
    tokens = scorecard.multiple(RLM_ARM, baseline, "tokens")
    energy = scorecard.multiple(RLM_ARM, baseline, "energy")
    clause = (f"{_fmt_mult(wall)} median wall-clock and "
              f"{_fmt_mult(tokens)} median tokens")
    if energy is not None:
        clause += f" and {_fmt_mult(energy)} energy"
    return clause


def render_report(verdict: Verdict, scorecard: Scorecard | None,
                  leaks: Mapping[str, Mapping[str, int]], *,
                  escalated: Verdict | None = None,
                  report_path: str | pathlib.Path | None = None) -> str:
    """The S4 report, generated half only (everything above NARRATIVE_MARKER).

    ORDER IS AN ARGUMENT. The pre-registered rule is stated in full BEFORE any
    number appears, so no reader — including the author — can check the rule
    against the result it produced. Every margin carries its p-value and CI in
    the same line; every win claim carries its cost multiple; the per-category
    table is printed with the refusal of per-category gates printed beside it.

    WHEN `escalated` IS GIVEN IT IS THE VERDICT. §8 makes the post-escalation
    recomputation the decision and the pre-escalation figures a reporting
    obligation, not an alternative — so the gate heading, the margins and the
    findings all come from the escalated grid, and the pre-escalation figures
    appear (in full) under Escalation. Rendering the pre-escalation gate as the
    headline would let the two be chosen between, which is the thing §8's
    "recomputed once" forbids. `_final` refuses an `escalated` verdict that is
    not this run's, or that was not computed on an escalated grid.
    """
    final = _final(verdict, escalated)
    arms = [a for a in ARMS if a in final.arms] + \
           [a for a in final.arms if a not in ARMS]
    L: list[str] = []
    L += [f"# S4 verdict — run `{verdict.run_id}`", "",
          "Generated by `rlm.measure.verdict` from the closed trace store. Every "
          "figure below is recomputable from the record: the sign test and "
          "bootstrap are exact and dependency-free (`src/rlm/measure/stats.py`), and the "
          "grid is the primary artifact, not these summaries.", ""]

    # 1. the rule, before the numbers -------------------------------------- #
    L += ["## The decision rule (pre-registered, stated before the numbers)", "",
          DECISION_RULE, ""]

    # 2. the gate ----------------------------------------------------------- #
    L += [f"## S4 GATE: {'PASS' if final.gate_pass else 'FAIL'}", ""]
    chunk = (f"chunk_size={final.chunk_size}" if final.chunk_size is not None
             else "chunk_size=unrecorded" if not final.chunk_sizes
             else f"chunk_size=MIXED{list(final.chunk_sizes)}")
    clean = ("clean pass" if final.clean_pass
             else "NOT a clean pass" if final.gate_pass
             else "gate failed")
    # A gate decided on a grid that still owes seeds {4,5} is not the final
    # decision §8 pre-registered; saying so is the difference between reporting
    # a result and claiming one.
    provisional = ("" if escalated is not None or not final.escalation_plan
                   else " · **PROVISIONAL: escalation owed** (see below)")
    L += [f"**RLM {len(final.passes.get(RLM_ARM, ()))}/{final.n_tasks} tasks "
          f"({final.success_rate.get(RLM_ARM, 0.0):.3f})** · {chunk} · "
          f"N={final.n_tasks} · "
          f"{'post-escalation grid' if final.escalated else 'pre-escalation grid'}"
          f" · {clean}{provisional}.", ""]
    # §8: "the S4 verdict states the p-value and CI next to the margin" -- so a
    # margin never appears alone, not even in the summary line.
    for b in BASELINES:
        pair = final.pairs.get(b)
        if pair is None:
            continue
        L.append(f"- margin **{_fmt_margin(pair.margin)}** vs {b.upper()} — "
                 f"{_fmt_p(pair.p)}, {_fmt_ci(pair.ci)}"
                 + ("" if pair.present else " _(arm absent from this grid)_"))
    L.append("")

    # 3. margins with p and CI ---------------------------------------------- #
    L += ["### Margins, each with its exact sign test and bootstrap CI", "",
          "| pair | RLM tasks | baseline tasks | margin | sign test | "
          "bootstrap CI (95%) | discordant (w/l) | escalation band |",
          "|---|---|---|---|---|---|---|---|"]
    for b in BASELINES:
        pair = final.pairs.get(b)
        if pair is None:
            continue
        L.append(
            f"| RLM vs {b.upper()} | {pair.rlm_passes} | "
            f"{pair.baseline_passes if pair.present else 'n/a (arm absent)'} | "
            f"{_fmt_margin(pair.margin)} | {_fmt_p(pair.p)} | {_fmt_ci(pair.ci)} | "
            f"{len(pair.discordant)} ({pair.wins}/{pair.losses}) | "
            f"{'yes' if pair.escalates else 'no'} |")
    L.append("")

    for b in BASELINES:
        pair = final.pairs.get(b)
        if pair is None or not pair.present or pair.margin is None:
            continue
        if pair.beats:
            L.append(f"RLM beats {b.upper()} by {_fmt_margin(pair.margin)} tasks "
                     f"({_fmt_p(pair.p)}, {_fmt_ci(pair.ci)}) at "
                     f"{_cost_clause(scorecard, b)} vs {b.upper()}.")
        elif pair.margin > 0:
            # §8 DEFINES "beats" as a margin of +3 at N=30. A smaller positive
            # margin is a lead and nothing more -- calling it a win in prose
            # while the gate calls it a failure is how a report ends up
            # arguing with its own verdict.
            L.append(f"RLM leads {b.upper()} by {_fmt_margin(pair.margin)} tasks "
                     f"({_fmt_p(pair.p)}, {_fmt_ci(pair.ci)}) — **below the "
                     f"+{MARGIN_GATE} threshold, so it does not beat "
                     f"{b.upper()}** — at {_cost_clause(scorecard, b)} vs "
                     f"{b.upper()}.")
        elif pair.margin == 0:
            L.append(f"RLM ties {b.upper()} ({_fmt_margin(pair.margin)}, "
                     f"{_fmt_p(pair.p)}, {_fmt_ci(pair.ci)}) — a tie fails the "
                     f"gate, at {_cost_clause(scorecard, b)} vs {b.upper()}.")
        else:
            L.append(f"RLM loses to {b.upper()} by {_fmt_margin(pair.margin)} "
                     f"tasks ({_fmt_p(pair.p)}, {_fmt_ci(pair.ci)}), at "
                     f"{_cost_clause(scorecard, b)} vs {b.upper()}.")
    L.append("")

    # 4. findings ------------------------------------------------------------ #
    L += ["## Findings (pre-registered consequences)", ""]
    if final.findings:
        L += [f"- **[{f.kind}]** {f.text}" for f in final.findings]
    else:
        L.append("- none of §8's named consequences fired.")
    L.append("")

    # 5. per-category table -------------------------------------------------- #
    L += ["## Per-category results (reported, never gated)", "",
          "§8 refuses **per-category margin gates** outright — \"at 4–6 tasks "
          "per category any margin is noise, and a per-category gate would "
          "generate random vetoes of a valid aggregate result\". This table is "
          "reported for interpretation; the only per-category RULE is the "
          "zero-floor tripwire, which appears under Findings above.", "",
          "| category | tasks | " + " | ".join(a.upper() for a in arms) + " |",
          "|---" * (len(arms) + 2) + "|"]
    for row in final.categories:
        L.append(f"| {row.category} | {row.n_tasks} | "
                 + " | ".join(str(row.passes.get(a, 0)) for a in arms) + " |")
    L.append("")

    # 6. cost scorecard ------------------------------------------------------ #
    L += ["## Cost scorecard", "",
          "Per-arm medians over per-task medians (tokens = `tokens_in + "
          "tokens_out` over **all** steps including retries; wall = "
          "`ended_at − started_at`). `null` means the column was never "
          "measured on this host, which is not a measurement of zero.", "",
          "| arm | tasks | episodes | median tokens | median wall s | "
          "median energy J | median avg power W | ×wall vs B1 | ×wall vs B2 | "
          "×wall vs B3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    if scorecard is None:
        L.append("| _no scorecard supplied_ | | | | | | | | | |")
    else:
        for arm in arms:
            c = scorecard.arms.get(arm)
            if c is None:
                continue
            L.append(
                f"| {arm} | {c.n_tasks} | {c.n_episodes} | "
                f"{_fmt_num(c.median_tokens, digits=0)} | "
                f"{_fmt_num(c.median_wall_s)} | "
                f"{_fmt_num(c.median_energy_j)} | "
                f"{_fmt_num(c.median_power_w)} | "
                + " | ".join(_fmt_mult(scorecard.multiple(arm, b, "wall"))
                             for b in BASELINES) + " |")
    L.append("")

    # 7. R13 --------------------------------------------------------------- #
    L += ["## R13 foreign-string detector, per arm", "",
          "§8 (v0.2.6) makes this mandatory: contamination is a bias channel "
          "that splits **two chunked-and-exposed arms (RLM, B2) against two "
          "single-shot-and-spared ones (B1, B3)**, so an undetected serving "
          "bug would manufacture both pre-registered kill findings at once. "
          "The verdict is **tri-state and is never collapsed**: `hits` = a "
          "foreign identifier was found, `checked_clean` = checked and none "
          "found, `not_checked` = never checked. `checked_clean` is evidence, "
          "not a certificate — 138 clean calls bound the rate at 2.2%.", "",
          "| arm | leaf llm_calls | hits | checked_clean | not_checked |",
          "|---|---|---|---|---|"]
    if leaks:
        for arm in arms:
            row = leaks.get(arm)
            if row is None:
                continue
            L.append(f"| {arm} | {row['leaf_attempts']} | {row['hits']} | "
                     f"{row['checked_clean']} | {row['not_checked']} |")
    else:
        L.append("| _no leak report supplied_ | | hits n/a | checked_clean n/a "
                 "| not_checked n/a |")
    L.append("")

    # 8. Pareto ------------------------------------------------------------- #
    L += ["## Success vs cost (Pareto)", ""]
    if report_path is not None:
        L += [f"Also written beside this report as "
              f"`{_svg_path(report_path).name}`.", ""]
    L += [pareto_svg(final, scorecard), ""]

    # 9. escalation --------------------------------------------------------- #
    L += ["## Escalation (§8, pre-registered at freeze)", ""]
    if escalated is not None:
        L += ["Seeds {4, 5} were run on the discordant tasks of every pair in "
              "the band, those tasks re-decided at ≥3/5, and the sign test and "
              "bootstrap recomputed **once**. Both figures stand below; no "
              "other recomputation is permitted.", "",
              "| pair | pre-escalation margin | pre p / CI | post-escalation "
              "margin | post p / CI |", "|---|---|---|---|---|"]
        for b in BASELINES:
            pre, post = verdict.pairs.get(b), escalated.pairs.get(b)
            if pre is None or post is None:
                continue
            L.append(f"| RLM vs {b.upper()} | {_fmt_margin(pre.margin)} | "
                     f"{_fmt_p(pre.p)}, {_fmt_ci(pre.ci)} | "
                     f"{_fmt_margin(post.margin)} | "
                     f"{_fmt_p(post.p)}, {_fmt_ci(post.ci)} |")
        L += ["",
              f"**Post-escalation gate: "
              f"{'PASS' if escalated.gate_pass else 'FAIL'}** — this is the "
              f"final decision; the pre-escalation figures above are reported "
              f"because §8 requires both, not because either may be chosen.", ""]
    elif verdict.escalation_plan:
        L += ["The net margin lands in the {+1, +2, +3} band against the "
              "pair(s) below, so §8 owes seeds **{4, 5}** on **those pairs' "
              "discordant tasks only**, re-decided at ≥3/5:", ""]
        for b, tasks in verdict.escalation_plan.items():
            L.append(f"- **RLM vs {b.upper()}** "
                     f"({_fmt_margin(verdict.pairs[b].margin)}): "
                     f"{len(tasks)} discordant task(s) — "
                     + ", ".join(f"`{t}`" for t in tasks))
        L += ["", f"Cost: {sum(len(t) for t in verdict.escalation_plan.values())} "
                  f"task(s) × {len(ESCALATION_SEEDS)} seed(s) × the arms of each "
                  f"pair.", ""]
    elif final.escalated:
        L += ["This grid already carries seeds {4, 5}: §8 escalates **once** "
              "and permits no second recomputation, whatever the recomputed "
              "margins above land on.", ""]
    else:
        L += ["No pair's margin lands in the {+1, +2, +3} band; no escalation "
              "is owed and none may be run.", ""]

    L += [NARRATIVE_MARKER, ""]
    return "\n".join(L)


def _svg_path(report_path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(report_path)
    return p.with_suffix(".pareto.svg")


def regenerate(report_path: str | pathlib.Path, generated: str) -> str:
    """Rebuild the generated half, preserving the hand-written half.

    `milestones/s1/run_s1.py:521-535`'s contract verbatim: the tables and the verdict come
    from the record, and the findings — what the run actually meant — cannot be
    generated and must not be destroyed by re-running the report.
    """
    path = pathlib.Path(report_path)
    narrative = ""
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if NARRATIVE_MARKER in old:
            narrative = old.split(NARRATIVE_MARKER, 1)[1]
    head = generated.split(NARRATIVE_MARKER, 1)[0].rstrip("\n")
    return f"{head}\n\n{NARRATIVE_MARKER}\n{narrative.lstrip(chr(10))}"


def write_report(report_path: str | pathlib.Path, verdict: Verdict,
                 scorecard: Scorecard | None,
                 leaks: Mapping[str, Mapping[str, int]], *,
                 escalated: Verdict | None = None) -> pathlib.Path:
    """Render, preserve the narrative, and write the report plus its Pareto."""
    path = pathlib.Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = render_report(verdict, scorecard, leaks, escalated=escalated,
                         report_path=path)
    _svg_path(path).write_text(
        pareto_svg(_final(verdict, escalated), scorecard) + "\n",
        encoding="utf-8", newline="\n")
    path.write_text(regenerate(path, body), encoding="utf-8", newline="\n")
    return path


__all__ = [
    "ARMS",
    "BASELINES",
    "DECISION_RULE",
    "ESCALATION_SEEDS",
    "MARGIN_GATE",
    "NARRATIVE_MARKER",
    "RLM_ARM",
    "ArmCost",
    "CategoryRow",
    "EpisodeCost",
    "Finding",
    "Grid",
    "PairResult",
    "Scorecard",
    "TaskCost",
    "Verdict",
    "VerdictError",
    "cost_scorecard",
    "decide",
    "leak_report",
    "load_grid",
    "pareto_svg",
    "regenerate",
    "render_report",
    "write_report",
]
