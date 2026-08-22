"""§8:343's escalation plan: which (task, arm) cells a run must re-measure.

A gate that lands inside its own uncertainty is not a result. §8 answers that
with escalation seeds -- extra replicates on the cells whose margin was
inconclusive -- and this module is the PLAN for them: which cells, at which
seeds, written next to the ledger so an escalation run is reproducible from the
artifact rather than from whoever remembered to write it down.

The plan only. `run_escalation` stays in `rlm/cli.py` because executing it
means driving arms against live servers, which is composition-root work.

WHY THIS IS ITS OWN MODULE. Choosing and serialising cells is JSON and verdict
inspection -- no server, no dispatcher, no HTTP client -- so it belongs UNDER
§5's dependency-rule lint (`tests/test_import_rules.py` ISOLATED). The lint is
load-bearing here rather than tidy: a planner that could ASK a server which
cells to re-run would be choosing its own replicates, and §8's pre-registration
exists to prevent exactly that.

Extracted from `rlm/cli.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from rlm.errors import ConfigError
from rlm.measure.verdict import BASELINES, PairResult, RLM_ARM, Verdict


def escalation_plan_path(ledger_path, run_id: str) -> Path:
    """Beside the ledger, and named by run: a plan is per-run state, exactly
    like the ledger it sits next to."""
    return Path(ledger_path).parent / f"escalation-{run_id}.json"


def save_escalation_plan(path, verdict, *, seeds) -> Path:
    """Write the pre-escalation verdict's REPORTING FACTS and the plan, BEFORE
    a single escalation episode runs.

    WHY THIS FILE EXISTS. §8 permits exactly one recomputation, and the report
    must state both the pre- and post-escalation figures. Both of those become
    unrecoverable the moment the first escalation episode lands: the store then
    holds a 5-seed grid, and no query over it can reproduce what the 3-seed
    grid decided (`load_grid` builds a cell from every seed present). A run that
    crashed mid-escalation could therefore be resumed to completion and STILL
    not be reportable -- the pre-escalation column would be gone, and the only
    honest thing left to print would be "this grid was escalated against a
    baseline we can no longer name".

    So the figures are written down while they are still true. `pairs` carries
    the whole `PairResult` per baseline (margin, p, CI, the discordant list),
    which is everything `render_report` reads off the pre-escalation verdict,
    and the discordant lists double as the work list a resume finishes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": verdict.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "escalation_seeds": list(seeds),
        "n_tasks": verdict.n_tasks,
        "n_manifest_tasks": verdict.n_manifest_tasks,
        "task_ids": list(verdict.task_ids),
        "arms": list(verdict.arms),
        "passes": {arm: list(tasks) for arm, tasks in verdict.passes.items()},
        "success_rate": dict(verdict.success_rate),
        "gate_pass": verdict.gate_pass,
        "clean_pass": verdict.clean_pass,
        "chunk_sizes": list(verdict.chunk_sizes),
        "escalation_plan": {b: list(t) for b, t in verdict.escalation_plan.items()},
        "pairs": {
            b: {"baseline": p.baseline, "present": p.present,
                "rlm_passes": p.rlm_passes, "baseline_passes": p.baseline_passes,
                "margin": p.margin, "wins": p.wins, "losses": p.losses,
                "discordant": list(p.discordant), "p": p.p,
                "ci": list(p.ci) if p.ci is not None else None,
                "mean_delta": p.mean_delta, "escalates": p.escalates,
                "beats": p.beats}
            for b, p in verdict.pairs.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                    newline="\n")
    return path


def load_escalation_plan(path):
    """The pre-escalation verdict, rebuilt from `save_escalation_plan`'s file,
    or `None` when no escalation was ever planned for this run.

    Rebuilt as a real `Verdict` so `render_report`/`_final` take it unchanged:
    they read `run_id`, `pairs` and `escalation_plan` off the pre-escalation
    verdict and everything else off the post-escalation one. `scores`,
    `categories` and `findings` are left empty on purpose -- the report never
    reads them from this side, and inventing them would be inventing figures.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"the escalation plan at {path} could not be read ({exc}). It "
            f"records the pre-escalation figures §8 requires reported beside "
            f"the post-escalation ones, and they cannot be recomputed from a "
            f"store that already carries seeds {{4, 5}}; move the file aside "
            f"only if you accept losing them") from exc
    pairs = {
        b: PairResult(
            baseline=d["baseline"], present=d["present"],
            rlm_passes=d["rlm_passes"], baseline_passes=d["baseline_passes"],
            margin=d["margin"], wins=d["wins"], losses=d["losses"],
            discordant=tuple(d["discordant"]), p=d["p"],
            ci=tuple(d["ci"]) if d["ci"] is not None else None,
            mean_delta=d["mean_delta"], escalates=d["escalates"],
            beats=d["beats"])
        for b, d in (raw.get("pairs") or {}).items()}
    verdict = Verdict(
        run_id=raw["run_id"], n_tasks=raw["n_tasks"],
        n_manifest_tasks=raw["n_manifest_tasks"],
        task_ids=tuple(raw["task_ids"]), arms=tuple(raw["arms"]),
        passes={a: tuple(t) for a, t in (raw.get("passes") or {}).items()},
        success_rate=dict(raw.get("success_rate") or {}), scores={},
        pairs=pairs, categories=(), findings=(),
        escalation_plan={b: tuple(t)
                         for b, t in (raw.get("escalation_plan") or {}).items()},
        gate_pass=raw["gate_pass"], clean_pass=raw["clean_pass"],
        chunk_sizes=tuple(raw.get("chunk_sizes") or ()), escalated=False)
    return verdict, [int(s) for s in raw.get("escalation_seeds") or ()]
