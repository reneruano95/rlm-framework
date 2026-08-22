"""§8's wall-clock projection model, and the calibration table `--smoke` prints.

These are PROJECTIONS from measured inputs, never measurements. That distinction
is the reason the smoke run prints them beside the numbers it has just measured
instead of trusting them, and the reason they live in a module you can find
rather than buried in the middle of the operator surface.

WHY THIS IS ITS OWN MODULE. Pure arithmetic over manifest entries and ledger
records: no server, no dispatcher, no HTTP client. It therefore belongs UNDER
§5's dependency-rule lint (`tests/test_import_rules.py` ISOLATED), where
something checks that it stays that way. `src/rlm/cli.py` re-exports every public
name, so the existing import sites keep working.

Extracted from `src/rlm/cli.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

# §8's pre-registered within-block arm order. Imported rather than duplicated:
# the calibration table projects a cost per arm, and a projection over a
# DIFFERENT arm list than the one the grid runs is a silently wrong number.
# `rlm.measure.bench` is itself lint-covered, so this does not widen the dependency
# surface -- see tests/test_import_rules.py.
from rlm.measure.bench import ARM_ORDER


# ---- the projection constants a --smoke run calibrates against -------------- #
#
# Copied from `milestones/s2/aggregation_options.py:19-38` rather than imported: `milestones/s2/` is
# an analysis directory, not part of the shipped wheel (`pyproject.toml`
# packages = ["rlm"]), and `rlm/` importing it would break `rlm` on install for
# the sake of four floats. They are PROJECTIONS from measured inputs, never
# measurements -- which is exactly why the smoke run prints them beside the
# numbers it just measured instead of trusting them.
PROJ_S_PER_WINDOW = 2.78          # §8, serial, both sub-calls per window
PROJ_CHEAP_ARM_S = 60.0           # a single-shot arm: one big call + scoring
PROJ_NON_AGG_EXPENSIVE_S = 450.0  # needle/synthesis/codeQA on a chunked arm
PROJ_ROOT_OVERHEAD_FRAC = 0.30    # share of an agg episode NOT in leaf windows
PROJ_S4_BUDGET_H = 60             # §8's pre-registered wall budget for S4
#: §8: "two chunked-and-exposed (RLM, B2) versus two single-shot-and-spared".
CHUNKED_ARMS = ("rlm", "b2")


# --------------------------------------------------------------------------- #
# --smoke: the calibration table
# --------------------------------------------------------------------------- #


def projected_episode_s(entry, arm: str, *, wall_cap: float) -> float:
    """What `milestones/s2/aggregation_options.py` predicts one (task, arm) episode costs.

    Aggregation on a chunked arm is the only size-dependent case: its windows
    are stated in the manifest (§8 requires it, so "the affordability claim is
    checkable rather than assumed"), and the root's share is added back the
    way `s2.episode_seconds` does before the per-episode wall cap applies.
    """
    if arm not in CHUNKED_ARMS:
        return PROJ_CHEAP_ARM_S
    if entry.category == "aggregation" and entry.windows:
        leaf_s = entry.windows * PROJ_S_PER_WINDOW
        return min(leaf_s / (1.0 - PROJ_ROOT_OVERHEAD_FRAC), float(wall_cap))
    return PROJ_NON_AGG_EXPENSIVE_S


def projected_grid_hours(manifest, *, seeds, arms, wall_cap: float,
                          measured: dict | None = None) -> float:
    """The full frozen grid in hours, per (arm, CATEGORY) seconds.

    `measured` (keyed `(arm, category)`) overrides the projection wherever the
    smoke run actually timed something; everything it did not reach falls back
    to the pre-registered constant. Per category rather than per task because
    that is the granularity a 4-episode smoke run can support: one measured
    needle episode says something about the other seven needle tasks and
    nothing at all about aggregation.
    """
    total = 0.0
    for entry in manifest.tasks:
        for arm in arms:
            per = (measured or {}).get((arm, entry.category))
            if per is None:
                per = projected_episode_s(entry, arm, wall_cap=wall_cap)
            total += per * len(seeds)
    return total / 3600.0


def _median(values) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def print_calibration(records, manifest, cfg, *, arms, run_id, out) -> None:
    """§8's affordability claim, measured against its own projection.

    Prints one row per measured cell and then the projected full-grid hours
    twice: once from the pre-registered constants (what the plan was costed
    with) and once with the measurements substituted in. Both, because a
    single number would hide which of the two moved.
    """
    by_id = {t.task_id: t for t in manifest.tasks}
    wall_cap = float(cfg.scaffold.budgets.max_wall_clock_s)
    print(f"\n--- smoke calibration (run_id {run_id}; NOT scored, no report "
          f"written) ---", file=out)
    print(f"projection constants (milestones/s2/aggregation_options.py): "
          f"{PROJ_NON_AGG_EXPENSIVE_S:.0f} s chunked-arm non-aggregation, "
          f"{PROJ_CHEAP_ARM_S:.0f} s single-shot, {PROJ_S_PER_WINDOW} s/window "
          f"aggregation (+{PROJ_ROOT_OVERHEAD_FRAC:.0%} root overhead)",
          file=out)
    print(f"{'arm':<5} {'task':<12} {'category':<12} {'measured s':>11} "
          f"{'projected s':>12} {'ratio':>7}  outcome", file=out)
    for record in records:
        entry = by_id.get(record["task_id"])
        if entry is None:
            continue
        want = projected_episode_s(entry, record["arm"], wall_cap=wall_cap)
        got = record.get("wall_s")
        # A cell that refused never opened an episode, so it has no measured
        # seconds. Printing 0.0 for it would read as "instant" -- a
        # measurement -- on the one table whose job is to be believed.
        measured_s = "n/a" if got is None else f"{got:.1f}"
        ratio = f"{got / want:.2f}x" if got and want else "n/a"
        print(f"{record['arm']:<5} {record['task_id']:<12} {entry.category:<12} "
              f"{measured_s:>11} {want:>12.1f} "
              f"{ratio:>7}  {record['outcome']}"
              + (f" ({record['reason']})" if record.get("reason") else ""),
              file=out)

    measured: dict[tuple[str, str], float] = {}
    for arm in arms:
        for category in {t.category for t in manifest.tasks}:
            walls = [r["wall_s"] for r in records
                     if r["arm"] == arm and by_id.get(r["task_id"]) is not None
                     and by_id[r["task_id"]].category == category]
            median = _median(walls)
            if median is not None:
                measured[(arm, category)] = median

    full_seeds = list(cfg.benchmark.seeds)
    from_constants = projected_grid_hours(manifest, seeds=full_seeds,
                                           arms=ARM_ORDER, wall_cap=wall_cap)
    from_measured = projected_grid_hours(manifest, seeds=full_seeds,
                                          arms=ARM_ORDER, wall_cap=wall_cap,
                                          measured=measured)
    agg_ep = max((projected_episode_s(t, "rlm", wall_cap=wall_cap)
                  for t in manifest.tasks if t.category == "aggregation"),
                 default=0.0)
    escalation_h = 32 * agg_ep * 0.5 / 3600.0     # §8: "typically 8-32 extra episodes"
    total = from_measured + escalation_h
    print(f"\nfull grid = {len(manifest.tasks)} tasks x {len(full_seeds)} seeds "
          f"x {len(ARM_ORDER)} arms, plus §8's escalation allowance "
          f"({escalation_h:.1f} h, up to 32 extra episodes)", file=out)
    print(f"  from the pre-registered constants:  {from_constants:>6.1f} h grid "
          f"+ {escalation_h:.1f} h = {from_constants + escalation_h:>6.1f} h",
          file=out)
    print(f"  with these measurements substituted:{from_measured:>6.1f} h grid "
          f"+ {escalation_h:.1f} h = {total:>6.1f} h", file=out)
    print(f"  the measured figure is the one to judge: {total:.1f} h against "
          f"§8's {PROJ_S4_BUDGET_H} h budget — "
          f"{'WITHIN' if total <= PROJ_S4_BUDGET_H else 'OVER'}", file=out)
    if total > PROJ_S4_BUDGET_H:
        print(f"  ** the projection breaches the pre-registered "
              f"{PROJ_S4_BUDGET_H} h budget: that is a decision for a human, "
              f"not a number to proceed past **", file=out)


