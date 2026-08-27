"""The gate's accept rule and audit record (spec §4.2, plan step 4.1/4.2).

    python gate/decide.py --root <decision-dir> --id <decision-id> [--audit gate/audit.jsonl]

A decision directory holds `on/<task>/rep<N>/` and `off/<task>/rep<N>/` as written by
gate/run_episode.sh. Nothing here re-runs anything; it reads what the runner recorded.

THE RULE, pre-registered in the spec before any run and not restated after:

  Q  quality non-inferiority. No held-out task that passes in the OFF arm may fail in
     the ON arm. A task passes an arm at >= 2/3 reps, per ARCHITECTURE.md:377. One
     lost task rejects outright; no cost trade is permitted against it.

  K  cost improvement. On per-task total tokens, the median ON/OFF ratio must be
     <= 0.90 AND the upper bound of a 10,000-resample paired bootstrap 95% CI on that
     median must be < 1.00.

  ACCEPT iff Q and K. Otherwise REJECT, and a rejection is a result: the candidate,
  its screens, its grid and its statistic are all kept.

WHY COST GATES HERE WHEN §8 SAYS IT DOES NOT. `ARCHITECTURE.md:384` -- "deliberately
no hard cost gate -- the decision rule stays single" -- governs the S4 decision, where
the question is whether the architecture wins on tasks. This gate asks a different
question about a saturated instrument, and uses the repo's own primary metric:
`ARCHITECTURE.md:345`, "wall-clock per task at fixed quality". Q is the "at fixed
quality" half; K is the cost half, promoted from annotation to gate because on a
benchmark the agent already passes there is no quality headroom to gate on. Scoped to
this gate; §8's S4 rule is untouched (spec D-S3).

WALL IS RECORDED BUT DOES NOT GATE. Thermal drift on this box is of the same order as
the effect being looked for -- +9.9% measured across the spike's 2,000-request
endurance run, against R9's plateau on record. Tokens are the gated quantity; wall and
energy travel as annotations, per §8's rule that a win claim states its cost multiple.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import random
import re
import statistics
import sys
from typing import Any, Sequence

REPO_DEFAULT = pathlib.Path(__file__).resolve().parent.parent

K_MEDIAN_MAX = 0.90        # median ON/OFF per-task token ratio must be at or below this
K_CI_UPPER_MAX = 1.00      # and the bootstrap CI upper bound strictly below this
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260827  # fixed so a verdict is reproducible from the same grid
PASS_FRACTION = 2 / 3      # ARCHITECTURE.md:377


# --------------------------------------------------------------------------- io
def _metric(path: pathlib.Path, key: str) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(key + " "):
            try:
                return float(line.split(" ", 1)[1])
            except ValueError:
                return None
    return None


def _tokens(run: pathlib.Path) -> int | None:
    """Processed prompt + predicted tokens, from the server's own counters.

    The /metrics delta is the [V] source. prime-agent's `usage.cacheRead` disagreed
    with the server across the spike's runs, so the harness's own accounting is [R]
    here and is not what the gate is scored on.
    """
    total = 0
    for key in ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total"):
        pre = _metric(run / "metrics.pre", key)
        post = _metric(run / "metrics.post", key)
        if pre is None or post is None:
            return None
        total += int(post - pre)
    return total


def _answer(run: pathlib.Path) -> str | None:
    ans = run / "answer.txt"
    if ans.exists() and ans.stat().st_size:
        text = ans.read_text(encoding="utf-8", errors="replace").strip()
        m = re.search(r"^FINAL:\s*(.+)$", text, re.M)
        return (m.group(1) if m else text).strip()
    return None


def _in_window(run: pathlib.Path) -> int | None:
    led = run / "ledger.jsonl"
    if not led.exists():
        return None
    kept = None
    for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("event") == "prompt_filtered":
            kept = o.get("kept")
    return kept


# ---------------------------------------------------------------------- scoring
@dataclasses.dataclass
class Cell:
    task: str
    arm: str
    rep: int
    passed: bool
    tokens: int | None
    wall: float | None
    exit_code: int | None
    harness_sha: str | None
    in_window: int | None
    void: bool
    answer: str | None


def collect(root: pathlib.Path, repo: pathlib.Path) -> list[Cell]:
    sys.path.insert(0, str(repo / "src"))
    from rlm.measure.checkers import check  # noqa: E402

    cells: list[Cell] = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ("on", "off")):
        for task_dir in sorted(p for p in arm_dir.iterdir() if p.is_dir() and p.name != "sessions"):
            task = task_dir.name
            spec = json.loads((repo / "bench" / "tasks" / f"{task}.json").read_text(encoding="utf-8"))
            for rep_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                rep = int(re.sub(r"\D", "", rep_dir.name) or 0)
                got = _answer(rep_dir)
                passed = bool(got) and check(spec["checker"], got, spec["answer"])
                wall_txt = (rep_dir / "wall.txt")
                exit_txt = (rep_dir / "exit.txt")
                sha_txt = (rep_dir / "harness.sha256")
                cells.append(Cell(
                    task=task, arm=arm_dir.name, rep=rep, passed=passed,
                    tokens=_tokens(rep_dir),
                    wall=float(wall_txt.read_text().strip()) if wall_txt.exists() else None,
                    exit_code=int(exit_txt.read_text().strip()) if exit_txt.exists() else None,
                    harness_sha=sha_txt.read_text().strip() if sha_txt.exists() else None,
                    in_window=_in_window(rep_dir),
                    void=(rep_dir / "VOID").exists(),
                    answer=got,
                ))
    return cells


def task_passes(cells: Sequence[Cell], task: str, arm: str) -> bool | None:
    reps = [c for c in cells if c.task == task and c.arm == arm]
    if not reps:
        return None
    return sum(c.passed for c in reps) >= PASS_FRACTION * len(reps)


def task_tokens(cells: Sequence[Cell], task: str, arm: str) -> float | None:
    vals = [c.tokens for c in cells if c.task == task and c.arm == arm and c.tokens is not None]
    return statistics.median(vals) if vals else None


def bootstrap_ci(ratios: Sequence[float], resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """Percentile CI on the median of per-task ratios, resampled at TASK level.

    Task level, not cell level: the tasks are the independent units. With 9 held-out
    tasks this interval is wide, and that is the honest width -- a narrower one would
    be manufactured by resampling reps that share a task.
    """
    if not ratios:
        return (float("nan"), float("nan"))
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(ratios)
    meds = []
    for _ in range(resamples):
        meds.append(statistics.median(rng.choices(ratios, k=n)))
    meds.sort()
    lo = meds[int(0.025 * resamples)]
    hi = meds[min(int(0.975 * resamples), resamples - 1)]
    return (lo, hi)


def decide(cells: Sequence[Cell]) -> dict[str, Any]:
    tasks = sorted({c.task for c in cells})
    grid = []
    ratios: list[float] = []
    q_lost: list[str] = []

    for t in tasks:
        on_p, off_p = task_passes(cells, t, "on"), task_passes(cells, t, "off")
        on_tok, off_tok = task_tokens(cells, t, "on"), task_tokens(cells, t, "off")
        ratio = (on_tok / off_tok) if (on_tok and off_tok) else None
        if ratio is not None:
            ratios.append(ratio)
        if off_p and not on_p:
            q_lost.append(t)
        grid.append({
            "task": t,
            "on_pass": on_p, "off_pass": off_p,
            "on_reps": [c.passed for c in cells if c.task == t and c.arm == "on"],
            "off_reps": [c.passed for c in cells if c.task == t and c.arm == "off"],
            "on_tokens": on_tok, "off_tokens": off_tok, "token_ratio": ratio,
            "on_wall": _median_wall(cells, t, "on"), "off_wall": _median_wall(cells, t, "off"),
        })

    med = statistics.median(ratios) if ratios else None
    lo, hi = bootstrap_ci(ratios)
    q_ok = not q_lost
    k_ok = med is not None and med <= K_MEDIAN_MAX and hi < K_CI_UPPER_MAX

    voids = [f"{c.arm}/{c.task}/rep{c.rep}" for c in cells if c.void]
    on_windows = sorted({c.in_window for c in cells if c.arm == "on" and c.in_window is not None})
    off_windows = sorted({c.in_window for c in cells if c.arm == "off" and c.in_window is not None})

    return {
        "verdict": "ACCEPT" if (q_ok and k_ok and not voids) else "REJECT",
        "Q": {"passed": q_ok, "lost_tasks": q_lost,
              "rule": "no held-out task that passes OFF may fail ON (>=2/3 reps)"},
        "K": {"passed": k_ok, "median_ratio": med, "ci95": [lo, hi],
              "thresholds": {"median_max": K_MEDIAN_MAX, "ci_upper_max": K_CI_UPPER_MAX},
              "n_tasks": len(ratios),
              "rule": "median per-task ON/OFF token ratio, task-level paired bootstrap"},
        "voids": voids,
        "in_window_entries": {"on": on_windows, "off": off_windows},
        "grid": grid,
    }


def _median_wall(cells: Sequence[Cell], task: str, arm: str) -> float | None:
    vals = [c.wall for c in cells if c.task == task and c.arm == arm and c.wall is not None]
    return round(statistics.median(vals), 1) if vals else None


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="decision directory (holds on/ and off/)")
    ap.add_argument("--id", required=True, help="decision id, recorded in the audit row")
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--audit", default=None, help="append the audit row here (JSONL)")
    ap.add_argument("--screens", default=None, help="a screens verdict JSON to fold into the row")
    ap.add_argument("--note", default="", help="free text recorded with the decision")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    root = pathlib.Path(args.root)
    cells = collect(root, repo)
    if not cells:
        print(f"no episodes under {root}", file=sys.stderr)
        return 2
    result = decide(cells)

    cand = root / "candidate.json"
    cand_sha = hashlib.sha256(cand.read_bytes()).hexdigest() if cand.exists() else None
    split_path = repo / "bench" / "splits" / "s6lite-v0.json"
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.exists() else None

    print(f"=== decision {args.id}: {result['verdict']}")
    print(f"    candidate sha  {(cand_sha or 'none')[:16]}")
    print(f"    split sha      {(split_sha or 'none')[:16]}")
    print(f"    episodes       {len(cells)}   voids {len(result['voids'])}")
    print(f"    in-window      on={result['in_window_entries']['on']} off={result['in_window_entries']['off']}")
    print()
    print(f"    {'task':11s} {'OFF':>10s} {'ON':>10s}  {'off tok':>9s} {'on tok':>9s} {'ratio':>7s}"
          f"  {'off s':>7s} {'on s':>7s}")
    for row in result["grid"]:
        def fmt(p, reps):
            return f"{'pass' if p else 'FAIL'} {sum(reps)}/{len(reps)}"
        print(f"    {row['task']:11s} {fmt(row['off_pass'], row['off_reps']):>10s}"
              f" {fmt(row['on_pass'], row['on_reps']):>10s}"
              f"  {row['off_tokens'] or 0:9.0f} {row['on_tokens'] or 0:9.0f}"
              f" {row['token_ratio'] or float('nan'):7.3f}"
              f"  {row['off_wall'] or 0:7.1f} {row['on_wall'] or 0:7.1f}")
    print()
    q, k = result["Q"], result["K"]
    print(f"    Q  {'PASS' if q['passed'] else 'FAIL'}   lost: {q['lost_tasks'] or 'none'}")
    ci = k["ci95"]
    print(f"    K  {'PASS' if k['passed'] else 'FAIL'}   median ratio "
          f"{k['median_ratio'] if k['median_ratio'] is None else round(k['median_ratio'], 3)}"
          f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  n={k['n_tasks']} tasks"
          f"  (need <= {K_MEDIAN_MAX} and CI upper < {K_CI_UPPER_MAX})")
    if result["voids"]:
        print(f"    VOID blocks: {result['voids']}")

    row = {
        "decision_id": args.id, "verdict": result["verdict"],
        "candidate_sha256": cand_sha, "split_sha256": split_sha,
        "episodes": len(cells), "note": args.note,
        "screens": json.loads(pathlib.Path(args.screens).read_text(encoding="utf-8")) if args.screens else None,
        **{k2: v for k2, v in result.items() if k2 != "verdict"},
    }
    (root / "decision.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.audit:
        with open(args.audit, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"\n    audit row appended to {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
