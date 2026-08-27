#!/usr/bin/env python3
"""Join the scored spike runs with the scaffold baselines (plan section 5).

    python compare.py --runs-root <dir> [--usage-csv <file> ...] [--out <csv>]

Inputs
------
* every ``score.json`` under ``--runs-root`` (written by ``score.py``) -- the
  authority on pass/fail, and on ``task`` / ``arm`` / ``run``;
* the per-run usage CSV(s) written by ``usage.py`` -- the authority on
  ``wall_s``, ``turns`` and tokens.  Given explicitly with ``--usage-csv``
  (repeatable), otherwise every ``*.csv`` under ``--runs-root`` whose header
  carries ``task`` and ``wall_s`` is used.  The CSV columns read are those the
  plan pins: ``task, run, arm, pass, wall_s, turns, prompt_tokens_metrics,
  predicted_tokens_metrics, tokens_in_harness, tokens_out_harness``.

Join key is ``(task, run, arm)`` with the run id normalised (``run1`` -> ``1``),
because the plan's run dirs are ``/home/spike/work/A/<task>/run<i>`` and the
same task and run number occur again in the A' arm.  When the two sides carry
different arm labels for one (task, run) and each side has exactly one such row,
they are joined anyway and the group is flagged ``arm_mismatch``.  A CSV row
with no score.json is still used (its own ``pass`` column is then the verdict),
and a score.json with no CSV row contributes pass/fail but no cost.

Outputs
-------
* the A-cost table: task, arm, pass_count/n, median wall, wall ratios against
  the DFlash2 and S4 scaffold medians, median tokens from ``/metrics`` and the
  token ratio against DFlash2;
* the pre-registered A1/A2/A3 reading (section 2): a task counts as passing
  when at least two thirds of its runs pass, then >=6 tasks = A1, 3-5 = A2,
  <=2 = A3.  Printed per arm, plus a best-of-arms line (a task passing in A
  *or* A', which is what section 2's A2 row uses to admit a category to Phase B).
  Until all eight tasks of an arm are scored the verdict prints as
  "provisional", so a half-finished Day 1 cannot be quoted as an A3;
* with ``--out``, one CSV row per (task, arm) carrying every column above plus
  the turn medians.

Ratios are prime-agent / scaffold, so >1 means the spike was slower or dearer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import median

# --------------------------------------------------------------------------- #
# Scaffold per-task medians, hardcoded from plan section 4 (S4 run 1cbafb8f and
# DFlash2 run c1740386, both verified 2026-08-26).  wall in seconds, tokens
# total, turns count.

BASELINES = {
    "agg-03":    {"s4": (50.7, 7306, 4),  "dflash2": (37.1, 7601, 4)},
    "agg-04":    {"s4": (97.0, 17918, 6), "dflash2": (76.8, 19394, 7)},
    "agg-07":    {"s4": (84.8, 14405, 6), "dflash2": (95.5, 21802, 7)},
    "codeqa-01": {"s4": (69.3, 8856, 5),  "dflash2": (66.7, 11405, 6)},
    "codeqa-03": {"s4": (72.2, 8788, 5),  "dflash2": (108.0, 24567, 10)},
    "codeqa-05": {"s4": (87.5, 13816, 7), "dflash2": (98.4, 17149, 7)},
    "needle-02": {"s4": (78.8, 13077, 6), "dflash2": (43.3, 8872, 5)},
    "synth-02":  {"s4": (95.4, 15733, 6), "dflash2": (142.9, 36555, 8)},
}
PHASE_A_TASKS = list(BASELINES)

# section 2: >=6/8 tasks pass = A1, 3-5 = A2, <=2 = A3.
A1_MIN, A2_MIN = 6, 3
PASS_FRACTION = 2.0 / 3.0          # ">=2/3 runs pass" makes a task pass


# --------------------------------------------------------------------------- #
# small helpers


def norm_run(value) -> str:
    """`run1`, `run_1`, `1`, `/work/A/agg-03/run1` -> `1`."""
    s = str(value or "").strip().replace("\\", "/").rstrip("/")
    s = s.rsplit("/", 1)[-1]
    m = re.search(r"(\d+)\s*$", s)
    return m.group(1) if m else s


def to_float(value):
    try:
        f = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def to_bool(value):
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "pass", "passed", "ok"):
        return True
    if s in ("false", "0", "no", "n", "fail", "failed"):
        return False
    return None


def fmt(value, spec="%.1f"):
    return "n/a" if value is None else (spec % value)


# --------------------------------------------------------------------------- #
# loading


def load_scores(runs_root: Path) -> dict:
    """(task, run, arm) -> score record, from every score.json under the root."""
    rows = {}
    for path in sorted(runs_root.rglob("score.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("compare.py: warning: unreadable %s (%r)" % (path, exc),
                  file=sys.stderr)
            continue
        task = str(rec.get("task") or "").strip()
        if not task:
            continue
        run = norm_run(rec.get("run") or path.parent.name)
        arm = str(rec.get("arm") or infer_arm(path.parent)).strip() or "P1"
        rows[(task, run, arm)] = {
            "task": task, "run": run, "arm": arm,
            "pass": bool(rec.get("pass")),
            "final_changed": bool(rec.get("final_changed")),
            "source": rec.get("source"),
            "score_path": str(path),
        }
    return rows


def infer_arm(run_dir: Path) -> str:
    low = str(run_dir).replace("\\", "/").lower()
    for token in ("/p2", "p2/", "aprime", "a-prime", "a_prime", "strat"):
        if token in low:
            return "P2"
    return "P1"


def find_usage_csvs(runs_root: Path, given, exclude=None) -> list:
    """Every per-run usage CSV under the root, by exact header fields.

    The match is on the field names `task` and `wall_s`, not on a substring:
    this tool's own --out file has `median_wall_s` and must never be read back
    in as input (which would double every run)."""
    if given:
        return [Path(p) for p in given]
    exclude = {Path(p).resolve() for p in (exclude or []) if p}
    out = []
    for path in sorted(runs_root.rglob("*.csv")):
        if path.resolve() in exclude:
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                fields = {f.strip().lower()
                          for f in (fh.readline() or "").split(",")}
        except Exception:
            continue
        if {"task", "wall_s"} <= fields and not (fields & set(CSV_HEAD_MARKERS)):
            out.append(path)
    return out


# header fields that identify compare.py's own output, never an input
CSV_HEAD_MARKERS = ("median_wall_s", "n_runs", "pass_count")


def load_usage(paths: list) -> dict:
    """(task, run, arm) -> usage row, from usage.py's CSV(s)."""
    rows = {}
    for path in paths:
        if not path.is_file():
            print("compare.py: warning: no usage CSV at %s" % path,
                  file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            for raw in csv.DictReader(fh):
                rec = {(k or "").strip().lower(): v
                       for k, v in raw.items() if k}
                task = str(rec.get("task") or "").strip()
                if not task:
                    continue
                run = norm_run(rec.get("run"))
                arm = (str(rec.get("arm") or "").strip() or "P1").upper()
                metrics = None
                p_in = to_float(rec.get("prompt_tokens_metrics"))
                p_out = to_float(rec.get("predicted_tokens_metrics"))
                if p_in is not None or p_out is not None:
                    metrics = (p_in or 0.0) + (p_out or 0.0)
                harness = None
                h_in = to_float(rec.get("tokens_in_harness"))
                h_out = to_float(rec.get("tokens_out_harness"))
                if h_in is not None or h_out is not None:
                    harness = (h_in or 0.0) + (h_out or 0.0)
                rows[(task, run, arm)] = {
                    "task": task, "run": run, "arm": arm,
                    "wall_s": to_float(rec.get("wall_s")),
                    "turns": to_float(rec.get("turns")),
                    "tokens_metrics": metrics if metrics else None,
                    "tokens_harness": harness if harness else None,
                    "pass_csv": to_bool(rec.get("pass")),
                    "csv_path": str(path),
                }
    return rows


# --------------------------------------------------------------------------- #
# joining and grouping


def _record(task, run, arm, s, u, arm_mismatch=False) -> dict:
    passed = (s or {}).get("pass")
    if passed is None:
        passed = (u or {}).get("pass_csv")
    return {
        "task": task, "run": run, "arm": arm,
        "pass": bool(passed),
        "have_score": s is not None,
        "have_usage": u is not None,
        "arm_mismatch": arm_mismatch,
        "wall_s": (u or {}).get("wall_s"),
        "turns": (u or {}).get("turns"),
        "tokens_metrics": (u or {}).get("tokens_metrics"),
        "tokens_harness": (u or {}).get("tokens_harness"),
    }


def join(scores: dict, usage: dict) -> list:
    """One record per run.  Exact (task, run, arm) matches first; then a single
    leftover usage row for the same (task, run) is attached to a single leftover
    score (the two sides can disagree on the arm only by inference), flagged as
    arm_mismatch.  A run never lands in two arm groups: agg-04 legitimately runs
    in both P1 and P2 with the same run number, so the arm is part of identity."""
    out = []
    score_left = dict(scores)
    usage_left = dict(usage)
    for key in sorted(set(scores) & set(usage)):
        out.append(_record(key[0], key[1], key[2],
                           score_left.pop(key), usage_left.pop(key)))
    for key in sorted(score_left):
        task, run, arm = key
        alt = [k for k in usage_left if k[0] == task and k[1] == run]
        same = [k for k in score_left if k[0] == task and k[1] == run]
        u = usage_left.pop(alt[0]) if (len(alt) == 1 and len(same) == 1) else None
        out.append(_record(task, run, arm, score_left[key], u,
                           arm_mismatch=u is not None))
    for key in sorted(usage_left):        # a usage row with no score.json
        out.append(_record(key[0], key[1], key[2], None, usage_left[key]))
    return sorted(out, key=lambda r: (r["task"], r["arm"], r["run"]))


def med(values):
    vals = [v for v in values if v is not None]
    return median(vals) if vals else None


def group(records: list) -> dict:
    """(task, arm) -> aggregate."""
    buckets = {}
    for rec in records:
        buckets.setdefault((rec["task"], rec["arm"]), []).append(rec)
    out = {}
    for key, runs in buckets.items():
        task, arm = key
        n = len(runs)
        passes = sum(1 for r in runs if r["pass"])
        tokens = [r["tokens_metrics"] for r in runs]
        source = "metrics"
        if not any(t is not None for t in tokens):
            tokens = [r["tokens_harness"] for r in runs]
            source = "harness" if any(t is not None for t in tokens) else "-"
        base = BASELINES.get(task)
        wall = med(r["wall_s"] for r in runs)
        tok = med(tokens)
        out[key] = {
            "task": task, "arm": arm, "n": n, "passes": passes,
            "wall_n": sum(1 for r in runs if r["wall_s"] is not None),
            "tokens_n": sum(1 for t in tokens if t is not None),
            "arm_mismatch": sum(1 for r in runs if r["arm_mismatch"]),
            "task_pass": n > 0 and passes >= math.ceil(PASS_FRACTION * n - 1e-9),
            "median_wall": wall,
            "median_turns": med(r["turns"] for r in runs),
            "median_tokens": tok,
            "tokens_source": source,
            "s4": base["s4"] if base else None,
            "dflash2": base["dflash2"] if base else None,
            "ratio_s4_wall": (wall / base["s4"][0]) if (base and wall) else None,
            "ratio_df_wall": (wall / base["dflash2"][0]) if (base and wall) else None,
            "ratio_s4_tok": (tok / base["s4"][1]) if (base and tok) else None,
            "ratio_df_tok": (tok / base["dflash2"][1]) if (base and tok) else None,
            "missing_usage": sum(1 for r in runs if not r["have_usage"]),
            "missing_score": sum(1 for r in runs if not r["have_score"]),
        }
    return out


# --------------------------------------------------------------------------- #
# output


HEAD = ["task", "arm", "pass", "med_wall_s", "vs_dflash2_wall", "vs_s4_wall",
        "med_tokens", "vs_dflash2_tok"]


def print_table(groups: dict) -> None:
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        g = groups[key]
        star = "*" if g["tokens_source"] == "harness" else ""
        rows.append([
            g["task"], g["arm"], "%d/%d" % (g["passes"], g["n"]),
            fmt(g["median_wall"]),
            fmt(g["ratio_df_wall"], "%.2fx"),
            fmt(g["ratio_s4_wall"], "%.2fx"),
            (fmt(g["median_tokens"], "%.0f") + star),
            fmt(g["ratio_df_tok"], "%.2fx"),
        ])
    widths = [max(len(HEAD[i]), *(len(r[i]) for r in rows)) if rows
              else len(HEAD[i]) for i in range(len(HEAD))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(HEAD))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    if any(g["tokens_source"] == "harness" for g in groups.values()):
        print("* tokens from prime-agent's own accounting [R]; the /metrics "
              "columns were absent or zero in the usage CSV")
    short = ["%s/%s (wall %d/%d, tokens %d/%d runs)"
             % (g["task"], g["arm"], g["wall_n"], g["n"], g["tokens_n"], g["n"])
             for g in (groups[k] for k in sorted(groups))
             if g["wall_n"] < g["n"] or g["tokens_n"] < g["n"]]
    if short:
        print("median over fewer runs than were scored: " + "; ".join(short))
    missing = ["%s/%s" % k for k, g in sorted(groups.items())
               if g["missing_score"]]
    if missing:
        print("no score.json (pass taken from the usage CSV) for: "
              + ", ".join(missing))
    mism = ["%s/%s" % k for k, g in sorted(groups.items()) if g["arm_mismatch"]]
    if mism:
        print("arm disagreed between score.json and the usage CSV for: "
              + ", ".join(mism) + " (score.json's arm kept)")


def reading(n_pass: int) -> str:
    if n_pass >= A1_MIN:
        return "A1 (drives it)"
    if n_pass >= A2_MIN:
        return "A2 (partial)"
    return "A3 (collapse)"


def print_reading(groups: dict) -> None:
    arms = sorted({k[1] for k in groups})
    print("\nPhase A reading (a task passes when >=2/3 of its runs pass; the "
          "reading is defined over the 8 Phase-A tasks)")

    def line(label, passed, scored):
        verdict, note = reading(len(passed)), ""
        if scored < len(PHASE_A_TASKS):
            # Never quote a bare A3 off a half-run phase -- and the A' arm only
            # ever runs the failing categories, so its denominator is never 8.
            verdict = "provisional " + verdict
            note = "  (only %d of the 8 tasks scored)" % scored
        print("  %-22s %d/8 tasks pass -> %s%s%s"
              % (label, len(passed), verdict, note,
                 ("\n%26spassed: %s" % ("", ", ".join(passed))) if passed else ""))

    for arm in arms:
        passed = sorted(k[0] for k, g in groups.items()
                        if k[1] == arm and g["task_pass"] and k[0] in BASELINES)
        scored = len({k[0] for k in groups if k[1] == arm and k[0] in BASELINES})
        line("arm %s" % arm, passed, scored)
    if len(arms) > 1:
        best = sorted({k[0] for k, g in groups.items()
                       if g["task_pass"] and k[0] in BASELINES})
        scored = len({k[0] for k in groups if k[0] in BASELINES})
        line("best-of-arms (A or A')", best, scored)
    not_run = [t for t in PHASE_A_TASKS
               if not any(k[0] == t for k in groups)]
    if not_run:
        print("  not scored yet: " + ", ".join(not_run))


CSV_HEAD = ["task", "arm", "n_runs", "pass_count", "task_pass",
            "wall_n", "tokens_n", "median_wall_s", "s4_wall_s", "dflash2_wall_s",
            "ratio_vs_s4_wall", "ratio_vs_dflash2_wall",
            "median_tokens", "tokens_source", "s4_tokens", "dflash2_tokens",
            "ratio_vs_s4_tokens", "ratio_vs_dflash2_tokens",
            "median_turns", "s4_turns", "dflash2_turns",
            "runs_missing_usage", "runs_missing_score", "arm_mismatch"]


def write_csv(groups: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEAD)
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            g = groups[key]
            s4 = g["s4"] or (None, None, None)
            df = g["dflash2"] or (None, None, None)
            writer.writerow([
                g["task"], g["arm"], g["n"], g["passes"], g["task_pass"],
                g["wall_n"], g["tokens_n"],
                fmt(g["median_wall"], "%.2f"), fmt(s4[0]), fmt(df[0]),
                fmt(g["ratio_s4_wall"], "%.3f"), fmt(g["ratio_df_wall"], "%.3f"),
                fmt(g["median_tokens"], "%.0f"), g["tokens_source"],
                fmt(s4[1], "%.0f"), fmt(df[1], "%.0f"),
                fmt(g["ratio_s4_tok"], "%.3f"), fmt(g["ratio_df_tok"], "%.3f"),
                fmt(g["median_turns"], "%.1f"), fmt(s4[2], "%.0f"),
                fmt(df[2], "%.0f"),
                g["missing_usage"], g["missing_score"], g["arm_mismatch"],
            ])
    print("\nwrote %s" % out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Join scored spike runs with the scaffold baselines.")
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--usage-csv", action="append",
                    help="usage.py CSV (repeatable); default: discovered "
                         "under --runs-root")
    ap.add_argument("--out", type=Path, help="write the joined table as CSV")
    args = ap.parse_args(argv)

    runs_root = args.runs_root.resolve()
    if not runs_root.is_dir():
        print("compare.py: error: --runs-root %s is not a directory" % runs_root,
              file=sys.stderr)
        return 2

    scores = load_scores(runs_root)
    csvs = find_usage_csvs(runs_root, args.usage_csv, [args.out])
    usage = load_usage(csvs)
    records = join(scores, usage)
    if not records:
        print("compare.py: nothing to compare: no score.json and no usage CSV "
              "row under %s" % runs_root, file=sys.stderr)
        return 1
    groups = group(records)

    print("%d run(s) from %d score.json and %d usage row(s) in %s"
          % (len(records), len(scores), len(usage),
             ", ".join(str(p) for p in csvs) or "no usage CSV"))
    print()
    print_table(groups)
    print_reading(groups)
    if args.out:
        write_csv(groups, args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
