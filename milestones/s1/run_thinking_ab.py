"""S1 re-run on the swapped root, with the root's THINKING as the arm.

Two things are owed at once and this settles both in one pass:

1. **The S5 swap invalidated S1's gate.** S1 passed 3/3 crediting
   "root-as-programmer", measured on Qwen3.6-27B. The root is now Qwen3.8-27B,
   so that result does not transfer and has to be re-taken.

2. **`scaffold.root.enable_thinking` is contested.** `milestones/s2/ROOT-THINKING.md`
   measured thinking OFF at 5/8 against 8/8 for every thinking arm on
   single-turn arithmetic -- but thinking DESTROYS multi-turn prefix reuse
   (85% -> 4.7%, ~24 s of extra prefill per serial root turn). Single-turn
   accuracy and multi-turn cost disagree, and only a whole EPISODE has both in
   scope at once.

WHAT THIS DELIBERATELY DOES NOT DO. `milestones/s1/run_s1.py --phase ab` runs the root
prompt A/B over variants v1 and v2. §9 S1 CLOSED that A/B at two variants and
pinned root.v3 by tie-break; re-running it would produce v1-vs-v2 numbers that
read as reopening a closed decision. So this driver calls the same
`rlm_attempt` with `variant=None`, which `variant_config` treats as "leave the
shipped root prompt alone" -- the episodes therefore run **root.v3, what
actually ships**, and the only thing that varies is the config file.

The two configs differ in exactly one key (verified: same root model, same
prompt sha256 pins, same leaf block), so the arm is the flag and nothing else.

Both servers must already be running at the shipped launch lines -- this driver
does not own them, exactly like `run_s1.py`.

    uv run --python 3.12 --no-project python -m s1.run_thinking_ab
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import yaml

from rlm.config import load_config
from rlm.lifecycle import Lifecycle
from s1.run_s1 import FIXTURES, _lifecycle_path, load_fixture, rlm_attempt

ARMS = {"think-off": "config.yaml", "think-on": "config-thinkon.yaml"}
OUT = Path(__file__).resolve().parent / "results" / "thinking_ab.jsonl"


async def main_async(args) -> int:
    fixtures = {n: load_fixture(n) for n in args.fixtures}
    rows: list[dict] = []

    for arm, cfg_path in ARMS.items():
        cfg = load_config(Path(cfg_path))
        raw_cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        thinking = cfg.scaffold.root.enable_thinking
        print(f"\n=== arm {arm}  ({cfg_path}, "
              f"root.enable_thinking={thinking}) ===")
        lifecycle = Lifecycle(_lifecycle_path(cfg, None))
        try:
            for name in args.fixtures:
                for seed in args.seeds:
                    t0 = time.perf_counter()
                    rec = await rlm_attempt(raw_cfg, fixtures[name],
                                            None, seed, lifecycle)
                    rec = dict(rec)
                    rec["arm"] = arm
                    rec["enable_thinking"] = thinking
                    rec["driver_wall_s"] = round(time.perf_counter() - t0, 1)
                    rows.append(rec)
                    print(f"  {name}/seed{seed}: {rec['outcome']} "
                          f"({rec.get('outcome_reason')}) "
                          f"turns={rec.get('root_turns')} "
                          f"leaf={rec.get('leaf_calls')} "
                          f"{rec['wall_clock_s']}s "
                          f"answer={str(rec.get('final_answer'))[:90]!r}")
        finally:
            lifecycle.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def passed(r: dict) -> bool:
        """Use the RECORDED outcome, not a re-derived substring test.

        `rlm_attempt` already stores `passed = outcome == "success"`, and that
        outcome comes from `Task.check` -- the checker the gate is defined
        against, with its own normalisation. Re-implementing it here would let
        this driver score an episode differently from the gate it is feeding."""
        return bool(r.get("passed"))

    print(f"\n=== S1 re-run on the swapped root: thinking A/B ===")
    print(f"  {'arm':<11} {'passed':>8} {'med wall':>9} {'med turns':>10} "
          f"{'med leaf':>9} {'budget/ctx kills':>17}")
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        if not sel:
            continue
        ok = sum(passed(r) for r in sel)
        kills = sum(1 for r in sel
                    if str(r.get("outcome_reason") or "") in
                    ("budget_kill", "context_exhausted"))
        print(f"  {arm:<11} {ok:>5}/{len(sel):<2} "
              f"{statistics.median(r['wall_clock_s'] for r in sel):>8.1f}s "
              f"{statistics.median(r.get('root_turns') or 0 for r in sel):>10.0f} "
              f"{statistics.median(r.get('leaf_calls') or 0 for r in sel):>9.0f} "
              f"{kills:>17}")
    print(f"\n  wrote {OUT}")
    print("  I5 reads this as wall-clock at FIXED QUALITY: thinking only wins "
          "if it is more correct, and its wall cost is stated beside the win.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", nargs="+", default=list(FIXTURES),
                    choices=list(FIXTURES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
