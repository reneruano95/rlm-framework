"""The reliability reading (spec §4.2b), scored on discordant ON/OFF pairs.

    python gate/reliability.py --root <decision-dir> --id <decision-id>

WHY THIS EXISTS ALONGSIDE decide.py. Measured across decisions pc-01 and pc-02, seven
of the nine held-out tasks pass 3/3 in BOTH arms every time. They carry no quality
signal, and on the cost statistic their per-task ratios cluster at 1.0 and drag the
median toward "no effect" whatever an artifact does elsewhere. The only dynamic range
in v1 is agg-06 and agg-07, which fail intermittently in both arms at roughly 15-20%
of episodes -- and that is exactly where §4.2's design has the least support, at 2
tasks x 3 reps.

A cost gate wants breadth: many tasks, few reps, median over tasks. A reliability gate
wants depth: the tasks that actually fail, many reps, a failure-rate comparison. The
blocked A/B is right for both; the sampling is not. This file is the second reading,
not a replacement for the first.

THE RULE, fixed in the spec before decision pc-03 ran:

  unit        the episode, in (task, rep) blocks with ON and OFF adjacent
  statistic   b = OFF passed / ON failed;  c = ON passed / OFF failed
              exact one-sided binomial (McNemar) on the b + c discordant pairs
  ACCEPT      c > b and p <= 0.05.  A tie, b > c, or c > b at p > 0.05 is a REJECT.
  cost        recorded, does not gate

POWER, stated before the run rather than after: at a ~15-20% failure rate, 20 pairs
gives roughly 3-4 discordant pairs under a real effect, and p <= 0.05 needs about
5-0 or 6-1. This design can confirm a large effect and cannot rule out a moderate
one. A REJECT here means "not demonstrated at this n", never "no effect".
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

REPO_DEFAULT = pathlib.Path(__file__).resolve().parent.parent
ALPHA = 0.05


def exact_mcnemar_one_sided(b: int, c: int) -> float:
    """P(X >= c) for X ~ Binomial(b + c, 0.5) -- the exact one-sided sign test.

    One-sided in the direction the artifact is supposed to help: `c` is the count of
    pairs where the ON arm passed and OFF failed. With no discordant pairs there is no
    evidence either way and the p-value is 1.0, not 0.
    """
    n = b + c
    if n == 0:
        return 1.0
    total = sum(math.comb(n, k) for k in range(c, n + 1))
    return total / (2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--audit", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    sys.path.insert(0, str(repo))
    from gate.decide import collect  # noqa: E402

    cells = collect(pathlib.Path(args.root), repo)
    if not cells:
        print(f"no episodes under {args.root}", file=sys.stderr)
        return 2

    by_block: dict[tuple[str, int], dict[str, object]] = {}
    for c in cells:
        by_block.setdefault((c.task, c.rep), {})[c.arm] = c

    pairs, b, cc, voids = [], 0, 0, []
    for (task, rep), arms in sorted(by_block.items()):
        on, off = arms.get("on"), arms.get("off")
        if on is None or off is None:
            continue
        if on.void or off.void:
            voids.append(f"{task}/rep{rep}")
            continue
        pairs.append((task, rep, off.passed, on.passed, off.answer, on.answer))
        if off.passed and not on.passed:
            b += 1
        elif on.passed and not off.passed:
            cc += 1

    p = exact_mcnemar_one_sided(b, cc)
    accept = cc > b and p <= ALPHA

    on_fail = sum(1 for _, _, _, onp, _, _ in pairs if not onp)
    off_fail = sum(1 for _, _, offp, _, _, _ in pairs if not offp)

    def med(arm: str, field: str):
        vals = [getattr(c, field) for c in cells if c.arm == arm and getattr(c, field) is not None]
        return round(statistics.median(vals), 1) if vals else None

    print(f"=== reliability {args.id}: {'ACCEPT' if accept else 'REJECT'}")
    print(f"    pairs {len(pairs)}   voids {len(voids)}")
    print(f"    failures  OFF {off_fail}/{len(pairs)}   ON {on_fail}/{len(pairs)}")
    print(f"    discordant  b(OFF pass, ON fail) = {b}   c(ON pass, OFF fail) = {cc}")
    print(f"    exact one-sided p = {p:.4f}   (need c > b and p <= {ALPHA})")
    print(f"    cost (not gating)  median tokens OFF {med('off','tokens')} ON {med('on','tokens')}"
          f"   median wall OFF {med('off','wall')}s ON {med('on','wall')}s")
    print()
    print(f"    {'block':16s} {'OFF':>6s} {'ON':>6s}   discordant")
    for task, rep, offp, onp, offa, ona in pairs:
        mark = ""
        if offp and not onp:
            mark = f"b  ON gave {ona!r}"
        elif onp and not offp:
            mark = f"c  OFF gave {offa!r}"
        print(f"    {task+'/rep'+str(rep):16s} {'pass' if offp else 'FAIL':>6s}"
              f" {'pass' if onp else 'FAIL':>6s}   {mark}")
    if voids:
        print(f"\n    VOID blocks excluded: {voids}")

    row = {
        "decision_id": args.id, "reading": "reliability", "spec": "§4.2b",
        "verdict": "ACCEPT" if accept else "REJECT",
        "pairs": len(pairs), "b": b, "c": cc, "p_one_sided": p, "alpha": ALPHA,
        "failures": {"off": off_fail, "on": on_fail},
        "voids": voids, "note": args.note,
        "blocks": [{"task": t, "rep": r, "off_pass": o, "on_pass": n,
                    "off_answer": oa, "on_answer": na} for t, r, o, n, oa, na in pairs],
    }
    out = pathlib.Path(args.root) / "reliability.json"
    out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.audit:
        with open(args.audit, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"\n    audit row appended to {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
