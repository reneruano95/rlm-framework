"""Which reading of `N_resident` reproduces `cache_n` on 9-call slots?

Candidate models, scored on milestones/s2/results/refusal-ab-640.jsonl (28 slot groups x 9
calls) and cross-checked on milestones/s2/results/cache_instrument.jsonl.

Read-only, stdlib, no GPU.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RA = ROOT / "milestones" / "s2" / "results" / "refusal-ab-640.jsonl"
CI = ROOT / "milestones" / "s2" / "results" / "cache_instrument.jsonl"
UB = 512


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def law(n, lcp, ub=UB):
    if not n or n <= 0 or lcp <= 0:
        return 0
    if lcp >= n:
        return n - 4
    if lcp >= n - 4:
        return lcp
    rb = n - ub - 4
    return max(rb, 0) if lcp >= rb else 0


def main():
    ra = rows(RA)
    groups = defaultdict(list)
    for r in ra:
        groups[(r["arm"], r["cell_uid"], r["id_slot"])].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r["call_idx"])

    print(f"groups={len(groups)} sizes={dict(Counter(len(v) for v in groups.values()))}")
    g0 = groups[sorted(groups)[0]]
    print("\nsample group (arm/cell/slot):", sorted(groups)[0])
    for r in g0:
        print(f"  idx={r['call_idx']} cold={r['cold']} qtype={r['question_type']:10s} "
              f"tin={r['tokens_in']:5d} tout={r['tokens_out']:4d} "
              f"cached={r['tokens_cached']:5d} prefill={r['prefill_ms']:8.1f}")

    models = {
        "prev_prompt": lambda st, r: st["prev_in"],
        "prev_prompt_plus_out": lambda st, r: (st["prev_in"] + st["prev_out"]) if st["prev_in"] else None,
        "highwater_prompt": lambda st, r: st["hi_in"],
        "highwater_prompt_plus_out": lambda st, r: st["hi_in_out"],
    }
    score = {k: [0, 0] for k in models}
    deltas = defaultdict(list)
    for key, g in groups.items():
        st = {"prev_in": None, "prev_out": 0, "hi_in": 0, "hi_in_out": 0}
        for i, r in enumerate(g):
            if i > 0:
                # LCP is unknown here (no tokenization was recorded), but every
                # call on the slot shares [head][chunk] and differs only in the
                # trailing question, so the divergence is deep and the only
                # reachable branch is the rollback point. Assume lcp >= rollback
                # and test the identity `cache_n == N - ub - 4` directly.
                for name, f in models.items():
                    n = f(st, r)
                    pred = max((n or 0) - UB - 4, 0)
                    score[name][1] += 1
                    if pred == r["tokens_cached"]:
                        score[name][0] += 1
                    else:
                        deltas[name].append(r["tokens_cached"] - pred)
            st["prev_in"] = r["tokens_in"]
            st["prev_out"] = r["tokens_out"]
            st["hi_in"] = max(st["hi_in"], r["tokens_in"])
            st["hi_in_out"] = max(st["hi_in_out"], r["tokens_in"] + r["tokens_out"])

    print("\n== identity `cache_n == N_resident - 512 - 4` on the 8 re-queries per slot")
    for name in models:
        ok, n = score[name]
        d = deltas[name]
        rng = f"[{min(d)}, {max(d)}]" if d else "-"
        print(f"  {name:26s} {ok:3d}/{n}  miss-delta range {rng}")

    # cold denominator vs re-query numerator, per arm
    print("\n== (b2) prefill ratio per arm  (median re-query / median cold)")
    for arm in sorted({r["arm"] for r in ra}):
        sel = [r for r in ra if r["arm"] == arm]
        cold = [r["prefill_ms"] for r in sel if r["cold"]]
        warm = [r["prefill_ms"] for r in sel if not r["cold"]]
        print(f"  {arm:16s} cold n={len(cold)} med={statistics.median(cold):8.1f} | "
              f"warm n={len(warm)} med={statistics.median(warm):8.1f} | "
              f"ratio {statistics.median(warm)/statistics.median(cold):.3f}")

    # --- cross-check on cache_instrument: does prompt+out ever fit better? ---
    ci = rows(CI)
    print("\n== cache_instrument: same four models (per condition)")
    for cond in sorted({r["condition"] for r in ci}):
        sel = sorted([r for r in ci if r["condition"] == cond], key=lambda x: x["ordinal"])
        ub = 512
        extra = (sel[0].get("extra") or "").split()
        if "-ub" in extra:
            ub = int(extra[extra.index("-ub") + 1])
        st = defaultdict(lambda: {"prev_in": None, "prev_out": 0, "hi_in": 0, "hi_in_out": 0})
        res = {k: 0 for k in models}
        for r in sel:
            s = st[r["slot_key"]]
            lcp = r["truth_lcp_prev_same_slot"]
            for name, f in models.items():
                n = f(s, r)
                if models[name] is models["highwater_prompt"] or name.startswith("highwater"):
                    lcp_use = r.get("truth_lcp_best_same_slot", lcp)
                else:
                    lcp_use = lcp
                if law(n, lcp_use, ub) == r["reported_cache_n"]:
                    res[name] += 1
            s["prev_in"] = r["prompt_tokens_true"]
            s["prev_out"] = r["predicted_n"] or 0
            s["hi_in"] = max(s["hi_in"], r["prompt_tokens_true"])
            s["hi_in_out"] = max(s["hi_in_out"], r["prompt_tokens_true"] + (r["predicted_n"] or 0))
        print(f"  {cond:14s} n={len(sel):4d} ub={ub:4d} " +
              "  ".join(f"{k}={res[k]}" for k in models))

    # how many calls per slot in the requery-relevant cache_instrument cases?
    per = Counter()
    for r in ci:
        per[(r["condition"], r["slot_key"])] += 1
    threeplus = [k for k, v in per.items() if v >= 3]
    print("\n  cache_instrument slots with >=3 calls:", len(threeplus),
          "cases:", dict(Counter(
              r["case"] for r in ci
              if (r["condition"], r["slot_key"]) in set(threeplus))))


if __name__ == "__main__":
    main()
