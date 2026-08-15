"""(b2) prefill ratios by window, and the 38/38 / 42/42 re-query counts.

Read-only, stdlib, no GPU.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / "s2" / "results" / "cache_instrument.jsonl"


def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ci = rows(CI)
    print("== roles present:", dict(Counter(r["role"] for r in ci)))
    print("== cases present:", dict(Counter(r["case"] for r in ci)))

    # requery pairs: cold vs requery, matched by (condition, slot_key)
    print("\n== requery / cold prefill by condition and chunk size")
    for cond in sorted({r["condition"] for r in ci}):
        sel = [r for r in ci if r["condition"] == cond]
        cold = [r for r in sel if r["case"] == "requery" and r["role"] == "cold"]
        warm = [r for r in sel if r["case"] == "requery" and r["role"] == "requery"]
        if not warm:
            continue
        by_tk = defaultdict(lambda: ([], []))
        for r in cold:
            by_tk[r["chunk_tokens_target"]][0].append(r["prompt_ms"])
        for r in warm:
            by_tk[r["chunk_tokens_target"]][1].append(r["prompt_ms"])
        for tk in sorted(by_tk):
            c, w = by_tk[tk]
            if c and w:
                print(f"  {cond:14s} chunk~{tk:5d} cold n={len(c):3d} med={statistics.median(c):8.1f}"
                      f" | warm n={len(w):3d} med={statistics.median(w):8.1f}"
                      f" | ratio {statistics.median(w)/statistics.median(c):.3f}")

    # window-length view for cram0-len / default-len: what are their prompt sizes?
    print("\n== *-len conditions: chunk_tokens_target x case x role")
    for cond in ("cram0-len", "default-len"):
        sel = [r for r in ci if r["condition"] == cond]
        print(f"  {cond}: n={len(sel)}",
              dict(Counter((r["case"], r["role"], r["chunk_tokens_target"]) for r in sel)))

    # a2: divergence re-queries that reused anything, exactness
    print("\n== (a2) intra-window re-query identity, per condition")
    for cond in sorted({r["condition"] for r in ci}):
        sel = sorted([r for r in ci if r["condition"] == cond], key=lambda x: x["ordinal"])
        ub = 512
        extra = (sel[0].get("extra") or "").split()
        if "-ub" in extra:
            ub = int(extra[extra.index("-ub") + 1])
        prev = {}
        ok = tot = reused = 0
        for r in sel:
            k = r["slot_key"]
            n = prev.get(k)
            if r["case"] == "requery" and r["role"] == "requery" and n:
                tot += 1
                pred = max(n - ub - 4, 0)
                ok += (r["reported_cache_n"] == pred)
                reused += (r["reported_cache_n"] > 0)
            prev[k] = r["prompt_tokens_true"]
        if tot:
            print(f"  {cond:14s} requery calls={tot} identity_ok={ok} reused>0={reused}")

    # all same-slot divergences (any case) that reused anything
    print("\n== all same-slot second+ calls, divergence branch")
    tot = ok = reused = 0
    for cond in sorted({r["condition"] for r in ci}):
        sel = sorted([r for r in ci if r["condition"] == cond], key=lambda x: x["ordinal"])
        ub = 512
        extra = (sel[0].get("extra") or "").split()
        if "-ub" in extra:
            ub = int(extra[extra.index("-ub") + 1])
        prev = {}
        for r in sel:
            k = r["slot_key"]
            n = prev.get(k)
            if n:
                lcp = r["truth_lcp_prev_same_slot"]
                if lcp < n - 4:
                    tot += 1
                    pred = max(n - ub - 4, 0) if lcp >= n - ub - 4 else 0
                    ok += (r["reported_cache_n"] == pred)
                    reused += (r["reported_cache_n"] > 0)
            prev[k] = r["prompt_tokens_true"]
    print(f"  divergences={tot} identity_ok={ok} reused>0={reused}")


if __name__ == "__main__":
    main()
