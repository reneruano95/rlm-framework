"""Offline verification of the gates-ab survey plan's empirical claims.

Read-only. No GPU, no HTTP. Stdlib only.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / "s2" / "results" / "cache_instrument.jsonl"
RA = ROOT / "s2" / "results" / "refusal-ab-640.jsonl"


def rows(p: Path) -> list[dict]:
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ci = rows(CI)
    print(f"== cache_instrument.jsonl: {len(ci)} rows")
    print("  conditions:", dict(Counter(r.get("condition") for r in ci)))
    print("  head_tokens values:", dict(Counter(r.get("head_tokens") for r in ci)))
    print("  prefix_body_tokens:", dict(Counter(r.get("prefix_body_tokens") for r in ci)))
    print("  rows with 'ub' key:", sum(1 for r in ci if "ub" in r))
    print("  keys sample:", sorted(ci[0].keys()))

    # --- the reuse law, prev vs high-water, per condition -------------------
    print("\n== reuse law scoring on cache_instrument (prev vs high-water)")
    for cond in sorted({r["condition"] for r in ci}):
        sel = sorted([r for r in ci if r["condition"] == cond],
                     key=lambda x: x["ordinal"])
        ub = 512
        extra = (sel[0].get("extra") or "").split()
        if "-ub" in extra:
            ub = int(extra[extra.index("-ub") + 1])
        prev: dict[str, int] = {}
        hi: dict[str, int] = {}
        ok_prev = ok_hi = 0
        for r in sel:
            k = r["slot_key"]
            lcp = r["truth_lcp_prev_same_slot"]
            p_prev = law(prev.get(k), lcp, ub)
            p_hi = law(hi.get(k), r.get("truth_lcp_best_same_slot", lcp), ub)
            ok_prev += (p_prev == r["reported_cache_n"])
            ok_hi += (p_hi == r["reported_cache_n"])
            n = r["prompt_tokens_true"]
            prev[k] = n
            hi[k] = max(hi.get(k, 0), n)
        print(f"  {cond:16s} n={len(sel):4d} ub={ub:4d} prev={ok_prev}/{len(sel)} "
              f"hi={ok_hi}/{len(sel)}")

    # --- how many calls does any cache_instrument slot ever see? ------------
    calls_per_slot = Counter()
    for r in ci:
        calls_per_slot[(r["condition"], r["slot_key"])] += 1
    print("  calls-per-slot distribution:",
          dict(Counter(calls_per_slot.values())))

    # ---------------------------------------------------------------- ra ---
    ra = rows(RA)
    print(f"\n== refusal-ab-640.jsonl: {len(ra)} rows")
    print("  keys:", sorted(ra[0].keys()))
    print("  arms:", dict(Counter(r.get("arm") for r in ra)))
    print("  cells:", dict(Counter(r.get("cell") for r in ra)))
    print("  prefix_tokens overall:", dict(Counter(r.get("prefix_tokens") for r in ra)))
    a = [r for r in ra if r.get("arm") == "a-v1-plain"]
    print("  a-v1-plain rows:", len(a),
          "prefix_tokens:", dict(Counter(r.get("prefix_tokens") for r in a)))
    print("  slot_ok true:", sum(1 for r in ra if r.get("slot_ok")), "/", len(ra))
    print("  prefix_sha256:", dict(Counter(r.get("prefix_sha256") for r in ra)))
    print("  cold flag present:", sum(1 for r in ra if "cold" in r))

    # per-slot sequences
    seq = defaultdict(list)
    for r in ra:
        seq[(r.get("arm"), r.get("cell"), r.get("slot"), r.get("id_slot"))].append(r)
    print("  distinct (arm,cell,slot) groups:", len(seq))
    print("  group sizes:", dict(Counter(len(v) for v in seq.values())))


def law(n_resident, lcp, ub):
    if not n_resident or n_resident <= 0 or lcp <= 0:
        return 0
    if lcp >= n_resident:
        return n_resident - 4
    if lcp >= n_resident - 4:
        return lcp
    rb = n_resident - ub - 4
    return max(rb, 0) if lcp >= rb else 0


if __name__ == "__main__":
    main()
