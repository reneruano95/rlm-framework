"""OFFLINE: does the server's own prefill TIME corroborate cache_n = 0?

`cache_n` is a counter the server reports; if the host prompt cache had silently
restored state onto a virgin slot, the honest tell would be a prefill that is
too FAST for the number of tokens the server claims it had to process.
s2/CACHE-INSTRUMENT.md measured this build's pooled cold prefill rate at
961 tok/s (786-802 t/s at 320 chunk tokens, 977-989 t/s at ~1,900).

So: for every COLD call in distance.jsonl, compare prefill_ms against
(tokens_in - cache_n) / rate. A cold call that reports cache_n = 0 and prefills
at 5x the cold rate is a silent restore. One that lands on the cold line is not.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S2 = REPO / "s2"

recs = [json.loads(l) for l in (S2 / "results" / "distance.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

q = [r for r in recs if r.get("status") == "ok" and r.get("phase") != "cache"]

print("=== COLD calls (first call on a virgin slot), by arm and size ===")
print(f"{'arm':12s} {'size':>5s} {'n':>3s} {'tok_in':>7s} {'cache_n':>7s} "
      f"{'prefill_ms':>10s} {'implied t/s':>11s}")
groups = defaultdict(list)
for r in q:
    if r.get("cold"):
        groups[(r["arm"], r["size_target"])].append(r)
for k in sorted(groups):
    rows = groups[k]
    ti = statistics.median(r["tokens_in"] for r in rows)
    cn = statistics.median(r["tokens_cached"] for r in rows)
    pm = statistics.median(r["prefill_ms"] for r in rows)
    rate = (ti - cn) / (pm / 1000)
    print(f"{k[0]:12s} {k[1]:5d} {len(rows):3d} {ti:7.0f} {cn:7.0f} "
          f"{pm:10.1f} {rate:11.0f}")

print("\n=== WARM calls (re-query of the SAME document on that cell's slot) ===")
print(f"{'arm':12s} {'size':>5s} {'n':>3s} {'tok_in':>7s} {'cache_n':>7s} "
      f"{'prefill_ms':>10s} {'implied t/s':>11s}")
groups = defaultdict(list)
for r in q:
    if r.get("cold") is False:
        groups[(r["arm"], r["size_target"])].append(r)
for k in sorted(groups):
    rows = groups[k]
    ti = statistics.median(r["tokens_in"] for r in rows)
    cn = statistics.median(r["tokens_cached"] for r in rows)
    pm = statistics.median(r["prefill_ms"] for r in rows)
    rate = (ti - cn) / (pm / 1000)
    print(f"{k[0]:12s} {k[1]:5d} {len(rows):3d} {ti:7.0f} {cn:7.0f} "
          f"{pm:10.1f} {rate:11.0f}")

print("\n=== Any COLD call whose implied prefill rate exceeds 1,600 t/s "
      "(i.e. too fast to be a real cold prefill) ===")
fast = []
for r in q:
    if not r.get("cold"):
        continue
    todo = r["tokens_in"] - r["tokens_cached"]
    if r["prefill_ms"] <= 0:
        continue
    rate = todo / (r["prefill_ms"] / 1000)
    if rate > 1600:
        fast.append((rate, r))
print(f"count: {len(fast)}")
for rate, r in sorted(fast, reverse=True)[:15]:
    print(f"  {r['arm']} {r['cell_uid']} slot {r['requested_slot']} "
          f"tok_in={r['tokens_in']} cache_n={r['tokens_cached']} "
          f"prefill={r['prefill_ms']}ms -> {rate:.0f} t/s")

print("\n=== REPLICATION phase: 640 vs 1024 cold prefill, the two cells that "
      "carry the headline ===")
for size in (640, 1024):
    rows = [r for r in q if r.get("phase") == "replicate"
            and r["size_target"] == size and r.get("cold")]
    if rows:
        print(f"  size {size}: n={len(rows)} "
              f"median tok_in={statistics.median(r['tokens_in'] for r in rows):.0f} "
              f"cache_n set={sorted({r['tokens_cached'] for r in rows})} "
              f"median prefill={statistics.median(r['prefill_ms'] for r in rows):.0f}ms")
