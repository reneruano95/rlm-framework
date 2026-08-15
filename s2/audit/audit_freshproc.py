"""Two strict follow-ups to audit_slot_reuse.py.

(1) STRICT grouping: ignore `phase`, treat each file as ONE server process, and
    re-ask whether any slot served more than one document. This is the reading
    that holds if the prose's "fresh process per cell" claim is wrong.

(2) TEST the fresh-process claim from telemetry the runner recorded anyway:
    on a genuinely fresh llama-server process the FIRST call must report
    tokens_cached == 0 (nothing in KV, nothing in the host prompt cache). Print,
    per (phase, size_target) cell in call order, the tokens_cached of the first
    call and of every call, so a restart boundary is visible or absent.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results")
TARGETS = ("sweep.jsonl", "sweep-run1-shared-server.jsonl", "distance.jsonl",
           "refusal-ab.jsonl", "refusal-ab-640.jsonl", "diag.jsonl")


def load(p: Path):
    out = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


print("### (1) STRICT: one process per FILE, slot -> distinct documents")
for name in TARGETS:
    p = RESULTS / name
    if not p.exists():
        continue
    recs = load(p)
    per_slot = defaultdict(set)
    for r in recs:
        s = r.get("slot_id", r.get("id_slot"))
        d = r.get("chunk_sha256")
        if s is not None and d is not None:
            per_slot[s].add(d)
        elif s is not None:
            per_slot[s].add(r.get("cell_uid") or r.get("cell_id"))
    multi = {k: v for k, v in per_slot.items() if len(v) > 1}
    print(f"  {name}: slots={len(per_slot)}  slots holding >1 doc={len(multi)}"
          + (f"  e.g. slot {sorted(multi)[0]} held {len(multi[sorted(multi)[0]])} docs"
             if multi else ""))

print()
print("### (2) fresh-process evidence: tokens_cached in call order, per cell")
for name in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
    p = RESULTS / name
    if not p.exists():
        continue
    recs = load(p)
    print(f"--- {name}")
    cells = defaultdict(list)
    for r in recs:
        cells[(r.get("phase"), r.get("size_target"))].append(r)
    for key in sorted(cells, key=lambda k: (str(k[0]), k[1] or 0)):
        rs = cells[key]
        tc = [r.get("tokens_cached") for r in rs]
        chunks = []
        for r in rs:
            c = r.get("chunk_sha256")
            if not chunks or chunks[-1] != c:
                chunks.append(c)
        print(f"   phase={key[0]!r} size={key[1]} n={len(rs)}  "
              f"distinct chunks in cell={len(set(chunks))}")
        print(f"        tokens_cached in order: {tc}")

print()
print("### (3) distance/refusal: was any slot's FIRST call warm? "
      "(nonzero tokens_cached on a slot's first appearance = state it did not "
      "create, i.e. a host-cache restore or a reused slot)")
for name in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl"):
    p = RESULTS / name
    if not p.exists():
        continue
    recs = load(p)
    firstseen = {}
    warm_first = []
    for r in recs:
        s = r.get("slot_id")
        if s is None:
            continue
        if s not in firstseen:
            firstseen[s] = r
            if (r.get("tokens_cached") or 0) > 0:
                warm_first.append((s, r.get("tokens_cached"),
                                   r.get("chunk_sha256", "")[:10]))
    print(f"  {name}: slots seen={len(firstseen)}  "
          f"slots whose FIRST call was already warm={len(warm_first)}")
    for w in warm_first[:10]:
        print(f"      slot={w[0]} tokens_cached={w[1]} chunk={w[2]}")
