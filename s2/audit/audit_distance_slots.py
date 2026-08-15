"""The 6 distance.jsonl slots that served two documents: which, when, and did a
phase boundary (i.e. possibly a server restart) sit between them?

Also dumps the slot-allocation timeline so it is visible whether the runner
walked the pool monotonically (never-reuse) or wrapped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

P = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results\distance.jsonl")

recs = []
with P.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

per_slot = defaultdict(list)
for i, r in enumerate(recs):
    s = r.get("slot_id")
    if s is None:
        continue
    per_slot[s].append((i, r.get("phase"), r.get("chunk_sha256"),
                        r.get("arm"), r.get("size_target"),
                        r.get("tokens_cached"), r.get("cell_uid")))

print("slots that served >1 distinct chunk:")
for s in sorted(per_slot):
    docs = {t[2] for t in per_slot[s]}
    if len(docs) > 1:
        print(f"  slot {s}: {len(per_slot[s])} calls, {len(docs)} docs")
        prev = None
        for (i, ph, ch, arm, size, tc, uid) in per_slot[s]:
            mark = "  <== DOC CHANGE" if prev is not None and ch != prev else ""
            print(f"      rec#{i:4d} phase={ph!r} arm={arm!r} size={size} "
                  f"chunk={str(ch)[:10]} cached={tc} uid={uid}{mark}")
            prev = ch

print()
print("slot allocation order (first record index at which each slot appears):")
firsts = sorted(((min(t[0] for t in v), s) for s, v in per_slot.items()))
print("  " + ", ".join(f"{s}@{i}" for i, s in firsts))

print()
print("phases present:", sorted({r.get("phase") for r in recs}, key=str))
for ph in sorted({r.get("phase") for r in recs}, key=str):
    sl = sorted({r.get("slot_id") for r in recs if r.get("phase") == ph
                 and r.get("slot_id") is not None})
    n = sum(1 for r in recs if r.get("phase") == ph)
    print(f"  phase {ph!r}: n={n} slots={sl[:6]}...{sl[-3:] if len(sl) > 6 else ''} "
          f"({len(sl)} slots)")
