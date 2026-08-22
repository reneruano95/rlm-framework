"""Slot/flag hygiene of the probes whose findings ARCHITECTURE.md cites:
distance.jsonl (the ~1,000-token horizon), refusal-ab*.jsonl, sweep.jsonl,
cache_instrument.jsonl, occupancy.jsonl, r14.jsonl.

For each: recorded argv, slot policy actually observed (was any slot given two
different documents?), the run's own leak verdicts, and slot mismatches.

Offline only.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

RES = Path(r"D:/PROJECTS/rlm-halo-framework/milestones/s2/results")

FILES = ["distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
         "sweep.jsonl", "sweep-run1-shared-server.jsonl",
         "cache_instrument.jsonl", "occupancy.jsonl", "r14.jsonl"]

DOC_KEYS = ("cell_id", "cell", "doc", "chunk_sha256", "prompt_sha256", "rendered_sha256")
SLOT_REQ = ("requested_slot", "id_slot_requested", "id_slot", "slot")
SLOT_GOT = ("slot_id", "id_slot_returned")


def first(r: dict, keys) -> object:
    for k in keys:
        if k in r and r[k] is not None:
            return r[k]
    return None


for name in FILES:
    p = RES / name
    if not p.exists():
        print(f"MISSING {name}")
        continue
    n = 0
    argvs: Counter = Counter()
    leak = Counter()
    mism = 0
    docs_per_slot: dict[object, set] = defaultdict(set)
    slot_field_used = None
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            a = r.get("argv")
            if isinstance(a, list):
                argvs[" ".join(a)] += 1
            elif isinstance(r.get("server_flags"), (list, str)):
                argvs[str(r["server_flags"])] += 1
            if "leak_detected" in r:
                leak[r["leak_detected"]] += 1
            req, got = first(r, SLOT_REQ), first(r, SLOT_GOT)
            if req is not None and got is not None and req != got:
                mism += 1
            if req is not None:
                slot_field_used = slot_field_used or [k for k in SLOT_REQ if k in r][0]
                d = first(r, DOC_KEYS)
                if d is not None:
                    docs_per_slot[req].add(d)
    multi = {s: len(v) for s, v in docs_per_slot.items() if len(v) > 1}
    print("=" * 92)
    print(f"{name}: n={n}")
    print(f"  leak_detected tally: {dict(leak) or 'NO leak field recorded'}")
    print(f"  slot field: {slot_field_used}; slots used: {len(docs_per_slot)}; "
          f"slot/answer mismatches: {mism}")
    print(f"  slots that saw MORE THAN ONE document: {len(multi)}"
          + (f"  -> {dict(list(multi.items())[:8])}" if multi else "  (virgin-slot rule held)"))
    if argvs:
        print(f"  distinct recorded argvs: {len(argvs)}")
        for a, c in argvs.most_common(8):
            print(f"     x{c:<5d} {a[-190:]}")
    else:
        print("  NO argv RECORDED IN THIS FILE -- server flags are not auditable from it")
