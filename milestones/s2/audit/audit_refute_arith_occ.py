"""INDEPENDENT re-derivation of the occupancy/r14 hygiene numbers claimed by the
r13-truth auditor:
   'occupancy.jsonl 1,400/1,400 leak_detected:false, 0 mismatches, no
    (condition, slot) pair repeated; r14.jsonl 1,151 clean, 0 slots with two
    documents.'
The files carry id_slot / id_slot_returned / slot_mismatch / leak_detected /
doc, not slot_id, so the denominators must be re-derived from those.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(r"D:/PROJECTS/rlm-halo-framework/milestones/s2")


def audit(name: str):
    rows = [json.loads(l) for l in (S2 / "results" / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    st = Counter(r.get("status") for r in rows)
    ld = Counter(r.get("leak_detected") for r in rows)
    sm = Counter(r.get("slot_mismatch") for r in rows)
    mism_field = sum(1 for r in rows
                     if r.get("id_slot") is not None
                     and r.get("id_slot_returned") is not None
                     and r["id_slot"] != r["id_slot_returned"])
    # documents per (run_id, condition, served slot)
    docs = defaultdict(set)
    for r in rows:
        s = r.get("id_slot_returned", r.get("id_slot"))
        if s is None:
            continue
        d = r.get("doc")
        if d is None:
            continue
        docs[(r.get("run_id"), r.get("condition"), s)].add(json.dumps(d, sort_keys=True) if not isinstance(d, str) else d)
    multi = {k: len(v) for k, v in docs.items() if len(v) > 1}
    # documents per served slot ignoring run/condition
    docs2 = defaultdict(set)
    for r in rows:
        s = r.get("id_slot_returned", r.get("id_slot"))
        d = r.get("doc")
        if s is None or d is None:
            continue
        docs2[s].add(d if isinstance(d, str) else json.dumps(d, sort_keys=True))
    multi2 = {k: len(v) for k, v in docs2.items() if len(v) > 1}
    print("=" * 100)
    print(f"{name}: lines={len(rows)}")
    print(f"  status: {dict(st)}")
    print(f"  leak_detected: {dict(ld)}")
    print(f"  slot_mismatch field: {dict(sm)}")
    print(f"  id_slot != id_slot_returned (recomputed): {mism_field}")
    print(f"  (run_id,condition,slot) groups={len(docs)}; with >1 distinct doc: {len(multi)}")
    print(f"  served slots={len(docs2)}; slots that held >1 distinct doc: {len(multi2)} "
          f"{dict(list(multi2.items())[:8])}")
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_ans = sum(1 for r in rows if r.get("answer"))
    print(f"  rows with status=='ok': {n_ok}; rows with a non-empty answer: {n_ans}")
    print(f"  distinct argv: {len({r.get('argv') if isinstance(r.get('argv'), str) else json.dumps(r.get('argv')) for r in rows})}")


if __name__ == "__main__":
    audit("occupancy.jsonl")
    audit("r14.jsonl")
