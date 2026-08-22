"""Two tests the prior audit did not run.

(1) sweep.jsonl puts 13 different chunks on slot 0 of a 128-slot server. If the
    process really restarted per cell, the KV is empty at each cell boundary and
    tokens_cached MUST be 0 there; if one process served the file, the 311-token
    shared prefix would be resident and reused. tokens_cached is recorded, so
    the launcher fact is testable after all.

(2) distance --phase cache deliberately puts TWO DIFFERENT documents on ONE slot
    under the production flags. That is an in-corpus positive-opportunity control
    for leakage: if pinned-slot + cache_prompt=True leaks, it leaks here.
"""
import json, collections
from pathlib import Path
import hashlib, re
ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

chunks = {}
for p in sorted(ROOT.glob("fixtures*/**/*.chunk.txt")):
    t = p.read_text(encoding="utf-8")
    chunks[hashlib.sha256(t.encode()).hexdigest()] = (p, t)
uuid_owner = {}
for h, (p, t) in chunks.items():
    for u in UUID.findall(t):
        uuid_owner[u.lower()] = p.name

def load(f):
    return [json.loads(l) for l in (ROOT / "results" / f).open(encoding="utf-8") if l.strip()]

for f in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
    rows = [r for r in load(f) if r.get("status") == "ok"]
    print(f"\n===== {f} : tokens_cached at cell boundaries vs within cells =====")
    prev_sha = None
    b_zero = b_nz = i_zero = i_nz = 0
    for r in rows:
        sha = r["chunk_sha256"]
        boundary = sha != prev_sha
        tc = r.get("tokens_cached")
        if boundary:
            print(f"  BOUNDARY {r['ts']} {r['cell_id']:<14} slot={r['slot_id']} "
                  f"tokens_cached={tc}/{r['tokens_in']}")
            (b_zero if tc == 0 else b_nz).__class__  # noop
            if tc == 0: b_zero += 1
            else: b_nz += 1
        else:
            if tc == 0: i_zero += 1
            else: i_nz += 1
        prev_sha = sha
    print(f"  -> first-call-of-cell : cached==0 {b_zero}, cached>0 {b_nz}")
    print(f"  -> later calls in cell: cached==0 {i_zero}, cached>0 {i_nz}")

print("\n===== distance --phase cache : two DIFFERENT docs on ONE slot =====")
rows = [r for r in load("distance.jsonl") if r.get("phase") == "cache"]
for r in rows:
    raw = r.get("raw_output") or ""
    us = [u.lower() for u in UUID.findall(raw)]
    print(f"  slot={r['slot_id']} arm={r['arm']:<11} doc={r['doc_index']} "
          f"cached={r['tokens_cached']:>5}/{r['tokens_in']} "
          f"answer_owner={[uuid_owner.get(u,'UNKNOWN') for u in us]}  {raw[:44]!r}")
print("\n  (doc 0 and doc 1 are different documents on the SAME slot; a doc-0 answer"
      "\n   repeated on doc 1 would be a leak under the PRODUCTION flag combination.)")
