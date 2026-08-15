"""INDEPENDENT re-derivation of the never-reuse claim in the r13-truth report.

For each headline probe file, in FILE ORDER (= wall-clock order), track for
every slot_id the set of distinct documents (chunk_sha256, else
rendered_sha256) it has already held. A call is "on a dirty slot" if the slot
already held a DIFFERENT document earlier in the same run.

Also re-derives:
  * requested-slot vs served-slot mismatches
  * tokens_cached == 0 on every slot's first call
  * per-slot document counts INCLUDING the phase=cache priming records
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(r"D:/PROJECTS/rlm-halo-framework/s2")


def doc_key(r: dict):
    return r.get("chunk_sha256") or r.get("rendered_sha256") or r.get("chunk_id")


def audit(name: str, req_field: str, include_all_phases: bool = True):
    p = S2 / "results" / name
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    seen: dict = defaultdict(list)          # slot -> [doc,...] distinct in order
    first_call: dict = {}
    dirty = 0
    dirty_rows = []
    mismatch = 0
    mismatch_rows = []
    firstcall_cached_nonzero = 0
    n_first = 0
    phases = Counter()
    for i, r in enumerate(rows):
        phases[r.get("phase")] += 1
        s = r.get("slot_id")
        if s is None:
            continue
        req = r.get(req_field)
        if req is not None and req != s:
            mismatch += 1
            mismatch_rows.append((i, req, s))
        d = doc_key(r)
        if s not in first_call:
            first_call[s] = i
            n_first += 1
            tc = r.get("tokens_cached")
            if tc not in (0, None):
                firstcall_cached_nonzero += 1
        elif d is not None and d not in seen[s]:
            dirty += 1
            dirty_rows.append((i, s, r.get("phase"), r.get("cell_id"), len(seen[s])))
        if d is not None and d not in seen[s]:
            seen[s].append(d)
    multi = {s: len(v) for s, v in seen.items() if len(v) > 1}
    print("=" * 100)
    print(f"{name}: rows={len(rows)} slots={len(seen)} phases={dict(phases)}")
    print(f"  requested!=served: {mismatch}  {mismatch_rows[:5]}")
    print(f"  first-call-on-slot rows={n_first}, of which tokens_cached NOT 0/None: "
          f"{firstcall_cached_nonzero}")
    print(f"  slots that ever held >1 distinct document: {len(multi)} {dict(list(multi.items())[:10])}")
    print(f"  calls landing on an already-dirty slot: {dirty}")
    for d in dirty_rows[:10]:
        print(f"     row{d[0]} slot={d[1]} phase={d[2]} cell={d[3]} prior_docs={d[4]}")


if __name__ == "__main__":
    audit("distance.jsonl", "requested_slot")
    audit("refusal-ab.jsonl", "requested_slot")
    audit("refusal-ab-640.jsonl", "requested_slot")
    audit("sweep.jsonl", "id_slot")
    audit("sweep-run1-shared-server.jsonl", "id_slot")
    audit("occupancy.jsonl", "requested_slot")
    audit("r14.jsonl", "requested_slot")
