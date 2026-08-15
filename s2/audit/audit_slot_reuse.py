"""LEAK-POSSIBLE, tested structurally rather than by reading prose.

For each result file, group the calls by the slot the server actually served
them on (`slot_id` / `id_slot_returned`), and ask: did any single slot ever hold
MORE THAN ONE distinct document within one server process?

That is the R13 precondition. A slot that has only ever seen one document cannot
leak another document's content into an answer, whatever the cache flags say.
A slot that has seen two CAN.

Document identity is taken from whatever the file records: chunk_sha256, doc,
doc_key, cell_id/cell_uid, prompt_sha256, rendered_sha256 (fallback).
Process identity is taken from run_id / condition where recorded; where it is
NOT recorded the file is treated as ONE process, which is the conservative
(leak-favouring) reading, and that is flagged.

Offline. stdlib only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results")

SLOT_KEYS = ("slot_id", "id_slot_returned", "returned_slot", "id_slot",
             "requested_slot", "id_slot_requested")
DOC_KEYS = ("chunk_sha256", "doc", "doc_key", "cell_uid", "cell_id",
            "prompt_sha256", "slot_key", "uid")
PROC_KEYS = ("run_id", "condition", "phase")


def first(rec, keys):
    for k in keys:
        if k in rec and rec[k] is not None:
            return k, rec[k]
    return None, None


def main() -> int:
    for p in sorted(RESULTS.glob("*.jsonl")):
        recs = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not recs:
            continue
        sk, _ = first(recs[0], SLOT_KEYS)
        dk, _ = first(recs[0], DOC_KEYS)
        pk, _ = first(recs[0], PROC_KEYS)
        print("=" * 78)
        print(f"{p.name}: n={len(recs)}  slot_key={sk}  doc_key={dk}  proc_key={pk}")
        if sk is None or dk is None:
            print("   -> CANNOT TEST (no slot and/or no document identity recorded)")
            continue
        # (proc, slot) -> set(docs)
        seen: dict[tuple, set] = defaultdict(set)
        order: dict[tuple, list] = defaultdict(list)
        for r in recs:
            slot = r.get(sk)
            doc = r.get(dk)
            proc = r.get(pk) if pk else "<single-process-assumed>"
            if slot is None or doc is None:
                continue
            key = (proc, slot)
            seen[key].add(doc)
            if not order[key] or order[key][-1] != doc:
                order[key].append(doc)
        multi = {k: v for k, v in seen.items() if len(v) > 1}
        tot_slots = len(seen)
        print(f"   distinct (proc,slot) pairs: {tot_slots}; "
              f"pairs that held >1 document: {len(multi)}")
        if multi:
            worst = sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
            for (proc, slot), docs in worst:
                trans = len(order[(proc, slot)]) - 1
                print(f"      proc={proc!r} slot={slot}: {len(docs)} distinct docs, "
                      f"{trans} document TRANSITIONS on that slot")
            print("   -> LEAK-POSSIBLE: YES (a slot served >1 document)")
        else:
            print("   -> slot never served a second document in the same process "
                  "grouping -> leak via slot-state carryover NOT possible")
        if pk is None:
            print("   NOTE: no process identity recorded; treated as one process "
                  "(conservative). Prose may claim fresh processes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
