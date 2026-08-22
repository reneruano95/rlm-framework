"""Slot discipline is claimed PER FILE. But two probe files can be served by the
SAME llama-server process; a slot virgin within refusal-ab-640 may already hold
another document from refusal-ab. Test the union, ordered by wall-clock ts.
"""
import json, collections, datetime as dt
from pathlib import Path
ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2/results")

FILES = ["refusal-ab.jsonl", "refusal-ab-640.jsonl", "distance.jsonl",
         "sweep.jsonl", "sweep-run1-shared-server.jsonl"]
rows = []
for f in FILES:
    for line in (ROOT / f).open(encoding="utf-8"):
        if not line.strip(): continue
        d = json.loads(line)
        if d.get("status") != "ok" or "ts" not in d: continue
        rows.append((dt.datetime.fromisoformat(d["ts"]), f, d))
rows.sort(key=lambda x: x[0])

print("=== wall-clock spans per file ===")
span = collections.defaultdict(lambda: [None, None, 0])
for t, f, d in rows:
    s = span[f]
    s[0] = t if s[0] is None else min(s[0], t); s[1] = t if s[1] is None else max(s[1], t); s[2] += 1
for f in FILES:
    if f in span:
        s = span[f]; print(f"  {f:<34} {s[0]} .. {s[1]}  n={s[2]}")

print("\n=== do file spans OVERLAP or abut (same process plausible)? ===")
ordered = sorted(((v[0], v[1], k) for k, v in span.items()))
for i in range(len(ordered) - 1):
    gap = (ordered[i+1][0] - ordered[i][1]).total_seconds()
    print(f"  {ordered[i][2]:<34} -> {ordered[i+1][2]:<34} gap {gap:>9.0f} s")

print("\n=== union view: does any slot_id hold >1 distinct chunk across files? ===")
per_slot = collections.defaultdict(list)
for t, f, d in rows:
    sha = d.get("chunk_sha256")
    if sha is None: continue
    per_slot[d["slot_id"]].append((t, f, sha, d.get("cell_uid") or d.get("cell_id")))
bad = 0
for slot, seq in sorted(per_slot.items()):
    shas = []
    for t, f, sha, cid in seq:
        if not shas or shas[-1][0] != sha: shas.append((sha, f, cid, t))
    if len({s[0] for s in shas}) > 1:
        bad += 1
        if bad <= 12:
            print(f"  slot {slot}: {len({s[0] for s in shas})} distinct chunks ->")
            for sha, f, cid, t in shas:
                print(f"      {t}  {f:<32} {cid}  {sha[:10]}")
print(f"\n  slots holding >1 distinct chunk in the UNION: {bad} of {len(per_slot)}")

print("\n=== distance --phase cache rows (deliberate cross-document slot reuse) ===")
for t, f, d in rows:
    if f == "distance.jsonl" and d.get("phase") == "cache":
        print(f"  {t} arm={d['arm']} slot={d['slot_id']} doc_index={d.get('doc_index')} "
              f"cached={d.get('tokens_cached')}/{d.get('tokens_in')} "
              f"raw={ (d.get('raw_output') or '')[:70]!r}")
