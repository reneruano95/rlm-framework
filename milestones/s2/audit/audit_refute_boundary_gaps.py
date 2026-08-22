import json, re, hashlib, datetime as dt, statistics
from pathlib import Path
ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

owner = {}
for p in sorted(ROOT.glob("fixtures*/**/*.chunk.txt")):
    t = p.read_text(encoding="utf-8")
    for u in UUID.findall(t):
        owner.setdefault(u.lower(), []).append(str(p.relative_to(ROOT)))

def load(f):
    return [json.loads(l) for l in (ROOT / "results" / f).open(encoding="utf-8") if l.strip()]

print("=== phase-cache answers: which fixture FILE owns each answer? ===")
for r in [x for x in load("distance.jsonl") if x.get("phase") == "cache"]:
    u = UUID.findall(r.get("raw_output") or "")
    print(f"  slot={r['slot_id']} {r['arm']:<11} doc={r['doc_index']} "
          f"cached={r['tokens_cached']:>5}/{r['tokens_in']} -> {[owner.get(x.lower()) for x in u]}")

print("\n=== boundary vs intra-cell wall-clock gaps (restart signature) ===")
for f in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
    rows = [r for r in load(f) if r.get("status") == "ok"]
    b, i, prev, prev_sha = [], [], None, None
    for r in rows:
        t = dt.datetime.fromisoformat(r["ts"])
        if prev is not None:
            g = (t - prev).total_seconds() - r["wall_s"]   # dead time before this call
            (b if r["chunk_sha256"] != prev_sha else i).append(round(g, 1))
        prev, prev_sha = t, r["chunk_sha256"]
    print(f"\n  {f}")
    print(f"    boundary dead-time gaps  n={len(b)} median={statistics.median(b) if b else '-'} -> {b}")
    print(f"    intra-cell dead-time     n={len(i)} median={statistics.median(i) if i else '-'} "
          f"min={min(i) if i else '-'} max={max(i) if i else '-'}")
