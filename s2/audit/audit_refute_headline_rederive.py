import json, datetime as dt, statistics, collections
from pathlib import Path
ROOT = Path("D:/PROJECTS/rlm-halo-framework/s2")
def load(f):
    return [json.loads(l) for l in (ROOT / "results" / f).open(encoding="utf-8") if l.strip()]

print("=== gap convention check (subtract wall_s of the PREVIOUS call instead) ===")
for f in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
    rows = [r for r in load(f) if r.get("status") == "ok"]
    b, i = [], []
    for k in range(1, len(rows)):
        g = (dt.datetime.fromisoformat(rows[k]["ts"]) - dt.datetime.fromisoformat(rows[k-1]["ts"])).total_seconds() - rows[k-1]["wall_s"]
        (b if rows[k]["chunk_sha256"] != rows[k-1]["chunk_sha256"] else i).append(round(g, 1))
    print(f"  {f:<34} boundary median={statistics.median(b):>6} {b}")
    print(f"  {'':<34} intra    median={statistics.median(i):>6} min={min(i)} max={max(i)}")

print("\n=== distance REPLICATION arm: false positives on ABSENT, by size ===")
rows = [r for r in load("distance.jsonl")
        if r.get("phase") == "replicate" and r.get("status") == "ok"]
t = collections.Counter()
for r in rows:
    if r["question_type"] != "absent": continue
    t[(r["size_target"], r["label"])] += 1
for size in sorted({s for s, _ in t}):
    lab = {l: c for (s, l), c in t.items() if s == size}
    n = sum(lab.values())
    fp = sum(c for l, c in lab.items() if l not in ("CORRECT_REFUSAL", "REFUSAL", "MISS"))
    print(f"  size {size}: n={n}  labels={lab}")

print("\n=== distance GRID arm A-shipped: ABSENT labels by size ===")
rows = [r for r in load("distance.jsonl")
        if r.get("phase") == "grid" and r.get("arm") == "A-shipped" and r.get("status") == "ok"]
t = collections.Counter()
for r in rows:
    if r["question_type"] != "absent": continue
    t[(r["size_target"], r["label"])] += 1
for size in sorted({s for s, _ in t}):
    print(f"  size {size}: {{" + ", ".join(f"{l}: {c}" for (s, l), c in sorted(t.items()) if s == size) + "}")

print("\n=== all distinct labels present in distance.jsonl ===")
print(sorted({r.get("label") for r in load("distance.jsonl")}, key=str))
