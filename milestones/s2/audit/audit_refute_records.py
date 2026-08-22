import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/milestones/s2/results"

def load(fn):
    recs=[]
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: recs.append(json.loads(line))
    return recs

def keys_report(fn):
    r=load(fn)
    print(f"--- {fn}: {len(r)} records; keys sample: {sorted(r[0].keys())}")
    return r

for fn in ["sweep.jsonl","distance.jsonl","refusal-ab.jsonl","refusal-ab-640.jsonl","occupancy.jsonl","r14.jsonl","cache_instrument.jsonl"]:
    r=load(fn)
    print(f"{fn:28s} n={len(r):5d} keys={sorted(r[0].keys())[:18]}")
