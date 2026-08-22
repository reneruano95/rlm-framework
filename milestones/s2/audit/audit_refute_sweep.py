import json, os, datetime, collections
R=r"D:/PROJECTS/rlm-halo-framework/milestones/s2/results"
def load(fn):
    with open(os.path.join(R,fn),encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def gaps(fn):
    r=load(fn)
    tk=[k for k in r[0] if k in ("ts","t_start","timestamp")]
    key=tk[0] if tk else None
    print(f"\n{fn}: n={len(r)} timekey={key} sample={r[0].get(key)}")
    if not key: return
    def pt(s):
        try: return datetime.datetime.fromisoformat(str(s).replace("Z","+00:00"))
        except Exception: return None
    prev=None; prevcell=None
    for x in r:
        t=pt(x.get(key)); cell=x.get("cell_id")
        if prev and t:
            d=(t-prev).total_seconds()
            if cell!=prevcell: print(f"   CELL BOUNDARY {prevcell} -> {cell}: gap {d:.1f}s")
        prev=t or prev; prevcell=cell
gaps("sweep.jsonl")
gaps("sweep-run1-shared-server.jsonl")
