import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/milestones/s2/results"
def load(fn):
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

r=load("distance.jsonl")
calls=[x for x in r if x.get("id_slot") is not None]
print("distance call records:", len(calls))
nochunk=[x for x in calls if "chunk_sha256" not in x]
print("  of which missing chunk_sha256:", len(nochunk))
if nochunk:
    print("  sample keys:", sorted(nochunk[0].keys()))
    print("  sample:", {k:nochunk[0].get(k) for k in ("id_slot","error","label","arm","cell_id","status","phase")})
print("  distinct slots over ALL call records:", len(set(x["id_slot"] for x in calls)))
# use cell_uid as doc proxy where chunk missing
pairs=collections.defaultdict(set)
for x in calls:
    pairs[x["id_slot"]].add(x.get("chunk_sha256") or ("NOCHUNK:"+str(x.get("cell_uid"))))
print("  slots with >1 doc-proxy:", sum(1 for v in pairs.values() if len(v)>1))
tc=[x for x in calls if x.get("tokens_cached") is None]
print("  call records missing tokens_cached:", len(tc))

print()
ci=load("cache_instrument.jsonl")
print("cache_instrument keys:", sorted(ci[0].keys()))
print("sample extra:", json.dumps(ci[0].get("extra"))[:300])
slotkeys=[k for k in ci[0] if "slot" in k.lower()]
print("slot-ish keys:", slotkeys)
