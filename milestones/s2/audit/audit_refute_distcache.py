import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/milestones/s2/results"
def load(fn):
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
r=load("distance.jsonl")
calls=[x for x in r if x.get("id_slot") is not None]
print("phases:", collections.Counter(x.get("phase") for x in calls))
print("arms:", collections.Counter(x.get("arm") for x in calls))
cache=[x for x in calls if x.get("phase")=="cache"]
print("\ncache-phase records:", len(cache))
for x in cache:
    print("  slot",x["id_slot"],"cell_uid",x.get("cell_uid"),"doc_index",x.get("doc_index"),
          "rendered_sha",str(x.get("rendered_sha256"))[:12],"tokens_cached",x.get("tokens_cached"),
          "tokens_in",x.get("tokens_in"),"chunk_sha",str(x.get("chunk_sha256"))[:12])
# which slots do cache-phase records share with non-cache records
cslots=set(x["id_slot"] for x in cache)
print("\ncache-phase slots:", sorted(cslots))
for s in sorted(cslots):
    recs=[x for x in calls if x["id_slot"]==s]
    shas=set(str(x.get("rendered_sha256"))[:12] for x in recs)
    chunks=set(str(x.get("chunk_sha256"))[:12] for x in recs)
    print(f"  slot {s}: {len(recs)} calls, phases={sorted(set(x.get('phase') for x in recs))}, distinct rendered_sha={len(shas)}, distinct chunk_sha={chunks}")
