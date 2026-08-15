import json, os, collections
R=r"D:/PROJECTS/rlm-halo-framework/s2/results"
def load(fn):
    with open(os.path.join(R,fn),encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
# distance: 91/91 first-call tokens_cached==0 over ALL 777 calls
d=[x for x in load("distance.jsonl") if x.get("id_slot") is not None]
seen=set(); z=0; nz=[]
for x in d:
    s=x["id_slot"]
    if s in seen: continue
    seen.add(s); v=x.get("tokens_cached")
    (nz.append((s,v)) if v else None) or (z:=z+1) if not v else None
z=sum(1 for s in seen for x in [next(y for y in d if y["id_slot"]==s)] if not x.get("tokens_cached"))
print(f"distance: distinct slots={len(seen)}  first-call tokens_cached==0: {z}/{len(seen)}")
# the 8 non-call records
nc=[x for x in load("distance.jsonl") if x.get("id_slot") is None]
print("distance non-call records status:", collections.Counter(x.get("status") for x in nc), [x.get("cell_uid") for x in nc][:8])
# cache_instrument with correct slot key
ci=load("cache_instrument.jsonl")
pairs=collections.defaultdict(set)
for x in ci: pairs[(x.get("run_id"), x.get("returned_slot"))].add(x.get("doc_key"))
multi={k:v for k,v in pairs.items() if len(v)>1}
print(f"cache_instrument: (run_id,returned_slot) pairs={len(pairs)} with>1 doc_key={len(multi)}")
pairs2=collections.defaultdict(set)
for x in ci: pairs2[(x.get("run_id"), x.get("requested_slot"))].add(x.get("doc_key"))
print(f"cache_instrument: (run_id,requested_slot) pairs={len(pairs2)} with>1={sum(1 for v in pairs2.values() if len(v)>1)}")
# r14 pairs
r14=load("r14.jsonl")
p=collections.defaultdict(set)
for x in r14: p[(x.get("run_id"), x.get("id_slot"))].add(x.get("doc"))
print(f"r14: pairs={len(p)} multi={sum(1 for v in p.values() if len(v)>1)} calls={len(r14)}")
# occupancy argv presence
occ=load("occupancy.jsonl")
print("occupancy: records with argv:", sum(1 for x in occ if x.get("argv")), "/", len(occ))
print("r14: records with argv:", sum(1 for x in r14 if x.get("argv")), "/", len(r14))
print("sample occ argv:", str(occ[0].get("argv"))[:260])
