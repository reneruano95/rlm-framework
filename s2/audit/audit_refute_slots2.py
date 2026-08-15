import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/s2/results"
def load(fn):
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

r=load("distance.jsonl")
noslot=[x for x in r if x.get("id_slot") is None]
print("distance: no-slot records =", len(noslot))
print("  their keys:", sorted(noslot[0].keys())[:20] if noslot else None)
print("  their 'error'/'admissible' fields:", [(x.get("error"), x.get("admissible"), x.get("label")) for x in noslot][:10])
nochunk=[x for x in r if "chunk_sha256" not in x]
print("distance: records missing chunk_sha256 =", len(nochunk))
ok=[x for x in r if x.get("id_slot") is not None and "chunk_sha256" in x]
pairs=collections.defaultdict(set)
order=[]
for x in ok:
    pairs[x["id_slot"]].add(x["chunk_sha256"])
    if not order or order[-1]!=x["id_slot"]: order.append(x["id_slot"])
print(f"distance: usable={len(ok)} slots={len(pairs)} range=[{min(pairs)},{max(pairs)}] slots_with_2plus_chunks={sum(1 for v in pairs.values() if len(v)>1)}")
seen=set(); ret=0
for s in order:
    if s in seen: ret+=1
    seen.add(s)
print("distance: slot-return events in record order =", ret)
print("distance: monotonic increasing distinct order?", order==sorted(order))
# first call cache
seen=set(); z=0; nz=[]
for x in ok:
    s=x["id_slot"]
    if s in seen: continue
    seen.add(s)
    v=x.get("tokens_cached", x.get("cache_n", x.get("cache_hit_fraction")))
    if not v: z+=1
    else: nz.append((s,v))
print(f"distance: first-call-on-slot cache-zero={z} nonzero={len(nz)} {nz[:5]}")
print("distance cache keys present:", [k for k in ok[0] if "cach" in k or "token" in k])

for fn in ["refusal-ab.jsonl","refusal-ab-640.jsonl"]:
    r=load(fn); ok=[x for x in r if x.get("id_slot") is not None and "chunk_sha256" in x]
    pairs=collections.defaultdict(set)
    for x in ok: pairs[x["id_slot"]].add(x["chunk_sha256"])
    print(f"{fn}: n={len(r)} usable={len(ok)} slots={len(pairs)} range=[{min(pairs)},{max(pairs)}] multi={sum(1 for v in pairs.values() if len(v)>1)}")

def slotdoc(fn, slotkey, dockey, runkey):
    r=load(fn)
    pairs=collections.defaultdict(set)
    miss=0
    for rec in r:
        s=rec.get(slotkey); d=rec.get(dockey)
        if s is None: miss+=1; continue
        pairs[(rec.get(runkey),s)].add(d)
    multi={k:v for k,v in pairs.items() if len(v)>1}
    print(f"{fn}: n={len(r)} noslot={miss} pairs={len(pairs)} multi_doc_pairs={len(multi)}")
slotdoc("occupancy.jsonl","id_slot","doc","run_id")
slotdoc("r14.jsonl","id_slot","doc","run_id")
slotdoc("cache_instrument.jsonl","id_slot","doc_key","run_id")
