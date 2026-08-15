import json, os
from collections import defaultdict, Counter

def load(p):
    out=[]
    with open(p, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def slotdoc(path, slotkey, dockey, groupkey=None, label=None):
    recs=load(path)
    print(f"\n=== {label or path}  n={len(recs)}")
    # group -> slot -> set(docs)
    m=defaultdict(lambda: defaultdict(set))
    order=[]
    for r in recs:
        g = r.get(groupkey) if groupkey else "_"
        s = r.get(slotkey)
        d = r.get(dockey)
        m[g][s].add(d)
        order.append((g,s))
    pairs=sum(len(v) for v in m.values())
    multi=[(g,s,len(ds)) for g,v in m.items() for s,ds in v.items() if len(ds)>1]
    print(f"  distinct ({groupkey or '-'},{slotkey}) pairs = {pairs}")
    print(f"  pairs holding >1 distinct {dockey} = {len(multi)}")
    if multi[:8]: print("   examples:", multi[:8])
    allslots = sorted({s for g,v in m.items() for s in v if s is not None})
    nnone = sum(1 for g,v in m.items() for s in v if s is None)
    print(f"  pairs with slot=None: {nnone}")
    print(f"  distinct {slotkey} values = {len(allslots)} min={min(allslots)} max={max(allslots)}")
    return recs, m

# DISTANCE
recs,_ = slotdoc("s2/results/distance.jsonl","slot_id","chunk_sha256",None,"DISTANCE distance.jsonl")
st = Counter(r.get("status") for r in recs)
print("  status counter:", st)
adm = [r for r in recs if r.get("status")=="ok"]
print("  status==ok:", len(adm))
# monotonic allocation check
seq=[r["slot_id"] for r in recs if r.get("slot_id") is not None]
firstseen={}; revisits=0
for i,s in enumerate(seq):
    if s in firstseen and firstseen[s]!=i-1: revisits+=1
    firstseen[s]=i
print(f"  non-contiguous slot revisits in record order = {revisits}")
# tokens_cached==0 on first call per slot
firstcall={}
for r in recs:
    s=r.get("slot_id")
    if s is None: continue
    if s not in firstcall: firstcall[s]=r
zc=sum(1 for s,r in firstcall.items() if r.get("tokens_cached")==0)
print(f"  slots with tokens_cached==0 on FIRST call: {zc}/{len(firstcall)}")
# phases
print("  phases:", Counter(r.get("phase") for r in recs))
ph_slots=defaultdict(set)
for r in recs: ph_slots[r.get("phase")].add(r.get("slot_id"))
for p,ss in ph_slots.items():
    ss2={x for x in ss if x is not None}
    print(f"    phase {p}: slots {min(ss2)}..{max(ss2)} n={len(ss2)} (+None:{len(ss)-len(ss2)})")
# overlap between phases
phs=list(ph_slots)
for i in range(len(phs)):
    for j in range(i+1,len(phs)):
        ov=ph_slots[phs[i]]&ph_slots[phs[j]]
        if ov: print(f"    OVERLAP {phs[i]} & {phs[j]}: {sorted(ov)[:10]}")

# REFUSAL-AB / 640
for p,lab in [("s2/results/refusal-ab.jsonl","REFUSAL-AB"),("s2/results/refusal-ab-640.jsonl","REFUSAL-AB-640")]:
    recs,_=slotdoc(p,"slot_id","chunk_sha256",None,lab)
    firstcall={}
    for r in recs:
        s=r.get("slot_id")
        if s is None: continue
        if s not in firstcall: firstcall[s]=r
    zc=sum(1 for s,r in firstcall.items() if r.get("tokens_cached")==0)
    print(f"  slots with tokens_cached==0 on FIRST call: {zc}/{len(firstcall)}")
    seq=[r["slot_id"] for r in recs if r.get("slot_id") is not None]; firstseen={}; rev=0
    for i,s in enumerate(seq):
        if s in firstseen and firstseen[s]!=i-1: rev+=1
        firstseen[s]=i
    print(f"  non-contiguous revisits = {rev}")

# OCCUPANCY / R14 / CACHE-INSTRUMENT
slotdoc("s2/results/occupancy.jsonl","id_slot","doc","run_id","OCCUPANCY")
slotdoc("s2/results/r14.jsonl","id_slot","doc","run_id","R14")
slotdoc("s2/results/cache_instrument.jsonl","returned_slot","prompt_sha256","run_id","CACHE-INSTRUMENT (returned_slot x prompt_sha)")
slotdoc("s2/results/cache_instrument.jsonl","requested_slot","doc_key","run_id","CACHE-INSTRUMENT (requested_slot x doc_key)")

# SWEEP
recs,_=slotdoc("s2/results/sweep.jsonl","slot_id","chunk_sha256",None,"SWEEP")
print("  distinct chunk_sha256:", len({r["chunk_sha256"] for r in recs}))
print("  slot_id values:", Counter(r.get("slot_id") for r in recs))
print("  id_slot values:", Counter(r.get("id_slot") for r in recs))
recs,_=slotdoc("s2/results/sweep-run1-shared-server.jsonl","slot_id","chunk_sha256",None,"SWEEP-RUN1-SHARED")
print("  distinct chunk_sha256:", len({r["chunk_sha256"] for r in recs}))
