import json, io, re, os
from collections import Counter
out=io.open("milestones/s2/audit/_xcheck_out.txt","w",encoding="utf-8",errors="replace")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")

byid=re.compile(r"selected slot by id\s*\((\d+)\)")
lcp=re.compile(r"selected slot by LCP similarity")
lru=re.compile(r"selected slot by LRU")

def logslots(p):
    ids=[]; nl=nr=0
    for line in io.open(p,encoding="utf-8",errors="replace"):
        m=byid.search(line)
        if m: ids.append(int(m.group(1)))
        elif lcp.search(line): nl+=1
        elif lru.search(line): nr+=1
    return ids,nl,nr

# DISTANCE cross-check
ids,nl,nr=logslots("traces/logs/distance-leaf.err.log")
recs=[json.loads(l) for l in open("milestones/s2/results/distance.jsonl",encoding="utf-8") if l.strip()]
ok=[r for r in recs if r.get("status")=="ok"]
P("DISTANCE: log by-id selections=%d (LCP=%d LRU=%d); ok records=%d" % (len(ids),nl,nr,len(ok)))
P("  log slot multiset == record slot multiset ? %s" % (Counter(ids)==Counter(r["slot_id"] for r in ok)))
P("  log slot ORDER == record slot order ? %s" % (ids==[r["slot_id"] for r in ok]))
d1=Counter(ids); d2=Counter(r["slot_id"] for r in ok)
diff={k:(d1.get(k,0),d2.get(k,0)) for k in set(d1)|set(d2) if d1.get(k,0)!=d2.get(k,0)}
P("  disagreements (slot: log,record):", diff if diff else "NONE")
P("  requested_slot == slot_id on every ok record ? %s" %
  all(r.get("requested_slot")==r.get("slot_id") for r in ok))
P("  slot_ok False count:", sum(1 for r in ok if r.get("slot_ok") is False))

# REFUSAL cross-check
ids,nl,nr=logslots("traces/logs/leaf-server.err.log")
a=[json.loads(l) for l in open("milestones/s2/results/refusal-ab.jsonl",encoding="utf-8") if l.strip()]
b=[json.loads(l) for l in open("milestones/s2/results/refusal-ab-640.jsonl",encoding="utf-8") if l.strip()]
P("\nREFUSAL: log by-id=%d (LCP=%d LRU=%d); records a=%d b=%d total=%d ; UNACCOUNTED log calls=%d"
  % (len(ids),nl,nr,len(a),len(b),len(a)+len(b),len(ids)-len(a)-len(b)))
logset=set(ids); recset={r["slot_id"] for r in a}|{r["slot_id"] for r in b}
P("  distinct slots in log=%d ; in records=%d ; log-only slots=%s" %
  (len(logset),len(recset),sorted(logset-recset)))
P("  record-only slots (should be empty):", sorted(recset-logset))
P("  requested==slot_id on all: %s / %s" % (all(r.get("requested_slot")==r.get("slot_id") for r in a),
                                            all(r.get("requested_slot")==r.get("slot_id") for r in b)))

# OCCUPANCY / R14 server logs
P("\n-- OCCUPANCY / R14 server logs: selection kinds --")
tot_id=tot_lcp=tot_lru=0
for f in sorted(os.listdir("traces/logs")):
    if f.startswith(("occ-","r14-")):
        p="traces/logs/"+f
        ids,nl,nr=logslots(p)
        tot_id+=len(ids); tot_lcp+=nl; tot_lru+=nr
        if nl or nr: P(f"  !! {f}: by-id={len(ids)} LCP={nl} LRU={nr}")
P(f"  TOTAL across occ-*/r14-* logs: by-id={tot_id} LCP={tot_lcp} LRU={tot_lru}")

# cache_instrument logs
tot_id=tot_lcp=tot_lru=0
for f in sorted(os.listdir("traces/logs")):
    if f.startswith("cacheinst-"):
        ids,nl,nr=logslots("traces/logs/"+f)
        tot_id+=len(ids); tot_lcp+=nl; tot_lru+=nr
P(f"  TOTAL across cacheinst-* logs: by-id={tot_id} LCP={tot_lcp} LRU={tot_lru}")
out.close()
