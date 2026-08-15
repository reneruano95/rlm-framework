import json, io, datetime as dt
from collections import Counter, defaultdict
out=io.open("s2/audit/_sweep_out.txt","w",encoding="utf-8")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")

recs=[json.loads(l) for l in open("s2/results/sweep.jsonl",encoding="utf-8") if l.strip()]
P("sweep n=",len(recs))
P("question_type:",Counter(r.get("question_type") for r in recs))
P("expected_kind:",Counter(r.get("expected_kind") for r in recs))
P("phase:",Counter(r.get("phase") for r in recs))
P("cell_id count:",len({r["cell_id"] for r in recs}))
P("status:",Counter(r.get("status") for r in recs))
P("size_target:",Counter(r.get("size_target") for r in recs))

# cell boundaries & timestamps
def ts(r):
    s=r["ts"].replace("Z","+00:00")
    return dt.datetime.fromisoformat(s)
P("\n-- cell order & inter-record gaps (sweep.jsonl) --")
seq=[(ts(r), r["cell_id"], r.get("phase"), r.get("size_target")) for r in recs]
gaps_boundary=[]; gaps_within=[]
for i in range(1,len(seq)):
    d=(seq[i][0]-seq[i-1][0]).total_seconds()
    if seq[i][1]!=seq[i-1][1]:
        gaps_boundary.append((seq[i-1][1],seq[i][1],round(d,1)))
    else:
        gaps_within.append(d)
for g in gaps_boundary: P("  BOUNDARY",g)
gw=sorted(gaps_within)
P("  within-cell gaps: n=%d median=%.1f min=%.1f max=%.1f"%(len(gw), gw[len(gw)//2], gw[0], gw[-1]))

recs1=[json.loads(l) for l in open("s2/results/sweep-run1-shared-server.jsonl",encoding="utf-8") if l.strip()]
P("\n-- run1 (KNOWN single process) --")
seq=[(ts(r), r["cell_id"]) for r in recs1]
gb=[];gwn=[]
for i in range(1,len(seq)):
    d=(seq[i][0]-seq[i-1][0]).total_seconds()
    if seq[i][1]!=seq[i-1][1]: gb.append((seq[i-1][1],seq[i][1],round(d,1)))
    else: gwn.append(d)
for g in gb: P("  BOUNDARY",g)
gwn=sorted(gwn)
P("  within-cell gaps: n=%d median=%.1f min=%.1f max=%.1f"%(len(gwn), gwn[len(gwn)//2], gwn[0], gwn[-1]))
out.close()
print(open("s2/audit/_sweep_out.txt",encoding="utf-8").read())
