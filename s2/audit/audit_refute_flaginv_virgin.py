import json, io
from collections import Counter, defaultdict
out=io.open("s2/audit/_virgin_out.txt","w",encoding="utf-8",errors="replace")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")
recs=[json.loads(l) for l in open("s2/results/cache_instrument.jsonl",encoding="utf-8") if l.strip()]
sel=[r for r in recs if r.get("role") in ("virgin-slot-repeat","virgin-slot-lagged")]
P("virgin-slot roles total:", len(sel))
P("by (condition, role):", Counter((r["condition"], r["role"]) for r in sel))
P("by (case, condition, role):", Counter((r.get("case"), r["condition"], r["role"]) for r in sel))
nz=[r for r in sel if (r.get("reported_cache_n") or 0) != 0]
P("virgin-slot calls with reported_cache_n != 0:", len(nz))
P("reported_cache_n values:", Counter(r.get("reported_cache_n") for r in sel))
P("truth_lcp_best_any_slot values:", Counter(r.get("truth_lcp_best_any_slot") for r in sel))
# exclude smoke
sel2=[r for r in sel if r["condition"]!="smoke"]
P("\nexcluding condition 'smoke': n =", len(sel2), Counter((r["condition"],r["role"]) for r in sel2))
P("  of those, cache_n != 0:", sum(1 for r in sel2 if (r.get("reported_cache_n") or 0)!=0))
# the 'elsewhere'/'seen-elsewhere' rows for contrast
oth=[r for r in recs if r.get("role") in ("elsewhere","seen-elsewhere")]
P("\nelsewhere/seen-elsewhere n =", len(oth))
P("  by (condition,role):", Counter((r["condition"],r["role"]) for r in oth))
P("  reported_cache_n:", Counter(r.get("reported_cache_n") for r in oth))
out.close()
