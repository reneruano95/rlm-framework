import json, io, os
from collections import Counter
out=io.open("s2/audit/_newruns_out.txt","w",encoding="utf-8",errors="replace")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")
for f in sorted(os.listdir("s2/results")):
    if f.startswith("arch_ladder_"):
        p="s2/results/"+f
        rs=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
        P(f"\n== {f}  n={len(rs)}  mtime={os.path.getmtime(p)}")
        P("   keys:", sorted(rs[0].keys()))
        P("   by (size,qtype,cls):", sorted(Counter((r.get("size"),r.get("qtype"),r.get("cls")) for r in rs).items()))
        if "slot" in rs[0]:
            P("   slots requested:", Counter(r.get("slot") for r in rs))
            P("   slots served:", Counter(r.get("slot_served") for r in rs))
out.close()
