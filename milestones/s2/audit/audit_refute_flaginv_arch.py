import json, io, re
from collections import Counter, defaultdict
out=io.open("milestones/s2/audit/_arch2_out.txt","w",encoding="utf-8",errors="replace")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")

for p,lab in [("milestones/s2/results/arch_ladder_qwen-hybrid.jsonl","QWEN-HYBRID"),
              ("milestones/s2/results/arch_ladder_gemma-fullattn.jsonl","GEMMA-FULLATTN")]:
    recs=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
    P(f"\n===== {lab} n={len(recs)}")
    P("  sizes:", Counter(r["size"] for r in recs))
    P("  qtypes:", Counter(r["qtype"] for r in recs))
    P("  trials:", Counter(r["trial"] for r in recs))
    P("  distinct (size,trial) i.e. DOCUMENTS:", len({(r["size"],r["trial"]) for r in recs}))
    P("  distinct uid:", len({r["uid"] for r in recs}))
    P("  distinct chunk_tokens:", sorted({r["chunk_tokens"] for r in recs}))
    P("  cls counter:", Counter(r["cls"] for r in recs))
    P("  by (size,qtype,cls):", sorted(Counter((r["size"],r["qtype"],r["cls"]) for r in recs).items()))
    P("  rendered_tokens min/max:", min(r["rendered_tokens"] for r in recs), max(r["rendered_tokens"] for r in recs))

# LCP similarity distribution in arch logs
pat=re.compile(r"id\s+(\d+)\s*\|.*selected slot by (LCP similarity, f_sim_best = ([\d.]+).*f_keep = ([\d.]+)|LRU, t_last = (-?\d+))")
for p,lab in [("milestones/s2/logs/arch-qwen.err","arch-qwen"),("milestones/s2/logs/arch-gemma.err","arch-gemma"),("milestones/s2/logs/bos.err","bos")]:
    rows=[]
    for line in io.open(p,encoding="utf-8",errors="replace"):
        m=pat.search(line)
        if m:
            slot=int(m.group(1))
            if m.group(3) is not None: rows.append((slot,"LCP",float(m.group(3)),float(m.group(4))))
            else: rows.append((slot,"LRU",None,None))
    P(f"\n===== {lab}: {len(rows)} selections")
    P("  slot usage:", Counter(r[0] for r in rows))
    P("  kinds:", Counter(r[1] for r in rows))
    sims=[r[2] for r in rows if r[2] is not None]
    hi=[s for s in sims if s>=0.9]; lo=[s for s in sims if s<0.9]
    P(f"  LCP f_sim_best: n={len(sims)} >=0.90: {len(hi)}  <0.90: {len(lo)}  min={min(sims) if sims else None} max={max(sims) if sims else None}")
    P(f"  <0.90 values (new-document landings on a dirty slot): {sorted(lo)[:20]}")
    # consecutive same-slot transitions
    prev={}
    newdoc_on_dirty=0; total_after_first=0
    for slot,kind,sim,keep in rows:
        if slot in prev:
            total_after_first+=1
            if kind=="LCP" and sim is not None and sim<0.9: newdoc_on_dirty+=1
        prev[slot]=1
    P(f"  selections after a slot's first use: {total_after_first}; of those, low-similarity (different doc): {newdoc_on_dirty}")
out.close()
