import json, io
from collections import defaultdict, Counter
out=io.open("s2/audit/_argv_out.txt","w",encoding="utf-8")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")

for path,lab in [("s2/results/occupancy.jsonl","OCCUPANCY"),("s2/results/r14.jsonl","R14")]:
    recs=[json.loads(l) for l in open(path,encoding="utf-8") if l.strip()]
    P(f"\n===== {lab}  n={len(recs)}")
    byc=defaultdict(list)
    for r in recs: byc[r["condition"]].append(r)
    P(f"  distinct conditions: {len(byc)}")
    n_cram0=0; n_nocacheidle=0
    for c in sorted(byc):
        rs=byc[c]
        argvs={tuple(r.get("argv") or []) for r in rs}
        assert len(argvs)==1, (c, len(argvs))
        a=list(argvs)[0]
        has_cram = "--cache-ram" in a
        cramval = a[a.index("--cache-ram")+1] if has_cram else None
        nocidle = "--no-cache-idle-slots" in a
        npv = a[a.index("-np")+1] if "-np" in a else "?"
        ctxv= a[a.index("-c")+1] if "-c" in a else "?"
        ub  = a[a.index("-ub")+1] if "-ub" in a else "?"
        if has_cram and cramval=="0": n_cram0+=1
        if nocidle: n_nocacheidle+=1
        P(f"    {c:<24} n={len(rs):<5} -np={npv:<5} -c={ctxv:<7} -ub={ub:<5} --cache-ram={cramval} --no-cache-idle-slots={nocidle}")
    P(f"  conditions WITH --cache-ram 0     : {n_cram0}")
    P(f"  conditions WITH --no-cache-idle-slots: {n_nocacheidle}")
    P(f"  distinct -np values: {sorted({(r['argv'][r['argv'].index('-np')+1]) for r in recs if '-np' in (r.get('argv') or [])})}")
    P(f"  distinct -c values : {sorted({(r['argv'][r['argv'].index('-c')+1]) for r in recs if '-c' in (r.get('argv') or [])})}")
    # full argv sample
    P(f"  SAMPLE argv: {' '.join(recs[0]['argv'])}")

# CACHE-INSTRUMENT conditions (no argv field -> check 'extra')
recs=[json.loads(l) for l in open("s2/results/cache_instrument.jsonl",encoding="utf-8") if l.strip()]
P(f"\n===== CACHE-INSTRUMENT n={len(recs)}")
P("  conditions:", Counter(r["condition"] for r in recs))
P("  run_ids:", len({r["run_id"] for r in recs}))
P("  extra values:", Counter(r.get("extra") for r in recs))
P("  np values:", Counter(r.get("np") for r in recs))
P("  ctx values:", Counter(r.get("ctx") for r in recs))
P("  leak_detected true:", sum(1 for r in recs if r.get("leak_detected")))
P("  slot_ok false:", sum(1 for r in recs if r.get("slot_ok") is False))
P("  answer_correct not None:", sum(1 for r in recs if r.get("answer_correct") is not None))
P("  role counter:", Counter(r.get("role") for r in recs))
out.close()
print(open("s2/audit/_argv_out.txt",encoding="utf-8").read())
