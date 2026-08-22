"""For EVERY ABSENT call in the arch ladder, how many prior bindings of the
ASKED entity had already been served ON THAT SLOT? If donor availability tracks
the 'collapse at 2,048', chunk size explains nothing."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_refute_log_parse import parse
STEMS = ["Prylfennwick","Orstlornholm","Quinfennsted","Selkdaleridge",
         "Hurnshawfield","Marnwickstead","Talverstrand","Bryndlecombe"]
def cell(size, trial):
    rng = random.Random(1000*size+trial)
    pool = rng.sample(STEMS,4)
    pres=[(f"{s} Trust","%08x-%04x-%04x-%04x-%012x"%(rng.getrandbits(32),rng.getrandbits(16),
          rng.getrandbits(16),rng.getrandbits(16),rng.getrandbits(48))) for s in pool[:3]]
    return pres, f"{pool[3]} Trust"

for jsonl, log in [("milestones/s2/results/arch_ladder_qwen-hybrid.jsonl","milestones/s2/logs/arch-qwen.err"),
                   ("milestones/s2/results/arch_ladder_gemma-fullattn.jsonl","milestones/s2/logs/arch-gemma.err")]:
    R=[json.loads(l) for l in Path(jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    L=parse(log); al=None
    for d in (0,1):
        for s0 in range(len(L)-len(R)+1):
            if all(L[s0+i].get("prompt_n")==R[i]["rendered_tokens"]+d for i in range(len(R))):
                al=(s0,d); break
        if al: break
    s0,_=al
    print(f"\n##### {Path(jsonl).name}   (aligned at log call {s0})")
    print(f"{'cell':>10} {'slot':>4} {'asked entity':<22} {'donors on slot':>14}  outcome")
    hist={}
    tab={}
    for i,r in enumerate(R):
        c=L[s0+i]; slot=c["slot"]; s,t=r["size"],r["trial"]
        pres,absent=cell(s,t)
        if r["qtype"]=="ABSENT":
            donors=[u for (cs,ct) in hist.get(slot,[]) for e,u in cell(cs,ct)[0] if e==absent]
            out = "REFUSED" if r["cls"]=="REFUSED" else f"LEAKED {r['answer'][:8]}"
            hit = (out!="REFUSED") and any(u in r["answer"] for u in donors)
            print(f"{s:>6}/t{t} {slot:>4} {absent:<22} {len(donors):>14}  {out}"
                  + ("  <- donor match" if hit else ""))
            tab.setdefault(s,[]).append((len(donors), r["cls"]!="REFUSED"))
        hist.setdefault(slot,[]).append((s,t))
    print("  --- by size ---")
    for s in sorted(tab):
        v=tab[s]
        print(f"   {s:>5}: n={len(v)}  non-refusals={sum(x[1] for x in v)}/{len(v)}  "
              f"donor counts on slot={[x[0] for x in v]}  "
              f"cells with >=1 donor={sum(x[0]>0 for x in v)}/{len(v)}")
