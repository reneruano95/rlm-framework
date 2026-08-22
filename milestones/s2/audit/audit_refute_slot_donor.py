"""Align each arch_ladder JSONL row to its server-log call, then ask:
is the leaked UUID the most recent binding of the ASKED entity ON THAT SLOT?

Global-recency and per-slot-recency make DIFFERENT predictions for qwen 2048/t1.
"""
import json, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_refute_log_parse import parse

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]

def cell(size, trial):
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust", "%08x-%04x-%04x-%04x-%012x" % (
        rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(16),
        rng.getrandbits(16), rng.getrandbits(48))) for s in pool[:3]]
    return present, f"{pool[3]} Trust"

def rows(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

for jsonl, log, off in [("milestones/s2/results/arch_ladder_qwen-hybrid.jsonl", "milestones/s2/logs/arch-qwen.err", None),
                        ("milestones/s2/results/arch_ladder_gemma-fullattn.jsonl", "milestones/s2/logs/arch-gemma.err", None)]:
    R = rows(jsonl); L = parse(log)
    # align: find the tail window of L whose prompt_n matches rendered_tokens (allow +0 or +1 BOS)
    aligned = None
    for delta in (0, 1):
        for start in range(len(L) - len(R) + 1):
            if all(L[start+i].get("prompt_n") == R[i]["rendered_tokens"] + delta for i in range(len(R))):
                aligned = (start, delta); break
        if aligned: break
    print(f"\n##### {jsonl}")
    print(f"log completions={len(L)}  jsonl rows={len(R)}  aligned_at={aligned}")
    if not aligned:
        print("  NO ALIGNMENT -> cannot attribute slots"); continue
    start, delta = aligned
    hist = {}          # slot -> list of (call_idx, size, trial)
    verdicts = []
    for i, r in enumerate(R):
        c = L[start+i]
        slot = c["slot"]; s, t = r["size"], r["trial"]
        pres, absent = cell(s, t)
        own = {u for _, u in pres}
        if r["qtype"] == "ABSENT" and r["cls"] != "REFUSED":
            ans = r["answer"].strip()
            # every binding of the asked entity ever served ON THIS SLOT, earlier
            same_slot = [(ci, cs, ct) for (ci, cs, ct) in hist.get(slot, []) if ci < start + i]
            on_slot = [(ci, cs, ct, u) for (ci, cs, ct) in same_slot
                       for e, u in cell(cs, ct)[0] if e == absent]
            # every binding anywhere earlier in the run (global)
            glob = [(ci, cs, ct, u) for sl, v in hist.items() for (ci, cs, ct) in v if ci < start + i
                    for e, u in cell(cs, ct)[0] if e == absent]
            on_slot.sort(); glob.sort()
            pred_slot = on_slot[-1] if on_slot else None
            pred_glob = glob[-1] if glob else None
            hit_slot = bool(pred_slot and pred_slot[3] in ans)
            hit_glob = bool(pred_glob and pred_glob[3] in ans)
            print(f"\n  {s}/t{t} ABSENT  slot={slot} call#{start+i}  asked '{absent}'")
            print(f"    answer                     : {ans}")
            print(f"    in the document that was sent: {any(u in ans for u in own)}")
            print(f"    ALL earlier bindings on slot {slot}: {[(ci,cs,ct,u[:8]) for ci,cs,ct,u in on_slot]}")
            print(f"    ALL earlier bindings anywhere    : {[(ci,cs,ct,u[:8]) for ci,cs,ct,u in glob]}")
            print(f"    per-SLOT most-recent predicts {pred_slot[1] if pred_slot else None}/t{pred_slot[2] if pred_slot else None} -> {'HIT' if hit_slot else 'MISS'}")
            print(f"    GLOBAL   most-recent predicts {pred_glob[1] if pred_glob else None}/t{pred_glob[2] if pred_glob else None} -> {'HIT' if hit_glob else 'MISS'}")
            verdicts.append((hit_slot, hit_glob))
        hist.setdefault(slot, []).append((start + i, s, t))
    if verdicts:
        print(f"\n  SUMMARY {jsonl}: per-slot-recency {sum(v[0] for v in verdicts)}/{len(verdicts)}"
              f"   global-recency {sum(v[1] for v in verdicts)}/{len(verdicts)}")
