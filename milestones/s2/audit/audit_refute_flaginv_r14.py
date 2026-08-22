import json
from collections import Counter, defaultdict
recs=[json.loads(l) for l in open("milestones/s2/results/r14.jsonl",encoding="utf-8") if l.strip()]
print("r14 records:", len(recs))
pairs=Counter((r["run_id"], r["id_slot"]) for r in recs)
print("distinct (run_id,id_slot) pairs:", len(pairs))
print("pairs used more than once:", sum(1 for v in pairs.values() if v>1))
print("top repeated:", pairs.most_common(3))
# id_slot_returned variant
p2=Counter((r["run_id"], r.get("id_slot_returned")) for r in recs)
print("distinct (run_id,id_slot_returned) pairs:", len(p2), " repeated:", sum(1 for v in p2.values() if v>1))
print("  most common:", p2.most_common(3))
mm=sum(1 for r in recs if r.get("slot_mismatch"))
print("slot_mismatch true:", mm)
# docs per pair using id_slot_returned
m=defaultdict(set)
for r in recs: m[(r["run_id"], r.get("id_slot_returned"))].add(r.get("doc"))
print("returned-slot pairs holding >1 doc:", sum(1 for v in m.values() if len(v)>1))
print("conditions:", len(set(r["condition"] for r in recs)), sorted(set(r["condition"] for r in recs)))
print("run_ids:", len(set(r["run_id"] for r in recs)))
# leak flags
print("leak_detected true:", sum(1 for r in recs if r.get("leak_detected")))

# OCCUPANCY same
recs=[json.loads(l) for l in open("milestones/s2/results/occupancy.jsonl",encoding="utf-8") if l.strip()]
print("\noccupancy records:", len(recs))
pairs=Counter((r["run_id"], r["id_slot"]) for r in recs)
print("distinct (run_id,id_slot) pairs:", len(pairs), " repeated:", sum(1 for v in pairs.values() if v>1))
print("conditions:", len(set(r["condition"] for r in recs)))
print("leak_detected true:", sum(1 for r in recs if r.get("leak_detected")))
print("slot_mismatch true:", sum(1 for r in recs if r.get("slot_mismatch")))
m=defaultdict(set)
for r in recs: m[(r["run_id"], r.get("id_slot_returned"))].add(r.get("doc"))
print("returned-slot pairs holding >1 doc:", sum(1 for v in m.values() if len(v)>1))

# leak arms
import os
tot=0
for f in ["sweep-run1-shared-server","leak-nocacheidle","leak-cram0","leak-nocram","leak-ctxcp0","leak-slotiso","leak-erase"]:
    p=f"milestones/s2/results/{f}.jsonl"
    rs=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
    docs=len({r.get("chunk_sha256") for r in rs})
    slots=Counter(r.get("slot_id") for r in rs)
    tot+=len(rs)
    print(f"  {f}: n={len(rs)} distinct_chunks={docs} slots={dict(slots)}")
print("leak-arms TOTAL calls:", tot)

# R13 files
tot=0
for f in ["r13_replay_erase","r13_replay_gemma_fullattn","r13_replay_hybrid","r13_replay_paired","r13_twoprompt_matrix","r13_twoprompt_sizesweep"]:
    p=f"milestones/s2/results/{f}.jsonl"
    rs=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
    tot+=len(rs)
    print(f"  {f}: n={len(rs)}")
print("R13 TOTAL calls:", tot)
rs=[json.loads(l) for l in open("milestones/s2/results/r13_mitigation.jsonl",encoding="utf-8") if l.strip()]
print("r13_mitigation.jsonl n=", len(rs))
import glob
print("r13_mit_*.json count:", len(glob.glob("milestones/s2/results/r13_mit_*.json")))
