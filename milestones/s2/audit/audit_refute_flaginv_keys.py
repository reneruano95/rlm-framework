import json, os
files = ["milestones/s2/results/distance.jsonl","milestones/s2/results/refusal-ab.jsonl","milestones/s2/results/refusal-ab-640.jsonl",
         "milestones/s2/results/occupancy.jsonl","milestones/s2/results/r14.jsonl","milestones/s2/results/cache_instrument.jsonl",
         "milestones/s2/results/sweep.jsonl","milestones/s2/results/sweep-run1-shared-server.jsonl",
         "milestones/s2/results/arch_ladder_qwen-hybrid.jsonl","milestones/s2/results/arch_ladder_gemma-fullattn.jsonl"]
for p in files:
    if not os.path.exists(p): print(p,"MISSING"); continue
    n=0; keys=None
    with open(p, encoding="utf-8") as f:
        first=f.readline()
        rec=json.loads(first)
        keys=sorted(rec.keys())
        n=1
        for line in f:
            if line.strip(): n+=1
    print(f"\n=== {p}  records={n}")
    print("  keys:", keys)
    for k in ("slot_id","slot","chunk_sha256","tokens_cached","cache_n","run_id","argv","extra","doc_id","chunk_id"):
        if k in rec: print(f"    {k} = {str(rec[k])[:160]}")
