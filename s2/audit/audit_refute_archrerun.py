import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/s2/results"
def load(fn):
    p=os.path.join(ROOT,fn)
    with open(p,encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
for fn in ["arch_ladder_qwen-hybrid.jsonl","arch_ladder_gemma-fullattn.jsonl",
           "arch_ladder_qwen_virgin.jsonl","arch_ladder_qwen_shared.jsonl",
           "arch_ladder_qwen-origgeom_auto.jsonl","arch_ladder_og-t0_auto.jsonl","arch_ladder_og-t0_shared.jsonl"]:
    try: r=load(fn)
    except Exception as e:
        print(fn,"ERR",e); continue
    print(f"\n### {fn}  n={len(r)}")
    print("keys:", sorted(r[0].keys()))
    for k in ("model","arm","condition","chunk_tokens","size","id_slot","requested_slot","question_kind","kind","verdict","refused","correct"):
        vals=[x.get(k) for x in r if k in x]
        if vals: print(f"  {k}: {collections.Counter(map(str,vals))}")
