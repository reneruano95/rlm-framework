import os, re, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework"
PATS={"prompt_cache_room":"making room for prompt cache entry",
      "prompt_cache_any":"prompt cache",
      "state_save":"state save","state_restore":"state restore",
      "alloc_line":"alloc:"}
for rel in ["s2/logs/arch-qwen.err","s2/logs/arch-gemma.err","s2/logs/origgeom.err",
            "s2/logs/slotfix-qwen.err","s2/logs/bos.err",
            "traces/logs/distance-leaf.err.log","traces/logs/leaf-server.err.log"]:
    p=os.path.join(ROOT,rel)
    if not os.path.exists(p): print(rel,"MISSING"); continue
    c=collections.Counter(); n=0
    with open(p,encoding="utf-8",errors="replace") as f:
        for line in f:
            n+=1
            for k,s in PATS.items():
                if s in line: c[k]+=1
    print(f"{rel:38s} lines={n:7d} " + " ".join(f"{k}={c[k]}" for k in PATS))
# what distinct 'alloc:' messages appear in the arch logs
for rel in ["s2/logs/origgeom.err","s2/logs/slotfix-qwen.err","s2/logs/arch-qwen.err"]:
    p=os.path.join(ROOT,rel); seen=collections.Counter()
    with open(p,encoding="utf-8",errors="replace") as f:
        for line in f:
            if "alloc:" in line or "cache" in line.lower():
                key=re.sub(r"\d+","N",line.split("|")[-1].strip())[:90]
                seen[key]+=1
    print(f"\n--- {rel} cache-ish message shapes:")
    for k,v in seen.most_common(12): print(f"   {v:5d}  {k}")
