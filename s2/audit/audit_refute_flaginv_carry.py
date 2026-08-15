import re, io, os
out=io.open("s2/audit/_carry_out.txt","w",encoding="utf-8")
def P(*a): out.write(" ".join(str(x) for x in a)+"\n")

pe = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
ev = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*\s*eval time =\s*[\d.]+ ms /\s*(\d+) tokens")
tt = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*total time =\s*[\d.]+ ms /\s*(\d+) tokens")
rl = re.compile(r"id\s+(\d+)\s*\|\s*task\s+(\d+)\s*\|\s*stop processing:\s*n_tokens\s*=\s*(\d+)")

for name,p in [("arch-qwen","s2/logs/arch-qwen.err"),("arch-gemma","s2/logs/arch-gemma.err"),
               ("bos","s2/logs/bos.err"),
               ("distance-leaf","traces/logs/distance-leaf.err.log"),
               ("leaf-server","traces/logs/leaf-server.err.log")]:
    if not os.path.exists(p): P(name,"MISSING"); continue
    prompt={}; evald={}; total={}; ntok={}
    with open(p,encoding="utf-8",errors="replace") as f:
        for line in f:
            if "prompt eval time" in line:
                m=pe.search(line)
                if m: prompt[(int(m.group(1)),int(m.group(2)))]=int(m.group(3))
            elif "total time" in line:
                m=tt.search(line)
                if m: total[(int(m.group(1)),int(m.group(2)))]=int(m.group(3))
            elif "eval time" in line:
                m=ev.search(line)
                if m: evald[(int(m.group(1)),int(m.group(2)))]=int(m.group(3))
            elif "stop processing" in line:
                m=rl.search(line)
                if m: ntok[(int(m.group(1)),int(m.group(2)))]=int(m.group(3))
    keys=sorted(set(ntok)|set(prompt))
    carried=[]
    for k in keys:
        pn=prompt.get(k); en=evald.get(k); nt=ntok.get(k)
        if pn is None or en is None or nt is None: continue
        c = nt - (pn + en)          # tokens resident that were never prefilled or decoded
        carried.append((k, pn, en, nt, c))
    P(f"\n===== {name}  tasks_with_full_triple={len(carried)}  (prompt_evals={len(prompt)}, releases={len(ntok)})")
    if not carried: continue
    cs=[c for _,_,_,_,c in carried]
    gt8=[x for x in carried if x[4]>8]
    P(f"  carry-over (n_tokens - prompt_eval - eval): min={min(cs)} max={max(cs)} sum={sum(cs)}")
    P(f"  tasks with carry-over > 8 tokens: {len(gt8)} / {len(carried)}")
    P(f"  tasks with carry-over > 0 tokens: {sum(1 for x in cs if x>0)} / {len(carried)}")
    if gt8[:5]: P(f"   examples (slot,task,prompt_n,eval_n,n_tokens,carry): {gt8[:5]}")
    P(f"  max prompt_eval tokens: {max(prompt.values())}   max n_tokens: {max(ntok.values())}")
out.close()
print(open("s2/audit/_carry_out.txt",encoding="utf-8").read())
