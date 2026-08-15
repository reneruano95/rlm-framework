import re, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework"
PE=re.compile(r"task (\d+) \| prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
EV=re.compile(r"task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
REL=re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")
def analyse(rel, name):
    p=os.path.join(ROOT,rel)
    if not os.path.exists(p): print(name,"MISSING"); return
    pe={}; ev={}; rl={}
    with open(p,encoding="utf-8",errors="replace") as f:
        for line in f:
            m=PE.search(line);  ev_m=EV.search(line); r=REL.search(line)
            if m: pe[int(m.group(1))]=int(m.group(2))
            if ev_m: ev[int(ev_m.group(1))]=int(ev_m.group(2))
            if r: rl[int(r.group(1))]=int(r.group(2))
    tasks=sorted(set(pe)&set(rl))
    carry=[]
    for t in tasks:
        c = rl[t] - (pe[t] + ev.get(t,0)) + 1   # release n_tokens excludes last sampled tok
        carry.append((t,c,pe[t],ev.get(t,0),rl[t]))
    big=[x for x in carry if x[1]>8]
    tot=sum(max(0,x[1]) for x in carry)
    print(f"{name:24s} tasks={len(tasks):4d} carryover>8tok: {len(big):4d}  total_carry={tot:7d}  max_carry={max((x[1] for x in carry), default=0)}")
    if big[:3]: print("   sample:", big[:3])
for rel,name in [("s2/logs/arch-qwen.err","arch-qwen"),("s2/logs/arch-gemma.err","arch-gemma"),
                 ("s2/logs/bos.err","bos"),("s2/logs/slotfix-qwen.err","slotfix-virgin+shared"),
                 ("s2/logs/origgeom.err","origgeom"),
                 ("traces/logs/distance-leaf.err.log","distance-leaf"),
                 ("traces/logs/leaf-server.err.log","leaf-server(refusal)")]:
    analyse(rel,name)
