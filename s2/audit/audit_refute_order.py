import re, os
p=r"D:/PROJECTS/rlm-halo-framework/s2/logs/slotfix-qwen.err"
SEL=re.compile(r"selected slot by id \((\d+)\)")
seq=[]
with open(p,encoding="utf-8",errors="replace") as f:
    for line in f:
        m=SEL.search(line)
        if m: seq.append(int(m.group(1)))
print("slotfix-qwen.err selection order (48):")
print(seq)
print("\nfirst 24 (virgin arm):", seq[:24])
print("last  24 (shared arm):", seq[24:])
print("shared arm all slot 0?", all(s==0 for s in seq[24:]))
print("slot 0 used during virgin arm?", 0 in seq[:24])

p2=r"D:/PROJECTS/rlm-halo-framework/s2/logs/origgeom.err"
seq2=[]
kinds=[]
K=re.compile(r"selected slot by (id \((\d+)\)|LCP similarity|LRU)")
with open(p2,encoding="utf-8",errors="replace") as f:
    for line in f:
        m=K.search(line)
        if m:
            if m.group(1).startswith("id"): kinds.append(("id",int(m.group(2))))
            else: kinds.append((m.group(1),None))
print("\noriggeom.err first 24:", kinds[:24])
print("origgeom.err 25-48   :", kinds[24:48])
print("origgeom.err 49-72   :", kinds[48:])
