import re
P=r"D:/PROJECTS/rlm-halo-framework/milestones/s2/logs/origgeom.err"
SEL=re.compile(r"id\s+(\d+) \| task (-?\d+) \| selected slot by (id \(\d+\)|LCP similarity[^\n]*|LRU[^\n]*)")
rows=[]
with open(P,encoding="utf-8",errors="replace") as f:
    for line in f:
        m=SEL.search(line)
        if m: rows.append((int(m.group(1)), m.group(3)[:28]))
print("total", len(rows))
# fixture/question order: 12 fixtures x (LITERAL, ABSENT); arm = 24 calls
labels=[]
for size in (640,1024,2048):
    for t in range(4):
        for q in ("LIT","ABS"): labels.append(f"{size}/t{t}/{q}")
for arm,name in ((0,"origgeom_auto (fresh proc)"),(1,"og-t0_auto"),(2,"og-t0_shared")):
    seg=rows[arm*24:(arm+1)*24]
    print(f"\n--- arm {arm}: {name}")
    hist={}
    for i,(slot,kind) in enumerate(seg):
        lab=labels[i]
        prev=hist.get(slot,[])
        mark=""
        if lab.startswith("2048/t0/ABS") or lab.startswith("2048/t2/ABS"): mark="   <== LEAKED ROW"
        print(f"  {lab:16s} slot={slot} {kind:26s} slot_prev_held={prev}{mark}")
        hist.setdefault(slot,[]).append(lab.rsplit('/',1)[0])
