import re, collections
for P,name in [(r"D:/PROJECTS/rlm-halo-framework/s2/logs/arch-qwen.err","arch-qwen"),
               (r"D:/PROJECTS/rlm-halo-framework/s2/logs/arch-gemma.err","arch-gemma")]:
    SEL=re.compile(r"id\s+(\d+) \| task (-?\d+) \| selected slot by (id \(\d+\)|LCP similarity|LRU)")
    rows=[]
    with open(P,encoding="utf-8",errors="replace") as f:
        for line in f:
            m=SEL.search(line)
            if m: rows.append((int(m.group(1)), m.group(3)))
    print(f"\n{name}: {len(rows)} selections; slot histogram {collections.Counter(s for s,_ in rows)}")
    for arm in range(len(rows)//24):
        seg=rows[arm*24:(arm+1)*24]
        print(f"  arm{arm}: slots {collections.Counter(s for s,_ in seg)}  kinds {collections.Counter(k for _,k in seg)}")
