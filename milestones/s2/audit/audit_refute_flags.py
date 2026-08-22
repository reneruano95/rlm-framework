import re, os, json, collections
ROOT = r"D:/PROJECTS/rlm-halo-framework"

SEL = re.compile(r"selected slot by (id \((\d+)\)|LCP similarity|LRU)")
def sel_counts(path):
    c = collections.Counter(); ids=[]
    if not os.path.exists(path): return None, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEL.search(line)
            if m:
                kind = "id" if m.group(1).startswith("id") else m.group(1)
                c[kind]+=1
                if kind=="id": ids.append(int(m.group(2)))
    return c, ids

def nslots(path):
    if not os.path.exists(path): return None
    with open(path, encoding="utf-8", errors="replace") as f:
        for i,line in enumerate(f,1):
            if "n_slots" in line:
                return i, line.strip()
    return None

logs = {
 "arch-qwen": "milestones/s2/logs/arch-qwen.err",
 "arch-gemma": "milestones/s2/logs/arch-gemma.err",
 "arch-q35": "milestones/s2/logs/arch-q35.err",
 "bos": "milestones/s2/logs/bos.err",
 "slotfix-qwen": "milestones/s2/logs/slotfix-qwen.err",
 "origgeom": "milestones/s2/logs/origgeom.err",
 "distance-leaf": "traces/logs/distance-leaf.err.log",
 "leaf-server(refusal)": "traces/logs/leaf-server.err.log",
}
for name, rel in logs.items():
    p = os.path.join(ROOT, rel)
    c, ids = sel_counts(p)
    ns = nslots(p)
    if c is None:
        print(f"{name:22s} MISSING {rel}"); continue
    tot = sum(c.values())
    extra=""
    if ids:
        # monotonic-never-return check
        seen=set(); returns=0; last=None
        order=[]
        for i in ids:
            if i!=last: order.append(i)
            last=i
        s=set(); ret=0
        prev=None
        for i in order:
            if i in s and i!=prev: ret+=1
            s.add(i); prev=i
        extra=f" distinct_ids={len(set(ids))} min={min(ids)} max={max(ids)} returns_after_other_slot={ret}"
    print(f"{name:22s} total_sel={tot:5d} by_id={c['id']:5d} LCP={c['LCP similarity']:4d} LRU={c['LRU']:4d}{extra}")
    print(f"{'':22s} n_slots line {ns[0] if ns else '?'}: {ns[1] if ns else '?'}")
