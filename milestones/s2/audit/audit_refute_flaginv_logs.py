import re, sys, os
from collections import Counter

LOGS = {
 "distance-leaf": "traces/logs/distance-leaf.err.log",
 "leaf-server":   "traces/logs/leaf-server.err.log",
 "arch-qwen":     "milestones/s2/logs/arch-qwen.err",
 "arch-gemma":    "milestones/s2/logs/arch-gemma.err",
 "arch-q35":      "milestones/s2/logs/arch-q35.err",
 "bos":           "milestones/s2/logs/bos.err",
}

pat_byid  = re.compile(r"selected slot by id")
pat_lcp   = re.compile(r"selected slot by LCP similarity")
pat_lru   = re.compile(r"selected slot by LRU")
pat_evict = re.compile(r"making room for prompt cache entry, removing oldest entry")
pat_slots = re.compile(r"n_slots\s*=\s*(\d+)")
pat_ctxslot = re.compile(r"n_ctx_slot\s*=\s*(\d+)")
pat_kvu   = re.compile(r"kv_unified\s*=\s*'?(\w+)'?")
# slot id from "selected slot by id (N)" or "slot ... id  N"
pat_byid_n = re.compile(r"selected slot by id\s*\((\d+)\)")
pat_lcp_n  = re.compile(r"selected slot by LCP similarity\s*\((\d+)\)")
pat_lru_n  = re.compile(r"selected slot by LRU\s*\((\d+)\)")

for name, p in LOGS.items():
    if not os.path.exists(p):
        print(f"{name}: MISSING {p}"); continue
    n_byid=n_lcp=n_lru=n_evict=0
    slots_seq=[]
    hdr={}
    lines=0
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            lines+=1
            if pat_byid.search(line):
                n_byid+=1
                m=pat_byid_n.search(line)
                slots_seq.append(("id", int(m.group(1)) if m else -1))
            if pat_lcp.search(line):
                n_lcp+=1
                m=pat_lcp_n.search(line)
                slots_seq.append(("lcp", int(m.group(1)) if m else -1))
            if pat_lru.search(line):
                n_lru+=1
                m=pat_lru_n.search(line)
                slots_seq.append(("lru", int(m.group(1)) if m else -1))
            if pat_evict.search(line): n_evict+=1
            if "n_slots" in line and "n_slots" not in hdr:
                m=pat_slots.search(line)
                if m: hdr["n_slots"]=(m.group(1), i)
            if "n_ctx_slot" in line and "n_ctx_slot" not in hdr:
                m=pat_ctxslot.search(line)
                if m: hdr["n_ctx_slot"]=(m.group(1), i)
            if "kv_unified" in line and "kv_unified" not in hdr:
                m=pat_kvu.search(line)
                if m: hdr["kv_unified"]=(m.group(1), i)
    uniq=sorted({s for _,s in slots_seq})
    print(f"\n=== {name} ({p}) lines={lines}")
    print(f"  header: {hdr}")
    print(f"  selected-by-id={n_byid}  LCP={n_lcp}  LRU={n_lru}  total_sel={n_byid+n_lcp+n_lru}")
    print(f"  prompt-cache-evictions={n_evict}")
    print(f"  distinct slot ids selected={len(uniq)} min={min(uniq) if uniq else None} max={max(uniq) if uniq else None}")
    # revisit detection: did a slot get selected again after a DIFFERENT slot was used in between?
    seen_last=None; revisits=0; last_seen_index={}
    for idx,(kind,s) in enumerate(slots_seq):
        if s in last_seen_index and last_seen_index[s] != idx-1:
            revisits+=1
        last_seen_index[s]=idx
    print(f"  non-contiguous revisits (slot selected again after another slot in between)={revisits}")
