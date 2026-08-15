import json, os, collections, re
ROOT=r"D:/PROJECTS/rlm-halo-framework/s2/results"
def load(fn):
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
FOREIGN = {
 "48e81295-9489-33be-cc30-430d702be6c3":"Marnwickstead(1024/t3)",
 "d9f804c1-fa2b-8d32-a160-adccebcd8978":"Quinfennsted(1024/t1)",
 "7e41c11e-4a6f-131b-df64-d2385eb09ba3":"Selkdaleridge(1024/t3)",
}
files=["arch_ladder_qwen-hybrid.jsonl","arch_ladder_gemma-fullattn.jsonl",
       "arch_ladder_qwen_virgin.jsonl","arch_ladder_qwen_shared.jsonl",
       "arch_ladder_qwen-origgeom_auto.jsonl","arch_ladder_og-t0_auto.jsonl","arch_ladder_og-t0_shared.jsonl"]
for fn in files:
    r=load(fn)
    print(f"\n### {fn}  policy={set(str(x.get('policy')) for x in r)}")
    by=collections.defaultdict(lambda: collections.Counter())
    for x in r: by[(x.get("qtype"),x.get("size"))][x.get("cls")]+=1
    for k in sorted(by, key=lambda t:(str(t[0]),t[1])):
        print(f"  qtype={k[0]:10s} size={k[1]:5} -> {dict(by[k])}")
    # foreign uuid hunt
    hits=[]
    for x in r:
        a=str(x.get("answer",""))
        for u,tag in FOREIGN.items():
            if u in a: hits.append((x.get("qtype"),x.get("size"),x.get("trial"),tag,x.get("slot"),x.get("slot_served")))
    print("  FOREIGN-UUID hits:", hits if hits else "NONE")
    # any uuid at all in ABSENT answers
    UU=re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    absent=[x for x in r if str(x.get("qtype")).lower().startswith("absent")]
    print(f"  ABSENT rows={len(absent)} cls={collections.Counter(x.get('cls') for x in absent)}")
    for x in absent:
        us=UU.findall(str(x.get("answer","")))
        if us: print(f"    size={x.get('size')} trial={x.get('trial')} slot={x.get('slot')} uuid={us[:2]} cls={x.get('cls')}")
