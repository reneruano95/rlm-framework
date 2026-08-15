import json, os, collections
ROOT=r"D:/PROJECTS/rlm-halo-framework/s2/results"
def load(fn):
    with open(os.path.join(ROOT,fn),encoding="utf-8") as f:
        return {(x["size"],x["trial"],x["qtype"]): x for x in (json.loads(l) for l in f if l.strip())}
runs={
 "orig_hybrid(auto,4x16384)": "arch_ladder_qwen-hybrid.jsonl",
 "virgin(32x4096)":           "arch_ladder_qwen_virgin.jsonl",
 "shared0(32x4096)":          "arch_ladder_qwen_shared.jsonl",
 "origgeom_auto(4x16384)":    "arch_ladder_qwen-origgeom_auto.jsonl",
 "ogt0_auto(4x16384)":        "arch_ladder_og-t0_auto.jsonl",
 "ogt0_shared0(4x16384)":     "arch_ladder_og-t0_shared.jsonl",
}
data={k:load(v) for k,v in runs.items()}
keys=sorted(set().union(*[set(d) for d in data.values()]))
names=list(runs)
print("PAIRWISE byte-identical answer counts (out of 24):")
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=data[names[i]],data[names[j]]
        same=sum(1 for k in keys if k in a and k in b and a[k]["answer"]==b[k]["answer"])
        n=sum(1 for k in keys if k in a and k in b)
        print(f"  {names[i]:26s} vs {names[j]:26s}: {same}/{n} identical")
print("\nABSENT rows, class per run:")
hdr=f"{'size/trial':<12}"+"".join(f"{n[:18]:<20}" for n in names)
print(hdr)
for k in [k for k in keys if k[2]=="ABSENT"]:
    row=f"{str(k[0])+'/t'+str(k[1]):<12}"
    for n in names:
        x=data[n].get(k)
        row+=f"{(x['cls'] if x else '-'):<20}"
    print(row)
print("\nABSENT 2048 answers verbatim:")
for k in [k for k in keys if k[2]=="ABSENT" and k[0]==2048]:
    print(f"-- {k}")
    for n in names:
        x=data[n].get(k)
        if x: print(f"   {n:26s} slot={str(x.get('slot')):>4} served={str(x.get('slot_served')):>4} {x['cls']:<14} {x.get('detail','')!r}\n        {x['answer'][:150]!r}")
