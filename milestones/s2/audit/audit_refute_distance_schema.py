import json, collections
from pathlib import Path
p = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2/results/distance.jsonl")
keysets = collections.Counter()
byphase = collections.Counter()
missing = []
rows = []
for i, line in enumerate(p.open(encoding="utf-8")):
    if not line.strip(): continue
    d = json.loads(line); rows.append(d)
    keysets[tuple(sorted(d.keys()))] += 1
    byphase[(d.get("phase"), d.get("arm"), d.get("layout"))] += 1
    if "chunk_sha256" not in d: missing.append(i)
print(f"rows={len(rows)}  distinct key-schemas={len(keysets)}  rows missing chunk_sha256={len(missing)}")
for ks, c in keysets.most_common():
    print(f"\n-- schema n={c}")
    print("   keys:", ", ".join(ks))
print("\nphase/arm/layout counts:")
for k, v in sorted(byphase.items(), key=lambda x: str(x)):
    print("  ", k, v)
if missing:
    d = rows[missing[0]]
    print("\nexample row lacking chunk_sha256:")
    for k, v in d.items():
        s = repr(v)
        print(f"   {k}: {s[:120]}")
