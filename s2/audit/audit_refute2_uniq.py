import hashlib, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ENT = re.compile(r"\bENT-\d{4,6}\b")


def ids(t):
    return set(u.lower() for u in UUID.findall(t)) | set(ENT.findall(t))


idx = {}
for f in sorted(ROOT.glob("s2/fixtures*/**/*.chunk.txt")):
    t = f.read_text(encoding="utf-8")
    idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (str(f.as_posix()), t)

for fp, label in (("s2/results/distance.jsonl", "distance"),
                  ("s2/results/refusal-ab.jsonl", "refusal-ab")):
    rows = [json.loads(l) for l in (ROOT / fp).read_text(encoding="utf-8").splitlines() if l.strip()]
    for sz in ("640", "1024", "2048", "ALL"):
        used = {r["chunk_sha256"] for r in rows
                if r.get("chunk_sha256") in idx
                and (sz == "ALL" or str(r.get("size_target")) == sz)}
        m = defaultdict(set)
        for h in used:
            for i in ids(idx[h][1]):
                m[i].add(h)
        uniq = sum(1 for i, hs in m.items() if len(hs) == 1)
        print(f"{label:10s} size={sz:5s} chunks={len(used):3d} unique={uniq}/{len(m)}")
