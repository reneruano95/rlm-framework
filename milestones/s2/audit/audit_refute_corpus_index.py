"""Build an INDEPENDENT chunk index over every fixture .chunk.txt on disk,
keyed by sha256 of the raw bytes AND of the decoded text, so a record's
chunk_sha256 can be resolved without trusting any prior audit artefact.
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2")
UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# other identifier shapes seen in these fixtures
OTHER = re.compile(r"\b(?:[A-Z]{2,5}-[0-9A-Z]{2,}(?:-[0-9A-Z]+)*|[0-9A-F]{16,})\b")

idx = {}
files = sorted(ROOT.glob("fixtures*/**/*.chunk.txt"))
for p in files:
    raw = p.read_bytes()
    txt = raw.decode("utf-8")
    for h, how in ((hashlib.sha256(raw).hexdigest(), "bytes"),
                   (hashlib.sha256(txt.encode("utf-8")).hexdigest(), "text")):
        idx.setdefault(h, {"paths": [], "how": how})
        if str(p) not in idx[h]["paths"]:
            idx[h]["paths"].append(str(p))
        idx[h]["text"] = txt
print(f"fixture chunk files: {len(files)}   distinct sha256 keys: {len(idx)}")

# resolve every chunk_sha256 referenced by the four probes + control
probes = ["results/sweep.jsonl", "results/refusal-ab.jsonl", "results/refusal-ab-640.jsonl",
          "results/distance.jsonl", "results/sweep-run1-shared-server.jsonl"]
for pr in probes:
    p = ROOT / pr
    shas, n = set(), 0
    for line in p.open(encoding="utf-8"):
        if not line.strip(): continue
        n += 1
        shas.add(json.loads(line)["chunk_sha256"])
    unres = [s for s in shas if s not in idx]
    print(f"{pr:<42} rows={n:>5} distinct chunk_sha256={len(shas):>3} UNRESOLVED={len(unres)}")
    for s in unres[:5]:
        print(f"     !! {s}")

out = {h: {"paths": v["paths"], "uuids": sorted(set(UUID.findall(v["text"]))),
           "other": sorted(set(OTHER.findall(v["text"]))), "chars": len(v["text"])}
       for h, v in idx.items()}
Path(ROOT / "audit" / "_refute_chunk_index.json").write_text(json.dumps(out), encoding="utf-8")
print("\nwrote milestones/s2/audit/_refute_chunk_index.json")
# uniqueness of identifiers across the whole fixture corpus
owner = {}
for h, v in out.items():
    for u in v["uuids"]:
        owner.setdefault(u, set()).add(h)
multi = {u: len(s) for u, s in owner.items() if len(s) > 1}
print(f"distinct UUIDs in corpus: {len(owner)}   appearing in >1 chunk: {len(multi)}")
if multi:
    for u, c in list(multi.items())[:20]:
        print(f"   {u} in {c} chunks: {sorted(owner[u])[:3]}")
