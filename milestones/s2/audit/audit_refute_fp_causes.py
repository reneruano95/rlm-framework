"""Independent decomposition of every false positive: does the answer carry an
identifier that is IN the chunk that was sent (misattribution) or from elsewhere
(leak) or from nowhere (fabrication)?"""
import json, re, hashlib, collections
from pathlib import Path
ROOT = Path("D:/PROJECTS/rlm-halo-framework/milestones/s2")
PATS = {"uuid": re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"),
        "ent":  re.compile(r"\bENT-\d{4,6}\b"),
        "hex":  re.compile(r"\b[0-9a-fA-F]{16,}\b")}
chunks = {}
for p in sorted(ROOT.glob("fixtures*/**/*.chunk.txt")):
    t = p.read_text(encoding="utf-8"); chunks[hashlib.sha256(t.encode()).hexdigest()] = t
owner = collections.defaultdict(set)
for h, t in chunks.items():
    for k in ("uuid", "ent"):
        for v in PATS[k].findall(t): owner[v.lower()].add(h)

def load(f): return [json.loads(l) for l in (ROOT/"results"/f).open(encoding="utf-8") if l.strip()]
FPLAB = {"FALSE-POSITIVE", "CONFABULATION"}
for f in ("sweep.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl", "distance.jsonl"):
    rows = [r for r in load(f) if r.get("status") == "ok" and r.get("question_type") == "absent"]
    fps = [r for r in rows if r.get("label") in FPLAB]
    c = collections.Counter(); samples = collections.defaultdict(list)
    for r in fps:
        sha = r.get("chunk_sha256"); raw = r.get("raw_output") or ""
        if sha not in chunks: c["no_chunk"] += 1; continue
        sent = chunks[sha] + "\n\n" + (r.get("question") or "")
        found = {(k, m.lower()) for k, pat in PATS.items() for m in pat.findall(raw)}
        if not found:
            c["no identifier at all"] += 1; samples["no identifier at all"].append(raw[:90]); continue
        if any(v in sent.lower() for _, v in found):
            c["quoted-own (in sent chunk)"] += 1
        elif any(k in ("uuid","ent") and owner.get(v, set()) - {sha} for k, v in found):
            c["FOREIGN (leak)"] += 1; samples["FOREIGN (leak)"].append(raw[:90])
        else:
            c["fabricated identifier"] += 1; samples["fabricated identifier"].append(raw[:90])
    n = len(fps)
    print(f"\n===== {f}: absent calls={len(rows)}  false positives={n} =====")
    for k, v in c.most_common():
        print(f"   {k:<30} {v:>4}  ({v/n*100:.0f}%)" if n else k)
    for k in ("no identifier at all", "fabricated identifier", "FOREIGN (leak)"):
        for s in samples[k][:3]:
            print(f"     [{k}] {s!r}")
print("\nlabels seen on absent calls:",
      sorted({r.get('label') for f in ("sweep.jsonl","refusal-ab.jsonl","refusal-ab-640.jsonl","distance.jsonl")
              for r in load(f) if r.get("question_type")=="absent"}, key=str))
