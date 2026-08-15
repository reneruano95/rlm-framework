"""Inspect the distance.jsonl rows my foreign-scan flagged: are they real
foreign hits, or rows whose chunk_sha256 my index simply could not resolve?"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

rows = [json.loads(l) for l in (ROOT / "s2/results/distance.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

idx = {}
for f in sorted(ROOT.glob("s2/fixtures*/**/*.chunk.txt")):
    t = f.read_text(encoding="utf-8")
    idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (str(f.relative_to(ROOT)), t)

no_sha = [r for r in rows if not r.get("chunk_sha256")]
print(f"rows total {len(rows)}; rows WITHOUT chunk_sha256: {len(no_sha)}")
print("  their phase :", dict(Counter(r.get("phase") for r in no_sha)))
print("  their arm   :", dict(Counter(r.get("arm") for r in no_sha)))
print("  their label :", dict(Counter(r.get("label") for r in no_sha)))
print("  their keys  :", sorted(no_sha[0].keys()) if no_sha else "-")
print()
for r in no_sha[:6]:
    print({k: (str(v)[:90]) for k, v in r.items()
           if k in ("phase", "arm", "layout", "cell_id", "cell_uid", "label",
                    "question", "expected", "raw_output", "slot_id", "id_slot",
                    "leak_detected", "note", "kind", "chunk_sha256")})
print()
have_sha = [r for r in rows if r.get("chunk_sha256")]
unres = [r for r in have_sha if r["chunk_sha256"] not in idx]
print(f"rows WITH chunk_sha256: {len(have_sha)}; unresolved against fixture index: {len(unres)}")
print("  unresolved phases:", dict(Counter(r.get("phase") for r in unres)))

# strict foreign scan restricted to resolvable rows
corpus = {}
for h, (p, t) in idx.items():
    for u in set(x.lower() for x in UUID.findall(t)):
        corpus.setdefault(u, set()).add(h)
nfor = 0
for r in have_sha:
    if r["chunk_sha256"] not in idx:
        continue
    mytxt = idx[r["chunk_sha256"]][1].lower()
    got = set(x.lower() for x in UUID.findall(r.get("raw_output") or ""))
    foreign = {u for u in got if u not in mytxt and u in corpus}
    if foreign:
        nfor += 1
        print("  FOREIGN:", r.get("cell_id"), foreign)
print(f"strict foreign rows among {sum(1 for r in have_sha if r['chunk_sha256'] in idx)} "
      f"resolvable distance rows: {nfor}")
