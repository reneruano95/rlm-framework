"""REFUTE pass 2, adversarial arm: try to KILL the 'clean' verdict.

The previous audit's detector is identifier-scoped (UUID / ENT code). R13's own
strongest artifact was 'ENT-#####:hex pairs', but a cross-request leak could
equally surface as a foreign ENTITY NAME with no identifier attached, which an
identifier-only detector cannot see. This scans every answer in the four
headline probes for entity names that belong to a DIFFERENT fixture chunk.

Entity names are harvested from the chunk corpus itself (the fixture generator
uses invented bigrams like 'Prylfennwick Trust' / 'Ilvkeldwick Depository'), so
the vocabulary is derived from data, not guessed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["s2/results/sweep.jsonl", "s2/results/refusal-ab.jsonl",
         "s2/results/refusal-ab-640.jsonl", "s2/results/distance.jsonl",
         "s2/results/sweep-run1-shared-server.jsonl"]

# invented-stem + institution-word bigrams, as the fixtures build them
NAME = re.compile(r"\b([A-Z][a-z]{4,}) (Trust|Chapterhouse|Depository|Bureau|"
                  r"Registry|Consortium|Syndicate|Foundation|Institute|Society|"
                  r"Company|Exchange|Guild|Fund|Authority|Board|Office|Works)\b")


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


idx = {}
for f in sorted(ROOT.glob("s2/fixtures*/**/*.chunk.txt")):
    t = f.read_text(encoding="utf-8")
    idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (f.as_posix(), t)

corpus_names = defaultdict(set)
for h, (p, t) in idx.items():
    for m in NAME.finditer(t):
        corpus_names[m.group(0)].add(h)

print(f"chunks={len(idx)}  distinct entity names in corpus={len(corpus_names)}")
uniq = sum(1 for n, hs in corpus_names.items() if len(hs) == 1)
print(f"names occurring in exactly one chunk: {uniq}/{len(corpus_names)}")
print("sample:", sorted(corpus_names)[:8])

# cell_uid -> chunk sha, so cache-phase rows (no chunk_sha256) still resolve
uid2sha = {}
for fp in FILES:
    for r in load(fp):
        if r.get("cell_uid") and r.get("chunk_sha256"):
            uid2sha[r["cell_uid"]] = r["chunk_sha256"]

for fp in FILES:
    rows = load(fp)
    nfor = 0
    unresolved = 0
    detail = []
    for r in rows:
        sha = r.get("chunk_sha256") or uid2sha.get(r.get("cell_uid"))
        out = r.get("raw_output") or ""
        if not out:
            continue
        if sha not in idx:
            unresolved += 1
            continue
        mine = idx[sha][1]
        # a name is foreign only if it is NOT in the sent chunk, NOT in the
        # question that was asked, and IS owned by some other chunk
        q = (r.get("question") or "")
        got = {m.group(0) for m in NAME.finditer(out)}
        foreign = {n for n in got
                   if n not in mine and n not in q and n in corpus_names}
        if foreign:
            nfor += 1
            detail.append((r.get("cell_id"), sorted(foreign),
                           sorted({idx[h][0].split("/")[-2] + "/" + idx[h][0].split("/")[-1]
                                   for n in foreign for h in corpus_names[n]}),
                           out[:80]))
    print(f"\n{fp}: rows with FOREIGN ENTITY NAME = {nfor}/{len(rows)} "
          f"(rows whose chunk could not be resolved: {unresolved})")
    for d in detail[:15]:
        print(f"   {d[0]}  {d[1]}  owned-by {d[2]}\n      out={d[3]!r}")
