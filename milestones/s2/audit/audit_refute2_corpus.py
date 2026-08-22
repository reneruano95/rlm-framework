"""REFUTE pass 2: independent corpus forensics over the four 'headline' probes
plus the known-contaminated control run.

Builds its OWN chunk index by hashing every fixture .chunk.txt on disk and
matching sha256 against each record's chunk_sha256 -- no reliance on cell_id
naming conventions, and no reliance on the previous auditor's _chunk_index.json.

Reports, per file:
  - row count, label histogram, arm/layout/size breakdown
  - identifier extraction from raw_output
  - own / FOREIGN / nowhere classification of every extracted identifier
  - identifier uniqueness across the whole chunk corpus
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = ["milestones/s2/results/sweep.jsonl", "milestones/s2/results/refusal-ab.jsonl",
         "milestones/s2/results/refusal-ab-640.jsonl", "milestones/s2/results/distance.jsonl",
         "milestones/s2/results/sweep-run1-shared-server.jsonl"]

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def chunk_index():
    idx = {}
    for f in ROOT.glob("milestones/s2/fixtures*/**/*.chunk.txt"):
        t = f.read_text(encoding="utf-8")
        for enc in (t.encode("utf-8"),):
            h = hashlib.sha256(enc).hexdigest()
            idx[h] = (str(f.relative_to(ROOT)), t)
    return idx


def main():
    idx = chunk_index()
    print(f"chunk files hashed: {len(idx)}")

    # global identifier ownership over the WHOLE fixture corpus
    owner = defaultdict(set)
    for h, (path, txt) in idx.items():
        for u in set(UUID_RE.findall(txt)):
            owner[u.lower()].add(h)

    print(f"distinct uuids in fixture corpus: {len(owner)}")
    multi = {u: hs for u, hs in owner.items() if len(hs) > 1}
    print(f"uuids appearing in >1 chunk file: {len(multi)}")

    for fp in FILES:
        rows = load(fp)
        print(f"\n================ {fp}  rows={len(rows)} ================")
        print("  labels:", dict(Counter(r.get("label") for r in rows)))
        if any("arm" in r for r in rows):
            print("  arms  :", dict(Counter(r.get("arm") for r in rows)))
        if any("layout" in r for r in rows):
            print("  layout:", dict(Counter(r.get("layout") for r in rows)))
        print("  sizes :", dict(Counter(r.get("size_target") for r in rows)))
        if any("phase" in r for r in rows):
            print("  phase :", dict(Counter(r.get("phase") for r in rows)))

        # sha resolution
        unres = [r for r in rows if r["chunk_sha256"] not in idx]
        print(f"  chunk_sha256 resolved: {len(rows)-len(unres)}/{len(rows)}")
        if unres:
            print("   UNRESOLVED shas:", sorted({r['chunk_sha256'][:12] for r in unres}))

        # identifier forensics
        n_ids = 0
        own = foreign = nowhere = 0
        foreign_detail = []
        rows_with_foreign = set()
        for i, r in enumerate(rows):
            out = r.get("raw_output") or ""
            sha = r["chunk_sha256"]
            mine = idx.get(sha)
            mytxt = (mine[1] if mine else "").lower()
            for u in set(x.lower() for x in UUID_RE.findall(out)):
                n_ids += 1
                if u in mytxt:
                    own += 1
                elif u in owner:
                    foreign += 1
                    rows_with_foreign.add(i)
                    foreign_detail.append((i, u, r.get("cell_id"),
                                           sorted(idx[h][0] for h in owner[u])))
                else:
                    nowhere += 1
        print(f"  uuids in answers: {n_ids}  own={own}  FOREIGN={foreign}  "
              f"not-in-corpus={nowhere}")
        print(f"  ROWS containing >=1 foreign uuid: {len(rows_with_foreign)}/{len(rows)}")
        for d in foreign_detail[:25]:
            print(f"    row{d[0]} {d[1]} asked-in={d[2]} owned-by={d[3]}")

        # uniqueness of identifiers restricted to the chunks this file uses
        used = {r["chunk_sha256"] for r in rows if r["chunk_sha256"] in idx}
        ids_here = defaultdict(set)
        for h in used:
            for u in set(UUID_RE.findall(idx[h][1])):
                ids_here[u.lower()].add(h)
        uniq = sum(1 for u, hs in ids_here.items() if len(hs) == 1)
        print(f"  identifiers across this file's {len(used)} chunks: "
              f"{uniq}/{len(ids_here)} occur in exactly one chunk")


if __name__ == "__main__":
    main()
