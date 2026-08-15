"""REFUTE pass 2: re-derive the false-positive cause decomposition independently.

Identifier classes present in the fixtures (verified by inspection of
s2/fixtures/s2-1024-p50.chunk.txt):
    UUID          1251d802-86aa-4e75-96be-aefc175c1e8e
    ENT code      ENT-95082
    4-hex tag     b66f          (low entropy -- reported separately)

For every row the probe's OWN scorer labelled FALSE-POSITIVE, classify:
    OWN      >=1 extracted identifier is literally in the chunk that was sent
    FOREIGN  >=1 extracted identifier is in a DIFFERENT fixture chunk and not
             in the sent chunk
    FABRICATED  identifier-shaped output that is in no chunk anywhere
    NO-ID    no identifier-shaped token at all
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

UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ENT = re.compile(r"\bENT-\d{4,6}\b")
HEX4 = re.compile(r"(?<![0-9a-zA-Z-])[0-9a-f]{4}(?![0-9a-zA-Z-])")


def ids(text, with_hex4=True):
    out = set(u.lower() for u in UUID.findall(text))
    out |= set(e.upper() for e in ENT.findall(text))
    if with_hex4:
        # strip uuids first so their internal groups don't count
        stripped = UUID.sub(" ", text)
        out |= set(h.lower() for h in HEX4.findall(stripped))
    return out


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def build_index():
    idx = {}
    for f in sorted(ROOT.glob("s2/fixtures*/**/*.chunk.txt")):
        t = f.read_text(encoding="utf-8")
        idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (
            str(f.relative_to(ROOT)).replace("\\", "/"), t)
    return idx


def main():
    idx = build_index()
    for with_hex4 in (True, False):
        print("\n" + "#" * 70)
        print(f"# identifier classes: UUID + ENT" + (" + 4-hex tag" if with_hex4 else ""))
        print("#" * 70)

        corpus = defaultdict(set)   # id -> set of chunk shas
        for h, (path, txt) in idx.items():
            for i in ids(txt, with_hex4):
                corpus[i].add(h)

        grand = Counter()
        for fp in FILES:
            rows = load(fp)
            fps = [r for r in rows if r.get("label") == "FALSE-POSITIVE"]
            cnt = Counter()
            foreign_rows = []
            for r in fps:
                sha = r.get("chunk_sha256")
                mine = idx.get(sha)
                mytxt = mine[1] if mine else ""
                myids = ids(mytxt, with_hex4) if mine else set()
                out = r.get("raw_output") or ""
                got = ids(out, with_hex4)
                if not got:
                    cnt["NO-ID"] += 1
                    continue
                if got & myids:
                    cnt["OWN"] += 1
                elif got & set(corpus):
                    cnt["FOREIGN"] += 1
                    foreign_rows.append((r.get("cell_id"), sorted(got & set(corpus)),
                                         [idx[h][0] for i in (got & set(corpus))
                                          for h in corpus[i]][:3]))
                else:
                    cnt["FABRICATED"] += 1
            print(f"\n{fp}: FALSE-POSITIVE rows = {len(fps)}")
            print(f"   {dict(cnt)}")
            if fp != "s2/results/sweep-run1-shared-server.jsonl":
                grand.update(cnt)
            for fr in foreign_rows[:12]:
                print(f"     FOREIGN: cell={fr[0]} ids={fr[1]} owner={fr[2]}")
        print(f"\n  GRAND TOTAL over the four headline probes: {dict(grand)} "
              f"(n={sum(grand.values())})")

    # ---- whole-row foreign scan over ALL rows (not just FPs), uuid+ENT only
    print("\n" + "=" * 70)
    print("WHOLE-FILE foreign scan (all rows, all labels), UUID+ENT identifiers")
    corpus = defaultdict(set)
    for h, (path, txt) in idx.items():
        for i in ids(txt, False):
            corpus[i].add(h)
    for fp in FILES:
        rows = load(fp)
        nfor = 0
        detail = []
        for r in rows:
            sha = r.get("chunk_sha256")
            mine = idx.get(sha)
            myids = ids(mine[1], False) if mine else set()
            got = ids(r.get("raw_output") or "", False)
            f = (got & set(corpus)) - myids
            if f:
                nfor += 1
                detail.append((r.get("cell_id"), sorted(f),
                               sorted({idx[h][0] for i in f for h in corpus[i]})))
        print(f"  {fp}: rows with foreign identifier = {nfor}/{len(rows)}")
        for d in detail[:20]:
            print(f"     {d[0]}  {d[1]}  owned-by {d[2]}")


if __name__ == "__main__":
    main()
