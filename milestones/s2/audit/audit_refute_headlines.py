"""Reproduce the spec's headline numbers straight from the raw jsonl, and probe
the report's two weakest provenance claims.

H1  distance.jsonl: the 640-vs-1024 false-positive contrast (spec §4:101-102)
H2  distance.jsonl: the needle-distance cliff (spec §7 #2)
H3  refusal-ab*: the 30/30 vs 0/21 contrast
H4  the 12 distance records whose chunk_sha256 does not resolve
H5  sweep process provenance: does "first-call cache_n=0 per cell" actually
    discriminate a fresh process from a shared one?  (the known-shared
    positive control is the test)
H6  occupancy: the 640-vs-1024 recall gap by host-cache flag
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

RES = Path(r"D:\PROJECTS\rlm-halo-framework\milestones\s2\results")


def load(name):
    out = []
    with (RES / name).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    d = [r for r in load("distance.jsonl") if r.get("status") in (None, "ok")]
    print("=" * 74)
    print("H0  distance.jsonl composition")
    print("  phases :", dict(collections.Counter(r.get("phase") for r in d)))
    print("  arms   :", dict(collections.Counter(r.get("arm") for r in d)))
    print("  qtypes :", dict(collections.Counter(r.get("question_type") for r in d)))
    print("  density:", dict(collections.Counter(r.get("density") for r in d)))
    print("  labels :", dict(collections.Counter(r.get("label") for r in d)))
    print("  sizes  :", dict(collections.Counter(r.get("size_target") for r in d)))

    print("\n" + "=" * 74)
    print("H1  false-positive rate on ABSENT questions, by size / arm / density")
    ab = [r for r in d if r.get("question_type") == "absent"]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in ab:
        k = (r.get("arm"), r.get("size_target"), r.get("density"))
        agg[k][1] += 1
        if r.get("supplied_identifier"):
            agg[k][0] += 1
    for k in sorted(agg, key=lambda x: (str(x[0]), x[1] or 0, str(x[2]))):
        fp, n = agg[k]
        print(f"  arm={str(k[0]):<22} size={str(k[1]):>6} density={str(k[2]):<16}"
              f" supplied-an-identifier {fp}/{n}")

    print("\n" + "=" * 74)
    print("H2  literal recall vs needle distance (the cliff)")
    lit = [r for r in d if r.get("question_type") == "literal"]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in lit:
        k = (r.get("arm"), r.get("size_target"), r.get("density"))
        agg[k][1] += 1
        if r.get("label") == "CORRECT":
            agg[k][0] += 1
    for k in sorted(agg, key=lambda x: (str(x[0]), x[1] or 0, str(x[2]))):
        c, n = agg[k]
        print(f"  arm={str(k[0]):<22} size={str(k[1]):>6} density={str(k[2]):<16}"
              f" CORRECT {c}/{n}")

    print("\n" + "=" * 74)
    print("H3  refusal-ab / refusal-ab-640 false positives")
    for f in ("refusal-ab.jsonl", "refusal-ab-640.jsonl"):
        rs = [r for r in load(f) if r.get("status") in (None, "ok")]
        abq = [r for r in rs if r.get("question_type") == "absent"]
        agg = collections.defaultdict(lambda: [0, 0])
        for r in abq:
            k = (r.get("arm"), r.get("size_target"))
            agg[k][1] += 1
            if r.get("supplied_identifier"):
                agg[k][0] += 1
        print(f"  {f}: {len(rs)} ok, {len(abq)} absent-questions")
        for k in sorted(agg, key=lambda x: (str(x[0]), x[1] or 0)):
            fp, n = agg[k]
            print(f"     arm={str(k[0]):<26} size={str(k[1]):>6}  supplied {fp}/{n}")

    print("\n" + "=" * 74)
    print("H4  distance records whose chunk_sha256 does not resolve")
    import hashlib
    S2 = Path(r"D:\PROJECTS\rlm-halo-framework\milestones\s2")
    on_disk = set()
    for p in S2.glob("fixtures*/**/*.chunk.txt"):
        on_disk.add(hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
        on_disk.add(hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest())
    miss = [r for r in d if (r.get("chunk_sha256") or "") not in on_disk]
    print(f"  unresolved records: {len(miss)}")
    print("  their phases/arms/cells:",
          collections.Counter((r.get("phase"), r.get("arm"),
                               r.get("cell_uid")) for r in miss))

    print("\n" + "=" * 74)
    print("H5  does 'first-call cache_n=0 per cell' discriminate fresh vs shared?")
    for f in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        rs = [r for r in load(f) if r.get("status") in (None, "ok")]
        cells = collections.OrderedDict()
        for r in rs:
            cells.setdefault(r.get("cell_id"), []).append(r)
        zeros = sum(1 for c in cells.values() if (c[0].get("tokens_cached") or 0) == 0)
        print(f"  {f}: {zeros}/{len(cells)} cells whose FIRST call had tokens_cached==0")
    print("  -> if the KNOWN-SHARED run also scores 100%, the signature proves nothing")

    print("\n" + "=" * 74)
    print("H6  occupancy: recall by chunk size and host-cache flag")
    o = load("occupancy.jsonl")
    agg = collections.defaultdict(lambda: [0, 0])
    argvs = collections.Counter()
    for r in o:
        argv = r.get("argv")
        s = " ".join(argv) if isinstance(argv, list) else str(argv)
        argvs[s] += 1
        if r.get("answer_correct") is None:
            continue
        agg[(r.get("condition"), r.get("chunk_tokens"))][1] += 1
        if r["answer_correct"]:
            agg[(r.get("condition"), r.get("chunk_tokens"))][0] += 1
    for k in sorted(agg, key=lambda x: (str(x[0]), x[1] or 0)):
        c, n = agg[k]
        print(f"  cond={str(k[0]):<16} chunk_tokens={str(k[1]):>6}  correct {c}/{n}")
    print("\n  distinct argv:")
    for a, c in argvs.most_common():
        print(f"   [{c:5d}] {a[:150]}")


if __name__ == "__main__":
    main()
