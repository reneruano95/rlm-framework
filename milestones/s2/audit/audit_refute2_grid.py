"""Chase the two distance.jsonl discrepancies: A-shipped@1024 FP denominator,
and the '30 and 23 MISS' figures for arms B and C."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                  r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ENT = re.compile(r"\bENT-\d{4,6}\b")

rows = [json.loads(l) for l in (ROOT / "milestones/s2/results/distance.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]

print("question_type:", dict(Counter(r.get("question_type") for r in rows)))
print("density      :", dict(Counter(r.get("density") for r in rows)))
print("phase x arm  :", dict(Counter((r.get("phase"), r.get("arm")) for r in rows)))
print()

idx = {}
for f in sorted(ROOT.glob("milestones/s2/fixtures*/**/*.chunk.txt")):
    t = f.read_text(encoding="utf-8")
    idx[hashlib.sha256(t.encode("utf-8")).hexdigest()] = (str(f.relative_to(ROOT)), t)


def ids(t):
    return set(u.lower() for u in UUID.findall(t)) | set(ENT.findall(t))


corpus = defaultdict(set)
for h, (p, t) in idx.items():
    for i in ids(t):
        corpus[i].add(h)

print(f"{'phase':10s} {'arm':12s} {'dens':9s} {'size':>5s} "
      f"{'ABS':>4s} {'FP':>4s} {'own':>4s} {'nid':>4s} {'FGN':>4s} "
      f"{'LIT':>4s} {'CORR':>5s} {'MISS':>5s}")
tot = Counter()
for key in sorted({(r.get("phase"), r.get("arm"), r.get("density"),
                    str(r.get("size_target"))) for r in rows},
                  key=lambda k: (str(k[0]), str(k[1]), str(k[2]), str(k[3]))):
    ph, arm, dens, size = key
    g = [r for r in rows if (r.get("phase"), r.get("arm"), r.get("density"),
                             str(r.get("size_target"))) == key]
    ab = [r for r in g if r.get("question_type") == "absent"]
    lit = [r for r in g if r.get("question_type") == "literal"]
    fp = [r for r in ab if r.get("label") == "FALSE-POSITIVE"]
    c = Counter()
    for r in fp:
        mine = idx.get(r.get("chunk_sha256"))
        myids = ids(mine[1]) if mine else set()
        got = ids(r.get("raw_output") or "")
        if not got:
            c["nid"] += 1
        elif got & myids:
            c["own"] += 1
        elif got & set(corpus):
            c["FGN"] += 1
        else:
            c["nid"] += 1
    print(f"{str(ph):10s} {str(arm):12s} {str(dens):9s} {size:>5s} "
          f"{len(ab):>4d} {len(fp):>4d} {c['own']:>4d} {c['nid']:>4d} {c['FGN']:>4d} "
          f"{len(lit):>4d} {sum(1 for r in lit if r.get('label')=='CORRECT'):>5d} "
          f"{sum(1 for r in lit if r.get('label')=='MISS'):>5d}")

print()
print("MISS totals by arm (all phases/sizes/question types):",
      dict(Counter(r.get("arm") for r in rows if r.get("label") == "MISS")))
print("MISS by (arm,size):",
      dict(Counter((r.get("arm"), str(r.get("size_target")))
                   for r in rows if r.get("label") == "MISS")))
print("MISS by (arm,size,qtype):",
      dict(Counter((r.get("arm"), str(r.get("size_target")), r.get("question_type"))
                   for r in rows if r.get("label") == "MISS")))

# identifiers QUOTED IN ANSWERS -- alternative basis for the 140/50/29 figures
print()
for fp_ in ("milestones/s2/results/distance.jsonl", "milestones/s2/results/refusal-ab.jsonl",
            "milestones/s2/results/refusal-ab-640.jsonl", "milestones/s2/results/sweep.jsonl"):
    rr = [json.loads(l) for l in (ROOT / fp_).read_text(encoding="utf-8").splitlines() if l.strip()]
    quoted = set()
    for r in rr:
        quoted |= ids(r.get("raw_output") or "")
    inchunks = {i for i in quoted if i in corpus}
    uniq = sum(1 for i in inchunks if len(corpus[i]) == 1)
    print(f"  {fp_:36s} distinct identifiers quoted in answers={len(quoted)} "
          f"of which in-corpus={len(inchunks)} unique={uniq}")
