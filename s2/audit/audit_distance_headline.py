"""The DISTANCE headline, re-derived independently: false positives and recall
by size, with each false positive's CAUSE attached.

The claim under audit (ARCHITECTURE.md §4 "instruction decay", config.yaml's
window 640/stride 480): 30/30 false positives at 1,024 vs 0/45 at 640 with
45/45 literal recall. If those false positives are answers quoting an
identifier that is IN the document that was sent, no cross-request leak is
needed to explain them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2.parent))

from rlm.leakcheck import identifier_tokens  # noqa: E402
from s2.run_sweep import classify  # noqa: E402

TEXT_BY_SHA: dict[str, str] = {}
TOKEN_OWNERS: dict[str, set[str]] = defaultdict(set)
for d in sorted(S2.glob("fixtures*")):
    man = d / "manifest.json"
    if not man.exists():
        continue
    for cell in json.loads(man.read_text(encoding="utf-8"))["cells"].values():
        p = Path(cell["chunk_path"])
        if not p.exists():
            p = d / Path(cell["chunk_path"]).name
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        TEXT_BY_SHA[hashlib.sha256(t.encode()).hexdigest()] = t
        for tok in identifier_tokens(t):
            TOKEN_OWNERS[tok.lower()].add(f"{d.name}/{cell['cell_id']}")

recs = [json.loads(l) for l in (S2 / "results" / "distance.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

agg: dict[tuple, Counter] = defaultdict(Counter)
for r in recs:
    if r.get("status") != "ok" or not r.get("question_type"):
        continue
    lab = classify(r.get("raw_output", ""), question_type=r["question_type"],
                   expected=r.get("expected"), expected_kind=r.get("expected_kind"))
    key = (r.get("phase"), r.get("arm"), r.get("size_target"), r.get("density"))
    c = agg[key]
    c[f"{r['question_type']}:{lab['label']}"] += 1
    if r["question_type"] == "absent" and lab["label"] == "FALSE-POSITIVE":
        sent = TEXT_BY_SHA.get(r.get("chunk_sha256", ""), "") + "\n\n" + str(
            r.get("question", ""))
        own = {t.lower() for t in identifier_tokens(sent)}
        toks = identifier_tokens(r.get("raw_output", ""))
        if any(t.lower() in own for t in toks):
            c["fp_quoted_own"] += 1
        elif any(t.lower() in TOKEN_OWNERS for t in toks):
            c["fp_FOREIGN"] += 1
        elif toks:
            c["fp_fabricated"] += 1
        else:
            c["fp_no_identifier"] += 1
    if r["question_type"] == "absent" and r.get("leak_detected") is True:
        c["leak_flag"] += 1

print(f"{'phase':<10} {'arm':<12} {'size':>5} {'density':<14} "
      f"{'absent n':>8} {'FP':>5} {'own':>4} {'FOREIGN':>7} {'fab':>4} "
      f"{'noid':>5} | {'lit n':>5} {'lit ok':>6} {'MISS':>5}")
for key in sorted(agg, key=lambda k: (str(k[0]), str(k[1]), k[2] or 0, str(k[3]))):
    c = agg[key]
    an = sum(v for k, v in c.items() if k.startswith("absent:"))
    fp = c["absent:FALSE-POSITIVE"]
    ln = sum(v for k, v in c.items() if k.startswith("literal:"))
    lok = c["literal:CORRECT"]
    miss = c["literal:MISS"] + c["paraphrase:MISS"]
    print(f"{str(key[0]):<10} {str(key[1]):<12} {str(key[2]):>5} "
          f"{str(key[3]):<14} {an:>8} {fp:>5} {c['fp_quoted_own']:>4} "
          f"{c['fp_FOREIGN']:>7} {c['fp_fabricated']:>4} "
          f"{c['fp_no_identifier']:>5} | {ln:>5} {lok:>6} {miss:>5}")
