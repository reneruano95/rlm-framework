"""Refinement of the entity-binding detector, offline, from data already on disk.

audit_envelope_entity_binding.py showed the crude rule ("some evidence span
contains the question's entity") catches 29/29 wrong answers but rejects 55/96
correct ones. This script decomposes those 55 to find out whether the rule is
wrong or the PROMPT is, and checks the rule is satisfiable at all:

  * are the 55 rejections abstentions (no span exists to check -- the rule must
    exempt them), or substantive answers whose spans omit the entity?
  * for every (cell, question), does the question's entity even OCCUR in the
    chunk? If it does, a span containing it exists and the requirement is
    satisfiable; if it does not, the correct behaviour is to abstain and the
    scaffold could have known that without the model.

Stdlib only. Reads; writes nothing but stdout.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_WS = re.compile(r"\s+")


def norm(t: str) -> str:
    return _WS.sub(" ", t or "")


def ent_key(e: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", norm(e).strip(), flags=re.IGNORECASE)


def main() -> int:
    fixtures: dict[tuple[str, str], dict] = {}
    for d in sorted(ROOT.joinpath("s2").glob("fixtures-refusal*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        for cid, cell in m.get("cells", {}).items():
            cell = dict(cell)
            ch = d / f"{cid}.chunk.txt"
            cell["_chunk"] = ch.read_text(encoding="utf-8") if ch.exists() else ""
            fixtures[(cid, str(m.get("seed")))] = cell

    # ---- satisfiability: is the question's entity in the chunk? -------------
    print("-- is the question's entity present in the chunk? (fixture ground truth) --")
    tally: dict[str, dict[bool, int]] = defaultdict(lambda: {True: 0, False: 0})
    for (cid, seed), cell in sorted(fixtures.items()):
        if not cell["_chunk"]:
            continue
        hay = norm(cell["_chunk"])
        for qt, q in cell.get("questions", {}).items():
            tally[qt][ent_key(q.get("entity", "")) in hay] += 1
    for qt in sorted(tally):
        t = tally[qt]
        print(f"  {qt:<12} present={t[True]:<4} absent={t[False]:<4}")

    # ---- decompose the 55 false rejections ---------------------------------
    path = ROOT / "milestones" / "s2" / "results" / "refusal-ab-640.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    print("\n-- CORRECT parsed envelope replies rejected by 'entity in some span' --")
    buckets: dict[tuple, int] = defaultdict(int)
    samples: dict[tuple, list[str]] = defaultdict(list)
    n_ok = 0
    strict: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in recs:
        if str(r.get("envelope")) != "True" or str(r.get("envelope_ok")) != "True":
            continue
        cell = fixtures.get((r["cell_id"], str(r["fixture_seed"])))
        if not cell:
            continue
        q = cell["questions"].get(r["question_type"])
        ent = ent_key(q.get("entity", ""))
        spans = r.get("evidence") or []
        abstain = bool(r.get("abstain"))
        hit = any(ent and ent in norm(s) for s in spans)
        n_ok += 1

        # THE PROPOSED RULE, stated properly: it applies only to a SUBSTANTIVE
        # answer (abstain False and a non-empty answer). An abstention has no
        # span to bind and is exempt.
        substantive = (not abstain) and bool((r.get("answer") or r.get("reduced_text") or "").strip())
        verdict = "EXEMPT(abstain)" if not substantive else ("PASS" if hit else "REJECT")
        strict[r["question_type"]][f"{r['label']}/{verdict}"] += 1

        if r["label"] == "CORRECT" and not hit:
            key = (r["question_type"], abstain, len(spans))
            buckets[key] += 1
            if len(samples[key]) < 2:
                samples[key].append(json.dumps(spans)[:180])

    print(f"  (over {n_ok} parsed envelope replies)")
    for k in sorted(buckets, key=str):
        print(f"  qtype={k[0]:<12} abstain={str(k[1]):<6} n_spans={k[2]:<3} count={buckets[k]}")
        for s in samples[k]:
            print(f"      e.g. {s}")

    print("\n-- the rule as it should be stated (abstentions exempt) --")
    for qt in sorted(strict):
        print(f"  {qt}: {dict(strict[qt])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
