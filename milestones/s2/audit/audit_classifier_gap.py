"""Does the pinned classifier mislabel real refusals as FALSE-POSITIVE?

Enumerates every distinct normalized answer the leaf actually produced to an
ABSENT question that scored FALSE-POSITIVE and carries NO identifier-shaped
token -- the population where a "verbose refusal scored as a lie" would hide --
and re-tests each against the pinned `is_refusal`.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2.parent))

from rlm.leakcheck import identifier_tokens  # noqa: E402
from s2.run_sweep import REFUSAL_LEAD_RE, classify, is_refusal, normalize  # noqa: E402


def scan(name: str) -> None:
    p = S2 / "results" / name
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    texts = Counter()
    lead_but_residue = Counter()
    for r in recs:
        if r.get("status") != "ok" or r.get("question_type") != "absent":
            continue
        raw = r.get("raw_output", "")
        c = classify(raw, question_type="absent", expected=None,
                     expected_kind=r.get("expected_kind"))
        if c["label"] != "FALSE-POSITIVE":
            continue
        norm = c["normalized"]
        if not identifier_tokens(raw):
            texts[norm[:150]] += 1
        if REFUSAL_LEAD_RE.match(norm) and not is_refusal(norm):
            lead_but_residue[norm[:110]] += 1
    print(f"\n=== {name} ===")
    print(f"FALSE-POSITIVE answers carrying NO identifier "
          f"({sum(texts.values())} calls, {len(texts)} distinct):")
    for t, n in texts.most_common(20):
        print(f"  [{n}x] {t!r}")
    print(f"answers that START with a pinned refusal phrase but were rejected "
          f"by the residue rule ({sum(lead_but_residue.values())} calls):")
    for t, n in lead_but_residue.most_common(10):
        print(f"  [{n}x] {t!r}")


for f in ("sweep.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
          "distance.jsonl"):
    scan(f)
