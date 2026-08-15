"""FALSE-POSITIVE, re-scored with EACH probe's own scorer, then decomposed.

Fixes two things `audit_probe_defects.py` got wrong on the A/B files: envelope
arms must be reduced by `s2.run_refusal_ab.score` before the sweep's classifier
sees them, and a FALSE-POSITIVE deserves a cause, not just a count.

Buckets, per file:
  quoted-own      the answer carries an identifier that IS in the document sent
  foreign         carries an identifier absent from the document sent and
                  present in ANOTHER fixture of the whole s2 corpus (a leak)
  fabricated      carries an identifier present in no fixture at all
  prose-refusal   no identifier, and the text is a refusal the pinned
                  `is_refusal` phrase list does not match (a SCORER error)
  fragment        no identifier, under 12 chars, not a refusal (degenerate)
  other
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S2.parent))

from rlm.leakcheck import identifier_tokens  # noqa: E402
from s2.run_refusal_ab import score as ab_score  # noqa: E402
from s2.run_sweep import classify, normalize  # noqa: E402

PROSE_REFUSAL = re.compile(
    r"(does not contain|contains no|does not (mention|include|list|state|specify)"
    r"|is not (provided|present|specified|listed|mentioned|given|included)"
    r"|not provided in|no (information|mention|record|entry|such)"
    r"|i cannot answer|cannot be determined|unable to)", re.I)

TEXT_BY_SHA: dict[str, str] = {}
TOKEN_OWNERS: dict[str, set[str]] = defaultdict(set)


def load_corpus() -> None:
    for d in sorted(S2.glob("fixtures*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        for cell in m["cells"].values():
            p = Path(cell["chunk_path"])
            if not p.exists():
                p = d / Path(cell["chunk_path"]).name
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            TEXT_BY_SHA[hashlib.sha256(text.encode()).hexdigest()] = text
            for t in identifier_tokens(text):
                TOKEN_OWNERS[t.lower()].add(f"{d.name}/{cell['cell_id']}")


def label_of(rec: dict, envelope_aware: bool) -> dict:
    if envelope_aware:
        return ab_score(rec.get("raw_output", ""),
                        envelope=bool(rec.get("envelope")),
                        chunk=TEXT_BY_SHA.get(rec.get("chunk_sha256", "")),
                        question_type=rec["question_type"],
                        expected=rec.get("expected"),
                        expected_kind=rec.get("expected_kind"))
    return classify(rec.get("raw_output", ""), question_type=rec["question_type"],
                    expected=rec.get("expected"),
                    expected_kind=rec.get("expected_kind"))


def scan(name: str, *, envelope_aware: bool) -> None:
    p = S2 / "results" / name
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    buckets = Counter()
    labels = Counter()
    examples: dict[str, str] = {}
    n_absent = 0
    for r in recs:
        if r.get("status") != "ok" or not r.get("question_type"):
            continue
        lab = label_of(r, envelope_aware)
        labels[lab["label"]] += 1
        if r["question_type"] != "absent":
            continue
        n_absent += 1
        if lab["label"] != "FALSE-POSITIVE":
            continue
        raw = r.get("raw_output", "")
        sent = TEXT_BY_SHA.get(r.get("chunk_sha256", ""), "") + "\n\n" + str(
            r.get("question", ""))
        own = {t.lower() for t in identifier_tokens(sent)}
        toks = identifier_tokens(raw)
        if any(t.lower() in own for t in toks):
            b = "quoted-own"
        elif any(t.lower() in TOKEN_OWNERS for t in toks):
            b = "FOREIGN (leak)"
        elif toks:
            b = "fabricated"
        elif PROSE_REFUSAL.search(raw):
            b = "prose-refusal (SCORER ERROR)"
        elif len(normalize(raw)) < 12:
            b = "fragment/degenerate"
        else:
            b = "other"
        buckets[b] += 1
        examples.setdefault(b, " ".join(raw.split())[:120])
    fp = sum(buckets.values())
    print(f"\n=== {name}  (envelope-aware scorer: {envelope_aware}) ===")
    print(f"  all labels: {dict(labels)}")
    print(f"  ABSENT calls: {n_absent};  FALSE-POSITIVE: {fp} "
          f"({fp / n_absent:.0%})" if n_absent else "")
    for b, n in buckets.most_common():
        print(f"    {b:32s} {n:4d}  ({n / fp:.0%})   e.g. {examples[b]!r}")


load_corpus()
print(f"corpus: {len(TEXT_BY_SHA)} chunks, {len(TOKEN_OWNERS)} identifiers")
scan("sweep.jsonl", envelope_aware=False)
scan("refusal-ab.jsonl", envelope_aware=True)
scan("refusal-ab-640.jsonl", envelope_aware=True)
scan("distance.jsonl", envelope_aware=False)
