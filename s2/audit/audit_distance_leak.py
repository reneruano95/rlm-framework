"""Re-score s2/results/distance.jsonl -- the run that produced the ~1,000-token
horizon and the false-positive rate -- with the FULL R13 oracle (identifiers
AND the corpus's coined proper nouns), not just the shipped identifier-only
detector.

The shipped detector (rlm/leakcheck.py) is measured elsewhere in this audit to
catch 2 of 34 R13 leaks on the hybrid leaf, so "leak_detected: false" on 765
calls is a weak certificate. This script asks the stronger question directly.

Corpus = every *.chunk.txt under every s2/fixtures* directory, so a leak from
ANY document this box ever served through these probes would be caught.

Offline only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

S2 = Path(r"D:/PROJECTS/rlm-halo-framework/s2")

_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|ENT-\d{4,6}", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]{6,}\b")

CORPUS: dict[str, str] = {}
for d in sorted(S2.glob("fixtures*")):
    if not d.is_dir():
        continue
    for p in sorted(d.glob("*.chunk.txt")):
        CORPUS[f"{d.name}/{p.name.replace('.chunk.txt', '')}"] = p.read_text(encoding="utf-8")
print(f"corpus: {len(CORPUS)} chunks from {len(set(k.split('/')[0] for k in CORPUS))} fixture dirs")

LOWER = {k: v.lower() for k, v in CORPUS.items()}


def score(path: Path, own_key_of) -> None:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = leaked = 0
    tally = Counter()
    detail = []
    for r in rows:
        ans = r.get("raw_output") or ""
        if not ans:
            continue
        own = own_key_of(r)
        if own is None:
            continue
        n += 1
        own_txt = CORPUS[own].lower()
        q = (r.get("question") or "").lower()
        hits = []
        for tok in set(_ID.findall(ans)) | set(_PROPER.findall(ans)):
            t = tok.lower()
            if t in own_txt or t in q:
                continue
            where = [k for k, v in LOWER.items() if k != own and t in v]
            if where:
                hits.append((tok, where[0] if len(where) == 1 else f"{len(where)} chunks"))
        if hits:
            leaked += 1
            for h in hits:
                tally[h] += 1
            detail.append((r.get("cell_id"), r.get("arm"), r.get("question_type"),
                           r.get("requested_slot"), hits[:3], ans[:110].replace("\n", " ")))
    print("=" * 96)
    print(f"{path.name}: scored {n} answers; FULL-ORACLE leaks = {leaked} ({leaked/max(n,1):.1%})")
    for h, c in tally.most_common(12):
        print(f"    x{c:<4d} {h[0]:<38s} -> {h[1]}")
    for d in detail[:12]:
        print(f"    {d[0]} {d[1]} {d[2]} slot={d[3]} {d[4]}")
        print(f"       {d[5]}")


def dist_own(r):
    """Resolve the chunk this call ACTUALLY sent.

    Must prefer the seed-specific fixture directory; matching `fixtures/`
    first would mis-label a correct answer as a leak, because the same
    cell_id exists in several fixture sets with DIFFERENT identifiers.
    Falls back to the recorded chunk_sha256 when the name is ambiguous.
    """
    cell = str(r.get("cell_id"))
    seed = r.get("fixture_seed")
    sha = r.get("chunk_sha256")
    cands = [k for k in CORPUS if k.endswith("/" + cell)]
    if sha:
        for k in cands:
            if hashlib.sha256(CORPUS[k].encode("utf-8")).hexdigest() == sha:
                return k
    if seed is not None:
        for k in cands:
            if k.endswith(f"-s{seed}/{cell}"):
                return k
    return cands[0] if len(cands) == 1 else None


if __name__ == "__main__":
    for f in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl", "sweep.jsonl"):
        p = S2 / "results" / f
        if p.exists():
            score(p, dist_own)
