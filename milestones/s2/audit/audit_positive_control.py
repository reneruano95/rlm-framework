"""OFFLINE POSITIVE CONTROL: does this audit's detector actually fire?

A CLEAN verdict on distance.jsonl is worthless unless the same code, run
unchanged over data that is KNOWN to be contaminated, comes back dirty. The
known-dirty sets are the pre-mitigation sweep (milestones/s2/results/sweep-run1-shared-
server.jsonl -- the run that produced R13) and the R13 reproducer's shared-slot
arms.

Also: re-derive 59/66 under the LOOSER criterion the original check may have
used (the emitted string occurs verbatim in its own chunk), so the audit reports
the claim's real support rather than a strawman.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "milestones" / "s2"
sys.path.insert(0, str(REPO))
from rlm.leakcheck import identifier_tokens  # noqa: E402


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha_map():
    out = {}
    for d in sorted(S2.glob("fixtures*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        for cid, cell in m["cells"].items():
            p = Path(cell["chunk_path"])
            if not p.exists():
                p = d / f"{cid}.chunk.txt"
            if not p.exists():
                continue
            t = p.read_text(encoding="utf-8")
            out[hashlib.sha256(t.encode("utf-8")).hexdigest()] = {
                "uid": f"{d.name}/{cid}", "text": t,
                "ids": {x.lower() for x in identifier_tokens(t)}}
    return out


def audit_file(path: Path, SM):
    recs = load(path)
    all_ids = {}
    for sha, v in SM.items():
        for t in v["ids"]:
            all_ids.setdefault(t, set()).add(v["uid"])
    cat = Counter()
    foreign = []
    for r in recs:
        if r.get("status") != "ok":
            continue
        raw = r.get("raw_output") or r.get("answer") or ""
        toks = {t.lower() for t in identifier_tokens(raw)}
        if not toks:
            continue
        ent = SM.get(r.get("chunk_sha256"))
        if ent is None:
            cat["chunk not resolvable"] += 1
            continue
        own = ent["ids"] | {t.lower() for t in identifier_tokens(r.get("question") or "")}
        out = toks - own
        elsewhere = {t: sorted(all_ids[t]) for t in out if t in all_ids}
        if elsewhere:
            cat["OUT-OF-CHUNK"] += 1
            foreign.append((r, elsewhere))
        elif out:
            cat["unknown string (fabricated/mangled)"] += 1
        else:
            cat["in-chunk"] += 1
    return recs, cat, foreign


def main() -> int:
    SM = sha_map()
    for name in ("sweep-run1-shared-server.jsonl", "sweep.jsonl",
                 "distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl"):
        p = S2 / "results" / name
        if not p.exists():
            print(f"\n### {name}: MISSING")
            continue
        recs, cat, foreign = audit_file(p, SM)
        print(f"\n### {name} ({len(recs)} records)")
        for k, v in cat.most_common():
            print(f"   {v:5d}  {k}")
        for r, e in foreign[:12]:
            print(f"     LEAK: {r.get('phase')}/{r.get('arm')} "
                  f"{r.get('cell_id') or r.get('cell_uid')} "
                  f"{r.get('question_type')} slot={r.get('requested_slot')} "
                  f"{(r.get('raw_output') or '')[:70]!r} -> {e}")

    # ------------------------------------------------------------------ #
    print("\n=== 59/66: the LOOSER criterion (emitted string verbatim in own chunk) ===")
    recs = load(S2 / "results" / "sweep.jsonl")
    WRONG = {"CONFABULATION", "FALSE-POSITIVE"}
    wrong = [r for r in recs if r.get("status") == "ok" and r.get("label") in WRONG]
    strict = loose = 0
    unexplained = []
    for r in wrong:
        ent = SM.get(r.get("chunk_sha256"))
        raw = (r.get("normalized") or r.get("raw_output") or "").strip()
        chunk = ent["text"] if ent else ""
        toks = {t.lower() for t in identifier_tokens(raw)}
        own = ent["ids"] if ent else set()
        if toks and toks <= (own | {t.lower() for t in identifier_tokens(r.get('question') or '')}):
            strict += 1
        # loose: ANY maximal alnum run of >=4 chars from the answer occurs in the chunk
        runs = [x for x in re.findall(r"[0-9a-zA-Z-]{4,}", raw)]
        low = chunk.lower()
        if runs and any(x.lower() in low for x in runs):
            loose += 1
        else:
            unexplained.append(r)
    print(f"wrong answers: {len(wrong)}")
    print(f"  strict (every identifier-shaped token is in the chunk sent): {strict}/{len(wrong)}")
    print(f"  loose  (some >=4-char run of the answer occurs in the chunk): {loose}/{len(wrong)}")
    print(f"  neither: {len(unexplained)}")
    for r in unexplained[:20]:
        print(f"    {r.get('cell_id')} {r.get('question_type')} {r.get('label')}: "
              f"{(r.get('normalized') or '')[:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
