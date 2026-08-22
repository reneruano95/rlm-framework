"""INDEPENDENT re-derivation of the r13-truth auditor's full-oracle leak counts.

Differences from milestones/s2/audit/audit_distance_leak.py (deliberately independent):
  * proper-noun detector is built from the GENERATOR's own syllable lists
    (make_sweep_fixtures.py:125-129) -> exact 20x17x10 name space, instead of
    the loose `[A-Z][a-z]{6,}` regex.
  * own-chunk resolution is reported per method, and UNRESOLVED rows are
    counted and printed instead of silently dropped.
  * denominators are printed three ways: file lines / rows with an answer /
    rows scored.

Offline. Reads only files already on disk.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

S2 = Path(r"D:/PROJECTS/rlm-halo-framework/milestones/s2")

_SYL_A = ("ald bry cen dov erl fask gorm hurn ilv jent korr lums merv nield "
          "orst pryl quin ravv selk tors").split()
_SYL_B = ("mora dale vint holt shaw regn brae carn dune fenn gild hask "
          "irme jost keld lorn mest").split()
_SYL_C = ("wick sted holm ridge combe thorpe gate ness field bourne").split()

NAME_RE = re.compile(
    r"\b(?:%s)(?:%s)(?:%s)\b" % ("|".join(_SYL_A), "|".join(_SYL_B), "|".join(_SYL_C)),
    re.I,
)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
ENT_RE = re.compile(r"\bENT-\d{3,8}\b", re.I)

CORPUS: dict[str, str] = {}
for d in sorted(S2.glob("fixtures*")):
    if not d.is_dir():
        continue
    for p in sorted(d.glob("*.chunk.txt")):
        CORPUS[f"{d.name}/{p.name.replace('.chunk.txt', '')}"] = p.read_text(encoding="utf-8")

LOWER = {k: v.lower() for k, v in CORPUS.items()}
BY_SHA = {hashlib.sha256(v.encode("utf-8")).hexdigest(): k for k, v in CORPUS.items()}


def tokens_of(text: str) -> set[str]:
    return {t.lower() for t in UUID_RE.findall(text)} | \
           {t.lower() for t in ENT_RE.findall(text)} | \
           {t.lower() for t in NAME_RE.findall(text)}


CORPUS_TOKENS = {k: tokens_of(v) for k, v in CORPUS.items()}
ALL_TOKENS: Counter = Counter()
for ts in CORPUS_TOKENS.values():
    for t in ts:
        ALL_TOKENS[t] += 1


def resolve(r: dict) -> tuple[str | None, str]:
    sha = r.get("chunk_sha256")
    if sha and sha in BY_SHA:
        return BY_SHA[sha], "sha"
    cell = str(r.get("cell_id") or r.get("label") or "")
    cell = cell.split(":")[0]
    cands = [k for k in CORPUS if k.endswith("/" + cell)]
    seed = r.get("fixture_seed")
    if seed is not None:
        for k in cands:
            if k.endswith(f"-s{seed}/{cell}"):
                return k, "seed"
    if len(cands) == 1:
        return cands[0], "unique-name"
    if cands:
        return None, f"ambiguous({len(cands)})"
    return None, "no-candidate"


def score(path: Path, verbose: int = 6) -> dict:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]
    n_lines = len(rows)
    n_ans = n = leaked = 0
    unresolved: Counter = Counter()
    methods: Counter = Counter()
    tally: Counter = Counter()
    detail = []
    for i, r in enumerate(rows):
        ans = r.get("raw_output") or ""
        if not ans:
            continue
        n_ans += 1
        own, how = resolve(r)
        methods[how] += 1
        if own is None:
            unresolved[f"{how}:{r.get('cell_id') or r.get('label')}"] += 1
            continue
        n += 1
        own_toks = CORPUS_TOKENS[own]
        q_toks = tokens_of((r.get("question") or "") + " " + (r.get("asked_about") or ""))
        hits = []
        for t in tokens_of(ans):
            if t in own_toks or t in q_toks:
                continue
            where = [k for k in CORPUS_TOKENS if k != own and t in CORPUS_TOKENS[k]]
            if where:
                hits.append((t, where[0] if len(where) == 1 else f"{len(where)}chunks"))
        if hits:
            leaked += 1
            for h in hits:
                tally[h[0]] += 1
            detail.append((i, r.get("cell_id") or r.get("label"), r.get("slot_id"),
                           hits[:3], ans[:100].replace("\n", " ")))
    print("=" * 100)
    print(f"{path.name}: lines={n_lines} with_answer={n_ans} scored={n} "
          f"unresolved={n_ans - n}  FOREIGN={leaked} ({leaked / max(n, 1):.1%})")
    if unresolved:
        print("   unresolved:", dict(unresolved.most_common(8)))
    print("   resolve methods:", dict(methods))
    for t, c in tally.most_common(8):
        print(f"     x{c:<4d} {t}")
    for d in detail[:verbose]:
        print(f"     row{d[0]} {d[1]} slot={d[2]} {d[3]}")
        print(f"        {d[4]}")
    return {"lines": n_lines, "with_answer": n_ans, "scored": n, "leaked": leaked}


if __name__ == "__main__":
    print(f"corpus: {len(CORPUS)} chunks from "
          f"{len({k.split('/')[0] for k in CORPUS})} dirs; "
          f"{len(ALL_TOKENS)} distinct identifier/name tokens")
    targets = sys.argv[1:] or [
        "distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl", "sweep.jsonl",
        "sweep-run1-shared-server.jsonl",
        "leak-erase.jsonl", "leak-slotiso.jsonl", "leak-nocacheidle.jsonl",
        "leak-nocram.jsonl", "leak-cram0.jsonl", "leak-ctxcp0.jsonl",
    ]
    out = {}
    for f in targets:
        p = S2 / "results" / f
        if p.exists():
            out[f] = score(p)
    print("\nSUMMARY", json.dumps(out, indent=1))
