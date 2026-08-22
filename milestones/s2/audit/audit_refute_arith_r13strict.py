"""Score the r13_replay_* files with MY OWN strict oracle and compare against
(a) the shipped `verdict` field and (b) the numbers R13.md prints.

Own chunk = milestones/s2/fixtures/<cell>  (the replay ran --replay-fixtures milestones/s2/fixtures,
R13.md:251-253).  Strict oracle = coined-name/UUID/ENT tokens that are absent
from the sent chunk AND absent from the question/asked_about, and present in
some other chunk of milestones/s2/fixtures.

Also reports the LOOSE variant (no question exclusion) so the gap between
R13.md's table and the shipped scorer can be measured.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import audit_refute_arith_oracle as O

FIX = O.S2 / "fixtures"
CHUNKS = {p.name.replace(".chunk.txt", ""): p.read_text(encoding="utf-8")
          for p in FIX.glob("*.chunk.txt")}
TOKS = {k: O.tokens_of(v) for k, v in CHUNKS.items()}

# the question text is not stored in the r13 replay rows; reconstruct the
# question-token set from `asked_about` plus the manifest question for the cell
MAN = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
QTOK = {}
for cell, spec in MAN["cells"].items():
    s = " ".join(q.get("question", "") for q in spec.get("questions", {}).values())
    QTOK[cell] = O.tokens_of(s)


def fisher(a, b, c, d):
    def logC(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    po = math.exp(logC(r1, a) + logC(r2, c1 - a) - logC(n, c1))
    t = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        p = math.exp(logC(r1, x) + logC(r2, c1 - x) - logC(n, c1))
        if p <= po * (1 + 1e-9):
            t += p
    return t


def score(name, group=None):
    rows = [json.loads(l) for l in (O.S2 / "results" / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = defaultdict(lambda: {"n": 0, "strict": 0, "loose": 0, "verdict": 0})
    ex_strict = []
    for r in rows:
        cell = str(r.get("label") or "").split(":")[0]
        if cell not in TOKS:
            continue
        ans = r.get("raw_output") or ""
        k = r.get(group) if group else "ALL"
        o = out[k]
        o["n"] += 1
        if r.get("verdict") == "FOREIGN":
            o["verdict"] += 1
        own = TOKS[cell]
        qt = O.tokens_of(r.get("asked_about") or "") | QTOK.get(cell, set())
        at = O.tokens_of(ans)
        loose = [t for t in at if t not in own and any(t in TOKS[c] for c in TOKS if c != cell)]
        strict = [t for t in loose if t not in qt]
        if loose:
            o["loose"] += 1
        if strict:
            o["strict"] += 1
            ex_strict.append((cell, r.get("condition"), strict[:3]))
    print(f"--- {name}" + (f"  by {group}" if group else ""))
    for k, o in out.items():
        print(f"    {str(k):40s} n={o['n']:4d}  shipped_verdict={o['verdict']:3d}  "
              f"MY_loose={o['loose']:3d}  MY_strict={o['strict']:3d}")
    return out


if __name__ == "__main__":
    score("r13_replay_hybrid.jsonl")
    score("r13_replay_erase.jsonl")
    score("r13_replay_gemma_fullattn.jsonl")
    p = score("r13_replay_paired.jsonl", group="condition")
    ks = list(p)
    if len(ks) == 2:
        a = p[ks[0]]["strict"]; b = p[ks[0]]["n"] - a
        c = p[ks[1]]["strict"]; d = p[ks[1]]["n"] - c
        print(f"    STRICT 2x2 [[{a},{b}],[{c},{d}]] Fisher p = {fisher(a, b, c, d):.3e}")
        a = p[ks[0]]["verdict"]; b = p[ks[0]]["n"] - a
        c = p[ks[1]]["verdict"]; d = p[ks[1]]["n"] - c
        print(f"    SHIPPED 2x2 [[{a},{b}],[{c},{d}]] Fisher p = {fisher(a, b, c, d):.3e}")
    score("r13_twoprompt_matrix.jsonl")
    score("r13_twoprompt_sizesweep.jsonl")
