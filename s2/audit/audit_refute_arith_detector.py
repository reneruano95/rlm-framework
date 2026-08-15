"""Measure the SHIPPED detector (rlm.leakcheck.ChunkIndex.foreign) against the
strict oracle, on the very runs R13 is built from.

r13-truth report claims: 2/34 on r13_replay_hybrid (6%), 2/24 on the decisive
paired run (8%), 31/33 missed in the erase run, 27/39 on gemma, and that
ARCHITECTURE.md:436 calls it 'strictly stronger than R5's evidence-span check
(measured 11%)'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:/PROJECTS/rlm-halo-framework")
from rlm.leakcheck import ChunkIndex  # noqa: E402

import audit_refute_arith_oracle as O  # noqa: E402

FIX = O.S2 / "fixtures"
CHUNKS = {p.name.replace(".chunk.txt", ""): p.read_text(encoding="utf-8")
          for p in FIX.glob("*.chunk.txt")}
TOKS = {k: O.tokens_of(v) for k, v in CHUNKS.items()}
MAN = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
QTOK = {c: O.tokens_of(" ".join(q.get("question", "") for q in s.get("questions", {}).values()))
        for c, s in MAN["cells"].items()}
IDX = ChunkIndex.from_chunks(CHUNKS)


def run(name, cond=None):
    rows = [json.loads(l) for l in (O.S2 / "results" / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    if cond:
        rows = [r for r in rows if r.get("condition") == cond]
    n = strict = caught = 0
    for r in rows:
        cell = str(r.get("label") or "").split(":")[0]
        if cell not in CHUNKS:
            continue
        ans = r.get("raw_output") or ""
        n += 1
        own = TOKS[cell]
        qt = O.tokens_of(r.get("asked_about") or "") | QTOK.get(cell, set())
        hits = [t for t in O.tokens_of(ans)
                if t not in own and t not in qt
                and any(t in TOKS[c] for c in TOKS if c != cell)]
        if not hits:
            continue
        strict += 1
        v = IDX.foreign(ans, sent=CHUNKS[cell])
        if v.detected:
            caught += 1
    lab = f"{name}" + (f" [{cond}]" if cond else "")
    print(f"{lab:56s} strict_leaks={strict:3d}/{n:3d}   shipped detector caught "
          f"{caught}/{strict} = {caught / max(strict, 1):.0%}  (missed {strict - caught})")


if __name__ == "__main__":
    run("r13_replay_hybrid.jsonl")
    run("r13_replay_erase.jsonl")
    run("r13_replay_gemma_fullattn.jsonl")
    run("r13_replay_paired.jsonl", "replay_paired")
    run("r13_replay_paired.jsonl", "replay_paired__virgin_per_chunk")
