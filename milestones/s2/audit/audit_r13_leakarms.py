"""Offline audit of milestones/s2/results/leak-*.jsonl -- the R13 server-flag arms.

Re-scores every recorded answer with the SAME leak oracle used by
milestones/s2/r13_repro.py (foreign_strings): a token is "foreign" if it appears in the
answer, is absent from the chunk actually sent, and occurs in some OTHER chunk
of the fixture corpus.

No GPU, no network. Reads only files already on disk.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")
FIX = ROOT / "milestones" / "s2" / "fixtures"
RES = ROOT / "milestones" / "s2" / "results"

_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|ENT-\d{4,6}", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]{6,}\b")


def load_texts() -> dict[str, str]:
    return {
        p.name.replace(".chunk.txt", ""): p.read_text(encoding="utf-8")
        for p in sorted(FIX.glob("*.chunk.txt"))
    }


def foreign_strings(answer: str, own: str, texts: dict[str, str], own_cell: str):
    hits = []
    for tok in set(_ID.findall(answer)) | set(_PROPER.findall(answer)):
        if tok.lower() in own.lower():
            continue
        for cell, txt in texts.items():
            if cell != own_cell and tok.lower() in txt.lower():
                hits.append((tok, cell))
                break
    return hits


def main() -> None:
    texts = load_texts()
    files = sorted(RES.glob("leak-*.jsonl"))
    grand = []
    for f in files:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("=" * 78)
        print(f"{f.name}  n={len(rows)}  window={rows[0]['ts']} .. {rows[-1]['ts']}")
        # per-slot ordering of cells as served
        seen_by_slot: dict[int, list[str]] = {}
        leaked_any = 0
        leaked_prior = 0
        cache_n_zero = 0
        per_cell: dict[str, list[int]] = {}
        details = []
        for i, r in enumerate(rows):
            cell = r["cell_id"]
            slot = r.get("id_slot", r.get("slot_id"))
            own = texts[cell]
            ans = r.get("raw_output") or ""
            hits = foreign_strings(ans, own, texts, cell)
            prior = seen_by_slot.get(slot, [])
            hits_prior = [(t, c) for (t, c) in hits if c in prior]
            if r.get("tokens_cached", 0) == 0:
                cache_n_zero += 1
            if hits:
                leaked_any += 1
            if hits_prior:
                leaked_prior += 1
            per_cell.setdefault(cell, [0, 0])
            per_cell[cell][1] += 1
            if hits:
                per_cell[cell][0] += 1
            details.append(
                dict(
                    i=i,
                    cell=cell,
                    slot=slot,
                    q=r["question_type"],
                    trial=r.get("trial"),
                    cached=r.get("tokens_cached"),
                    label=r.get("label"),
                    hits=hits,
                    hits_prior=hits_prior,
                    ans=ans[:160].replace("\n", " "),
                )
            )
            if cell not in prior:
                seen_by_slot.setdefault(slot, []).append(cell)
        slots = sorted({d["slot"] for d in details})
        print(
            f"  slots used: {slots} | cells: {list(per_cell)} | "
            f"cache_n==0 on {cache_n_zero}/{len(rows)}"
        )
        print(f"  LEAK (foreign to any other cell): {leaked_any}/{len(rows)}")
        print(f"  LEAK (foreign to a cell THIS SLOT previously held): {leaked_prior}/{len(rows)}")
        print("  per cell (leaked/n): " + ", ".join(f"{c}={v[0]}/{v[1]}" for c, v in per_cell.items()))
        for d in details:
            if d["hits"]:
                print(
                    f"    [{d['i']:2d}] {d['cell']:14s} slot={d['slot']} {d['q']:10s} t{d['trial']} "
                    f"cached={d['cached']:>6} {d['label']:14s} foreign={d['hits'][:4]}"
                )
                print(f"         ans: {d['ans']}")
        grand.append((f.name, leaked_any, leaked_prior, len(rows), cache_n_zero))

    print("=" * 78)
    print("SUMMARY")
    for name, la, lp, n, cz in grand:
        print(f"  {name:28s} leak_any={la:3d}/{n:<3d} leak_prior={lp:3d}/{n:<3d} cache_n0={cz}/{n}")


if __name__ == "__main__":
    main()
