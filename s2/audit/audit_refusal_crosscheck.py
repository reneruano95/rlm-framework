"""OFFLINE cross-checks that close the gaps the run's OWN detector had.

  1. Which fixture families does each run file actually touch?  (the run-time
     ChunkIndex only covered the dirs passed on that run's command line, so a
     string leaked from the 1,024 corpus into a 640 answer -- same process --
     could not have been detected at run time.)
  2. Are the 'absent' entities really absent from the chunk that was sent?
  3. The 11% catch rate itself: re-derive it from s2/results/sweep.jsonl and
     ask, of the 66 wrong answers, how many quote a string that is in their own
     chunk vs in another chunk of the sweep corpus.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")
sys.path.insert(0, str(S2.parent))
sys.path.insert(0, str(S2 / "audit"))

from rlm.envelope import normalize_ws  # noqa: E402
from rlm.leakcheck import identifier_tokens  # noqa: E402
from audit_refusal_ab import CELLS, BY_UID, load_runs, sent_cell_for  # noqa: E402


def main():
    recs = load_runs()
    ok = [r for r in recs if r.get("status") == "ok"]

    print("=== 1. which fixture family did each run touch? ===")
    fam = Counter()
    for r in ok:
        c = sent_cell_for(r)
        fam[(r["_run"], c["dir"] if c else "?")] += 1
    for k, v in sorted(fam.items()):
        print("  ", k, v)
    print("  -> the 640 run's own ChunkIndex covered only its 640 dirs, so any")
    print("     1,024-corpus string appearing in a 640 answer was UNDETECTABLE")
    print("     at run time. This audit indexes all",
          len({c['dir'] for c in CELLS.values()}), "families together.")
    print()

    print("=== 2. is the 'absent' entity really absent from the chunk sent? ===")
    bad = 0
    n = 0
    for uid, cands in sorted(BY_UID.items()):
        for c in cands:
            q = c["questions"].get("absent")
            if not q:
                continue
            n += 1
            ent = (q.get("entity") or "").strip()
            core = ent[4:] if ent.lower().startswith("the ") else ent
            if core and core.lower() in c["text"].lower():
                bad += 1
                print("   PRESENT AFTER ALL:", c["dir"], c["cell_id"], core)
    print(f"  cells with an absent-question: {n}; entity found in its own chunk: {bad}")
    print()

    print("=== 3. re-deriving the 11% span-check catch rate from sweep.jsonl ===")
    sweep = [json.loads(l) for l in (S2 / "results" / "sweep.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    sweep_ok = [r for r in sweep if r.get("status") == "ok"]
    labs = Counter(r.get("label") for r in sweep_ok)
    print(f"  sweep records: {len(sweep)} ({len(sweep_ok)} ok); labels: {dict(labs)}")

    # the sweep's cells live in s2/fixtures (seed 1)
    sweep_cells = {c["cell_id"]: c for c in CELLS.values() if c["dir"] == "fixtures"}
    print(f"  sweep fixture cells on disk: {len(sweep_cells)}")

    wrong = [r for r in sweep_ok
             if r.get("label") in ("CONFABULATION", "FALSE-POSITIVE")]
    print(f"  non-refusal wrong answers (CONFABULATION + FALSE-POSITIVE): {len(wrong)}")
    tally = Counter()
    foreign_rows = []
    for r in wrong:
        cell = sweep_cells.get(r.get("cell_id"))
        if cell is None:
            tally["no-fixture-on-disk"] += 1
            continue
        low = cell["text"].lower()
        toks = sorted(identifier_tokens(r.get("raw_output") or ""))
        if not toks:
            tally["no-identifier"] += 1
            continue
        if all(t.lower() in low for t in toks):
            tally["all-identifiers-in-own-chunk"] += 1
            continue
        outs = [t for t in toks if t.lower() not in low]
        elsewhere = {t: [(c2["dir"], c2["cell_id"]) for c2 in CELLS.values()
                         if t.lower() in c2["text"].lower()][:3] for t in outs}
        if any(elsewhere.values()):
            tally["identifier-from-ANOTHER-chunk"] += 1
            foreign_rows.append((r.get("cell_id"), r.get("question_type"),
                                 r.get("label"), outs, elsewhere))
        else:
            tally["identifier-nowhere"] += 1
            foreign_rows.append((r.get("cell_id"), r.get("question_type"),
                                 r.get("label"), outs, "NOWHERE"))
    for k, v in tally.most_common():
        print(f"    {k:32s} {v}")
    for row in foreign_rows[:30]:
        print("      ", row)
    print()
    print("  (ARCHITECTURE.md R5 claims 59/66 = 89% quote a real identifier from")
    print("   their own chunk, so the span check catches 7/66 = 11%.)")


if __name__ == "__main__":
    main()
