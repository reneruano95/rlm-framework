"""Re-derive arch_ladder.py's fixtures OFFLINE (pure RNG, no server, no GPU).

`build_chunk` draws the 4 stems and the 3 UUIDs BEFORE it ever calls /tokenize,
so the (entity -> key) bindings and the ABSENT entity of every (size, trial)
cell are reproducible with zero network. Only the filler length depends on the
model's tokenizer.

Two questions the probe's own JSONL cannot answer, because it records only
`present[0]`'s uuid and neither the question nor the absent entity:

  Q1  Was each wrong ABSENT answer one of the OTHER TWO keys planted in the
      SAME document (within-document misattribution -- no leak needed)?
  Q2  Or the true key of the asked-about entity in a DIFFERENT cell of the
      same run (cross-request retrieval)?
"""
from __future__ import annotations

import json
import random
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]


def cell(size: int, trial: int):
    """Exactly arch_ladder.build_chunk's draws, up to the first ntok() call."""
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust"


def main() -> None:
    rows = []
    for f in ("arch_ladder_qwen-hybrid.jsonl", "arch_ladder_gemma-fullattn.jsonl"):
        p = S2 / "results" / f
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    sizes = sorted({r["size"] for r in rows})
    trials = sorted({r["trial"] for r in rows})

    # every (entity, key) the run ever planted, and where
    planted: dict[str, list[str]] = {}
    print("=== fixtures re-derived offline ===")
    for size in sizes:
        for t in trials:
            present, absent = cell(size, t)
            print(f"  {size}/t{t}: absent={absent:<22} "
                  + " ".join(f"[{e}={u}]" for e, u in present))
            for e, u in present:
                planted.setdefault(u, []).append(f"{size}/t{t}:{e}")

    print("\n=== every non-refusal ABSENT answer, resolved ===")
    for r in rows:
        if r["qtype"] != "ABSENT" or r["cls"] != "ANSWERED_ANYWAY":
            continue
        present, absent = cell(r["size"], r["trial"])
        ans = r["answer"].strip()
        own = {u: e for e, u in present}
        where = planted.get(ans, [])
        same_doc = ans in own
        asked_key = next((u for e, u in present if e == absent), None)
        print(f"  {r['label']:14s} {r['size']}/t{r['trial']} asked about "
              f"{absent!r} -> {ans}")
        print(f"      in its OWN document? {same_doc}"
              + (f"  (as {own[ans]})" if same_doc else ""))
        print(f"      planted in: {where or 'NOWHERE in this run'}")
        for w in where:
            cellname, ent = w.split(":", 1)
            print(f"        -> {ent} in cell {cellname}"
                  f"{'   <-- SAME ENTITY as the question' if ent == absent else ''}")
        print(f"      order: this cell is call #"
              f"{sizes.index(r['size']) * len(trials) * 2 + trials.index(r['trial']) * 2 + 2}"
              f" of {len(sizes) * len(trials) * 2} in the run")

    print("\n=== does the ABSENT entity of each cell appear in an EARLIER cell? ===")
    seq = [(s, t) for s in sizes for t in trials]
    seen: dict[str, str] = {}
    for s, t in seq:
        present, absent = cell(s, t)
        prior = seen.get(absent)
        print(f"  {s}/t{t}: absent={absent:<22} previously planted in "
              f"{prior or '-'}")
        for e, _u in present:
            seen.setdefault(e, f"{s}/t{t}")


if __name__ == "__main__":
    main()
