"""Replicate arch_ladder.py's fixture RNG offline (no server) and ask the only
question that matters about the ANSWERED_ANYWAY rows:

    was the wrong UUID a REAL binding inside that request's own chunk
    (-> misattribution, a model behaviour)
    or did it belong to an EARLIER request in the run
    (-> cross-request leakage, a server/harness defect)?

build_chunk() draws pool + the three UUIDs from rng BEFORE any /tokenize call,
so the bindings are reproducible without the server. Only the filler length
(binary search) depends on the server, and that consumes rng afterwards.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]

RESULTS = Path(__file__).resolve().parent / "results"


def bindings(size: int, trial: int) -> tuple[list[tuple[str, str]], str]:
    """Exactly the draws arch_ladder.build_chunk makes, in the same order."""
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
    for path in sorted(RESULTS.glob("arch_ladder_*.jsonl")):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"\n=== {path.name} ===")

        # every uid this run planted, keyed by where it came from
        planted: dict[str, str] = {}
        for r in rows:
            for i, (ent, uid) in enumerate(bindings(r["size"], r["trial"])[0]):
                planted.setdefault(uid, f"{r['size']}/t{r['trial']} slot{i} ({ent})")

        for idx, r in enumerate(rows):
            if r["cls"] != "ANSWERED_ANYWAY":
                continue
            ans = r["answer"].strip()
            own, absent_ent = bindings(r["size"], r["trial"])
            own_uids = {u: e for e, u in own}
            print(f"\n  row {idx}  {r['size']}/t{r['trial']} {r['qtype']}")
            print(f"    asked about : {absent_ent}  (genuinely absent)")
            print(f"    answered    : {ans}")
            if ans in own_uids:
                print(f"    -> IN ITS OWN CHUNK, bound to '{own_uids[ans]}'"
                      f"  == MISATTRIBUTION (model behaviour)")
            elif ans in planted:
                print(f"    -> NOT in its own chunk. Belongs to {planted[ans]}"
                      f"  == CROSS-REQUEST LEAKAGE (harness/server defect)")
            else:
                print("    -> not any planted uid anywhere == FABRICATION")

        # full binding table for the rows in question
        print("\n  --- planted bindings per fixture ---")
        seen = set()
        for r in rows:
            key = (r["size"], r["trial"])
            if key in seen:
                continue
            seen.add(key)
            own, absent_ent = bindings(*key)
            print(f"  {r['size']}/t{r['trial']}  ABSENT-asks={absent_ent}")
            for ent, uid in own:
                print(f"      {uid}  <- {ent}")


if __name__ == "__main__":
    main()
