"""REFUTE pass 3: independently re-derive arch_ladder fixture bindings and test
every non-leakage mechanism for the cross-fixture UUIDs.

Checks, in order:
 1. Is the offline RNG replication FAITHFUL? Every row records `uid` = present[0]
    uuid. If our re-derivation reproduces that field for 100% of rows, the
    replication is proven against the run's own recorded ground truth.
 2. For each non-CORRECT answer containing a uuid: is it in its OWN chunk
    (misattribution), in another fixture (cross-fixture), or nowhere (fabrication)?
 3. If cross-fixture: is the donor fixture EARLIER or LATER in run order?
    A donor that runs LATER cannot be leakage (no causal path).
 4. Is the donor binding entity-CORRECT (same entity the question asked about)?
 5. Chance-collision sanity: do any two fixtures share a uuid?
"""
from __future__ import annotations

import json
import random
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
RESULTS = S2 / "results"

STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]

UUID_CHARS = "0123456789abcdef"


def bindings(size: int, trial: int):
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust"


import re
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def main() -> None:
    files = sorted(RESULTS.glob("arch_ladder_*.jsonl"))
    for path in files:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n================ {path.name}  ({len(rows)} rows) ================")
        pol = rows[0].get("policy", "(none recorded / HEAD script)")
        print(f"  policy field: {pol}   label: {rows[0].get('label')}")

        # ---- check 1: replication faithfulness against the recorded uid field
        ok = bad = 0
        for r in rows:
            pres, _ = bindings(r["size"], r["trial"])
            if pres[0][1].lower() == r["uid"].lower():
                ok += 1
            else:
                bad += 1
                print(f"  MISMATCH row {r['size']}/t{r['trial']}: "
                      f"recorded uid={r['uid']} derived={pres[0][1]}")
        print(f"  [1] replication faithful on recorded uid: {ok}/{ok+bad}")

        # ---- build donor table with run ORDER
        order = {}          # (size,trial) -> first row index where it was served
        for i, r in enumerate(rows):
            order.setdefault((r["size"], r["trial"]), i)
        donors = {}         # uuid -> (size,trial,entity)
        for (size, trial) in order:
            pres, _ = bindings(size, trial)
            for ent, uid in pres:
                if uid.lower() in donors:
                    print(f"  !! UUID COLLISION across fixtures: {uid}")
                donors[uid.lower()] = (size, trial, ent)

        # ---- checks 2-4
        n_cross = n_own = n_fab = 0
        for i, r in enumerate(rows):
            pres, absent_ent = bindings(r["size"], r["trial"])
            own = {u.lower(): e for e, u in pres}
            asked = absent_ent if r["qtype"] == "ABSENT" else pres[0][0]
            expect = None if r["qtype"] == "ABSENT" else pres[0][1].lower()
            found = [m.lower() for m in UUID_RE.findall(r["answer"])]
            if not found:
                continue
            if expect and expect in found:
                continue           # correct
            for u in found:
                if u in own:
                    n_own += 1
                    print(f"  [OWN ] row{i:>3} {r['size']}/t{r['trial']} {r['qtype']:<7} "
                          f"asked '{asked}' -> uuid of '{own[u]}' (same chunk)")
                elif u in donors:
                    ds, dt, dent = donors[u]
                    di = order[(ds, dt)]
                    direction = "EARLIER" if di < i else ("LATER" if di > i else "SAME")
                    entity_match = "ENTITY-MATCH" if dent == asked else f"entity differs (donor={dent})"
                    n_cross += 1
                    print(f"  [XFIX] row{i:>3} {r['size']}/t{r['trial']} {r['qtype']:<7} "
                          f"asked '{asked}' -> donor {ds}/t{dt} (row {di}, {direction}) "
                          f"{entity_match}")
                else:
                    n_fab += 1
                    print(f"  [FAB ] row{i:>3} {r['size']}/t{r['trial']} {r['qtype']:<7} "
                          f"asked '{asked}' -> {u} matches nothing planted")
        print(f"  [2-4] own={n_own} cross-fixture={n_cross} fabricated={n_fab}")

        # class histogram
        from collections import Counter
        print("  cls histogram:", dict(Counter(r.get("cls") for r in rows)))
        # slots if recorded
        if "slot_served" in rows[0]:
            print("  slots served:", [r["slot_served"] for r in rows])


if __name__ == "__main__":
    main()
