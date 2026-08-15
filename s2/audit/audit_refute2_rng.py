"""REFUTE pass 2: independently re-derive the arch_ladder fixture RNG.

No server needed: the entity pool and the three UUIDs are drawn from
random.Random(1000*size+trial) BEFORE any /tokenize call, so the fixture
identity is reproducible offline. Only the filler length (and hence the
final char count) depends on the server tokenizer.

Verifies, from the raw JSONL only:
  - which (size,trial) cell each ABSENT non-refusal answer belongs to
  - which entity that cell ASKED about (pool[3])
  - whether the answered UUID is bound to that entity in ANY cell
  - whether the answered UUID appears in the cell's OWN document
  - the serving order (call index) of donor vs receiver
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEMS = ["Prylfennwick", "Orstlornholm", "Quinfennsted", "Selkdaleridge",
         "Hurnshawfield", "Marnwickstead", "Talverstrand", "Bryndlecombe"]


def cell(size: int, trial: int):
    rng = random.Random(1000 * size + trial)
    pool = rng.sample(STEMS, 4)
    present = [(f"{s} Trust",
                "%08x-%04x-%04x-%04x-%012x" % (
                    rng.getrandbits(32), rng.getrandbits(16),
                    rng.getrandbits(16), rng.getrandbits(16),
                    rng.getrandbits(48)))
               for s in pool[:3]]
    return present, f"{pool[3]} Trust"


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    for fname in ("s2/results/arch_ladder_qwen-hybrid.jsonl",
                  "s2/results/arch_ladder_gemma-fullattn.jsonl"):
        rows = load(fname)
        print(f"\n=== {fname}  rows={len(rows)} ===")
        sizes = sorted({r["size"] for r in rows})
        trials = sorted({r["trial"] for r in rows})
        print(f"sizes={sizes} trials={trials}")

        # rebuild the full run's fixture table in SERVING ORDER (script order:
        # for size in sizes: for trial in range(trials): LITERAL then ABSENT)
        order = []           # (call_idx, size, trial, qtype)
        table = {}           # (size,trial) -> (present, absent)
        i = 0
        for s in sizes:
            for t in trials:
                table[(s, t)] = cell(s, t)
                for q in ("LITERAL", "ABSENT"):
                    i += 1
                    order.append((i, s, t, q))
        callidx = {(s, t, q): i for i, s, t, q in order}

        # sanity: does the replicated present[0] uuid match the recorded uid?
        bad = 0
        for r in rows:
            pres, _ = table[(r["size"], r["trial"])]
            if pres[0][1] != r["uid"]:
                bad += 1
                print(f"  RNG MISMATCH {r['size']}/t{r['trial']}: "
                      f"recorded {r['uid']} vs replicated {pres[0][1]}")
        print(f"present[0] uuid matches recorded uid in {len(rows)-bad}/{len(rows)} rows")

        # ownership map: uuid -> list of (size,trial,entity)
        owner = {}
        for (s, t), (pres, _) in table.items():
            for e, u in pres:
                owner.setdefault(u, []).append((s, t, e))
        # entity -> cells where planted
        planted = {}
        for (s, t), (pres, _) in table.items():
            for e, u in pres:
                planted.setdefault(e, []).append((s, t, u))

        nonref = [r for r in rows if r["qtype"] == "ABSENT" and r["cls"] != "REFUSED"]
        print(f"non-refusal ABSENT rows: {len(nonref)}")
        for r in nonref:
            s, t = r["size"], r["trial"]
            pres, absent_ent = table[(s, t)]
            ans = r["answer"].strip()
            own_uuids = {u for _, u in pres}
            in_own_doc = any(u in ans for u in own_uuids)
            rc = callidx[(s, t, "ABSENT")]
            print(f"\n  cell {s}/t{t} ABSENT cls={r['cls']} (call #{rc})")
            print(f"    asked about  : {absent_ent}")
            print(f"    doc contains : {[(e, u) for e, u in pres]}")
            print(f"    answer       : {ans}")
            print(f"    answer in own doc? {in_own_doc}")
            hit = [(u, o) for u, o in owner.items() if u in ans]
            if not hit:
                print("    answer matches NO fixture uuid anywhere (fabricated)")
            for u, o in hit:
                for (os_, ot, oe) in o:
                    print(f"    -> {u} is TRUE key of '{oe}' in cell {os_}/t{ot} "
                          f"(LITERAL call #{callidx[(os_, ot, 'LITERAL')]}, "
                          f"ABSENT call #{callidx[(os_, ot, 'ABSENT')]})")
                    print(f"       entity-correct for the ASKED entity? "
                          f"{oe == absent_ent}")
                    print(f"       donor served BEFORE receiver? "
                          f"{callidx[(os_, ot, 'ABSENT')] < rc}  "
                          f"(gap {rc - callidx[(os_, ot, 'ABSENT')]} calls)")
            # recency: all cells where the asked entity was planted
            pl = planted.get(absent_ent, [])
            print(f"    '{absent_ent}' planted in {len(pl)} cell(s): "
                  + ", ".join(f"{a}/t{b}->{c[:8]}" for a, b, c in pl))


if __name__ == "__main__":
    main()
