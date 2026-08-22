"""Does the existing leaf-envelope A/B data meet the probe-hygiene rules that
were adopted AFTER it was collected? Offline, zero GPU.

Checks, on `milestones/s2/results/refusal-ab-640.jsonl` and `refusal-ab.jsonl`:
  * sampling: greedy (temperature 0) or not;
  * slot discipline: one never-reused slot per (arm, cell), and id_slot ==
    requested_slot on every record;
  * R13: any foreign-identifier hits;
  * seeds and trials, so a re-run can be specified exactly.

Stdlib only. Reads; writes nothing but stdout.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    for name in ("refusal-ab-640.jsonl", "refusal-ab.jsonl"):
        p = ROOT / "milestones" / "s2" / "results" / name
        recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        print(f"\n=== {name}: {len(recs)} records ===")
        print("  temperature :", dict(Counter(str(r.get("temperature")) for r in recs)))
        print("  top_p       :", dict(Counter(str(r.get("top_p")) for r in recs)))
        print("  seed        :", dict(Counter(str(r.get("seed")) for r in recs)))
        print("  trials      :", dict(Counter(str(r.get("trial")) for r in recs)))
        print("  max_predict :", dict(Counter(str(r.get("max_predict")) for r in recs)))
        print("  slot_ok     :", dict(Counter(str(r.get("slot_ok")) for r in recs)))
        print("  leak        :", dict(Counter(str(r.get("leak_detected")) for r in recs)))
        print("  status      :", dict(Counter(str(r.get("status")) for r in recs)))
        print("  cold        :", dict(Counter(str(r.get("cold")) for r in recs)))
        print("  cache_hit   :", dict(Counter(str(r.get("cache_hit_fraction")) for r in recs)))

        slots: dict[int, set] = defaultdict(set)
        for r in recs:
            slots[int(r["requested_slot"])].add((r["arm"], r["cell_uid"]))
        shared = {s: v for s, v in slots.items() if len(v) > 1}
        print(f"  distinct slots: {len(slots)}; slots serving >1 (arm,cell): {len(shared)}")
        for s, v in list(shared.items())[:5]:
            print(f"    slot {s} -> {sorted(v)}")
        mism = [r for r in recs if str(r.get("id_slot")) != str(r.get("requested_slot"))]
        print(f"  id_slot != requested_slot: {len(mism)}")
        cells = sorted({(r["cell_id"], str(r["fixture_seed"])) for r in recs})
        print(f"  cells: {len(cells)} -> {cells[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
