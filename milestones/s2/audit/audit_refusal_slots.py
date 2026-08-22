"""OFFLINE: slot hygiene of the refusal A/B inside its single server process.

  * does every record's chunk_sha256 match the fixture cell I compared it to?
  * did any slot carry traffic from more than one cell_uid?
  * the 690 log tasks vs the 639 recorded calls: which slots do the ~51
    unrecorded tasks land on, and do those slots overlap the A/B's?
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\milestones\s2")
sys.path.insert(0, str(S2.parent))
sys.path.insert(0, str(S2 / "audit"))
from audit_refusal_ab import BY_UID, load_runs, sent_cell_for  # noqa: E402

LOG = Path(r"D:\PROJECTS\rlm-halo-framework\traces\logs\leaf-server.err.log")
RE_LAUNCH = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+) \|")


def main():
    recs = load_runs()
    ok = [r for r in recs if r.get("status") == "ok"]

    # uid ambiguity
    amb = {u: [c["dir"] for c in v] for u, v in BY_UID.items() if len(v) > 1}
    print(f"uids resolving to >1 fixture dir: {len(amb)} {amb}")

    # chunk_sha256 agreement -- proof the reference text is the text that was sent
    import hashlib
    good = bad = 0
    for r in ok:
        c = sent_cell_for(r)
        h = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        if h == r.get("chunk_sha256"):
            good += 1
        else:
            bad += 1
            if bad < 5:
                print("  SHA MISMATCH", r["_file"], r["_line"], r.get("cell_uid"),
                      c["dir"], h[:12], r.get("chunk_sha256", "")[:12])
    print(f"records whose chunk_sha256 == sha256(fixture text I compared against): "
          f"{good} ok, {bad} mismatched")

    # one cell per slot?
    per_slot = defaultdict(set)
    for r in ok:
        per_slot[r.get("slot_id")].add((r["_run"], r.get("cell_uid"), r["arm"]))
    multi = {s: v for s, v in per_slot.items() if len(v) > 1}
    print(f"slots carrying more than one (run, cell, arm): {len(multi)}")
    for s, v in list(multi.items())[:10]:
        print("   slot", s, v)
    print(f"distinct slots used by the A/B: {len(per_slot)} "
          f"(min {min(per_slot)}, max {max(per_slot)})")

    # log-side slots
    lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    launches = [(int(m.group(1)), int(m.group(2))) for l in lines
                if (m := RE_LAUNCH.search(l))]
    log_slots = Counter(s for s, _ in launches)
    print(f"log: {len(launches)} launch_slot events over {len(log_slots)} slots")
    ab_slots = set(per_slot)
    extra = {s: n for s, n in log_slots.items() if s not in ab_slots}
    print(f"slots the log used that the A/B never requested: {len(extra)} "
          f"-> {sorted(extra)[:40]}")
    print(f"total tasks on those slots: {sum(extra.values())}")
    shared = {s: log_slots[s] - sum(1 for r in ok if r.get('slot_id') == s)
              for s in ab_slots}
    over = {s: d for s, d in shared.items() if d}
    print(f"A/B slots with MORE log tasks than recorded calls: {len(over)} {dict(list(over.items())[:20])}")


if __name__ == "__main__":
    main()
