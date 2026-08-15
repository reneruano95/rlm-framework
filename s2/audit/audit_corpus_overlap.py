"""OFFLINE: how discriminating is the in-chunk/out-of-chunk test, really?

If two fixtures shared identifiers, a leaked token could be mis-scored as
"in-chunk" and the CLEAN verdict would be an artefact of a blunt instrument.
This quantifies the instrument's resolution:

  * identifier-shaped tokens that occur in more than one fixture cell,
  * whether any two cells in the distance grid share a UUID or an ENT- code,
  * how many identifiers each request's chunk contributes vs the rest of the
    corpus (the ratio that sets the chance a leak is invisible),
  * serial-dispatch check: did any two distance calls overlap in wall time?
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
S2 = REPO / "s2"
sys.path.insert(0, str(REPO))
from rlm.leakcheck import identifier_tokens  # noqa: E402

cells = {}
for d in sorted(S2.glob("fixtures*")):
    man = d / "manifest.json"
    if not man.exists():
        continue
    m = json.loads(man.read_text(encoding="utf-8"))
    for cid, cell in m["cells"].items():
        p = Path(cell["chunk_path"])
        if not p.exists():
            p = d / f"{cid}.chunk.txt"
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        cells[f"{d.name}/{cid}"] = {
            "ids": {x.lower() for x in identifier_tokens(t)},
            "sha": hashlib.sha256(t.encode("utf-8")).hexdigest(),
            "uuid": (cell.get("questions", {}).get("literal") or {}).get("expected"),
        }

home = defaultdict(set)
for uid, v in cells.items():
    for t in v["ids"]:
        home[t].add(uid)

shared = {t: sorted(v) for t, v in home.items() if len(v) > 1}
print(f"fixture cells: {len(cells)}")
print(f"distinct identifier-shaped tokens: {len(home)}")
print(f"tokens occurring in MORE THAN ONE cell: {len(shared)}")
for t, v in list(shared.items())[:20]:
    print(f"   {t!r} in {v}")

dist = {k: v for k, v in cells.items() if "fixtures-distance" in k}
print(f"\ndistance-grid cells: {len(dist)}")
duuid = Counter(v["uuid"] for v in dist.values())
print(f"distinct planted UUIDs among them: {len([u for u in duuid if u])} "
      f"(duplicates: {[u for u, c in duuid.items() if c > 1]})")
ent_home = defaultdict(set)
for uid, v in dist.items():
    for t in v["ids"]:
        if t.startswith("ent-"):
            ent_home[t].add(uid)
print(f"ENT- codes shared between two distance cells: "
      f"{ {t: sorted(v) for t, v in ent_home.items() if len(v) > 1} }")

sizes = [len(v["ids"]) for v in cells.values()]
print(f"\nidentifiers per cell: min {min(sizes)} median "
      f"{sorted(sizes)[len(sizes)//2]} max {max(sizes)}")
print(f"identifiers in the whole corpus: {len(home)} -- so a leaked identifier "
      f"had a {1 - len(shared)/len(home):.3%} chance of being distinguishable "
      f"from an in-chunk one, corpus-wide")

# ------------------------------------------------------------------ #
print("\n=== SERIAL DISPATCH CHECK (R14: concurrency corrupts the leaf) ===")
recs = [json.loads(l) for l in (S2 / "results" / "distance.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
ok = [r for r in recs if r.get("status") == "ok" and r.get("ts")]
prev_end = None
overlaps = 0
for r in ok:
    end = datetime.fromisoformat(r["ts"])
    start = end.timestamp() - (r.get("wall_s") or 0) - (r.get("overhead_s") or 0)
    if prev_end is not None and start + 1.0 < prev_end:   # 1s = ts resolution
        overlaps += 1
    prev_end = end.timestamp()
print(f"calls: {len(ok)}; calls whose start precedes the previous call's end "
      f"by more than the 1 s timestamp resolution: {overlaps}")
tot = sum((r.get('wall_s') or 0) + (r.get('overhead_s') or 0) for r in ok)
span = (datetime.fromisoformat(ok[-1]['ts'])
        - datetime.fromisoformat(ok[0]['ts'])).total_seconds()
print(f"summed call wall time {tot:.0f} s over a {span:.0f} s wall span "
      f"(ratio {tot/span:.2f}; >1 would prove concurrency)")

# ------------------------------------------------------------------ #
print("\n=== arch_ladder's leaked UUIDs vs the distance corpus ===")
for u in ("48e81295-9489-33be-cc30-430d702be6c3",
          "d9f804c1-fa2b-8d32-a160-adccebcd8978",
          "7e41c11e-4a6f-131b-df64-d2385eb09ba3"):
    print(f"  {u}: in fixture corpus? {u.lower() in home} "
          f"{sorted(home.get(u.lower(), []))}")
