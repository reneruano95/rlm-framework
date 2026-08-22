"""OFFLINE: the arch_ladder failure mode, looked for in the distance grid.

milestones/s2/arch_ladder.py's contamination was ENTITY-CORRECT retrieval across requests:
the ABSENT organisation of the fixture under test was the LITERAL organisation of
a DIFFERENT fixture the same process had served, and the model returned that
other fixture's true key. If the distance grid has the same structure, its
false positives could be leakage wearing the costume of misattribution.

So: for every distance cell, is its ABSENT organisation a real, keyed
organisation anywhere else in the corpus? And if it is, did the model return
that organisation's true key?
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "milestones" / "s2"

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
        qs = cell.get("questions") or {}
        cells[f"{d.name}/{cid}"] = {
            "text": t, "low": t.lower(),
            "sha": hashlib.sha256(t.encode("utf-8")).hexdigest(),
            "lit_org": (qs.get("literal") or {}).get("entity"),
            "lit_key": (qs.get("literal") or {}).get("expected"),
            "par_org": (qs.get("paraphrase") or {}).get("entity"),
            "abs_org": (qs.get("absent") or {}).get("entity"),
        }

# index: organisation (lowercased) -> cells where it is the KEYED literal org
lit_home = defaultdict(list)
for uid, v in cells.items():
    if v["lit_org"]:
        lit_home[v["lit_org"].lower()].append(uid)

print(f"cells: {len(cells)}; distinct literal (keyed) organisations: {len(lit_home)}")

overlap = []
for uid, v in cells.items():
    a = (v["abs_org"] or "").lower()
    if not a:
        continue
    elsewhere = [u for u in lit_home.get(a, []) if u != uid]
    # also: does the ABSENT org's *text* appear in any other chunk at all?
    in_other_text = [u for u, w in cells.items() if u != uid and a and a in w["low"]]
    if elsewhere or in_other_text:
        overlap.append((uid, v["abs_org"], elsewhere, in_other_text))

print(f"\ncells whose ABSENT organisation is keyed / mentioned in ANOTHER "
      f"fixture: {len(overlap)}")
for uid, org, el, it in overlap[:30]:
    print(f"   {uid}: absent={org!r} keyed-elsewhere={el} mentioned-in={it[:4]}")

# distance grid specifically
dgrid = {k: v for k, v in cells.items() if "fixtures-distance" in k}
print(f"\ndistance-grid cells: {len(dgrid)}")
bad = [(k, v['abs_org']) for k, v in dgrid.items()
       if (v['abs_org'] or '').lower() in lit_home]
print(f"   ...whose ABSENT organisation is a keyed organisation anywhere in "
      f"the corpus: {len(bad)} {bad}")

# do the observed FALSE-POSITIVE answers match ANY other cell's key?
recs = [json.loads(l) for l in (S2 / "results" / "distance.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
sha_uid = {v["sha"]: u for u, v in cells.items()}
key_home = {v["lit_key"].lower(): u for u, v in cells.items() if v["lit_key"]}
tab = Counter()
for r in recs:
    if r.get("label") != "FALSE-POSITIVE" or r.get("status") != "ok":
        continue
    raw = (r.get("raw_output") or "").strip().lower()
    own = sha_uid.get(r.get("chunk_sha256"))
    src = None
    for k, u in key_home.items():
        if k in raw:
            src = u
            break
    if src is None:
        tab["answer matches NO fixture's planted key"] += 1
    elif src == own:
        tab["answer is THIS cell's own planted key (in-chunk misattribution)"] += 1
    else:
        tab[f"answer is ANOTHER cell's planted key: {src} (LEAK)"] += 1
print("\n=== every FALSE-POSITIVE in distance.jsonl, resolved against every "
      "planted key in the corpus ===")
for k, v in tab.most_common():
    print(f"   {v:4d}  {k}")
