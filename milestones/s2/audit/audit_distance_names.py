"""OFFLINE: the checks rlm.leakcheck CANNOT do, applied to distance.jsonl.

1. COINED-NAME leakage. The fixtures' organisation and person names are
   procedurally coined (`_coined`/`_org` in milestones/s2/make_sweep_fixtures.py), so
   unlike a real corpus they ARE safely usable as a proper-noun oracle -- the
   same argument milestones/s2/r13_repro.py's `_PROPER` makes. A distance answer that
   names an entity belonging only to a DIFFERENT fixture is leakage that the
   identifier-shaped detector is blind to (leakcheck.py limit 1).

2. Cache and slot telemetry: cold calls that arrived with a non-zero cache_n,
   slot_id != requested_slot, and the temporal ordering of the whole run.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "milestones" / "s2"
sys.path.insert(0, str(REPO))

RUNS = S2 / "results" / "distance.jsonl"

# A coined name is a capitalised word of >=8 chars that is NOT ordinary English.
# We do not need a dictionary: we take the names straight out of the manifests
# (needle text + question entities), which is exact.
NAME_RE = re.compile(r"\b[A-Z][a-z]{7,}\b")


def fixture_names() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(name -> {fixture uids that contain it}, uid -> names)."""
    home: dict[str, set[str]] = defaultdict(set)
    per: dict[str, set[str]] = {}
    for d in sorted(S2.glob("fixtures*")):
        man = d / "manifest.json"
        if not man.exists():
            continue
        m = json.loads(man.read_text(encoding="utf-8"))
        seed = m.get("seed")
        for cid, cell in m["cells"].items():
            p = Path(cell["chunk_path"])
            if not p.exists():
                p = d / f"{cid}.chunk.txt"
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            qs = " ".join(q.get("question", "") + " " + str(q.get("expected"))
                          for q in (cell.get("questions") or {}).values())
            names = set(NAME_RE.findall(text)) | set(NAME_RE.findall(qs))
            uid = f"{d.name}/{cid}"
            per[uid] = names
            for n in names:
                home[n].add(uid)
    return home, per


def main() -> int:
    home, per = fixture_names()
    # A name is DISCRIMINATING if it lives in exactly one fixture cell.
    unique = {n: next(iter(v)) for n, v in home.items() if len(v) == 1}
    print(f"coined names across the corpus: {len(home)}; "
          f"unique to one fixture cell: {len(unique)}")

    recs = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]

    # map cell_uid -> the fixture uid it came from, via chunk_sha256
    import hashlib
    sha_to_uid = {}
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
            sha_to_uid[hashlib.sha256(t.encode("utf-8")).hexdigest()] = f"{d.name}/{cid}"

    hits = []
    checked = 0
    for r in recs:
        if r.get("status") != "ok" or not r.get("question_type"):
            continue
        uid = sha_to_uid.get(r.get("chunk_sha256"))
        if not uid:
            continue
        checked += 1
        answer_names = set(NAME_RE.findall(r.get("raw_output") or ""))
        own = per.get(uid, set()) | set(NAME_RE.findall(r.get("question") or ""))
        for n in sorted(answer_names - own):
            if n in unique:  # lives in exactly one fixture, and not this one
                hits.append((r, n, unique[n]))

    print(f"\nanswers checked for coined-name leakage: {checked}")
    print(f"FOREIGN-NAME HITS: {len(hits)}")
    for r, n, src in hits[:40]:
        print(f"  {r['phase']}/{r['arm']} {r['cell_uid']} {r['question_type']} "
              f"{r['label']}: name {n!r} lives only in {src} | "
              f"raw={(r.get('raw_output') or '')[:90]!r}")

    # ------------------------------------------------------------------ #
    # cache / slot telemetry
    # ------------------------------------------------------------------ #
    print("\n=== CACHE / SLOT TELEMETRY (quality phases only) ===")
    q = [r for r in recs if r.get("phase") != "cache" and r.get("status") == "ok"]
    cold_nonzero = [r for r in q if r.get("cold") and (r.get("tokens_cached") or 0) > 0]
    warm = [r for r in q if r.get("cold") is False]
    print(f"ok quality calls: {len(q)}")
    print(f"cold (first call on a virgin slot) calls: {sum(1 for r in q if r.get('cold'))}")
    print(f"  ...of those, cache_n > 0 (i.e. arrived with reused KV): {len(cold_nonzero)}")
    for r in cold_nonzero[:20]:
        print(f"    {r['arm']} {r['cell_uid']} slot {r['requested_slot']}->"
              f"{r['slot_id']} cache_n={r['tokens_cached']}/{r['tokens_in']}")
    print(f"warm (re-query on the same cell's slot) calls: {len(warm)}")
    mm = [r for r in q if r.get("slot_ok") is False]
    print(f"slot_id != requested_slot: {len(mm)}")

    # slot reuse across cells
    owners = defaultdict(set)
    for r in recs:
        if r.get("phase") == "cache" or r.get("requested_slot") is None:
            continue
        owners[r["requested_slot"]].add((r.get("arm"), r.get("cell_uid")))
    shared = {s: sorted(o) for s, o in owners.items() if len(o) > 1}
    print(f"slots serving more than one (arm, cell): {len(shared)} {list(shared)[:10]}")

    # server-assigned slot_id collisions
    sid_owners = defaultdict(set)
    for r in recs:
        if r.get("slot_id") is None or r.get("phase") == "cache":
            continue
        sid_owners[r["slot_id"]].add((r.get("arm"), r.get("cell_uid")))
    sid_shared = {s: sorted(o) for s, o in sid_owners.items() if len(o) > 1}
    print(f"server-RETURNED slot_ids serving more than one (arm, cell): "
          f"{len(sid_shared)} {list(sid_shared)[:10]}")

    # ------------------------------------------------------------------ #
    # temporal ordering: one process? sequential?
    # ------------------------------------------------------------------ #
    print("\n=== TEMPORAL ORDER ===")
    ts = [(r.get("ts"), r.get("phase"), r.get("arm"), r.get("requested_slot"))
          for r in recs if r.get("ts")]
    print(f"first: {ts[0]}")
    print(f"last : {ts[-1]}")
    phases = []
    for t, ph, arm, slot in ts:
        tag = f"{ph}/{arm}"
        if not phases or phases[-1][0] != tag:
            phases.append([tag, t, t, slot, slot])
        else:
            phases[-1][2] = t
            phases[-1][4] = slot
    for tag, t0, t1, s0, s1 in phases:
        print(f"  {tag:24s} {t0} -> {t1}   slots {s0}..{s1}")
    mono = all(ts[i][0] <= ts[i + 1][0] for i in range(len(ts) - 1))
    print(f"timestamps non-decreasing across the whole file: {mono}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
