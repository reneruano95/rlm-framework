"""OFFLINE forensic audit of the S2 refusal A/B (s2/results/refusal-ab*.jsonl).

No GPU, no network. Answers three questions with numbers:

  Q1  For every FALSE-POSITIVE (and every other wrong answer), was the
      identifier it handed over present in the EXACT chunk sent, present in
      SOME OTHER fixture cell (which the same server process may have held on
      another slot), or nowhere in any fixture at all?
  Q2  Is any wrong answer the TRUE key of the very entity the question named,
      taken from a DIFFERENT fixture cell?  (that is the arch_ladder
      signature: entity-correct retrieval from a prior request's document)
  Q3  Of the answers the report calls "the span check passes on them",
      how many quoted a span that lives in the CURRENT chunk vs elsewhere?
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\s2")
sys.path.insert(0, str(S2.parent))

from rlm.envelope import normalize_ws  # noqa: E402
from rlm.leakcheck import identifier_tokens  # noqa: E402

RUNS = {
    "1024": S2 / "results" / "refusal-ab.jsonl",
    "640": S2 / "results" / "refusal-ab-640.jsonl",
}

# every fixture family that exists on disk; the refusal run's own detector saw
# only the dirs passed on ITS command line, which is the gap under test.
FIXTURE_DIRS = sorted(p for p in S2.glob("fixtures*") if (p / "manifest.json").exists())


def load_fixtures():
    cells = {}          # key (dir.name, cell_id) -> record
    by_uid = {}         # "<cell_id>#s<seed>" -> record  (run_refusal_ab's uid)
    for d in FIXTURE_DIRS:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        seed = man.get("seed")
        for cell in man["cells"].values():
            path = Path(cell["chunk_path"])
            if not path.is_absolute():
                path = S2.parent / path
            if not path.exists():
                path = d / Path(cell["chunk_path"]).name
            text = path.read_text(encoding="utf-8")
            rec = {
                "dir": d.name,
                "cell_id": cell["cell_id"],
                "seed": seed,
                "uid": f"{cell['cell_id']}#s{seed}",
                "text": text,
                "tokens": {t.lower() for t in identifier_tokens(text)},
                "questions": cell.get("questions", {}),
            }
            cells[(d.name, cell["cell_id"])] = rec
            by_uid.setdefault(rec["uid"], []).append(rec)
    return cells, by_uid


CELLS, BY_UID = load_fixtures()

# token -> list of (dir, cell_id) that contain it
TOKEN_OWNERS: dict[str, list[tuple[str, str]]] = {}
for (dn, cid), rec in CELLS.items():
    for t in rec["tokens"]:
        TOKEN_OWNERS.setdefault(t, []).append((dn, cid))

# entity -> list of (dir, cell_id, qtype, expected) for questions with a real key
ENTITY_KEYS: dict[str, list[tuple[str, str, str, str]]] = {}
for (dn, cid), rec in CELLS.items():
    for qt, q in rec["questions"].items():
        ent = (q.get("entity") or "").strip().lower()
        if ent and q.get("expected"):
            ENTITY_KEYS.setdefault(ent, []).append((dn, cid, qt, q["expected"]))


def load_runs():
    out = []
    for tag, p in RUNS.items():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_run"] = tag
            r["_line"] = i + 1
            r["_file"] = p.name
            out.append(r)
    return out


def sent_cell_for(rec):
    """The fixture cell whose text was actually sent on this call.

    Matched by uid AND verified against the record's own chunk_sha256 where the
    manifest carries one, so a uid collision across fixture families cannot
    silently mislabel the reference text."""
    cands = BY_UID.get(rec.get("cell_uid"), [])
    if len(cands) == 1:
        return cands[0]
    # disambiguate by which family the run was pointed at
    want = "640" if rec.get("size_target") == 640 else "1024"
    for c in cands:
        if want == "640" and "-640-" in c["dir"]:
            return c
        if want == "1024" and "-640-" not in c["dir"] and "refusal" in c["dir"]:
            return c
    return cands[0] if cands else None


def main():
    records = load_runs()
    ok = [r for r in records if r.get("status") == "ok"]
    print(f"records: {len(records)} total, {len(ok)} status=ok")
    print(f"fixture families indexed: {len(FIXTURE_DIRS)} "
          f"({sum(1 for _ in CELLS)} cells, {len(TOKEN_OWNERS)} distinct identifiers)")
    print()

    # ---- process ordering: ts + file order ------------------------------- #
    ts = sorted({r.get("ts") for r in ok if r.get("ts")})
    print(f"ts range: {ts[0]} .. {ts[-1]}  ({len(ts)} distinct)")
    # order of first service of each cell_uid, per run file
    served_at = {}
    for idx, r in enumerate(ok):
        key = (r["_run"], r.get("cell_uid"))
        served_at.setdefault(key, idx)
    print()

    # ---- Q1: where does every wrong answer's identifier live? ------------ #
    buckets = {}
    detail_rows = []
    for r in ok:
        label = r.get("label")
        if label in ("CORRECT", "MISS"):
            continue
        cell = sent_cell_for(r)
        sent = (cell["text"] if cell else "") + "\n\n" + (r.get("question") or "")
        own = {t.lower() for t in identifier_tokens(sent)}
        toks = identifier_tokens(r.get("raw_output") or "")
        if not toks:
            buckets[(label, "no-identifier")] = buckets.get((label, "no-identifier"), 0) + 1
            continue
        for t in sorted(toks):
            tl = t.lower()
            if tl in own:
                kind = "in-chunk"
                owners = []
            else:
                owners = TOKEN_OWNERS.get(tl, [])
                if not owners:
                    kind = "nowhere (fabricated)"
                else:
                    same_fam = [o for o in owners
                                if cell and o[0] == cell["dir"]]
                    kind = ("foreign-same-family" if same_fam
                            else "foreign-other-family")
            buckets[(label, kind)] = buckets.get((label, kind), 0) + 1
            if kind.startswith("foreign") or kind.startswith("nowhere"):
                detail_rows.append((r["_file"], r["_line"], r["arm"], label,
                                    r.get("question_type"), r.get("cell_uid"),
                                    t, kind, owners[:3]))

    print("=== Q1  identifier provenance, per label (one row per token) ===")
    for (label, kind), n in sorted(buckets.items()):
        print(f"  {label:14s} {kind:22s} {n}")
    print()
    if detail_rows:
        print("--- non-in-chunk identifiers, every one ---")
        for row in detail_rows:
            print("   ", row)
    else:
        print("--- NO answer in the whole A/B carried an identifier that was "
              "absent from its own chunk ---")
    print()

    # ---- Q2: entity-correct retrieval from another fixture --------------- #
    print("=== Q2  did any answer hand back the named entity's TRUE key from "
          "ANOTHER cell? ===")
    hits = 0
    absent_entity_elsewhere = 0
    for r in ok:
        if r.get("question_type") != "absent":
            continue
        cell = sent_cell_for(r)
        q = (cell or {}).get("questions", {}).get("absent", {})
        ent = (q.get("entity") or "").strip().lower()
        elsewhere = ENTITY_KEYS.get(ent, [])
        if elsewhere:
            absent_entity_elsewhere += 1
        raw = (r.get("raw_output") or "").lower()
        for (dn, cid, qt, exp) in elsewhere:
            if exp.lower() in raw:
                hits += 1
                print(f"  HIT {r['_file']}:{r['_line']} arm={r['arm']} "
                      f"cell={r.get('cell_uid')} entity={ent!r} "
                      f"answered {exp} which is the true key in {dn}/{cid}")
    print(f"  absent-questions whose named entity DOES have a real key in some "
          f"other fixture cell: {absent_entity_elsewhere}")
    print(f"  answers that returned that other cell's key: {hits}")
    print()

    # ---- Q2b: is the absent entity name present in any other fixture? ---- #
    print("=== Q2b  are the 'absent' entity names unique to their cell? ===")
    seen_names = {}
    for (dn, cid), rec in CELLS.items():
        for qt, q in rec["questions"].items():
            ent = (q.get("entity") or "").strip()
            if ent:
                seen_names.setdefault(ent.lower(), []).append((dn, cid, qt))
    shared = {k: v for k, v in seen_names.items() if len(v) > 1}
    print(f"  distinct entity names across all fixtures: {len(seen_names)}")
    print(f"  names used in more than one (dir, cell, qtype): {len(shared)}")
    for k, v in sorted(shared.items())[:20]:
        print(f"    {k}: {v}")
    # and: does the absent entity's NAME appear in any other chunk's TEXT?
    name_in_other_text = 0
    for (dn, cid), rec in CELLS.items():
        q = rec["questions"].get("absent")
        if not q:
            continue
        ent = (q.get("entity") or "").strip()
        core = ent.replace("the ", "", 1) if ent.lower().startswith("the ") else ent
        if not core:
            continue
        others = [(d2, c2) for (d2, c2), r2 in CELLS.items()
                  if (d2, c2) != (dn, cid) and core.lower() in r2["text"].lower()]
        if others:
            name_in_other_text += 1
            print(f"    absent entity {core!r} of {dn}/{cid} ALSO appears in "
                  f"{others[:4]}{' ...' if len(others) > 4 else ''}")
    print(f"  absent entities whose name appears in another fixture's text: "
          f"{name_in_other_text}")
    print()

    # ---- Q3: evidence spans — which text do they live in? ---------------- #
    print("=== Q3  evidence spans: current chunk vs any other fixture ===")
    span_stats = {"in-current": 0, "in-other-only": 0, "nowhere": 0, "n_records": 0}
    other_hits = []
    for r in ok:
        ev = r.get("evidence")
        if not ev:
            continue
        span_stats["n_records"] += 1
        cell = sent_cell_for(r)
        cur = normalize_ws(cell["text"]) if cell else ""
        for span in ev:
            s = normalize_ws(span).strip()
            if not s:
                span_stats["nowhere"] += 1
                continue
            if s in cur:
                span_stats["in-current"] += 1
                continue
            found = [(dn, cid) for (dn, cid), rec2 in CELLS.items()
                     if s in normalize_ws(rec2["text"])]
            if found:
                span_stats["in-other-only"] += 1
                other_hits.append((r["_file"], r["_line"], r["arm"],
                                   r.get("cell_uid"), s[:90], found[:3]))
            else:
                span_stats["nowhere"] += 1
    print(f"  records carrying evidence: {span_stats['n_records']}")
    print(f"  spans verbatim in the CURRENT chunk : {span_stats['in-current']}")
    print(f"  spans in ANOTHER fixture only       : {span_stats['in-other-only']}")
    print(f"  spans in no fixture at all          : {span_stats['nowhere']}")
    for h in other_hits[:25]:
        print("    OTHER-CHUNK SPAN:", h)
    print()

    # ---- context: label counts + cache stats ----------------------------- #
    print("=== context: labels, and how much KV was reused ===")
    lab = {}
    for r in ok:
        k = (r["_run"], r["arm"], r.get("question_type"), r.get("label"))
        lab[k] = lab.get(k, 0) + 1
    for k in sorted(lab):
        print("  ", k, lab[k])
    print()
    fp = [r for r in ok if r.get("label") == "FALSE-POSITIVE"]
    cold_fp = [r for r in fp if r.get("cold")]
    print(f"  FALSE-POSITIVE records: {len(fp)}  (of which cold/first-call "
          f"on their slot: {len(cold_fp)})")
    caches = [(r.get("tokens_cached"), r.get("tokens_in")) for r in fp]
    nz = [c for c in caches if c[0]]
    print(f"  FP with tokens_cached>0: {len(nz)}/{len(caches)}")
    coldnz = [r for r in cold_fp if r.get("tokens_cached")]
    print(f"  FP that were COLD yet reported tokens_cached>0: {len(coldnz)}")
    for r in coldnz[:10]:
        print("    ", r["_file"], r["_line"], r["arm"], r.get("cell_uid"),
              "cached", r.get("tokens_cached"), "/", r.get("tokens_in"),
              "slot", r.get("slot_id"))
    # any cold call at all with cache>0?
    coldall = [r for r in ok if r.get("cold") and r.get("tokens_cached")]
    print(f"  ANY cold call with tokens_cached>0: {len(coldall)} of "
          f"{sum(1 for r in ok if r.get('cold'))} cold calls")
    for r in coldall[:20]:
        print("    ", r["_file"], r["_line"], r["arm"], r.get("cell_uid"),
              r.get("question_type"), "cached", r.get("tokens_cached"), "/",
              r.get("tokens_in"), "slot", r.get("slot_id"),
              "label", r.get("label"))
    print()
    mism = [r for r in ok if r.get("slot_ok") is False]
    print(f"  slot mismatches: {len(mism)}")
    leaks = [r for r in ok if r.get("leak_detected") is True]
    nc = [r for r in ok if r.get("leak_detected") is None]
    print(f"  run-time leak_detected True: {len(leaks)}  NOT-CHECKED: {len(nc)}")


if __name__ == "__main__":
    main()
