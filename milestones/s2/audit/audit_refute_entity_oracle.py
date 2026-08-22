"""Entity-level leak oracle for the LOAD-BEARING runs -- sharper than the
blast-radius report's substring test.

The arch-ladder leak signature was not merely "a uuid from another fixture";
it was "the CORRECT key for the ASKED entity, from another fixture". That is
the signature that cannot be produced by hallucination or by misattribution.

This script builds, from every fixtures*/manifest.json, the full
   (fixture, entity) -> expected key
ground truth, then classifies every identifier emitted in distance /
refusal-ab / refusal-ab-640 / sweep / sweep-run1-shared-server as:

   OWN-CHUNK-OTHER-ENTITY : key of a different entity in this record's own
                            chunk               -> R5 misattribution
   FOREIGN-ASKED-ENTITY   : correct key for the ASKED entity, planted in a
                            DIFFERENT fixture   -> arch-ladder leak signature
   FOREIGN-OTHER-ENTITY   : some other fixture's key -> leak signature (weak)
   FABRICATED             : matches nothing planted anywhere

Also reports DETECTOR POWER: could signature (2) even occur? i.e. is the asked
absent entity planted in any OTHER fixture of the same run?

Plus: per-slot FIRST-CALL cache state, and sweep process-restart evidence.

Offline, stdlib only.
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

S2 = Path(r"D:\PROJECTS\rlm-halo-framework\milestones\s2")
RES = S2 / "results"

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")


def load(name):
    out = []
    with (RES / name).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_truth():
    """sha256 -> {'label':..., 'text':..., 'entity_keys': {key: entity},
                  'questions': {...}}  from manifests + chunk files."""
    by_sha = {}
    for mf in sorted(S2.glob("fixtures*/manifest.json")):
        man = json.loads(mf.read_text(encoding="utf-8"))
        cells = man.get("cells", man)
        for cid, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            sha = cell.get("sha256")
            path = mf.parent / f"{cid}.chunk.txt"
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            if text:
                real = hashlib.sha256(text.encode("utf-8")).hexdigest()
            else:
                real = sha
            ent_keys = {}
            qs = cell.get("questions", {})
            for qt, q in qs.items():
                exp, ent = q.get("expected"), q.get("entity")
                if exp and ent:
                    ent_keys[str(exp).lower()] = ent
            # every uuid literally in the text, entity unknown unless mapped
            text_ids = set(x.lower() for x in UUID_RE.findall(text))
            by_sha.setdefault(real, {
                "label": f"{mf.parent.name}/{cid}",
                "text": text, "entity_keys": ent_keys,
                "text_ids": text_ids,
                "absent_entity": (qs.get("absent") or {}).get("entity"),
            })
            if sha and sha != real:
                by_sha.setdefault(sha, by_sha[real])
    return by_sha


def main():
    truth = build_truth()
    print(f"fixtures with manifest ground truth: {len(truth)} sha keys")

    # global: key -> list of (fixture_label, entity)
    global_keys = collections.defaultdict(list)
    global_text_ids = collections.defaultdict(list)
    for sha, t in truth.items():
        for k, e in t["entity_keys"].items():
            if (t["label"], e) not in global_keys[k]:
                global_keys[k].append((t["label"], e))
        for i in t["text_ids"]:
            if t["label"] not in global_text_ids[i]:
                global_text_ids[i].append(t["label"])

    for name in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl",
                 "sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        recs = [r for r in load(name) if r.get("status") in (None, "ok")]
        print("\n" + "=" * 74)
        print(f"{name}: {len(recs)} ok records")

        # ---- detector power: is the asked-absent entity plantable elsewhere?
        run_shas = {r.get("chunk_sha256") for r in recs} - {None}
        run_labels = {truth[s]["label"] for s in run_shas if s in truth}
        absent_ents = collections.Counter()
        for s in run_shas:
            if s in truth and truth[s]["absent_entity"]:
                absent_ents[truth[s]["absent_entity"]] += 1
        ent_to_fix = collections.defaultdict(set)
        for s in run_shas:
            if s not in truth:
                continue
            for k, e in truth[s]["entity_keys"].items():
                ent_to_fix[e].add(truth[s]["label"])
        shared_ents = {e: f for e, f in ent_to_fix.items() if len(f) > 1}
        reusable_absent = [e for e in absent_ents if e in ent_to_fix]
        print(f"  documents in run: {len(run_shas)} ({len(run_labels)} with truth)")
        print(f"  DETECTOR POWER: entities appearing in >1 fixture of this run: "
              f"{len(shared_ents)}")
        print(f"  DETECTOR POWER: asked-ABSENT entities that are PLANTED in some "
              f"fixture of this run: {len(reusable_absent)}/{len(absent_ents)}"
              f"  {'<-- foreign-asked-entity signature IS possible' if reusable_absent else '<-- signature IMPOSSIBLE by construction'}")

        counts = collections.Counter()
        examples = []
        for r in recs:
            raw = r.get("raw_output") or ""
            own = truth.get(r.get("chunk_sha256") or "")
            for ident in set(x.lower() for x in UUID_RE.findall(raw)):
                if own is None:
                    counts["own-chunk UNRESOLVED"] += 1
                    continue
                asked = None
                qt = r.get("question_type")
                if qt == "absent":
                    asked = own["absent_entity"]
                if ident in own["text_ids"]:
                    ent = own["entity_keys"].get(ident)
                    counts["OWN-CHUNK" + (f" (mapped:{'asked' if ent==asked else 'other'})"
                                          if ent else " (unmapped entity)")] += 1
                elif ident in global_keys or ident in global_text_ids:
                    src = global_keys.get(ident) or [(l, "?") for l in global_text_ids[ident]]
                    hit_asked = asked is not None and any(e == asked for _, e in src)
                    key = ("FOREIGN-ASKED-ENTITY (LEAK)" if hit_asked
                           else "FOREIGN-OTHER-ENTITY (leak-shaped)")
                    counts[key] += 1
                    if len(examples) < 15:
                        examples.append({
                            "cell": r.get("cell_uid") or r.get("cell_id"),
                            "qtype": qt, "asked": asked, "emitted": ident,
                            "src": src[:3], "own": own["label"],
                            "slot": r.get("slot_id"),
                            "tokens_cached": r.get("tokens_cached")})
                else:
                    counts["FABRICATED (nowhere on disk)"] += 1
        for k, v in counts.most_common():
            print(f"    {k:44s} {v}")
        for e in examples:
            print("    >>", e)

    # ---------------- per-slot FIRST call cache state, per run
    print("\n" + "=" * 74)
    print("FIRST-CALL-PER-SLOT cache state (host-cache cross-slot restore would show here)")
    for name in ("distance.jsonl", "refusal-ab.jsonl", "refusal-ab-640.jsonl"):
        recs = [r for r in load(name) if r.get("status") in (None, "ok")]
        seen, firsts = set(), []
        for r in recs:
            s = r.get("slot_id")
            if s is None or s in seen:
                continue
            seen.add(s)
            firsts.append(r)
        bad = [r for r in firsts if (r.get("tokens_cached") or 0) > 0]
        print(f"  {name}: {len(firsts)} slots; first call tokens_cached>0: {len(bad)}")
        for r in bad[:5]:
            print("     ", {k: r.get(k) for k in
                            ("cell_uid", "slot_id", "tokens_cached", "tokens_in", "cold")})

    # ---------------- sweep: process-restart evidence
    print("\n" + "=" * 74)
    print("SWEEP process-restart evidence (report claims one fresh process per cell)")
    for name in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
        recs = [r for r in load(name) if r.get("status") in (None, "ok")]
        by_cell = collections.OrderedDict()
        for r in recs:
            by_cell.setdefault(r.get("cell_id") or r.get("cell_uid"), []).append(r)
        print(f"\n  {name}: {len(by_cell)} cells")
        for cell, rs in by_cell.items():
            tc = [(r.get("tokens_cached") or 0) for r in rs]
            print(f"    {str(cell):22s} n={len(rs):3d} first_tokens_cached={tc[0]:6d} "
                  f"max={max(tc):6d}  slots={sorted({r.get('slot_id') for r in rs})} "
                  f"ts0={str(rs[0].get('ts'))[-14:]}")


if __name__ == "__main__":
    main()
