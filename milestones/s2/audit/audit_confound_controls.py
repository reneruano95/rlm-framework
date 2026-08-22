"""OFFLINE: kill (or confirm) the two live alternatives to the distance story.

ALT-1  Slot index / pool occupancy, not chunk size. Within every run the 640
       cells are served BEFORE the 1,024 cells (load_cells sorts by size), so
       size is confounded with slot index and with elapsed time. If leakage or
       state accumulates with slot index, that alone reproduces "0/12 at 640,
       12/12 at 1,024".

ALT-2  The ABSENT question is always asked WARM, on a slot where the same
       document's literal question was answered with the key moments earlier.

Both are tested against records that already exist: the A/B ran 640 and 1,024
in SEPARATE fresh processes, each starting at slot 0, and the distance run's
latency CONTROL re-ran 1,024 at slots 100-103.

Also re-derives, independently, the "59/66 wrong answers quote a real in-chunk
identifier" figure from milestones/s2/results/sweep.jsonl, and reports whether the
identifier is in THE CHUNK THAT REQUEST SENT or merely somewhere in the corpus.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S2 = REPO / "milestones" / "s2"
sys.path.insert(0, str(REPO))
from rlm.leakcheck import identifier_tokens  # noqa: E402


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha_map():
    out = {}
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
            out[hashlib.sha256(t.encode("utf-8")).hexdigest()] = {
                "uid": f"{d.name}/{cid}", "text": t,
                "ids": {x.lower() for x in identifier_tokens(t)},
                "cell": cell}
    return out


def fp_by(recs, keyfn, *, absent_only=True):
    agg = defaultdict(lambda: [0, 0])
    for r in recs:
        if r.get("status") != "ok":
            continue
        if absent_only and r.get("question_type") != "absent":
            continue
        k = keyfn(r)
        agg[k][1] += 1
        if r.get("label") == "FALSE-POSITIVE":
            agg[k][0] += 1
    return agg


def main() -> int:
    SM = sha_map()

    # ------------------------------------------------------------------ #
    print("=== ALT-1: is the 640/1024 split really a slot-index split? ===")
    for name in ("refusal-ab.jsonl", "refusal-ab-640.jsonl"):
        p = S2 / "results" / name
        if not p.exists():
            continue
        recs = load(p)
        slots = [r.get("requested_slot") for r in recs if r.get("requested_slot") is not None]
        sizes = sorted({r.get("size_target") for r in recs if r.get("size_target")})
        print(f"\n{name}: {len(recs)} records, sizes {sizes}, "
              f"slots {min(slots)}..{max(slots)}")
        for k, (a, n) in sorted(fp_by(recs, lambda r: (r.get("arm"), r.get("size_target"))).items()):
            if n:
                print(f"   arm={k[0]:22s} size={k[1]}  FALSE-POSITIVE {a}/{n}"
                      f"  ({a/n:.0%})")
        # slot range per size
        per = defaultdict(list)
        for r in recs:
            if r.get("requested_slot") is not None and r.get("size_target"):
                per[r["size_target"]].append(r["requested_slot"])
        for s, v in sorted(per.items()):
            print(f"   size {s}: slots {min(v)}..{max(v)}")

    print("\n--- distance.jsonl: arm A at 1024, low slots vs high slots ---")
    dist = load(S2 / "results" / "distance.jsonl")
    for phase in ("grid", "latency-control"):
        rows = [r for r in dist if r.get("phase") == phase and r.get("arm") == "A-shipped"
                and r.get("size_target") == 1024 and r.get("status") == "ok"]
        if not rows:
            continue
        slots = sorted({r["requested_slot"] for r in rows})
        a, n = 0, 0
        for r in rows:
            if r.get("question_type") == "absent":
                n += 1
                a += r.get("label") == "FALSE-POSITIVE"
        print(f"   {phase:16s} slots {slots[0]}..{slots[-1]}  "
              f"FALSE-POSITIVE {a}/{n}")
    # arm A slot ranges by size
    print("   arm A grid slot ranges by size:")
    per = defaultdict(list)
    for r in dist:
        if r.get("phase") == "grid" and r.get("arm") == "A-shipped" and r.get("requested_slot") is not None:
            per[r["size_target"]].append(r["requested_slot"])
    for s, v in sorted(per.items()):
        print(f"     size {s}: slots {min(v)}..{max(v)}")

    # ------------------------------------------------------------------ #
    print("\n=== ALT-2: does the ABSENT answer track the call that preceded it? ===")
    # sequence per (arm, cell_uid) in file order
    seq = defaultdict(list)
    for r in dist:
        if r.get("status") == "ok" and r.get("question_type"):
            seq[(r.get("phase"), r.get("arm"), r.get("cell_uid"))].append(r)
    tab = Counter()
    for k, rows in seq.items():
        for i, r in enumerate(rows):
            if r.get("question_type") != "absent":
                continue
            prev = rows[i - 1] if i else None
            tab[(r.get("size_target"), prev["question_type"] if prev else "COLD",
                 r.get("label") == "FALSE-POSITIVE")] += 1
    print("  size | preceding question | FP? | n")
    for (size, prev, isfp), n in sorted(tab.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        print(f"  {size:5d} | {prev:10s} | {str(isfp):5s} | {n}")

    # ------------------------------------------------------------------ #
    print("\n=== 59/66 RE-DERIVED from milestones/s2/results/sweep.jsonl ===")
    sw = S2 / "results" / "sweep.jsonl"
    if sw.exists():
        recs = load(sw)
        print(f"sweep.jsonl records: {len(recs)}")
        WRONG = {"CONFABULATION", "FALSE-POSITIVE"}
        wrong = [r for r in recs if r.get("status") == "ok" and r.get("label") in WRONG]
        print(f"non-refusal wrong answers (CONFABULATION+FALSE-POSITIVE): {len(wrong)}")
        cat = Counter()
        foreign_rows = []
        for r in wrong:
            ent = SM.get(r.get("chunk_sha256"))
            toks = {t.lower() for t in identifier_tokens(r.get("raw_output") or "")}
            if not toks:
                cat["no identifier-shaped token"] += 1
                continue
            if ent is None:
                cat["chunk NOT resolvable on disk"] += 1
                continue
            q = {t.lower() for t in identifier_tokens(r.get("question") or "")}
            own = ent["ids"] | q
            inc = toks & own
            out = toks - own
            elsewhere = {t: [v["uid"] for v in SM.values() if t in v["ids"]] for t in out}
            elsewhere = {t: v for t, v in elsewhere.items() if v}
            if inc and not out:
                cat["IN-CHUNK only"] += 1
            elif elsewhere:
                cat["OUT-OF-CHUNK: identifier lives in another fixture"] += 1
                foreign_rows.append((r, elsewhere))
            elif inc and out:
                cat["MIXED (in-chunk + unknown string)"] += 1
            else:
                cat["FABRICATED (in no fixture)"] += 1
        for k, v in cat.most_common():
            print(f"  {v:4d}  {k}")
        print(f"\n  OUT-OF-CHUNK detail ({len(foreign_rows)}):")
        for r, e in foreign_rows[:30]:
            print(f"    {r.get('phase')} {r.get('cell_id')} {r.get('question_type')} "
                  f"{r.get('label')} slot={r.get('requested_slot')}: "
                  f"{(r.get('raw_output') or '')[:70]!r} -> {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
