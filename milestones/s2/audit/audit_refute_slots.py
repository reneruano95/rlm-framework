"""Test the auditor's SLOT HYGIENE claims literally, including the rows they
may have filtered out.

Claims under test:
  distance : "91 slots used, EVERY slot held exactly 1 distinct chunk_sha256";
             "R13's precondition -- a slot that previously held another
             document -- never occurred"
  sweep    : "one fresh process per cell confirmed in-data (13 cells, 9 calls
             each, first-call cache_n = 0 in every cell)"
"""
import collections
import datetime as dt
import json
import pathlib

S2 = pathlib.Path(__file__).resolve().parents[1]
RES = S2 / "results"


def load(name):
    return [json.loads(l) for l in (RES / name).open(encoding="utf-8") if l.strip()]


def slot_timeline(name, key="chunk_sha256"):
    recs = [r for r in load(name) if r.get("status") in (None, "ok")]
    per_slot = collections.defaultdict(list)
    for i, r in enumerate(recs):
        per_slot[r.get("slot_id")].append((i, r))
    print(f"\n=== {name}: {len(recs)} ok records, {len(per_slot)} slots")
    multi = []
    for sid, seq in sorted(per_slot.items(), key=lambda kv: (kv[0] is None, kv[0])):
        docs = [r.get(key) for _, r in seq]
        distinct_nonnull = {d for d in docs if d}
        nulls = sum(1 for d in docs if not d)
        # did the slot ever serve document B after document A?
        seen = []
        switches = 0
        for _, r in seq:
            d = r.get(key) or ("<null:" + str(r.get("cell_uid")) + ">")
            if seen and d != seen[-1]:
                switches += 1
            seen.append(d)
        if switches:
            first_reuse = [
                (r.get("cell_uid") or r.get("cell_id"), r.get("phase"),
                 r.get("tokens_cached"))
                for _, r in seq
            ]
            multi.append((sid, len(distinct_nonnull), nulls, switches,
                          first_reuse[:6]))
    print(f"slots whose document CHANGED mid-slot: {len(multi)}")
    for m in multi[:20]:
        print(f"   slot {m[0]}: distinct-nonnull={m[1]} nulls={m[2]} "
              f"switches={m[3]}")
        for e in m[4]:
            print(f"        cell={e[0]} phase={e[1]} tokens_cached={e[2]}")
    hist_nonnull = collections.Counter(
        len({r.get(key) for _, r in seq if r.get(key)}) for seq in per_slot.values()
    )
    print(f"docs-per-slot histogram (null chunk_sha256 excluded): "
          f"{dict(sorted(hist_nonnull.items()))}")


def sweep_restarts(name):
    recs = load(name)
    print(f"\n=== {name}: fresh-process-per-cell test")
    per_cell = collections.OrderedDict()
    for r in recs:
        per_cell.setdefault(r["cell_id"], []).append(r)
    print(f"cells={len(per_cell)}  calls/cell="
          f"{sorted({len(v) for v in per_cell.values()})}")
    prev_end = None
    for cell, v in per_cell.items():
        first = v[0]
        t0 = dt.datetime.fromisoformat(v[0]["ts"])
        t1 = dt.datetime.fromisoformat(v[-1]["ts"])
        gap = (t0 - prev_end).total_seconds() if prev_end else None
        nz = [x["tokens_cached"] for x in v[:1]]
        print(f"  {cell:16s} n={len(v):2d} first_call cache={first['tokens_cached']:5d} "
              f"cold={first['cold']}  first_prefill_ms={first['prefill_ms']:8.1f} "
              f"gap_from_prev_cell={gap if gap is None else round(gap,1)}s")
        prev_end = t1
    bad = [c for c, v in per_cell.items() if v[0]["tokens_cached"] != 0]
    print(f"cells whose FIRST call had tokens_cached != 0: {len(bad)} {bad}")


if __name__ == "__main__":
    slot_timeline("distance.jsonl")
    slot_timeline("refusal-ab.jsonl")
    slot_timeline("refusal-ab-640.jsonl")
    sweep_restarts("sweep.jsonl")
    sweep_restarts("sweep-run1-shared-server.jsonl")
