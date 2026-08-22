"""Offline audit: for every s2 leaf-quality measurement the spec cites, record
(a) the server launch flags actually used, (b) whether the run shows evidence of
cross-request KV reuse on calls the run believed were cold/virgin.

Stdlib only. No GPU, no HTTP. Reads existing result files.
"""
from __future__ import annotations

import json
import pathlib
import collections

ROOT = pathlib.Path(__file__).resolve().parents[3]
RES = ROOT / "milestones" / "s2" / "results"


def load(name):
    out = []
    with (RES / name).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarise_cached(recs, name, cachekey="tokens_cached", coldkey="cold"):
    n = len(recs)
    ok = [r for r in recs if r.get("status") in (None, "ok")]
    cached = [r for r in ok if (r.get(cachekey) or 0) > 0]
    cold = [r for r in ok if r.get(coldkey) is True]
    cold_cached = [r for r in cold if (r.get(cachekey) or 0) > 0]
    print(f"\n=== {name}: {n} records, {len(ok)} ok")
    print(f"  calls with {cachekey}>0 : {len(cached)}/{len(ok)}")
    if cached:
        vals = sorted((r.get(cachekey) or 0) for r in cached)
        print(f"    {cachekey} min/med/max = {vals[0]}/{vals[len(vals)//2]}/{vals[-1]}")
    print(f"  calls flagged cold      : {len(cold)}   of which {cachekey}>0: {len(cold_cached)}")
    if cold_cached:
        vals = sorted((r.get(cachekey) or 0) for r in cold_cached)
        print(f"    COLD-but-cached {cachekey} min/med/max = {vals[0]}/{vals[len(vals)//2]}/{vals[-1]}")
        ex = cold_cached[0]
        print("    example:", {k: ex.get(k) for k in
                               ("cell_id", "cell_uid", "arm", "size_target", "position",
                                "question_type", "requested_slot", "slot_id", cachekey,
                                "tokens_in", "prefill_ms")})
    # slot discipline
    mism = [r for r in ok if r.get("requested_slot") is not None
            and r.get("slot_id") is not None and r["requested_slot"] != r["slot_id"]]
    print(f"  requested_slot != slot_id: {len(mism)}")
    slots = collections.Counter(r.get("slot_id") for r in ok if r.get("slot_id") is not None)
    reused = {s: c for s, c in slots.items() if c > 1}
    print(f"  distinct slots used: {len(slots)}; slots serving >1 call: {len(reused)}"
          f" (max calls on one slot: {max(slots.values()) if slots else 0})")
    leaks = [r for r in ok if r.get("leak_detected")]
    print(f"  R13 foreign-string detector hits: {len(leaks)}")
    return ok


def main():
    # ---- DISTANCE: the cliff + instruction decay + one-horizon-two-distances
    d = load("distance.jsonl")
    ok = summarise_cached(d, "distance.jsonl (DISTANCE.md: cliff, instruction decay)")
    # phase breakdown
    ph = collections.Counter(r.get("phase") for r in ok)
    print("  phases:", dict(ph))

    # ---- REFUSAL A/B: the 30/30 vs 0/21 false-positive contrast
    for f in ("refusal-ab.jsonl", "refusal-ab-640.jsonl"):
        r = load(f)
        summarise_cached(r, f"{f} (REFUSAL-AB: false-positive rate)")

    # ---- SWEEP: the 95% false-positive + the original cliff
    s = load("sweep.jsonl")
    summarise_cached(s, "sweep.jsonl (95% FP, distance cliff)")

    # ---- OCCUPANCY: launch flags per condition + the accidental 640/1024 recall gap
    o = load("occupancy.jsonl")
    print(f"\n=== occupancy.jsonl: {len(o)} records")
    by_cond = collections.defaultdict(list)
    for r in o:
        by_cond[r.get("condition")].append(r)
    for cond, recs in sorted(by_cond.items()):
        argv = recs[0].get("argv")
        argv_s = " ".join(argv) if isinstance(argv, list) else str(argv)
        cram = "--cache-ram" in argv_s
        noidle = "--no-cache-idle-slots" in argv_s
        correct = [r for r in recs if r.get("answer_correct") is not None]
        nc = sum(1 for r in correct if r["answer_correct"])
        toks = collections.Counter(r.get("chunk_tokens") for r in recs)
        print(f"  {cond:28s} n={len(recs):5d} conc={sorted({r.get('concurrency') for r in recs})} "
              f"np={sorted({r.get('np') for r in recs})} chunk_tokens={dict(toks)} "
              f"cache-ram-flag={cram} no-cache-idle={noidle} correct={nc}/{len(correct)}")
    # the §8 recall gap, split by chunk size AND by whether host cache was off
    print("\n  --- occupancy correctness by (chunk_tokens, host-cache-off, concurrency) ---")
    agg = collections.defaultdict(lambda: [0, 0])
    for r in o:
        if r.get("answer_correct") is None:
            continue
        argv = r.get("argv")
        argv_s = " ".join(argv) if isinstance(argv, list) else str(argv)
        off = ("--cache-ram 0" in argv_s) or ("--cache-ram" in argv_s and "0" in argv_s) or \
              ("--no-cache-idle-slots" in argv_s)
        key = (r.get("chunk_tokens"), off, r.get("concurrency"))
        agg[key][1] += 1
        if r["answer_correct"]:
            agg[key][0] += 1
    for k in sorted(agg, key=lambda x: (x[0] or 0, x[1], x[2] or 0)):
        c, n = agg[k]
        print(f"    chunk={k[0]} host_cache_off={k[1]} conc={k[2]}: {c}/{n}")

    # exact argv strings seen (for the record)
    print("\n  --- distinct argv seen in occupancy ---")
    seen = collections.Counter()
    for r in o:
        argv = r.get("argv")
        seen[" ".join(argv) if isinstance(argv, list) else str(argv)] += 1
    for a, c in seen.most_common():
        print(f"    [{c:5d}] {a}")


if __name__ == "__main__":
    main()
