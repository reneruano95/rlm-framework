"""Re-derive the auditor's occupancy claim:

  'chunk 622 tok = 128/128 correct with host cache ON (condition w640) AND
   128/128 with --cache-ram 0 (cram0-w640); chunk 1008 tok = 122/128 with cache
   ON (baseline), 122/128 with --cache-ram 0 (cram0), 122/128 with
   --no-cache-idle-slots (nocacheidle), 122/128 with -sps 0, 122/128 shuffled.
   Four independent 128-call conditions.'

and the argv claim ('4 distinct argv seen, including --cache-ram 0 (528
records) and --no-cache-idle-slots (128 records)').
"""
import collections
import json
import pathlib

S2 = pathlib.Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (S2 / "results" / "occupancy.jsonl").open(encoding="utf-8") if l.strip()]
print(f"occupancy.jsonl rows: {len(rows)}")

argvs = collections.Counter(" ".join(r["argv"]) for r in rows)
print(f"\ndistinct argv: {len(argvs)}")
for a, n in argvs.most_common():
    tail = a.split("--host 127.0.0.1", 1)[-1]
    print(f"  n={n:5d}  ...{tail}")

print("\nrecords whose argv contains --cache-ram 0        :",
      sum(1 for r in rows if "--cache-ram" in r["argv"]))
print("records whose argv contains --no-cache-idle-slots:",
      sum(1 for r in rows if "--no-cache-idle-slots" in r["argv"]))

print("\ncondition x chunk_tokens -> correct/total  (status ok only)")
agg = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r.get("status") != "ok":
        agg[(r.get("condition"), r.get("chunk_tokens"))][2] += 1
        continue
    k = (r.get("condition"), r.get("chunk_tokens"))
    agg[k][1] += 1
    if r.get("answer_correct"):
        agg[k][0] += 1
for k in sorted(agg, key=lambda x: (str(x[0]), x[1])):
    c, n, bad = agg[k]
    flag = ""
    if n:
        flag = f"  {100*c/n:5.1f}%"
    print(f"  {str(k[0]):22s} chunk={k[1]:6d}  {c:4d}/{n:<4d} not-ok={bad}{flag}")

print("\nrun_id x condition (are the '128-call conditions' really distinct runs?)")
runs = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    runs[(r.get("condition"), r.get("chunk_tokens"))][r.get("run_id")] += 1
for k in sorted(runs, key=lambda x: (str(x[0]), x[1])):
    print(f"  {str(k[0]):22s} chunk={k[1]:6d} -> {dict(runs[k])}")

print("\nslot mismatch / cache_n hygiene")
print("  slot_mismatch True:", sum(1 for r in rows if r.get("slot_mismatch")))
print("  leak_detected True:", sum(1 for r in rows if r.get("leak_detected")))
firstcache = collections.Counter()
seen = set()
for r in rows:
    k = (r.get("run_id"), r.get("id_slot"))
    if k not in seen:
        seen.add(k)
        firstcache[(r.get("condition"), (r.get("cache_n") or 0) > 0)] += 1
print("  first call per (run_id,slot) with cache_n>0:",
      {k: v for k, v in firstcache.items() if k[1]})
