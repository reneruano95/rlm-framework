"""Did s2/results/sweep.jsonl really get a freshly launched leaf per cell?

RESULTS.md claims it did, but the PowerShell drivers that did it were throwaway
and are not in the repo, and the records carry no process id. One artefact does
survive: the `ts` on every record. A llama-server restart on this box costs
~5-10 s of model load (s2/R13-slotcount.md measures 5.65 s warm / 9.9 s cold)
plus process teardown, so a genuine per-cell restart must show a multi-second
gap between the LAST call of one cell and the FIRST call of the next -- larger
than the gap between consecutive calls inside a cell.

Compare that against sweep-run1-shared-server.jsonl, which is the KNOWN
single-process run, as the negative control.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

RESULTS = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


for name in ("sweep.jsonl", "sweep-run1-shared-server.jsonl"):
    p = RESULTS / name
    recs = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    recs.sort(key=lambda r: r["ts"])
    print(f"=== {name}: {len(recs)} records, "
          f"{ts(recs[0]['ts'])} .. {ts(recs[-1]['ts'])}")
    prev = None
    for r in recs:
        if prev is not None:
            gap = (ts(r["ts"]) - ts(prev["ts"])).total_seconds()
            boundary = (r["chunk_sha256"] != prev["chunk_sha256"]
                        or r["phase"] != prev["phase"])
            if boundary:
                print(f"   CELL BOUNDARY {prev['phase']}/{prev['size_target']} "
                      f"-> {r['phase']}/{r['size_target']}: gap = {gap:.1f} s "
                      f"(prev call wall {prev.get('wall_s')} s)")
        prev = r
    # within-cell gap distribution for scale
    gaps = []
    prev = None
    for r in recs:
        if prev is not None and r["chunk_sha256"] == prev["chunk_sha256"] \
                and r["phase"] == prev["phase"]:
            gaps.append((ts(r["ts"]) - ts(prev["ts"])).total_seconds())
        prev = r
    if gaps:
        gaps.sort()
        print(f"   within-cell gaps: min {gaps[0]:.1f}s median "
              f"{gaps[len(gaps)//2]:.1f}s max {gaps[-1]:.1f}s  (n={len(gaps)})")
    print()
