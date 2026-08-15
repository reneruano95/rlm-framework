"""Extract the RECORDED llama-server argv per condition from the result files
that store one (occupancy.jsonl, r14.jsonl), plus the recorded cache_prompt.

Offline only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results")


def flags_of(argv: list[str]) -> dict[str, str]:
    """Pull the leak-relevant flags out of a full argv."""
    out: dict[str, str] = {}
    for i, tok in enumerate(argv):
        if tok in ("-np", "--parallel"):
            out["-np"] = argv[i + 1]
        elif tok in ("-c", "--ctx-size"):
            out["-c"] = argv[i + 1]
        elif tok == "--cache-ram":
            out["--cache-ram"] = argv[i + 1]
        elif tok == "--no-cache-idle-slots":
            out["--no-cache-idle-slots"] = "PRESENT"
        elif tok == "--cache-idle-slots":
            out["--cache-idle-slots"] = "PRESENT"
        elif tok == "-sps":
            out["-sps"] = argv[i + 1]
        elif tok in ("-ub", "--ubatch-size"):
            out["-ub"] = argv[i + 1]
        elif tok in ("-b", "--batch-size"):
            out["-b"] = argv[i + 1]
        elif tok == "--no-cont-batching":
            out["--no-cont-batching"] = "PRESENT"
        elif tok == "--cont-batching":
            out["--cont-batching"] = "PRESENT"
        elif tok == "--no-kv-unified":
            out["--no-kv-unified"] = "PRESENT"
        elif tok == "--no-context-shift":
            out["--no-context-shift"] = "PRESENT"
    return out


for name in ("occupancy.jsonl", "r14.jsonl", "cache_instrument.jsonl"):
    p = RESULTS / name
    if not p.exists():
        continue
    print("=" * 78)
    print(name)
    per_cond: dict[str, set] = defaultdict(set)
    per_cond_extra: dict[str, set] = defaultdict(set)
    per_cond_n: dict[str, int] = defaultdict(int)
    full_argvs: set[str] = set()
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cond = r.get("condition", "?")
            per_cond_n[cond] += 1
            argv = r.get("argv")
            if isinstance(argv, list):
                full_argvs.add(" ".join(argv))
                fl = flags_of(argv)
                per_cond[cond].add(json.dumps(fl, sort_keys=True))
            if "extra" in r:
                per_cond_extra[cond].add(repr(r["extra"]))
    for cond in sorted(per_cond_n):
        print(f"  --- {cond}  (n={per_cond_n[cond]})")
        for e in sorted(per_cond_extra.get(cond, {"<no extra field>"})):
            print(f"        extra = {e}")
        for f in sorted(per_cond.get(cond, set())) or ["<NO argv RECORDED>"]:
            print(f"        flags = {f}")
    if full_argvs:
        print(f"  distinct full argvs: {len(full_argvs)}")
        for a in sorted(full_argvs)[:6]:
            print(f"      {a}")
