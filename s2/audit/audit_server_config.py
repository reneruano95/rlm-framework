"""Offline inventory: what server configuration did each s2 result file record?

Reads every *.jsonl / *.json under s2/results/ WITHOUT loading them whole into a
human's context, and reports, per file:
  - record count, first/last record keys
  - any key that looks like server configuration (np/parallel/ctx/cache/flags/argv)
  - the distinct values of those keys
No network, no GPU, stdlib only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(r"D:\PROJECTS\rlm-halo-framework\s2\results")

# Substrings that mark a key as server-configuration-bearing.
CFG_MARKS = (
    "np", "parallel", "ctx", "cache", "flag", "argv", "launch", "extra",
    "slot", "server", "port", "cont_batch", "ub", "batch", "condition",
    "model", "temperature", "seed", "concurrency", "drain",
)


def looks_cfg(key: str) -> bool:
    k = key.lower()
    return any(m in k for m in CFG_MARKS)


def walk(obj, prefix="", out=None, depth=0):
    """Flatten a dict/list into dotted paths, capped in depth and width."""
    if out is None:
        out = {}
    if depth > 3:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                walk(v, p, out, depth + 1)
            else:
                out[p] = v
    elif isinstance(obj, list):
        # only look at the first element of a list, enough for shape
        if obj and isinstance(obj[0], (dict, list)):
            walk(obj[0], prefix + "[0]", out, depth + 1)
        elif obj:
            out[prefix] = obj
    return out


def main() -> int:
    files = sorted(list(RESULTS.glob("*.jsonl")) + list(RESULTS.glob("*.json")))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for f in files:
        if only and only not in f.name:
            continue
        print("=" * 78)
        print(f"FILE {f.name}  ({f.stat().st_size} bytes)")
        recs = []
        if f.suffix == ".jsonl":
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            recs = [json.loads(f.read_text(encoding="utf-8"))]
        print(f"  records: {len(recs)}")
        if not recs:
            continue
        # union of config-ish paths with their distinct values (capped)
        vals: dict[str, set] = {}
        for r in recs:
            flat = walk(r)
            for k, v in flat.items():
                if looks_cfg(k):
                    s = vals.setdefault(k, set())
                    if len(s) < 12:
                        try:
                            s.add(v if not isinstance(v, list) else tuple(v))
                        except TypeError:
                            s.add(str(v))
        for k in sorted(vals):
            vs = sorted(vals[k], key=lambda x: str(x))
            shown = ", ".join(repr(x)[:110] for x in vs[:8])
            more = "" if len(vs) <= 8 else f"  (+{len(vs) - 8} more)"
            print(f"    {k} = {shown}{more}")
        # top-level keys of the first record, for orientation
        print(f"  first-record top keys: {sorted(recs[0].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
