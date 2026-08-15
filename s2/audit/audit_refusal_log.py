"""OFFLINE: what does the leaf server's own log say about the refusal A/B run?

Checks, in order:
  * how many server launches the file covers, and the geometry each reported
  * how slots were SELECTED ("by id" vs "by LCP similarity" -- the router that
    produced the arch_ladder cross-request hit)
  * every non-"by id" selection, verbatim
  * per-task prompt-eval token counts, to see whether any task prefilled fewer
    tokens than the client's rendered prompt (partial KV reuse)
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

LOG = Path(r"D:\PROJECTS\rlm-halo-framework\traces\logs\leaf-server.err.log")

RE_LOAD = re.compile(r"load_model: loading model '(.+?)'")
RE_INIT = re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")
RE_SEL = re.compile(r"get_availabl: id\s+(\d+) \| task (-?\d+) \| selected slot (.+)$")
RE_PE = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens")
RE_REL = re.compile(r"release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)")


def main():
    text = LOG.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print(f"{LOG}: {len(lines)} lines, {len(text)} bytes")

    loads = [(i + 1, m.group(1)) for i, l in enumerate(lines) if (m := RE_LOAD.search(l))]
    print(f"\nserver launches in this file: {len(loads)}")
    for ln, model in loads:
        print(f"  line {ln}: {model}")
    for i, l in enumerate(lines):
        if m := RE_INIT.search(l):
            print(f"  line {i+1}: n_slots={m.group(1)} n_ctx_slot={m.group(2)} "
                  f"kv_unified={m.group(3)}")

    sel = collections.Counter()
    odd = []
    for i, l in enumerate(lines):
        if m := RE_SEL.search(l):
            how = m.group(3)
            key = "by id" if how.startswith("by id") else how.split("(")[0].strip()
            sel[key] += 1
            if not how.startswith("by id"):
                odd.append((i + 1, l.strip()))
    print(f"\nslot selections: {sum(sel.values())}")
    for k, v in sel.most_common():
        print(f"  {k!r}: {v}")
    print(f"non-'by id' selections: {len(odd)}")
    for ln, l in odd[:40]:
        print("   ", ln, l)

    # any mention of the LCP router / cache restore at all?
    for needle in ("similarity", "f_keep", "f_sim", "restore", "prompt cache",
                   "cache_reuse", "erased", "SWA", "shift"):
        n = sum(1 for l in lines if needle in l)
        if n:
            print(f"\nlines containing {needle!r}: {n}")
            for l in [x for x in lines if needle in x][:5]:
                print("   ", l.strip())

    pe = [(int(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)))
          for l in lines if (m := RE_PE.search(l))]
    rel = {(int(m.group(1)), int(m.group(2))): int(m.group(3))
           for l in lines if (m := RE_REL.search(l))}
    print(f"\ntasks with a prompt-eval line: {len(pe)}")
    full = [t for t in pe if t[3] > 1200]
    part = [t for t in pe if t[3] <= 1200]
    print(f"  prefilled >1200 tokens (cold, full prompt): {len(full)}")
    print(f"  prefilled <=1200 tokens (warm re-query)   : {len(part)}")
    # for each task, total context after release vs tokens prefilled
    diffs = collections.Counter()
    for sid, task, ms, ntok in pe:
        n_total = rel.get((sid, task))
        if n_total is None:
            continue
        diffs[n_total - ntok] += 1
    print("  (n_tokens at release) - (tokens prefilled), histogram top 12:")
    for d, n in diffs.most_common(12):
        print(f"    reused_prefix={d:6d}  x{n}")
    # first task on each slot: was any of them a partial prefill?
    first_by_slot = {}
    for sid, task, ms, ntok in pe:
        first_by_slot.setdefault(sid, (task, ntok))
    partial_first = {s: v for s, v in first_by_slot.items() if v[1] <= 1200}
    print(f"\n  slots whose FIRST task prefilled <=1200 tokens: "
          f"{len(partial_first)} of {len(first_by_slot)}")
    for s, v in sorted(partial_first.items())[:20]:
        print("    slot", s, "task", v[0], "prefilled", v[1])


if __name__ == "__main__":
    main()
