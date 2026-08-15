"""Recover, from each llama-server stderr log the repo still holds, the facts
that decide LEAK-POSSIBLE:

  n_slots / n_ctx_slot          -> the -np / -c the process actually ran with
  "selected slot by id (N)"     -> the client PINNED the slot (never-reuse policy
                                   is at least attempted)
  "selected slot by LCP simi"   -> the SERVER chose the slot by prefix similarity
                                   (a slot that already holds another document
                                   can be handed the next one)
  "selected slot by LRU"        -> server chose, no similarity
  host-prompt-cache activity    -> lines mentioning `cache state` / `prompt cache`
                                   / `alloc`+`cache`, which only appear when the
                                   host prompt cache (--cache-idle-slots,
                                   --cache-ram > 0) is saving/restoring. Their
                                   presence proves the host cache was ON; their
                                   absence is weaker evidence (verbosity 3 may
                                   not print them).

Reads big logs line-by-line; nothing is loaded whole.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\PROJECTS\rlm-halo-framework")

LOGS = [
    "traces/logs/distance-leaf.err.log",
    "traces/logs/leaf-server.err.log",
    "traces/logs/leaf-server-ub128.log.err",
    "traces/logs/leaf-server-ub512.log.err",
    "traces/logs/leaf-server-ub512cacheram0.log.err",
    "traces/logs/leaf-server-ub512cacheram0ctxcp0.log.err",
    "traces/logs/leaf-server-ub512ctxcp0.log.err",
    "traces/logs/leaf-server-ub512nocacheidleslots.log.err",
    "s2/logs/arch-qwen.err",
    "s2/logs/arch-gemma.err",
    "s2/logs/bos.err",
]

INIT = re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")
CACHEY = re.compile(r"cache state|prompt cache|cache_ram|state save|state restore|"
                    r"saving slot|restoring slot|checkpoint", re.I)


for rel in LOGS:
    p = ROOT / rel
    if not p.exists():
        print(f"--- {rel}: MISSING")
        continue
    sel = Counter()
    inits = []
    cachey = Counter()
    n_lines = 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            m = INIT.search(line)
            if m:
                inits.append(m.groups())
            if "get_availabl" in line:
                if "by id (" in line:
                    sel["by id"] += 1
                elif "LCP similarity" in line:
                    sel["by LCP similarity"] += 1
                elif "LRU" in line:
                    sel["by LRU"] += 1
                else:
                    sel["other"] += 1
            if CACHEY.search(line):
                # normalise to the message stem so the counter stays small
                stem = re.sub(r"[\d.]+", "#", line.strip())[-90:]
                cachey[stem] += 1
    print(f"--- {rel}  ({n_lines} lines, {p.stat().st_size} bytes)")
    for g in inits:
        print(f"      init: n_slots={g[0]} n_ctx_slot={g[1]} kv_unified={g[2]} "
              f"-> implies -np {g[0]} -c {int(g[0]) * int(g[1])}")
    if not inits:
        print("      init: NOT FOUND in this log")
    print(f"      slot selection: {dict(sel)}")
    if cachey:
        print(f"      host-cache-ish lines ({sum(cachey.values())} total):")
        for stem, n in cachey.most_common(4):
            print(f"          x{n}  ...{stem}")
    else:
        print("      host-cache-ish lines: NONE (weak evidence only)")
