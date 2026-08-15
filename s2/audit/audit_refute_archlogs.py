"""Re-derive the auditor's claims about the arch-ladder server logs.

  qwen : '72 tasks routed onto only 4 slots (launch_slot_ counts 32/12/6/22)
          with 66 of 72 selections logged as selected slot by LCP similarity'
  gemma: 'ALL 24 tasks launched on slot id 3'
  and: 'n_slots = 4, n_ctx_slot = 16384, kv_unified = false' at arch-qwen.err:12
"""
import collections
import pathlib
import re

S2 = pathlib.Path(__file__).resolve().parents[1]
LOGS = S2 / "logs"

LAUNCH = re.compile(r"launch_slot_with_task:\s*id\s*(-?\d+)")
LAUNCH2 = re.compile(r"slot\s+launch_slot_:\s*id\s+(-?\d+)")
LCP = re.compile(r"selected slot by LCP similarity", re.I)
LRU = re.compile(r"selected slot by LRU", re.I)
SLOTID = re.compile(r"id\s+(-?\d+)\s*\|\s*task\s+(-?\d+)")


def scan(name):
    p = LOGS / name
    if not p.exists():
        print(f"{name}: MISSING")
        return
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print(f"\n=== {name} ({len(lines)} lines)")
    for i, l in enumerate(lines[:20], 1):
        if "n_slots" in l or "n_ctx_slot" in l or "kv_unified" in l:
            print(f"   :{i}  {l.strip()}")
    launch = collections.Counter()
    tasks = set()
    for l in lines:
        if "launch_slot_" in l:
            m = SLOTID.search(l)
            if m:
                launch[int(m.group(1))] += 1
                tasks.add(int(m.group(2)))
    print(f"   launch_slot_ lines: {sum(launch.values())} across "
          f"{len(launch)} slots -> {dict(sorted(launch.items()))}")
    print(f"   distinct task ids launched: {len(tasks)}")
    print(f"   'selected slot by LCP similarity' lines: "
          f"{len(LCP.findall(text))}")
    print(f"   'selected slot by LRU' lines: {len(LRU.findall(text))}")
    # any explicit cache-ram / prompt cache lines
    for i, l in enumerate(lines, 1):
        if ("cache_ram" in l or "cache-ram" in l or "prompt cache" in l.lower()
                or "--cache-ram" in l):
            print(f"   :{i} {l.strip()[:160]}")


for n in ("arch-qwen.err", "arch-gemma.err", "arch-q35.err",
          "slotfix-qwen.err", "origgeom.err"):
    scan(n)
