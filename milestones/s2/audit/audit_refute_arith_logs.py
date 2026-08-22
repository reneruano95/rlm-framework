"""INDEPENDENT re-derivation of every server-log count in the r13-truth report.

Checks:
  * request counts per leak-* server log (36 / 18 / 18 / 18 / 27)
  * cold-prefill counts (prompt eval token count == full prompt)
  * 'selected slot by id' vs 'selected slot by LCP similarity' tallies in the
    five leak-* logs, distance-leaf.err.log, leaf-server.err.log,
    milestones/s2/logs/arch-qwen.err, milestones/s2/logs/arch-gemma.err
  * n_slots / n_ctx_slot / kv_unified on line 8 of each leak log
  * count of 'making room for prompt cache entry' in distance-leaf.err.log
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")

RE_PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
RE_EVAL = re.compile(r"^\s*eval time\s*=", re.M)
RE_BY_ID = re.compile(r"selected slot by id")
RE_BY_LCP = re.compile(r"selected slot by LCP similarity")
RE_FSIM = re.compile(r"f_sim_best\s*=\s*([\d.]+)")
RE_KEEP = re.compile(r"f_keep\s*=\s*([\d.]+)")
RE_ROOM = re.compile(r"making room for prompt cache entry")
RE_TASK = re.compile(r"n_past\s*=\s*(\d+)")
RE_SLOTID = re.compile(r"slot\s+\w+:\s*id\s+(\d+)")


def scan(path: Path, label: str = ""):
    txt = path.read_text(encoding="utf-8", errors="replace")
    pe = RE_PROMPT_EVAL.findall(txt)
    byid = len(RE_BY_ID.findall(txt))
    bylcp = len(RE_BY_LCP.findall(txt))
    room = len(RE_ROOM.findall(txt))
    fsim = RE_FSIM.findall(txt)
    tokcounts = Counter(int(n) for _, n in pe)
    print("-" * 96)
    print(f"{label or path.name}  ({path})")
    print(f"  'prompt eval time = .. / N tokens' lines: {len(pe)}")
    print(f"  selected slot by id: {byid}   by LCP similarity: {bylcp}   f_sim_best values: {len(fsim)}")
    print(f"  'making room for prompt cache entry': {room}")
    if len(tokcounts) <= 14:
        print(f"  prompt-eval token histogram: {dict(sorted(tokcounts.items()))}")
    else:
        print(f"  prompt-eval token counts: {len(tokcounts)} distinct, "
              f"min={min(tokcounts)} max={max(tokcounts)}")
    for ln in txt.splitlines()[:14]:
        if "n_slots" in ln or "kv_unified" in ln or "n_ctx_slot" in ln:
            print(f"  header: {ln.strip()[:160]}")
    return {"pe": [int(n) for _, n in pe], "byid": byid, "bylcp": bylcp, "room": room}


if __name__ == "__main__":
    L = ROOT / "traces" / "logs"
    res = {}
    for f, lab in [
        ("leaf-server-ub512nocacheidleslots.log.err", "--no-cache-idle-slots (leak-nocacheidle)"),
        ("leaf-server-ub512cacheram0ctxcp0.log.err", "--cache-ram 0 -ctxcp 0 (leak-nocram)"),
        ("leaf-server-ub512cacheram0.log.err", "--cache-ram 0 (leak-cram0)"),
        ("leaf-server-ub512ctxcp0.log.err", "-ctxcp 0 (leak-ctxcp0)"),
        ("leaf-server-ub512.log.err", "S0 default (leak-slotiso + leak-erase)"),
        ("distance-leaf.err.log", "distance run"),
        ("leaf-server.err.log", "refusal runs"),
    ]:
        p = L / f
        if p.exists():
            res[f] = scan(p, lab)
    for f in ("arch-qwen.err", "arch-gemma.err"):
        p = ROOT / "milestones" / "s2" / "logs" / f
        if p.exists():
            res[f] = scan(p)
