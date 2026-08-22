"""Did the arch-ladder server actually REUSE KV, or only re-select a dirty slot?

llama-server logs, per task:
    prompt eval time = ... / P tokens        <- tokens it actually PREFILLED
    eval time        = ... / D tokens        <- tokens it DECODED
    release: ... n_tokens = T                <- final sequence length in the slot

If the prompt was fully prefilled then P == T - D (+/- 1 for the sampled token
bookkeeping). If P < T - D by a lot, the difference is KV carried over from
whatever the slot held before -- real cross-request reuse.

Run over every server log we still have, so the arch-ladder (unpinned slots) can
be compared against distance/refusal (pinned virgin slots).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"D:\PROJECTS\rlm-halo-framework")
LOGS = ["milestones/s2/logs/arch-qwen.err", "milestones/s2/logs/arch-gemma.err", "milestones/s2/logs/bos.err",
        "traces/logs/distance-leaf.err.log", "traces/logs/leaf-server.err.log"]

PE = re.compile(r"id\s+(\d+) \| task (\d+) \| prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
EV = re.compile(r"id\s+(\d+) \| task (\d+) \|\s+eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
REL = re.compile(r"id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+)")
SEL = re.compile(r"id\s+(\d+) \| task -1 \| selected slot by (.+?)(?:,|$)")

for rel in LOGS:
    p = ROOT / rel
    if not p.exists():
        continue
    pe: dict[str, int] = {}
    ev: dict[str, int] = {}
    rl: dict[str, int] = {}
    sel_order: list[str] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PE.search(line)
            if m:
                pe[m.group(2)] = int(m.group(3))
                continue
            m = EV.search(line)
            if m:
                ev[m.group(2)] = int(m.group(3))
                continue
            m = REL.search(line)
            if m:
                rl[m.group(2)] = int(m.group(3))
                continue
            m = SEL.search(line)
            if m:
                sel_order.append(m.group(2).strip())
    rows = []
    for task in pe:
        if task in ev and task in rl:
            prompt_len = rl[task] - ev[task]          # tokens of prompt in the slot
            reused = prompt_len - pe[task]            # tokens NOT prefilled
            rows.append((task, pe[task], prompt_len, reused))
    if not rows:
        print(f"--- {rel}: no complete task triples parsed")
        continue
    reuse_rows = [r for r in rows if r[3] > 8]        # >8 tokens = real reuse
    tot_reused = sum(r[3] for r in rows if r[3] > 0)
    print(f"--- {rel}")
    print(f"      tasks parsed: {len(rows)}")
    print(f"      tasks with >8 tokens carried over (NOT prefilled): "
          f"{len(reuse_rows)} / {len(rows)}")
    print(f"      total carried-over tokens: {tot_reused}")
    if reuse_rows:
        worst = sorted(reuse_rows, key=lambda r: -r[3])[:5]
        for t, prefilled, plen, reused in worst:
            print(f"          task {t}: prefilled {prefilled} of {plen} "
                  f"prompt tokens -> {reused} reused")
