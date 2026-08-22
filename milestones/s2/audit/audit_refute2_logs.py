"""REFUTE pass 2: re-derive the arch-ladder server-log claims from raw .err files.

Checks:
  - count of completions (task finished / prompt eval lines) vs JSONL rows
  - slot routing decisions: 'selected slot by LCP similarity' vs 'available slot'
  - f_sim_best distribution
  - prompt-eval token counts vs the JSONL rendered_tokens, element-for-element
  - n_slots / n_ctx_slot / kv_unified init line
  - presence/absence of the launch argv line
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_jsonl(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    for log, res in (("milestones/s2/logs/arch-qwen.err", "milestones/s2/results/arch_ladder_qwen-hybrid.jsonl"),
                     ("milestones/s2/logs/arch-gemma.err", "milestones/s2/results/arch_ladder_gemma-fullattn.jsonl")):
        txt = (ROOT / log).read_text(encoding="utf-8", errors="replace")
        lines = txt.splitlines()
        rows = load_jsonl(res)
        print(f"\n================ {log} ({len(lines)} lines) ================")

        # init line
        for i, l in enumerate(lines, 1):
            if "n_slots" in l and "n_ctx_slot" in l:
                print(f"  init  {log}:{i}: {l.strip()[:200]}")
                break

        # any argv / command line recorded?
        argv_hits = [(i, l) for i, l in enumerate(lines, 1)
                     if re.search(r"llama-server(\.exe)?\s+-", l) or "--cache-ram" in l
                     or "--cache-idle" in l or "system_info" in l.lower()]
        print(f"  launch-argv-like lines: {len(argv_hits)}")
        for i, l in argv_hits[:5]:
            print(f"    {log}:{i}: {l.strip()[:200]}")

        # routing decisions
        lcp = [l for l in lines if "LCP similarity" in l]
        avail = [l for l in lines if re.search(r"selected slot by (LRU|available)", l)]
        print(f"  'LCP similarity' lines: {len(lcp)}   'LRU/available' lines: {len(avail)}")
        sims = [float(m.group(1)) for l in lcp
                for m in [re.search(r"f_sim_best\s*=\s*([0-9.]+)", l)] if m]
        if sims:
            print(f"    f_sim_best n={len(sims)} min={min(sims):.3f} max={max(sims):.3f}")
        keeps = [float(m.group(1)) for l in lcp
                 for m in [re.search(r"f_keep\s*=\s*([0-9.]+)", l)] if m]
        if keeps:
            print(f"    f_keep     n={len(keeps)} min={min(keeps):.3f} max={max(keeps):.3f}")

        # slot ids used
        sid = re.findall(r"slot\s+(?:launch_slot_|update_slots|release|print_timi)\w*:\s*id\s+(\d+)", txt)
        from collections import Counter
        print(f"  slot-id mentions: {Counter(sid).most_common()}")

        # prompt eval counts, in order
        pe = re.findall(r"prompt eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*tokens", txt)
        pe = [int(x) for x in pe]
        ev = re.findall(r"\beval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)", txt)
        ev = [int(x) for x in ev]
        print(f"  completions with 'prompt eval time': {len(pe)}   'eval time' blocks: {len(ev)}")
        print(f"  JSONL rows: {len(rows)}  (expected server completions = rows, /tokenize is separate)")
        rt = [r["rendered_tokens"] for r in rows]
        print(f"  first 12 prompt-eval counts: {pe[:12]}")
        print(f"  last  {len(rt)} prompt-eval counts: {pe[-len(rt):] if len(pe) >= len(rt) else pe}")
        print(f"  JSONL rendered_tokens      : {rt}")
        if len(pe) >= len(rt):
            tail = pe[-len(rt):]
            exact = sum(a == b for a, b in zip(tail, rt))
            print(f"  tail matches rendered_tokens exactly: {exact}/{len(rt)}")
            # also try offsets
            best = None
            for off in range(0, max(1, len(pe) - len(rt) + 1)):
                seg = pe[off:off + len(rt)]
                m = sum(a == b for a, b in zip(seg, rt))
                if best is None or m > best[1]:
                    best = (off, m)
            print(f"  best-aligned window: offset={best[0]} matches={best[1]}/{len(rt)}")
        # predicted token counts
        pred = re.findall(r"\beval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*runs", txt)
        pred = [int(x) for x in pred]
        if pred:
            print(f"  predicted-token counts (all): {pred}")


if __name__ == "__main__":
    main()
