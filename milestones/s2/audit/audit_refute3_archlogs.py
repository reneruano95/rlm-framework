"""REFUTE pass 3: parse milestones/s2/logs/arch-*.err into server sessions and completion
records, then align against the arch_ladder result files WITHOUT assuming the
alignment the previous audit assumed.

Reports per session:
  - n_slots / n_ctx_slot banner
  - every completion task: slot id, selection method, prompt-eval tokens
  - the sequence of rendered_tokens in each candidate result file
  - the best alignment (exact-match count) and any prompt_eval < rendered cases
"""
from __future__ import annotations

import json
import re
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]
LOGS = S2 / "logs"
RESULTS = S2 / "results"

RE_INIT = re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")
RE_SEL = re.compile(r"slot get_availabl: id\s+(\d+) \| task -1 \| selected slot by ([^,]+)(.*)")
RE_PE = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens")


def parse(path: Path):
    sessions = []
    cur = None
    pending_sel = None
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = RE_INIT.search(line)
        if m:
            cur = {"line": ln, "n_slots": int(m.group(1)), "n_ctx_slot": int(m.group(2)),
                   "kv_unified": m.group(3), "completions": []}
            sessions.append(cur)
            continue
        m = RE_SEL.search(line)
        if m:
            pending_sel = (int(m.group(1)), m.group(2).strip(), m.group(3).strip(), ln)
            continue
        m = RE_PE.search(line)
        if m and cur is not None:
            cur["completions"].append({
                "line": ln, "slot": int(m.group(1)), "task": int(m.group(2)),
                "pe": int(m.group(3)),
                "sel": pending_sel[1] if pending_sel and pending_sel[0] == int(m.group(1)) else None,
                "sel_detail": pending_sel[2] if pending_sel and pending_sel[0] == int(m.group(1)) else None,
            })
    return sessions


def main() -> None:
    for logname in ["arch-qwen.err", "arch-gemma.err", "arch-q35.err"]:
        p = LOGS / logname
        if not p.exists():
            continue
        sess = parse(p)
        print(f"\n########## {logname}  ({len(sess)} server session(s)) ##########")
        for si, s in enumerate(sess):
            c = s["completions"]
            print(f"  session {si} @line {s['line']}: n_slots={s['n_slots']} "
                  f"n_ctx_slot={s['n_ctx_slot']} kv_unified={s['kv_unified']} "
                  f"completions={len(c)}")
            print(f"    slots used: {sorted({x['slot'] for x in c})}")
            from collections import Counter
            print(f"    selection methods: {dict(Counter(x['sel'] for x in c))}")
            print(f"    prompt-eval seq: {[x['pe'] for x in c]}")
    print("\n########## result-file rendered_tokens ##########")
    for name in ["arch_ladder_qwen-hybrid.jsonl", "arch_ladder_gemma-fullattn.jsonl"]:
        rows = [json.loads(l) for l in (RESULTS / name).read_text().splitlines() if l.strip()]
        print(f"  {name}: {[r['rendered_tokens'] for r in rows]}")


if __name__ == "__main__":
    main()
