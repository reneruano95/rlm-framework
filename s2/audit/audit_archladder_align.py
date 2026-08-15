"""Align s2/results/arch_ladder_*.jsonl (which records `rendered_tokens`, the
full tokenized prompt length) against the matching llama-server stderr log's
`prompt eval time = ... / N tokens`.

If N < rendered_tokens for the same request, the server reused KV despite
cache_prompt:false. If N == rendered_tokens, it did not.

Offline only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(r"D:/PROJECTS/rlm-halo-framework")

PE = re.compile(r"print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens")
SEL = re.compile(r"get_availabl: id\s+(\d+) \| task (-?\d+) \| selected slot by (.*)$")
LAUNCH = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+) \| processing task")


def log_requests(path: Path):
    """(ts, slot, how, f_sim, f_keep, pe_tok) in wire order."""
    out = []
    cur = None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts = ln.split(" ", 1)[0]
        m = SEL.search(ln)
        if m:
            how = m.group(3)
            f = re.search(r"f_sim_best\s*=\s*([0-9.]+)", how)
            k = re.search(r"f_keep\s*=\s*([0-9.]+)", how)
            cur = dict(ts=ts, slot=int(m.group(1)), how=how.split(",")[0].strip(),
                       f_sim=f and float(f.group(1)), f_keep=k and float(k.group(1)),
                       task=None, pe=None)
            out.append(cur)
            continue
        m = LAUNCH.search(ln)
        if m and cur is not None and cur["task"] is None:
            cur["task"] = int(m.group(2))
            continue
        m = PE.search(ln)
        if m:
            t = int(m.group(2))
            for r in reversed(out):
                if r["task"] == t:
                    r["pe"] = int(m.group(4))
                    break
    return [r for r in out if r["pe"] is not None]


def align(jsonl: Path, log: Path) -> None:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    want = [r["rendered_tokens"] for r in rows]
    reqs = log_requests(log)
    got = [r["pe"] for r in reqs]
    print("=" * 96)
    print(f"{jsonl.name}: {len(rows)} recorded calls | {log.name}: {len(reqs)} server requests")
    # find the offset where the log's pe-token sequence matches the jsonl's
    best = None
    for off in range(0, len(got) - len(want) + 1):
        d = [got[off + i] - want[i] for i in range(len(want))]
        score = sum(1 for x in d if x == 0)
        if best is None or score > best[1]:
            best = (off, score, d)
    off, score, d = best
    print(f"best alignment: log offset {off}, exact matches {score}/{len(want)}")
    print(f"{'#':>3} {'size':>5} {'qtype':<8} {'cls':<16} {'rendered':>8} {'pe_tok':>7} "
          f"{'delta':>6} {'slot':>4} {'how':<16} {'f_sim':>6} {'f_keep':>6}")
    for i, r in enumerate(rows):
        q = reqs[off + i]
        print(f"{i:>3} {r['size']:>5} {r['qtype']:<8} {r['cls']:<16} {r['rendered_tokens']:>8} "
              f"{q['pe']:>7} {q['pe'] - r['rendered_tokens']:>+6} {q['slot']:>4} {q['how'][:16]:<16} "
              f"{('' if q['f_sim'] is None else f'{q['f_sim']:.3f}'):>6} "
              f"{('' if q['f_keep'] is None else f'{q['f_keep']:.3f}'):>6}")
    neg = sum(1 for i, r in enumerate(rows) if reqs[off + i]["pe"] < r["rendered_tokens"])
    print(f"\n  requests whose prompt-eval token count was BELOW the rendered prompt: {neg}/{len(rows)}")


if __name__ == "__main__":
    align(ROOT / "s2/results/arch_ladder_qwen-hybrid.jsonl", ROOT / "s2/logs/arch-qwen.err")
    align(ROOT / "s2/results/arch_ladder_gemma-fullattn.jsonl", ROOT / "s2/logs/arch-gemma.err")
