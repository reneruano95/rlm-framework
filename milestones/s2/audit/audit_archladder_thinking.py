"""Did arch_ladder's two arms actually run in the SAME thinking mode?

`arch_ladder.apply_template` tries `chat_template_kwargs={"enable_thinking":
False}` and SILENTLY falls back to no kwarg on any exception, and it strips
everything up to the last `</think>` before recording the answer. So the JSONL
cannot show whether a reasoning block was emitted. The server log can: compare
PREDICTED tokens per call against the length of the recorded answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

S2 = Path(__file__).resolve().parents[1]

EVAL = re.compile(r"eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
PROMPT = re.compile(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
LCP = re.compile(r"selected slot by LCP similarity, f_sim_best = ([\d.]+)")
LRU = re.compile(r"selected slot by (LRU)")
SLOT = re.compile(r"slot\s+\w+:\s+id\s+(\d+)")


def arm(logname: str, resname: str) -> None:
    log = (S2 / "logs" / logname).read_text(encoding="utf-8", errors="replace")
    rows = [json.loads(l) for l in (S2 / "results" / resname)
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    prompts, evals, slots = [], [], []
    for line in log.splitlines():
        m = PROMPT.search(line)
        if m:
            prompts.append(int(m.group(1)))
            ms = SLOT.search(line)
            if ms:
                slots.append(int(ms.group(1)))
        m = EVAL.search(line)
        if m and "prompt eval time" not in line:
            evals.append(int(m.group(1)))
    sims = [float(x) for x in LCP.findall(log)]

    print(f"\n=== {logname} / {resname} ===")
    print(f"  completions in log: {len(prompts)}   records in jsonl: {len(rows)}")
    print(f"  slots the server chose: {sorted(set(slots))}")
    print(f"  routing decisions: {len(LRU.findall(log))} LRU, {len(sims)} LCP "
          f"(f_sim_best min {min(sims):.3f} max {max(sims):.3f})" if sims else "")
    print(f"  predicted tokens per call: {evals}")
    print(f"  prompt-eval tokens per call: {prompts}")
    print(f"  rendered tokens from jsonl : {[r['rendered_tokens'] for r in rows]}")
    print(f"  answer chars from jsonl    : {[len(r['answer']) for r in rows]}")
    # a 36-char UUID is ~20 tokens; 'NONE' is 1-2. Anything far above that was
    # generated and then discarded by the </think> strip.
    if len(evals) == len(rows):
        over = [(e, len(r["answer"])) for e, r in zip(evals, rows)
                if e > max(4, len(r["answer"]) // 2)]
        print(f"  calls whose PREDICTED tokens exceed ~half the answer's chars: "
              f"{len(over)}/{len(rows)}  {over[:8]}")


def main() -> None:
    arm("arch-qwen.err", "arch_ladder_qwen-hybrid.jsonl")
    arm("arch-gemma.err", "arch_ladder_gemma-fullattn.jsonl")


if __name__ == "__main__":
    main()
