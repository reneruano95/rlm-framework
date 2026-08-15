"""Join the arch_ladder server log against its own results file.

Question: for each completion request, how many tokens did the server actually
evaluate, versus how many tokens the rendered prompt contained? A shortfall means
the slot kept KV cells from whatever it held before -- and the arch_ladder client
sent cache_prompt:false, so any shortfall is reuse the client did not ask for.

Also reports which slot served each request and how the slot was chosen (LRU vs
LCP similarity, with f_keep), so the leaking requests can be traced to a donor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = [("arch-qwen.err", "arch_ladder_qwen-hybrid.jsonl"),
         ("arch-gemma.err", "arch_ladder_gemma-fullattn.jsonl")]

RE_AVAIL = re.compile(r"slot get_availabl: id\s+(\d+) \| task -1 \| (.*)")
RE_LAUNCH = re.compile(r"slot launch_slot_: id\s+(\d+) \| task (\d+) \|")
RE_PROMPT = re.compile(r"slot print_timing: id\s+(\d+) \| task (\d+) \| "
                       r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens")
RE_RELEASE = re.compile(r"slot\s+release: id\s+(\d+) \| task (\d+) \| "
                        r"stop processing: n_tokens = (\d+)")


def parse_log(path: Path) -> list[dict]:
    """One record per completion, in service order."""
    recs: list[dict] = []
    by_task: dict[int, dict] = {}
    pending_choice = None
    for line in path.read_text(errors="replace").splitlines():
        m = RE_AVAIL.search(line)
        if m:
            pending_choice = m.group(2).strip()
            continue
        m = RE_LAUNCH.search(line)
        if m:
            slot, task = int(m.group(1)), int(m.group(2))
            r = {"slot": slot, "task": task, "choice": pending_choice,
                 "eval_tokens": None, "n_tokens": None}
            by_task[task] = r
            recs.append(r)
            pending_choice = None
            continue
        m = RE_PROMPT.search(line)
        if m and int(m.group(2)) in by_task:
            by_task[int(m.group(2))]["eval_tokens"] = int(m.group(3))
            continue
        m = RE_RELEASE.search(line)
        if m and int(m.group(2)) in by_task:
            by_task[int(m.group(2))]["n_tokens"] = int(m.group(3))
    # only records that actually ran a prompt eval are completions
    return [r for r in recs if r["eval_tokens"] is not None]


def main() -> None:
    for logname, resname in PAIRS:
        log = HERE / "logs" / logname
        res = HERE / "results" / resname
        if not log.exists() or not res.exists():
            print(f"skip {logname}: missing")
            continue
        recs = parse_log(log)
        rows = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]

        print(f"\n{'='*78}\n{logname}  ->  {resname}")
        print(f"completions in log: {len(recs)}   rows in results: {len(rows)}")
        if len(recs) != len(rows):
            print("  NOTE: counts differ; aligning on the LAST len(rows) completions")
            recs = recs[-len(rows):] if len(recs) > len(rows) else recs

        print(f"\n{'#':>3} {'size/t':>9} {'qtype':<8} {'cls':<16} "
              f"{'slot':>4} {'rendered':>9} {'evaluated':>10} {'skipped':>8} "
              f"{'slot n_tok':>10}  chosen-by")
        for i, (rec, row) in enumerate(zip(recs, rows)):
            rendered = row["rendered_tokens"]
            skipped = rendered - rec["eval_tokens"]
            flag = " <== LEAK ROW" if row["cls"] == "ANSWERED_ANYWAY" else ""
            choice = (rec["choice"] or "")[:46]
            print(f"{i:>3} {str(row['size'])+'/t'+str(row['trial']):>9} "
                  f"{row['qtype']:<8} {row['cls']:<16} "
                  f"{rec['slot']:>4} {rendered:>9} {rec['eval_tokens']:>10} "
                  f"{skipped:>8} {str(rec['n_tokens']):>10}  {choice}{flag}")

        skipped_all = [row["rendered_tokens"] - rec["eval_tokens"]
                       for rec, row in zip(recs, rows)]
        print(f"\n  tokens skipped per request: min={min(skipped_all)} "
              f"max={max(skipped_all)} mean={sum(skipped_all)/len(skipped_all):.1f}")
        print(f"  requests with ANY skipped tokens: "
              f"{sum(1 for s in skipped_all if s > 0)}/{len(skipped_all)}")
        print("  (client sent cache_prompt:false for every one of these)")


if __name__ == "__main__":
    main()
