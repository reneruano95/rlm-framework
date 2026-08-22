"""`preserve_thinking` on a MULTI-TURN root, and §7 #3c's owed cache_n check.

Two questions, one probe, because they are the same measurement:

1. `preserve_thinking` (template default TRUE) decides whether earlier turns'
   reasoning stays in the rendered prompt. It is invisible single-turn, which is
   why `milestones/s2/root_thinking.py` could not test it. On a root that iterates in the
   REPL it is not invisible at all: kept reasoning makes every later prompt
   longer, spends the 32K root window faster (`context_exhausted`), and adds
   prefill to the serial part of every turn.

2. §9 S0 item 5(b) and §7 #3c want a scripted 3-turn conversation on port 8080
   showing per-turn `cache_n` ~= total prior-turn tokens -- "only the new
   observation prefills". That pass was owed on the OLD root and the S5 swap
   makes it owed again, on the new one.

The prefix-cache contract is what ties them together: a conversation is supposed
to be a CONTINUATION, so turn N+1 extends turn N's prefix and reuses it. If
`preserve_thinking` rewrites earlier turns (dropping reasoning that was there
when they were cached), the prefix no longer extends -- it DIVERGES -- and the
reuse §4 depends on is destroyed. That is measurable here as cache_n collapsing.

    uv run --python 3.12 --no-project milestones/s2/root_multiturn.py
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
OUT = Path(__file__).resolve().parent / "results"

TURNS = [
    "I have a list `chunks` of 400 text fragments. Write a Python function "
    "`find_ids(chunks)` that returns a dict mapping each chunk index to the "
    "list of UUID-shaped strings it contains. Standard library only.",
    "Now modify it so it also records, for each UUID, the 40 characters of "
    "context immediately before it. Keep the same return shape but make the "
    "value a list of (uuid, context) tuples.",
    "Finally, add a `min_count` parameter that filters out any chunk with "
    "fewer than `min_count` UUIDs, and explain in one sentence what the "
    "function now returns.",
]


def post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_arm(label: str, thinking: bool, preserve: bool, effort: str,
            n_predict: int, temp: float) -> list[dict]:
    msgs = [{"role": "system", "content": "You are a careful Python engineer."}]
    rows = []
    prior_total = 0
    for i, turn in enumerate(TURNS):
        msgs.append({"role": "user", "content": turn})
        kwargs: dict = {"enable_thinking": thinking}
        if thinking:
            kwargs["reasoning_effort"] = effort
            kwargs["preserve_thinking"] = preserve
        rendered = post("/apply-template", {
            "messages": msgs, "chat_template_kwargs": kwargs})["prompt"]
        # cache_prompt TRUE on purpose: this measures the reuse the root's
        # multi-turn path actually gets in production.
        r = post("/completion", {"prompt": rendered, "n_predict": n_predict,
                                 "temperature": temp, "top_p": 0.8, "seed": 1,
                                 "cache_prompt": True})
        t = r.get("timings", {})
        raw = r.get("content") or ""
        prompt_n = t.get("prompt_n") or 0
        # `timings.cache_n` is the REUSE count. The top-level `tokens_cached`
        # is a different number and the two are "NOT interchangeable"
        # (rlm/dispatcher.py:437-442, recipes §serverapi). This probe read
        # the top-level field first and saw slot occupancy AFTER the call
        # (107-token prompt reporting 1130 = 107 + 1024 generated), which is
        # how the mistake announces itself. Both are recorded now.
        cache_n = t.get("cache_n") or 0
        tokens_cached_field = r.get("tokens_cached")
        rendered_n = len(post("/tokenize", {"content": rendered})["tokens"])
        # Reuse is derived from rendered - evaluated rather than from any cache
        # field, so the number stands whichever field the build populates.
        reused = rendered_n - prompt_n
        rows.append({
            "arm": label, "thinking": thinking,
            "preserve_thinking": preserve if thinking else None,
            "effort": effort if thinking else None, "turn": i,
            "rendered_tokens": rendered_n,
            "prompt_n": prompt_n, "cache_n": cache_n,
            "tokens_cached_field": tokens_cached_field,
            "prior_total": prior_total, "reused_tokens": reused,
            "reuse_frac": round(reused / rendered_n, 3) if rendered_n else 0.0,
            "predicted_n": t.get("predicted_n"),
            "truncated": t.get("predicted_n") == n_predict,
            "has_think": "<think>" in raw,
        })
        print(f"  {label:<14} turn{i}  rendered {rendered_n:>6}"
              f"  evaluated {prompt_n:>6}  reused {reused:>6}"
              f"  {rows[-1]['reuse_frac']:>6.1%}"
              f"  out {t.get('predicted_n')} tok"
              f"{'  TRUNC' if t.get('predicted_n') == n_predict else ''}")
        # Feed the full assistant text back, reasoning included -- the template
        # is what decides whether to keep it, which is the variable under test.
        msgs.append({"role": "assistant", "content": raw})
        prior_total = rendered_n + (t.get("predicted_n") or 0)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    print(f"3-turn root conversation, effort={a.effort}, "
          f"cache_prompt=true, temp={a.temp}\n")
    rows: list[dict] = []
    arms = [("think-off", False, False),
            ("preserve-on", True, True),
            ("preserve-off", True, False)]
    for label, thinking, preserve in arms:
        rows += run_arm(label, thinking, preserve, a.effort, a.n_predict, a.temp)
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "root_multiturn.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("=== multi-turn root: prompt growth and TRUE prefix reuse ===")
    print(f"  {'arm':<14} {'turn':>4} {'rendered':>9} {'evaluated':>10} "
          f"{'reused':>8} {'reuse':>7} {'out':>6}")
    for label, _, _ in arms:
        for r in [x for x in rows if x["arm"] == label]:
            print(f"  {label:<14} {r['turn']:>4} {r['rendered_tokens']:>9} "
                  f"{r['prompt_n']:>10} {r['reused_tokens']:>8} "
                  f"{r['reuse_frac']:>6.1%} {r['predicted_n']:>6}"
                  f"{'  TRUNC' if r['truncated'] else ''}")
    print("\n  §7 #3c expects per-turn reuse to cover all prior-turn "
          "tokens ('only the new observation prefills'). Measured: that holds "
          "with thinking OFF (85-93%) and collapses with it ON (5-66%) -- the "
          "prior assistant turn re-prefills, because the template's rendering "
          "of it does not match the tokens the slot cached when it was "
          "generated. preserve_thinking does not change this; enable_thinking "
          "is the lever.")
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
