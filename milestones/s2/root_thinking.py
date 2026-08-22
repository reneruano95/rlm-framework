"""Should the ROOT think, and at what effort? (§4/D15 currently says no.)

Production sets `enable_thinking: false` for both roles. That was decided on the
LEAF, where thinking measurably burns the whole generation budget on a preamble
and never emits an answer. The root is a different job -- it writes Python and
reasons about a plan -- so the setting is inherited, not measured, and this
measures it.

Qwen3.8-27B's chat template exposes three kwargs (read straight out of the GGUF,
`milestones/s2/gguf_compare.py`):
    enable_thinking     default TRUE
    preserve_thinking   default TRUE  (keeps earlier turns' reasoning)
    reasoning_effort    default 'xhigh', one of ('xhigh', 'medium', 'low')

The default matters: turning thinking on without naming an effort silently buys
the most expensive mode. 'xhigh' and 'low' each inject an explicit instruction
into the system block; 'medium' injects none.

THE TASK SHAPE IS THE ROOT'S OWN. Each item is a small record set plus a
filter-and-aggregate question -- the aggregation shape the root has to
orchestrate -- with a ground truth computed in Python, so scoring is exact match
on an integer and needs no judge. Multi-step filtering then arithmetic is
precisely where reasoning should pay if it pays anywhere.

Cost is reported beside accuracy, because §7's rule is wall-clock at fixed
quality (I5): thinking that buys +1 correct answer for 6x the tokens is not
obviously a win on a serial, decode-bound root turn.

One server launch serves every arm -- these are per-request template kwargs.
    uv run --python 3.12 --no-project milestones/s2/root_thinking.py --tasks 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
OUT = Path(__file__).resolve().parent / "results"

CATS = ["alpha", "beta", "gamma", "delta"]
REGIONS = ["north", "south", "east", "west"]


def post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def make_task(seed: int, n_records: int = 24):
    """A filter-and-aggregate question with a computed ground truth."""
    rng = random.Random(seed)
    recs = [{"id": i,
             "cat": rng.choice(CATS),
             "region": rng.choice(REGIONS),
             "value": rng.randint(10, 99),
             "active": rng.random() < 0.6}
            for i in range(n_records)]
    cat = rng.choice(CATS)
    region = rng.choice(REGIONS)
    truth = sum(r["value"] for r in recs
                if r["cat"] == cat and r["region"] != region and r["active"])
    table = "\n".join(
        f"  {{id: {r['id']:>2}, cat: {r['cat']:<5}, region: {r['region']:<5}, "
        f"value: {r['value']:>2}, active: {str(r['active']).lower()}}}"
        for r in recs)
    q = (f"Here are {n_records} records:\n\n{table}\n\n"
         f"Compute the SUM of `value` over every record whose `cat` is "
         f"'{cat}' AND whose `region` is NOT '{region}' AND whose `active` is "
         f"true.\n\nReply with the integer only, nothing else.")
    return q, truth


def render(question: str, thinking: bool, effort: str | None) -> str:
    msgs = [{"role": "system", "content": "You are a careful Python engineer."},
            {"role": "user", "content": question}]
    kwargs: dict = {"enable_thinking": thinking}
    if thinking and effort:
        kwargs["reasoning_effort"] = effort
    return post("/apply-template",
                {"messages": msgs, "chat_template_kwargs": kwargs})["prompt"]


ANSWER_RE = re.compile(r"-?\d[\d,]*")


def parse(raw: str) -> tuple[int | None, bool]:
    """Return (answer, thinking_was_closed). With thinking on the answer sits
    after the LAST </think>; an unclosed block means the budget ran out before
    the model ever answered -- the exact failure the leaf showed."""
    closed = "</think>" in raw
    tail = raw.rsplit("</think>", 1)[1] if closed else raw
    if not closed and "<think>" in raw:
        return None, False
    nums = ANSWER_RE.findall(tail)
    if not nums:
        return None, closed
    return int(nums[-1].replace(",", "")), closed


ARMS = [
    ("off", False, None),          # production today (§4/D15)
    ("low", True, "low"),
    ("medium", True, "medium"),
    ("xhigh", True, "xhigh"),      # the template's DEFAULT when thinking is on
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=1536)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tasks = [make_task(4000 + i) for i in range(a.tasks)]
    print(f"{a.tasks} filter-and-aggregate tasks, n_predict {a.n_predict}, "
          f"temp {a.temp}, {len(ARMS)} arms\n")

    rows = []
    for label, thinking, effort in ARMS:
        for i, (q, truth) in enumerate(tasks):
            prompt = render(q, thinking, effort)
            t0 = time.monotonic()
            r = post("/completion", {"prompt": prompt, "n_predict": a.n_predict,
                                     "temperature": a.temp, "top_p": 0.8,
                                     "seed": 1, "cache_prompt": False})
            wall = time.monotonic() - t0
            raw = r.get("content") or ""
            got, closed = parse(raw)
            t = r.get("timings", {})
            ok = got == truth
            rows.append({"arm": label, "task": i, "truth": truth, "got": got,
                         "ok": ok, "thinking_closed": closed,
                         "predicted_n": t.get("predicted_n"),
                         "truncated": t.get("predicted_n") == a.n_predict,
                         "wall_s": round(wall, 2),
                         "decode_tps": t.get("decode_tps") or t.get("predicted_per_second"),
                         "raw_tail": raw[-160:]})
            mark = "ok " if ok else "MISS"
            print(f"  {label:<7} t{i} {mark} truth={truth:<5} got={str(got):<6} "
                  f"{t.get('predicted_n'):>5} tok {wall:>6.1f}s"
                  f"{'  TRUNCATED' if t.get('predicted_n') == a.n_predict else ''}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = Path(a.out or OUT / "root_thinking.jsonl")
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== root thinking / effort, n={a.tasks} per arm ===")
    print(f"  {'arm':<8} {'correct':>9} {'med tok':>8} {'med wall':>9} "
          f"{'truncated':>10} {'tok/correct':>12}")
    base_wall = None
    for label, _, _ in ARMS:
        sel = [r for r in rows if r["arm"] == label]
        ok = sum(r["ok"] for r in sel)
        med_tok = statistics.median(r["predicted_n"] or 0 for r in sel)
        med_wall = statistics.median(r["wall_s"] for r in sel)
        trunc = sum(r["truncated"] for r in sel)
        if base_wall is None:
            base_wall = med_wall
        per = (sum(r["predicted_n"] or 0 for r in sel) / ok) if ok else float("inf")
        print(f"  {label:<8} {ok:>6}/{len(sel):<2} {med_tok:>8.0f} "
              f"{med_wall:>8.1f}s {trunc:>10} {per:>12.0f}")
    print(f"\n  wrote {p}")
    print("  I5 reads this as wall-clock at FIXED QUALITY: an arm only wins if "
          "it is more correct, and its cost is stated beside the win.")


if __name__ == "__main__":
    main()
