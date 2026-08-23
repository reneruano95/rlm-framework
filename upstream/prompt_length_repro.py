#!/usr/bin/env python3
"""Reproducer: on the HIP/ROCm backend, a value placed more than roughly a
thousand tokens before the question cannot be repeated back. The same build's
Vulkan backend, same GPU, same model, same flags, same bytes, is clean.

Self-contained: standard library only. No fixtures, no corpus, no model of ours,
no repo checkout. Point it at a running `llama-server`.

    # terminal 1 -- HIP/ROCm build. -np 4 so the largest cell fits one slot: at
    # -np 8 the last cell is 5,779 tokens against 4,096 per slot and BOTH
    # backends return HTTP 400, which is a context limit and not this defect.
    llama-server -m Qwen2.5-7B-Instruct-Q8_0.gguf --host 127.0.0.1 --port 8081 \
      -c 32768 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048

    # terminal 2
    python prompt_length_repro.py --base http://127.0.0.1:8081

Then relaunch from the Vulkan build of the SAME commit and re-run.

THE TASK, and why it is this one. A 12-hex-digit marker is planted near the top
of a filler document and asked for at the bottom. Repeating a string you were
handed is the least a language model can do -- there is nothing to reason about,
nothing to count, and no world knowledge involved -- so a wrong answer is a
serving fault and not a capability limit. The marker is generated from `--seed`,
so it appears nowhere in any training corpus.

TWO TASKS THAT DO NOT WORK, recorded so nobody re-derives them:

  * COUNTING occurrences of a repeated sentence. A 7B model genuinely cannot do
    it, so both backends fail and the test discriminates nothing.
  * ASKING FOR A SUMMARY, or anything scored by eye. On a repetitive filler
    document, "repeating because degenerate" and "repeating because the source
    repeats" are the same string.

POSITIVE CONTROL. The shortest cell puts the marker ~60 tokens from the
question and must pass on every backend. If it fails, the model or the flags are
wrong and nothing below that line is evidence. This reproducer needs the control:
a first draft used a counting task and failed its own control on BOTH backends,
which is the only reason the result was not reported as a difference.

MEASURED, `b10375-ba360efe1`, AMD Radeon 8060S (gfx1151), ROCm 7.14 runtime DLLs,
`Qwen2.5-7B-Instruct-Q8_0`. Marker distance from the question, and whether the
model repeated it:

    marker back       80     584    1,200    1,760    2,936
    HIP / ROCm        ok    WRONG  HTTP 500  WRONG   HTTP 500
    Vulkan            ok      ok      ok       ok       ok

Both backends pass the 80-token control. HIP fails every longer cell, twice with
a 500. Vulkan repeats a 12-hex marker 2,936 tokens back without error.

`Meta-Llama-3.1-8B-Instruct-Q8_0` on the same box degenerates on HIP into "The
end of the end of the end of..." at ~1,200 tokens and is coherent on Vulkan at
every length. The onset is model-dependent -- ~584 tokens for Qwen2.5-7B, ~1,000
for a hybrid Qwen3.6-35B-A3B -- but the direction never is.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request

FILLER = ("The stowage draft was endorsed over to the vane holder in blank, and "
          "the tally clerk initialled the margin without further comment. ")

#: Filler repetitions per cell. ~26 tokens each, so the marker ends up roughly
#: 60 / 550 / 1,100 / 1,600 / 2,700 / 5,300 tokens before the question.
REPEATS = (2, 20, 42, 62, 104, 205)


def post(base: str, path: str, payload: dict, timeout: float):
    req = urllib.request.Request(f"{base}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def ask(base: str, prompt: str, timeout: float) -> tuple[str, str | None]:
    try:
        payload = post(base, "/v1/chat/completions", {
            "messages": [
                {"role": "system", "content": "Answer in plain text. Never emit "
                                              "JSON and never call a tool."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0, "top_p": 1, "max_tokens": 40, "stream": False,
        }, timeout)
        return payload["choices"][0]["message"]["content"], None
    except urllib.error.HTTPError as exc:
        return "", f"HTTP {exc.code}"
    except Exception as exc:                       # noqa: BLE001 - reported, not raised
        return "", f"{type(exc).__name__}: {exc}"


def ntokens(base: str, text: str) -> int | None:
    try:
        return len(post(base, "/tokenize", {"content": text}, 60)["tokens"])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-repeats", type=int, default=0,
                    help="drop cells above this repeat count, for a server "
                         "whose -c cannot hold the largest one")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"{'filler':>7} {'prompt tok':>11} {'marker back':>12}  {'verdict':<12} answer")
    print("-" * 92)
    rows = []
    for n in (r for r in REPEATS if not args.max_repeats or r <= args.max_repeats):
        marker = "".join(rng.choice("0123456789abcdef") for _ in range(12))
        head = f"MARKER: {marker}\n\n"
        tail = ("\n\nWhat value appears after MARKER: at the top of the text "
                "above? Reply with that value and nothing else.")
        prompt = head + FILLER * n + tail
        total = ntokens(args.base, prompt)
        back = ntokens(args.base, FILLER * n + tail)
        answer, err = ask(args.base, prompt, args.timeout)
        flat = " ".join((answer or "").split())
        if err:
            verdict = "ERROR"
        elif marker in flat.lower():
            verdict = "ok"
        else:
            words = flat.lower().split()
            degenerate = len(words) > 12 and len(set(words)) <= max(3, len(words) // 4)
            verdict = "DEGENERATE" if degenerate else "WRONG"
        rows.append((n, total, back, verdict))
        print(f"{n:>7} {str(total):>11} {str(back):>12}  {verdict:<12} "
              f"{(err or flat[:44] or '<empty>')!r}")

    print()
    if not rows or rows[0][3] != "ok":
        print("POSITIVE CONTROL FAILED: the shortest cell could not repeat a "
              "marker ~60 tokens away.")
        print("The model or the flags are wrong; nothing above this line is evidence.")
        return 2
    print(f"positive control: PASSED (marker {rows[0][2]} tokens back, repeated exactly)")
    bad = [r for r in rows[1:] if r[3] != "ok"]
    if not bad:
        print("every longer cell answered correctly -- this backend is clean")
        return 0
    first = min(bad, key=lambda r: r[2] or 0)
    print(f"FIRST FAILURE with the marker ~{first[2]} tokens before the question "
          f"({first[3]}); {len(bad)} of {len(rows) - 1} longer cells failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
