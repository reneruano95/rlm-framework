"""§7 optimization #4: MTP (and ngram) speculative decoding on the ROOT.

Root turns are decode-bound serial segments, so root decode t/s is the lever
§7 #4 targets at >=1.4x. Qwen3.8-27B ships the MTP head in its base quant
(`nextn_predict_layers = 1`), which Qwen3.6-27B's Q4_K_M did not, so the S5 root
swap is what makes this measurable at all.

TWO THINGS ARE MEASURED, AND THE SECOND IS THE ONE THAT DECIDES ADOPTION:

  1. decode t/s   -- the §7 #4 metric, target >=1.4x over no speculation.
  2. OUTPUT IDENTITY at temperature 0 -- I5 admits an optimization only if it is
     benchmark-neutral-or-better, and R4 requires MTP show *unchanged* success.
     Speculative decoding is free only if it is LOSSLESS: with greedy decoding
     and exact verification the emitted tokens must be byte-identical to the
     unspeculated run. If they are not, the flags are trading answers for
     speed and the gain is not free.

     Identity is asserted ONLY at temperature 0. At the production root sampling
     (0.7/0.8) llama.cpp's speculative path preserves the DISTRIBUTION but not
     the realized sample -- the RNG is consumed differently -- so a byte
     difference there is expected and proves nothing either way.

Run one arm per server launch, same prompts every time:

    # arm 0, baseline (no speculative flags)
    uv run --python 3.12 --no-project s2/mtp_bench.py --label base

    # arm 1, MTP at the spec's pinned draft-n
    #   ... --spec-type draft-mtp --spec-draft-n-max 2
    uv run --python 3.12 --no-project s2/mtp_bench.py --label mtp2

    # arm 2, MTP + ngram (user-supplied combination)
    #   ... --spec-type draft-mtp,ngram-mod --spec-draft-n-max 6 \
    #       --spec-draft-p-min 0.75 --spec-ngram-mod-n-match 24 \
    #       --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
    uv run --python 3.12 --no-project s2/mtp_bench.py --label mtp6-ngram

    uv run --python 3.12 --no-project s2/mtp_bench.py --compare
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"
OUT = Path(__file__).resolve().parent / "results"

# Root-shaped work: the root writes Python over `chunks` and reasons about it.
# Code is also where ngram speculation should pay best (highly repetitive
# token structure), so a code-free probe would understate the combined arm.
PROMPTS = [
    ("plan", "You are given a Python list `chunks` of ~400 strings. Write a "
             "function that finds every chunk containing a UUID and returns a "
             "dict mapping chunk index to the list of UUIDs found. Use only "
             "the standard library. Explain your approach first, then give the "
             "code."),
    ("loop", "Write a Python async function that dispatches one question per "
             "chunk to `await llm_query(chunk, question)`, serially, collects "
             "the answers into a list, retries once on an empty answer, and "
             "returns the list. Include type hints and a docstring."),
    ("reduce", "Given a list of per-chunk partial counts as dicts, write code "
               "that merges them into one total count per key, sorted "
               "descending by count, and prints the top 10 as a table."),
    ("prose", "Explain, in about 200 words and without code, why reading a "
              "long document in overlapping windows can produce a different "
              "answer than reading it all at once."),
]


def post(path: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def render(question: str, thinking: bool) -> str:
    msgs = [{"role": "system", "content": "You are a careful Python engineer."},
            {"role": "user", "content": question}]
    body: dict = {"messages": msgs}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        return post("/apply-template", body)["prompt"]
    except Exception:
        return post("/apply-template", {"messages": msgs})["prompt"]


def run(label: str, temp: float, n_predict: int, thinking: bool,
        reps: int) -> None:
    rows = []
    for name, q in PROMPTS:
        prompt = render(q, thinking)
        for rep in range(reps):
            r = post("/completion", {"prompt": prompt, "n_predict": n_predict,
                                     "temperature": temp, "top_p": 0.8,
                                     "seed": 1, "cache_prompt": False})
            t = r.get("timings", {})
            rows.append({
                "label": label, "prompt": name, "rep": rep, "temp": temp,
                "n_predict": n_predict, "thinking": thinking,
                "predicted_n": t.get("predicted_n"),
                "decode_tps": t.get("predicted_per_second"),
                "prompt_n": t.get("prompt_n"),
                "prefill_tps": t.get("prompt_per_second"),
                "content": r.get("content") or "",
            })
            print(f"  {name:<7} rep{rep} temp{temp} -> "
                  f"{t.get('predicted_per_second', 0):7.2f} tok/s  "
                  f"({t.get('predicted_n')} tok)")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"mtp_{label}_t{temp}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    med = statistics.median(r["decode_tps"] for r in rows if r["decode_tps"])
    print(f"\n  median decode {med:.2f} tok/s over {len(rows)} runs -> {p}")


def compare() -> None:
    files = sorted(OUT.glob("mtp_*.jsonl"))
    if not files:
        raise SystemExit("no arms recorded yet")
    arms: dict[tuple[str, float], list[dict]] = {}
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                arms.setdefault((r["label"], r["temp"]), []).append(r)

    print(f"\n{'arm':<16} {'temp':>5} {'n':>4} {'median tok/s':>13} "
          f"{'vs base':>8}  identity-vs-base")
    print("-" * 74)
    for temp in sorted({k[1] for k in arms}):
        base = arms.get(("base", temp))
        base_med = (statistics.median(r["decode_tps"] for r in base)
                    if base else None)
        for (label, t), rows in sorted(arms.items()):
            if t != temp:
                continue
            med = statistics.median(r["decode_tps"] for r in rows)
            speed = f"{med / base_med:.2f}x" if base_med else "-"
            # WITHIN-ARM control first: comparing an arm's bytes against the
            # baseline's only means something if the baseline reproduces
            # ITSELF. Measured on this stack it does not -- 0/4 at temperature
            # 0 with the same prompt and seed -- which §8 already anticipated
            # ("a fixed seed does not guarantee bitwise reproducibility --
            # seeds pin sampling identity, not numerics"). So a cross-arm byte
            # difference is NOT evidence that speculation is lossy, and this
            # column reports the control rather than a verdict it cannot
            # support. R4's actual gate is unchanged benchmark success (S4).
            prompts = {r["prompt"] for r in rows}
            by_rep = {(r["prompt"], r["rep"]): r["content"] for r in rows}
            self_same = sum(1 for p in prompts
                            if by_rep.get((p, 0)) == by_rep.get((p, 1)))
            ident = f"self {self_same}/{len(prompts)}"
            if base is not None and label != "base" and temp == 0.0:
                b = {(r["prompt"], r["rep"]): r["content"] for r in base}
                same = sum(1 for r in rows
                           if b.get((r["prompt"], r["rep"])) == r["content"])
                ident += f", vs base {same}/{len(rows)}"
            print(f"{label:<16} {t:>5} {len(rows):>4} {med:>13.2f} "
                  f"{speed:>8}  {ident}")
    print("\n§7 #4 target is >=1.40x on decode.")
    print("The identity column is a CONTROL, not a verdict: if an arm does not "
          "reproduce ITSELF at temp 0, a cross-arm byte difference says nothing "
          "about whether speculation changed the answer. R4's real gate is "
          "unchanged benchmark success, which needs S4.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
        return
    if not a.label:
        raise SystemExit("--label is required unless --compare")
    print(f"arm {a.label}: temp {a.temp}, n_predict {a.n_predict}, "
          f"thinking {a.thinking}, {a.reps} reps x {len(PROMPTS)} prompts")
    run(a.label, a.temp, a.n_predict, a.thinking, a.reps)


if __name__ == "__main__":
    main()
