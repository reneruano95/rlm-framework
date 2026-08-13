"""The pre-registered decisive test for §7 #3's gate (a) — is it meetable?

THE CLAIM UNDER TEST, stated before the run. Gate (a) as written is "prefix
integrity: `tokens_cached >= tokenized-prefix-length` on every warm-slot call".
The v0.2.4 measurement says that is not meetable on this stack for a FIRST-SIGHT
chunk, and gives the mechanism: llama.cpp's cache reuse is quantized at the
`-ub` boundary (observed `cache_n` 38, then 550, at `ub: 512`), while the
rendered leaf prefix is only ~311 tokens — shorter than one ubatch, so it never
survives a chunk change. The spec pre-registers exactly one experiment before
§4 is rewritten:

    "relaunch the leaf at `-ub 128` and repeat the pinned two-chunk test — if
     `cache_n` jumps to ~311 the gate is meetable and the fix is a config
     change, not a spec edit."

So this script takes `--ub` as an argument and reports the same four numbers
whichever value the server was launched with. It does NOT launch or configure
the server: `-ub` is a launch flag, the leaf is a long-lived process, and a
script that claimed to set it would be lying about what produced the number.
`--ub` is RECORDED (and cross-checked against `config.yaml`, loudly, when they
disagree) so a record can never be mistaken for the other arm.

THE SEQUENCE, on ONE pinned slot:

  1. **A cold** — chunk A, first sight. `cache_n` here is the floor.
  2. **A warm** — the identical prompt again. This is the CONTROL: if `cache_n`
     does not now approach the whole prompt, the pin did not work or the slot
     was evicted, and step 3 would measure the wrong thing entirely. Without
     this control, `cache_n(B) = 0` is unattributable.
  3. **B after A** — different chunk, byte-identical system prefix, same slot.
     **`cache_n` here is the measurement**: does reuse reach the prefix length?
  4. **A after B** — the alternation cost, for the record.

VERDICT: `cache_n(B) >= prefix_tokens` means the prefix survived a chunk change
and gate (a) is meetable as written (at this `-ub`). Anything less means the
honest assertion for a first-sight chunk is `tokens_cached < prefix_len`, and
§4/§7 #3 (a) must be restated conditionally on chunk identity.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `uv run s2/run_ub_experiment.py`
    sys.path.insert(0, str(REPO_ROOT))

from s2.make_sweep_fixtures import fit_to_tokens, make_filler  # noqa: E402

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "ub.jsonl"

#: Small on purpose. The boundary being tested is the PREFIX's (~311 tokens vs
#: one ubatch); chunk length changes only what the experiment costs.
DEFAULT_CHUNK_TOKENS = 2048

QUESTION = ("What archive key does this excerpt record? Reply with the key "
            "itself and nothing else.")


def build_chunks(count, *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                 seed: int = 1) -> tuple[str, str]:
    """Two chunks of the same size and different content, deterministically.

    Different CONTENT is the whole point — if A and B shared a long head, a
    high `cache_n(B)` would be reuse of that head rather than of the system
    prefix. Two independently generated fillers share nothing past the first
    few tokens.
    """
    a = fit_to_tokens(make_filler(random.Random(f"ub:{seed}:A"),
                                  int(chunk_tokens / 1.7 * 1.4) + 80),
                      chunk_tokens, count)
    b = fit_to_tokens(make_filler(random.Random(f"ub:{seed}:B"),
                                  int(chunk_tokens / 1.7 * 1.4) + 80),
                      chunk_tokens, count)
    return a, b


def ub_verdict(*, prefix_tokens: int | None, cache_n_b: int | None,
               cache_n_a_warm: int | None, ub: int) -> dict[str, Any]:
    """The reading of the numbers, as a pure function so it is unit-testable
    and so the verdict cannot be typed from memory afterwards.

    `control_ok` gates everything: a warm re-query of the SAME prompt that does
    not reuse more than the prefix means the slot was not held, and no
    conclusion about `cache_n(B)` is available from that run.
    """
    if prefix_tokens is None or cache_n_b is None:
        return {"verdict": "INVALID", "control_ok": False,
                "reaches_prefix": None, "ub": ub,
                "detail": "missing prefix_tokens or cache_n for chunk B"}
    control_ok = cache_n_a_warm is not None and cache_n_a_warm > prefix_tokens
    reaches = cache_n_b >= prefix_tokens
    if not control_ok:
        verdict = "INVALID"
        detail = (f"warm control re-used only {cache_n_a_warm} tokens, not more "
                  f"than the {prefix_tokens}-token prefix: the slot was not "
                  f"held, so cache_n(B)={cache_n_b} is unattributable")
    elif reaches:
        verdict = "GATE-A MEETABLE"
        detail = (f"cache_n(B)={cache_n_b} >= prefix {prefix_tokens} at ub={ub}: "
                  f"the byte-identical prefix survived a chunk change, so §7 #3 "
                  f"(a) holds as written and the fix is a config change")
    else:
        verdict = "GATE-A NOT MEETABLE"
        detail = (f"cache_n(B)={cache_n_b} < prefix {prefix_tokens} at ub={ub}: "
                  f"reuse is quantized below the prefix length, so gate (a) must "
                  f"be restated conditionally on chunk identity")
    return {"verdict": verdict, "control_ok": control_ok,
            "reaches_prefix": reaches, "ub": ub,
            "prefix_tokens": prefix_tokens, "cache_n_b": cache_n_b,
            "cache_n_a_warm": cache_n_a_warm,
            "ub_multiples_reused": (cache_n_b // ub) if ub else None,
            "detail": detail}


async def run_experiment(caller, *, ub: int, slot: int, chunk_a: str,
                         chunk_b: str, seed: int = 1, echo=print) -> dict:
    """The four calls, in order, on one pinned slot."""
    steps = [("A_cold", chunk_a), ("A_warm", chunk_a),
             ("B_after_A", chunk_b), ("A_after_B", chunk_a)]
    calls: list[dict] = []
    for label, chunk in steps:
        answer = await caller.ask(question=QUESTION, chunk=chunk, seed=seed,
                                  id_slot=slot)
        rec = {"step": label, **answer.as_record()}
        calls.append(rec)
        echo(f"[{label}] cache_n={rec['tokens_cached']} of {rec['tokens_in']} "
             f"prompt tokens, slot {rec['slot_id']}, prefill "
             f"{rec['prefill_ms']} ms, {rec['wall_s']} s")

    by_step = {c["step"]: c for c in calls}
    verdict = ub_verdict(prefix_tokens=caller.prefix_tokens,
                         cache_n_b=by_step["B_after_A"]["tokens_cached"],
                         cache_n_a_warm=by_step["A_warm"]["tokens_cached"],
                         ub=ub)
    cold_ms = by_step["A_cold"]["prefill_ms"] or 0.0
    warm_ms = by_step["A_warm"]["prefill_ms"] or 0.0
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": "ub-prefix-survival", "ub": ub, "slot": slot, "seed": seed,
        "prefix_tokens": caller.prefix_tokens,
        "cache_n": {c["step"]: c["tokens_cached"] for c in calls},
        "cold_warm_prefill_ratio": (round(cold_ms / warm_ms, 2)
                                    if warm_ms else None),
        "calls": calls,
        **verdict,
    }


async def _amain(args) -> int:
    from rlm.config import load_config
    from s1.make_fixtures import leaf_counter
    from s2.leafcall import PinnedLeafCaller

    cfg = load_config(Path(args.config))
    if cfg.servers.leaf.ub != args.ub:
        print(f"NOTE: --ub {args.ub} does not match config.yaml's "
              f"servers.leaf.ub ({cfg.servers.leaf.ub}). `-ub` is a LAUNCH "
              f"flag: this script records what you tell it the running server "
              f"was launched with and cannot verify it. Confirm the launch "
              f"line before trusting this record.", file=sys.stderr)

    count = leaf_counter(cfg.servers.leaf.port)
    chunk_a, chunk_b = build_chunks(count, chunk_tokens=args.chunk_tokens,
                                    seed=args.seed)
    caller = PinnedLeafCaller.from_config(cfg)
    try:
        prefix_tokens = await caller.prepare()
        print(f"leaf prefix renders to {prefix_tokens} tokens; ub={args.ub}; "
              f"chunks {count(chunk_a)}/{count(chunk_b)} tokens; slot {args.slot}")
        record = await run_experiment(caller, ub=args.ub, slot=args.slot,
                                      chunk_a=chunk_a, chunk_b=chunk_b,
                                      seed=args.seed)
    finally:
        await caller.aclose()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\ncache_n by step: {record['cache_n']}")
    print(f"prefix_tokens={record['prefix_tokens']} "
          f"cache_n(B)={record['cache_n_b']} "
          f"reaches_prefix={record['reaches_prefix']}")
    print(f"{record['verdict']}: {record['detail']}")
    print(f"appended to {args.out}")
    return 0 if record["verdict"] != "INVALID" else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="§7 #3 gate (a): does the leaf prefix survive a chunk "
                    "change on a pinned slot at this -ub?")
    parser.add_argument("--ub", type=int, required=True,
                        help="the -ub value the RUNNING leaf server was "
                             "launched with (512 and 128 are the two pre-"
                             "registered arms). Recorded, not applied.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=RUNS_PATH)
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
