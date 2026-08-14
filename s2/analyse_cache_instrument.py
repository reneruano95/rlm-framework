"""Tables for `s2/CACHE-INSTRUMENT.md`, read from `results/cache_instrument.jsonl`.

Nothing here re-measures anything: every number is either a field the runner
recorded or an arithmetic combination of such fields, so the report and the raw
records cannot drift apart.

THE TWO TRUTH MODELS, which is the whole argument. `truth_lcp_prev_same_slot`
is what §4 implies ("the prompt cache is per-slot — there is no cross-slot
sharing"): the reuse ceiling is the longest common token prefix with whatever
that slot last held. `truth_lcp_best_any_slot` drops the per-slot assumption:
the ceiling is the best common prefix against EVERY prompt the process has
served. `cache_n` is scored against both; the one it tracks is the one that
describes this build.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results" / "cache_instrument.jsonl"


def load(path: Path, conditions: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if conditions:
        rows = [r for r in rows if r["condition"] in conditions]
    return rows


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def fmt(x: float, nd: int = 1) -> str:
    return "-" if x != x else f"{x:,.{nd}f}"


# --------------------------------------------------------------------------- #


def table_cases(rows: list[dict[str, Any]], condition: str) -> str:
    """Reported vs true, per case and step, in both directions."""
    sel = [r for r in rows if r["condition"] == condition and r["case"] != "diverge-sweep"]
    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    order: list[tuple[str, int, str]] = []
    for r in sel:
        key = (r["case"], r["step"], r["role"])
        if key not in groups:
            order.append(key)
        groups[key].append(r)
    out = [
        "| case | step | n | prompt tokens | reported `cache_n` | true LCP "
        "(prev on slot) | true LCP (any slot) | err vs per-slot | err vs any-slot "
        "| `prompt_n` | prefill ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in order:
        rs = groups[key]
        case, step, role = key
        rep = med([r["reported_cache_n"] for r in rs])
        tp = med([r["truth_lcp_prev_same_slot"] for r in rs])
        ta = med([r["truth_lcp_best_any_slot"] for r in rs])
        out.append(
            f"| `{case}` | {step} {role} | {len(rs)} | "
            f"{fmt(med([r['prompt_tokens_true'] for r in rs]), 0)} | "
            f"**{fmt(rep, 0)}** | {fmt(tp, 0)} | {fmt(ta, 0)} | "
            f"{fmt(rep - tp, 0)} | {fmt(rep - ta, 0)} | "
            f"{fmt(med([r['reported_prompt_n'] for r in rs]), 0)} | "
            f"{fmt(med([r['prompt_ms'] for r in rs]))} |")
    return "\n".join(out)


def table_conditions(rows: list[dict[str, Any]]) -> str:
    """The decisive comparison: the cross-slot cases under each host-cache flag."""
    cases = ("cross-slot", "cross-slot-lag", "layoutC-elsewhere", "requery",
             "identical", "prefix-only")
    conds = sorted({r["condition"] for r in rows})
    out = ["| case / step | " + " | ".join(f"`{c}`" for c in conds) + " |",
           "|---" * (len(conds) + 1) + "|"]
    keyorder: list[tuple[str, int, str]] = []
    for r in rows:
        if r["case"] in cases:
            k = (r["case"], r["step"], r["role"])
            if k not in keyorder:
                keyorder.append(k)
    keyorder.sort(key=lambda k: (cases.index(k[0]), k[1]))
    for case, step, role in keyorder:
        cells = []
        for c in conds:
            rs = [r for r in rows if r["condition"] == c and r["case"] == case
                  and r["step"] == step]
            if not rs:
                cells.append("-")
                continue
            cells.append(f"{fmt(med([r['reported_cache_n'] for r in rs]), 0)} / "
                         f"{fmt(med([r['truth_lcp_prev_same_slot'] for r in rs]), 0)} / "
                         f"{fmt(med([r['truth_lcp_best_any_slot'] for r in rs]), 0)}")
        out.append(f"| `{case}` s{step} {role} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_diverge(rows: list[dict[str, Any]]) -> str:
    sel = [r for r in rows if r["case"] == "diverge-sweep" and r["step"] == 1]
    conds = sorted({r["condition"] for r in sel})
    out = ["| divergence point (true LCP) | prompt tokens | " +
           " | ".join(f"`{c}` cache_n" for c in conds) + " |",
           "|---" * (len(conds) + 2) + "|"]
    roles = sorted({r["role"] for r in sel})
    for role in roles:
        rs0 = [r for r in sel if r["role"] == role]
        tp = med([r["truth_lcp_prev_same_slot"] for r in rs0])
        pt = med([r["prompt_tokens_true"] for r in rs0])
        cells = [fmt(med([r["reported_cache_n"] for r in rs0
                          if r["condition"] == c]), 0) for c in conds]
        out.append(f"| {role} → {fmt(tp, 0)} | {fmt(pt, 0)} | " +
                   " | ".join(cells) + " |")
    return "\n".join(out)


def calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cold prefill tokens/second, and the spread that becomes the error bar.

    Only rows the server itself called cold (`cache_n == 0`) enter, and the rate
    is per row (`prompt_n / prompt_ms`) rather than a regression, because the
    instrument being calibrated converts ONE call's `prompt_ms` into tokens.
    """
    out: dict[str, Any] = {}
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond
               and r["case"] == "calibration" and r["reported_cache_n"] == 0]
        rates = [r["reported_prompt_n"] / (r["prompt_ms"] / 1000.0) for r in sel]
        by_len = defaultdict(list)
        for r in sel:
            by_len[r["chunk_tokens_target"]].append(
                r["reported_prompt_n"] / (r["prompt_ms"] / 1000.0))
        out[cond] = {
            "n": len(sel),
            "rate_median": med(rates),
            "rate_min": min(rates) if rates else float("nan"),
            "rate_max": max(rates) if rates else float("nan"),
            "by_len": {k: (med(v), len(v)) for k, v in sorted(by_len.items())},
        }
    return out


def instrument_check(rows: list[dict[str, Any]], rate: float) -> str:
    """How well does `prompt_ms x cold rate` recover the tokens actually
    prefilled? Scored against `prompt_n`, which is the server's own count of
    exactly that — so this measures the TIMING instrument's resolution, not the
    counter's honesty."""
    sel = [r for r in rows if r["prompt_ms"] and r["reported_prompt_n"]]
    errs = [r["prompt_ms"] / 1000.0 * rate - r["reported_prompt_n"] for r in sel]
    rel = [e / r["prompt_tokens_true"] for e, r in zip(errs, sel)]
    return (f"n={len(sel)}  median abs error {fmt(med([abs(e) for e in errs]), 0)} tokens "
            f"({fmt(100 * med([abs(x) for x in rel]), 1)}% of prompt); "
            f"p95 abs error {fmt(sorted(abs(e) for e in errs)[int(0.95 * len(errs))], 0)} "
            f"tokens; range [{fmt(min(errs), 0)}, {fmt(max(errs), 0)}]")


#: Tokens a byte-identical re-send still re-evaluates (the generation prompt).
#: Measured as a constant 4 at both micro-batch sizes.
TAIL_RETOKENIZE = 4


def rollback_gap(ub: int) -> int:
    """How far behind its end a slot can be rewound: `-ub` + 4 tokens.

    MEASURED, not assumed, and at two micro-batch sizes: at `-ub 512` the
    resident-prompt-length minus the reported reuse was exactly **516** on every
    one of 63 same-slot divergences, across four prompt lengths spanning 3.5x
    and nine divergence points; at `-ub 128` it was exactly **132** at all four
    lengths. The `+4` is the same generation-prompt markup a byte-identical
    re-send re-evaluates. R11 applies: this is a property of a build and a flag,
    so it is re-measured on either changing, never carried forward.
    """
    return ub + TAIL_RETOKENIZE


def ub_of(row: dict[str, Any]) -> int:
    extra = (row.get("extra") or "").split()
    if "-ub" in extra:
        return int(extra[extra.index("-ub") + 1])
    return int(row.get("ub") or 512)


def predict_reuse(*, resident_tokens: int | None, lcp_with_resident: int,
                  ub: int = 512) -> int:
    """What SHOULD be reused, computed entirely scaffold-side.

    `resident_tokens` is the token length of the prompt that last occupied this
    slot (None for a virgin slot); `lcp_with_resident` is the longest common
    token prefix between the incoming prompt and that one. Both are things the
    scaffold already holds or can get from one `/tokenize` call — no server
    counter enters.

    Two regimes, and the whole instrument is the boundary between them:

      * CONTINUATION — the incoming prompt picks up where the slot left off
        (`lcp >= N - 4`). Reuse is the common prefix itself, except that a
        byte-identical re-send must still re-evaluate the 4-token generation
        prompt at the very end, because a model cannot be asked to predict from
        a state that has nothing left to condition on.
      * DIVERGENCE — the prompts part company strictly inside the resident
        prompt. There is exactly ONE rollback point per slot, at `N - ub - 4`.
        Reuse is that point if the divergence is at or after it, and ZERO
        otherwise: the true common prefix is NOT available, however long it is.
    """
    if resident_tokens is None:
        return 0
    n = resident_tokens
    if lcp_with_resident >= n:
        return n - TAIL_RETOKENIZE
    if lcp_with_resident >= n - TAIL_RETOKENIZE:
        return lcp_with_resident
    if lcp_with_resident >= n - rollback_gap(ub):
        return n - rollback_gap(ub)
    return 0


def score_law(rows: list[dict[str, Any]]) -> str:
    """`cache_n` against the predicted-reuse model, per condition."""
    out = ["| condition | calls | `cache_n` == predicted | max |error| | "
           "disagreements |", "|---|---|---|---|---|"]
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond]
        resident: dict[str, int] = {}
        ok = 0
        errs: list[tuple[int, dict]] = []
        for r in sorted(sel, key=lambda x: x["ordinal"]):
            pred = predict_reuse(
                resident_tokens=resident.get(r["slot_key"]),
                lcp_with_resident=r["truth_lcp_prev_same_slot"], ub=ub_of(r))
            if pred == r["reported_cache_n"]:
                ok += 1
            else:
                errs.append((r["reported_cache_n"] - pred, r))
            resident[r["slot_key"]] = r["prompt_tokens_true"]
        detail = "; ".join(
            f"`{r['case']}`/{r['role']} {e:+d}" for e, r in errs[:4]) or "none"
        out.append(f"| `{cond}` | {len(sel)} | **{ok}/{len(sel)}** | "
                   f"{max((abs(e) for e, _ in errs), default=0)} | {detail} |")
    return "\n".join(out)


def audit(rows: list[dict[str, Any]]) -> str:
    out = ["| condition | calls | slot mismatches | foreign-identifier hits | "
           "`cache_n+prompt_n != tokenized` | wrong/absent answer |",
           "|---|---|---|---|---|---|"]
    for cond in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == cond]
        out.append(
            f"| `{cond}` | {len(sel)} | "
            f"{sum(1 for r in sel if not r['slot_ok'])} | "
            f"{sum(1 for r in sel if r['leak_detected'])} | "
            f"{sum(1 for r in sel if not r['bookkeeping_ok'])} | "
            f"{sum(1 for r in sel if not r['answer_correct'])} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(RESULTS))
    ap.add_argument("--conditions", default="default,cram0,nocacheidle")
    args = ap.parse_args()
    conds = tuple(c for c in args.conditions.split(",") if c)
    rows = load(Path(args.path), conds)
    if not rows:
        raise SystemExit("no rows")

    cal = calibration(rows)
    print("## Cold-prefill calibration\n")
    print("| condition | n | median t/s | min | max | by chunk-token target |")
    print("|---|---|---|---|---|---|")
    for cond, c in cal.items():
        by = ", ".join(f"{k}: {fmt(v[0])}" for k, v in c["by_len"].items())
        print(f"| `{cond}` | {c['n']} | **{fmt(c['rate_median'])}** | "
              f"{fmt(c['rate_min'])} | {fmt(c['rate_max'])} | {by} |")
    rate = med([c["rate_median"] for c in cal.values()])
    print(f"\npooled median cold rate: {fmt(rate)} tok/s")
    print("\n**Timing instrument resolution** (`prompt_ms x cold rate` vs the "
          "server's own `prompt_n`):")
    for cond in cal:
        print(f"- `{cond}`: " + instrument_check(
            [r for r in rows if r["condition"] == cond], cal[cond]["rate_median"]))

    for cond in conds:
        if not any(r["condition"] == cond for r in rows):
            continue
        print(f"\n## Reported vs true — `{cond}`\n")
        print(table_cases(rows, cond))

    print("\n## The decisive comparison: the same cases under each host-cache flag\n")
    print("Each cell is `cache_n` / true-LCP-prev-on-slot / true-LCP-any-slot.\n")
    print(table_conditions(rows))

    print("\n## Where a PARTIAL match lands\n")
    print(table_diverge(rows))

    print("\n## The reuse law, scored on every call\n")
    print(score_law(rows))

    print("\n## Audit\n")
    print(audit(rows))


if __name__ == "__main__":
    main()
