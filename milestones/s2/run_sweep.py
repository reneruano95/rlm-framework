"""S2 chunk-size QUALITY sweep — runner and scorer (spec §7 #2, v0.2.4).

THE SCORING TAXONOMY IS THE DELIVERABLE. Everything else in this file is
plumbing around it. §7 #2 exists because a 32K leaf returned the true UUID with
its last character altered — a fluent, correctly-formatted, character-plausible
lie — while the same leaf at ~1K returned `NONE`. Those two failures have
OPPOSITE remedies (one wants a smaller chunk or a verification pass, the other
wants a bigger one or a better question), so a scorer that reports both as
"wrong" destroys the only signal the sweep exists to produce. Five labels,
exactly one per answer:

  CORRECT        the planted fact, under the normalization stated below — and,
                 for an ABSENT question, a refusal.
  MISS           a refusal when the fact IS present. The leaf read and did not
                 find. Cheap to detect at the root, safe to retry.
  CONFABULATION  a non-refusal answer that is wrong. For UUID-shaped answers the
                 Levenshtein distance to the true value is recorded, because
                 distance 1 is precisely the failure that motivated this sweep
                 and a `contains` checker scores it identically to distance 36.
  FALSE-POSITIVE any non-refusal answer to an ABSENT question. This is the rate
                 that makes a confabulation dangerous rather than merely wrong:
                 it says the leaf will invent a key on demand.
  MALFORMED      empty or degenerate output (nothing to score).

**Re-scored from `raw_output` at report time.** Every record stores the model's
output verbatim and the label computed at run time, but `--phase report`
re-classifies from the raw text rather than trusting the stored label — so a
classifier fix costs a re-read of a JSONL file, never a re-run of the GPU.

WHY THE SWEEP IS AFFORDABLE: warm re-query. §7 #3 (d) measured a second
question against a resident 30K chunk at 1.71 s wall / `tokens_cached` 29641,
against 35.45 s cold. So the unit of work here is a CHUNK, not a call: pin one
slot, pay one cold prefill, then ask all three question types x all trials
against the chunk that is already sitting in it. Six sizes x 3 types x 3 trials
= 54 calls but only 6 cold prefills. `tokens_cached`, `slot_id` and the cold
flag are on every record precisely so the analysis can CONFIRM the warm path
was used rather than assume it — if `tokens_cached` collapses, the timings are
measuring eviction, not quality.

Trials are three SEEDS at the config's real leaf temperature (0.3), not three
repeats of one seed: the cell reports a rate under production sampling. Seed 1
is the config's own, so trial 1 is exactly what production would draw.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `uv run milestones/s2/run_sweep.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.rootclient import strip_reasoning  # noqa: E402
from s2.make_sweep_fixtures import (  # noqa: E402
    FIXTURES_DIR, MANIFEST_NAME, QUESTION_TYPES, UUID_RE, load_manifest,
)

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "sweep.jsonl"
RESULTS_MD = S2_DIR / "RESULTS.md"

TRIAL_SEEDS: tuple[int, ...] = (1, 2, 3)

# --------------------------------------------------------------------------- #
# THE CLASSIFIER
# --------------------------------------------------------------------------- #

CORRECT = "CORRECT"
MISS = "MISS"
CONFABULATION = "CONFABULATION"
FALSE_POSITIVE = "FALSE-POSITIVE"
MALFORMED = "MALFORMED"
LABELS = (CORRECT, MISS, CONFABULATION, FALSE_POSITIVE, MALFORMED)

#: An answer longer than this is not an answer. The leaf prefix says "give the
#: bare value", "no preamble", "be brief"; `max_predict.leaf` is 512 tokens
#: (~2 KB). A 600+ character reply to "reply with the key itself and nothing
#: else" is the excerpt being restated, and crediting it as CORRECT because the
#: key happens to appear somewhere inside would score regurgitation as
#: extraction.
DEGENERATE_CHARS = 600
#: A token repeated this many times in a row is a decode loop.
DEGENERATE_RUN = 8
#: Words allowed AFTER a refusal phrase before the answer stops being a
#: refusal. "none" and "none found in this excerpt" are refusals; "none, but
#: the closest match is Kelvane Dorwold" is a confabulation wearing a hedge.
REFUSAL_RESIDUE_WORDS = 6

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```$", re.DOTALL)
_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:archive\s+)?(?:key|answer|value|custodian|name|result)"
    r"(?:\s+is)?\s*[:\-]\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

#: Whole-answer refusals, matched after normalization.
REFUSAL_EXACT = frozenset({
    "none", "no", "n/a", "na", "nil", "null", "nothing", "unknown",
    "not present", "not stated", "not found", "not given", "not applicable",
    "no answer", "no key", "no such key", "no such entry", "not in the excerpt",
    "not in this excerpt", "not in the document", "not specified",
})
#: Leading refusal phrases. A match only counts as a refusal when what follows
#: names nothing (see `is_refusal`).
REFUSAL_LEAD_RE = re.compile(
    r"^(?:none|no|n/a|nil|null|unknown|not\s+present|not\s+stated|not\s+found|"
    r"not\s+given|not\s+specified|not\s+mentioned|not\s+applicable|"
    r"no\s+such\b[\w\s]{0,20}|no\s+answer|"
    r"not\s+in\s+(?:the\s+|this\s+)?(?:excerpt|document|text|passage|register)|"
    r"(?:the|this)\s+excerpt\s+does\s+not\b[\w\s]{0,40}|"
    r"there\s+is\s+no\b[\w\s]{0,40}|"
    r"i\s+(?:can|could)\s*n[o']?t\s+find\b[\w\s]{0,40}|"
    r"cannot\s+be\s+determined)\b")


def normalize(raw: str) -> str:
    """The stated normalization. Every match in this module is against this.

    Strips (in order): a `<think>...</think>` block via the SAME
    `rlm.rootclient.strip_reasoning` the production root uses, a single
    wrapping code fence, a leading `Key:` / `Answer:` style label, surrounding
    quotes/backticks/asterisks; then collapses whitespace, drops trailing
    punctuation, and casefolds. Deliberately permissive about FORMAT and
    strict about CONTENT: a leaf that formats loosely is a prompt problem, a
    leaf that changes a character is the thing being measured.
    """
    text = strip_reasoning(raw or "").strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    text = _LABEL_RE.sub("", text, count=1)
    text = text.strip(" \t\r\n`\"'*")
    text = _WS_RE.sub(" ", text).strip()
    text = text.rstrip(" .,;:!")
    return text.casefold()


def _degenerate_reason(norm: str) -> str | None:
    if not norm:
        return "empty output"
    if not any(ch.isalnum() for ch in norm):
        return "no alphanumeric content"
    if len(norm) > DEGENERATE_CHARS:
        return (f"{len(norm)} chars (> {DEGENERATE_CHARS}): the excerpt was "
                f"restated, not answered")
    words = norm.split()
    run = 1
    for prev, cur in zip(words, words[1:]):
        run = run + 1 if cur == prev else 1
        if run >= DEGENERATE_RUN:
            return f"decode loop: {cur!r} repeated {run}x"
    return None


def uuid_candidates(norm: str) -> list[str]:
    """Every UUID-shaped token in a normalized answer, in order."""
    return UUID_RE.findall(norm)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein. 36-character strings — the O(nm) table is free, and an
    off-by-one-character UUID is exactly what this number has to expose."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_refusal(norm: str, *, expected_kind: str | None = None) -> bool:
    """Did the leaf decline to answer?

    Two rules, in order. (1) The whole normalized answer is a known refusal
    string. (2) It STARTS with a refusal phrase, and what remains after that
    phrase names nothing: no token of the expected shape, and at most
    `REFUSAL_RESIDUE_WORDS` words. Rule 2's residue test is what keeps
    "NONE. The closest is 4f2a-..." out of the MISS column — an answer that
    hedges and then supplies a value is a confabulation, and calling it a
    refusal would understate the danger this sweep is measuring.
    """
    if norm in REFUSAL_EXACT:
        return True
    match = REFUSAL_LEAD_RE.match(norm)
    if not match:
        return False
    residue = norm[match.end():].strip(" .,;:-–—")
    if expected_kind == "uuid" and uuid_candidates(norm):
        return False
    return len(residue.split()) <= REFUSAL_RESIDUE_WORDS


def classify(raw: str, *, question_type: str, expected: str | None,
             expected_kind: str | None = None) -> dict[str, Any]:
    """Exactly one label per answer. The decision rule, in order:

      1. MALFORMED       — normalized output is empty, has no alphanumeric
                           character, exceeds 600 chars, or repeats one token
                           >= 8x.
      2. CORRECT         — (fact-present questions only) the normalized
                           expected string occurs in the normalized answer AND,
                           when the expected value is UUID-shaped, no second
                           distinct UUID-shaped token occurs. Tested BEFORE the
                           refusal rule so an answer cannot be both.
      3. refusal?        — `is_refusal`. ABSENT -> CORRECT; fact-present -> MISS.
      4. otherwise       — ABSENT -> FALSE-POSITIVE; fact-present ->
                           CONFABULATION, with `uuid_edit_distance` set to the
                           Levenshtein distance from the true value to the
                           nearest UUID-shaped token in the answer (to the whole
                           normalized answer when it contains none).
    """
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {question_type!r}")
    absent = question_type == "absent"
    if absent and expected is not None:
        raise ValueError("an ABSENT question has no expected answer; the "
                         "correct answer is a refusal")
    if not absent and not expected:
        raise ValueError(f"{question_type!r} needs an expected answer")

    norm = normalize(raw)
    out: dict[str, Any] = {"label": None, "reason": "", "normalized": norm,
                           "normalized_len": len(norm),
                           "uuid_edit_distance": None,
                           "uuid_candidates": uuid_candidates(norm)}

    degenerate = _degenerate_reason(norm)
    if degenerate:
        out.update(label=MALFORMED, reason=degenerate)
        return out

    if not absent:
        want = normalize(expected or "")
        others = [u for u in out["uuid_candidates"] if u != want]
        if want and want in norm and not (expected_kind == "uuid" and others):
            out.update(label=CORRECT, reason="expected value present")
            return out

    if is_refusal(norm, expected_kind=expected_kind):
        if absent:
            out.update(label=CORRECT, reason="refused an absent fact")
        else:
            out.update(label=MISS, reason="refused a fact that is present")
        return out

    if absent:
        out.update(label=FALSE_POSITIVE,
                   reason="answered a question whose fact is not in the chunk")
        return out

    out.update(label=CONFABULATION, reason="non-refusal answer that is wrong")
    if expected_kind == "uuid":
        want = normalize(expected or "")
        cands = out["uuid_candidates"] or [norm]
        out["uuid_edit_distance"] = min(edit_distance(want, c) for c in cands)
    return out


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #


def append_run(record: dict, path: Path = RUNS_PATH) -> dict:
    """One JSONL line per CALL, appended immediately.

    Append-per-call, not write-at-the-end: an interrupted sweep must never cost
    the calls that already ran, and nothing may be lost to summarization —
    `raw_output` is the model's text verbatim, never truncated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_runs(path: Path = RUNS_PATH) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plan_calls(cells: list[dict], *, trials: int = 3,
               seeds: tuple[int, ...] = TRIAL_SEEDS) -> list[dict]:
    """The call schedule: for each cell, every (trial, question type) pair,
    ordered so the chunk is prefilled ONCE and then re-queried.

    Question order rotates by trial so no single question type permanently owns
    the one genuinely cold call — the cold/warm distinction stays a property of
    position in the schedule, recorded per call, and never a property of the
    question type being scored.
    """
    plan: list[dict] = []
    for cell in cells:
        idx = 0
        for trial in range(trials):
            rotation = [QUESTION_TYPES[(i + trial) % len(QUESTION_TYPES)]
                        for i in range(len(QUESTION_TYPES))]
            for qtype in rotation:
                plan.append({"cell_id": cell["cell_id"], "trial": trial + 1,
                             "seed": seeds[trial % len(seeds)],
                             "question_type": qtype, "call_idx": idx,
                             "cold": idx == 0})
                idx += 1
    return plan


def select_cells(manifest: dict, *, sizes: list[int] | None,
                 positions: list[float] | None) -> list[dict]:
    cells = [c for c in manifest["cells"].values()
             if (sizes is None or c["size_tokens"] in sizes)
             and (positions is None or any(abs(c["position"] - p) < 1e-9
                                           for p in positions))]
    return sorted(cells, key=lambda c: (c["size_tokens"], c["position"]))


async def sweep(caller, manifest: dict, cells: list[dict], *, slot: int,
                trials: int, seeds: tuple[int, ...], phase: str,
                out_path: Path = RUNS_PATH, cold_per_trial: bool = False,
                buster: str | None = None, echo=print) -> list[dict]:
    """Run the plan against a prepared `PinnedLeafCaller`. One record per call.

    `caller` is injected rather than built here so the whole runner is testable
    against the loopback mock server without a live leaf.
    """
    records: list[dict] = []
    for cell in cells:
        text = Path(cell["chunk_path"]).read_text(encoding="utf-8")
        if not caller.admits(cell["measured_tokens"]):
            echo(f"[skip] {cell['cell_id']}: {cell['measured_tokens']} tokens "
                 f"does not fit one slot")
            continue
        for call in plan_calls([cell], trials=trials, seeds=seeds):
            spec = cell["questions"][call["question_type"]]
            fresh = call["call_idx"] % len(QUESTION_TYPES) == 0
            if cold_per_trial and fresh and call["call_idx"]:
                # Evict this chunk from the pinned slot so the next trial pays a
                # real cold prefill. Off by default: it costs one full prefill
                # per trial (3x the sweep's GPU time) to remove a confound the
                # `cold` flag already records.
                try:
                    await caller.ask(question="Reply NONE.",
                                     chunk=buster or "cache buster",
                                     seed=0, id_slot=slot)
                except Exception as exc:  # noqa: BLE001
                    echo(f"[warn] cache-buster failed: {exc!r}")
            record = {
                "phase": phase,
                "cell_id": cell["cell_id"],
                "size_target": cell["size_tokens"],
                "size_measured": cell["measured_tokens"],
                "position": cell["position"],
                "needle_token_depth": (cell["needles"].get(call["question_type"], {})
                                       .get("token_depth")),
                "question_type": call["question_type"],
                "question": spec["question"],
                "expected": spec["expected"],
                "expected_kind": spec["expected_kind"],
                "trial": call["trial"], "call_idx": call["call_idx"],
                "cold": call["cold"] or (cold_per_trial and fresh),
                "chunk_sha256": cell["sha256"],
                "manifest_counter": manifest.get("token_counter"),
                "generator_sha256": manifest.get("generator_sha256"),
                "temperature": caller.temperature, "top_p": caller.top_p,
                "max_predict": caller.max_predict,
            }
            try:
                answer = await caller.ask(question=spec["question"], chunk=text,
                                          seed=call["seed"], id_slot=slot)
            except Exception as exc:  # noqa: BLE001 -- a failed call IS a result
                record.update(status="error", error=repr(exc), label=None,
                              seed=call["seed"], id_slot=slot)
                append_run(record, out_path)
                records.append(record)
                echo(f"[error] {record['cell_id']} {call['question_type']} "
                     f"trial {call['trial']}: {exc!r}")
                continue
            record.update(status="ok", error=None, **answer.as_record())
            record.update(classify(answer.raw, question_type=call["question_type"],
                                   expected=spec["expected"],
                                   expected_kind=spec["expected_kind"]))
            record["cache_hit_fraction"] = (
                round(record["tokens_cached"] / record["tokens_in"], 4)
                if record.get("tokens_in") else None)
            append_run(record, out_path)
            records.append(record)
            echo(f"[{record['label']}] {record['cell_id']} "
                 f"{call['question_type']} trial {call['trial']} "
                 f"(cached {record['tokens_cached']}/{record['tokens_in']}, "
                 f"slot {record['slot_id']}, {record['wall_s']}s) "
                 f"{record['normalized'][:70]!r}")
    return records


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def rescore(records: list[dict]) -> list[dict]:
    """Re-classify every OK record from its verbatim `raw_output`.

    The report never trusts the label written at run time: a classifier fix
    must be applyable to a finished sweep without touching the GPU.
    """
    out: list[dict] = []
    for rec in records:
        rec = dict(rec)
        if rec.get("status") == "ok":
            rec.update(classify(rec.get("raw_output", ""),
                                question_type=rec["question_type"],
                                expected=rec.get("expected"),
                                expected_kind=rec.get("expected_kind")))
        out.append(rec)
    return out


def summarize(records: list[dict]) -> dict:
    """Counts per (size, position, question type), plus the warm-path evidence.

    Rates are over OK calls only; errors are counted and reported separately
    rather than folded into any label, because an unreachable server is not a
    model failure.
    """
    cells: dict[tuple, dict] = {}
    for rec in records:
        key = (rec["size_target"], rec["position"], rec["question_type"])
        cell = cells.setdefault(key, {
            "size": rec["size_target"], "position": rec["position"],
            "question_type": rec["question_type"], "n": 0, "errors": 0,
            "labels": {label: 0 for label in LABELS},
            "uuid_edit_distances": [], "wall_s": [],
        })
        if rec.get("status") != "ok":
            cell["errors"] += 1
            continue
        cell["n"] += 1
        cell["labels"][rec["label"]] = cell["labels"].get(rec["label"], 0) + 1
        if rec.get("uuid_edit_distance") is not None:
            cell["uuid_edit_distances"].append(rec["uuid_edit_distance"])
        if rec.get("wall_s") is not None:
            cell["wall_s"].append(rec["wall_s"])

    ok = [r for r in records if r.get("status") == "ok"]
    cold = [r for r in ok if r.get("cold")]
    warm = [r for r in ok if not r.get("cold")]

    def med(values: list[float]) -> float | None:
        return round(statistics.median(values), 3) if values else None

    return {
        "cells": dict(sorted(cells.items())),
        "warm_path": {
            "cold_calls": len(cold), "warm_calls": len(warm),
            "median_cold_wall_s": med([r["wall_s"] for r in cold]),
            "median_warm_wall_s": med([r["wall_s"] for r in warm]),
            "median_cold_tokens_cached": med([r["tokens_cached"] for r in cold]),
            "median_warm_tokens_cached": med([r["tokens_cached"] for r in warm]),
            "median_warm_cache_fraction": med(
                [r["cache_hit_fraction"] for r in warm
                 if r.get("cache_hit_fraction") is not None]),
            "slots_used": sorted({r["slot_id"] for r in ok if r.get("slot_id") is not None}),
        },
    }


def render_report(records: list[dict]) -> str:
    """RESULTS.md's generated half, from the record rather than from memory."""
    records = rescore(records)
    summary = summarize(records)
    lines = [
        "# S2 — chunk-size QUALITY sweep: results",
        "",
        f"*Generated by `milestones/s2/run_sweep.py --phase report` from "
        f"`milestones/s2/results/sweep.jsonl` ({len(records)} calls) on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}. Labels are "
        f"RE-DERIVED from each record's verbatim `raw_output`, not read back "
        f"from the label written at run time.*",
        "",
        "Taxonomy: **CORRECT** / **MISS** (refused a fact that is present) / "
        "**CONFABULATION** (non-refusal answer that is wrong) / "
        "**FALSE-POSITIVE** (any non-refusal answer to an ABSENT question) / "
        "**MALFORMED** (empty or degenerate). MISS and CONFABULATION are never "
        "collapsed into \"wrong\": they have opposite remedies.",
        "",
        "| size | depth | question | n | CORRECT | MISS | CONFAB | FALSE-POS | "
        "MALFORMED | err | median wall s | UUID edit dist |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (size, position, qtype), cell in summary["cells"].items():
        labels = cell["labels"]
        dists = cell["uuid_edit_distances"]
        wall = cell["wall_s"]
        lines.append(
            f"| {size} | {position:.0%} | {qtype} | {cell['n']} | "
            f"{labels[CORRECT]} | {labels[MISS]} | {labels[CONFABULATION]} | "
            f"{labels[FALSE_POSITIVE]} | {labels[MALFORMED]} | {cell['errors']} | "
            f"{round(statistics.median(wall), 2) if wall else '-'} | "
            f"{sorted(dists) if dists else '-'} |")

    warm = summary["warm_path"]
    lines += [
        "", "## Warm re-query — was the resident chunk actually re-used?", "",
        f"- cold calls: {warm['cold_calls']}, median wall "
        f"{warm['median_cold_wall_s']} s, median `tokens_cached` "
        f"{warm['median_cold_tokens_cached']}",
        f"- warm calls: {warm['warm_calls']}, median wall "
        f"{warm['median_warm_wall_s']} s, median `tokens_cached` "
        f"{warm['median_warm_tokens_cached']} "
        f"({warm['median_warm_cache_fraction']} of prompt)",
        f"- slots observed: {warm['slots_used']} (one pinned slot means one entry)",
        "",
        "A warm median `tokens_cached` that does not approach the chunk length "
        "means the slot was evicted and the wall-clock column is measuring "
        "re-prefill, not quality.",
        "",
        "## Every CONFABULATION and FALSE-POSITIVE, verbatim", "",
        "| size | depth | question | trial | UUID edit dist | raw output |",
        "|---|---|---|---|---|---|",
    ]
    for rec in records:
        if rec.get("label") not in (CONFABULATION, FALSE_POSITIVE):
            continue
        raw = " ".join((rec.get("raw_output") or "").split())[:200]
        lines.append(
            f"| {rec['size_target']} | {rec['position']:.0%} | "
            f"{rec['question_type']} | {rec['trial']} | "
            f"{rec.get('uuid_edit_distance', '-')} | `{raw}` |")
    lines.append("")
    return "\n".join(lines)


NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"


def regenerate(records: list[dict], results_md: Path = RESULTS_MD) -> str:
    generated = render_report(records)
    narrative = ""
    if results_md.exists():
        old = results_md.read_text(encoding="utf-8")
        if NARRATIVE_MARKER in old:
            narrative = old.split(NARRATIVE_MARKER, 1)[1]
    return f"{generated}\n{NARRATIVE_MARKER}\n{narrative.lstrip(chr(10))}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def _amain(args) -> int:
    if args.phase == "report":
        RESULTS_MD.write_text(regenerate(read_runs(args.out)), encoding="utf-8",
                              newline="\n")
        print(f"wrote {RESULTS_MD}")
        return 0

    from rlm.config import load_config
    from s2.leafcall import PinnedLeafCaller

    manifest = load_manifest(args.fixtures)
    if manifest.get("token_counter") != "leaf:/tokenize":
        print(f"REFUSING: {args.fixtures / MANIFEST_NAME} was built with "
              f"{manifest.get('token_counter')!r}. The sweep's x-axis is leaf "
              f"tokens; rebuild the fixtures against the live leaf tokenizer.",
              file=sys.stderr)
        return 2
    cells = select_cells(manifest, sizes=args.sizes, positions=args.positions)
    if not cells:
        print("no cells selected", file=sys.stderr)
        return 2

    cfg = load_config(Path(args.config))
    caller = PinnedLeafCaller.from_config(cfg)
    try:
        prefix_tokens = await caller.prepare()
        print(f"leaf prefix renders to {prefix_tokens} tokens; pinning slot "
              f"{args.slot}; {len(cells)} cell(s) x {args.trials} trial(s) x "
              f"{len(QUESTION_TYPES)} question types = "
              f"{len(cells) * args.trials * len(QUESTION_TYPES)} calls, "
              f"{len(cells)} cold prefill(s)")
        await sweep(caller, manifest, cells, slot=args.slot, trials=args.trials,
                    seeds=tuple(args.seeds), phase=args.phase, out_path=args.out,
                    cold_per_trial=args.cold_per_trial)
    finally:
        await caller.aclose()

    if not args.no_report:
        RESULTS_MD.write_text(regenerate(read_runs(args.out)), encoding="utf-8",
                              newline="\n")
        print(f"wrote {RESULTS_MD}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the S2 chunk-size QUALITY sweep (spec §7 #2)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="main",
                        choices=("main", "position", "report"),
                        help="'main' = the six sizes at ~50%% depth; 'position' "
                             "= the depth sub-study, whose --sizes are chosen "
                             "from the main sweep's breakdown point; 'report' = "
                             "re-score the JSONL and regenerate RESULTS.md")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    parser.add_argument("--out", type=Path, default=RUNS_PATH)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--positions", nargs="+", type=float, default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRIAL_SEEDS))
    parser.add_argument("--slot", type=int, default=0,
                        help="the pinned leaf slot (§7 #3 (d)); one slot for "
                             "the whole sweep keeps every re-query warm")
    parser.add_argument("--cold-per-trial", action="store_true",
                        help="evict between trials so every trial pays a cold "
                             "prefill. ~3x the GPU time; the `cold` flag "
                             "already records the confound without it.")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    if args.positions is None:
        # The main sweep holds depth at 50%; the sub-study is the OTHER two
        # depths (50% is already measured, at every size, by the main sweep).
        args.positions = {"main": [0.5], "position": [0.1, 0.9]}.get(args.phase)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
