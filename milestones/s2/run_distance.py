"""S2 — INSTRUCTION DECAY: is the false-positive rate about DISTANCE or DENSITY?

THE CLAIM UNDER TEST (spec §4 "INSTRUCTION DECAY", §7 #2, §10 R5/R13). The same
`leaf-prefix.v1.md`, unchanged, produced 30/30 false positives at a 1,024-token
window and 0/21 at 640 (`milestones/s2/REFUSAL-AB.md` §2, Fisher p = 8.7e-15). If the cause
is DISTANCE — instructions decaying with distance from the point of generation
exactly as facts do — then §7 #2's retrieval cliff and the false-positive rate
are one phenomenon measured twice, and §4's `[system prefix][chunk][question]`
layout puts the rules at the worst possible place. If the cause is DENSITY — a
640-token chunk simply holds fewer entities to misattribute from — the
hypothesis is dead and the leaf is unreliable at this size for an ordinary
reason.

**The two moved together in the A/B, so that result decides nothing as it
stands.** This file separates them, and if it does not separate them it is
worthless.

THE DESIGN, three factors, one un-edited prompt.

  FACTOR 1 — instruction LAYOUT, composed scaffold-side by
  `s2.leafcall.PinnedLeafCaller.compose` (never by the model, §5 C4):
      A  `[system prefix][chunk][question]`             — today's shipped layout
      B  `[system prefix][chunk][prefix again][question]` — cache contract intact
      C  `[chunk][system prefix][question]`             — no leading prefix
  All three carry `prompts/leaf-prefix.v1.md` VERBATIM. The variable is
  POSITION, not wording: authoring new prompt text against measurement fixtures
  is the overfitting §8's freeze rule forbids, and it is exactly what the A/B
  already showed does not work (v2 moved 30/30 to 29/30, p = 1.0).

  FACTOR 2 — chunk size {640, 1024, 2048} target leaf tokens.

  FACTOR 3 — distractor DENSITY, `matched` (3 entity bindings at every size,
  the 640-token cell's count, larger chunks padded with entity-free neutral
  filler) vs `natural` (the measured per-token rate of the fixtures the A/B ran:
  3 / 6 / 11). See `milestones/s2/make_distance_fixtures.py` for what a "binding" is and
  why it is that.

  QUESTIONS — all three types in every cell, always reported together. ABSENT
  measures the false-positive rate; LITERAL and PARAPHRASE measure RECALL, and
  they are how over-refusal is caught. **An arm at 0% false positives and 0%
  recall is a failure, not a success**, and the report may not print the first
  number without the second two.

REPLICATION FIRST (`--phase replicate`). Before the grid, arm A re-runs the
exact 640-vs-1024 comparison on the A/B's own fixture directories. If 30/30 →
0/21 does not reproduce, everything downstream is moot and the report says so.

WHAT THIS RUNNER DOES NOT DO. No retries (a failed call is a recorded fact — see
`milestones/s2/leafcall.py`), no prompt edits, no arm added after the first call, and no
scoring instrument of its own: labels come from `s2.run_sweep.classify`, the
same function `milestones/s2/RESULTS.md` and `milestones/s2/REFUSAL-AB.md` were scored with, so the
numbers here sit beside those without a translation step.

R13 (§10, §5 C4). One never-reused slot per (arm, cell), drawn from the pinned
128-slot pool, with the server's returned `slot_id` ASSERTED against the one
requested; intra-cell re-queries share that cell's slot on purpose (same-document
reuse, measured clean). R13's foreign-string detector runs over every answer
against the whole fixture corpus and its hit count is reported per arm. The one
deliberate exception is `--phase cache`, which reuses a slot across two
documents BECAUSE the quantity it measures is cross-document cache reuse; its
records are stamped `phase="cache"` and are excluded from every quality table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `uv run milestones/s2/run_distance.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.leakcheck import ChunkIndex  # noqa: E402
from s2.leafcall import LAYOUTS  # noqa: E402
from s2.make_sweep_fixtures import MANIFEST_NAME, QUESTION_TYPES  # noqa: E402
from s2.run_refusal_ab import supplied_identifier  # noqa: E402
from s2.run_sweep import (  # noqa: E402
    CONFABULATION,
    CORRECT,
    FALSE_POSITIVE,
    LABELS,
    MALFORMED,
    MISS,
    append_run,
    classify,
    plan_calls,
    read_runs,
)

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "distance.jsonl"
REPORT_MD = S2_DIR / "DISTANCE.md"

TRIAL_SEEDS: tuple[int, ...] = (1, 2, 3)

#: FACTOR 1. `(arm_id, layout)`; the prefix file is the same in all three.
ARMS: tuple[tuple[str, str], ...] = (
    ("A-shipped", "A"),
    ("B-repeated", "B"),
    ("C-after", "C"),
)
LEAF_PREFIX = "prompts/leaf-prefix.v1.md"

#: The A/B's own fixture directories, for the replication arm. Not rebuilt and
#: not regenerated: a replication that used new documents would be a new
#: experiment wearing the word "replication".
REPLICATION_1024 = ("fixtures", "fixtures-refusal-s2", "fixtures-refusal-s3")
REPLICATION_1024_CELLS = ("s2-1024-p10", "s2-1024-p30", "s2-1024-p50",
                          "s2-1024-p70")
REPLICATION_640 = tuple(f"fixtures-refusal-640-s{s}" for s in (2, 3, 4, 5, 7, 9, 10))

NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"


# --------------------------------------------------------------------------- #
# statistics: exact, and dependency-free
# --------------------------------------------------------------------------- #


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Written out rather than imported: the repo has no scipy, the tables here
    are tiny, and a headline p-value that cannot be recomputed from the record
    by anyone reading it is not evidence. Sums the hypergeometric probability
    of every table with the same margins whose probability is <= the observed
    one (the conventional two-sided definition).
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return (math.comb(row1, x) * math.comb(n - row1, col1 - x)
                / math.comb(n, col1))

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    observed = prob(a)
    total = sum(prob(x) for x in range(lo, hi + 1)
                if prob(x) <= observed * (1 + 1e-9))
    return min(1.0, total)


def wilson_upper95(k: int, n: int) -> float | None:
    """One-sided 95% upper bound on a rate — used only where a 0/n is reported,
    so "zero observed" is never printed as "zero"."""
    if not n:
        return None
    z = 1.6449
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(min(1.0, (centre + half) / denom), 4)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def load_cells(dirs: list[Path], *, only: tuple[str, ...] | None = None
               ) -> tuple[list[dict], dict[str, str]]:
    """Cells from every directory, tagged by manifest seed.

    The uid is `{cell_id}#s{seed}`, as in the A/B: two cells built at the same
    (size, density) from different seeds are INDEPENDENT generated facts, and
    that is where this experiment's statistical power comes from.
    """
    cells: list[dict] = []
    corpus: dict[str, str] = {}
    for d in dirs:
        manifest = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("token_counter") != "leaf:/tokenize":
            raise SystemExit(
                f"REFUSING: {d / MANIFEST_NAME} was built with "
                f"{manifest.get('token_counter')!r}. Every size claim here is in "
                f"leaf tokens; rebuild against the live tokenizer.")
        seed = manifest.get("seed")
        for cell in manifest["cells"].values():
            if only is not None and cell["cell_id"] not in only:
                continue
            cell = dict(cell)
            cell["fixture_seed"] = seed
            cell["uid"] = f"{cell['cell_id']}#s{seed}"
            cell.setdefault("density", "natural-sweep")
            cell.setdefault("entity_bindings", None)
            if cell["uid"] in corpus:
                raise SystemExit(f"duplicate fixture cell {cell['uid']}")
            corpus[cell["uid"]] = Path(cell["chunk_path"]).read_text(encoding="utf-8")
            cells.append(cell)
    cells.sort(key=lambda c: (c["size_tokens"], str(c["density"]),
                              c["fixture_seed"]))
    return cells, corpus


def replication_dirs() -> tuple[list[Path], list[Path]]:
    return ([S2_DIR / d for d in REPLICATION_1024],
            [S2_DIR / d for d in REPLICATION_640])


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def score(raw: str, *, question_type: str, expected: str | None,
          expected_kind: str | None) -> dict[str, Any]:
    """The sweep's classifier, plus the A/B's identifier sub-question.

    `supplied_identifier` is imported from `milestones/s2/run_refusal_ab.py` for the reason
    stated there: the pinned `is_refusal` scores a verbose refusal ("the
    provided text does not contain ...") as FALSE-POSITIVE, and layouts that put
    the rules next to the question may well produce more verbose refusals — so
    the treatment arms would be penalised by the instrument. The pinned label
    stays primary; this records whether the reply actually handed a root a value
    it could submit.
    """
    labels = classify(raw, question_type=question_type, expected=expected,
                      expected_kind=expected_kind)
    labels["supplied_identifier"] = supplied_identifier(labels["normalized"])
    return labels


async def run_arm(caller, *, arm_id: str, phase: str, cells: list[dict],
                  corpus: dict[str, str], index: ChunkIndex, next_slot,
                  trials: int, seeds: tuple[int, ...], out_path: Path,
                  echo=print) -> list[dict]:
    """One arm over every cell it admits. A fresh, never-reused slot per cell."""
    records: list[dict] = []
    for cell in cells:
        text = corpus[cell["uid"]]
        if not caller.admits(cell["measured_tokens"]):
            echo(f"[skip] {arm_id} {cell['uid']}: {cell['measured_tokens']} "
                 f"chunk tokens + a {caller.head_tokens()}-token layout head "
                 f"does not fit one {caller.slot_capacity_tokens}-token slot")
            # Recorded, not merely printed: "this arm could not run this cell"
            # is a RESULT — layout B costs slot capacity — and a report built
            # from the JSONL alone has to be able to say so.
            records.append(append_run({
                "phase": phase, "arm": arm_id, "layout": caller.layout,
                "cell_uid": cell["uid"], "cell_id": cell["cell_id"],
                "size_target": cell["size_tokens"], "density": cell["density"],
                "entity_bindings": cell["entity_bindings"],
                "status": "inadmissible", "label": None,
                "head_tokens": caller.head_tokens(),
                "slot_capacity_tokens": caller.slot_capacity_tokens,
            }, out_path))
            continue
        slot = next_slot()
        for call in plan_calls([cell], trials=trials, seeds=seeds):
            spec = cell["questions"][call["question_type"]]
            record = {
                "phase": phase,
                "arm": arm_id,
                "layout": caller.layout,
                "prefix_sha256": caller.prefix_sha256,
                "prefix_tokens": caller.prefix_tokens,
                "prefix_body_tokens": caller.prefix_body_tokens,
                "head_tokens": caller.head_tokens(),
                "cell_uid": cell["uid"],
                "cell_id": cell["cell_id"],
                "fixture_seed": cell["fixture_seed"],
                "size_target": cell["size_tokens"],
                "size_measured": cell["measured_tokens"],
                "density": cell["density"],
                "entity_bindings": cell["entity_bindings"],
                "position": cell["position"],
                "question_type": call["question_type"],
                "question": spec["question"],
                "expected": spec["expected"],
                "expected_kind": spec["expected_kind"],
                "trial": call["trial"],
                "call_idx": call["call_idx"],
                "cold": call["cold"],
                "chunk_sha256": cell["sha256"],
                "temperature": caller.temperature,
                "top_p": caller.top_p,
                "max_predict": caller.max_predict,
                "requested_slot": slot,
            }
            try:
                answer = await caller.ask(question=spec["question"], chunk=text,
                                          seed=call["seed"], id_slot=slot)
            except Exception as exc:  # noqa: BLE001 -- a failed call IS a result
                record.update(status="error", error=repr(exc), label=None,
                              seed=call["seed"])
                append_run(record, out_path)
                records.append(record)
                echo(f"[error] {arm_id} {cell['uid']} {call['question_type']} "
                     f"trial {call['trial']}: {exc!r}")
                continue
            record.update(status="ok", error=None, **answer.as_record())
            # R13's slot assertion, the one C4 makes: an out-of-range id_slot is
            # silently reassigned with HTTP 200, so a run that trusts its own
            # request believes it holds a virgin slot while sharing a used one.
            record["slot_ok"] = (answer.slot_id == slot)
            record.update(score(answer.raw, question_type=call["question_type"],
                                expected=spec["expected"],
                                expected_kind=spec["expected_kind"]))
            verdict = index.foreign(answer.raw, sent=f"{text}\n\n{spec['question']}")
            record["leak_detected"] = verdict.detected
            record["leak_detail"] = verdict.detail
            record["cache_hit_fraction"] = (
                round(record["tokens_cached"] / record["tokens_in"], 4)
                if record.get("tokens_in") else None)
            append_run(record, out_path)
            records.append(record)
            echo(f"[{record['label']:14s}] {arm_id} {cell['uid']} "
                 f"{call['question_type']:10s} t{call['trial']} "
                 f"slot {answer.slot_id}{'' if record['slot_ok'] else ' !!MISMATCH'} "
                 f"cached {record['tokens_cached']}/{record['tokens_in']} "
                 f"{record['wall_s']}s {record['normalized'][:56]!r}")
    return records


async def run_cache_probe(caller, *, arm_id: str, cells: list[dict],
                          corpus: dict[str, str], slot: int, out_path: Path,
                          echo=print) -> list[dict]:
    """Price §4's cache contract under this layout, and ONLY that.

    §4's argument for question-last is that a re-queried chunk extends the
    cached prefix. What the layout change actually threatens is the OTHER half:
    a NEW document arriving on a slot that already holds the prefix. Layouts A
    and B keep a byte-identical leading prefix, so that new document should
    reuse it; layout C has no leading prefix, so it should reuse nothing.

    This deliberately reuses one slot across two documents — the thing R13
    forbids — because cross-document reuse is precisely the quantity. The
    answers are stamped `phase="cache"` and never enter a quality table.
    """
    records: list[dict] = []
    previous: list[int] = []
    for idx, cell in enumerate(cells):
        spec = cell["questions"]["literal"]
        # The TRUE shared prefix, tokenized on the same server that will serve
        # the call. Recorded because `timings.cache_n` is the number under
        # suspicion here (§10 R8): a reuse figure that cannot be checked against
        # what the two prompts actually share is not evidence of reuse.
        rendered = await caller.client.apply_template(
            caller.compose(question=spec["question"], chunk=corpus[cell["uid"]]),
            chat_template_kwargs={"enable_thinking": caller.enable_thinking})
        tokens = await caller.client.tokenize(rendered, add_special=True)
        shared = 0
        for a, b in zip(previous, tokens):
            if a != b:
                break
            shared += 1
        previous = tokens
        answer = await caller.ask(question=spec["question"],
                                  chunk=corpus[cell["uid"]], seed=1, id_slot=slot)
        rec = {
            "phase": "cache", "arm": arm_id, "layout": caller.layout,
            "cell_uid": cell["uid"], "size_target": cell["size_tokens"],
            "density": cell["density"], "doc_index": idx,
            "shared_prefix_tokens": shared, "prompt_tokens_measured": len(tokens),
            "requested_slot": slot, "status": "ok", "label": None,
            "prefix_tokens": caller.prefix_tokens,
            "prefix_body_tokens": caller.prefix_body_tokens,
            "head_tokens": caller.head_tokens(),
            **answer.as_record(),
        }
        rec["slot_ok"] = (answer.slot_id == slot)
        append_run(rec, out_path)
        records.append(rec)
        echo(f"[cache] {arm_id} doc{idx} {cell['uid']} slot {answer.slot_id} "
             f"cached {rec['tokens_cached']}/{rec['tokens_in']} (truly shared "
             f"{shared}) prefill {rec['prefill_ms']}ms {rec['wall_s']}s")
    return records


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def rescore(records: list[dict]) -> list[dict]:
    """Re-derive every label from the verbatim `raw_output` — a classifier fix
    costs a re-read of a JSONL file, never a re-run of the GPU."""
    out: list[dict] = []
    for rec in records:
        rec = dict(rec)
        if rec.get("status") == "ok" and rec.get("question_type"):
            rec.update(score(rec.get("raw_output", ""),
                             question_type=rec["question_type"],
                             expected=rec.get("expected"),
                             expected_kind=rec.get("expected_kind")))
        out.append(rec)
    return out


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.0%})" if den else "-"


def tally(records: list[dict]) -> dict:
    """Label counts per question type over a set of records, plus the timing
    and cache columns every table in this report has to carry."""
    out = {q: {"n": 0, "labels": {lab: 0 for lab in LABELS}}
           for q in QUESTION_TYPES}
    extra = {"errors": 0, "fp_with_identifier": 0, "leaks": 0,
             "leak_not_checked": 0, "slot_mismatch": 0, "wall_s": [],
             "tokens_cached": [], "tokens_in": [], "warm_tokens_cached": [],
             "cold_tokens_cached": []}
    for rec in records:
        if rec.get("status") == "inadmissible":
            continue
        if rec.get("status") != "ok":
            extra["errors"] += 1
            continue
        q = out[rec["question_type"]]
        q["n"] += 1
        q["labels"][rec["label"]] = q["labels"].get(rec["label"], 0) + 1
        if rec["label"] == FALSE_POSITIVE and rec.get("supplied_identifier"):
            extra["fp_with_identifier"] += 1
        if rec.get("leak_detected") is True:
            extra["leaks"] += 1
        elif rec.get("leak_detected") is None:
            extra["leak_not_checked"] += 1
        if rec.get("slot_ok") is False:
            extra["slot_mismatch"] += 1
        for key in ("wall_s", "tokens_cached", "tokens_in"):
            if rec.get(key) is not None:
                extra[key].append(rec[key])
        if rec.get("tokens_cached") is not None:
            (extra["cold_tokens_cached"] if rec.get("cold")
             else extra["warm_tokens_cached"]).append(rec["tokens_cached"])
    return {"by_qtype": out, **extra}


def _fp(t: dict) -> tuple[int, int]:
    a = t["by_qtype"]["absent"]
    return a["labels"][FALSE_POSITIVE], a["n"]


def _recall(t: dict) -> tuple[int, int]:
    lit, par = t["by_qtype"]["literal"], t["by_qtype"]["paraphrase"]
    return lit["labels"][CORRECT] + par["labels"][CORRECT], lit["n"] + par["n"]


def _miss(t: dict) -> tuple[int, int]:
    lit, par = t["by_qtype"]["literal"], t["by_qtype"]["paraphrase"]
    return lit["labels"][MISS] + par["labels"][MISS], lit["n"] + par["n"]


def _med(values: list[float]) -> float | str:
    return round(statistics.median(values), 2) if values else "-"


def render_report(records: list[dict]) -> str:
    records = rescore(records)
    rep = [r for r in records if r.get("phase") == "replicate"]
    grid = [r for r in records if r.get("phase") == "grid"]
    cache = [r for r in records if r.get("phase") == "cache"]
    lines = [
        "# S2 — INSTRUCTION DECAY: distance vs distractor density",
        "",
        f"*Generated by `milestones/s2/run_distance.py --phase report` from "
        f"`milestones/s2/results/distance.jsonl` ({len(records)} records) on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}. Labels are "
        f"RE-DERIVED from each record's verbatim `raw_output` by "
        f"`s2.run_sweep.classify` — the instrument `milestones/s2/RESULTS.md` and "
        f"`milestones/s2/REFUSAL-AB.md` were scored with — not read back from the label "
        f"written at run time.*",
        "",
        "Taxonomy unchanged: **CORRECT** / **MISS** (refused a fact that is "
        "present) / **CONFABULATION** (non-refusal answer that is wrong) / "
        "**FALSE-POSITIVE** (any non-refusal answer to an ABSENT question) / "
        "**MALFORMED**. Every false-positive rate below is printed beside the "
        "RECALL it was bought with: an arm at 0% false positives and 0% recall "
        "is a failure, not a success.",
        "",
        "## 1. Replication — the 640-vs-1,024 result, arm A, the A/B's own fixtures",
        "",
    ]

    # ---- replication ------------------------------------------------------ #
    rep_by_size: dict[int, list[dict]] = {}
    for r in rep:
        rep_by_size.setdefault(r["size_target"], []).append(r)
    lines += [
        "| size | cells | FALSE-POSITIVE (absent) | ...handed over an identifier |"
        " CORRECT literal | CORRECT paraphrase | MISS (fact present) | MALFORMED |"
        " median wall s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rep_fp: dict[int, tuple[int, int]] = {}
    for size in sorted(rep_by_size):
        t = tally(rep_by_size[size])
        fp, fpn = _fp(t)
        rep_fp[size] = (fp, fpn)
        lit, par = t["by_qtype"]["literal"], t["by_qtype"]["paraphrase"]
        miss, missn = _miss(t)
        malformed = sum(q["labels"][MALFORMED] for q in t["by_qtype"].values())
        n_all = sum(q["n"] for q in t["by_qtype"].values())
        cells = len({r["cell_uid"] for r in rep_by_size[size]})
        lines.append(
            f"| {size} | {cells} | **{_rate(fp, fpn)}** | "
            f"{_rate(t['fp_with_identifier'], fpn)} | "
            f"{_rate(lit['labels'][CORRECT], lit['n'])} | "
            f"{_rate(par['labels'][CORRECT], par['n'])} | "
            f"{_rate(miss, missn)} | {_rate(malformed, n_all)} | "
            f"{_med(t['wall_s'])} |")
    if 640 in rep_fp and 1024 in rep_fp:
        f640, n640 = rep_fp[640]
        f1024, n1024 = rep_fp[1024]
        p = fisher_exact_two_sided(f1024, n1024 - f1024, f640, n640 - f640)
        lines += [
            "",
            f"**Replication: {f1024}/{n1024} at 1,024 vs {f640}/{n640} at 640, "
            f"Fisher two-sided p = {p:.3g}.** `milestones/s2/REFUSAL-AB.md` recorded 30/30 "
            f"vs 0/21 (p = 8.7e-15) on these same documents under this same "
            f"prefix. Everything below is conditional on this line.",
        ]

    # ---- the grid --------------------------------------------------------- #
    lines += [
        "",
        "## 2. The grid — layout x size x density",
        "",
        "Every cell: 3 question types x trials, one never-reused slot per "
        "(arm, cell), fixture seeds pooled as independent facts.",
        "",
        "| arm (layout) | size | density | bindings | n absent | FALSE-POS | "
        "...w/ identifier | n present | RECALL | MISS | MALFORMED | median "
        "wall s | median tokens_cached |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    key = lambda r: (r["arm"], r["size_target"], r["density"])  # noqa: E731
    groups: dict[tuple, list[dict]] = {}
    for r in grid:
        groups.setdefault(key(r), []).append(r)
    for (arm, size, density) in sorted(groups):
        rows = groups[(arm, size, density)]
        if all(r.get("status") == "inadmissible" for r in rows):
            head = rows[0].get("head_tokens")
            cap = rows[0].get("slot_capacity_tokens")
            lines.append(
                f"| `{arm}` | {size} | {density} | - | - | INADMISSIBLE | - | - | "
                f"- | - | - | - | {head}-token head + {size} chunk > {cap}/slot |")
            continue
        t = tally(rows)
        fp, fpn = _fp(t)
        rec, recn = _recall(t)
        miss, missn = _miss(t)
        malformed = sum(q["labels"][MALFORMED] for q in t["by_qtype"].values())
        n_all = sum(q["n"] for q in t["by_qtype"].values())
        bindings = rows[0].get("entity_bindings")
        lines.append(
            f"| `{arm}` | {size} | {density} | {bindings} | {fpn} | "
            f"**{_rate(fp, fpn)}** | {_rate(t['fp_with_identifier'], fpn)} | "
            f"{recn} | **{_rate(rec, recn)}** | {_rate(miss, missn)} | "
            f"{_rate(malformed, n_all)} | {_med(t['wall_s'])} | "
            f"{_med(t['tokens_cached'])} |")

    # ---- the verdict, computed rather than asserted ----------------------- #
    lines += [
        "",
        "## 3. DISTANCE vs DENSITY — the comparison this experiment exists for",
        "",
        "Both contrasts are within ONE arm, so layout cannot explain either.",
        "",
        "**(a) Distance, at FIXED density.** Matched-density cells carry the "
        "same 3 entity bindings at every size, so size varies distance alone.",
        "",
        "| arm | contrast | FALSE-POS smaller | FALSE-POS larger | Fisher p |",
        "|---|---|---|---|---|",
    ]

    def fp_of(arm: str, size: int, density: str) -> tuple[int, int]:
        rows = groups.get((arm, size, density), [])
        return _fp(tally(rows)) if rows else (0, 0)

    sizes = sorted({r["size_target"] for r in grid})
    for arm, _layout in ARMS:
        for small, large in zip(sizes, sizes[1:]):
            a1, n1 = fp_of(arm, small, "matched")
            a2, n2 = fp_of(arm, large, "matched")
            if not n1 or not n2:
                continue
            p = fisher_exact_two_sided(a2, n2 - a2, a1, n1 - a1)
            lines.append(f"| `{arm}` | matched {small} -> {large} | "
                         f"{_rate(a1, n1)} | {_rate(a2, n2)} | {p:.3g} |")
    lines += [
        "",
        "**(b) Density, at FIXED size.** Same distance, 3 bindings vs the "
        "sweep's natural rate.",
        "",
        "| arm | size | FALSE-POS matched | FALSE-POS natural | Fisher p |",
        "|---|---|---|---|---|",
    ]
    for arm, _layout in ARMS:
        for size in sizes:
            a1, n1 = fp_of(arm, size, "matched")
            a2, n2 = fp_of(arm, size, "natural")
            if not n1 or not n2:
                continue
            p = fisher_exact_two_sided(a2, n2 - a2, a1, n1 - a1)
            lines.append(f"| `{arm}` | {size} | {_rate(a1, n1)} | "
                         f"{_rate(a2, n2)} | {p:.3g} |")

    # ---- per-arm summary -------------------------------------------------- #
    lines += [
        "",
        "## 4. Per-arm totals — the false-positive rate and what it cost",
        "",
        "| arm (layout) | cells | calls | FALSE-POS | ...w/ identifier | RECALL "
        "| MISS | MALFORMED | median wall s | median tokens_in | median "
        "tokens_cached (warm) | median tokens_cached (cold) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, layout in ARMS:
        rows = [r for r in grid if r["arm"] == arm]
        if not rows:
            continue
        t = tally(rows)
        fp, fpn = _fp(t)
        rec, recn = _recall(t)
        miss, missn = _miss(t)
        malformed = sum(q["labels"][MALFORMED] for q in t["by_qtype"].values())
        n_all = sum(q["n"] for q in t["by_qtype"].values())
        cells = len({r["cell_uid"] for r in rows if r.get("status") == "ok"})
        lines.append(
            f"| `{arm}` ({layout}) | {cells} | {n_all} | **{_rate(fp, fpn)}** | "
            f"{_rate(t['fp_with_identifier'], fpn)} | **{_rate(rec, recn)}** | "
            f"{_rate(miss, missn)} | {_rate(malformed, n_all)} | "
            f"{_med(t['wall_s'])} | {_med(t['tokens_in'])} | "
            f"{_med(t['warm_tokens_cached'])} | {_med(t['cold_tokens_cached'])} |")

    zeros = []
    for arm, _layout in ARMS:
        rows = [r for r in grid if r["arm"] == arm]
        if not rows:
            continue
        fp, fpn = _fp(tally(rows))
        if fp == 0 and fpn:
            zeros.append(f"`{arm}` 0/{fpn} (95% upper bound "
                         f"{wilson_upper95(0, fpn):.1%})")
    if zeros:
        lines += ["", "Zero observed is never zero: " + "; ".join(zeros) + "."]

    lines += [
        "",
        "Layout is compared over the sizes EVERY arm admits; a row marked "
        "INADMISSIBLE in §2 contributes to neither its arm's totals nor anyone "
        "else's, and the arms are re-tallied over the common sizes here:",
        "",
        "| arm | common sizes | FALSE-POS | RECALL | MISS |",
        "|---|---|---|---|---|",
    ]
    admitted: dict[str, set[int]] = {}
    for r in grid:
        if r.get("status") == "ok":
            admitted.setdefault(r["arm"], set()).add(r["size_target"])
    common = set.intersection(*admitted.values()) if admitted else set()
    for arm, _layout in ARMS:
        rows = [r for r in grid if r["arm"] == arm and r["size_target"] in common]
        if not rows:
            continue
        t = tally(rows)
        fp, fpn = _fp(t)
        rec, recn = _recall(t)
        miss, missn = _miss(t)
        lines.append(f"| `{arm}` | {sorted(common)} | **{_rate(fp, fpn)}** | "
                     f"**{_rate(rec, recn)}** | {_rate(miss, missn)} |")
    if len(admitted) > 1:
        base = [r for r in grid if r["arm"] == ARMS[0][0]
                and r["size_target"] in common]
        a1, n1 = _fp(tally(base))
        r1, rn1 = _recall(tally(base))
        tests = []
        for arm, _layout in ARMS[1:]:
            rows = [r for r in grid if r["arm"] == arm
                    and r["size_target"] in common]
            if not rows:
                continue
            a2, n2 = _fp(tally(rows))
            r2, rn2 = _recall(tally(rows))
            tests.append(
                f"`{arm}` vs `{ARMS[0][0]}` — false positives "
                f"{_rate(a2, n2)} vs {_rate(a1, n1)}, Fisher p = "
                f"{fisher_exact_two_sided(a2, n2 - a2, a1, n1 - a1):.3g}; recall "
                f"{_rate(r2, rn2)} vs {_rate(r1, rn1)}, Fisher p = "
                f"{fisher_exact_two_sided(r2, rn2 - r2, r1, rn1 - r1):.3g}")
        lines += ["", "Layout tested against the shipped control, over the "
                  "common sizes, on BOTH axes:", ""]
        lines += [f"- {t}" for t in tests]

    # ---- the wall-clock confound, and its control ------------------------- #
    control = [r for r in records if r.get("phase") == "latency-control"
               and r.get("status") == "ok"]
    if control:
        lines += [
            "",
            "### 4b. The wall-clock column is CONFOUNDED — slot-pool occupancy",
            "",
            "The arms ran in order A, B, C, and R13 gives every cell a fresh "
            "never-reused slot, so arm order is also slot-index order and "
            "pool-occupancy order. This CONTROL re-runs the SHIPPED layout (arm "
            "A, unchanged) at the high slot indices arm C used, on cells arm A "
            "already ran at low indices. Server-reported prefill and decode are "
            "the comparison's own control: they are flat while wall is not.",
            "",
            "| run | layout | slots | n | median wall s | median prefill ms | "
            "median decode ms | median tokens_out | FALSE-POS |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        base = [r for r in grid if r["arm"] == "A-shipped"
                and r["size_target"] == 1024 and r.get("status") == "ok"]
        for label, rows in (("grid (early)", base), ("control (late)", control)):
            slots = sorted({r["requested_slot"] for r in rows})
            t = tally(rows)
            fp, fpn = _fp(t)
            lines.append(
                f"| {label} | A | {slots[0]}–{slots[-1]} | {len(rows)} | "
                f"**{_med([r['wall_s'] for r in rows])}** | "
                f"{_med([r['prefill_ms'] for r in rows])} | "
                f"{_med([r['decode_ms'] for r in rows])} | "
                f"{_med([r['tokens_out'] for r in rows])} | {_rate(fp, fpn)} |")
        lines += [
            "",
            "So the per-arm wall-clock in §4 prices the slot index, not the "
            "layout, and it is reported here rather than deleted. What the "
            "layout actually costs is TOKENS, which are exact: see the head "
            "column in §2 and the `tokens_in` column in §4.",
        ]

    # ---- cache probe ------------------------------------------------------ #
    lines += [
        "",
        "## 5. The cache price of the layout (§4's prefix contract)",
        "",
        "One slot per layout, two DIFFERENT documents in sequence — deliberately "
        "the reuse R13 forbids, because cross-document prefix reuse is exactly "
        "the quantity §4 trades the layout for. `doc 0` is a cold slot; `doc 1` "
        "arrives on a slot that already holds the prefix.",
        "",
        "| arm (layout) | slot | head tokens | true shared prefix | doc 0 "
        "tokens_cached | doc 1 tokens_cached | doc 0 prefill ms | doc 1 prefill "
        "ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, layout in ARMS:
        for slot in sorted({r["requested_slot"] for r in cache
                            if r["arm"] == arm}):
            rows = sorted([r for r in cache if r["arm"] == arm
                           and r["requested_slot"] == slot],
                          key=lambda r: r["doc_index"])
            if len(rows) < 2:
                continue
            lines.append(
                f"| `{arm}` ({layout}) | {slot} | {rows[0].get('head_tokens')} | "
                f"{rows[1].get('shared_prefix_tokens', '?')} | "
                f"{rows[0]['tokens_cached']} | **{rows[1]['tokens_cached']}** | "
                f"{rows[0]['prefill_ms']} | {rows[1]['prefill_ms']} |")

    # ---- R13 audit -------------------------------------------------------- #
    lines += [
        "",
        "## 6. R13 contamination and slot audit",
        "",
        "| arm | calls | foreign-identifier hits | NOT CHECKED | slot mismatches "
        "| errors |",
        "|---|---|---|---|---|---|",
    ]
    for arm, _layout in ARMS + (("REPLICATION", "A"),):
        rows = [r for r in records
                if r.get("arm") == arm and r.get("phase") != "cache"]
        if not rows:
            continue
        t = tally(rows)
        n_all = sum(q["n"] for q in t["by_qtype"].values())
        lines.append(f"| `{arm}` | {n_all} | **{t['leaks']}** | "
                     f"{t['leak_not_checked']} | {t['slot_mismatch']} | "
                     f"{t['errors']} |")
    slot_owners: dict[int, set[str]] = {}
    for r in records:
        if r.get("phase") == "cache" or r.get("requested_slot") is None:
            continue
        slot_owners.setdefault(r["requested_slot"], set()).add(
            f"{r.get('arm')}:{r.get('cell_uid')}")
    shared = {s: sorted(o) for s, o in slot_owners.items() if len(o) > 1}
    lines += [
        "",
        f"**Never-reused-slot check:** {len(slot_owners)} slot(s) used across the "
        f"quality phases; slots serving more than one (arm, cell): "
        f"{shared if shared else 'none'}. A nonzero entry invalidates the arms it "
        f"lands in. Per §10 R13 a clean verdict is evidence, never a certificate.",
        "",
        "## 7. Verbatim — every FALSE-POSITIVE and MALFORMED in the grid, and a "
        "sample of every other class",
        "",
        "| arm | cell | size | density | question | trial | label | raw output |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in grid:
        if r.get("label") not in (FALSE_POSITIVE, MALFORMED):
            continue
        raw = " ".join((r.get("raw_output") or "").split())[:200]
        lines.append(
            f"| `{r['arm']}` | {r['cell_uid']} | {r['size_target']} | "
            f"{r['density']} | {r['question_type']} | {r['trial']} | "
            f"{r['label']} | `{raw}` |")
    seen: set[tuple] = set()
    for r in grid:
        k = (r.get("arm"), r.get("label"), r.get("question_type"))
        if r.get("label") in (None, FALSE_POSITIVE, MALFORMED) or k in seen:
            continue
        seen.add(k)
        raw = " ".join((r.get("raw_output") or "").split())[:200]
        lines.append(
            f"| `{r['arm']}` | {r['cell_uid']} | {r['size_target']} | "
            f"{r['density']} | {r['question_type']} | {r['trial']} | "
            f"{r['label']} | `{raw}` |")
    lines.append("")
    return "\n".join(lines)


def regenerate(records: list[dict], report_md: Path = REPORT_MD) -> str:
    generated = render_report(records)
    narrative = ""
    if report_md.exists():
        old = report_md.read_text(encoding="utf-8")
        if NARRATIVE_MARKER in old:
            narrative = old.split(NARRATIVE_MARKER, 1)[1]
    return f"{generated}\n{NARRATIVE_MARKER}\n{narrative.lstrip(chr(10))}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def _build_caller(cfg, layout: str):
    from rlm.config import PromptRegistry
    from s2.leafcall import PinnedLeafCaller

    registry = PromptRegistry.from_files(
        root_path=REPO_ROOT / "prompts/root.v3.md",
        leaf_prefix_path=REPO_ROOT / LEAF_PREFIX,
        leaf_envelope_path=None,
        strategy_paths={},
    ).load()
    caller = PinnedLeafCaller.from_config(
        cfg, system_prefix=registry.render_leaf(envelope=False), envelope=False)
    caller.layout = layout
    await caller.prepare()
    return caller


async def _amain(args) -> int:
    if args.phase == "report":
        report_md = Path(args.report_md) if args.report_md else REPORT_MD
        report_md.write_text(regenerate(read_runs(args.out), report_md),
                             encoding="utf-8", newline="\n")
        print(f"wrote {report_md}")
        return 0

    from rlm.config import load_config

    cfg = load_config(Path(args.config))
    state = {"next": args.first_slot}

    def next_slot() -> int:
        slot = state["next"]
        state["next"] += 1
        if slot >= cfg.servers.leaf.parallel:
            raise SystemExit(f"slot pool exhausted at {slot} (R13)")
        return slot

    if args.phase == "replicate":
        d1024, d640 = replication_dirs()
        cells_1024, corpus_1024 = load_cells(d1024, only=REPLICATION_1024_CELLS)
        cells_640, corpus_640 = load_cells(d640)
        cells = cells_640 + cells_1024
        corpus = {**corpus_640, **corpus_1024}
        index = ChunkIndex.from_chunks(corpus)
        caller = await _build_caller(cfg, "A")
        print(f"REPLICATION: layout A, {len(cells)} cells "
              f"({len(cells_640)} @640, {len(cells_1024)} @1024), "
              f"prefix {caller.prefix_tokens} rendered / "
              f"{caller.prefix_body_tokens} body tokens")
        try:
            await run_arm(caller, arm_id="REPLICATION", phase="replicate",
                          cells=cells, corpus=corpus, index=index,
                          next_slot=next_slot, trials=args.trials,
                          seeds=tuple(args.seeds), out_path=args.out)
        finally:
            await caller.aclose()

    elif args.phase in ("grid", "cache"):
        dirs = [Path(d) for d in args.fixtures]
        cells, corpus = load_cells(dirs)
        if args.sizes:
            cells = [c for c in cells if c["size_tokens"] in args.sizes]
        index = ChunkIndex.from_chunks(corpus)
        arms = [a for a in ARMS if not args.arms or a[0] in args.arms]
        for arm_id, layout in arms:
            caller = await _build_caller(cfg, layout)
            print(f"\n=== arm {arm_id} (layout {layout}): head "
                  f"{caller.head_tokens()} tokens, slot capacity "
                  f"{caller.slot_capacity_tokens} ===")
            try:
                if args.phase == "grid":
                    await run_arm(caller, arm_id=arm_id,
                                  phase=args.phase_label or "grid",
                                  cells=cells, corpus=corpus, index=index,
                                  next_slot=next_slot, trials=args.trials,
                                  seeds=tuple(args.seeds), out_path=args.out)
                else:
                    probe = [c for c in cells
                             if c["size_tokens"] == args.cache_size][:2]
                    await run_cache_probe(caller, arm_id=arm_id, cells=probe,
                                          corpus=corpus, slot=next_slot(),
                                          out_path=args.out)
            finally:
                await caller.aclose()

    if not args.no_report:
        report_md = Path(args.report_md) if args.report_md else REPORT_MD
        report_md.write_text(regenerate(read_runs(args.out), report_md),
                             encoding="utf-8", newline="\n")
        print(f"wrote {report_md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S2 instruction-decay experiment: distance vs density")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="grid",
                        choices=("replicate", "grid", "cache", "report"))
    parser.add_argument("--fixtures", nargs="+",
                        default=[str(S2_DIR / f"fixtures-distance-s{s}")
                                 for s in (11, 12, 13, 14)])
    parser.add_argument("--out", type=Path, default=RUNS_PATH)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--layouts", nargs="*", default=list(LAYOUTS),
                        help="informational: the arms carry the layouts")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRIAL_SEEDS))
    parser.add_argument("--first-slot", type=int, default=0,
                        help="first never-reused slot index (R13). Phases are "
                             "separate processes, so this must not overlap a "
                             "range an earlier phase already spent; the report "
                             "audits it.")
    parser.add_argument("--sizes", nargs="*", type=int, default=None,
                        help="restrict to these chunk sizes")
    parser.add_argument("--phase-label", default=None,
                        help="stamp records with a phase other than the grid — "
                             "used by the slot-occupancy latency CONTROL, whose "
                             "calls must never enter a quality table")
    parser.add_argument("--cache-size", type=int, default=1024)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
