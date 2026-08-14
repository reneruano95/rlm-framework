"""S2 — the leaf REFUSAL A/B: does an explicit abstention channel move the 95%
false-positive rate? (spec §5 "leaf-envelope A/B", §10 R5, §7 #2.)

THE NUMBER THIS EXISTS TO ATTACK. `s2/RESULTS.md` finding 3: asked about a fact
that is NOT in the chunk, the leaf answered anyway **37 of 39 times (95%)**, and
that rate is the one figure in the sweep that is FLAT across every chunk size
from 1K to 32K. Finding: **a leaf answer carries no information about whether
the fact was present.** Every "root as orchestrator of leaves" claim in §2 is
downstream of that being fixable.

§10 R5 pre-registers the replacements and this file measures two of them, as one
2x2, pre-registered before the run and not extended afterwards:

    (a) leaf-prefix.v1, no envelope   — the control: the recorded 95% baseline
    (b) leaf-prefix.v1, envelope      — abstention as a structured field
    (c) leaf-prefix.v2, no envelope   — abstention argued for in prose
    (d) leaf-prefix.v2, envelope      — both

**Exactly two prefixes, authored before the first call, neither edited after.**
Unbounded prompt search against measurement fixtures is the overfitting §8's
freeze rule exists to forbid, and it would be especially cheap to do here
because the fixtures are generated and the metric is a single rate.

WHY THE ENVELOPE MIGHT WORK, stated as a falsifiable claim rather than a hope:
v1 already offers a refusal token (`NONE`, and its rules say in as many words
that `NONE` is correct and useful), so "the model was never offered a way to
refuse" is FALSE going in. What v1 does not offer is a refusal that is
structurally separate from the answer — the model must produce the same kind of
object (a line of text) for "here it is" and for "not here", and the format of
the reply carries no signal. The envelope makes abstention a boolean the
scaffold reads directly. If that changes nothing, the honest reading is that the
false-positive rate is a property of this model at this quantization rather than
of the prompt, and §10 R1 commits this project to publishing that rather than
tuning it away.

THE MEASUREMENT'S OWN RULES.

  * **Fact-present questions run in every arm**, and their MISS column is as
    load-bearing as the false-positive column. A leaf that abstains on
    everything has a 0% false-positive rate and is useless; any arm that buys
    refusals with recall must be reported as having done so.
  * **The scoring taxonomy is the sweep's, unchanged** (CORRECT / MISS /
    CONFABULATION / FALSE-POSITIVE / MALFORMED), and envelope replies are
    scored by REDUCING them to the plain-text answer the same classifier sees,
    so the four arms are compared by one scorer and not by four.
  * **No retries.** `s2/leafcall.py` deliberately has none, and adding one here
    would confound the arms: an arm that emits more malformed replies would get
    more draws at the question. A parse failure is recorded as MALFORMED and
    reported as such; production (C4) retries these, and the report says so
    rather than quietly modelling it.
  * **R13 slot discipline (§10 R13).** One never-reused slot per (arm, cell) —
    a fresh slot for every document, from a pool of 128, with C4's own
    assertion that the server answered on the slot that was asked for. The
    intra-cell re-queries share that slot on purpose: same-document reuse is
    the one reuse pattern measured clean. R13's foreign-string detector then
    runs over every answer against the whole fixture corpus, and its hit count
    is reported per arm — a contaminated measurement of a contamination-adjacent
    property would be worthless.

WHY 1,024-TOKEN CELLS ONLY. The pinned leaf topology is `-np 128 -c 327680` =
**2,560 tokens per slot** (`s2/R13-slotcount.md`), which is `chunk 1024 +
overhead 1536` — the shipped window geometry exactly. A 2,048-token cell plus
the envelope's own prefix does not fit one slot, and re-launching the server
with a bigger slot to fit bigger chunks would measure a topology the runtime
does not use. Since the false-positive rate is the one quantity the sweep
measured as FLAT in chunk size, measuring it at the production window costs
nothing and buys the right generalization. Statistical power comes from more
CELLS (independent generated facts, several fixture seeds) rather than more
trials against one fact.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `uv run s2/run_refusal_ab.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.envelope import parse as envelope_parse  # noqa: E402
from rlm.envelope import verify_evidence  # noqa: E402
from rlm.leakcheck import ChunkIndex, identifier_tokens  # noqa: E402
from s2.make_sweep_fixtures import MANIFEST_NAME, QUESTION_TYPES  # noqa: E402
from s2.run_sweep import (  # noqa: E402
    CONFABULATION,
    CORRECT,
    FALSE_POSITIVE,
    LABELS,
    MALFORMED,
    MISS,
    append_run,
    classify,
    is_refusal,
    normalize,
    plan_calls,
    read_runs,
)

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "refusal-ab.jsonl"
REPORT_MD = S2_DIR / "REFUSAL-AB.md"

TRIAL_SEEDS: tuple[int, ...] = (1, 2, 3)

#: The 2x2, pre-registered. `(arm_id, prefix file, envelope on)`.
ARMS: tuple[tuple[str, str, bool], ...] = (
    ("a-v1-plain", "prompts/leaf-prefix.v1.md", False),
    ("b-v1-envelope", "prompts/leaf-prefix.v1.md", True),
    ("c-v2-plain", "prompts/leaf-prefix.v2.md", False),
    ("d-v2-envelope", "prompts/leaf-prefix.v2.md", True),
)
ENVELOPE_BLOCK = "prompts/leaf-envelope.v2.md"

NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"


# --------------------------------------------------------------------------- #
# scoring: reduce an envelope to what the sweep's classifier already scores
# --------------------------------------------------------------------------- #

#: What a reduced envelope refusal looks like to `s2.run_sweep.classify`. The
#: literal token both leaf prefixes name, so the plain arms and the envelope
#: arms are judged by the same `is_refusal` rules rather than by a second,
#: envelope-only notion of refusal.
REFUSAL_TOKEN = "NONE"


def reduce_envelope(raw: str, *, chunk: str | None) -> dict[str, Any]:
    """Turn one envelope reply into `(text, facts)` the sweep's scorer can take.

    THE ABSTAIN-WITH-AN-ANSWER RULE, and it is a scoring decision so it lives
    here rather than in `rlm.envelope`: `abstain: true` beside a substantive
    `answer` is NOT counted as a refusal. It is the structured twin of the
    plain-text hedge `is_refusal` already refuses to credit ("NONE, but the
    closest match is ..."), and crediting it would understate exactly the
    quantity this experiment measures. The `answer` field is what a root would
    read and submit; if it holds a value, a value was supplied.
    """
    result = envelope_parse(raw)
    out: dict[str, Any] = {
        "envelope_ok": result.ok,
        "envelope_error": result.error,
        "envelope_salvaged": result.salvaged,
        "abstain": None,
        "evidence": None,
        "evidence_verified": None,
        "evidence_ok": None,
        "extras": None,
        "reduced_text": raw,
    }
    if not result.ok:
        # MALFORMED by construction: `classify` labels an unparseable reply from
        # its own degeneracy rules, and an envelope that did not parse is a
        # reply the runtime could not have used.
        out["reduced_text"] = ""
        return out

    env = result.envelope
    verified = verify_evidence(env.evidence, chunk=chunk)
    out.update(
        abstain=env.abstain,
        evidence=list(env.evidence),
        evidence_verified=list(verified),
        evidence_ok=(None if (not verified or any(v is None for v in verified))
                     else all(verified)),
        extras=list(env.extras),
    )
    answer_norm = normalize(env.answer)
    supplied = bool(answer_norm) and not is_refusal(answer_norm)
    if env.abstain and not supplied:
        out["reduced_text"] = REFUSAL_TOKEN
    else:
        out["reduced_text"] = env.answer
    out["abstain_with_answer"] = bool(env.abstain and supplied)
    return out


def supplied_identifier(text: str) -> bool:
    """Does this answer hand a root an identifier-shaped token it could submit?

    WHY THIS EXISTS, and it is an honesty fix rather than a nicety. The sweep's
    pinned `is_refusal` matches a phrase list, and the list does not contain
    "the provided text does not contain ..." — it has `(the|this) excerpt does
    not`, and the leaf says "provided text". A verbose refusal therefore scores
    FALSE-POSITIVE. That biases the measurement in the worst possible
    direction here: the treatments under test (v2's refusal argument, the
    envelope's abstain field) are exactly the things that make refusals more
    verbose, so the pinned classifier penalises the arms it is measuring.

    Widening the phrase list is not available: it is the instrument
    `s2/RESULTS.md` was scored with, and moving it would move the 95% baseline
    the whole experiment is compared against. So the pinned label stays
    primary, and this reports the mechanical sub-question underneath it —
    **did the reply actually hand over a value?** — using `rlm.leakcheck`'s
    identifier patterns, which are pattern-matching on shape, not on prose.

    It is the number the architecture actually cares about: §2's error model
    breaks when a root is handed a wrong key and submits it, not when a leaf is
    wordy. A FALSE-POSITIVE carrying no identifier at all is a leaf failing to
    say `NONE` crisply; one carrying a UUID is a leaf lying.
    """
    return bool(identifier_tokens(text or ""))


def score(raw: str, *, envelope: bool, chunk: str | None, question_type: str,
          expected: str | None, expected_kind: str | None) -> dict[str, Any]:
    """One record's labels and envelope facts. `classify` is imported, never
    re-implemented: the whole point is that all four arms are scored by the
    instrument `s2/RESULTS.md` was scored by."""
    facts: dict[str, Any] = {"envelope_ok": None, "envelope_error": None,
                             "envelope_salvaged": None, "abstain": None,
                             "evidence": None, "evidence_verified": None,
                             "evidence_ok": None, "extras": None,
                             "abstain_with_answer": None, "reduced_text": raw}
    if envelope:
        facts.update(reduce_envelope(raw, chunk=chunk))
    labels = classify(facts["reduced_text"], question_type=question_type,
                      expected=expected, expected_kind=expected_kind)
    # Measured on the REDUCED text -- what a root would actually read and be
    # able to submit -- so an envelope arm is judged on its `answer` field
    # rather than on the JSON punctuation around it.
    labels["supplied_identifier"] = supplied_identifier(facts["reduced_text"])
    return {**facts, **labels}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def load_cells(dirs: list[Path]) -> tuple[list[dict], dict[str, str]]:
    """Every cell from every fixture directory, plus {chunk_id: text} for the
    R13 detector. Cells are tagged with the manifest seed so two cells built at
    the same (size, position) from different seeds stay distinguishable — that
    is how this experiment gets independent facts at ONE chunk size."""
    cells: list[dict] = []
    corpus: dict[str, str] = {}
    for d in dirs:
        manifest = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("token_counter") != "leaf:/tokenize":
            raise SystemExit(
                f"REFUSING: {d / MANIFEST_NAME} was built with "
                f"{manifest.get('token_counter')!r}. The slot-fit arithmetic and "
                f"every size claim here are in leaf tokens; rebuild against the "
                f"live tokenizer.")
        seed = manifest.get("seed")
        for cell in manifest["cells"].values():
            cell = dict(cell)
            cell["fixture_seed"] = seed
            cell["uid"] = f"{cell['cell_id']}#s{seed}"
            text = Path(cell["chunk_path"]).read_text(encoding="utf-8")
            if cell["uid"] in corpus:
                raise SystemExit(f"duplicate fixture cell {cell['uid']}")
            corpus[cell["uid"]] = text
            cells.append(cell)
    cells.sort(key=lambda c: (c["size_tokens"], c["position"], c["fixture_seed"]))
    return cells, corpus


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


async def run_arm(caller, arm_id: str, cells: list[dict], corpus: dict[str, str],
                  index: ChunkIndex, *, next_slot, trials: int,
                  seeds: tuple[int, ...], out_path: Path, echo=print) -> list[dict]:
    """One arm over every cell. A fresh, never-reused slot per cell (R13)."""
    records: list[dict] = []
    for cell in cells:
        text = corpus[cell["uid"]]
        if not caller.admits(cell["measured_tokens"]):
            echo(f"[skip] {arm_id} {cell['uid']}: {cell['measured_tokens']} tokens "
                 f"+ a {caller.prefix_tokens}-token prefix does not fit one "
                 f"{caller.slot_capacity_tokens}-token slot")
            continue
        slot = next_slot()
        for call in plan_calls([cell], trials=trials, seeds=seeds):
            spec = cell["questions"][call["question_type"]]
            record = {
                "arm": arm_id,
                "prefix_sha256": caller.prefix_sha256,
                "envelope": caller.envelope,
                "cell_uid": cell["uid"],
                "cell_id": cell["cell_id"],
                "fixture_seed": cell["fixture_seed"],
                "size_target": cell["size_tokens"],
                "size_measured": cell["measured_tokens"],
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
            # R13's slot assertion, the same one C4 makes: an out-of-range
            # id_slot is silently reassigned with HTTP 200, so a run that trusts
            # its own request believes it holds a virgin slot while sharing a
            # used one.
            record["slot_ok"] = (answer.slot_id == slot)
            record.update(score(answer.raw, envelope=caller.envelope, chunk=text,
                                question_type=call["question_type"],
                                expected=spec["expected"],
                                expected_kind=spec["expected_kind"]))
            # R13's free detector, over the WHOLE fixture corpus: an
            # identifier-shaped token absent from what was sent and present in
            # another cell cannot have come from this call's document.
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
                 f"{record['wall_s']}s {record['normalized'][:60]!r}")
    return records


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def rescore(records: list[dict]) -> list[dict]:
    """Re-derive every label from the verbatim `raw_output`, exactly as
    `s2/run_sweep.py --phase report` does: a classifier fix must cost a re-read
    of a JSONL file, never a re-run of the GPU."""
    out: list[dict] = []
    for rec in records:
        rec = dict(rec)
        if rec.get("status") == "ok":
            chunk = rec.get("_chunk")
            rec.update(score(rec.get("raw_output", ""),
                             envelope=bool(rec.get("envelope")),
                             chunk=chunk,
                             question_type=rec["question_type"],
                             expected=rec.get("expected"),
                             expected_kind=rec.get("expected_kind")))
        out.append(rec)
    return out


def summarize(records: list[dict]) -> dict:
    arms: dict[str, dict] = {}
    for rec in records:
        arm = arms.setdefault(rec["arm"], {
            "arm": rec["arm"], "envelope": rec.get("envelope"),
            "by_qtype": {q: {"n": 0, "labels": {lab: 0 for lab in LABELS}}
                         for q in QUESTION_TYPES},
            "errors": 0, "leaks": 0, "leak_not_checked": 0, "slot_mismatch": 0,
            "envelope_parsed": 0, "envelope_failed": 0, "envelope_salvaged": 0,
        "fp_with_identifier": 0,
            "abstain_true": 0, "abstain_with_answer": 0,
            "evidence_spans": 0, "evidence_verified": 0,
            "wall_s": [], "tokens_out": [],
        })
        if rec.get("status") != "ok":
            arm["errors"] += 1
            continue
        cell = arm["by_qtype"][rec["question_type"]]
        cell["n"] += 1
        cell["labels"][rec["label"]] = cell["labels"].get(rec["label"], 0) + 1
        if rec["label"] == FALSE_POSITIVE and rec.get("supplied_identifier"):
            arm["fp_with_identifier"] += 1
        if rec.get("leak_detected") is True:
            arm["leaks"] += 1
        elif rec.get("leak_detected") is None:
            arm["leak_not_checked"] += 1
        if rec.get("slot_ok") is False:
            arm["slot_mismatch"] += 1
        if rec.get("envelope"):
            if rec.get("envelope_ok"):
                arm["envelope_parsed"] += 1
                if rec.get("envelope_salvaged"):
                    arm["envelope_salvaged"] += 1
                if rec.get("abstain"):
                    arm["abstain_true"] += 1
                if rec.get("abstain_with_answer"):
                    arm["abstain_with_answer"] += 1
                for v in (rec.get("evidence_verified") or []):
                    arm["evidence_spans"] += 1
                    if v is True:
                        arm["evidence_verified"] += 1
            else:
                arm["envelope_failed"] += 1
        if rec.get("wall_s") is not None:
            arm["wall_s"].append(rec["wall_s"])
        if rec.get("tokens_out") is not None:
            arm["tokens_out"].append(rec["tokens_out"])
    return {"arms": dict(sorted(arms.items()))}


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.0%})" if den else "-"


def common_cells(records: list[dict]) -> set[str]:
    """Cells every arm actually ran.

    THE ARMS DO NOT ADMIT THE SAME CELLS, and pretending otherwise would be the
    experiment's worst available bug. The four prefixes render to 311 / 773 /
    577 / 1039 tokens, so against a 2,560-token slot the plain-v1 arm admits a
    2,048-token cell that the envelope arms cannot fit. Comparing arms over
    their own admitted sets would compare four different corpora and call the
    difference an effect. The headline table is therefore over the
    INTERSECTION; the per-arm grid below it still shows everything each arm ran.
    """
    by_arm: dict[str, set[str]] = {}
    for rec in records:
        if rec.get("cell_uid"):
            by_arm.setdefault(rec["arm"], set()).add(rec["cell_uid"])
    if not by_arm:
        return set()
    return set.intersection(*by_arm.values())


def render_report(records: list[dict]) -> str:
    records = rescore(records)
    shared = common_cells(records)
    summary = summarize([r for r in records if r.get("cell_uid") in shared])
    arms = summary["arms"]
    cells = sorted({r["cell_uid"] for r in records if r.get("cell_uid")})
    lines = [
        "# S2 — the leaf refusal A/B: does an abstention channel move the 95%?",
        "",
        f"*Generated by `s2/run_refusal_ab.py --phase report` from "
        f"`{RUNS_PATH.relative_to(REPO_ROOT).as_posix()}` ({len(records)} calls, "
        f"{len(cells)} cells) on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}. Labels are "
        f"RE-DERIVED from each record's verbatim `raw_output`, not read back from "
        f"the label written at run time.*",
        "",
        f"**Every table below is over the {len(shared)} cell(s) EVERY arm ran.** "
        f"The four prefixes render to different lengths, so against a fixed "
        f"2,560-token slot they do not admit the same cells; scoring each arm "
        f"over its own admitted set would compare four different corpora and "
        f"report the difference as an effect. Cells run by some arms but not all "
        f"({len(cells) - len(shared)} of {len(cells)}) are excluded here and "
        f"remain in the JSONL.",
        "",
        "Taxonomy is `s2/RESULTS.md`'s, unchanged, so the numbers are comparable: "
        "**CORRECT** / **MISS** (refused a fact that is present) / "
        "**CONFABULATION** (non-refusal answer that is wrong) / **FALSE-POSITIVE** "
        "(any non-refusal answer to an ABSENT question) / **MALFORMED** (empty, "
        "degenerate, or — in the envelope arms — unparseable).",
        "",
        "## The four arms",
        "",
        "| arm | prefix | envelope | FALSE-POSITIVE rate (absent) | "
        "...of which handed over an identifier | CORRECT (literal) | "
        "CORRECT (paraphrase) | MISS (fact present) | MALFORMED | "
        "median wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm_id, arm in arms.items():
        absent = arm["by_qtype"]["absent"]
        lit = arm["by_qtype"]["literal"]
        par = arm["by_qtype"]["paraphrase"]
        present_n = lit["n"] + par["n"]
        present_miss = lit["labels"][MISS] + par["labels"][MISS]
        malformed = sum(q["labels"][MALFORMED] for q in arm["by_qtype"].values())
        n_all = sum(q["n"] for q in arm["by_qtype"].values())
        wall = arm["wall_s"]
        prefix = "v2" if arm_id.split("-")[1] == "v2" else "v1"
        lines.append(
            f"| `{arm_id}` | {prefix} | {'on' if arm['envelope'] else 'off'} | "
            f"**{_rate(absent['labels'][FALSE_POSITIVE], absent['n'])}** | "
            f"**{_rate(arm['fp_with_identifier'], absent['n'])}** | "
            f"{_rate(lit['labels'][CORRECT], lit['n'])} | "
            f"{_rate(par['labels'][CORRECT], par['n'])} | "
            f"**{_rate(present_miss, present_n)}** | "
            f"{_rate(malformed, n_all)} | "
            f"{round(statistics.median(wall), 2) if wall else '-'} |")

    lines += [
        "",
        "The bold columns are the trade this experiment exists to price. A leaf "
        "that abstains on everything scores 0% false positives and is useless, "
        "so any drop in the first must be read against MISS.",
        "",
        "**\"...of which handed over an identifier\"** is the column that matters "
        "for §2's error model, and it exists because the pinned classifier has a "
        "known gap that biases AGAINST the treatments: `is_refusal` matches a "
        "phrase list containing `(the|this) excerpt does not`, and this leaf "
        "writes \"the provided text does not contain ...\", so a verbose refusal "
        "scores FALSE-POSITIVE. Widening that list is not available — it is the "
        "instrument `s2/RESULTS.md` was scored with, and moving it moves the "
        "baseline. So the pinned label stays primary and this reports the "
        "mechanical sub-question underneath it: did the reply actually hand a "
        "root a UUID or `ENT-` code it could submit (`rlm.leakcheck` patterns, "
        "shape not prose)? A false positive carrying no identifier is a leaf "
        "being wordy; one carrying a UUID is a leaf lying.",
        "",
        "## Full label grid",
        "",
        "| arm | question | n | CORRECT | MISS | CONFAB | FALSE-POS | MALFORMED |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm_id, arm in arms.items():
        for qtype in QUESTION_TYPES:
            q = arm["by_qtype"][qtype]
            lab = q["labels"]
            lines.append(
                f"| `{arm_id}` | {qtype} | {q['n']} | {lab[CORRECT]} | {lab[MISS]} | "
                f"{lab[CONFABULATION]} | {lab[FALSE_POSITIVE]} | {lab[MALFORMED]} |")

    lines += [
        "",
        "## Envelope mechanics (envelope arms only)",
        "",
        "| arm | parsed | of which salvaged from prose | parse failures | "
        "`abstain: true` | abstain WITH an answer | evidence spans verified |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm_id, arm in arms.items():
        if not arm["envelope"]:
            continue
        total = arm["envelope_parsed"] + arm["envelope_failed"]
        lines.append(
            f"| `{arm_id}` | {_rate(arm['envelope_parsed'], total)} | "
            f"{arm['envelope_salvaged']} | {arm['envelope_failed']} | "
            f"{arm['abstain_true']} | {arm['abstain_with_answer']} | "
            f"{_rate(arm['evidence_verified'], arm['evidence_spans'])} |")

    lines += [
        "",
        "`abstain WITH an answer` is counted as a NON-refusal (see "
        "`reduce_envelope`): it is the structured twin of the plain-text hedge "
        "the sweep's `is_refusal` already declines to credit.",
        "",
        "The evidence column is a direct re-measurement of §10 R5's finding, on "
        "fresh data: a high verified rate beside a high false-positive rate is "
        "the whole point — it means the span check passes on answers that are "
        "wrong, which is what makes it near-inert as a defence.",
        "",
        "## The same envelope arms, re-scored as PLAIN TEXT (format ignored)",
        "",
        "SECONDARY, and a decomposition rather than a second chance. The tables "
        "above score an unparseable reply MALFORMED because that is what the "
        "runtime gets: C4 retries it and then raises `EnvelopeParseError`. But a "
        "reply can fail the FORMAT while still carrying the CONTENT this "
        "experiment is about — several unparseable replies below are refusals "
        "written in prose. Re-scoring the same raw bytes with the plain-text "
        "classifier separates *the block changed what the model says* from *the "
        "block broke the reply*. Only the headline table decides anything.",
        "",
        "| arm | FALSE-POSITIVE rate (absent) | CORRECT (literal) | "
        "CORRECT (paraphrase) | MISS (fact present) | MALFORMED |",
        "|---|---|---|---|---|---|",
    ]
    as_plain = rescore([{**r, "envelope": False} for r in records
                        if r.get("envelope") and r.get("cell_uid") in shared])
    for arm_id, arm in summarize(as_plain)["arms"].items():
        absent = arm["by_qtype"]["absent"]
        lit = arm["by_qtype"]["literal"]
        par = arm["by_qtype"]["paraphrase"]
        present_n = lit["n"] + par["n"]
        present_miss = lit["labels"][MISS] + par["labels"][MISS]
        n_all = sum(q["n"] for q in arm["by_qtype"].values())
        malformed = sum(q["labels"][MALFORMED] for q in arm["by_qtype"].values())
        lines.append(
            f"| `{arm_id}` | "
            f"**{_rate(absent['labels'][FALSE_POSITIVE], absent['n'])}** | "
            f"{_rate(lit['labels'][CORRECT], lit['n'])} | "
            f"{_rate(par['labels'][CORRECT], par['n'])} | "
            f"**{_rate(present_miss, present_n)}** | {_rate(malformed, n_all)} |")

    lines += [
        "",
        "## R13 contamination audit (every answer, every arm)",
        "",
        "| arm | calls | foreign-identifier hits | NOT CHECKED | slot mismatches | errors |",
        "|---|---|---|---|---|---|",
    ]
    for arm_id, arm in arms.items():
        n_all = sum(q["n"] for q in arm["by_qtype"].values())
        lines.append(
            f"| `{arm_id}` | {n_all} | **{arm['leaks']}** | "
            f"{arm['leak_not_checked']} | {arm['slot_mismatch']} | {arm['errors']} |")

    lines += [
        "",
        "One never-reused slot per (arm, cell), from the pinned 128-slot pool; "
        "intra-cell re-queries share that slot (same-document reuse, measured "
        "clean). A nonzero hit count invalidates the arm it lands in — a "
        "contaminated measurement of a contamination-adjacent property would be "
        "worthless. Per §10 R13 this is evidence, never a certificate: the 95% "
        "upper bound from 138 clean virgin-slot calls is 2.2%, not zero.",
        "",
        "## Every FALSE-POSITIVE and every MALFORMED, verbatim",
        "",
        "| arm | cell | question | trial | label | raw output |",
        "|---|---|---|---|---|---|",
    ]
    for rec in records:
        if rec.get("cell_uid") not in shared:
            continue
        if rec.get("label") not in (FALSE_POSITIVE, MALFORMED):
            continue
        raw = " ".join((rec.get("raw_output") or "").split())[:220]
        lines.append(
            f"| `{rec['arm']}` | {rec['cell_uid']} | {rec['question_type']} | "
            f"{rec['trial']} | {rec['label']} | `{raw}` |")

    lines += [
        "",
        "## A sample of every other outcome class, verbatim",
        "",
        "| arm | cell | question | label | raw output |",
        "|---|---|---|---|---|",
    ]
    seen: set[tuple] = set()
    for rec in records:
        key = (rec.get("arm"), rec.get("label"), rec.get("question_type"))
        if rec.get("cell_uid") not in shared:
            continue
        if rec.get("label") in (None, FALSE_POSITIVE, MALFORMED) or key in seen:
            continue
        seen.add(key)
        raw = " ".join((rec.get("raw_output") or "").split())[:220]
        lines.append(
            f"| `{rec['arm']}` | {rec['cell_uid']} | {rec['question_type']} | "
            f"{rec['label']} | `{raw}` |")
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


def _attach_chunks(records: list[dict], corpus: dict[str, str]) -> list[dict]:
    """Re-scoring an envelope arm needs the chunk (the span check is against
    it), and the JSONL deliberately does not carry a copy of the document on
    every one of its rows."""
    for rec in records:
        rec["_chunk"] = corpus.get(rec.get("cell_uid", ""))
    return records


async def _amain(args) -> int:
    dirs = [Path(d) for d in args.fixtures]
    cells, corpus = load_cells(dirs)

    if args.phase == "report":
        records = _attach_chunks(read_runs(args.out), corpus)
        REPORT_MD.write_text(regenerate(records), encoding="utf-8", newline="\n")
        print(f"wrote {REPORT_MD}")
        return 0

    from rlm.config import PromptRegistry, load_config
    from s2.leafcall import PinnedLeafCaller

    cfg = load_config(Path(args.config))
    index = ChunkIndex.from_chunks(corpus)
    arms = [a for a in ARMS if not args.arms or a[0] in args.arms]
    print(f"{len(cells)} cell(s) x {len(QUESTION_TYPES)} question types x "
          f"{args.trials} trial(s) x {len(arms)} arm(s) = "
          f"{len(cells) * len(QUESTION_TYPES) * args.trials * len(arms)} calls")

    # R13: slots are handed out once, across the WHOLE run -- never per arm and
    # never per cell. A counter that reset between arms would hand arm (b) the
    # slot arm (a) had just filled with the same document under a different
    # prefix, which is the reproducing condition exactly.
    state = {"next": args.first_slot}

    def next_slot() -> int:
        slot = state["next"]
        state["next"] += 1
        if slot >= cfg.servers.leaf.parallel:
            raise SystemExit(
                f"slot pool exhausted at {slot}: this run needs "
                f"{len(cells) * len(arms)} never-reused slots and the server was "
                f"launched with --parallel {cfg.servers.leaf.parallel} (R13)")
        return slot

    for arm_id, prefix_path, envelope in arms:
        registry = PromptRegistry.from_files(
            root_path=REPO_ROOT / "prompts/root.v3.md",
            leaf_prefix_path=REPO_ROOT / prefix_path,
            leaf_envelope_path=(REPO_ROOT / ENVELOPE_BLOCK) if envelope else None,
            strategy_paths={},
        ).load()
        caller = PinnedLeafCaller.from_config(
            cfg, system_prefix=registry.render_leaf(envelope=envelope),
            envelope=envelope)
        try:
            prefix_tokens = await caller.prepare()
            print(f"\n=== arm {arm_id}: prefix {prefix_path} "
                  f"{'+ envelope block' if envelope else ''} renders to "
                  f"{prefix_tokens} tokens; slot capacity "
                  f"{caller.slot_capacity_tokens} ===")
            await run_arm(caller, arm_id, cells, corpus, index,
                          next_slot=next_slot, trials=args.trials,
                          seeds=tuple(args.seeds), out_path=args.out)
        finally:
            await caller.aclose()

    if not args.no_report:
        records = _attach_chunks(read_runs(args.out), corpus)
        REPORT_MD.write_text(regenerate(records), encoding="utf-8", newline="\n")
        print(f"wrote {REPORT_MD}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="The S2 leaf refusal A/B (spec §5 leaf-envelope A/B, §10 R5)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="run", choices=("run", "report"))
    parser.add_argument("--fixtures", nargs="+",
                        default=[str(S2_DIR / "fixtures")],
                        help="one or more fixture directories; cells from all of "
                             "them are pooled, tagged by manifest seed")
    parser.add_argument("--out", type=Path, default=RUNS_PATH)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="arm ids to run (default: all four)")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRIAL_SEEDS))
    parser.add_argument("--first-slot", type=int, default=0,
                        help="first never-reused slot index (R13)")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
