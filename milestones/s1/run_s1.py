"""S1 gate runner (spec §9 S1, task-17 brief).

Runs the whole S1 protocol against the REAL servers and appends one JSON
record per attempt to `milestones/s1/results/runs.jsonl`, from which `--phase report`
regenerates `milestones/s1/RESULTS.md`. The phases are separable on purpose: an
interrupted gate must never cost the attempts that already ran, and
`runs.jsonl` -- not this process's memory -- is the record.

  control     arm (a): the root ALONE on the control-truncated document, three
              attempts, scored by the task's own checker. MUST score 0/3. A
              control that passes means the FIXTURE is broken (the needle
              survived the cut, or the model can guess the answer), not that
              the scaffold is good, and the run stops there.
  ab          arm (b) and the R1 prompt A/B in one pass: {root.v1, root.v2} x
              {needle, paraphrase} x 3 attempts, full scaffold, real leaf.
  leafprobe   a direct, scaffold-side leaf call on each fixture's needle-
              bearing chunk. Not an episode and not scored: it exists so that
              "the leaf path works" and "the root chose to use the leaf path"
              are separate findings rather than one confounded one.
  report      regenerate RESULTS.md from runs.jsonl + the trace store.

THREE ATTEMPTS MEANS THREE SEEDS. Sampling is temperature 0.7 / top_p 0.8 at
the root, and llama.cpp's continuous batching means even a fixed seed is not
bitwise reproducible (§8) -- but re-running one seed three times would still be
the wrong experiment. Each attempt overrides `scaffold.sampling.{root,leaf}.
seed` with 1, 2, 3 -- the §8 benchmark's own seed list -- and the seed is
recorded on every record.

The prompt A/B is CAPPED AT THE TWO PRE-AUTHORED VARIANTS (§9 S1, R1). This
runner cannot express a third: `VARIANTS` is the whole space, both files are
already in the registry, and nothing here edits prompt text.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `uv run milestones/s1/run_s1.py` from the repo root
    sys.path.insert(0, str(REPO_ROOT))

from rlm.cli import _lifecycle_path, _scaffold_git_sha, recover  # noqa: E402
from rlm.config import Config, load_config  # noqa: E402
from rlm.dispatcher import LLMDispatcher, ServerClient  # noqa: E402
from rlm.episode import Task, run_episode  # noqa: E402
from rlm.lifecycle import Lifecycle  # noqa: E402
from rlm.trace import TraceLogger  # noqa: E402

S1_DIR = Path(__file__).resolve().parent
TASKS_DIR = S1_DIR / "tasks"
RESULTS_DIR = S1_DIR / "results"
RUNS_PATH = RESULTS_DIR / "runs.jsonl"
RESULTS_MD = S1_DIR / "RESULTS.md"

FIXTURES = ("needle", "paraphrase")
VARIANTS = {"v1": "prompts/root.v1.md", "v2": "prompts/root.v2.md"}
ATTEMPT_SEEDS = (1, 2, 3)

# Arm (a)'s system prompt. Deliberately minimal and deliberately NOT the RLM
# root prompt: the control measures what a 32K-window model can do when the
# document is simply handed to it (§8's B1 shape), so a prompt describing a
# REPL it does not have would sandbag the control and flatter the scaffold.
CONTROL_SYSTEM = (
    "You are a careful analyst. Answer the question using only the document "
    "provided. If the document does not contain the answer, say NOT PRESENT."
)


# --------------------------------------------------------------------------- #
# fixtures, configs, records
# --------------------------------------------------------------------------- #


def load_fixture(name: str) -> dict:
    return json.loads((TASKS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def fixture_task(meta: dict) -> Task:
    """The Task the episode runs -- and, for arm (a), the object whose
    `.check()` scores the control. Both arms are scored by the same code."""
    return Task(task_id=meta["task_id"], text=meta["text"], context=meta["context"],
                category=meta["category"], answer=meta["answer"],
                checker=meta["checker"])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_config(raw_cfg: dict, *, variant: str | None, seed: int) -> Config:
    """A Config identical to the shipped one except for the root prompt under
    test and the attempt's seed. Built by patching the RAW dict and
    re-validating, so every cross-field rule in `rlm.config` still runs and the
    prompt's sha256 is re-pinned rather than dropped."""
    raw = copy.deepcopy(raw_cfg)
    if variant is not None:
        path = REPO_ROOT / VARIANTS[variant]
        raw["scaffold"]["prompts"]["root"] = {
            "path": str(Path(VARIANTS[variant])), "sha256": _sha256_file(path)}
    raw["scaffold"]["sampling"]["root"]["seed"] = seed
    raw["scaffold"]["sampling"]["leaf"]["seed"] = seed
    return Config.model_validate(raw)


def append_run(record: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with RUNS_PATH.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_runs() -> list[dict]:
    if not RUNS_PATH.exists():
        return []
    return [json.loads(line) for line in RUNS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --------------------------------------------------------------------------- #
# arm (a): the root alone, no scaffold
# --------------------------------------------------------------------------- #


async def control_attempt(cfg: Config, meta: dict, seed: int) -> dict:
    """One control attempt: system + (truncated document, then the question),
    rendered through the same /apply-template the scaffold uses, generated with
    the same n_predict and sampling, scored by the same checker."""
    doc = Path(meta["control"]["path"]).read_text(encoding="utf-8")
    task = fixture_task(meta)
    client = ServerClient(f"http://127.0.0.1:{cfg.servers.root.port}",
                           timeout=cfg.scaffold.retries.per_call_timeout_s)
    t0 = time.perf_counter()
    try:
        messages = [{"role": "system", "content": CONTROL_SYSTEM},
                    {"role": "user", "content": f"{doc}\n\n{meta['text']}"}]
        rendered = await client.apply_template(
            messages,
            chat_template_kwargs={"enable_thinking": cfg.scaffold.root.enable_thinking})
        result = await client.completion(
            rendered, n_predict=cfg.scaffold.budgets.max_predict.root,
            temperature=cfg.scaffold.sampling.root.temperature,
            top_p=cfg.scaffold.sampling.root.top_p, seed=seed, stream=True)
    finally:
        await client.aclose()
    elapsed = round(time.perf_counter() - t0, 1)
    answer = result.content.strip()
    return append_run({
        "phase": "control", "arm": "a", "fixture": meta["task_id"], "seed": seed,
        "attempt_id": f"control-{meta['task_id']}-seed{seed}",
        "passed": bool(task.check(answer)),
        "answer": answer[:2000],
        "control_tokens": meta["control"]["tokens"],
        "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
        "tokens_cached": result.cache_n, "stop_type": result.stop_type,
        "truncated": result.truncated,
        "prefill_ms": round(result.prompt_ms, 1),
        "decode_ms": round(result.predicted_ms, 1),
        "wall_clock_s": elapsed,
        "system_prompt_sha256": hashlib.sha256(
            CONTROL_SYSTEM.encode("utf-8")).hexdigest(),
    })


# --------------------------------------------------------------------------- #
# arm (b) + the R1 A/B: full scaffold episodes
# --------------------------------------------------------------------------- #


async def run_one_episode(cfg: Config, task: Task, lifecycle: Lifecycle):
    """`rlm.cli._run_one` minus the D21 bundle export (S3/S4 territory, and
    re-exporting per episode buys the gate nothing)."""
    dispatcher = LLMDispatcher.from_config(cfg)
    trace = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root, lifecycle=lifecycle)
    await trace.start()
    try:
        return await run_episode(
            task, cfg, dispatcher=dispatcher, trace=trace, lifecycle=lifecycle,
            scaffold_instance_id=f"{os.getpid()}",
            scaffold_git_sha=_scaffold_git_sha(),
            benchmark_version=cfg.benchmark.version)
    finally:
        await trace.aclose()
        await dispatcher.aclose()


def episode_stats(db_path: Path, episode_id: str) -> dict:
    """What actually happened, read back out of the trace store -- never out of
    this process's memory. If a number is not in the trace it does not go in
    the report."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ep = con.execute(
            "SELECT started_at, ended_at, outcome, outcome_reason FROM episodes "
            "WHERE episode_id = ?", [episode_id]).fetchone()
        rows = con.execute(
            "SELECT step_idx, parent_step_idx, call_id, retry_idx, actor, "
            "action_type, status, error_detail, tokens_in, tokens_out, "
            "tokens_cached, slot_id, latency_prefill_ms, latency_decode_ms, "
            "action_payload, observation_view FROM steps WHERE episode_id = ? "
            "ORDER BY step_idx", [episode_id]).fetchall()
    finally:
        con.close()
    cols = ("step_idx", "parent_step_idx", "call_id", "retry_idx", "actor",
            "action_type", "status", "error_detail", "tokens_in", "tokens_out",
            "tokens_cached", "slot_id", "latency_prefill_ms", "latency_decode_ms",
            "action_payload", "observation_view")
    steps = [dict(zip(cols, r)) for r in rows]
    leaf = [s for s in steps if s["action_type"] == "llm_call"]
    leaf_ok = [s for s in leaf if s["status"] == "ok"]
    turns = [s for s in steps if s["action_type"] == "repl_exec"]
    final = [s for s in steps if s["action_type"] == "final"]
    return {
        "started_at": str(ep[0]) if ep else None,
        "ended_at": str(ep[1]) if ep else None,
        "db_outcome": str(ep[2]) if ep else None,
        "db_outcome_reason": ep[3] if ep else None,
        "root_turns": len(turns),
        "turn_statuses": [s["status"] for s in turns],
        "leaf_calls": len({s["call_id"] for s in leaf if s["call_id"]}),
        "leaf_attempts": len(leaf),
        "leaf_errors": [s["error_detail"] for s in leaf if s["status"] != "ok"],
        "leaf_tokens_out": [s["tokens_out"] for s in leaf_ok if s["tokens_out"] is not None],
        "leaf_tokens_cached": [s["tokens_cached"] for s in leaf_ok],
        "root_tokens_in": [s["tokens_in"] for s in turns],
        "root_tokens_out": [s["tokens_out"] for s in turns],
        "final_payload": (final[0]["action_payload"] if final else None),
        "transcript": [
            {"step_idx": s["step_idx"], "action": s["action_type"],
             "status": s["status"], "error_detail": s["error_detail"],
             "payload_head": (s["action_payload"] or "")[:700],
             "observation_head": (s["observation_view"] or "")[:700]}
            for s in steps
        ],
    }


async def rlm_attempt(raw_cfg: dict, meta: dict, variant: str, seed: int,
                       lifecycle: Lifecycle) -> dict:
    cfg = variant_config(raw_cfg, variant=variant, seed=seed)
    task = fixture_task(meta)
    t0 = time.perf_counter()
    error: str | None = None
    try:
        result = await run_one_episode(cfg, task, lifecycle)
        episode_id, outcome, reason = (result.episode_id, str(result.outcome),
                                        result.reason)
        final_answer = result.final_answer
    except Exception as exc:  # noqa: BLE001 -- an attempt that blew up IS a result
        episode_id, outcome, reason = "", "runner_exception", repr(exc)
        final_answer = None
        error = repr(exc)
    elapsed = round(time.perf_counter() - t0, 1)
    stats = (episode_stats(Path(cfg.trace.db_path), episode_id)
             if episode_id else {})
    return append_run({
        "phase": "ab", "arm": "b", "fixture": meta["task_id"], "variant": variant,
        # `variant_config` already accepts None to mean "leave the SHIPPED root
        # prompt alone"; only this record line assumed a variant, so an episode
        # against the pinned prompt died on KeyError(None). Recording the config
        # path matters more than the variant name here: after the §9 S1 A/B was
        # closed and root.v3 pinned, a re-run should exercise what ships, not
        # re-open a decided comparison.
        "root_prompt": (VARIANTS[variant] if variant is not None
                        else str(cfg.scaffold.prompts.root.path)),
        "seed": seed,
        "episode_id": episode_id, "outcome": outcome, "outcome_reason": reason,
        "passed": outcome == "success",
        "final_answer": (str(final_answer)[:2000] if final_answer is not None else None),
        "expected_answer": meta["answer"],
        "wall_clock_s": elapsed, "runner_error": error,
        **stats,
    })


# --------------------------------------------------------------------------- #
# leaf probe: is the leaf path working, independent of whether the root uses it
# --------------------------------------------------------------------------- #


async def leaf_probe(cfg: Config, meta: dict) -> dict:
    """One direct leaf call, composed exactly as the root prompt tells the root
    to compose one (chunk text verbatim and first, question last), against the
    needle-bearing slice of the fixture. Scaffold-side, one call, not scored
    into any arm."""
    text = Path(meta["context_path"]).read_text(encoding="utf-8")
    offset = meta["needle_char_offset"]
    # A PRODUCTION-SIZED window around the needle, not a 120,000-char slab.
    #
    # This line used to take text[offset-60_000 : offset+60_000] and call it
    # "capped well under one slot". That was true when the leaf ran at `-np 8`
    # (327,680 / 8 = 40,960 tokens per slot). R13's mitigation moved the leaf to
    # `-np 128` -- 2,560 tokens per slot -- and the comment silently became
    # false: ~31,800 tokens against a 2,560-token slot, which C4's pre-flight
    # rejects before any call is made (status=rejected in 0.1 s, measured
    # 2026-08-15). It went unnoticed because the leaf path was broken for other
    # reasons (F3, no chat template), so the probe never got far enough to hit
    # the cap.
    #
    # The window is now sized from config -- `scaffold.chunk.size_tokens` at the
    # measured 3.7727 chars/token (§8) -- so the probe sends what production
    # actually sends a leaf, which is what "does one direct leaf call work"
    # should have meant all along.
    half = int(cfg.scaffold.chunk.size_tokens * 3.7727 / 2)
    chunk = text[max(0, offset - half):offset + half]
    question = f"{meta['text']}\nIf the document does not state it, reply NONE."
    dispatcher = LLMDispatcher.from_config(cfg)
    t0 = time.perf_counter()
    try:
        answer = await dispatcher.query(f"{chunk}\n\n{question}", role="leaf",
                                         call_id="s1-leaf-probe")
        step = dispatcher.last_step or {}
        err = None
    except Exception as exc:  # noqa: BLE001
        answer, step, err = "", (dispatcher.last_step or {}), repr(exc)
    finally:
        await dispatcher.aclose()
    return append_run({
        "phase": "leafprobe", "fixture": meta["task_id"],
        "answer": answer[:1000], "expected_answer": meta["answer"],
        "contains_expected": meta["answer"].casefold() in answer.casefold(),
        "tokens_in": step.get("tokens_in"), "tokens_out": step.get("tokens_out"),
        "tokens_cached": step.get("tokens_cached"), "status": step.get("status"),
        "error": err, "wall_clock_s": round(time.perf_counter() - t0, 1),
    })


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def derive_max_predict(runs: list[dict]) -> dict:
    """Re-derive the leaf `max_predict` default from OBSERVED answer lengths
    (§9 S1: "S1 traces also re-derive the leaf max_predict default").

    Reported with its own n, its max, and the count of answers that hit the
    configured ceiling -- a distribution of three samples is a finding about
    how little the root used the leaf, not a calibrated default, and the
    report has to say which one it is.
    """
    lens: list[int] = []
    for run in runs:
        lens.extend(int(n) for n in run.get("leaf_tokens_out") or [] if n is not None)
        if run.get("phase") == "leafprobe" and run.get("tokens_out"):
            lens.append(int(run["tokens_out"]))
    if not lens:
        return {"n": 0}
    lens.sort()

    def pct(p: float) -> int:
        return lens[min(len(lens) - 1, max(0, int(round(p * (len(lens) - 1)))))]

    return {"n": len(lens), "min": lens[0], "median": int(statistics.median(lens)),
            "p90": pct(0.90), "p95": pct(0.95), "max": lens[-1],
            "mean": round(statistics.fmean(lens), 1), "all": lens}


def _score(runs: list[dict], **match: Any) -> tuple[int, int, list[dict]]:
    sel = [r for r in runs if all(r.get(k) == v for k, v in match.items())]
    return sum(1 for r in sel if r.get("passed")), len(sel), sel


def render_report(runs: list[dict]) -> str:
    """RESULTS.md, generated from the record rather than typed from memory.

    The verdict rule is stated BEFORE the numbers, and applied mechanically:
    arm (a) must be 0/3, and the RLM arm is scored on the A/B WINNER (the
    prompt that gets pinned), never on whichever variant happened to do best
    per task. Both variants' full scores are printed either way.
    """
    lines: list[str] = []
    control = [r for r in runs if r.get("phase") == "control"]
    ab = [r for r in runs if r.get("phase") == "ab"]
    probes = [r for r in runs if r.get("phase") == "leafprobe"]

    needle_control_pass, needle_control_n, _ = _score(
        control, fixture="s1-needle")
    ab_scores = {
        (v, f): _score(ab, variant=v, fixture=f)[:2]
        for v in VARIANTS for f in ("s1-needle", "s1-paraphrase")
    }
    totals = {v: (sum(ab_scores[(v, f)][0] for f in ("s1-needle", "s1-paraphrase")),
                  sum(ab_scores[(v, f)][1] for f in ("s1-needle", "s1-paraphrase")))
              for v in VARIANTS}
    # Winner: total passes across both S1 fixtures; ties break to the SIMPLER
    # prompt (v1, tips-only) because a tie is not evidence for the exemplars.
    winner = max(VARIANTS, key=lambda v: (totals[v][0], v == "v1"))

    # Arm (b) is scored on the PINNED prompt when episodes against it exist.
    # The rule below says "the variant that WINS the R1 A/B and is therefore
    # the one pinned into config.yaml" -- those were the same thing while the
    # A/B was open. It closed, root.v3 was pinned, and they stopped being the
    # same thing: episodes against the shipped prompt carry `variant: None`,
    # which the v1/v2 lookup scored as 0/0 and reported as a gate FAIL while
    # every episode had in fact passed. Scoring the pinned prompt is what the
    # rule asks for; falling back to the A/B winner keeps the historical
    # records (which have no pinned-prompt rows) scoring exactly as before.
    pinned_needle = _score(ab, variant=None, fixture="s1-needle")[:2]
    scored_on_pinned = pinned_needle[1] > 0
    win_needle = pinned_needle if scored_on_pinned else ab_scores[(winner, "s1-needle")]
    gate_a = needle_control_n > 0 and needle_control_pass == 0
    gate_b = win_needle[1] > 0 and win_needle[0] >= 2
    verdict = "PASS" if (gate_a and gate_b) else "FAIL"

    lines += [
        "# S1 — Minimal loop: gate results",
        "",
        f"*Generated by `milestones/s1/run_s1.py --phase report` from `milestones/s1/results/runs.jsonl` "
        f"({len(runs)} records) on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.*",
        "",
        f"## S1 GATE: {verdict}",
        "",
        "Verdict rule, stated before the numbers (spec §9 S1): arm (a) control must "
        "score **0/3**, arm (b) RLM must score **>= 2/3** on the needle fixture with "
        "the prompt variant that WINS the R1 A/B and is therefore the one pinned into "
        "`config.yaml`. Scoring the gate on whichever variant did best per task would "
        "be selection after the fact; both variants' scores are printed in full below.",
        "",
        f"- arm (a) control, root alone on the <=28K-token truncation: "
        f"**{needle_control_pass}/{needle_control_n}** "
        f"({'PASS — 0/3 as required' if gate_a else 'GATE VIOLATION'})",
        (f"- arm (b) RLM, full scaffold, **pinned prompt** "
         f"(`{'?' if not ab else next((r.get('root_prompt') for r in ab if r.get('variant') is None), '?')}`): "
         f"**{win_needle[0]}/{win_needle[1]}** "
         f"({'PASS' if gate_b else 'FAIL'})"
         if scored_on_pinned else
         f"- arm (b) RLM, full scaffold, winner `{winner}` "
         f"(`{VARIANTS[winner]}`): **{win_needle[0]}/{win_needle[1]}** "
         f"({'PASS' if gate_b else 'FAIL'})"),
        "",
    ]

    lines += ["## Arm (a) — control: root alone, document truncated", "",
              "System prompt (verbatim):", "", "```", CONTROL_SYSTEM, "```", "",
              "| attempt_id | fixture | seed | doc tokens | passed | tokens_in |"
              " tokens_out | wall s | answer (head) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in control:
        ans = " ".join((r.get("answer") or "").split())[:110]
        lines.append(
            f"| `{r['attempt_id']}` | {r['fixture']} | {r['seed']} | "
            f"{r.get('control_tokens')} | {'PASS' if r['passed'] else 'fail'} | "
            f"{r.get('tokens_in')} | {r.get('tokens_out')} | {r.get('wall_clock_s')} | "
            f"{ans} |")
    lines.append("")

    lines += ["## Arm (b) + R1 prompt A/B — full scaffold", "",
              "| episode_id | fixture | variant | seed | outcome | reason | turns |"
              " leaf calls | wall s | final answer (head) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ab:
        ans = " ".join((r.get("final_answer") or "")[:80].split())
        lines.append(
            f"| `{r.get('episode_id') or '(no episode)'}` | {r['fixture']} | "
            f"{r['variant']} | {r['seed']} | {r.get('outcome')} | "
            f"{r.get('outcome_reason') or ''} | {r.get('root_turns')} | "
            f"{r.get('leaf_calls')} | {r.get('wall_clock_s')} | {ans} |")
    lines += ["", "### A/B scores", "",
              "| variant | needle | paraphrase | total |", "|---|---|---|---|"]
    for v in VARIANTS:
        n, p = ab_scores[(v, "s1-needle")], ab_scores[(v, "s1-paraphrase")]
        lines.append(f"| `{v}` ({VARIANTS[v]}) | {n[0]}/{n[1]} | {p[0]}/{p[1]} | "
                     f"{totals[v][0]}/{totals[v][1]} |")
    margin = totals[winner][0] - totals[min(VARIANTS, key=lambda v: (totals[v][0], v == "v1"))][0]
    lines += ["", f"**Winner: `{winner}` ({VARIANTS[winner]}), margin "
                  f"{margin} episode(s) over the other variant across both fixtures.**",
              ""]

    if probes:
        lines += ["## Leaf-path probe (not an arm, not scored)", "",
                  "| fixture | status | tokens_in | tokens_out | contains expected |"
                  " wall s | answer (head) |", "|---|---|---|---|---|---|---|"]
        for r in probes:
            ans = " ".join((r.get("answer") or "").split())[:90]
            lines.append(
                f"| {r['fixture']} | {r.get('status')} | {r.get('tokens_in')} | "
                f"{r.get('tokens_out')} | {r.get('contains_expected')} | "
                f"{r.get('wall_clock_s')} | {ans} |")
        lines.append("")

    mp = derive_max_predict(runs)
    lines += ["## Re-derived leaf `max_predict` (C5)", ""]
    if mp["n"] == 0:
        lines += ["No leaf answer lengths were observed at all — the re-derivation "
                  "has no data, which is itself the finding. `max_predict.leaf` "
                  "stays at its configured value.", ""]
    else:
        lines += [
            f"Observed leaf answer lengths (`steps.tokens_out` over every "
            f"`llm_call` with `status=ok`, plus the leaf probes): n={mp['n']}, "
            f"min {mp['min']}, median {mp['median']}, p90 {mp['p90']}, p95 "
            f"{mp['p95']}, max {mp['max']}, mean {mp['mean']}.",
            "", f"Raw: `{mp['all']}`", ""]

    lines += ["## Per-episode wall clock", "",
              "| phase | fixture | variant | seed | wall s |", "|---|---|---|---|---|"]
    for r in runs:
        if r.get("phase") in ("control", "ab"):
            lines.append(f"| {r['phase']} | {r.get('fixture')} | "
                         f"{r.get('variant', '-')} | {r.get('seed')} | "
                         f"{r.get('wall_clock_s')} |")
    lines.append("")
    return "\n".join(lines)


NARRATIVE_MARKER = "<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->"


def regenerate(runs: list[dict]) -> str:
    """Rebuild the generated half of RESULTS.md, preserving the hand-written
    half. The tables and the verdict come from the record; the findings — what
    the scaffold got wrong against real models, and what that means — cannot be
    generated and must not be destroyed by re-running `--phase report`."""
    generated = render_report(runs)
    narrative = ""
    if RESULTS_MD.exists():
        old = RESULTS_MD.read_text(encoding="utf-8")
        if NARRATIVE_MARKER in old:
            narrative = old.split(NARRATIVE_MARKER, 1)[1]
    return f"{generated}\n{NARRATIVE_MARKER}\n{narrative.lstrip(chr(10))}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


async def _amain(args) -> int:
    cfg = load_config(Path(args.config))
    raw_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    lifecycle = Lifecycle(_lifecycle_path(cfg, None))
    fixtures = {name: load_fixture(name) for name in FIXTURES}
    try:
        tombstoned = recover(cfg, lifecycle)
        if tombstoned:
            print(f"recovery: tombstoned {len(tombstoned)} orphaned episode(s)")

        if args.phase in ("control", "all"):
            for name in args.fixtures:
                meta = fixtures[name]
                for seed in args.seeds:
                    rec = await control_attempt(
                        variant_config(raw_cfg, variant=None, seed=seed), meta, seed)
                    print(f"[control] {rec['attempt_id']}: "
                          f"{'PASS' if rec['passed'] else 'fail'} "
                          f"({rec['wall_clock_s']}s, {rec['tokens_out']} tok out) "
                          f"{rec['answer'][:120]!r}")
            control_needle = _score(
                [r for r in read_runs() if r.get("phase") == "control"],
                fixture="s1-needle")
            if control_needle[0]:
                print("CONTROL PASSED — the fixture is broken, not the scaffold. "
                      "Stopping before arm (b) (spec §9 S1).", file=sys.stderr)
                return 3

        if args.phase in ("leafprobe", "all"):
            for name in args.fixtures:
                rec = await leaf_probe(variant_config(raw_cfg, variant=None, seed=1),
                                        fixtures[name])
                print(f"[leafprobe] {rec['fixture']}: status={rec['status']} "
                      f"contains_expected={rec['contains_expected']} "
                      f"({rec['wall_clock_s']}s) {rec['answer'][:160]!r}")

        if args.phase in ("ab", "all"):
            for variant in args.variants:
                for name in args.fixtures:
                    for seed in args.seeds:
                        rec = await rlm_attempt(raw_cfg, fixtures[name], variant,
                                                 seed, lifecycle)
                        print(f"[ab] {name}/{variant}/seed{seed}: "
                              f"{rec['outcome']} ({rec.get('outcome_reason')}) "
                              f"turns={rec.get('root_turns')} "
                              f"leaf={rec.get('leaf_calls')} "
                              f"{rec['wall_clock_s']}s "
                              f"answer={str(rec.get('final_answer'))[:120]!r}")

        if args.phase in ("report", "all"):
            RESULTS_MD.write_text(regenerate(read_runs()), encoding="utf-8",
                                   newline="\n")
            print(f"wrote {RESULTS_MD}")
        return 0
    finally:
        lifecycle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the S1 gate (spec §9 S1)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", default="all",
                        choices=("control", "leafprobe", "ab", "report", "all"))
    parser.add_argument("--fixtures", nargs="+", default=list(FIXTURES),
                        choices=list(FIXTURES))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ATTEMPT_SEEDS))
    args = parser.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
