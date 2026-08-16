"""§8's baseline arms — the controls RLM is measured against (ARCHITECTURE.md §8).

Three arms live here (B1 now; B3 and B2 land beside it), and they exist so the
S4 verdict can say what the scaffold is worth **relative to something**:

  * **B1** — the leaf model, single shot, its full native window. "Does raw long
    context beat the scaffold?"
  * **B3** — deterministic BM25-RAG single shot over the C2 chunker's windows.
  * **B2** — deterministic map-reduce: chunk, summarise, reduce. The honest
    control; if it matches RLM, the root's agency is not earning its complexity.

**EVERY ARM LOGS A REAL EPISODE.** An arm that reported a score without a trace
row would be unscoreable under I4 ("a run that is not logged did not happen")
and unauditable under §8, whose entire comparison rests on the per-task ×
per-arm × per-seed grid coming out of one store. So each arm opens an episode,
writes one `llm_call` step per model call, and closes with §6's outcome
semantics:

    answer + checker pass      -> success
    answer + checker fail      -> fail / checker_failed
    wall-clock breach          -> budget_kill / wall_clock
    dispatcher/server failure  -> error / server_unreachable

**THE STEPS COME FROM C4, NOT FROM HERE.** Every model call is routed through
the injected dispatcher, so the R13 leak columns, the retry attempts and the
timing/token fields are recorded by the one component that measures them. This
module only copies C4's attempt dicts onto trace rows (the pattern
`rlm/episode.py:720-757` owns), which is why an arm's steps are comparable with
the RLM arm's rather than merely similar to them.

**KILL GRANULARITY IS ONE MODEL CALL.** C5's wall clock is checked immediately
before and immediately after each dispatch, and nowhere else. The RLM arm needs
a 0.1 s watchdog because a sandbox cell can loop forever and never return to the
turn loop; a baseline arm has no sandbox and no cell — it makes between one and
a few hundred calls and does nothing else — so the coarser bound is the honest
one, and it is stated here rather than discovered from the code. A breach is
therefore detected at most one model call late (bounded by
`scaffold.retries.per_call_timeout_s` × `max_attempts`).

**THIS MODULE NEVER IMPORTS C4.** `rlm/episode.py` is the composition root and
the only module permitted to import both C4 and the isolated components; arms
stay on the isolated side by taking the dispatcher (and the trace logger, and
the registry) as injected arguments, with `Task` imported only for typing.
`tests/test_import_rules.py` lints it.

**B1'S OVERFLOW POLICY IS PRE-REGISTERED (§8):** a task whose tokenized corpus
exceeds the window is head+tail truncated to fit, 50/50, and the truncation is
recorded in `config_snapshot["bench"]["b1_truncation"]` so the B1-infeasible
subset is identifiable in the results rather than silently scored as if it had
been readable. Two rulings inside that policy, written down because either could
otherwise be re-decided after seeing results:

  * The kept head and tail are joined with `rlm.context.DOCUMENT_SEPARATOR`
    ("\\n\\n"), NOT an elision banner. C2 already refuses per-document banners on
    the ground that scaffold text the model cannot distinguish from corpus text
    is exactly what §8's adversarial tasks turn on; an elision marker is the
    same object, and "tell the model it was truncated" is a prompt-side
    intervention B1 was pre-registered without.
  * The split is char-proportional, and then VERIFIED once against the real
    tokenizer (`run_b1`'s fit loop). Char-proportionality is an estimate: if the
    tail half is denser than the head, a prompt sized on it overruns the slot
    and C4's pre-flight REJECTS the call — which would land in the results as
    `error/server_unreachable`, i.e. as a server fault, for what is really an
    arithmetic slip. The loop costs at most three `/tokenize` round trips and
    removes that failure mode.
"""
from __future__ import annotations

import inspect
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rlm import trace as tracemod
from rlm.budget import BudgetEnforcer
from rlm.budget import Budgets as BudgetLimits
from rlm.config import Config, PromptRegistry, config_snapshot
from rlm.context import DOCUMENT_SEPARATOR, load_context
from rlm.errors import (
    ActionType,
    Actor,
    BudgetBreach,
    ConfigError,
    DispatchError,
    Outcome,
    StepStatus,
)

if TYPE_CHECKING:                       # pragma: no cover - typing only
    # Import-time isolation: `rlm.episode` reaches C4, and an arm that pulled it
    # in would put an HTTP client behind a module the trace/analysis side must
    # be able to import. `Task` is used structurally (text/context/check), so
    # the type is all that is needed.
    from rlm.episode import Task

__all__ = [
    "ARM_ERROR",
    "CHECKER_FAILED",
    "FIT_SLACK_TOKENS",
    "MAX_FIT_VERIFY_ROUNDS",
    "NO_ANSWER",
    "SERVER_UNREACHABLE",
    "SNAPSHOT_KEYS",
    "ArmEpisode",
    "ArmResult",
    "bench_slot_capacity",
    "outcome_for_error",
    "run_b1",
    "truncate_head_tail",
]

# --- §6 outcome_reason vocabulary ------------------------------------------ #
# DUPLICATED FROM `rlm/episode.py` ON PURPOSE: importing it from there would
# drag C4 into this module (see the docstring). `tests/test_arms.py` pins the
# two definitions equal, so the vocabulary cannot drift in silence.
CHECKER_FAILED = "checker_failed"
SERVER_UNREACHABLE = "server_unreachable"

#: A call completed but produced nothing to score. Deliberately NOT
#: `no_final_emitted`: a baseline arm has no `final_answer` channel to emit on —
#: the reply IS the answer — so borrowing the RLM arm's reason would attribute
#: the failure to a mechanism this arm does not have.
NO_ANSWER = "no_answer"

#: The arm itself failed in a way §6 has no vocabulary for (a bug, an unexpected
#: exception). The episode is closed rather than orphaned — a NULL-outcome row
#: tombstones as `orphaned_at_recovery` at the next startup and loses the
#: attribution — and the exception is re-raised, so it stays loud.
ARM_ERROR = "arm_error"

#: Slack between the fitted prompt and the slot's true capacity, in tokens. It
#: absorbs the two separators this module inserts and the chat template's markup,
#: neither of which a raw text token count sees.
FIT_SLACK_TOKENS = 64

#: Bound on `run_b1`'s fit verify loop. Each round costs one `/tokenize` of the
#: whole prompt; three is far more than a char-density skew can need, and the
#: bound is what makes the loop's termination structural rather than hopeful.
MAX_FIT_VERIFY_ROUNDS = 3

#: The top-level `config_snapshot` keys this module adds. `config_snapshot`
#: merges `extra` OVER the config dump, so any collision here would silently
#: replace a whole config section; `tests/test_arms.py` pins them disjoint from
#: `Config`'s own fields.
SNAPSHOT_KEYS = ("prompt_hashes", "pinned_prompt_hashes", "task", "bench")


@dataclass(slots=True)
class ArmResult:
    """One baseline episode's verdict — the same four facts `EpisodeResult`
    carries for the RLM arm, so §8's grid is built from one shape."""

    episode_id: str
    outcome: Outcome
    reason: str | None
    answer: str | None


# --------------------------------------------------------------------------- #
# §8's pre-registered B1 overflow policy
# --------------------------------------------------------------------------- #


def truncate_head_tail(corpus: str, *, corpus_tokens: int,
                        fit_tokens: int) -> tuple[str, dict]:
    """§8's head+tail overflow policy: keep the first and last halves of
    `fit_tokens`, drop the middle. Pure and deterministic.

    Returns `(text, record)`. The record is what lands in the episode's
    `config_snapshot` and therefore what makes the B1-infeasible subset
    identifiable at scoring time:

        {"truncated": bool, "kept_head_tokens": int,
         "kept_tail_tokens": int, "corpus_tokens": int}

    When nothing was dropped the whole corpus IS the kept head, so
    `kept_head_tokens == corpus_tokens` and the tail is empty — the record
    stays readable as "what survived" in both cases, and `truncated` is the
    field a scoring query filters on.

    The split is char-proportional (`len(corpus) / corpus_tokens` chars per
    token). It is an estimate by construction — token density varies across a
    document — which is why `run_b1` verifies the assembled prompt against the
    real tokenizer instead of trusting this number.
    """
    if corpus_tokens <= 0 or (fit_tokens >= corpus_tokens and fit_tokens > 0):
        # Nothing to drop: an empty corpus, or one that already fits.
        return corpus, {"truncated": False,
                        "kept_head_tokens": max(corpus_tokens, 0),
                        "kept_tail_tokens": 0,
                        "corpus_tokens": corpus_tokens}
    if fit_tokens <= 0:
        # No room for corpus at all (the head and the question already fill the
        # slot). Recorded as a truncation of everything rather than refused:
        # the task is B1-infeasible, and that is a RESULT §8 wants in the grid.
        return "", {"truncated": True, "kept_head_tokens": 0,
                    "kept_tail_tokens": 0, "corpus_tokens": corpus_tokens}

    head_tokens = fit_tokens // 2
    tail_tokens = fit_tokens - head_tokens
    chars_per_token = len(corpus) / corpus_tokens
    head_chars = min(round(head_tokens * chars_per_token), len(corpus))
    tail_chars = min(round(tail_tokens * chars_per_token), len(corpus) - head_chars)
    head = corpus[:head_chars]
    tail = corpus[len(corpus) - tail_chars:] if tail_chars else ""
    return (f"{head}{DOCUMENT_SEPARATOR}{tail}" if tail else head,
            {"truncated": True, "kept_head_tokens": head_tokens,
             "kept_tail_tokens": tail_tokens, "corpus_tokens": corpus_tokens})


def bench_slot_capacity(cfg: Config) -> int:
    """Tokens one slot of §8's B1/B3 relaunch profile holds.

    Refused loudly on a config with no `servers.bench_leaf`: the single-shot
    arms are DEFINED by that profile (`--parallel 2 -c 524288`, one slot each,
    per §8's v0.2.6 correction), and falling back to `servers.leaf` would run
    them in the 2,560-token slot R13's pool arithmetic gives the RLM arm — i.e.
    it would quietly measure a different baseline than the one pre-registered.
    """
    bench = cfg.servers.bench_leaf
    if bench is None:
        raise ConfigError(
            "servers.bench_leaf is not configured; §8's B1/B3 single-shot arms "
            "run on that relaunch profile and cannot borrow servers.leaf's "
            "slot geometry (ARCHITECTURE.md §8)")
    return bench.ctx // bench.parallel


# --------------------------------------------------------------------------- #
# shared arm plumbing
# --------------------------------------------------------------------------- #


def outcome_for_error(exc: BaseException) -> tuple[Outcome, str]:
    """§6's attribution for a failed arm, in the one place it is decided.

    A `BudgetBreach` already carries the outcome C5 decided, and it wins as-is:
    an arm killed for `wall_clock` whose dispatcher then reports a failure is
    `budget_kill`, not `error`. Everything from C4 is `error/server_unreachable`
    — a baseline arm has no route around a leaf that will not answer.
    """
    if isinstance(exc, BudgetBreach):
        return exc.outcome, exc.reason
    if isinstance(exc, DispatchError):
        return Outcome.ERROR, SERVER_UNREACHABLE
    return Outcome.ERROR, ARM_ERROR


def _accepts_slot_id(query: Any) -> bool:
    """Whether this dispatcher's `query` DECLARES a `slot_id` parameter.

    Deliberately not satisfied by `**kwargs`: a dispatcher that swallows an
    unknown keyword would report a pin it never applied, and the whole point of
    the pin is that B1 and B3 stop sharing slot 0 (R13's smallest repro).
    """
    try:
        params = inspect.signature(query).parameters
    except (TypeError, ValueError):          # builtins, C callables
        return False
    return "slot_id" in params


def _answer_text(raw: Any) -> str:
    """C4 returns the answer STRING, or the parsed envelope dict when the target
    asks for one. Baselines score text, so the envelope's `answer` field is the
    answer; the envelope is off by default and this costs two lines."""
    if isinstance(raw, dict):
        return str(raw.get("answer", ""))
    return raw if isinstance(raw, str) else str(raw)


class ArmEpisode:
    """One baseline episode's trace + budget plumbing, shared by B1/B2/B3.

    Built as an object rather than a set of free functions because the four
    things an arm must not get wrong — the episode id, the step counter, the
    already-logged attempt set, and the enforcer — have to be the SAME four
    across every call the arm makes, and passing them around individually is
    how one of them gets forgotten on the failure path.

    Not a context manager: `close()` must be callable from an `except`
    (including a cancellation unwind) and is therefore synchronous — every
    `TraceLogger` method it uses is a queue put, never I/O. The caller drains.
    """

    def __init__(self, task: "Task", cfg: Config, *, dispatcher: Any,
                 trace: Any, registry: PromptRegistry, arm: str,
                 bench_extra: dict[str, Any] | None) -> None:
        self.task = task
        self.cfg = cfg
        self.dispatcher = dispatcher
        self.trace = trace
        self.registry = registry
        self.arm = arm
        self.bench_extra = dict(bench_extra or {})
        # REAL uuid4: the DuckDB column is UUID-typed and the writer loop
        # swallows conversion failures, so a malformed id loses every write for
        # this episode in silence.
        self.episode_id = str(uuid.uuid4())
        self.enforcer = BudgetEnforcer(
            BudgetLimits(
                max_depth=cfg.scaffold.budgets.max_depth,
                max_subcalls=cfg.scaffold.budgets.max_subcalls,
                max_wall_clock_s=cfg.scaffold.budgets.max_wall_clock_s,
                max_total_tokens=cfg.scaffold.budgets.max_total_tokens,
                max_predict={"root": cfg.scaffold.budgets.max_predict.root,
                             "leaf": cfg.scaffold.budgets.max_predict.leaf},
            ),
            root_window_kill_fraction=cfg.scaffold.root_window_kill_fraction,
        )
        self._next_idx = 0
        self._logged_attempts: set[tuple[str, int]] = set()
        #: The blob holding the answer, reused as `episodes.final_answer_ref`
        #: rather than written twice (it is the same bytes).
        self.answer_ref: str | None = None
        self._closed: ArmResult | None = None
        self._slot_kwarg: bool | None = None

    # -- C5 wall clock ------------------------------------------------------- #

    def start_clock(self) -> None:
        """Start C5's clock. Called BEFORE the corpus is loaded and counted:
        `/tokenize` on a multi-megabyte corpus is real wall time the episode
        spent, and untimed it would be work the budget cannot see."""
        self.enforcer.start_clock()

    def check_wall_clock(self) -> None:
        self.enforcer.check_wall_clock()

    # -- C6 ------------------------------------------------------------------ #

    def snapshot(self, arm_snapshot: dict[str, Any] | None = None) -> dict:
        """§6 `config_snapshot`: the validated config plus what a replay or a
        scoring query needs and the schema has no column for.

        Shaped to match `_EpisodeRun._snapshot` key for key, so one query reads
        prompt hashes and task identity across all four arms. `bench` is the
        arm's own dict: the identity Task 9 supplies (arm/run_id/seed/block)
        plus whatever the arm records about how it ran (B1's truncation).
        """
        bench = {"arm": self.arm, **self.bench_extra, **(arm_snapshot or {})}
        extra = {
            # The bytes each arm ACTUALLY ran. §8 pre-registers the baseline
            # prompts and forbids iterating them against benchmark content; this
            # is what makes that auditable after the fact rather than asserted.
            "prompt_hashes": self.registry.hashes(),
            "pinned_prompt_hashes": self.cfg.pinned_prompt_hashes(),
            "task": {
                "task_id": self.task.task_id,
                "text": self.task.text,
                "category": self.task.category,
                "checker": self.task.checker,
                "answer": self.task.answer,
            },
            "bench": bench,
        }
        return config_snapshot(self.cfg, extra)

    def open_episode(self, *, arm_snapshot: dict[str, Any] | None = None,
                      tokenized_task_len: int | None = None) -> str:
        """Write the episode row, with a NULL outcome (what makes tombstoning
        possible) and a NULL `sandbox_pid` (a baseline arm runs no sandbox, so
        there is no pid for §6 recovery to reap)."""
        self.trace.open_episode({
            "episode_id": self.episode_id,
            "task_id": self.task.task_id,
            "task_hash": self.task.task_hash,
            "tokenized_task_len": tokenized_task_len,
            "dry_run": self.cfg.scaffold.dispatcher == "mock",
            "sandbox_pid": None,
            "config_snapshot": self.snapshot(arm_snapshot),
            "benchmark_version": self.cfg.benchmark.version,
        })
        return self.episode_id

    def alloc_step_idx(self) -> int:
        idx = self._next_idx
        self._next_idx += 1
        return idx

    def put_step(self, row: dict[str, Any],
                  blobs: dict[str, bytes] | None = None) -> None:
        row.setdefault("episode_id", self.episode_id)
        self.trace.put_step(row, blobs)

    def log_call(self, call_id: str, prompt: str, *, answer: str | None = None,
                  parent: int | None = None, actor: str = Actor.LEAF,
                  depth: int = 1) -> str | None:
        """One trace step per dispatch ATTEMPT, copied from C4's own records.

        Idempotent per `(call_id, retry_idx)` so the failure path and a later
        flush cannot double-write the same attempt (`rlm/episode.py:720-757`).
        Returns the blob path holding the answer, or None when no attempt
        produced one.
        """
        ref = None
        for attempt in [s for s in self.dispatcher.steps
                        if s.get("call_id") == call_id]:
            key = (call_id, attempt.get("retry_idx", 0))
            if key in self._logged_attempts:
                continue
            self._logged_attempts.add(key)
            idx = self.alloc_step_idx()
            row = {k: v for k, v in attempt.items() if k in tracemod.STEP_COLS}
            row.update(step_idx=idx, parent_step_idx=parent, depth=depth,
                        actor=actor, action_type=ActionType.LLM_CALL,
                        action_payload=prompt)
            blobs: dict[str, bytes] = {}
            rendered = attempt.get("rendered")
            if rendered is not None:
                # The EXACT rendered request, stored so a prompt that the
                # server rejected is investigable at all — for B1 the fit
                # arithmetic is the thing under suspicion, and only the bytes
                # plus the measured prefix length can settle it.
                blobs["root_request_ref"] = tracemod.pack_blob({
                    "rendered": rendered.encode("utf-8"),
                    "meta": tracemod.safe_json(
                        {"layout": attempt.get("layout"),
                         "prefix_tokens": attempt.get("prefix_tokens")}
                    ).encode("utf-8", "replace"),
                })
            if answer is not None and attempt.get("status") == StepStatus.OK:
                blobs["observation_full_ref"] = answer.encode("utf-8", "replace")
                ref = tracemod.blob_rel(self.episode_id, idx,
                                         "observation_full_ref")
            self.put_step(row, blobs or None)
        if ref is not None:
            self.answer_ref = ref
        return ref

    # -- the model call ------------------------------------------------------ #

    async def call_leaf(self, prompt: str, *, chunk: str | None = None,
                         slot_id: int | None = None,
                         parent: int | None = None) -> str:
        """One leaf call through C4, wall-clock-checked on both sides and
        logged either way.

        The post-call check is what makes the clock enforceable at all for an
        arm with no watchdog, and it runs AFTER the step is written: a call that
        overran still happened, and an episode whose kill erased its own
        evidence would be unauditable.
        """
        self.check_wall_clock()
        call_id = str(uuid.uuid4())
        if self._slot_kwarg is None:
            self._slot_kwarg = _accepts_slot_id(self.dispatcher.query)
        kwargs: dict[str, Any] = {"role": "leaf", "call_id": call_id,
                                  "chunk": chunk}
        if slot_id is not None and self._slot_kwarg:
            kwargs["slot_id"] = slot_id
        try:
            raw = await self.dispatcher.query(prompt, **kwargs)
        except BaseException:
            self.log_call(call_id, prompt, parent=parent)
            raise
        answer = _answer_text(raw)
        self.log_call(call_id, prompt, answer=answer, parent=parent)
        self.check_wall_clock()
        return answer

    # -- §6 outcome ---------------------------------------------------------- #

    def outcome_for_answer(self, answer: str | None) -> tuple[Outcome, str | None]:
        """§6: `success` = an answer was produced AND the task's checker passes."""
        if answer is None:
            return Outcome.FAIL, NO_ANSWER
        return ((Outcome.SUCCESS, None) if self.task.check(answer)
                else (Outcome.FAIL, CHECKER_FAILED))

    def finish(self, answer: str | None) -> ArmResult:
        return self.close(*self.outcome_for_answer(answer), answer=answer)

    def close(self, outcome: Outcome, reason: str | None = None,
               answer: str | None = None) -> ArmResult:
        """Close the episode exactly once. Idempotent so the failure path may
        close defensively without overwriting an outcome already recorded."""
        if self._closed is not None:
            return self._closed
        self.trace.close_episode(self.episode_id, outcome, reason,
                                  self.answer_ref)
        self._closed = ArmResult(self.episode_id, outcome, reason, answer)
        return self._closed


# --------------------------------------------------------------------------- #
# B1 — single shot, full native window
# --------------------------------------------------------------------------- #


async def _b1_prompt(task: "Task", cfg: Config, *, dispatcher: Any,
                      registry: PromptRegistry) -> tuple[str, dict, int]:
    """Assemble B1's one prompt and the truncation record that describes it.

    Runs BEFORE the episode row is opened — deliberately. Everything here can
    refuse (a missing bench profile, an unreadable corpus, an unreachable
    tokenizer), and a refusal must not leave a NULL-outcome row behind for crash
    recovery to tombstone as `orphaned_at_recovery`. It mirrors the RLM arm,
    which likewise opens its row only after C2 has run.
    """
    head = registry.render_baseline("b1_single_shot")
    capacity = bench_slot_capacity(cfg)
    corpus = load_context(task.context)
    corpus_tokens = await dispatcher.count_tokens(corpus, role="leaf")
    overhead = await dispatcher.count_tokens(head + task.text, role="leaf")
    # §8's fit budget: one slot, less the instruction + question, less the room
    # the answer must decode into, less slack for the separators and the chat
    # template's markup.
    fit = capacity - overhead - cfg.scaffold.budgets.max_predict.leaf - FIT_SLACK_TOKENS
    ceiling = capacity - cfg.scaffold.budgets.max_predict.leaf

    rounds = 0
    while True:
        text, record = truncate_head_tail(corpus, corpus_tokens=corpus_tokens,
                                           fit_tokens=fit)
        prompt = f"{head}\n\n{text}\n\n{task.text}"
        total = await dispatcher.count_tokens(prompt, role="leaf")
        if total <= ceiling or fit <= 0 or rounds >= MAX_FIT_VERIFY_ROUNDS:
            break
        # Char-proportionality under-counted. Shrink by the measured overshoot
        # plus the same slack, and re-cut: the estimate is wrong by a density
        # ratio, so one correction converges and three is a hard stop.
        fit -= (total - ceiling) + FIT_SLACK_TOKENS
        rounds += 1
    record.update(fit_tokens=fit, prompt_tokens=total,
                   slot_capacity_tokens=capacity, verify_rounds=rounds)
    return prompt, record, corpus_tokens


async def run_b1(task: "Task", cfg: Config, *, dispatcher: Any, trace: Any,
                  registry: PromptRegistry, bench_extra: dict[str, Any],
                  slot_id: int = 0) -> ArmResult:
    """§8's B1: the leaf model, one shot, its full native window.

    `slot_id` is the pre-registered pin (B1 = slot 0, B3 = slot 1). §8's v0.2.6
    correction: landing both single-shot arms on slot 0 of one process is R13's
    smallest reproducing case verbatim — two documents, one slot — in the
    configuration measured at 4/18 leaked, and it would have contaminated
    precisely the two arms that are supposed to be spared. It is passed to the
    dispatcher when the dispatcher can pin, and recorded in the snapshot either
    way; `steps.slot_id` stays C4's, because that column means the slot the
    server actually served on.
    """
    ep = ArmEpisode(task, cfg, dispatcher=dispatcher, trace=trace,
                     registry=registry, arm="b1", bench_extra=bench_extra)
    ep.start_clock()
    try:
        prompt, truncation, corpus_tokens = await _b1_prompt(
            task, cfg, dispatcher=dispatcher, registry=registry)
        ep.open_episode(arm_snapshot={"slot_id": slot_id,
                                       "b1_truncation": truncation},
                         tokenized_task_len=corpus_tokens)
        try:
            answer = await ep.call_leaf(prompt, slot_id=slot_id)
        except (BudgetBreach, DispatchError) as exc:
            return ep.close(*outcome_for_error(exc))
        except BaseException:
            # Never orphan the row (§6). The exception still propagates: a bug
            # in an arm must not be scoreable as an ordinary `error` episode.
            ep.close(Outcome.ERROR, ARM_ERROR)
            raise
        return ep.finish(answer)
    finally:
        # D21: the trace is durable before the caller reads or exports it. The
        # TraceLogger itself belongs to the bench run, not to one episode, so it
        # is drained here and closed there.
        await trace.drain()
