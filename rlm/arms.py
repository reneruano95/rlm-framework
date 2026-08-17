"""§8's baseline arms — the controls RLM is measured against (ARCHITECTURE.md §8).

Three arms live here (B1 and B3 now; B2 lands beside them), and they exist so
the S4 verdict can say what the scaffold is worth **relative to something**:

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
    leaf slot pool exhausted   -> error / slot_pool_exhausted  (B2, see below)
    a rotation could not run   -> error / rotation_failed      (B2, see below)

**THE STEPS COME FROM C4, NOT FROM HERE.** Every model call is routed through
the injected dispatcher, so the R13 leak columns, the retry attempts and the
timing/token fields are recorded by the one component that measures them. This
module only copies C4's attempt dicts onto trace rows (the pattern
`rlm/episode.py:720-757` owns), which is why an arm's steps are comparable with
the RLM arm's rather than merely similar to them.

**B2'S MAP CAN DRAIN THE LEAF'S NEVER-REUSE SLOT POOL, AND ROTATES.** R13's
mitigation gives every window one never-reused slot, sized to `--parallel`
(128 on the measured config); a §8 aggregation corpus needs ~300 of them, so
without a rotation `SlotPoolExhausted` would end every such task `error`
partway through the map -- a manufactured contamination-class loss, not a
finding about the task. `ArmEpisode` therefore accepts an optional
`process_manager` (the `rlm.serverproc.ProcessManager` duck type: one method,
`.restart()`) and `call_leaf` rotates through it on `SlotPoolExhausted` ONLY
(§5 C4: a FAILED server is never restarted, only a HEALTHY one whose pool
served its `--parallel` windows), mirroring `rlm/episode.py::_rotate_leaf`'s
quiesce -> restart -> `rotate_pool()` -> resume sequence with one guarantee
deliberately narrowed and documented (`ArmEpisode._rotate_leaf`'s docstring):
this module cannot re-run §4's `/props` handshake against the restarted
process, because it cannot talk HTTP at all (`FORBIDDEN_ROOTS` in
`tests/test_import_rules.py`), not merely `rlm.dispatcher`. `None` (no
`process_manager` injected) is today's behaviour, unchanged: a clean
`error/slot_pool_exhausted`, never a crash. B1/B3 never pass one -- their one
call each cannot exhaust a pool sized >= 1.

**KILL GRANULARITY IS ONE MODEL CALL.** C5's wall clock is checked immediately
before and immediately after each dispatch, and nowhere else. The RLM arm needs
a 0.1 s watchdog because a sandbox cell can loop forever and never return to the
turn loop; a baseline arm has no sandbox and no cell — it makes between one and
a few hundred calls and does nothing else — so the coarser bound is the honest
one, and it is stated here rather than discovered from the code. A breach is
therefore detected at most one model call late (bounded by that call's timeout
× `max_attempts` -- `servers.<role>.per_call_timeout_s` where the profile sets
one, `scaffold.retries.per_call_timeout_s` otherwise; B1/B3 run on the bench
profile's 900 s, sized against a 262K-token slot's measured prefill).

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

**B3'S SELECTION RULE IS PRE-REGISTERED (§8), AND DELIBERATELY THE DUMBEST ONE
THAT IS STILL DETERMINISTIC.** `bm25_select` ranks C2's chunks by BM25 score,
descending, ties broken to the lower chunk index; then greedily takes chunks in
that order while the running token sum still fits the 80%-of-slot budget; then
STOPS at the first chunk that does not fit. It does NOT skip that chunk and try
the next-best one that might still fit — a skip-and-continue rule is a knob (how
many chunks does it look past? by what margin?) and every knob is a place S4's
verdict could have been tuned toward a result after seeing one. Stopping at the
first miss has no such knob. The consequence is priced in on purpose: a single
large chunk ranked just above the cutoff can waste budget that several smaller,
lower-ranked chunks would have filled, and B3 is measured with that inefficiency
left in rather than optimised away. The record `bm25_select` returns —
`{"ranked", "selected", "budget_tokens", "fts": True}` — is what makes the rule
auditable after the fact: `selected` is reconstructable as a prefix-by-fit of
`ranked` against the same token counts, so a result cannot silently have used a
smarter rule than the one pre-registered here. `run_b3` writes it into
`config_snapshot["bench"]` verbatim (§8).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import duckdb

from rlm import trace as tracemod
from rlm.budget import BudgetEnforcer
from rlm.budget import Budgets as BudgetLimits
from rlm.chunker import ChunkConfig, split
from rlm.config import Config, PromptRegistry, config_snapshot
from rlm.context import DOCUMENT_SEPARATOR, load_context
from rlm.errors import (
    ActionType,
    Actor,
    BudgetBreach,
    ConfigError,
    DispatchError,
    Outcome,
    ServerRotationError,
    SlotPoolExhausted,
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
    "NO_SUMMARY",
    "ROTATION_FAILED",
    "SERVER_UNREACHABLE",
    "SLOT_POOL_EXHAUSTED",
    "SNAPSHOT_KEYS",
    "ArmEpisode",
    "ArmResult",
    "b2_summary_n_predict",
    "bench_slot_capacity",
    "bm25_select",
    "is_transport_error",
    "outcome_for_error",
    "run_b1",
    "run_b2",
    "run_b3",
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

#: B2's deterministic stand-in for an empty/whitespace-only chunk summary
#: (module docstring, B2 section): never crash-and-error the arm on one bad
#: summary, and never silently drop the chunk from the numbered list either —
#: the root still sees a slot for it, just one that says nothing was there.
NO_SUMMARY = "[no summary]"

#: `rlm/episode.py`'s rotation vocabulary, DUPLICATED here for the same
#: reason `CHECKER_FAILED`/`SERVER_UNREACHABLE` are (importing `rlm.episode`
#: would drag C4 in) — `SLOT_POOL_EXHAUSTED` fires when a leaf's never-reuse
#: slot pool ran out and either no `process_manager` was injected or the pool
#: was error-drained (not a healthy pool that simply served its `--parallel`
#: windows, spec §5 C4 — see `ArmEpisode._rotate_leaf`); `ROTATION_FAILED`
#: fires when a `process_manager.restart()` that WAS attempted could not
#: complete. `tests/test_arms.py` pins both against `rlm.episode`'s.
SLOT_POOL_EXHAUSTED = "slot_pool_exhausted"
ROTATION_FAILED = "rotation_failed"

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


#: Exception FAMILIES that mean "the injected client could not reach the
#: server", recognised by the module their type is defined in.
#:
#: `arms.py` may not import `httpx` (the dependency rule -- `FORBIDDEN_ROOTS`
#: in `tests/test_import_rules.py` bars every HTTP library from this side), so
#: `except httpx.HTTPError` is not available here and the family is recognised
#: structurally instead. The alternative -- `except Exception` around the root
#: call -- would swallow the scaffold's own bugs and score them as an ordinary
#: `error` episode, which §6 forbids and which this module's own `except
#: BaseException: ep.close(...); raise` exists to prevent.
_TRANSPORT_MODULES = frozenset({"httpx", "httpcore", "ssl", "json"})


def is_transport_error(exc: BaseException) -> bool:
    """Is `exc` a TRANSPORT failure from an injected HTTP client?

    `OSError` covers the socket layer; the module check covers httpx/httpcore
    (connect, read, write, pool timeouts, and `HTTPStatusError` from
    `raise_for_status`) and a response body that would not parse as JSON --
    all of them "the server did not answer usefully", none of them a bug in
    this file.
    """
    if isinstance(exc, OSError):
        return True
    return (type(exc).__module__ or "").split(".")[0] in _TRANSPORT_MODULES


def outcome_for_error(exc: BaseException) -> tuple[Outcome, str]:
    """§6's attribution for a failed arm, in the one place it is decided.

    A `BudgetBreach` already carries the outcome C5 decided, and it wins as-is:
    an arm killed for `wall_clock` whose dispatcher then reports a failure is
    `budget_kill`, not `error`. `SlotPoolExhausted` (checked BEFORE the generic
    `DispatchError` branch below, since it is a subclass of it) is B2's honest
    degraded mode when a rotation could not run or was refused — see
    `ArmEpisode._rotate_leaf`'s docstring — and it is deliberately not lumped
    into `server_unreachable`: the leaf never failed to answer, the pool simply
    had nothing left to hand out. `ServerRotationError` is a rotation that WAS
    attempted (a `process_manager` was injected) and could not complete.
    Everything else from C4 is `error/server_unreachable` — a baseline arm has
    no route around a leaf that will not answer.
    """
    if isinstance(exc, BudgetBreach):
        return exc.outcome, exc.reason
    if isinstance(exc, SlotPoolExhausted):
        return Outcome.ERROR, SLOT_POOL_EXHAUSTED
    if isinstance(exc, ServerRotationError):
        return Outcome.ERROR, ROTATION_FAILED
    if isinstance(exc, DispatchError):
        return Outcome.ERROR, SERVER_UNREACHABLE
    return Outcome.ERROR, ARM_ERROR


def _settled_tokens(attempts: list[dict[str, Any]]) -> tuple[int, int]:
    """`rlm/episode.py::settled_tokens`, DUPLICATED ON PURPOSE: importing
    `rlm.episode` here would drag C4's composition root into this module (the
    module docstring's "THIS MODULE NEVER IMPORTS C4"). The sum itself is
    tiny — every attempt's tokens count against `max_total_tokens` (spec §5
    C4), zero for an attempt that reported none — so duplicating it costs two
    lines; `test_settled_tokens_matches_the_episode_runners_implementation`
    pins the two implementations equal so the duplication cannot drift in
    silence."""
    return (sum(a.get("tokens_in") or 0 for a in attempts),
            sum(a.get("tokens_out") or 0 for a in attempts))


#: `rlm.rootclient.strip_reasoning`'s think-block regex, copied verbatim —
#: see `_strip_reasoning`'s docstring for why it is copied rather than
#: imported.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """`rlm.rootclient.strip_reasoning`, DUPLICATED ON PURPOSE: `arms.py` may
    not import `rlm.rootclient` (`tests/test_import_rules.py`'s
    `FORBIDDEN_RLM` — it is C4's HTTP client, exactly what the module
    docstring's "THIS MODULE NEVER IMPORTS C4" forbids). B2's root call still
    needs D16's belt-and-braces strip (a leading `<think>...</think>` block,
    or text up to the LAST `</think>`) so a root that reopens or never closes
    a think block cannot leak reasoning into the scored answer.
    `test_strip_reasoning_matches_rootclients_implementation` pins this
    implementation equal to the original so the duplication cannot drift in
    silence."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()


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
                 bench_extra: dict[str, Any] | None,
                 scaffold_instance_id: str = "",
                 scaffold_git_sha: str = "",
                 process_manager: Any = None) -> None:
        self.task = task
        self.cfg = cfg
        self.dispatcher = dispatcher
        self.trace = trace
        self.registry = registry
        self.arm = arm
        self.bench_extra = dict(bench_extra or {})
        # §5 C4's rotation (v0.2.6), B2's leaf topology only: the object that
        # owns the leaf PROCESS (`rlm.serverproc.ProcessManager`'s duck type —
        # one method, `.restart()`), never constructed here (arms.py may not
        # start or stop processes any more than it may talk HTTP). `None` is
        # "current behaviour" -- `SlotPoolExhausted` propagates unrotated, the
        # honest degraded mode `outcome_for_error` maps cleanly (see
        # `_rotate_leaf`'s docstring). B1/B3 never pass one: their one call
        # each cannot exhaust a pool sized >= 1.
        self.process_manager = process_manager
        #: How many rotations THIS episode has completed. In-memory only —
        #: unlike `rlm/episode.py`, no `config_snapshot` field carries this
        #: (episode.py itself has none either; `steps.server_rotation`,
        #: stamped by `_rotate_leaf`, is the durable record). Exposed for
        #: tests and for a caller that wants a fast total without scanning
        #: steps.
        self.rotations = 0
        self._rotation_lock = asyncio.Lock()
        # §6's provenance columns, PASSED IN rather than computed here: `rlm
        # run` already owns both answers (`cli.py:529` -- the pid of the process
        # that ran the episode, and the sha of the tree it ran from), and a
        # second derivation inside an arm could disagree with the RLM arm's for
        # the same block. An empty string is what the schema defaults to and is
        # what an ad-hoc call gets; a bench run supplies both.
        self.scaffold_instance_id = scaffold_instance_id
        self.scaffold_git_sha = scaffold_git_sha
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

    # -- R13 ----------------------------------------------------------------- #

    def set_corpus(self, chunks: Any) -> None:
        """Hand C4 the corpus its foreign-string detector indexes against.

        MANDATORY FOR EVERY ARM, not an optimisation. §8 (v0.2.6) makes it
        binding — "R13's foreign-string detector runs on every leaf call during
        S4 and its hit count is reported per arm in the verdict" — and the
        column is tri-state: without this call C4 records NULL (*not checked*)
        rather than False, so the per-arm hit count would have no denominator
        and the verdict could not be written at all.

        Same entry point the RLM arm uses (`rlm/episode.py:817`), and
        deliberately unguarded: a dispatcher without `set_corpus` is not a
        dispatcher this benchmark may run on, and degrading to NULL silently is
        precisely the failure this call exists to prevent.
        """
        self.dispatcher.set_corpus(chunks)

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
            "scaffold_instance_id": self.scaffold_instance_id,
            "scaffold_git_sha": self.scaffold_git_sha,
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
                  depth: int = 1,
                  attempts: list[dict[str, Any]] | None = None) -> str | None:
        """One trace step per dispatch ATTEMPT, copied from C4's own records.

        Idempotent per `(call_id, retry_idx)` so the failure path and a later
        flush cannot double-write the same attempt (`rlm/episode.py:720-757`).
        Returns the blob path holding the answer, or None when no attempt
        produced one.

        `attempts` defaults to C4's own attempt records for `call_id`
        (`self.dispatcher.steps` filtered) — every leaf caller's behaviour.
        B2's root call passes ONE SYNTHESIZED attempt instead: root traffic
        goes through the injected `root_client`, never through `dispatcher`,
        so there is nothing on `dispatcher.steps` to filter. Shaping that one
        attempt like C4's own (same keys: `status`, `tokens_in/out`, `rendered`,
        ...) is what lets it flow through this SAME method — the blob-writing
        and `STEP_COLS` projection below — rather than a second, parallel
        implementation for root calls.
        """
        ref = None
        if attempts is None:
            attempts = [s for s in self.dispatcher.steps
                        if s.get("call_id") == call_id]
        for attempt in attempts:
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

    def can_pin_slot(self) -> bool:
        """Whether this dispatcher can actually honour a slot pin.

        Resolved (and cached) here so an arm can RECORD the answer next to the
        pin it asked for. A snapshot that says `slot_id: 0` while the keyword
        was dropped on the floor is worse than no record: §8's v0.2.6 correction
        obliges B1 and B3 to take their OWN slot, and that obligation has to be
        checkable from the trace rather than assumed from the config.
        """
        if self._slot_kwarg is None:
            self._slot_kwarg = _accepts_slot_id(self.dispatcher.query)
        return self._slot_kwarg

    async def call_leaf(self, prompt: str, *, chunk: str | None = None,
                         slot_id: int | None = None,
                         parent: int | None = None,
                         admit_tokens: int | None = None,
                         n_predict: int | None = None) -> str:
        """One leaf call through C4, wall-clock-checked on both sides and
        logged either way.

        The post-call check is what makes the clock enforceable at all for an
        arm with no watchdog, and it runs AFTER the step is written: a call that
        overran still happened, and an episode whose kill erased its own
        evidence would be unauditable.

        `admit_tokens`, when given, is the pre-flight PROMPT token count and
        routes this call through `BudgetEnforcer.admit()`/`.settle()` — B2's
        obligation (this task): its leaf summaries are admitted against
        `max_subcalls` AND `max_total_tokens` exactly as the RLM arm's leaf
        sub-calls are (`rlm/episode.py`'s `_EpisodeRun._on_llm_query`/
        `_settle`, mirrored here rather than imported — the dependency rule).
        B1/B3 never pass it: their ONE call per episode is already bounded by
        the bench_leaf relaunch profile's single-slot capacity, so admitting
        it besides would only duplicate that check with a different one.

        `n_predict`, when given, CAPS this call's decode. B2 passes
        `b2_summary_n_predict`'s value on every summary so that the budget §8
        pre-registers is ENFORCED rather than merely recorded: the whole point
        of that formula is that all `n_chunks` summaries fit 80% of the root
        window, and a 299-chunk corpus decoding at the leaf's full
        `max_predict` overflows the reduce prompt by construction. Omitted, C4
        uses the leaf role's `max_predict` exactly as before.

        A `BudgetBreach` from `.admit()` is raised BEFORE any dispatch and
        BEFORE any reservation is recorded (`BudgetEnforcer.admit` never
        partially reserves — see its docstring) — nothing was sent, so
        (matching `rlm/episode.py`'s own admission path) nothing is logged as
        a step; the breach itself is the episode's outcome.

        The dispatch itself goes through `_dispatch_leaf`, which rotates the
        leaf server on `SlotPoolExhausted` when a `process_manager` was
        injected (see its docstring) — B2's obligation (this task's second
        addition): B2's ~300-window map would otherwise hit
        `SlotPoolExhausted` at window `--parallel` (128) and every aggregation
        task would end `error` for a reason that has nothing to do with the
        task, a manufactured §8 contamination-class loss.
        """
        self.check_wall_clock()
        call_id = str(uuid.uuid4())
        reservation = (self.enforcer.admit(admit_tokens, "leaf", call_id)
                       if admit_tokens is not None else None)
        kwargs: dict[str, Any] = {
            "role": "leaf", "call_id": call_id, "chunk": chunk,
            # THIS EPISODE'S SEED, PER CALL -- never the one the dispatcher was
            # built with. §8 re-seeds the CONFIG for each of its three
            # replicates (`rlm.bench.seeded_config`) while one bench run holds
            # a single leaf dispatcher across all of them, so a
            # construction-time seed would give every replicate the same leaf
            # draw while `config_snapshot` recorded three different ones.
            "seed": self.cfg.scaffold.sampling.leaf.seed,
        }
        if n_predict is not None:
            kwargs["n_predict"] = n_predict
        if slot_id is not None and self.can_pin_slot():
            kwargs["slot_id"] = slot_id
        try:
            raw = await self._dispatch_leaf(prompt, call_id, kwargs)
        except BaseException:
            if reservation is not None:
                self._settle_admission(reservation, call_id)
            self.log_call(call_id, prompt, parent=parent)
            raise
        answer = _answer_text(raw)
        if reservation is not None:
            self._settle_admission(reservation, call_id)
        self.log_call(call_id, prompt, answer=answer, parent=parent)
        self.check_wall_clock()
        return answer

    async def _dispatch_leaf(self, prompt: str, call_id: str,
                              kwargs: dict[str, Any]) -> Any:
        """One `dispatcher.query()`, rotating the leaf server ONCE on
        `SlotPoolExhausted` when a `process_manager` was injected — spec §5
        C4's rotation, scoped to what `arms.py` can reach (see
        `_rotate_leaf`'s docstring for the one guarantee this narrows
        relative to `rlm/episode.py`'s `_dispatch_leaf`, which this mirrors).

        FIRES ONLY ON EXHAUSTION, NEVER ON ANY OTHER `DispatchError` (§5's
        rule: a server that FAILED is never restarted — that would mask the
        fault the trace exists to record — only a HEALTHY one whose pool ran
        out is a resource-lifecycle operation). `pool_error_drained` (when the
        dispatcher exposes it, matching `LLMDispatcher`) distinguishes the two
        shapes of exhaustion the same way `rlm/episode.py:586` does: a
        generation that answered NOTHING is a failed server wearing pool
        exhaustion's exception, not a healthy one that ran out of windows.

        Retries the SAME `call_id` ONCE after a successful rotation — B2's
        map is serial, so (unlike `rlm/episode.py`'s up-to-
        `MAX_ROTATIONS_PER_CALL` loop, sized for concurrent callers racing the
        same pool) one rotation is always enough to make forward progress on
        one call; a second exhaustion immediately against a freshly rotated,
        still-virgin pool would be a distinct bug, not something to loop on.
        """
        try:
            return await self.dispatcher.query(prompt, **kwargs)
        except SlotPoolExhausted:
            if self.process_manager is None:
                # Nobody owns the leaf process (launched outside `rlm run
                # bench`, or a caller that simply chose not to inject one) --
                # there is nothing to rotate. The refusal propagates
                # unchanged; `outcome_for_error` maps it to a clean
                # `error/slot_pool_exhausted`, never a crash and never a
                # silent wrap onto a slot that has held another document.
                raise
            if getattr(self.dispatcher, "pool_error_drained", False):
                raise
            await self._rotate_leaf(call_id)
            return await self.dispatcher.query(prompt, **kwargs)

    async def _rotate_leaf(self, call_id: str) -> None:
        """Replace the healthy leaf process and resume on a virgin pool --
        `rlm/episode.py::_rotate_leaf`'s sequence, mirrored: quiesce C4
        (`dispatcher.rotating()`, quiesce -> resume with the reopen in a
        `finally` INSIDE C4) -> `process_manager.restart()` -> `rotate_pool()`
        (a new process means a new pool) -> resume. Serialized on
        `_rotation_lock` for the same reason episode.py's is: a second caller
        that queued behind the first finds the pool already refilled and
        returns without rotating again (moot for B2 today, whose map is
        serial and therefore never has two calls racing this method, but
        `call_leaf` is shared plumbing and the lock costs nothing idle).

        ONE GUARANTEE THIS NARROWS, DELIBERATELY, DOCUMENTED RATHER THAN LEFT
        TO BE DISCOVERED: `rlm/episode.py` re-runs §4's `/props` handshake
        against the freshly restarted process (`_rehandshake_leaf`,
        `assert_props` — total_slots/n_ctx/build_info re-checked against
        config) as an EXTRA, independent verification that the replacement
        really is what config describes. `arms.py` cannot do that: it is one
        of the ISOLATED modules `tests/test_import_rules.py` forbids from
        importing an HTTP client AT ALL (`FORBIDDEN_ROOTS` blocks `httpx`
        *and* the stdlib `http`/`socket`, not merely `rlm.dispatcher`) — there
        is no way to GET `/props` from inside this module, so re-running the
        handshake here the way `_rehandshake_leaf` does is not "not imported
        for tidiness", it is architecturally unreachable. Two things still
        stand in for it, both weaker than a pre-emptive `/props` check but
        not nothing: `ProcessManager.restart()`'s OWN contract already
        promises "a fresh process of the SAME CONFIGURATION ... returning
        only once it answers /health" (`rlm/serverproc.py`'s `ProcessManager`
        Protocol), and C4's per-target prefix-hash check
        (`LLMDispatcher._query_once`) fires automatically on the very next
        call if the restarted process renders a different system head, so a
        template/config drift is still CAUGHT, one call later, as
        `PrefixDrift`, just not pre-emptively. A caller that wants the full
        §4 re-validation composes it into its own `process_manager` (e.g. a
        `.restart()` that also calls `rlm.episode.handshake` before
        returning) — `arms.py` only ever calls the one method the
        `ProcessManager` Protocol already promises.

        Stamps `steps.server_rotation` on the triggering (exhausted) attempt,
        the SAME mechanism `rlm/episode.py:683-696` (`_stamp_rotation`) uses:
        `self.dispatcher.steps` is a plain public list of dicts — the one
        `log_call` already reads — so mutating the last recorded attempt for
        `call_id` in place needs no C4 import, only data C4 already exposes.
        Also counted in `self.rotations` (in-memory; `episode.py` keeps no
        `config_snapshot` field for this either, only the per-step column and
        a lifecycle-log event this module has no lifecycle log to write to).
        """
        async with self._rotation_lock:
            if not getattr(self.dispatcher, "restart_required", True):
                return    # someone else already rotated while this queued
            # `process_manager.restart()` raising `ServerRotationError` (or
            # anything else, including cancellation) propagates UNCHANGED --
            # `dispatcher.rotating()`'s own `finally` still reopens the gate
            # (its docstring: parked calls must take a refusal, never hang),
            # and the caller's `outcome_for_error` maps `ServerRotationError`
            # to a dedicated `error/rotation_failed` rather than the generic
            # `arm_error` a bare `except: raise` would add nothing over.
            async with self.dispatcher.rotating():
                await self.process_manager.restart()
                self.dispatcher.rotate_pool()
            self.rotations += 1
            attempts = [s for s in self.dispatcher.steps
                        if s.get("call_id") == call_id]
            if attempts:
                attempts[-1]["server_rotation"] = self.rotations

    def _settle_admission(self, reservation: Any, call_id: str) -> None:
        """Release `reservation`'s hold and charge what the call ACTUALLY
        cost, summed over every attempt C4 recorded for `call_id` — spec §5
        C4's asymmetry: a retried call counts once against `max_subcalls`,
        but every attempt's tokens count against `max_total_tokens`. The mock
        dispatcher records no token usage, so this settles zero for it —
        stated, not a bug (`rlm/episode.py::_settle`'s pattern, mirrored)."""
        attempts = [s for s in self.dispatcher.steps if s.get("call_id") == call_id]
        tokens_in, tokens_out = _settled_tokens(attempts)
        self.enforcer.settle(reservation, tokens_in, tokens_out)

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
                      registry: PromptRegistry) -> tuple[str, dict, int, str]:
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
    return prompt, record, corpus_tokens, corpus


async def run_b1(task: "Task", cfg: Config, *, dispatcher: Any, trace: Any,
                  registry: PromptRegistry, bench_extra: dict[str, Any],
                  slot_id: int = 0, scaffold_instance_id: str = "",
                  scaffold_git_sha: str = "") -> ArmResult:
    """§8's B1: the leaf model, one shot, its full native window.

    `slot_id` is the pre-registered pin (B1 = slot 0, B3 = slot 1). §8's v0.2.6
    correction: landing both single-shot arms on slot 0 of one process is R13's
    smallest reproducing case verbatim — two documents, one slot — in the
    configuration measured at 4/18 leaked, and it would have contaminated
    precisely the two arms that are supposed to be spared. It is passed to the
    dispatcher when the dispatcher can pin, and the snapshot records BOTH the
    requested slot and whether it was applied — an unapplied pin recorded as if
    it had been is how that correction would get lost. `steps.slot_id` stays
    C4's, because that column means the slot the server actually served on.
    """
    ep = ArmEpisode(task, cfg, dispatcher=dispatcher, trace=trace,
                     registry=registry, arm="b1", bench_extra=bench_extra,
                     scaffold_instance_id=scaffold_instance_id,
                     scaffold_git_sha=scaffold_git_sha)
    ep.start_clock()
    try:
        prompt, truncation, corpus_tokens, corpus = await _b1_prompt(
            task, cfg, dispatcher=dispatcher, registry=registry)
        # R13's detector, indexed against the FULL corpus while the (possibly
        # truncated) document is what gets sent. Indexing the truncated text
        # instead would make the check vacuous by construction -- every token in
        # the index would also be in `sent`, so no answer could ever score a hit
        # and the column would be a row of trivially-False values, which
        # `schema.sql` warns must never be read as "leak-free". Indexed this
        # way, an identifier the model produced from the DROPPED middle is a
        # hit: text it was not shown, in the arm §8 relies on being spared.
        ep.set_corpus([corpus])
        ep.open_episode(arm_snapshot={"slot_id": slot_id,
                                       "slot_id_applied": ep.can_pin_slot(),
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


# --------------------------------------------------------------------------- #
# B3 -- deterministic BM25-RAG single shot over C2's chunk windows
# --------------------------------------------------------------------------- #


class _ChunkTokenCounter:
    """`rlm.episode`'s `_TokenCounter`, copied rather than imported.

    C2's `split()` is synchronous (it imports no LLM client, by the dependency
    rule §5) while `/tokenize` is an HTTP call, so it runs off-loop on a worker
    thread and each count is marshalled back onto the calling coroutine's event
    loop. `arms.py` cannot import `rlm.episode` (the module docstring's "THIS
    MODULE NEVER IMPORTS C4"), so this is the same three lines a second time
    rather than a shared helper that would put C4 one import away.
    """

    def __init__(self, dispatcher: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._dispatcher = dispatcher
        self._loop = loop

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        future = asyncio.run_coroutine_threadsafe(
            self._dispatcher.count_tokens(text, role="leaf"), self._loop)
        return future.result()


def _bm25_rank(chunks: list[str], question: str) -> list[int]:
    """BM25 rank of `chunks` against `question`, descending score, ties broken
    to the lower chunk index -- both done IN SQL (`ORDER BY s DESC, id ASC`) so
    the tie-break is not a second, possibly-different sort in Python.

    One in-memory DuckDB connection PER CALL, per §8's Step 1 pre-flight note:
    the index is cheap to build fresh (`PRAGMA create_fts_index`, once per leaf
    call at benchmark corpus sizes) and a shared connection across calls would
    let one episode's index leak into another's. Chunks with no matching term
    score NULL and are excluded by `WHERE s IS NOT NULL` -- absent from
    `ranked` entirely, not merely ranked last, which is what lets a corpus with
    no lexical overlap to the question select nothing rather than something
    arbitrary.
    """
    if not chunks:
        return []
    con = duckdb.connect()
    try:
        con.execute("INSTALL fts; LOAD fts")
        con.execute("CREATE TABLE chunks(id INTEGER, body TEXT)")
        con.executemany("INSERT INTO chunks VALUES (?, ?)",
                         list(enumerate(chunks)))
        con.execute("PRAGMA create_fts_index('chunks', 'id', 'body')")
        rows = con.execute(
            "SELECT id, fts_main_chunks.match_bm25(id, ?) AS s "
            "FROM chunks WHERE s IS NOT NULL ORDER BY s DESC, id ASC",
            [question]).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def bm25_select(chunks: list[str], question: str, *, budget_tokens: int,
                 token_counts: list[int]) -> tuple[list[int], dict]:
    """§8's pre-registered B3 selection rule (see the module docstring for the
    "why no skip-and-continue" ruling): BM25 rank desc / ties to the lower
    index, greedy-take while the running token sum fits `budget_tokens`, STOP
    at the first chunk that does not fit.

    Returns `(selected, record)`. `selected` is the chosen chunk indices in
    ORIGINAL corpus order -- what a caller assembles the prompt from, since a
    RAG prompt built in rank order would scramble the document's own sequence
    for no benefit the model could use. `record` is
    `{"ranked": [...], "selected": [...], "budget_tokens": int, "fts": True}`,
    exactly what `run_b3` writes into `config_snapshot["bench"]`: `ranked` is
    every chunk that matched at all, in the order the rule walked them, and
    `record["selected"]` is the same set `selected` returns (kept equal on
    purpose, so the snapshot needs no recomputation to show what was actually
    sent).
    """
    if len(chunks) != len(token_counts):
        raise ValueError(
            f"bm25_select: {len(chunks)} chunks but {len(token_counts)} "
            "token_counts -- exactly one count per chunk is required")
    ranked = _bm25_rank(chunks, question)
    selected: list[int] = []
    total = 0
    for idx in ranked:
        cost = token_counts[idx]
        if total + cost > budget_tokens:
            break  # the pre-registered stop -- no skip-and-continue
        selected.append(idx)
        total += cost
    selected.sort()
    record = {"ranked": ranked, "selected": list(selected),
              "budget_tokens": budget_tokens, "fts": True}
    return selected, record


async def _b3_prompt(task: "Task", cfg: Config, *, dispatcher: Any,
                      registry: PromptRegistry) -> tuple[str, dict, int, list[str]]:
    """Assemble B3's one prompt: C2's chunker VERBATIM (same config geometry,
    same snap rule as `rlm/episode.py:~808` -- B2/B3 sharing C2 verbatim is a
    §8 pre-registration), then `bm25_select`'s greedy pick restored to original
    order.

    Runs BEFORE the episode row is opened, for the same reason `_b1_prompt`
    does: a refusal here (no bench profile, an unreadable corpus, an
    unreachable tokenizer) must not leave a NULL-outcome row for crash
    recovery to tombstone.
    """
    head = registry.render_baseline("b3_single_shot")
    capacity = bench_slot_capacity(cfg)
    corpus = load_context(task.context)
    corpus_tokens = await dispatcher.count_tokens(corpus, role="leaf")
    loop = asyncio.get_running_loop()
    counter = _ChunkTokenCounter(dispatcher, loop)
    chunk_cfg = ChunkConfig(size_tokens=cfg.scaffold.chunk.size_tokens,
                             overhead_tokens=cfg.scaffold.chunk.overhead_tokens,
                             snap_to_boundary=cfg.scaffold.chunk.snap_to_boundary,
                             snap_tolerance=cfg.scaffold.chunk.snap_tolerance,
                             stride_tokens=cfg.scaffold.chunk.stride_tokens)
    chunks = await asyncio.to_thread(split, corpus, chunk_cfg, counter)
    overhead = await dispatcher.count_tokens(head + task.text, role="leaf")
    # §8's fit budget: 80% of the slot -- RAG's headroom against the retrieved
    # set overshooting, since (unlike B1) there is no post-hoc verify loop here
    # -- less the instruction + question, less the room the answer must decode
    # into, less slack for the separators and the chat template's markup.
    budget = (int(0.8 * capacity) - overhead
              - cfg.scaffold.budgets.max_predict.leaf - FIT_SLACK_TOKENS)
    token_counts = [await dispatcher.count_tokens(c, role="leaf") for c in chunks]
    # Off-loop, mirroring `split()` two lines up: `bm25_select` builds and
    # queries an in-memory DuckDB FTS index (`CREATE TABLE` + `INSERT` +
    # `PRAGMA create_fts_index` + a `match_bm25` scan), all synchronous CPU/IO
    # work with no `await` inside it -- measured ~720 ms at 450 chunks. Left
    # un-threaded it would block the bench process's one event loop, which
    # also has to keep servicing the power sampler and lifecycle machinery
    # while this call runs. `bm25_select` itself stays a plain sync function
    # (its unit tests call it directly, no loop required) -- only the call
    # site pays for the loop it happens to run on.
    selected, record = await asyncio.to_thread(
        bm25_select, chunks, task.text, budget_tokens=budget,
        token_counts=token_counts)
    body = "\n\n".join(chunks[i] for i in selected)
    prompt = f"{head}\n\n{body}\n\n{task.text}"
    return prompt, record, corpus_tokens, chunks


async def run_b3(task: "Task", cfg: Config, *, dispatcher: Any, trace: Any,
                  registry: PromptRegistry, bench_extra: dict[str, Any],
                  slot_id: int = 1, scaffold_instance_id: str = "",
                  scaffold_git_sha: str = "") -> ArmResult:
    """§8's B3: deterministic BM25-RAG, one shot, over C2's chunk windows.

    `slot_id` defaults to 1 -- §8's v0.2.6 correction pins B1 to slot 0 and B3
    to slot 1 of the bench relaunch profile, so the two single-shot arms never
    land on the same slot of one process (`run_b1`'s docstring has the R13
    story: two documents, one slot, is R13's smallest reproducing case). It is
    passed to the dispatcher when the dispatcher can pin, and the snapshot
    records BOTH the requested slot and whether it was applied, exactly as B1
    does, for the same reason: an unapplied pin recorded as if it had been
    would silently void the obligation.
    """
    ep = ArmEpisode(task, cfg, dispatcher=dispatcher, trace=trace,
                     registry=registry, arm="b3", bench_extra=bench_extra,
                     scaffold_instance_id=scaffold_instance_id,
                     scaffold_git_sha=scaffold_git_sha)
    ep.start_clock()
    try:
        prompt, record, corpus_tokens, chunks = await _b3_prompt(
            task, cfg, dispatcher=dispatcher, registry=registry)
        # R13's detector, indexed against EVERY chunk C2 produced, not just the
        # ones `bm25_select` kept. Indexing only the selected subset would make
        # the check vacuous for exactly the same reason B1's full-corpus index
        # is: an identifier the model produced from an UNSELECTED chunk is a
        # hit -- text this call never sent -- and that hit is invisible to an
        # index that only knows what was sent.
        ep.set_corpus(chunks)
        ep.open_episode(arm_snapshot={"slot_id": slot_id,
                                       "slot_id_applied": ep.can_pin_slot(),
                                       **record},
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


# --------------------------------------------------------------------------- #
# B2 -- deterministic map-reduce: chunk, summarise (serial), reduce (one root
# call). The honest control: if RLM does not beat this, the root's agency is
# not earning its complexity (module docstring).
#
# UNLIKE B1/B3, B2 runs on the RESIDENT leaf topology -- the RLM-profile
# `LLMDispatcher`, C4's never-reuse slot discipline included -- not the
# bench_leaf single-shot relaunch profile. Two consequences, both
# pre-registered here rather than discovered later: (1) `run_b2` never calls
# `bench_slot_capacity` and never requests a slot pin (`ep.call_leaf`'s
# `slot_id` stays unset for every summary call) -- C4 assigns the slot, the
# same discipline the RLM arm runs under; (2) the summary calls are admitted
# against the SAME episode budget (`max_subcalls`/`max_total_tokens`) an RLM
# episode would be, via `ArmEpisode.call_leaf`'s `admit_tokens` (this task's
# addition to the shared plumbing).
#
# THE SUMMARY BUDGET IS ENFORCED PER CALL, and the reasoning that once said
# otherwise is kept here because it was wrong in an instructive way. It ran:
# `LLMDispatcher.query` has no per-call `n_predict`, a target's budget is fixed
# at construction, and adding an override for one baseline arm would be drift
# into a component §8 needs identical across every arm -- so record the number
# and let the leaf role's `max_predict` do the enforcing.
#
# What that missed is that the number is not an aspiration, it is the thing
# that makes the arm fit: `b2_summary_n_predict` exists so ALL `n_chunks`
# summaries fit 80% of the root window. Unenforced, a 299-chunk aggregation
# corpus decodes 299 x max_predict (512) tokens into a reduce prompt sized for
# 0.8 x 32K -- B2 overflows the root by construction on every aggregation task,
# which is a manufactured §8 result and not a measurement of anything. And the
# drift argument had it backwards: an OPTIONAL, defaulted `n_predict` on
# `query` changes nothing for any other arm (omitted, every caller still gets
# `target.max_predict`), while a budget only one arm silently misses is exactly
# the asymmetry §8 cannot afford. So C4 gained the override (resolved once per
# call, beside the seed), `ep.call_leaf` passes it, and the recorded
# `config_snapshot["bench"]["summary_n_predict"]` is now the number that was
# actually sent rather than the number that was hoped for.
# --------------------------------------------------------------------------- #


def b2_summary_n_predict(cfg: Config, n_chunks: int) -> int:
    """§8's pre-registered B2 summary budget (no tuning): sized so that ALL
    `n_chunks` summaries fit 80% of the root window by construction --
    `n_chunks * n_predict <= 0.8 * window_tokens`, i.e.
    `n_predict = (8 * window_tokens) // (10 * n_chunks)` -- floored at 16
    tokens (a summary shorter than that is not a useful compression of a
    whole chunk) and capped at the leaf role's configured `max_predict`.

    ENFORCED, not merely recorded: `run_b2` passes this value to
    `call_leaf(n_predict=...)` on every summary call, so C4 decodes to it (the
    per-call override on `LLMDispatcher.query`). It was recorded-only once, and
    that made the "fits by construction" claim false for exactly the corpora
    the formula exists for -- at 299 chunks the cap is 87 tokens while the leaf
    would have decoded up to 512, putting ~150K tokens of summary into a
    reduce prompt sized for 0.8 x 32K.

    `n_chunks <= 0` (an empty corpus) returns the leaf role's `max_predict`
    unshrunk rather than dividing by zero -- there is no summary call to size
    for, so the number is moot, and a crash here would turn an empty-document
    task into an `arm_error` for an arithmetic reason that has nothing to do
    with the task.
    """
    if n_chunks <= 0:
        return cfg.scaffold.budgets.max_predict.leaf
    window = cfg.scaffold.root.window_tokens
    return max(16, min(cfg.scaffold.budgets.max_predict.leaf,
                        (8 * window) // (10 * n_chunks)))


async def _b2_chunks(task: "Task", cfg: Config, *,
                      dispatcher: Any) -> tuple[list[str], int]:
    """C2's chunker, verbatim -- the SAME construction `_b3_prompt` uses
    (field-for-field: `size_tokens`, `overhead_tokens`, `snap_to_boundary`,
    `snap_tolerance`, `stride_tokens`), because B2/B3 sharing C2 verbatim is a
    §8 pre-registration. Runs BEFORE the episode row is opened, for the same
    reason `_b1_prompt`/`_b3_prompt` do: a refusal here (an unreadable
    corpus, an unreachable tokenizer) must not leave a NULL-outcome row for
    crash recovery to tombstone.
    """
    corpus = load_context(task.context)
    corpus_tokens = await dispatcher.count_tokens(corpus, role="leaf")
    loop = asyncio.get_running_loop()
    counter = _ChunkTokenCounter(dispatcher, loop)
    chunk_cfg = ChunkConfig(size_tokens=cfg.scaffold.chunk.size_tokens,
                             overhead_tokens=cfg.scaffold.chunk.overhead_tokens,
                             snap_to_boundary=cfg.scaffold.chunk.snap_to_boundary,
                             snap_tolerance=cfg.scaffold.chunk.snap_tolerance,
                             stride_tokens=cfg.scaffold.chunk.stride_tokens)
    chunks = await asyncio.to_thread(split, corpus, chunk_cfg, counter)
    return chunks, corpus_tokens


async def _b2_root_final(cfg: Config, *, ep: ArmEpisode, registry: PromptRegistry,
                          root_client: Any, summaries: list[str],
                          question: str) -> str:
    """B2's reduce step: ONE root call, mirroring `s1/run_s1.py:control_attempt`
    (:144-156) -- `/apply-template` then `/completion`, with
    `cfg.scaffold.sampling.root` (temperature/top_p/seed) carried verbatim,
    via the injected `root_client` at the root port (never constructed here --
    `arms.py` may not import `rlm.dispatcher`, the dependency rule).

    The prompt is `render_baseline("b2_root_final") + "\\n\\n" + numbered
    summaries in order + "\\n\\n" + task.text` (pre-registered, §8). B2 parses
    the RAW completion text as the answer -- there is no REPL cell, the reply
    IS the answer -- with reasoning stripped by `_strip_reasoning` (D16's
    belt-and-braces strip, duplicated from `rlm.rootclient` rather than
    imported; see that function's docstring).

    Logged as ONE `llm_call` step with `actor="root"` through
    `ArmEpisode.log_call`'s shared blob-writing/`STEP_COLS` machinery, via a
    single SYNTHESIZED attempt built from the `CompletionResult` --
    `log_call`'s docstring explains why that is the same method every leaf
    call uses rather than a second implementation.
    """
    head = registry.render_baseline("b2_root_final")
    numbered = "\n\n".join(f"{i + 1}. {s}" for i, s in enumerate(summaries))
    prompt = f"{head}\n\n{numbered}\n\n{question}"
    messages = [{"role": "user", "content": prompt}]

    ep.check_wall_clock()
    call_id = str(uuid.uuid4())
    sampling = cfg.scaffold.sampling.root
    # THE ROOT SERVER IS A SERVER TOO. C4 wraps every leaf transport failure in
    # `DispatchError`, so a leaf that dies is `error/server_unreachable` and §8's
    # rerun-once applies to it. The reduce step talks to the root DIRECTLY (via
    # the injected client), so an httpx error here used to propagate out of
    # `run_b2` uncaught -- past the `(BudgetBreach, DispatchError,
    # ServerRotationError)` handler, through `except BaseException` (which
    # closes the row `arm_error` and RE-RAISES), and out of `rlm.bench._run_cell`,
    # which contains only `ConfigError`. One root hiccup at hour 12 would end
    # the whole grid. Mapped to the same `DispatchError` a leaf failure raises,
    # it becomes this cell's `error/server_unreachable` and gets §8's rerun.
    try:
        rendered = await root_client.apply_template(
            messages,
            chat_template_kwargs={"enable_thinking": cfg.scaffold.root.enable_thinking})
        result = await root_client.completion(
            rendered, n_predict=cfg.scaffold.budgets.max_predict.root,
            temperature=sampling.temperature, top_p=sampling.top_p,
            seed=sampling.seed, stream=True)
    except asyncio.CancelledError:
        raise                       # C5's abort path owns this one
    except BaseException as exc:
        # A BUG STAYS LOUD (`is_transport_error`'s docstring): only the
        # transport family is converted, everything else re-raises and closes
        # the row `arm_error` through `run_b2`'s own handler.
        if not is_transport_error(exc):
            raise
        raise DispatchError(
            f"B2's reduce step could not reach the root server at port "
            f"{cfg.servers.root.port}: {exc}") from exc
    answer = _strip_reasoning(result.content).strip()

    attempt = {
        "call_id": call_id, "retry_idx": 0, "status": StepStatus.OK,
        "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
        "tokens_cached": result.cache_n, "slot_id": result.slot_id,
        "t_first_byte": result.t_first_byte, "t_end": tracemod.utc_now(),
        "latency_prefill_ms": result.prompt_ms,
        "latency_decode_ms": result.predicted_ms,
        "root_view_hash": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "rendered": rendered,
    }
    # depth=0: the root sits at the top of the tree (§6), same convention
    # `rlm/episode.py`'s own root-turn logging uses -- LEAF steps are the
    # ones at depth 1 (`log_call`'s default).
    ep.log_call(call_id, prompt, answer=answer, actor=Actor.ROOT, depth=0,
                attempts=[attempt])
    ep.check_wall_clock()
    return answer


async def run_b2(task: "Task", cfg: Config, *, dispatcher: Any, root_client: Any,
                  trace: Any, registry: PromptRegistry, bench_extra: dict[str, Any],
                  scaffold_instance_id: str = "",
                  scaffold_git_sha: str = "",
                  process_manager: Any = None) -> ArmResult:
    """§8's B2: deterministic map-reduce -- chunk (C2 verbatim), summarise
    (SERIAL, one leaf call per chunk -- `scaffold.dispatch_concurrency` is
    pinned at 1 by config, R14: concurrent leaf dispatch corrupts the answer),
    reduce (one root call over the numbered summaries, in order).

    `root_client` is a second injected dependency alongside `dispatcher`: the
    map step dispatches through C4 exactly as an RLM leaf sub-call does (see
    the section docstring above for the slot/admission consequences of that),
    while the reduce step talks directly to the root server the way
    `s1/run_s1.py:control_attempt` does. Neither is constructed here --
    `arms.py` may not import `rlm.dispatcher`/`rlm.rootclient` (the dependency
    rule; `tests/test_import_rules.py` lints it).

    `process_manager` is a THIRD, optional injected dependency (the
    `rlm.serverproc.ProcessManager` duck type: one method, `.restart()`), and
    it exists because B2's map is the one baseline arm that can actually drain
    the leaf's never-reuse slot pool: an aggregation corpus's ~300 windows
    against `--parallel 128` slots means `SlotPoolExhausted` partway through
    every such task, without it. `None` (the default) is today's behaviour --
    a clean `error/slot_pool_exhausted` outcome via `outcome_for_error`, never
    a crash and never a silent wrap onto a slot that has held another
    document. See `ArmEpisode._rotate_leaf`'s docstring for the rotation
    sequence and the one guarantee it deliberately narrows relative to
    `rlm/episode.py`'s.
    """
    ep = ArmEpisode(task, cfg, dispatcher=dispatcher, trace=trace,
                     registry=registry, arm="b2", bench_extra=bench_extra,
                     scaffold_instance_id=scaffold_instance_id,
                     scaffold_git_sha=scaffold_git_sha,
                     process_manager=process_manager)
    ep.start_clock()
    try:
        chunks, corpus_tokens = await _b2_chunks(task, cfg, dispatcher=dispatcher)
        # R13's detector, indexed against EVERY chunk C2 produced -- B2 sends
        # ALL of them (no selection, unlike B3), so this is the same full-
        # coverage index `run_b3` builds, just over the complete chunk list
        # rather than a BM25 subset. Mandatory before the first dispatch
        # (`ArmEpisode.set_corpus`'s docstring: NULL means NOT CHECKED).
        ep.set_corpus(chunks)
        n_predict = b2_summary_n_predict(cfg, len(chunks))
        ep.open_episode(arm_snapshot={"n_chunks": len(chunks),
                                       "summary_n_predict": n_predict},
                         tokenized_task_len=corpus_tokens)
        try:
            head = registry.render_baseline("b2_leaf_summary")
            summaries: list[str] = []
            for chunk_text in chunks:
                prompt = f"{head}\n\n{chunk_text}"
                prompt_tokens = await dispatcher.count_tokens(prompt, role="leaf")
                # `admit_tokens`: this task's obligation -- every summary is
                # admitted against `max_subcalls`/`max_total_tokens` exactly
                # as an RLM leaf sub-call is. `call_leaf`'s own
                # `check_wall_clock()` at the top of every call is what makes
                # the clock "checked between calls" (Task 6's rule) without a
                # second check here.
                # `n_predict=n_predict`: §8's summary budget ENFORCED, not
                # merely recorded in the snapshot. Without it a 299-chunk
                # aggregation corpus decodes 299 x the leaf's max_predict into
                # a reduce prompt sized for 0.8 x the root window -- B2 would
                # overflow the root by construction on every aggregation task,
                # which is a manufactured §8 result, not a measurement.
                summary = await ep.call_leaf(prompt, admit_tokens=prompt_tokens,
                                              n_predict=n_predict)
                # Deterministic, never crash-and-error the arm on one bad
                # summary (module docstring): an empty/whitespace reply still
                # gets a numbered slot in the reduce prompt, just a literal
                # one instead of nothing.
                summaries.append(summary if summary.strip() else NO_SUMMARY)
            answer = await _b2_root_final(cfg, ep=ep, registry=registry,
                                            root_client=root_client,
                                            summaries=summaries,
                                            question=task.text)
        except (BudgetBreach, DispatchError, ServerRotationError) as exc:
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
