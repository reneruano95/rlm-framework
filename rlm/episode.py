"""The episode runner — the composition root (spec §5, §6).

THIS IS THE ONLY MODULE PERMITTED TO IMPORT BOTH C4 AND THE ISOLATED
COMPONENTS. `tests/test_import_rules.py` enforces the rest of the dependency
rule; nothing enforces this one but review, so it is written here: if a second
module ever needs both, the design has drifted and the fix is to move the
wiring here, not to widen the lint list.

It owns exactly six things, and nothing else does:

  * the §4 startup handshake (weakened per D27 — see `assert_props`),
  * C2 materialisation of `context` and `chunks` INTO THE SANDBOX,
  * the root turn loop (render -> hash -> parse -> exec -> observe -> append),
  * the `final_answer` channel,
  * budget wiring (C5) and D22's kill ordering,
  * outcome determination (§6).

Three invariants shape every line below.

I2 — THE FULL CONTEXT NEVER ENTERS A MESSAGE ARRAY. It is `setvar`'d into the
sandbox and referenced by name. The root sees only `observation_view` output
capped by C3. The untruncated observation is stored as a blob
(`observation_full_ref`); the capped string is `steps.observation_view`. The
two together are what let a trace reconstruct both what the root KNEW and what
actually HAPPENED (§6).

THE FINAL ANSWER ARRIVES ONLY VIA `final_answer(value)`. Root prose is never
parsed for an answer — that is the one channel able to smuggle untruncated
context past C3. `_on_final_answer` below is the only place an answer is ever
accepted, and it is fed exclusively by the sandbox bridge.

D26 APPEND-ONLY. Every mutable thing (turn counter, remaining sub-calls, the
task text) goes into the NEWEST user message; nothing already sent is ever
rewritten. Measured: append-only reuses the prefix cache cleanly, and a
mid-conversation edit collapses reuse to the edit point. `compose_user_message`
is deliberately a pure function of values recoverable from the trace, because
`rlm replay` re-derives the same array offline and compares.

STEP INDICES ARE ASSIGNED HERE, NOT BY THE WRITER. C6 assigns `step_idx` in
commit order unless the caller supplies one; this runner supplies one, because
`parent_step_idx` on a leaf call must name the `repl_exec` that spawned it and
that step has not been committed yet when the call goes out. A consequence
worth stating: commit order is NOT `step_idx` order (a turn's leaf calls commit
before the turn's own row). That is exactly why §6 says causality lives in
`parent_step_idx`/`call_id` and never in `step_idx` adjacency.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm import trace as tracemod
from rlm.budget import Budgets as BudgetLimits
from rlm.budget import BudgetEnforcer
from rlm.chunker import ChunkConfig, split
from rlm.config import Config, config_snapshot
from rlm.context import load_context
from rlm.dispatcher import ServerClient, compose_leaf_user
from rlm.errors import (
    ActionType,
    Actor,
    BudgetBreach,
    ConfigError,
    DispatchError,
    Outcome,
    RlmError,
    SlotPoolExhausted,
    StepStatus,
)
from rlm.rootclient import RootConversation
from rlm.sandbox.manager import SandboxManager
from rlm.truncate import observation_view

# The kill code every scaffold-initiated TerminateJobObject carries. D10: an
# exit code is not an outcome channel -- `outcome_reason` comes from the kill
# reason, and this constant exists only so every kill site agrees.
KILL_CODE = 0xC5

# §6 outcome_reason conventions that are produced here rather than by a budget.
NO_FINAL_EMITTED = "no_final_emitted"
CHECKER_FAILED = "checker_failed"
OPERATOR_ABORT = "operator_abort"
SERVER_UNREACHABLE = "server_unreachable"
NO_CELL_EXTRACTED = "no_cell_extracted"
#: A planned slot-pool rotation could not be completed (§5 C4). Distinct from
#: `server_unreachable` on purpose: the leaf may be perfectly reachable and
#: still be the WRONG leaf -- a process that came back with different flags is
#: exactly what the re-run handshake exists to catch, and calling that
#: "unreachable" would hide it in the one column §8 reads to explain errors.
ROTATION_FAILED = "rotation_failed"

#: The reason a rotation ever fires. There is exactly one (§5 C4): pool
#: exhaustion. Named as a constant so the lifecycle log and this module cannot
#: drift, and so `grep slot_pool_exhausted` finds every site that can restart a
#: server.
SLOT_POOL_EXHAUSTED = "slot_pool_exhausted"

#: …and the reason a rotation is REFUSED: the pool ran out because every window
#: on it failed, not because every window on it was served. `SlotPoolExhausted`
#: alone cannot tell those apart, and treating them alike relaunches a FAILED
#: server while logging it as a planned rotation -- which is precisely what §5
#: C4 forbids, and the distinction that made rotation permissible at all.
SLOT_POOL_ERROR_DRAINED = "slot_pool_error_drained"

#: Termination guard on the rotate-and-retry loop for ONE call. A rotation
#: frees a whole pool (`--parallel` slots), so a single call needs a second one
#: only when other calls in the same wave took every slot first; needing 16 of
#: them means a fan-out wider than 16 x --parallel windows, which is a
#: structural problem the wall clock should not have to discover slowly.
MAX_ROTATIONS_PER_CALL = 16


# --------------------------------------------------------------------------- #
# task
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Task:
    """One task: the instruction, the corpus spec, and the declared category.

    `category` selects the strategy template DETERMINISTICALLY (I1: the model
    never chooses its own strategy), and an unknown category is refused by
    `PromptRegistry.render_root`, not defaulted.
    """

    task_id: str
    text: str
    context: Any = ""
    category: str = "default"
    answer: str | None = None
    checker: str = "auto"

    @property
    def task_hash(self) -> str:
        """§6: sha256 of the INSTRUCTION text. Corpus documents are hashed
        separately, in the benchmark manifest -- so the same question over a
        re-generated corpus keeps a stable task_hash."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Task":
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read task file {p}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"task file {p} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"task file {p} did not parse to an object")
        known = {"task_id", "text", "context", "category", "answer", "checker"}
        unknown = set(raw) - known - {"context_path", "fixtures"}
        if unknown:
            raise ConfigError(f"task file {p}: unknown keys {sorted(unknown)}")
        if "context_path" in raw:
            if "context" in raw:
                raise ConfigError(
                    f"task file {p}: give either 'context' or 'context_path', not both")
            # Relative to the TASK FILE, not the cwd: a task file is a portable
            # artifact and must not depend on where `rlm run` was invoked from.
            raw["context"] = {"path": str((p.parent / raw.pop("context_path")).resolve())}
        if "text" not in raw:
            raise ConfigError(f"task file {p}: 'text' (the instruction) is required")
        raw.pop("fixtures", None)
        raw.setdefault("task_id", p.stem)
        return cls(**raw)

    def check(self, value: Any) -> bool:
        """§6: `success` = final emitted AND the checker passes.

        `auto` means "contains, normalised" when the task declares an answer
        and "always passes" when it does not -- an ad-hoc `rlm run` with no
        declared answer is not a scored episode, and pretending otherwise
        would make every exploratory run a `fail`.
        """
        mode = self.checker
        if mode == "auto":
            mode = "none" if self.answer is None else "contains"
        if mode == "none":
            return True
        if self.answer is None:
            raise ConfigError(f"checker {mode!r} requires the task to declare an answer")
        # Delegated to the registry (§8's "checker fn" per task). `exact` and
        # `contains` keep byte-identical semantics there -- same normalisation,
        # same comparison -- so S1's recorded `contains` results still mean what
        # they said, while the benchmark gains checkers strict enough to reject
        # a hedge. An unknown name still raises rather than defaulting: a
        # typo'd checker silently becoming `contains` is exactly the permissive
        # failure §8 warns converts R5 confabulation into false passes.
        from rlm.checkers import check as _run_checker
        return _run_checker(mode, value, self.answer)


def _normalise(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def settled_tokens(attempts: list[dict[str, Any]]) -> tuple[int, int]:
    """Total tokens one logical call actually cost, summed over EVERY attempt.

    §5 C4's asymmetry, in one place: a retried call counts once against
    `max_subcalls` (dedupe by `call_id`) but every attempt's tokens count
    against `max_total_tokens`. Attempts that reported no usage contribute
    zero, so this is safe to run over a mixed list of ok/error/cancelled
    attempts.

    Module-level and pure so the arithmetic is testable without an episode.
    """
    return (sum(a.get("tokens_in") or 0 for a in attempts),
            sum(a.get("tokens_out") or 0 for a in attempts))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    outcome: Outcome
    reason: str | None
    final_answer: Any


# --------------------------------------------------------------------------- #
# §4 startup handshake (D27-weakened)
# --------------------------------------------------------------------------- #


def assert_props(props: dict, server_cfg, role: str) -> None:
    """Assert a live server matches the config it is supposed to be running.

    D27 — WHAT `/props` CANNOT TELL YOU. Cache types are NOT assertable here.
    Two servers launched with `-ctk q8_0` and `-ctk f16` produce `/props`
    responses that differ in exactly one field, `media_marker`, which is a
    per-process nonce and carries no cache information at all. §4's original
    "assert ... cache types" is therefore unimplementable against this
    endpoint, and this function does not pretend otherwise: cache-type
    verification comes from parsing the server's `-lv 4` launch stderr log
    (`rlm.cli.parse_launch_log`), and `rlm validate` reports it separately.

    Build info is RECORDED, not asserted: `ServerConfig` carries no pinned
    build id, so there is nothing to compare against. Its absence is still a
    failure -- a server that will not say what it is cannot be admitted to a
    measurement run (R11).
    """
    reported = props.get("model_path", "")
    if os.path.normcase(str(reported)) != os.path.normcase(str(server_cfg.model)):
        raise ConfigError(
            f"{role} server /props reports model_path={reported!r}, config says "
            f"{str(server_cfg.model)!r}. llama-server echoes the path it was "
            "launched with, verbatim; refusing to start.")
    slots = props.get("total_slots")
    if slots != server_cfg.parallel:
        raise ConfigError(
            f"{role} server /props reports total_slots={slots}, config says "
            f"parallel={server_cfg.parallel}; refusing to start.")
    expected_ctx = server_cfg.ctx // server_cfg.parallel
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    if n_ctx != expected_ctx:
        raise ConfigError(
            f"{role} server /props reports per-slot n_ctx={n_ctx}, config implies "
            f"{server_cfg.ctx}//{server_cfg.parallel}={expected_ctx}; refusing to start.")
    if not props.get("build_info"):
        raise ConfigError(
            f"{role} server /props carries no build_info; a server that will not "
            "identify its build cannot be admitted to a measured run (R11).")


async def handshake(client: ServerClient, server_cfg, role: str,
                     lifecycle: Any = None) -> dict:
    """Probe one server and refuse the run on any mismatch (§4)."""
    try:
        props = await client.props()
    except Exception as exc:  # noqa: BLE001 -- unreachable is one refusal among many
        if lifecycle is not None:
            lifecycle.event("server_health", role=role, state="unreachable",
                             error=repr(exc))
        raise ConfigError(f"{role} server /props is unreachable: {exc}") from exc
    assert_props(props, server_cfg, role)
    if lifecycle is not None:
        lifecycle.event("server_health", role=role, state="ok",
                         build_info=str(props.get("build_info", "")))
    return props


# --------------------------------------------------------------------------- #
# D26 message composition -- a PURE function of trace-recoverable values
# --------------------------------------------------------------------------- #


def no_cell_observation(cfg: Config) -> str:
    """The scaffold-authored observation fed back after an extraction miss.

    Generated from `scaffold.cell_extraction` so the extractor, the shipped
    prompt text and this correction can never disagree (Conflict 5).
    """
    ce = cfg.scaffold.cell_extraction
    langs = ", ".join(f"`{lang}`" for lang in ce.languages)
    which = "first" if ce.select == "first" else "last"
    return (
        "[scaffold]\n"
        "No code cell was found in that reply, so nothing ran and the REPL is "
        "unchanged.\n\n"
        f"Reply with exactly one fenced code block tagged {langs}; the {which} "
        "such block is the one that runs. For example:\n\n"
        "```repl\n"
        "print(len(chunks))\n"
        "```\n\n"
        "Prose is never read as an answer. Submit with final_answer(value)."
    )


def compose_user_message(*, turn: int, subcalls_remaining: int,
                          task_text: str | None = None,
                          observation: str | None = None) -> str:
    """The newest user message (D26). Everything mutable lives HERE.

    Every value in the trailer is recoverable from the trace alone -- the turn
    index from step order, `subcalls_remaining` from the distinct `call_id`s
    already logged -- because `rlm replay` calls THIS function to re-derive the
    array offline. Nothing wall-clock-shaped or token-count-shaped may ever go
    in here: it would make the state rule unverifiable without also making the
    check meaningless.
    """
    if (task_text is None) == (observation is None):
        raise ValueError("compose_user_message takes exactly one of task_text/observation")
    body = task_text if task_text is not None else observation
    return f"{body}\n\n[turn {turn}; sub-calls remaining: {subcalls_remaining}]"


# --------------------------------------------------------------------------- #
# C2 token counting: a sync counter over C4's async /tokenize
# --------------------------------------------------------------------------- #


class _TokenCounter:
    """`split()` is synchronous by design (C2 imports no LLM client) while
    `/tokenize` is an HTTP call, so the chunker runs on a worker thread and
    each count is marshalled back onto the episode's loop. Bridging the other
    way -- making C2 async -- would put an await inside the chunker and hand
    the dependency rule a hole to grow through.

    Cost, stated: the boundary search is O(log n) counts per chunk, each a
    round trip. On the real leaf that is tens of `/tokenize` calls for a
    multi-megabyte corpus, once per episode, off the critical path of any
    generation."""

    def __init__(self, dispatcher: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._dispatcher = dispatcher
        self._loop = loop

    def __call__(self, text: str) -> int:
        if not text:
            return 0
        future = asyncio.run_coroutine_threadsafe(
            self._dispatcher.count_tokens(text, role="leaf"), self._loop)
        return future.result()


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #


def resolve_chunk_ref(ref: Any, chunks: list[str]) -> str:
    """Turn a delegation-arm handle index into the chunk text it names.

    RANGE-CHECKED, NEVER CLAMPED. A ref the scaffold did not issue is a bug in
    the emitting cell; answering about chunk 0 instead would come back as an
    ordinary answer and be SCORED as one, which is the failure mode §8 cannot
    detect after the fact. The error goes back to the cell, where the root can
    act on it, exactly like the type errors beside it.

    `bool` is excluded explicitly because it is an `int` subclass in Python, and
    `chunk=True` resolving to chunk 1 is precisely the silent wrong answer this
    function exists to refuse.
    """
    if not isinstance(ref, int) or isinstance(ref, bool):
        raise RlmError(
            f"llm_query(chunk=...) handle must carry an integer index, got "
            f"{type(ref).__name__}")
    if not chunks:
        raise RlmError(
            "llm_query(chunk=...) got a handle but this episode has no chunks")
    if not 0 <= ref < len(chunks):
        raise RlmError(
            f"llm_query(chunk=...) handle index {ref} is outside the corpus "
            f"(0..{len(chunks) - 1})")
    return chunks[ref]


@dataclass
class _Breach:
    """Whatever terminated the episode, recorded ONCE at the moment it
    happened. §6 outcomes are fixed at the point of breach, never re-derived
    from later state -- a sandbox that dies *because* we killed it must not
    overwrite `wall_clock` with `sandbox_death`."""

    outcome: Outcome
    reason: str
    kill_reason: str = ""


class _EpisodeRun:
    def __init__(self, task: Task, cfg: Config, *, dispatcher, trace, lifecycle,
                 registry, props: dict, max_turns: int | None,
                 scaffold_instance_id: str, scaffold_git_sha: str,
                 benchmark_version: str | None,
                 process_manager: Any = None,
                 snapshot_extra: dict | None = None,
                 restrict_chunks: bool = False) -> None:
        self.task = task
        self.cfg = cfg
        #: Delegation arm: hand the sandbox opaque handles instead of chunk
        #: text, so `llm_query` is the only route to content. Defaults OFF --
        #: this same runner serves S1 and S3, and the arm must be inert unless
        #: the bench asks for it.
        self.restrict_chunks = restrict_chunks
        self._chunks: list[str] = []
        self.dispatcher = dispatcher
        # Who owns the leaf PROCESS (rlm.serverproc.ProcessManager), or None
        # when nobody does -- the servers were launched outside `rlm run`, and
        # a rotation is then impossible rather than optional.
        self.process_manager = process_manager
        self._rotations = 0
        self._rotation_lock = asyncio.Lock()
        self.trace = trace
        self.lifecycle = lifecycle
        self.registry = registry
        self.props = props
        self.max_turns = max_turns
        self.scaffold_instance_id = scaffold_instance_id
        self.scaffold_git_sha = scaffold_git_sha
        self.benchmark_version = benchmark_version
        self.snapshot_extra = snapshot_extra

        self.episode_id = str(uuid.uuid4())
        self.session = None
        self.enforcer = BudgetEnforcer(
            BudgetLimits(
                max_depth=cfg.scaffold.budgets.max_depth,
                max_subcalls=cfg.scaffold.budgets.max_subcalls,
                max_wall_clock_s=(
                    cfg.scaffold.budgets.restricted_max_wall_clock_s
                    if (restrict_chunks
                        and cfg.scaffold.budgets.restricted_max_wall_clock_s)
                    else cfg.scaffold.budgets.max_wall_clock_s),
                max_total_tokens=cfg.scaffold.budgets.max_total_tokens,
                max_identical_turns=cfg.scaffold.budgets.max_identical_turns,
                max_predict={"root": cfg.scaffold.budgets.max_predict.root,
                             "leaf": cfg.scaffold.budgets.max_predict.leaf},
            ),
            root_window_kill_fraction=cfg.scaffold.root_window_kill_fraction,
        )

        self._next_idx = 0
        self._parent_idx: int | None = None
        self._logged_attempts: set[tuple[str, int]] = set()
        self._breach: _Breach | None = None
        self._final_value: Any = None
        self._final_emitted = False
        self._final_parent: int | None = None
        self._final_ref: str | None = None
        # step_idx -> {root_view_hash, root_request_ref} for each root turn, so
        # the terminal `final` step can name its parent's already-written blob.
        self._turn_meta: dict[int, dict[str, str]] = {}
        self._finished = asyncio.Event()

    # -- step bookkeeping ---------------------------------------------------- #

    def _alloc(self) -> int:
        idx = self._next_idx
        self._next_idx += 1
        return idx

    def _put(self, row: dict[str, Any], blobs: dict[str, bytes] | None = None) -> None:
        row.setdefault("episode_id", self.episode_id)
        self.trace.put_step(row, blobs)

    # -- D22: one kill owner, one order -------------------------------------- #

    async def _trip(self, outcome: Outcome, reason: str, *, kill: bool = True) -> None:
        """Record the terminating fact and run D22's sequence exactly once.

        kill(reason, code) -> TerminateJobObject -> the bridge cancels every
        in-flight handler -> each cancelled handler writes its
        `status=cancelled` step. `drain()`/`aclose()` follow at the end of
        `execute()`/at the owner of the TraceLogger, in that order.
        """
        if self._breach is not None:
            return
        self._breach = _Breach(outcome, reason, kill_reason=reason)
        self._finished.set()
        if kill and self.session is not None:
            with contextlib.suppress(Exception):
                await self.session.kill(reason, KILL_CODE)

    # -- C5 wall clock -------------------------------------------------------- #

    async def _wall_clock_watchdog(self) -> None:
        """The clock has to be enforced by something OTHER than the turn loop:
        a cell that never returns (`while True: pass`) never gives the loop a
        chance to check, and that is precisely the case the budget exists for.
        """
        while not self._finished.is_set():
            try:
                self.enforcer.check_wall_clock()
            except BudgetBreach as breach:
                await self._trip(breach.outcome, breach.reason)
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._finished.wait(), timeout=0.1)

    # -- the sandbox's two outbound frames ------------------------------------ #

    async def _on_llm_query(self, payload: dict) -> str:
        """C4 dispatch under C4's semaphore, admitted by C5, logged by C6 --
        all of it in THIS process (I1). Every step written here carries
        `parent_step_idx` = the `repl_exec` that spawned it.

        The payload carries `chunk` and `question` as SEPARATE fields, and the
        composition into §4's `[chunk][question]` user segment happens
        scaffold-side (`rlm.dispatcher.compose_leaf_user`) -- the same string
        is then what C5 admits against and what C6 logs as `action_payload`,
        because it is the string that was actually sent.
        """
        payload = payload or {}
        question = payload.get("question")
        chunk = payload.get("chunk")
        # Delegation arm: the cell passed a handle, so the text is resolved
        # HERE and never existed on the sandbox side.
        ref = payload.get("chunk_ref")
        if ref is not None:
            chunk = resolve_chunk_ref(ref, self._chunks)
        # The bridge carries whatever JSON the cell wrote, so the types are
        # checked HERE rather than discovered as a TypeError inside C4. The
        # message goes back to the emitting cell, where the root can act on it.
        for name, value in (("question", question), ("chunk", chunk)):
            if value is not None and not isinstance(value, str):
                raise RlmError(
                    f"llm_query({name}=...) must be a string, got "
                    f"{type(value).__name__}")
        question = question or ""
        prompt = compose_leaf_user(question, chunk)
        role = payload.get("role") or "leaf"
        if role != "leaf":
            # I1: routing is the scaffold's. There is exactly one level below
            # the root (max_depth 1), so "leaf" is the only reachable role.
            raise RlmError(
                f"role={role!r} is not dispatchable; llm_query's role is fixed to "
                "'leaf' by the scaffold")
        parent = self._parent_idx
        call_id = str(uuid.uuid4())

        prompt_tokens = await self.dispatcher.count_tokens(prompt, role="leaf")
        try:
            reservation = self.enforcer.admit(prompt_tokens, "leaf", call_id)
        except BudgetBreach as breach:
            # Nothing was dispatched, so there is no attempt to log -- the
            # breach is the episode's outcome, not a step. Logging a rejected
            # step per refused call would also turn one runaway `gather` into
            # a hundred rows describing the same single fact.
            await self._trip(breach.outcome, breach.reason)
            raise

        try:
            answer = await self._dispatch_leaf(question, chunk, call_id)
        except asyncio.CancelledError:
            self._settle(reservation, call_id)
            self._log_attempts(call_id, parent, prompt)
            raise
        except Exception:
            self._settle(reservation, call_id)
            self._log_attempts(call_id, parent, prompt)
            raise
        self._settle(reservation, call_id)
        self._log_attempts(call_id, parent, prompt, answer=answer)
        return answer

    async def _dispatch_leaf(self, question: str, chunk: str | None,
                              call_id: str) -> str:
        """One leaf call, rotating the leaf server if its slot pool is spent.

        §5 C4 (v0.2.6): the R13 mitigation gives every window a never-reused
        slot, so a pool of `--parallel` slots is spent after `--parallel`
        windows -- 128 on the measured config, against 424 windows for a 200K
        corpus. Without a rotation the mitigation is inert. With one, the
        distinction that keeps §5 C4's original rule intact is PLANNED versus
        REACTIVE: this fires on `SlotPoolExhausted` and on nothing else, so a
        server that FAILED is still never restarted (that would mask the fault
        the trace exists to record -- `DispatchError` propagates untouched,
        below).

        `SlotPoolExhausted` ALONE IS NOT ENOUGH TO ESTABLISH THAT, and that
        was the hole: a pool drained by three consecutive dispatch failures
        raises the same exception as a pool drained by three answered windows,
        so a failing leaf was relaunched and logged as a planned rotation.
        C4 therefore records why each slot was consumed and this checks
        `pool_error_drained` first -- a generation that answered nothing ends
        the episode `outcome=error` rather than being relaunched.

        The retry re-dispatches the SAME `call_id`: a rotation does not make a
        new sub-call, it makes the same one land on a virgin slot, and §5 C4
        counts a re-dispatched call once against `max_subcalls`. C4 continues
        that call's `retry_idx` sequence so both the refusal and the answer
        survive in the trace.
        """
        for _ in range(MAX_ROTATIONS_PER_CALL + 1):
            try:
                return await self.dispatcher.query(
                    question, role="leaf", call_id=call_id, chunk=chunk,
                    # THE SEED COMES FROM THIS EPISODE'S CONFIG, PER CALL --
                    # never from whenever the dispatcher happened to be built.
                    # A bench run holds one leaf dispatcher across §8's three
                    # seeds (`rlm/bench.py` re-seeds the CONFIG per attempt),
                    # so a construction-time seed would decode all three
                    # replicates identically while `config_snapshot` recorded
                    # that they differed.
                    seed=self.cfg.scaffold.sampling.leaf.seed)
            except SlotPoolExhausted:
                if self.process_manager is None:
                    # Nobody owns the leaf process (it was launched outside
                    # `rlm run`), so there is nothing to rotate. The refusal
                    # reaches the cell unchanged -- which is the honest
                    # degradation, because the alternative the pool is
                    # protecting against is a silent wrap-around onto a slot
                    # that has held another document.
                    self.lifecycle.event(
                        "server_health", role="leaf", state="rotation_unavailable",
                        episode_id=self.episode_id, reason=SLOT_POOL_EXHAUSTED)
                    raise
                if getattr(self.dispatcher, "pool_error_drained", False):
                    # WHY the pool emptied decides whether this is a rotation
                    # at all. Every window on this generation failed, so the
                    # leaf is not a healthy server that ran out of slots -- it
                    # is a FAILED server, and §5 C4 has never permitted
                    # restarting one: doing so masks the fault the trace exists
                    # to record, and the lifecycle log would have called it a
                    # planned rotation. The episode ends instead.
                    self.lifecycle.event(
                        "server_health", role="leaf", state="rotation_refused",
                        episode_id=self.episode_id,
                        reason=SLOT_POOL_ERROR_DRAINED)
                    await self._trip(Outcome.ERROR, SLOT_POOL_ERROR_DRAINED)
                    raise
                rotation = await self._rotate_leaf()
                self._stamp_rotation(call_id, rotation)
        raise DispatchError(
            f"leaf call {call_id} could not obtain a virgin slot after "
            f"{MAX_ROTATIONS_PER_CALL} rotations; the fan-out is wider than "
            f"{MAX_ROTATIONS_PER_CALL} x --parallel windows (spec §5 C4)")

    async def _rotate_leaf(self) -> int:
        """Replace the healthy leaf process and resume on a virgin pool.

        Serialized: concurrent calls that all exhausted the pool queue here,
        and whoever gets in second finds the pool already refilled and returns
        without rotating again. The sequence is §5 C4's, in order, and none of
        it is optional:

          quiesce C4  -> no call may be mid-flight while the process it is
                         talking to is replaced;
          restart     -> the injected process manager's, never C4's;
          /props      -> §4's handshake, RE-RUN. A rotation that silently comes
                         back with different flags is exactly the failure the
                         handshake exists to catch, and at `total_slots` it
                         would make C4 request slot ids the server silently
                         reassigns onto used slots (R13);
          rotate_pool -> a new process means a new pool;
          resume      -> on every path, including failure, so parked calls take
                         a refusal instead of hanging.

        The wall clock keeps running throughout, deliberately: §5 C4 says the
        rotation's time is included in the episode's measured time (2 rotations
        ≈ 13.4 s per 200K corpus), and §8 excludes between-ARM relaunch, never
        this. The C5 watchdog therefore ends an episode whose rotation outlives
        its budget, which is the correct reading of an overrun.
        """
        async with self._rotation_lock:
            if not self.dispatcher.restart_required:
                # Someone rotated while this call queued on the lock; the pool
                # already has virgin slots, and rotating again would throw away
                # a whole process's worth of them.
                return self._rotations
            rotation = self._rotations + 1
            started = time.perf_counter()
            self.lifecycle.event("server_health", role="leaf", state="rotating",
                                  episode_id=self.episode_id, rotation=rotation,
                                  reason=SLOT_POOL_EXHAUSTED)
            # `rotating()` is quiesce -> ... -> resume with the reopen in a
            # `finally` INSIDE C4, so no path out of this block -- including a
            # cancellation landing on the quiesce itself -- can leave the gate
            # closed with nobody to reopen it. That failure mode is a hang, not
            # a refusal: parked calls would produce no step and no outcome.
            async with self.dispatcher.rotating():
                try:
                    await self.process_manager.restart()
                    await self._rehandshake_leaf()
                    self.dispatcher.rotate_pool()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- any failure here is
                    # terminal for the episode: the alternative to a completed
                    # rotation is reusing a slot that has held another document.
                    self.lifecycle.event("server_health", role="leaf",
                                          state="rotation_failed",
                                          episode_id=self.episode_id,
                                          rotation=rotation, error=repr(exc))
                    await self._trip(Outcome.ERROR, ROTATION_FAILED)
                    raise
            self._rotations = rotation
            self.lifecycle.event(
                "server_health", role="leaf", state="rotated",
                episode_id=self.episode_id, rotation=rotation,
                duration_ms=round((time.perf_counter() - started) * 1000, 1))
            return rotation

    async def _rehandshake_leaf(self) -> None:
        """§4's handshake against the process that just came up."""
        leaf_cfg = self.cfg.servers.leaf
        client = ServerClient(f"http://127.0.0.1:{leaf_cfg.port}",
                               timeout=self.cfg.scaffold.retries.per_call_timeout_s)
        try:
            self.props["leaf"] = await handshake(client, leaf_cfg, "leaf",
                                                  self.lifecycle)
        finally:
            await client.aclose()

    def _stamp_rotation(self, call_id: str, rotation: int) -> None:
        """Mark the step that triggered a rotation (§5 C4).

        The lifecycle log carries the event, but the S3 gate runs with that log
        deleted, so the trace has to hold the fact by itself -- and only the
        step ties a rotation to the window whose slot request could not be
        served. Stamped on the refusal attempt, which C4 has already recorded
        by the time this runs; `_log_attempts` then writes it out.
        """
        if rotation < 1:
            return          # nothing rotated (another call had already done it)
        attempts = self._attempts(call_id)
        if attempts:
            attempts[-1]["server_rotation"] = rotation

    def _attempts(self, call_id: str) -> list[dict]:
        return [s for s in self.dispatcher.steps if s.get("call_id") == call_id]

    def _settle(self, reservation, call_id: str) -> None:
        """Release the hold and charge what the call ACTUALLY cost.

        §5 C4: a retried call counts ONCE against `max_subcalls`, but EVERY
        attempt's tokens count against `max_total_tokens` -- the whole point of
        that asymmetry is that retries are not free. Charging only the
        successful attempt would under-bill a call that failed twice by two
        attempts' prefill, which is exactly the case the rule exists for.

        The same summation runs on every exit path, success or not: a failed or
        timed-out attempt that got far enough to report usage spent those tokens
        server-side whether or not anyone got an answer. When no attempt
        reported any usage (the ordinary cancellation case -- a closed stream
        never yields a final event) this settles zero, which is precisely
        `cancel()`'s effect, so the two paths need not diverge.
        """
        tokens_in, tokens_out = settled_tokens(self._attempts(call_id))
        self.enforcer.settle(reservation, tokens_in, tokens_out)

    def _log_attempts(self, call_id: str, parent: int | None, prompt: str,
                       answer: str | None = None) -> None:
        """One trace step per dispatch ATTEMPT (C4's contract), idempotent so a
        cancel path and a later flush cannot double-write the same attempt."""
        for attempt in self._attempts(call_id):
            key = (call_id, attempt.get("retry_idx", 0))
            if key in self._logged_attempts:
                continue
            self._logged_attempts.add(key)
            row = {k: v for k, v in attempt.items() if k in tracemod.STEP_COLS}
            row.update(step_idx=self._alloc(), parent_step_idx=parent, depth=1,
                        actor=Actor.LEAF, action_type=ActionType.LLM_CALL,
                        action_payload=prompt)
            blobs: dict[str, bytes] = {}
            rendered = attempt.get("rendered")
            if rendered is not None:
                # The same instrument the root path uses (`root_view_hash` is
                # already on the row): the EXACT rendered request, stored so a
                # gate-(a) failure is investigable at all -- prefix drift and
                # slot eviction produce an identical symptom, and only the
                # bytes plus the prefix's token length tell them apart. The
                # `meta` stream carries what §6 has no column for and needs
                # none: which call form was used (so a gate can score only the
                # calls that supplied `chunk=`) and the measured prefix length,
                # next to the `tokens_cached` on this very row.
                blobs["root_request_ref"] = tracemod.pack_blob({
                    "rendered": rendered.encode("utf-8"),
                    "meta": json.dumps(
                        {"layout": attempt.get("layout"),
                         "prefix_tokens": attempt.get("prefix_tokens")},
                        ensure_ascii=True, separators=(",", ":")).encode("ascii"),
                })
            if answer is not None and attempt.get("status") == StepStatus.OK:
                # C6 must not truncate the stored record: the leaf's answer is
                # ground truth even though the ROOT never sees it directly (it
                # lands in a REPL variable), so it is a blob, not a view.
                blobs["observation_full_ref"] = answer.encode("utf-8", "replace")
            self._put(row, blobs or None)

    async def _on_final_answer(self, value: Any) -> None:
        """THE ONLY PLACE AN ANSWER IS EVER ACCEPTED.

        Raising here is not an error path: the child surfaces a refusal into
        the emitting cell's stderr, so a root that submits twice, or submits
        into an already-terminal episode, is TOLD so rather than silently
        overwriting a recorded answer.
        """
        if self._breach is not None:
            raise RlmError("episode is already terminal; final_answer refused")
        if self._final_emitted:
            raise RlmError("this episode already has a final answer; refused")
        self._final_emitted = True
        self._final_value = value
        self._final_parent = self._parent_idx
        # DELIBERATELY does not set `_finished`. That event stops the
        # wall-clock watchdog, and the cell that submitted this answer is
        # still running -- `final_answer` is synchronous and returns into the
        # cell. A cell that submits and then loops forever would otherwise
        # leave the episode with no clock at all, waiting on an `exec_cell`
        # that never returns: precisely the hang C5 exists to make impossible.
        # The turn loop returns on its own once the cell finishes; the
        # watchdog is cancelled there.

    # -- the episode ---------------------------------------------------------- #

    async def execute(self, manager: SandboxManager, root: ServerClient) -> EpisodeResult:
        cfg = self.cfg
        loop = asyncio.get_running_loop()

        # The clock starts BEFORE C2, not after: chunking does O(log n)
        # `/tokenize` round trips per boundary against the leaf server, and on
        # a multi-megabyte corpus that is real wall time the episode spent. Ran
        # untimed, it would be work the wall-clock budget cannot see and the
        # trace cannot attribute -- so it is charged, and its duration is
        # recorded so the cost stops being a guess.
        self.enforcer.start_clock()
        t_chunk = time.perf_counter()
        context_text = load_context(self.task.context)
        counter = _TokenCounter(self.dispatcher, loop)
        # Off-loop: every count is an HTTP round trip on the real leaf.
        context_tokens = await self.dispatcher.count_tokens(context_text, role="leaf")
        chunk_cfg = ChunkConfig(size_tokens=cfg.scaffold.chunk.size_tokens,
                                 overhead_tokens=cfg.scaffold.chunk.overhead_tokens,
                                 snap_to_boundary=cfg.scaffold.chunk.snap_to_boundary,
                                 snap_tolerance=cfg.scaffold.chunk.snap_tolerance,
                                 # §7 #2: `stride < size` makes `chunks`
                                 # OVERLAPPING windows, not a partition --
                                 # `context` remains the only non-repeating
                                 # view of the corpus.
                                 stride_tokens=cfg.scaffold.chunk.stride_tokens)
        chunks = await asyncio.to_thread(split, context_text, chunk_cfg, counter)
        # R13 detection (§5 C4): C4 holds the corpus so it can run the
        # foreign-string check on every leaf answer -- an identifier absent
        # from the chunk that was sent and present in another chunk is a leak,
        # at zero model cost. Handed over ONCE, here, because this is the only
        # place the whole corpus is known; without it C4 records "not checked"
        # rather than clean, which is the honest degradation.
        self.dispatcher.set_corpus(chunks)
        # The delegation arm resolves `chunk_ref` indices against this list,
        # scaffold-side (I1). Kept whether or not the arm is on, so the two
        # paths differ in one place only -- what `setvar` sends.
        self._chunks = chunks
        # Recorded in the trace (via config_snapshot), not the lifecycle log:
        # this is episode data, and I4 makes the trace store its sole home. No
        # allow-listed lifecycle kind covers it, and inventing one would make
        # the log the second source of truth it exists not to be.
        chunk_ms = round((time.perf_counter() - t_chunk) * 1000, 1)

        async with manager.session(self.episode_id, cfg) as session:
            self.session = session
            self.lifecycle.event("sandbox_spawn", episode_id=self.episode_id,
                                  pid=session.pid)
            # The row goes in HERE -- after the spawn, so `sandbox_pid` is real
            # (§6 recovery reaps by pid and a NULL there is unreapable), and
            # with a NULL outcome, which is what makes tombstoning possible.
            # The window between spawn and insert is covered by the Job's
            # KILL_ON_JOB_CLOSE: a scaffold that dies in it takes the sandbox
            # with it, so the row that was never written describes nothing that
            # is still running.
            self.trace.open_episode({
                "episode_id": self.episode_id,
                "task_id": self.task.task_id,
                "task_hash": self.task.task_hash,
                "tokenized_task_len": context_tokens,
                "dry_run": cfg.scaffold.dispatcher == "mock",
                "scaffold_instance_id": self.scaffold_instance_id,
                "sandbox_pid": session.pid,
                "config_snapshot": self._snapshot(context_text, chunks, chunk_ms),
                "scaffold_git_sha": self.scaffold_git_sha,
                "benchmark_version": self.benchmark_version,
            })

            session.on_llm_query(self._on_llm_query)
            session.on_final_answer(self._on_final_answer)

            # I2: the corpus crosses into the sandbox HERE and nowhere else.
            # In the delegation arm it does not cross at all: `chunks` becomes a
            # count the child turns into opaque handles, and `context` is
            # withheld, because leaving the whole corpus readable would make the
            # handles decorative. `llm_query` is then the only route to content,
            # which is the thing that arm exists to price.
            if self.restrict_chunks:
                await session.setvar("context", "")
                await session.setvar("chunks", {"__opaque_chunks__": len(chunks)})
            else:
                await session.setvar("context", context_text)
                await session.setvar("chunks", chunks)

            conv = RootConversation(root, cfg,
                                     system=self.registry.render_root(
                                         self.task.category,
                                         restricted=self.restrict_chunks))
            # The clock is already running (it started before C2 chunking); the
            # watchdog is what makes it enforceable against a cell that never
            # returns.
            watchdog = asyncio.create_task(self._wall_clock_watchdog(),
                                            name="c5-wall-clock")
            try:
                await self._turn_loop(conv)
            except asyncio.CancelledError:
                # Route the operator abort through D22's kill path HERE, while
                # the session still exists. Unwinding first and killing later
                # is not merely untidy: `close()` takes the graceful path when
                # no kill was issued, and a sandbox wedged by construction
                # (the exact case someone hits Ctrl-C for) then costs the full
                # shutdown grace twice -- 30 s measured, before the job is
                # terminated at all.
                await self._trip(Outcome.BUDGET_KILL, OPERATOR_ABORT)
                raise
            finally:
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog

        return await self._close()

    async def _turn_loop(self, conv: RootConversation) -> None:
        cfg = self.cfg
        cap = cfg.scaffold.truncation_cap_chars
        window = cfg.scaffold.root.window_tokens
        conv.append_user(compose_user_message(
            turn=1, subcalls_remaining=self._subcalls_remaining(),
            task_text=self.task.text))
        turn = 0
        while self._breach is None and not self._final_emitted:
            turn += 1
            if self.max_turns is not None and turn > self.max_turns:
                await self._trip(Outcome.FAIL, NO_FINAL_EMITTED)
                return
            try:
                rt = await conv.turn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- root server fault
                self.lifecycle.event("server_health", role="root", state="failed",
                                      error=repr(exc))
                await self._trip(Outcome.ERROR, SERVER_UNREACHABLE)
                return

            idx = self._alloc()
            request_blob = tracemod.pack_blob({
                # The rendered string is the state-rule instrument: its sha256
                # IS `root_view_hash`, so replay rehashes this stream byte for
                # byte. The message array rides along so mode (ii) can re-POST
                # it to /apply-template offline-derived -- see rlm.cli.
                "messages": json.dumps(conv.messages[:-1], ensure_ascii=True,
                                        separators=(",", ":")).encode("ascii"),
                "rendered": rt.rendered.encode("utf-8"),
            })
            base = {
                "step_idx": idx, "actor": Actor.ROOT,
                "action_type": ActionType.REPL_EXEC,
                # §6 "code or prompt (full)": for a root turn the full action is
                # the model's reply; the cell is the config-pinned extraction of
                # it, and replay re-derives the cell rather than storing it twice.
                "action_payload": rt.raw,
                "root_view_hash": rt.view_hash,
                "tokens_in": rt.usage.tokens_in, "tokens_out": rt.usage.tokens_out,
                "tokens_cached": rt.usage.cache_n, "slot_id": rt.usage.slot_id,
                "t_first_byte": rt.usage.t_first_byte, "t_end": tracemod.utc_now(),
                "latency_prefill_ms": rt.usage.prompt_ms,
                "latency_decode_ms": rt.usage.predicted_ms,
            }
            self._turn_meta[idx] = {
                "root_view_hash": rt.view_hash,
                "root_request_ref": tracemod.blob_rel(self.episode_id, idx,
                                                       "root_request_ref"),
            }

            if rt.cell is None:
                # Extraction miss: a normal, correctable observation, never an
                # episode error. It costs the root a turn (and therefore root
                # window and wall clock) but NOT a sub-call -- nothing was
                # dispatched, so there is nothing for max_subcalls to count.
                view = no_cell_observation(cfg)
                self._put({**base, "status": StepStatus.REJECTED,
                            "error_detail": NO_CELL_EXTRACTED,
                            "observation_view": view},
                           {"root_request_ref": request_blob})
                await self._note_root_usage(rt, window)
                if self._breach is not None:
                    return
                conv.append_user(compose_user_message(
                    turn=turn + 1, subcalls_remaining=self._subcalls_remaining(),
                    observation=view))
                continue

            self._parent_idx = idx
            try:
                out = await self.session.exec_cell(rt.cell)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- sandbox death/desync
                reason = getattr(self.session, "kill_reason", None) or "sandbox_death"
                self._put({**base, "status": StepStatus.CANCELLED,
                            "error_detail": tracemod.safe_text(str(exc)),
                            "observation_view": None},
                           {"root_request_ref": request_blob})
                if self._final_emitted:
                    # The answer crossed the bridge before the sandbox died, so
                    # it counts -- but the terminal step still has to exist, or
                    # the trace shows an accepted final with nothing recording
                    # it and `final_answer_ref` stays NULL.
                    self._log_final(idx)
                await self._trip(Outcome.ERROR, reason)
                return

            view = observation_view(out, cap)          # C3, scaffold-side (I1)
            self._put({**base, "status": StepStatus.OK, "observation_view": view},
                       {"root_request_ref": request_blob,
                        "observation_full_ref": _full_observation(out)})
            # The final is logged BEFORE the window is charged: an answer that
            # already arrived is not undone by the turn that carried it
            # crossing the kill fraction.
            if self._final_emitted:
                self._log_final(idx)
                return
            await self._note_root_usage(rt, window)
            if self._breach is not None:
                return
            conv.append_user(compose_user_message(
                turn=turn + 1, subcalls_remaining=self._subcalls_remaining(),
                observation=view))

    async def _note_root_usage(self, rt, window: int) -> None:
        """C5 root-window accounting from SERVER-REPORTED usage (§5): at >= the
        kill fraction of the root window the episode ends deterministically as
        `context_exhausted`. A multi-turn flail loop can fill 32K well inside
        the wall clock, and overflow must be an outcome, never an accident.

        Root turns are NOT admitted through `enforcer.admit()`, so root tokens
        do not count against `max_total_tokens`. Adjudicated as correct, not a
        gap: §5 scopes that budget to dispatch admission, the root's own spend
        is already bounded by the 32K window plus this 90% kill, and §8's cost
        scorecard reads `steps.tokens_in`/`tokens_out` -- which every root turn
        writes -- so cost reporting stays complete. Admitting root turns would
        also charge each one against `max_subcalls`, which counts any unseen
        `call_id`."""
        used = (rt.usage.tokens_in or 0) + (rt.usage.tokens_out or 0)
        try:
            self.enforcer.note_root_usage(used, window)
        except BudgetBreach as breach:
            # D22: ANY BudgetBreach runs the kill path, root-window included.
            await self._trip(breach.outcome, breach.reason)

    def _log_final(self, parent_idx: int) -> None:
        """The terminal step, logged AFTER its parent turn so `action_type`
        ordering in the trace reads the way the episode actually ran (the
        handler itself fires mid-cell, before the turn's own row exists).

        It carries the parent's `root_view_hash` and points at the parent's
        already-written request blob: the final IS attributable to that exact
        rendered request, and a second copy of a 32K-token render per episode
        buys nothing.
        """
        value = self._final_value
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        parent = self._final_parent if self._final_parent is not None else parent_idx
        parent_row = self._turn_meta.get(parent, {})
        idx = self._alloc()
        self._final_ref = tracemod.blob_rel(self.episode_id, idx,
                                             "observation_full_ref")
        self._put({
            "step_idx": idx, "parent_step_idx": parent, "actor": Actor.ROOT,
            "action_type": ActionType.FINAL, "status": StepStatus.OK,
            "action_payload": text,
            "root_view_hash": parent_row.get("root_view_hash"),
            "root_request_ref": parent_row.get("root_request_ref"),
            "observation_view": observation_view(
                _as_cell_output(text), self.cfg.scaffold.truncation_cap_chars),
            "t_end": tracemod.utc_now(),
        }, {"observation_full_ref": text.encode("utf-8", "replace")})

    def _subcalls_remaining(self) -> int:
        return max(0, self.cfg.scaffold.budgets.max_subcalls
                   - self.enforcer.subcalls_used)

    def _snapshot(self, context_text: str, chunks: list[str],
                   chunk_ms: float = 0.0) -> dict:
        """§6 config_snapshot: the validated config plus everything about THIS
        run that a replay or a scoring query needs and the schema has no column
        for -- prompt hashes, the `/props` responses, the chat-template hash,
        and the task's own instruction text (without which offline replay
        cannot re-derive the first user message)."""
        root_props = self.props.get("root") or {}
        template = root_props.get("chat_template") or ""
        base_extra = {
            "prompt_hashes": self.registry.hashes(),
            "pinned_prompt_hashes": self.cfg.pinned_prompt_hashes(),
            "props": self.props,
            "chat_template_sha256": hashlib.sha256(
                template.encode("utf-8")).hexdigest(),
            "task": {
                "task_id": self.task.task_id,
                "text": self.task.text,
                "category": self.task.category,
                "checker": self.task.checker,
                "answer": self.task.answer,
                "context_chars": len(context_text),
                "chunks": len(chunks),
                # R11: an arm invisible to the snapshot is an arm §8 cannot
                # score, and this flag changes what the sandbox can see.
                "restrict_chunks": self.restrict_chunks,
                # C2's real cost: O(log n) /tokenize round trips per boundary.
                # Charged against the wall clock (start_clock precedes it) and
                # recorded here so it is measurable rather than guessed at.
                "chunk_ms": chunk_ms,
            },
        }
        if self.snapshot_extra:
            # Merged LAST, so a caller-supplied key (e.g. bench identity)
            # never shadows anything this method itself records.
            extra = self.snapshot_extra
            assert not (set(extra) & set(base_extra))
            base_extra.update(extra)
        return config_snapshot(self.cfg, base_extra)

    def record_abort(self, reason: str) -> None:
        """Terminal attribution for the operator-abort path, SYNCHRONOUSLY.

        Everything here is deliberately await-free. Once the task is being
        cancelled, any `await` re-raises `CancelledError` immediately -- so an
        abort path that awaited would leave the row with a NULL outcome, and
        the episode would tombstone as `orphaned_at_recovery` at the next
        startup instead of carrying the `operator_abort` reason §5 promises.
        `close_episode` is a queue put, not I/O, so it lands regardless; the
        TraceLogger's owner drains it on `aclose()`.

        The sandbox is already dead by the time this runs: the session context
        manager kills it on the way out.
        """
        if self._breach is None:
            self._breach = _Breach(Outcome.BUDGET_KILL, reason, kill_reason=reason)
            self._finished.set()
        outcome, why = self._outcome()
        self.trace.close_episode(self.episode_id, outcome, why, self._final_ref)

    async def _close(self) -> EpisodeResult:
        outcome, reason = self._outcome()
        self.trace.close_episode(self.episode_id, outcome, reason,
                                  getattr(self, "_final_ref", None))
        # D22's tail: drain so the partial trace is durable before anything
        # else runs. `aclose()` belongs to whoever owns the TraceLogger -- one
        # logger serves a whole `rlm bench` run, and closing it here would end
        # the process's single writer after the first episode.
        await self.trace.drain()
        return EpisodeResult(self.episode_id, outcome, reason, self._final_value)


    def _outcome(self) -> tuple[Outcome, str | None]:
        """§6 outcome semantics, in the one place they are decided.

        An ACCEPTED final wins outright, and that ordering is load-bearing
        rather than generous: `_on_final_answer` refuses the moment `_breach`
        is set, so `_final_emitted` being true means the answer arrived FIRST.
        Checking the breach first would let a kill that happened strictly after
        a valid answer -- a wall-clock tick, or the same turn crossing the root
        window -- erase an episode that had genuinely finished.

        Otherwise a breach recorded at the moment it happened always wins: an
        episode killed for `wall_clock` whose sandbox then dies is
        `budget_kill`, not `error`.
        """
        if self._final_emitted:
            if self.task.check(self._final_value):
                return Outcome.SUCCESS, None
            return Outcome.FAIL, CHECKER_FAILED
        if self._breach is not None:
            return self._breach.outcome, self._breach.reason
        return Outcome.FAIL, NO_FINAL_EMITTED


def _full_observation(out) -> bytes:
    """The untruncated observation, as ONE blob with its streams named. C6
    must not truncate the stored record; C3 truncates only the root's view."""
    return tracemod.pack_blob({
        "stdout": out.stdout.encode("utf-8", "replace"),
        "stderr": out.stderr.encode("utf-8", "replace"),
        "repr": out.repr_.encode("utf-8", "replace"),
        "traceback": out.traceback.encode("utf-8", "replace"),
    })


def _as_cell_output(text: str):
    from rlm.truncate import CellOutput

    return CellOutput(stdout=text)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


async def run_episode(task: Task, cfg: Config, *, dispatcher, trace, lifecycle,
                       sandbox_manager: SandboxManager | None = None,
                       root_client: ServerClient | None = None,
                       probe: bool = True, max_turns: int | None = None,
                       scaffold_instance_id: str = "",
                       scaffold_git_sha: str = "",
                       benchmark_version: str | None = None,
                       process_manager: Any = None,
                       snapshot_extra: dict | None = None,
                       restrict_chunks: bool = False) -> EpisodeResult:
    """Run ONE episode end to end and return its §6 outcome.

    `dispatcher`, `trace` and `lifecycle` are injected because their lifetimes
    are longer than an episode's (a bench run reuses all three); the sandbox
    manager and the root client are built here when not supplied, and torn
    down here when they were.

    The §4 handshake runs BEFORE the episode row is opened: a server that does
    not match config refuses the run outright, and a refused run is not an
    episode -- it leaves no row to explain. The leaf is probed only when the
    dispatcher is real; a dry run has no leaf server to probe, and inventing
    one would make `dispatcher: mock` require the very thing it exists to
    avoid.
    """
    registry = cfg.prompt_registry().load()
    root = root_client or ServerClient(
        f"http://127.0.0.1:{cfg.servers.root.port}",
        timeout=cfg.scaffold.retries.per_call_timeout_s)
    manager = sandbox_manager or SandboxManager()
    owns_root = root_client is None
    owns_manager = sandbox_manager is None

    try:
        props: dict[str, Any] = {}
        if probe:
            props["root"] = await handshake(root, cfg.servers.root, "root", lifecycle)
            if cfg.scaffold.dispatcher == "real":
                leaf = ServerClient(f"http://127.0.0.1:{cfg.servers.leaf.port}",
                                     timeout=cfg.scaffold.retries.per_call_timeout_s)
                try:
                    props["leaf"] = await handshake(leaf, cfg.servers.leaf, "leaf",
                                                     lifecycle)
                finally:
                    await leaf.aclose()

        run = _EpisodeRun(task, cfg, dispatcher=dispatcher, trace=trace,
                           lifecycle=lifecycle, registry=registry, props=props,
                           max_turns=max_turns,
                           scaffold_instance_id=scaffold_instance_id,
                           scaffold_git_sha=scaffold_git_sha,
                           benchmark_version=benchmark_version,
                           process_manager=process_manager,
                           snapshot_extra=snapshot_extra,
                           restrict_chunks=restrict_chunks)
        try:
            return await run.execute(manager, root)
        except asyncio.CancelledError:
            # Operator Ctrl-C routes through C5's path (spec §5): an
            # attributable reason, the outcome recorded, never a mid-write
            # death. See `record_abort` for why nothing here is awaited.
            lifecycle.event("operator_abort", episode_id=run.episode_id)
            run.record_abort(OPERATOR_ABORT)
            raise
    finally:
        if owns_root:
            await root.aclose()
        if owns_manager:
            manager.close()
