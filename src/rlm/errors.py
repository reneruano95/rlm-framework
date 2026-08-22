"""Shared types. Imports nothing from rlm — every component may import this."""
from __future__ import annotations

from enum import StrEnum


# --------------------------------------------------------------------------- #
# the CLI's exit contract
# --------------------------------------------------------------------------- #
# Here rather than in `src/rlm/cli.py` because `src/rlm/trace/export.py` and `src/rlm/trace/replay.py`
# return them too, and importing them back out of the composition root would be
# a cycle. This module imports nothing from rlm, so everything may have them.
EXIT_OK = 0
EXIT_FAILED = 1        # the command itself failed to complete
EXIT_REFUSED = 2       # config/handshake/invariant refusal
EXIT_MISMATCH = 3      # replay found a discrepancy


class Outcome(StrEnum):
    """episodes.outcome (spec §6)."""

    SUCCESS = "success"
    FAIL = "fail"
    BUDGET_KILL = "budget_kill"
    CONTEXT_EXHAUSTED = "context_exhausted"
    ERROR = "error"


class StepStatus(StrEnum):
    """steps.status (spec §6)."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ActionType(StrEnum):
    """steps.action_type (spec §6)."""

    REPL_EXEC = "repl_exec"
    LLM_CALL = "llm_call"
    FINAL = "final"


class Actor(StrEnum):
    """steps.actor (spec §6)."""

    ROOT = "root"
    LEAF = "leaf"


class RlmError(Exception):
    """Base for all scaffold errors."""


class ConfigError(RlmError):
    """Config failed schema or cross-field validation; the run refuses to start."""


class SandboxError(RlmError):
    """The sandbox interpreter died, refused a cell, or could not be spawned."""


class DispatchError(RlmError):
    """A model-server call failed after its retry budget."""


class TransportError(DispatchError):
    """A model-server call never got an answer: the connection was refused,
    reset, or timed out, or the body would not parse (spec §5 C4).

    THE CLASS THAT KEEPS `httpx` INSIDE C4. Every `ServerClient` method maps
    the HTTP library's own exceptions onto this before they can leave the
    dispatcher, because `rlm.cli`'s exit-code taxonomy is written in terms of
    `RlmError`: an exception the scaffold cannot name is deliberately allowed
    to escape as an uncaught traceback (it is a bug), while an attributable
    failure becomes `refused: ...` and exit 2 with a `--resume` hint. A raw
    `httpx.ConnectError` is an attributable failure wearing a bug's clothes,
    and it ended a 15/16-cell S4 smoke run on the last cell.

    A `DispatchError` subclass so every `except DispatchError` handler in the
    arms, the episode runner and C5 keeps working unchanged. It is separate
    from its parent for one reason: it is the failure class that is worth
    RETRYING on an idempotent request, and `LLMDispatcher.count_tokens` retries
    exactly this and nothing else (a `/tokenize` that answered "0 tokens for
    non-empty input", or an HTTP 4xx, is a fault that a second identical
    request cannot fix).
    """


class PreflightFailed(DispatchError):
    """A leaf call could not be BUILT: `/apply-template` or the pre-flight
    `/tokenize` failed, so nothing was ever dispatched and no slot was burned.

    Distinct from a plain `DispatchError` because it is the one dispatch
    failure the scaffold may itself have caused: a slot-pool rotation replaces
    the leaf process, and a call whose pre-flight was talking to the old one
    dies here through no fault of its own. C4 re-runs the pre-flight against
    the new process in exactly that case, and in no other -- a pre-flight that
    failed with no rotation in sight is a server-death report and stays one
    (spec §5 C4).
    """


class EnvelopeParseError(DispatchError):
    """The leaf answered, repeatedly, with something that is not a valid
    envelope (spec §5, the S2 leaf-envelope A/B).

    THE STRUCTURED ERROR THE ROOT BRANCHES ON, and deliberately its own class:
    a dead server and a leaf that cannot emit JSON are both `DispatchError`, but
    they have opposite remedies -- one ends the episode, the other is a fact
    about this chunk and this question that the root can route around (ask
    again, ask differently, or treat the window as unanswered). Collapsing them
    would make "the envelope costs episodes" indistinguishable from "the server
    fell over" in the A/B.

    `raw` is the last output that failed to parse, verbatim: without it a run
    reports a count of envelope failures with no way to see what the leaf
    actually emitted, and the MALFORMED column would be unauditable.
    """

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class SlotPoolExhausted(DispatchError):
    """C4's never-reused slot pool has no virgin slot left for a NEW window,
    so the leaf server must be restarted before another window is served
    (spec §10 R13, §5 C4).

    Raised rather than wrapping around: reusing a slot that has held another
    document is the R13 defect itself, and wrapping silently would reintroduce
    it invisibly -- the scaffold would believe it held a virgin slot."""


class SlotMismatch(DispatchError):
    """The leaf answered on a slot other than the one C4 requested.

    An out-of-range `id_slot` is silently reassigned with HTTP 200 (measured:
    asked for 200 on a 128-slot server, got 72 -- `milestones/s2/R13-mitigations.md`
    §4.5), so this is a contaminated answer, not a warning: the slot that
    served it may have held other documents."""


class PrefixDrift(DispatchError):
    """The rendered system head changed under a live target (spec §7 #3 a1, R3).

    §4's prefix contract is that the head is ONE constant string for a target's
    lifetime: every cache measurement, the `cache_n == N_resident - ub - 4`
    identity, and the 311-token gate all assume it. If its sha256 moves, every
    number taken after the move describes a different prompt than the numbers
    taken before it, so this is never retried -- drawing again cannot un-change
    the prefix -- and it fails the episode with `outcome_reason=prefix_drift`
    rather than degrading quietly."""


class ServerRotationError(RlmError):
    """A planned slot-pool rotation could not be carried out (spec §5 C4).

    Raised by whoever owns the server process — never by C4, which starts and
    stops nothing. It is an error rather than a warning because the only
    alternative to a completed rotation is reusing a slot that has held another
    document, which is R13's defect: the scaffold must fail the episode, not
    carry on believing its pool is virgin.
    """


class BudgetBreach(RlmError):
    """A C5 budget was breached. Carries the outcome the episode must record."""

    def __init__(self, outcome: Outcome, reason: str) -> None:
        super().__init__(f"{outcome}: {reason}")
        self.outcome = outcome
        self.reason = reason
