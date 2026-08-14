"""Shared types. Imports nothing from rlm — every component may import this."""
from __future__ import annotations

from enum import StrEnum


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
    asked for 200 on a 128-slot server, got 72 -- `s2/R13-mitigations.md`
    §4.5), so this is a contaminated answer, not a warning: the slot that
    served it may have held other documents."""


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
