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


class BudgetBreach(RlmError):
    """A C5 budget was breached. Carries the outcome the episode must record."""

    def __init__(self, outcome: Outcome, reason: str) -> None:
        super().__init__(f"{outcome}: {reason}")
        self.outcome = outcome
        self.reason = reason
