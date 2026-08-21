"""C5 — BudgetEnforcer (spec ARCHITECTURE.md §5).

Per-episode limits from config. Termination is deterministic and
unconditional: the enforcer never warns and continues past a breach, and
never accepts a request to extend a cap (spec §5 *Must not*). Once a breach
happens, every subsequent `.admit()`/`.note_root_usage()`/`.check_wall_clock()`
call re-raises the SAME `BudgetBreach` that first tripped it — the outcome is
fixed at the moment of breach, not re-derived from later state.

Two counters are tracked, deliberately kept separate — this is what makes
the mandatory hypothesis stateful suite's overshoot invariant provable
(spec §5 Testing):

- `tokens_used` — SETTLED actuals. Grows only via `.settle()`, by exactly
  `actual_in + actual_out`. A retried call settles once per attempt, so
  every attempt's tokens land here (spec §5 C4: "every attempt's tokens
  count against max_total_tokens").
- `reserved_total` — the sum of the OUTPUT portion (`max_predict[role]`) of
  every currently in-flight (admitted, not yet settled/cancelled)
  reservation. The pre-flight prompt-token count is exact (measured via
  C4's /tokenize before admission), so it carries no uncertainty and is not
  counted here; `max_predict` is the only part of a reservation that is a
  worst-case guess. Because every in-flight reservation contributes exactly
  `max_predict[role]`, `reserved_total` is always exactly `sum(max_predict
  for each in-flight call)` — which is what makes "worst-case overshoot
  bounded by in-flight x max_predict" (spec §5 C5 *Admission*) a provable
  invariant rather than an aspiration.

  The admission GATE — whether a new call fits under `max_total_tokens` —
  uses a third, private running total that DOES include the pre-flight
  prompt tokens of every in-flight reservation, plus already-settled
  `tokens_used`: i.e. the full worst-case commitment to date. That number is
  intentionally not what `.reserved_total` exposes; `.reserved_total`
  exposes only the part that determines the overshoot bound.

A retried call (the same `call_id` re-admitted) counts once against
`max_subcalls`; every attempt still settles its own tokens independently.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

from rlm.errors import BudgetBreach, Outcome


@dataclass(frozen=True)
class Budgets:
    """Per-episode limits (spec §5 C5). Built from config, immutable."""

    max_depth: int = 1
    max_subcalls: int = 32
    max_wall_clock_s: int = 900
    max_total_tokens: int = 1_500_000
    max_predict: dict[str, int] = field(default_factory=dict)
    #: v0.3.16: the same (cell, observation) pair repeating on consecutive
    #: root turns is a budget. 0 disables. At max-1 the scaffold corrects,
    #: at max it kills -- measured (s2/REPLAY-LOOP-AB.md): once a cell has
    #: repeated once the root re-emits it ~64% of the time, once it has
    #: repeated a few times ~92%, and it never calls final_answer from there.
    max_identical_turns: int = 0


@dataclass(frozen=True)
class Reservation:
    """An admitted, not-yet-settled/cancelled dispatch admission.

    `id` is unique per `.admit()` call — even for retries sharing a
    `call_id` — so two in-flight reservations for the same logical call are
    never mistaken for each other by `.settle()`/`.cancel()`.
    """

    id: int
    call_id: str | None
    role: str
    prompt_tokens: int
    max_predict: int

    @property
    def reservation_tokens(self) -> int:
        """The full reservation: pre-flight prompt tokens + role max_predict."""
        return self.prompt_tokens + self.max_predict


class BudgetEnforcer:
    """Admission control + wall-clock + root-window accounting for one episode.

    *Must not* (spec §5): warn and continue past a breach; accept a model
    request for a cap extension. Raising `max_depth` above 1 requires
    evidence this project does not yet have (spec §5) — this enforcer does
    not itself gate depth (the scaffold does, by never spawning a call at
    depth > `budgets.max_depth`); `max_depth` is carried on `Budgets` as the
    config-level record of that limit.
    """

    def __init__(self, budgets: Budgets, *, root_window_kill_fraction: float = 0.90) -> None:
        self.budgets = budgets
        self._root_window_kill_fraction = root_window_kill_fraction

        self._tokens_used = 0
        self._predict_reserved = 0   # exposed as .reserved_total
        self._full_reserved = 0      # private: prompt + max_predict, for the admission gate
        self._subcalls_used = 0
        self._seen_call_ids: set[str] = set()
        self._inflight: dict[int, Reservation] = {}
        self._next_id = itertools.count(1)

        self._breached = False
        self._breach_outcome: Outcome | None = None
        self._breach_reason: str | None = None
        self._callbacks: list[Callable[[BudgetBreach], None]] = []

        self._clock_start: float | None = None

        self._last_turn_key: tuple[str, str] | None = None
        self._identical_run = 0

    # ------------------------------------------------------------------ #
    # breach plumbing
    # ------------------------------------------------------------------ #

    def on_breach(self, callback: Callable[[BudgetBreach], None]) -> None:
        """Register a callback fired exactly once, the moment a breach first
        happens (never again for later, redundant refusals)."""
        self._callbacks.append(callback)

    def _ensure_not_breached(self) -> None:
        if self._breached:
            assert self._breach_outcome is not None and self._breach_reason is not None
            raise BudgetBreach(self._breach_outcome, self._breach_reason)

    def _breach(self, outcome: Outcome, reason: str) -> None:
        """Trip a NEW breach (only called when not already breached) and raise."""
        self._breached = True
        self._breach_outcome = outcome
        self._breach_reason = reason
        exc = BudgetBreach(outcome, reason)
        for cb in self._callbacks:
            cb(exc)
        raise exc

    # ------------------------------------------------------------------ #
    # counters
    # ------------------------------------------------------------------ #

    @property
    def subcalls_used(self) -> int:
        return self._subcalls_used

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def reserved_total(self) -> int:
        return self._predict_reserved

    @property
    def full_reserved(self) -> int:
        """Sum of `prompt_tokens + max_predict[role]` over every currently
        in-flight reservation -- the FULL commitment `admit()` actually gates
        on (`tokens_used + full_reserved + new_reservation <= max_total_tokens`).
        Exposed read-only so the property suite can assert the cap directly
        against the quantity the gate enforces, instead of trusting that by
        construction (fix round 1: `reserved_total` alone, being the
        max_predict-only slice, doesn't exercise this)."""
        return self._full_reserved

    # ------------------------------------------------------------------ #
    # admission
    # ------------------------------------------------------------------ #

    def admit(self, prompt_tokens: int, role: str, call_id: str | None = None) -> Reservation:
        """Admit a dispatch call, or raise `BudgetBreach`.

        Reservation = pre-flight prompt tokens + `role`'s `max_predict`
        (spec §5 *Admission*). Refused when the admission gate
        `running_total + reservation > max_total_tokens`, when a NEW
        `call_id` would push `subcalls_used` past `max_subcalls`, or when
        already breached.
        """
        self._ensure_not_breached()

        max_predict = self.budgets.max_predict[role]
        full_amount = prompt_tokens + max_predict

        prospective_full = self._tokens_used + self._full_reserved + full_amount
        if prospective_full > self.budgets.max_total_tokens:
            self._breach(Outcome.BUDGET_KILL, "max_total_tokens")

        is_new_call = call_id is None or call_id not in self._seen_call_ids
        if is_new_call and self._subcalls_used + 1 > self.budgets.max_subcalls:
            self._breach(Outcome.BUDGET_KILL, "max_subcalls")

        self._full_reserved += full_amount
        self._predict_reserved += max_predict
        if is_new_call:
            self._subcalls_used += 1
            if call_id is not None:
                self._seen_call_ids.add(call_id)

        reservation = Reservation(id=next(self._next_id), call_id=call_id, role=role,
                                   prompt_tokens=prompt_tokens, max_predict=max_predict)
        self._inflight[reservation.id] = reservation
        return reservation

    def _release(self, reservation: Reservation) -> None:
        if self._inflight.pop(reservation.id, None) is None:
            return  # already settled/cancelled -- idempotent, not a second breach source
        self._full_reserved -= reservation.reservation_tokens
        self._predict_reserved -= reservation.max_predict

    def settle(self, reservation: Reservation, actual_in: int, actual_out: int) -> None:
        """Release `reservation`'s hold and record its real cost.

        Runs even after a breach: the scaffold must be able to settle
        in-flight calls during breach cleanup (spec §5 *On breach*).
        """
        self._release(reservation)
        self._tokens_used += actual_in + actual_out

    def cancel(self, reservation: Reservation) -> None:
        """Release `reservation`'s hold without it ever having consumed
        tokens (spec §5 *On breach*: "cancelled calls are logged as steps
        with status=cancelled" — no tokens were spent). Runs even after a
        breach, for the same reason `.settle()` does.
        """
        self._release(reservation)

    # ------------------------------------------------------------------ #
    # root window + wall clock
    # ------------------------------------------------------------------ #

    def note_root_usage(self, used: int, window: int) -> None:
        """Root-window accounting (spec §5 *Root window*): at >= the kill
        fraction (default 90%) of the root window, terminate deterministically
        as `context_exhausted`."""
        self._ensure_not_breached()
        if window <= 0:
            return
        if used / window >= self._root_window_kill_fraction:
            self._breach(Outcome.CONTEXT_EXHAUSTED, "root_window")

    def note_turn(self, cell: str, view: str) -> bool:
        """v0.3.16 `max_identical_turns`: count consecutive root turns whose
        (cell, observation) pair is identical. Returns True when a scaffold
        correction is due (the pair has now occurred max-1 times in a row);
        raises BudgetBreach(budget_kill, "max_identical_turns") at max.

        `cell` is compared stripped (fence whitespace is not a decision);
        `view` is the C3 observation BEFORE any scaffold note is appended --
        the caller must pass the un-annotated view, or the note it appended
        last turn would make every repeat look different and the budget would
        never fire. 0 disables.
        """
        self._ensure_not_breached()
        cap = self.budgets.max_identical_turns
        if cap <= 0:
            return False
        key = (cell.strip(), view)
        self._identical_run = self._identical_run + 1 if key == self._last_turn_key else 1
        self._last_turn_key = key
        if self._identical_run >= cap:
            self._breach(Outcome.BUDGET_KILL, "max_identical_turns")
        return self._identical_run == cap - 1

    def start_clock(self) -> None:
        self._clock_start = time.monotonic()

    def check_wall_clock(self) -> None:
        self._ensure_not_breached()
        if self._clock_start is None:
            return
        elapsed = time.monotonic() - self._clock_start
        if elapsed >= self.budgets.max_wall_clock_s:
            self._breach(Outcome.BUDGET_KILL, "wall_clock")
