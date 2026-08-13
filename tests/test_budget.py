"""C5 BudgetEnforcer unit tests + mandatory hypothesis stateful suite
(spec ARCHITECTURE.md §5 C5, §5 Testing).

CONTROLLER RULING applied here (task-13 brief defect): the brief's
``BudgetMachine`` referenced a helper ``st_int(...)`` inside an ``@rule(...)``
decorator while defining ``st_int`` *below* the class. Decorators evaluate at
class-definition time, so that ordering raises ``NameError`` on import and
the suite cannot even collect. Fixed here by using ``st.integers(...)``
directly in the decorator instead of a forward-referenced helper.
"""
from __future__ import annotations

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from rlm.budget import BudgetEnforcer, Budgets
from rlm.errors import BudgetBreach, Outcome

B = Budgets(max_depth=1, max_subcalls=32, max_wall_clock_s=900,
            max_total_tokens=1_500_000, max_predict={"root": 1024, "leaf": 512})


def test_admission_refuses_when_reservation_would_exceed_cap():
    b = BudgetEnforcer(Budgets(1, 32, 900, 1000, {"leaf": 500}))
    b.admit(400, "leaf")
    with pytest.raises(BudgetBreach):
        b.admit(400, "leaf")  # 400+500 + 400+500 > 1000


def test_retried_call_counts_once_against_max_subcalls():
    b = BudgetEnforcer(B)
    r = b.admit(10, "leaf", call_id="c1")
    b.settle(r, 10, 5)
    r2 = b.admit(10, "leaf", call_id="c1")  # same logical call, retry
    b.settle(r2, 10, 5)
    assert b.subcalls_used == 1
    assert b.tokens_used == 30  # every attempt's tokens count


def test_root_window_breach_is_context_exhausted():
    b = BudgetEnforcer(B)
    with pytest.raises(BudgetBreach) as exc:
        b.note_root_usage(used=29_500, window=32_768)  # >= 90%
    assert exc.value.outcome == Outcome.CONTEXT_EXHAUSTED


def test_no_admissions_after_breach():
    b = BudgetEnforcer(Budgets(1, 1, 900, 1_000_000, {"leaf": 10}))
    b.settle(b.admit(1, "leaf", call_id="a"), 1, 1)
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="b")
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="c")  # still refusing, never warn-and-continue


def test_cancel_releases_the_reservation_without_settling_tokens():
    b = BudgetEnforcer(Budgets(1, 32, 900, 1000, {"leaf": 500}))
    r = b.admit(400, "leaf")
    assert b.reserved_total == 500
    b.cancel(r)
    assert b.reserved_total == 0
    assert b.tokens_used == 0  # a cancelled call never settles tokens
    # the freed reservation makes room for another admission
    b.admit(400, "leaf")


def test_on_breach_callback_fires_exactly_once():
    b = BudgetEnforcer(Budgets(1, 1, 900, 1_000_000, {"leaf": 10}))
    seen: list[BudgetBreach] = []
    b.on_breach(seen.append)
    b.settle(b.admit(1, "leaf", call_id="a"), 1, 1)
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="b")
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="c")
    assert len(seen) == 1
    assert seen[0].outcome == Outcome.BUDGET_KILL


class BudgetMachine(RuleBasedStateMachine):
    """The mandatory C5 stateful suite (spec §5 Testing)."""

    calls = Bundle("calls")

    def __init__(self):
        super().__init__()
        self.b = BudgetEnforcer(Budgets(1, 8, 900, 10_000, {"leaf": 100}))
        self.inflight = []
        self.breached = False

    @rule(target=calls, tokens=st.integers(min_value=1, max_value=200))
    def dispatch(self, tokens):
        try:
            r = self.b.admit(tokens, "leaf", call_id=f"c{len(self.inflight)}")
        except BudgetBreach:
            self.breached = True
            return None
        self.inflight.append(r)
        return r

    @rule(r=calls)
    def complete(self, r):
        if r is not None and r in self.inflight:
            self.inflight.remove(r)
            self.b.settle(r, r.prompt_tokens, 10)

    @rule(r=calls)
    def cancel(self, r):
        if r is not None and r in self.inflight:
            self.inflight.remove(r)
            self.b.cancel(r)

    @invariant()
    def reservations_never_exceed_cap(self):
        assert self.b.reserved_total <= self.b.budgets.max_total_tokens

    @invariant()
    def full_reserved_never_exceeds_cap(self):
        """Fix round 1: admit() actually gates on tokens_used + full_reserved
        + reservation <= max_total_tokens (spec §5 "running_total +
        reservation <= cap") -- full_reserved (prompt_tokens + max_predict
        summed over in-flight), not reserved_total's max_predict-only slice.
        That strong quantity was previously never property-tested and was
        true only "by construction" of admit(); property-test it directly."""
        assert self.b.full_reserved <= self.b.budgets.max_total_tokens

    @invariant()
    def overshoot_is_bounded_by_inflight_times_max_predict(self):
        bound = len(self.inflight) * self.b.budgets.max_predict["leaf"]
        assert self.b.reserved_total - self.b.tokens_used <= bound

    @invariant()
    def no_admissions_after_breach(self):
        if self.breached:
            with pytest.raises(BudgetBreach):
                self.b.admit(1, "leaf", call_id="post-breach")


TestBudgetMachine = BudgetMachine.TestCase
# derandomize=True: a fixed internal seed, not a wall-clock/OS-entropy one --
# this is what pins the stateful suite's example sequence for CI (brief's
# "Hypothesis profiles/seeds pinned in CI", spec §5 Testing).
TestBudgetMachine.settings = settings(max_examples=200, stateful_step_count=40,
                                      deadline=None, derandomize=True)
