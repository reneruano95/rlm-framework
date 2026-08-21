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

import json
from pathlib import Path

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


FIXTURES = Path(__file__).parent / "fixtures" / "repetition"


def _guarded(max_identical_turns: int = 3) -> BudgetEnforcer:
    return BudgetEnforcer(Budgets(1, 32, 900, 1_000_000, {"root": 1024, "leaf": 512},
                                  max_identical_turns=max_identical_turns))


def test_identical_turns_correct_at_max_minus_one_and_kill_at_max():
    """Spec §5 C5 (v0.3.16): the SAME (cell, observation) pair repeating is a
    budget. At max-1 consecutive occurrences the scaffold corrects; at max it
    kills as budget_kill/max_identical_turns -- the existing outcome, the
    existing reason convention."""
    b = _guarded(3)
    assert b.note_turn("print(1)", "[stdout]\n1") is False      # first occurrence
    assert b.note_turn("print(1)", "[stdout]\n1") is True       # 2nd == max-1: correct
    with pytest.raises(BudgetBreach) as exc:
        b.note_turn("print(1)", "[stdout]\n1")                  # 3rd == max: kill
    assert exc.value.outcome == Outcome.BUDGET_KILL
    assert exc.value.reason == "max_identical_turns"


def test_a_different_cell_or_a_different_observation_resets_the_count():
    """CONTROLLER RULING (task-2 brief defect): the brief's version called
    note_turn a third time on ("print(2)", "[stdout]\\n3") before this final
    assert, making it the pair's 3rd consecutive occurrence -- which, at
    max_identical_turns=3, is exactly what test_identical_turns_correct_at_
    max_minus_one_and_kill_at_max (above) proves raises BudgetBreach, not
    returns True. The extra call is dropped so this exercises the 2nd
    occurrence after the reset, mirroring lines 1-2's unasserted-then-True
    shape for the "different cell" reset three lines up."""
    b = _guarded(3)
    b.note_turn("print(1)", "[stdout]\n1")
    assert b.note_turn("print(1)", "[stdout]\n1") is True
    assert b.note_turn("print(2)", "[stdout]\n2") is False       # different cell: reset
    b.note_turn("print(2)", "[stdout]\n2")
    assert b.note_turn("print(2)", "[stdout]\n3") is False       # same cell, new output: reset
    assert b.note_turn("print(2)", "[stdout]\n3") is True        # counting again from the reset


def test_cells_are_compared_stripped_and_views_exactly():
    b = _guarded(3)
    b.note_turn("print(1)\n", "v")
    assert b.note_turn("  print(1)", "v") is True                # whitespace around the cell is not a difference
    b2 = _guarded(3)
    b2.note_turn("print(1)", "v")
    assert b2.note_turn("print(1)", "v ") is False               # the observation is compared byte for byte


def test_zero_disables_the_identical_turns_budget():
    b = _guarded(0)
    for _ in range(50):
        assert b.note_turn("print(1)", "v") is False


def test_cap_two_kills_on_the_first_repeat_and_never_corrects():
    """Review ruling (2026-08-21): the correction exists only when a repeat
    can precede the kill. At cap 2 the first repeat IS the kill, and a
    fresh pair must never be reported as a repeat."""
    b = _guarded(2)
    assert b.note_turn("a", "v") is False
    assert b.note_turn("b", "w") is False
    assert b.note_turn("c", "x") is False          # distinct pairs: never a correction
    with pytest.raises(BudgetBreach) as exc:
        b.note_turn("c", "x")                       # first repeat at cap 2: kill, no correction
    assert exc.value.reason == "max_identical_turns"


def test_no_turns_are_noted_after_a_breach():
    b = _guarded(2)
    b.note_turn("x", "v")
    with pytest.raises(BudgetBreach):
        b.note_turn("x", "v")
    with pytest.raises(BudgetBreach):
        b.note_turn("y", "w")                                    # still refusing, never warn-and-continue


@pytest.mark.parametrize("episode, onset_turn", [("9d9e47fb", 9), ("0c1c397d", 5)])
def test_the_recorded_loops_are_killed_at_the_third_identical_turn(episode, onset_turn):
    """The two production loops (s4/RESULTS-dflash2-rlm-only.md), replayed
    through the enforcer: the correction lands on the first repeat and the
    kill on the second, i.e. turn onset+2 instead of turn 79 / 116.

    CONTROLLER RULING (task-2 brief defect): the brief wrote
    `exc.value.reason` inside a plain `except BudgetBreach as exc:` clause.
    `.value` is an `ExceptionInfo` attribute that only exists when the
    exception is caught via `pytest.raises(...) as exc`; a plain `except`
    binds `exc` to the exception instance itself, which carries `.reason`
    directly (rlm/errors.py). Fixed to `exc.reason`."""
    turns = json.loads((FIXTURES / f"{episode}.json").read_text(encoding="utf-8"))["turns"]
    b = _guarded(3)
    corrected_at = killed_at = None
    for t in turns:
        try:
            if b.note_turn(t["cell"], t["observation_view"]) and corrected_at is None:
                corrected_at = t["turn"]
        except BudgetBreach as exc:
            killed_at = t["turn"]
            assert exc.reason == "max_identical_turns"
            break
    assert corrected_at == onset_turn + 1
    assert killed_at == onset_turn + 2
