"""§8 inference layer: exact, dependency-free, deterministic (house rule:
a headline p-value must be recomputable from the record; no scipy)."""
import math

from rlm.measure.stats import (fractional_score, needs_escalation, paired_bootstrap_ci,
                        sign_test_p, task_passes)


def test_sign_test_matches_hand_computed_binomial():
    # 5 discordant tasks, 5 wins 0 losses: two-sided p = 2 * 0.5^5 = 0.0625
    assert math.isclose(sign_test_p(5, 0), 0.0625)
    # symmetric
    assert math.isclose(sign_test_p(0, 5), 0.0625)
    # 4-1: two-sided p = 2 * (C(5,4)+C(5,5)) * 0.5^5 = 0.375
    assert math.isclose(sign_test_p(4, 1), 0.375)


def test_sign_test_degenerate_cases():
    assert sign_test_p(0, 0) == 1.0
    # perfectly balanced can exceed 1 by the two-sided doubling; must clamp
    assert sign_test_p(3, 3) == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    deltas = [1/3, 0.0, 2/3, 1/3, -1/3, 1.0, 0.0, 1/3]
    lo1, hi1 = paired_bootstrap_ci(deltas)
    lo2, hi2 = paired_bootstrap_ci(deltas)
    assert (lo1, hi1) == (lo2, hi2)          # pinned seed => reproducible
    mean = sum(deltas) / len(deltas)
    assert lo1 <= mean <= hi1


def test_bootstrap_ci_collapses_on_constant_deltas():
    assert paired_bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_task_passes_thirds_and_fifths():
    assert task_passes([True, True, False])            # 2/3 passes
    assert not task_passes([True, False, False])       # 1/3 fails
    assert task_passes([True, True, True, False, False])       # 3/5 passes
    assert not task_passes([True, True, False, False, False])  # 2/5 fails


def test_fractional_score():
    assert math.isclose(fractional_score([True, False, False]), 1/3)
    assert math.isclose(fractional_score([True] * 5), 1.0)


def test_escalation_band_is_exactly_plus1_to_plus3():
    assert not needs_escalation(0)
    assert all(needs_escalation(m) for m in (1, 2, 3))
    assert not needs_escalation(4)
    assert not needs_escalation(-1)   # a loss is a loss; no escalation rescues it
