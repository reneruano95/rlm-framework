"""§8's checker-validity precondition, as tests.

  "each checker's unit tests include >=3 authored plausible-but-wrong answers
   that must fail, plus normalization edge cases (permissive checkers convert
   R5 confabulation into false passes in every arm)."

The near-misses are the point. A checker that accepts a hedged answer -- "the
key is either A or B", "candidates: A, B, C" -- turns the leaf's measured
false-positive behaviour into a PASS, in every arm at once, and the S4 verdict
then measures the checker rather than the roots. So each checker here is tested
against answers that a confabulating model actually produces, not against
strings chosen to make it look good.
"""
from __future__ import annotations

import pytest

from rlm.measure.checkers import CHECKERS, check, near_miss_suite

UUID_A = "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"
UUID_B = "48e81295-9489-33be-cc30-430d702be6c3"


# --------------------------------------------------------------------------- #
# uuid_exact


@pytest.mark.parametrize("answer", [
    UUID_A,
    f"  {UUID_A}  ",
    f"{UUID_A}.",
    f"The custody key is {UUID_A}",
    UUID_A.upper(),
])
def test_uuid_exact_accepts_the_key_however_it_is_framed(answer):
    assert check("uuid_exact", answer, UUID_A)


@pytest.mark.parametrize("answer", [
    UUID_B,                                            # a real key, wrong one
    f"either {UUID_A} or {UUID_B}",                    # hedged between two
    f"candidates: {UUID_A}, {UUID_B}, {UUID_A}",       # a shortlist
    "7311d8a3-c2ce-4f44-bed4-d57b1e2feb8",             # truncated by one char
    "NONE",
    "",
])
def test_uuid_exact_rejects_the_near_misses(answer):
    """The hedges matter most: `contains` passes three of these, and every one
    of them is a confabulation this project has actually measured."""
    assert not check("uuid_exact", answer, UUID_A)


# --------------------------------------------------------------------------- #
# int_exact


@pytest.mark.parametrize("answer", ["1234", " 1234 ", "1234.", "**1234**",
                                    "The total is 1234", "1,234"])
def test_int_exact_accepts_the_number_however_it_is_framed(answer):
    assert check("int_exact", answer, "1234")


@pytest.mark.parametrize("answer", [
    "1235",                                  # off by one
    "of the 24 records, the total is 1234",  # TWO integers: which is the answer?
    "somewhere between 1200 and 1300",
    "1234 or 1243",
    "NONE",
])
def test_int_exact_rejects_the_near_misses(answer):
    """'of the 24 records, the total is 1234' is the interesting rejection: a
    last-integer rule would PASS it, and would then also pass 'the total is
    1234 out of 24 records' as 24. Requiring one unambiguous integer is the
    only rule that does not depend on where the model puts the sentence."""
    assert not check("int_exact", answer, "1234")


def test_int_exact_is_indifferent_to_thousands_separators_both_ways():
    assert check("int_exact", "1,234", "1234")
    assert check("int_exact", "1234", "1,234")


# --------------------------------------------------------------------------- #
# set_exact


def test_set_exact_accepts_any_order_and_any_common_separator():
    want = "alpha, beta, gamma"
    assert check("set_exact", "gamma\nalpha\nbeta", want)
    assert check("set_exact", "alpha, beta, gamma", want)
    assert check("set_exact", "- alpha\n- beta\n- gamma", want)


@pytest.mark.parametrize("answer", [
    "alpha, beta",                    # missing one
    "alpha, beta, gamma, delta",      # one too many
    "alpha, beta, gamma, gamma",      # duplicate padding to the right count
    "alpha beta gamma",               # not separated -- one item, not three
])
def test_set_exact_rejects_the_near_misses(answer):
    assert not check("set_exact", answer, "alpha, beta, gamma")


# --------------------------------------------------------------------------- #
# name_exact


def test_name_exact_accepts_surrounding_punctuation_and_case():
    for a in ["Zanelade Holkerath", "  zanelade holkerath ",
              '"Zanelade Holkerath"', "Zanelade Holkerath."]:
        assert check("name_exact", a, "Zanelade Holkerath")


@pytest.mark.parametrize("answer", [
    "Zanelade Holkerith",              # one letter off
    "Holkerath",                       # surname only
    "Zanelade Holkerath or Marn Vell",  # hedged
    "The custodian is not named in the register",
])
def test_name_exact_rejects_the_near_misses(answer):
    assert not check("name_exact", answer, "Zanelade Holkerath")


# --------------------------------------------------------------------------- #
# the registry itself


def test_every_registered_checker_ships_a_near_miss_suite():
    """§8 requires >=3 authored plausible-but-wrong answers PER CHECKER. Keeping
    them next to the checker -- rather than only in this file -- means a new
    checker cannot be added without them, and the benchmark manifest can record
    that the requirement was met."""
    for name in CHECKERS:
        suite = near_miss_suite(name)
        assert len(suite) >= 3, f"{name} ships {len(suite)} near-misses, needs >=3"
        for wrong, want in suite:
            assert not check(name, wrong, want), (
                f"{name}: near-miss {wrong!r} PASSED against {want!r}")


def test_an_unknown_checker_is_refused_rather_than_defaulted():
    from rlm.errors import ConfigError
    with pytest.raises(ConfigError, match="unknown checker"):
        check("vibes", "anything", "anything")


def test_the_legacy_checkers_still_behave(  ):
    """`exact` and `contains` predate this registry and S1's tasks use
    `contains`. They must keep their exact old semantics -- normalised
    whitespace, casefolded -- or the S1 record stops meaning what it said."""
    assert check("exact", "  Hello   World ", "hello world")
    assert not check("exact", "hello world!", "hello world")
    assert check("contains", "the answer is hello world, I think", "hello world")
    assert not check("contains", "the answer is hello", "hello world")


def test_int_exact_does_not_read_an_integer_out_of_a_decimal():
    """A trailing full stop is framing; a decimal point is not. '1234.' is the
    integer 1234 with punctuation, '12.5' contains no integer answer at all --
    and neither does the '89' inside a UUID."""
    assert check("int_exact", "1234.", "1234")
    assert not check("int_exact", "12.5", "12")
    assert not check("int_exact", "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89", "89")
