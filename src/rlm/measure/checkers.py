"""Programmatic answer checkers for §8's frozen benchmark.

§8 requires each task to ship a checker function, and requires each checker's
unit tests to include "**>=3 authored plausible-but-wrong answers that must
fail**, plus normalization edge cases", with the reason stated plainly:
*permissive checkers convert R5 confabulation into false passes in every arm*.

That reason is not hypothetical here. This project has measured, repeatedly,
that the leaf answers unanswerable questions with a real identifier lifted from
somewhere plausible (§7 #2's 95% false-positive rate; `milestones/s2/DISTANCE.md`). A
`contains`-style checker passes exactly that failure whenever the model hedges
-- "either A or B", "candidates: A, B, C" -- and it passes it for EVERY arm, so
the S4 verdict would be measuring the checker rather than the roots.

So the rule these checkers follow is: **an answer must be unambiguous to pass.**
Framing is forgiven (leading prose, markdown bold, a trailing full stop, case,
thousands separators); ambiguity is not. Two distinct UUIDs is a fail even when
one of them is right, because a shortlist is not an answer.

Each checker declares its own near-miss suite next to it, so a new checker
cannot be added without the evidence §8 demands, and `near_miss_suite()` lets
the benchmark manifest record that the precondition was met.
"""
from __future__ import annotations

import re
from typing import Callable

from rlm.errors import ConfigError

__all__ = ["CHECKERS", "check", "near_miss_suite", "normalise"]

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
# A signed integer, optionally with thousands separators. The lookarounds do
# two different jobs and both are needed:
#   (?<![\w.-])  do not start mid-token -- no digits out of a UUID, an
#                identifier, or the fractional half of a decimal.
#   (?!\w|\.\d)  do not end mid-token, but DO allow a trailing sentence period:
#                "1234." is an integer with punctuation, "12.5" is not an
#                integer at all. An earlier `(?![\w.])` conflated the two and
#                rejected "1234." -- caught by the framing tests, which is what
#                they are for.
_INT_RE = re.compile(r"(?<![\w.-])[+-]?\d{1,3}(?:,\d{3})+(?!\w|\.\d)"
                     r"|(?<![\w.-])[+-]?\d+(?!\w|\.\d)")
_ITEM_SPLIT_RE = re.compile(r"[\n;,]+|^\s*[-*•]\s*", re.M)


def normalise(value: object) -> str:
    """The normalisation `Task.check` has always used: collapse all runs of
    whitespace to one space, then casefold. Kept identical, and imported rather
    than re-implemented, so S1's `contains` results keep meaning what they said.
    """
    return " ".join(str(value).split()).casefold()


def _strip_frame(s: str) -> str:
    """Remove the packaging a model puts around a bare answer, and nothing more:
    surrounding quotes, markdown emphasis, and trailing sentence punctuation."""
    s = s.strip()
    s = re.sub(r"^[\"'`*_\s]+|[\"'`*_\s]+$", "", s)
    return s.rstrip(".!?").strip()


# --------------------------------------------------------------------------- #
# checkers


def _exact(got: str, want: str) -> bool:
    return normalise(got) == normalise(want)


def _contains(got: str, want: str) -> bool:
    return normalise(want) in normalise(got)


def _uuid_exact(got: str, want: str) -> bool:
    """Exactly one DISTINCT uuid in the answer, and it is the expected one.

    Distinct, not total: a model that repeats the same key twice has still
    given one answer. Two different keys is a shortlist, and a shortlist is a
    refusal to answer wearing an answer's clothes."""
    found = {m.lower() for m in _UUID_RE.findall(got or "")}
    return len(found) == 1 and found.pop() == normalise(want).strip()


def _int_exact(got: str, want: str) -> bool:
    """Exactly one DISTINCT integer in the answer, equal to the expected one.

    The alternative rule -- "take the last integer" -- looks tolerant and is
    actually arbitrary: it passes "of the 24 records, the total is 1234" and
    fails the same sentence written the other way round ("the total is 1234,
    out of 24 records"), so it scores sentence order rather than arithmetic.
    Benchmark questions ask for a bare integer; one integer is the answer."""
    found = {int(m.replace(",", "")) for m in _INT_RE.findall(got or "")}
    try:
        target = int(str(want).replace(",", "").strip())
    except ValueError:
        raise ConfigError(f"int_exact expects an integer answer, got {want!r}")
    return len(found) == 1 and found.pop() == target


def _split_items(s: str) -> list[str]:
    parts = [_strip_frame(p) for p in _ITEM_SPLIT_RE.split(s or "")]
    return [normalise(p) for p in parts if p and p.strip()]


def _set_exact(got: str, want: str) -> bool:
    """The answer's items, as a SET, equal the expected items as a set.

    A set and not a list because §8's aggregation questions ask "which X",
    where order carries no information. Duplicates collapse, which is why the
    count is compared too: 'alpha, beta, gamma, gamma' must not pass as three
    distinct items when one is padding."""
    g, w = _split_items(got), _split_items(want)
    return bool(w) and set(g) == set(w) and len(set(g)) == len(g)


def _name_exact(got: str, want: str) -> bool:
    """A proper name, with framing forgiven but nothing else. Deliberately not
    `contains`: 'Holkerath' is a wrong answer to "what is the custodian's FULL
    name", and 'X or Y' is a hedge."""
    return normalise(_strip_frame(got)) == normalise(_strip_frame(want))


CHECKERS: dict[str, Callable[[str, str], bool]] = {
    "exact": _exact,
    "contains": _contains,
    "uuid_exact": _uuid_exact,
    "int_exact": _int_exact,
    "set_exact": _set_exact,
    "name_exact": _name_exact,
}

# §8: ">=3 authored plausible-but-wrong answers that must fail" PER CHECKER.
# Held here, beside the implementations, so a checker cannot be registered
# without them and so the benchmark manifest can record the precondition as met.
_NEAR_MISSES: dict[str, list[tuple[str, str]]] = {
    "exact": [("hello world!", "hello world"),
              ("hello  worl", "hello world"),
              ("world hello", "hello world")],
    "contains": [("the answer is hello", "hello world"),
                 ("HELLOWORLD", "hello world"),
                 ("hell world", "hello world")],
    "uuid_exact": [
        ("48e81295-9489-33be-cc30-430d702be6c3",
         "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"),
        ("either 7311d8a3-c2ce-4f44-bed4-d57b1e2feb89 or "
         "48e81295-9489-33be-cc30-430d702be6c3",
         "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"),
        ("7311d8a3-c2ce-4f44-bed4-d57b1e2feb8",
         "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"),
        ("NONE", "7311d8a3-c2ce-4f44-bed4-d57b1e2feb89"),
    ],
    "int_exact": [("1235", "1234"),
                  ("of the 24 records, the total is 1234", "1234"),
                  ("somewhere between 1200 and 1300", "1234"),
                  ("1234 or 1243", "1234")],
    "set_exact": [("alpha, beta", "alpha, beta, gamma"),
                  ("alpha, beta, gamma, delta", "alpha, beta, gamma"),
                  ("alpha, beta, gamma, gamma", "alpha, beta, gamma"),
                  ("alpha beta gamma", "alpha, beta, gamma")],
    "name_exact": [("Zanelade Holkerith", "Zanelade Holkerath"),
                   ("Holkerath", "Zanelade Holkerath"),
                   ("Zanelade Holkerath or Marn Vell", "Zanelade Holkerath"),
                   ("not named in the register", "Zanelade Holkerath")],
}


def near_miss_suite(name: str) -> list[tuple[str, str]]:
    """The authored (wrong_answer, expected_answer) pairs for one checker."""
    if name not in CHECKERS:
        raise ConfigError(f"unknown checker {name!r}")
    return list(_NEAR_MISSES.get(name, []))


def check(name: str, got: object, want: object) -> bool:
    """Run one named checker. Unknown names are refused, never defaulted --
    a typo'd checker silently becoming `contains` is precisely the permissive
    failure §8 warns about."""
    fn = CHECKERS.get(name)
    if fn is None:
        raise ConfigError(f"unknown checker {name!r}; known: {sorted(CHECKERS)}")
    return fn("" if got is None else str(got), "" if want is None else str(want))
