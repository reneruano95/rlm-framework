"""The leaf JSON envelope: parse, validate, verify spans (spec §5, §10 R5).

The envelope exists because of ONE measured number: asked about a fact that is
not in the chunk, the leaf answered anyway 37/39 times (95%, flat across every
chunk size -- `milestones/s2/RESULTS.md` finding 3). The `abstain` field is the mechanism
under test; everything in this file is the instrument that makes it scoreable.

The evidence half is measured near-inert going in (§10 R5: a span check passes
on 59/66 wrong answers, catch rate 7/66 = 11%) and is implemented anyway,
because the A/B has to price the whole envelope -- format cost included -- not a
flattering half of it.
"""
from __future__ import annotations

import pytest

from rlm.envelope import (
    Envelope,
    ParseResult,
    normalize_ws,
    parse,
    verify_evidence,
)

CLEAN = '{"answer": "ENT-40410", "evidence": ["the key issued is ENT-40410"], "abstain": false}'


# --------------------------------------------------------------------------- #
# parse: the happy path and the shapes a real model actually emits
# --------------------------------------------------------------------------- #


def test_parses_a_clean_envelope():
    result = parse(CLEAN)
    assert result.ok
    assert result.error is None
    assert result.envelope == Envelope(
        answer="ENT-40410",
        evidence=("the key issued is ENT-40410",),
        abstain=False,
    )
    assert not result.salvaged


def test_parses_an_abstention():
    result = parse('{"answer": "", "evidence": [], "abstain": true}')
    assert result.ok
    assert result.envelope.abstain is True
    assert result.envelope.answer == ""
    assert result.envelope.evidence == ()


def test_parses_through_a_code_fence():
    """```json fences are the single most common wrapper an instruct model puts
    round a JSON answer, and rejecting them would measure markdown habits rather
    than abstention."""
    result = parse(f"```json\n{CLEAN}\n```")
    assert result.ok
    assert result.envelope.answer == "ENT-40410"
    assert not result.salvaged      # a fence is not salvage, it is a wrapper


def test_parses_through_a_reasoning_block():
    """`enable_thinking` is off by default, but the leaf reopened a block anyway
    in S1 (F3), and a parser that dies on it would score the sampler."""
    result = parse(f"<think>the key is right there</think>\n{CLEAN}")
    assert result.ok
    assert result.envelope.answer == "ENT-40410"


def test_salvages_an_object_buried_in_prose_and_says_so():
    """Permissive, but never silently: the report needs raw compliance and
    salvaged compliance as separate numbers, or 'the envelope parses' would be a
    claim about this parser rather than about the model."""
    result = parse(f"Here is the answer you asked for:\n{CLEAN}\nHope that helps.")
    assert result.ok
    assert result.salvaged
    assert result.envelope.answer == "ENT-40410"


def test_salvage_finds_a_nested_object_whole():
    raw = ('prelude {"answer": "x", "evidence": ["a {brace} inside"], '
           '"abstain": false} coda')
    result = parse(raw)
    assert result.ok
    assert result.envelope.evidence == ("a {brace} inside",)


def test_a_brace_inside_a_string_does_not_end_the_object():
    """The salvage scan is a string-aware brace matcher, not a `rfind('}')`:
    corpus text quoted into `evidence` routinely contains braces and quotes."""
    raw = r'{"answer": "}", "evidence": ["he said \"stop\" }"], "abstain": false}'
    result = parse(raw)
    assert result.ok
    assert result.envelope.answer == "}"
    assert result.envelope.evidence == ('he said "stop" }',)


# --------------------------------------------------------------------------- #
# parse: every rejection, because each one is a retry the A/B has to pay for
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw, fragment", [
    ("", "empty"),
    ("   \n  ", "empty"),
    ("NONE", "no JSON object"),
    ("ENT-40410", "no JSON object"),
    ("[1, 2, 3]", "no JSON object"),
    ('{"answer": "x", "evidence": []', "no JSON object"),
    ('{"answer": "x", "evidence": []}', "abstain"),
    ('{"answer": "x", "abstain": false}', "evidence"),
    ('{"evidence": [], "abstain": false}', "answer"),
    ('{"answer": 7, "evidence": [], "abstain": false}', "answer"),
    ('{"answer": null, "evidence": [], "abstain": false}', "answer"),
    ('{"answer": "x", "evidence": "a span", "abstain": false}', "evidence"),
    ('{"answer": "x", "evidence": [3], "abstain": false}', "evidence"),
    ('{"answer": "x", "evidence": [], "abstain": "false"}', "abstain"),
    ('{"answer": "x", "evidence": [], "abstain": 0}', "abstain"),
])
def test_rejects_with_a_reason_naming_the_field(raw, fragment):
    result = parse(raw)
    assert not result.ok
    assert result.envelope is None
    assert fragment in result.error


def test_abstain_is_not_coerced_from_a_string():
    """`"abstain": "true"` is the failure mode that would quietly make every
    arm look like it abstains. Rejecting it costs one retry; coercing it would
    corrupt the only number this experiment produces."""
    assert not parse('{"answer": "", "evidence": [], "abstain": "true"}').ok


def test_extra_keys_are_kept_but_do_not_fail_the_parse():
    result = parse('{"answer": "x", "evidence": [], "abstain": false, '
                   '"confidence": 0.9}')
    assert result.ok
    assert result.envelope.extras == ("confidence",)


def test_the_error_is_a_short_single_line_string():
    """It lands in `steps.error_detail` and in a retry log line; a multi-line
    dump of the model's output there would bury the trace."""
    error = parse("not json at all").error
    assert "\n" not in error
    assert len(error) <= 200


def test_parse_result_is_falsey_when_it_failed():
    assert not ParseResult(envelope=None, error="x", raw="")
    assert parse(CLEAN)


# --------------------------------------------------------------------------- #
# normalization -- PINNED, because the span check's verdict depends on it
# --------------------------------------------------------------------------- #


def test_normalize_collapses_every_whitespace_run_to_one_space():
    """A whitespace collapse and NOTHING else -- the ends are not stripped, so
    the function stays a pure normalization. `verify_evidence` strips the
    needle, which is where the stripping actually matters."""
    assert normalize_ws("a  \n\t b \r\n c ") == "a b c "
    assert normalize_ws("\n\na\tb\n") == " a b "


def test_normalize_is_case_sensitive():
    """§5's rule is "whitespace-normalized substring match" and nothing more.
    Case-folding would only make the check MORE permissive, and the check is
    already measured at an 11% catch rate (§10 R5) -- widening it further would
    buy nothing and cost the ability to tell `ENT-4A1` from `ent-4a1`."""
    assert normalize_ws("ENT-4A1") != normalize_ws("ent-4a1")


def test_normalize_does_not_touch_punctuation_or_letters():
    """Every widening of this function widens what counts as evidence, so it
    does exactly one thing. Punctuation and accented letters pass through
    unchanged; only whitespace runs collapse -- and the pattern is UNICODE
    whitespace, so a non-breaking space in the corpus normalizes like any
    other."""
    assert normalize_ws("caf\u00e9,\n\nok.") == "caf\u00e9, ok."
    assert normalize_ws("a\u00a0b") == "a b"


# --------------------------------------------------------------------------- #
# evidence verification: in-process, zero model calls
# --------------------------------------------------------------------------- #

CHUNK = ("[custody note] The archive key issued to the Fenngate Ledger\n"
         "is  ENT-40410.  It was cut once, for that holder only.")


def test_verifies_a_span_that_survives_whitespace_normalization():
    assert verify_evidence(["The archive key issued to the Fenngate Ledger is "
                            "ENT-40410."], chunk=CHUNK) == (True,)


def test_rejects_a_span_that_is_not_in_the_chunk():
    assert verify_evidence(["the key issued is ENT-99999"], chunk=CHUNK) == (False,)


def test_verifies_each_span_independently():
    assert verify_evidence(["It was cut once", "never happened", "archive key"],
                           chunk=CHUNK) == (True, False, True)


def test_an_empty_span_never_verifies():
    """A bare `""` is a substring of every chunk, so trusting it would make the
    check pass on an envelope that quoted nothing at all."""
    assert verify_evidence(["", "   "], chunk=CHUNK) == (False, False)


def test_no_evidence_verifies_as_no_spans():
    assert verify_evidence([], chunk=CHUNK) == ()


def test_a_missing_chunk_is_NOT_CHECKED_rather_than_false():
    """`chunk=None` is the single-string `llm_query` form: the scaffold cannot
    see where the document ended, so it has not checked anything. Recording
    False there would read as 'checked and failed' -- the same mistake
    `rlm.leakcheck` refuses to make with its tri-state verdict."""
    assert verify_evidence(["anything"], chunk=None) == (None,)


def test_span_check_passes_on_the_R5_misattribution_shape():
    """The measured failure mode, reproduced as a unit test: the leaf hands over
    a DIFFERENT entity's real identifier from the same chunk. Every character of
    it is in the document, so the span check verifies -- 59/66 of the sweep's
    wrong answers had exactly this shape (§10 R5, catch rate 7/66 = 11%).

    This test asserts the DEFENCE IS WEAK. If it ever starts failing, the span
    check got stricter than the spec's rule and the A/B's control arm moved."""
    chunk = ("[custody note] the key issued to the Fenngate Ledger is ENT-11111. "
             "[custody note] the key issued to the Orstholm Trust is ENT-22222.")
    # asked about Fenngate, answered with Orstholm's key, quoting Orstholm's line
    assert verify_evidence(["the key issued to the Orstholm Trust is ENT-22222"],
                           chunk=chunk) == (True,)
