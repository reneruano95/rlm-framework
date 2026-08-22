"""The refusal A/B's scorer (`milestones/s2/run_refusal_ab.py`).

The experiment is worth exactly as much as this scorer. Its one rule: the four
arms are judged by the SAME instrument `milestones/s2/RESULTS.md` was judged by, so an
envelope reply is REDUCED to the plain-text answer `s2.run_sweep.classify`
already scores rather than getting a taxonomy of its own. A second notion of
"refusal" for the envelope arms would let the mechanism under test define the
metric that measures it.
"""
from __future__ import annotations

import json

import pytest

from s2.run_refusal_ab import (
    ARMS,
    ENVELOPE_BLOCK,
    common_cells,
    load_cells,
    reduce_envelope,
    render_report,
    score,
)
from s2.run_sweep import CONFABULATION, CORRECT, FALSE_POSITIVE, MALFORMED, MISS

KEY = "8243843a-ecc2-4f29-9122-60b53028b36b"
CHUNK = (f"[custody note] The archive key issued to the Fenngate Ledger is {KEY}. "
         "It was cut once, for that holder only.")


def env(answer: str, evidence: list[str], abstain: bool) -> str:
    return json.dumps({"answer": answer, "evidence": evidence, "abstain": abstain})


# --------------------------------------------------------------------------- #
# the pre-registered design
# --------------------------------------------------------------------------- #


def test_the_arms_are_the_pre_registered_two_by_two():
    """Four arms, two axes, and nothing else. The A/B is closed at two prefixes
    by §8's freeze rule; a fifth arm here would be prompt search on a
    measurement fixture."""
    assert len(ARMS) == 4
    prefixes = {a[1] for a in ARMS}
    assert prefixes == {"prompts/leaf-prefix.v1.md", "prompts/leaf-prefix.v2.md"}
    assert {a[2] for a in ARMS} == {True, False}
    assert len({(a[1], a[2]) for a in ARMS}) == 4      # a full factorial
    assert ENVELOPE_BLOCK == "prompts/leaf-envelope.v2.md"


# --------------------------------------------------------------------------- #
# reducing an envelope to something the sweep's classifier scores
# --------------------------------------------------------------------------- #


def test_an_abstention_reduces_to_the_refusal_token():
    out = reduce_envelope(env("", [], True), chunk=CHUNK)
    assert out["reduced_text"] == "NONE"
    assert out["abstain"] is True
    assert out["abstain_with_answer"] is False


def test_an_answer_reduces_to_the_answer_field():
    out = reduce_envelope(env(KEY, [f"key issued to the Fenngate Ledger is {KEY}"],
                              False), chunk=CHUNK)
    assert out["reduced_text"] == KEY
    assert out["evidence_verified"] == [True]
    assert out["evidence_ok"] is True


def test_abstain_beside_a_substantive_answer_is_NOT_a_refusal():
    """The structured twin of "NONE, but the closest match is ..." -- which the
    sweep's `is_refusal` already declines to credit. Crediting it here would
    understate the exact quantity this experiment measures, and it would do so
    in the direction that flatters the mechanism under test."""
    out = reduce_envelope(env(KEY, [], True), chunk=CHUNK)
    assert out["reduced_text"] == KEY
    assert out["abstain_with_answer"] is True


def test_abstain_beside_an_answer_that_is_ITSELF_a_refusal_stays_a_refusal():
    """`{"abstain": true, "answer": "NONE"}` is a model being redundant, not a
    model supplying a value."""
    out = reduce_envelope(env("NONE", [], True), chunk=CHUNK)
    assert out["reduced_text"] == "NONE"
    assert out["abstain_with_answer"] is False


def test_an_unparseable_envelope_reduces_to_nothing():
    out = reduce_envelope("I'm sorry, I can't do JSON", chunk=CHUNK)
    assert out["envelope_ok"] is False
    assert out["reduced_text"] == ""
    assert out["envelope_error"]


# --------------------------------------------------------------------------- #
# ... and the labels that come out the other side
# --------------------------------------------------------------------------- #


def _score(raw, *, envelope, qtype, expected, kind="uuid"):
    return score(raw, envelope=envelope, chunk=CHUNK, question_type=qtype,
                 expected=expected, expected_kind=kind)


def test_an_envelope_abstention_on_an_ABSENT_question_is_CORRECT():
    """The outcome the whole experiment is hunting for."""
    assert _score(env("", [], True), envelope=True, qtype="absent",
                  expected=None)["label"] == CORRECT


def test_an_envelope_answer_on_an_ABSENT_question_is_a_FALSE_POSITIVE():
    assert _score(env("ENT-40410", ["ENT-40410"], False), envelope=True,
                  qtype="absent", expected=None)["label"] == FALSE_POSITIVE


def test_an_envelope_abstention_on_a_PRESENT_fact_is_a_MISS():
    """The cost side of the trade, and it must be scored as its own label: a
    leaf that abstains on everything has a perfect false-positive rate."""
    assert _score(env("", [], True), envelope=True, qtype="literal",
                  expected=KEY)["label"] == MISS


def test_an_envelope_answer_that_is_right_is_CORRECT():
    assert _score(env(KEY, [f"is {KEY}"], False), envelope=True,
                  qtype="literal", expected=KEY)["label"] == CORRECT


def test_an_envelope_answer_that_is_wrong_is_a_CONFABULATION():
    other = "0f3aac07-d1fe-460c-907f-53ddb57cc79a"
    out = _score(env(other, [], False), envelope=True, qtype="literal",
                 expected=KEY)
    assert out["label"] == CONFABULATION
    assert out["uuid_edit_distance"] > 0


def test_an_unparseable_envelope_is_MALFORMED_in_every_question_type():
    for qtype, expected in (("absent", None), ("literal", KEY)):
        out = _score("here you go: the key is probably ENT-1", envelope=True,
                     qtype=qtype, expected=expected)
        assert out["label"] == MALFORMED
        assert out["envelope_ok"] is False


def test_a_plain_arm_is_scored_exactly_as_the_sweep_scored_it():
    """The control arms must reproduce `milestones/s2/RESULTS.md`'s own numbers on
    `milestones/s2/RESULTS.md`'s own strings, or the baseline column is not a baseline."""
    assert _score("NONE", envelope=False, qtype="absent",
                  expected=None)["label"] == CORRECT
    assert _score("ENT-17687", envelope=False, qtype="absent",
                  expected=None)["label"] == FALSE_POSITIVE
    assert _score(KEY, envelope=False, qtype="literal",
                  expected=KEY)["label"] == CORRECT
    assert _score("NONE", envelope=False, qtype="literal",
                  expected=KEY)["label"] == MISS


def test_a_plain_arm_never_reports_envelope_facts():
    out = _score("NONE", envelope=False, qtype="absent", expected=None)
    assert out["envelope_ok"] is None
    assert out["abstain"] is None
    assert out["evidence_ok"] is None


def test_the_span_check_verifies_the_R5_misattribution_shape():
    """R5's finding, re-measured through the A/B's own scorer: the leaf quotes a
    genuine span and answers with the wrong entity's value, so the evidence
    check PASSES on an answer that is wrong. If this ever fails, the span check
    got stricter than §5's rule and the arms stopped being comparable."""
    chunk = (f"key issued to the Fenngate Ledger is {KEY}. "
             "key issued to the Orstholm Trust is ENT-22222.")
    out = score(env("ENT-22222", ["key issued to the Orstholm Trust is ENT-22222"],
                    False),
                envelope=True, chunk=chunk, question_type="absent",
                expected=None, expected_kind="uuid")
    assert out["label"] == FALSE_POSITIVE
    assert out["evidence_ok"] is True        # verified, and wrong


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def _record(**over):
    rec = {
        "arm": "a-v1-plain", "envelope": False, "cell_uid": "s2-1024-p50#s1",
        "cell_id": "s2-1024-p50", "fixture_seed": 1, "size_target": 1024,
        "size_measured": 1021, "position": 0.5, "question_type": "absent",
        "question": "q", "expected": None, "expected_kind": "uuid", "trial": 1,
        "status": "ok", "raw_output": "NONE", "wall_s": 1.0, "tokens_out": 3,
        "tokens_in": 1300, "leak_detected": False, "slot_ok": True,
        "_chunk": CHUNK,
    }
    rec.update(over)
    return rec


def test_the_report_renders_all_four_arms_and_both_bold_columns():
    records = []
    for arm, envelope in (("a-v1-plain", False), ("b-v1-envelope", True),
                          ("c-v2-plain", False), ("d-v2-envelope", True)):
        absent = env("ENT-1", [], False) if envelope else "ENT-1"
        present = env(KEY, [], False) if envelope else KEY
        records.append(_record(arm=arm, envelope=envelope, question_type="absent",
                               raw_output=absent))
        records.append(_record(arm=arm, envelope=envelope, question_type="literal",
                               raw_output=present, expected=KEY))
    md = render_report(records)
    for arm, _, _ in ARMS:
        assert f"`{arm}`" in md
    assert "FALSE-POSITIVE rate" in md
    assert "MISS (fact present)" in md
    assert "R13 contamination audit" in md


def test_the_report_re_derives_labels_rather_than_trusting_stored_ones():
    """A classifier fix must cost a re-read of a JSONL file, never a re-run of
    the GPU -- the same rule `milestones/s2/run_sweep.py --phase report` follows."""
    md = render_report([_record(label="CORRECT", raw_output="ENT-1")])
    assert "FALSE-POSITIVE" in md
    # the stored label said CORRECT; the raw output says otherwise, and the
    # verbatim table is where a reader can check
    assert "`ENT-1`" in md


def test_the_report_quotes_every_false_positive_verbatim():
    md = render_report([_record(raw_output="ENT-40410 / c820")])
    assert "ENT-40410 / c820" in md


def test_a_contaminated_answer_is_counted_not_hidden():
    md = render_report([_record(leak_detected=True, raw_output="ENT-1")])
    assert "R13 contamination audit" in md
    assert "**1**" in md


def test_a_verbose_refusal_scores_FALSE_POSITIVE_but_hands_over_nothing():
    """The pinned classifier's known gap, pinned as a test rather than fixed:
    `is_refusal`'s phrase list has `(the|this) excerpt does not` and this leaf
    writes "the provided text does not contain ...", so a genuine refusal scores
    FALSE-POSITIVE. That biases against exactly the arms under test, since both
    treatments make refusals wordier. The list is NOT widened -- it is the
    instrument `milestones/s2/RESULTS.md` was scored with -- so the report carries the
    mechanical sub-question instead: did the reply hand over an identifier?"""
    verbose = ("The provided text does not contain any mention of the "
               "Orstholtthorpe Chapterhouse or an associated archive key.")
    out = _score(verbose, envelope=False, qtype="absent", expected=None)
    assert out["label"] == FALSE_POSITIVE          # the pinned label, unmoved
    assert out["supplied_identifier"] is False     # ... and it handed over nothing
    # while a real lie hands over a key
    lie = _score("ENT-17687", envelope=False, qtype="absent", expected=None)
    assert lie["label"] == FALSE_POSITIVE
    assert lie["supplied_identifier"] is True


def test_the_identifier_column_is_measured_on_the_envelope_ANSWER_field():
    """An envelope arm must be judged on what a root would read and submit, not
    on the JSON punctuation around it: an abstention whose `evidence` quotes the
    chunk has still handed over nothing."""
    out = _score(env("", [], True), envelope=True, qtype="absent", expected=None)
    assert out["supplied_identifier"] is False
    out = _score(env("ENT-17687", ["ENT-17687"], False), envelope=True,
                 qtype="absent", expected=None)
    assert out["supplied_identifier"] is True


def test_the_report_also_scores_envelope_arms_as_plain_text():
    """The decomposition: a reply can fail the FORMAT and still carry the
    CONTENT under test. A prose refusal to an ABSENT question is MALFORMED to
    the envelope scorer (the runtime could not have used it) and a correct
    refusal to the plain-text one, and the report must show both or the
    experiment cannot tell "the block broke the reply" from "the block changed
    nothing"."""
    records = [
        _record(arm="a-v1-plain", raw_output="ENT-1"),
        _record(arm="b-v1-envelope", envelope=True,
                raw_output="The excerpt does not mention that organisation."),
    ]
    md = render_report(records)
    assert "re-scored as PLAIN TEXT" in md
    # envelope scorer: unusable. plain-text scorer: a correct refusal, so the
    # false-positive rate for that arm is 0/1 in the second table.
    assert "0/1 (0%)" in md


def test_the_headline_tables_are_over_cells_EVERY_arm_ran():
    """The four prefixes render to different lengths (311 / 773 / 577 / 1039
    tokens measured), so against a fixed 2,560-token slot the plain-v1 arm
    admits a 2,048-token cell the envelope arms cannot fit. Scoring each arm
    over its own admitted set would compare four different corpora and call the
    difference an effect."""
    records = [
        # both arms ran the shared cell ...
        _record(arm="a-v1-plain", cell_uid="shared", raw_output="ENT-1"),
        _record(arm="b-v1-envelope", envelope=True, cell_uid="shared",
                raw_output=env("ENT-1", [], False)),
        # ... only the short-prefix arm fitted the big one, and it refused there
        _record(arm="a-v1-plain", cell_uid="big-2048", raw_output="NONE"),
    ]
    assert common_cells(records) == {"shared"}
    md = render_report(records)
    # a's false-positive rate is 1/1 on the shared cell, NOT 1/2 as it would be
    # if the cell only it could run were folded in
    assert "1/1 (100%)" in md
    assert "1/2" not in md
    assert "over the 1 cell(s) EVERY arm ran" in md


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def test_cells_from_different_fixture_seeds_stay_distinguishable(tmp_path):
    """Power comes from independent FACTS, so the run pools several fixture
    seeds at one chunk size. Two cells built at the same (size, position) from
    different seeds share a `cell_id` and must not collide."""
    for seed in (1, 2):
        d = tmp_path / f"s{seed}"
        d.mkdir()
        (d / "s2-1024-p50.chunk.txt").write_text(CHUNK, encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({
            "token_counter": "leaf:/tokenize", "seed": seed,
            "cells": {"s2-1024-p50": {
                "cell_id": "s2-1024-p50", "size_tokens": 1024,
                "measured_tokens": 1021, "position": 0.5, "sha256": "x",
                "chunk_path": str(d / "s2-1024-p50.chunk.txt"),
                "questions": {}, "needles": {}}},
        }), encoding="utf-8")
    cells, corpus = load_cells([tmp_path / "s1", tmp_path / "s2"])
    assert [c["uid"] for c in cells] == ["s2-1024-p50#s1", "s2-1024-p50#s2"]
    assert len(corpus) == 2


def test_offline_built_fixtures_are_refused(tmp_path):
    """The slot-fit arithmetic and every size claim are in LEAF tokens; a
    manifest built with the 4-chars-per-token proxy would measure the proxy."""
    (tmp_path / "manifest.json").write_text(json.dumps({
        "token_counter": "approx-offline", "seed": 1, "cells": {}}),
        encoding="utf-8")
    with pytest.raises(SystemExit, match="REFUSING"):
        load_cells([tmp_path])
