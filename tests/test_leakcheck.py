# tests/test_leakcheck.py
"""R13's foreign-string detector (spec §10 R13 (3), §5 C4).

The corpus fixtures below are cut down from the real leaked answers quoted in
`milestones/s2/R13.md` §4 -- the same identifier shapes, the same two documents, so the
detector is tested against the failure it was built for rather than against an
invented one.
"""
import time

import pytest

from rlm.leakcheck import (
    ChunkIndex,
    NOT_CHECKED,
    foreign_identifiers,
    identifier_tokens,
)

# milestones/s2/R13.md §4: chunk `s2-1024-p50` and chunk `s2-2048-p50`, whose keys and
# entity names the leaked answers mixed together.
CHUNK_1024 = (
    "Custody note for the Prylfennwick Trust. The archive key issued to the "
    "Prylfennwick Trust is 1251d802-86aa-4e75-96be-aefc175c1e8e. "
    "The Orstlornholm Trust holds record ENT-84213:9f3ac1d0e7b24518."
)
CHUNK_2048 = (
    "Custody note for the Quinfennsted Trust. The archive key issued to the "
    "Quinfennsted Trust is 0f3aac07-d1fe-460c-907f-53ddb57cc797. "
    "The Selkdaleridge Bureau holds record ENT-19022:c41d8fa0b6e39274."
)
CHUNK_8192 = (
    "Custody note for the Hurnshawfield Bureau. The archive key issued to the "
    "Hurnshawfield Bureau is 8243843a-ecc2-4f29-9122-60b53028b36b."
)
CORPUS = {"c0": CHUNK_1024, "c1": CHUNK_2048, "c2": CHUNK_8192}

QUESTION = "What archive key was issued to the Hurnshawfield Bureau?"


def check(answer: str, *, sent: str = CHUNK_8192, question: str = QUESTION,
          corpus: dict[str, str] | None = None):
    return foreign_identifiers(answer, sent=f"{sent}\n\n{question}",
                               corpus=CORPUS if corpus is None else corpus)


# --------------------------------------------------------------------------- #
# what it catches
# --------------------------------------------------------------------------- #


def test_a_uuid_belonging_to_another_chunk_is_a_leak():
    """milestones/s2/R13.md §4, trial 1: the 2048 chunk was asked an absent-fact question
    and answered with the 1024 chunk's key."""
    verdict = check("The archive key is 1251d802-86aa-4e75-96be-aefc175c1e8e.",
                    sent=CHUNK_2048)
    assert verdict.detected
    assert [h.token for h in verdict.hits] == ["1251d802-86aa-4e75-96be-aefc175c1e8e"]
    assert verdict.hits[0].source == "c0"
    assert "1251d802-86aa-4e75-96be-aefc175c1e8e" in verdict.detail
    assert "c0" in verdict.detail


def test_an_ent_code_and_its_hex_partner_are_both_identifier_shaped():
    """§10 R13: "a 1024-token window returned eight `ENT-#####:hex` pairs whose
    bindings are split verbatim between the current document and one previously
    held on that slot" -- the value-level artifact, so both halves must be
    detectable."""
    verdict = check("Record ENT-19022:c41d8fa0b6e39274 is held by that bureau.")
    assert verdict.detected
    tokens = {h.token for h in verdict.hits}
    assert "ENT-19022" in tokens
    assert "c41d8fa0b6e39274" in tokens
    assert all(h.source == "c1" for h in verdict.hits)


def test_matching_is_case_insensitive():
    verdict = check("the key is 1251D802-86AA-4E75-96BE-AEFC175C1E8E", sent=CHUNK_2048)
    assert verdict.detected


def test_a_long_alphanumeric_token_is_identifier_shaped():
    """The fourth shape: a long mixed letter+digit run that is neither a UUID,
    an ENT- code, nor pure hex."""
    corpus = {"own": "nothing but prose here", "other": "token k7x2mq9v4zt1 issued"}
    verdict = foreign_identifiers("the token is k7x2mq9v4zt1",
                                  sent="nothing but prose here", corpus=corpus)
    assert verdict.detected
    assert verdict.hits[0].token == "k7x2mq9v4zt1"


# --------------------------------------------------------------------------- #
# what it must NOT call a leak
# --------------------------------------------------------------------------- #


def test_an_identifier_from_the_chunk_that_was_sent_is_not_a_leak():
    """milestones/s2/R13.md §6 #2: 75 of the sweep's answers quote an identifier from
    their OWN chunk. Those are the ordinary case, not contamination."""
    verdict = check("The archive key is 8243843a-ecc2-4f29-9122-60b53028b36b.")
    assert not verdict.detected
    assert verdict.hits == ()
    assert verdict.detail is None


def test_an_identifier_present_in_no_chunk_is_not_a_leak():
    """milestones/s2/R13.md §6 #2: 3 answers quote `7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d`,
    a fabricated placeholder present in no chunk. Fabrication is R5's problem,
    not R13's -- the detector must not annex it."""
    verdict = check("The archive key is 7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d.")
    assert not verdict.detected


def test_an_identifier_echoed_from_the_question_is_not_a_leak():
    """R13's own oracle definition: a leak is absent from the document AND
    absent from the question. A model quoting the question back is not
    evidence of anything."""
    verdict = check(
        "You asked about 1251d802-86aa-4e75-96be-aefc175c1e8e; it is not here.",
        sent=CHUNK_8192,
        question="Is 1251d802-86aa-4e75-96be-aefc175c1e8e in this excerpt?")
    assert not verdict.detected


def test_ordinary_prose_produces_no_candidates_at_all():
    """The cheapness and the false-positive floor in one: only
    identifier-shaped tokens are ever candidates, so shared English words
    across chunks can never raise a hit."""
    assert identifier_tokens(
        "The provided text does not contain a custody note for that bureau."
    ) == set()


# --------------------------------------------------------------------------- #
# the two stated limits, pinned as tests so they stay honest
# --------------------------------------------------------------------------- #


def test_LIMIT_paraphrased_leakage_is_not_caught():
    """Stated limit 1 (§10 R13 (3)). An answer that carries another chunk's
    CONTENT without carrying any of its identifier-shaped strings is invisible
    to this detector, and no amount of pattern-set widening fixes it."""
    verdict = check("The key for that bureau belongs to a different trust "
                    "whose custody note was recorded earlier.")
    assert not verdict.detected


def test_LIMIT_a_contaminated_refusal_is_not_caught():
    """Stated limit 2 (§10 R13 (3)) -- and a REAL observed case, quoted
    verbatim in milestones/s2/R13.md §4: the answer correctly refuses (the fact really is
    absent) while enumerating four entities belonging to two documents held
    earlier on the same slot. The refusal is right, the model is contaminated,
    and the detector says nothing, because entity NAMES are not
    identifier-shaped: adding a proper-noun class would flag ordinary
    capitalised English across chunks."""
    verdict = check(
        'The provided text does not contain a custody note for the '
        '"Hurnshawfield Bureau." It contains custody notes for the '
        '"Prylfennwick Trust," "Orstlornholm Trust," "Quinfennsted Trust," '
        'and "Selkdaleridge Bureau."',
        sent=CHUNK_8192)
    assert not verdict.detected


# --------------------------------------------------------------------------- #
# shape of the API
# --------------------------------------------------------------------------- #


def test_it_is_a_pure_function_of_its_arguments():
    a = check("key 1251d802-86aa-4e75-96be-aefc175c1e8e", sent=CHUNK_2048)
    b = check("key 1251d802-86aa-4e75-96be-aefc175c1e8e", sent=CHUNK_2048)
    assert a == b


def test_no_corpus_means_NOT_CHECKED_never_a_clean_bill():
    """An empty corpus cannot certify anything. `detected` is None -- "not
    checked" -- and never False, which would read as "checked and clean"."""
    verdict = foreign_identifiers("anything at all", sent="x", corpus={})
    assert verdict is NOT_CHECKED
    assert verdict.detected is None
    assert verdict.detail is None


def test_the_detail_is_bounded_for_a_text_column():
    corpus = {"own": "", **{f"c{i}": f"key {i:08x}aaaaaaaa" for i in range(200)}}
    answer = " ".join(f"{i:08x}aaaaaaaa" for i in range(200))
    verdict = foreign_identifiers(answer, sent="", corpus=corpus)
    assert verdict.detected
    assert len(verdict.hits) == 200          # every hit is kept for analysis
    assert len(verdict.detail) <= 512        # the TEXT column is not a dump


def test_the_index_and_the_pure_function_agree():
    index = ChunkIndex.from_chunks(CORPUS)
    answer = "key 1251d802-86aa-4e75-96be-aefc175c1e8e"
    assert index.foreign(answer, sent=CHUNK_2048) == foreign_identifiers(
        answer, sent=CHUNK_2048, corpus=CORPUS)


@pytest.mark.timeout(60)
def test_detection_is_cheap_enough_to_run_on_every_leaf_call():
    """§10 R13 (3): "zero model calls, one set-membership test". The index is
    built once per corpus; a detector that re-scanned the corpus per answer
    would cost O(corpus) on each of ~848 leaf calls in a 200K-token episode.

    300 chunks x ~4 KB is ~1.2 MB of corpus; 400 detections against it must
    stay far under a second of scanning in total."""
    corpus = {f"c{i}": (f"filler prose {i} " * 200) + f" key {i:08x}deadbeef"
              for i in range(300)}
    index = ChunkIndex.from_chunks(corpus)
    t0 = time.perf_counter()
    for i in range(400):
        index.foreign(f"the key is {i:08x}deadbeef", sent="unrelated text")
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"400 detections took {elapsed:.2f}s -- not cheap"
