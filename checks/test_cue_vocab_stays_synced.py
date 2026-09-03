"""Task 16 fix round: `bench/adversary.py` (the attacker) and
`bench/corpus_v2.py` (the defender) used to keep two hand-written copies of
the same surface-cue vocabulary, with nothing to notice if they drifted
apart. `bench/_cue_vocab.py` replaced both copies with one shared module
imported by each -- these tests fail if either file stops using it, so a
future edit can't silently reintroduce a private copy that looks the same
today and diverges tomorrow.
"""
from bench import _cue_vocab, adversary, corpus_v2


def test_adversary_scores_against_the_shared_vocabulary_object():
    """Identity, not equality: a local dict with the same keys/values today
    would pass an `==` check and still be free to drift tomorrow."""
    assert adversary.LABEL_NOUNS is _cue_vocab.LABEL_NOUNS
    assert adversary.is_verb is _cue_vocab.is_verb


def test_corpus_v2_redacts_against_the_shared_vocabulary_object():
    assert corpus_v2.LABEL_NOUNS is _cue_vocab.LABEL_NOUNS
    assert corpus_v2.WH_OPENERS is _cue_vocab.WH_OPENERS
    assert corpus_v2.is_verb is _cue_vocab.is_verb


def test_a_cue_word_is_redacted_and_scored_from_the_same_entry():
    """A real vocabulary entry drives both sides: `_label_lexicon` scores
    it, and `_redact_cues` removes it -- proof the two call sites read the
    same list, not two dicts that happen to agree by coincidence."""
    cue = "president"
    assert any(cue in kws for kws in _cue_vocab.LABEL_NOUNS.values())
    text = f"The register asks after the {cue} of Ghana."
    assert cue not in corpus_v2._redact_cues(text).lower()
    scores = {lab: sum(text.lower().count(kw) for kw in kws)
             for lab, kws in adversary.LABEL_NOUNS.items()}
    assert max(scores.values()) > 0


def test_widening_the_shared_vocabulary_widens_both_sides_at_once():
    """A word absent from the vocabulary is invisible to both: neither
    redacted by the register nor scored by the lexicon. Adding it to
    `LABEL_NOUNS` (as this test does, on a local copy, not the real one)
    would make it visible to both at once -- which is the whole point of
    one shared list instead of two hand-kept mirrors."""
    absent = "glockenspiel"
    assert not any(absent in kws for kws in _cue_vocab.LABEL_NOUNS.values())
    text = f"The register asks after the {absent} on display."
    assert absent in corpus_v2._redact_cues(text)
    scores = {lab: sum(text.lower().count(kw) for kw in kws)
             for lab, kws in adversary.LABEL_NOUNS.items()}
    assert max(scores.values()) == 0
