"""Single source of truth for v2's linear-semantic surface-cue vocabulary.

Both `bench/adversary.py` (which SCORES a query against this vocabulary,
looking for cues) and `bench/corpus_v2.py` (which REDACTS this vocabulary
from a paraphrased query) import this module; it imports neither of them,
so importing it from both is not circular -- unlike importing one from the
other, which is why `corpus_v2._CUE_WORDS` used to be a hand-kept mirror of
`adversary._LEXICON` instead of the same object. See the Task 16 fix-round
ruling: widening the attacker and the defender from two copies of the same
list proves nothing (they trivially stay in sync); widening them from ONE
list means a broader vocabulary is broader for both at once, and the two
sides can only diverge by an actual code bug, which `checks/test_cue_vocab
_stays_synced.py` exists to catch.

LABEL_NOUNS is the per-label keyword/phrase list: what `_label_lexicon`
scores a query against, and what `_redact_cues` strips out of one. It is
NOT exhaustive by design -- TREC's coarse labels correlate with an
open-ended vocabulary, and the point of the paraphrase step is to survive
a NAMED, fixed adversary, not to solve open-vocabulary redaction. Growing
this list is the sanctioned way to broaden what both sides can see; see
`bench/adversary.py`'s module docstring for the strategies built on it.

AUX_WORDS / IRREGULAR_PAST_VERBS / PRESENT_TENSE_VERBS / is_verb: a
shared, deliberately-approximate verb-boundary detector, used by BOTH
sides of the same seam -- `corpus_v2._verb_pos` strips a record's Query
THROUGH the verb (the defender reads the boundary to know where a head
noun ENDS); `adversary._head_noun_rule` reads the same boundary to find
the word immediately BEFORE it (the attacker reads it to know where a
head noun STARTS). One boundary, two readers.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Per-label keyword/phrase vocabulary -- what `_label_lexicon` scores
# against, and what the register's cue-redaction pass removes.
# --------------------------------------------------------------------------- #

LABEL_NOUNS: dict[str, tuple[str, ...]] = {
    "HUM": ("who", "whom", "person", "author", "president", "actor",
           "invented", "founder", "wrote", "married"),
    "LOC": ("where", "country", "city", "state", "capital", "river",
           "mountain", "continent", "island", "ocean"),
    "NUM": ("how many", "how much", "when", "what year", "which year",
           "year", "what percentage", "percentage", "distance",
           "population", "date", "how long", "how far"),
    "ENTY": ("what is the name", "what kind of", "what type of",
            "what animal", "material", "language", "what color",
            "what movie", "disease", "species"),
    "DESC": ("why", "what does", "how did", "how does", "what causes",
            "define", "meaning of", "what happens", "cause of"),
    "ABBR": ("abbreviation", "stand for", "acronym", "short for"),
}


# --------------------------------------------------------------------------- #
# Fix round 3: `_label_lexicon` scores by plain substring counting
# (`ql.count(kw)`), which has always found "disease" inside "diseases" with
# no boundary check at all -- a plural is not a new concept, it's the SAME
# cue word the attacker already sees. Round 2 restored word-boundary
# anchoring on the DEFENDER's side (`_redact_cues`) to stop matching a cue
# substring inside an unrelated word ("date" inside "birthdate"), which
# also stopped it matching a cue's own regular plural ("mountain" inside
# "mountains") -- collateral, not the point of that fix. This completes the
# existing entries' regular English forms (trailing "s", or "es" after
# s/x/z/ch/sh, or "ies" after a consonant+y) so the redaction can still
# find them WITH the boundary kept -- no new nouns, no boundary dropped.
# --------------------------------------------------------------------------- #

def _plural(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if len(word) > 1 and word[-1] == "y" and word[-2].lower() not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def cue_words() -> frozenset[str]:
    """Every `LABEL_NOUNS` entry, plus the regular English plural of each
    single-word entry (a multi-word phrase like "how many" has no
    singular/plural form to complete, so it's left as-is). Not a widened
    vocabulary -- the same concepts `LABEL_NOUNS` already names, with their
    existing forms completed."""
    words: set[str] = set()
    for kws in LABEL_NOUNS.values():
        for w in kws:
            words.add(w)
            if " " not in w and w.isalpha():
                words.add(_plural(w))
    return frozenset(words)


def cue_pattern() -> re.Pattern[str]:
    """A single word-bounded, case-insensitive regex over `cue_words()` --
    the one matcher both `corpus_v2._redact_cues` (removes a match) and any
    future morphology-aware adversary strategy would consult, so the two
    sides can't drift on WHAT counts as a match the way `LABEL_NOUNS` alone
    already guarantees they can't drift on WHICH words are cues."""
    return re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in
                            sorted(cue_words(), key=len, reverse=True)) + r")\b",
        re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Wh-openers. `_WH_RE` in `corpus_v2.py` builds the LEADING-position regex
# out of these; `adversary._wh_noun_rule` scans for the same words at ANY
# position, since a non-leading TREC item ("In which year did...") carries
# its wh-word mid-sentence.
# --------------------------------------------------------------------------- #

WH_OPENERS = ("who", "whom", "whose", "where", "when", "which", "what",
             "why", "name", "define", "describe")

# --------------------------------------------------------------------------- #
# Verb-boundary detection, shared by the defender's strip and the
# attacker's extraction (see module docstring). Deliberately approximate:
# past tense has a reliable "-ed" suffix in English, present tense does
# not, so PRESENT_TENSE_VERBS is a closed, hand-picked list of the verbs
# common in trivia-question predicates ("X kills Y", "X means Y", "X goes
# Z") rather than a suffix heuristic -- an "-s" suffix is at least as often
# a plural noun ("Post-its", "the United States") as a verb, so guessing
# from it would strip content it shouldn't on the defence side and produce
# false leads on the attack side. Grown when a gap surfaces, same as
# LABEL_NOUNS.
# --------------------------------------------------------------------------- #

AUX_WORDS = {"'s", "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "will", "would", "has", "have", "had"}

IRREGULAR_PAST_VERBS = {
    "wrote", "gave", "took", "said", "did", "went", "came", "knew", "sang",
    "ran", "wore", "told", "sat", "stood", "held", "led", "bought",
    "brought", "caught", "fought", "taught", "thought", "spoke", "broke",
    "chose", "drove", "rode", "sold", "sent", "spent", "kept", "left",
    "met", "paid", "hit", "made", "got", "began", "won", "lost",
}

PRESENT_TENSE_VERBS = {
    "goes", "does", "means", "kills", "causes", "weighs", "contains",
    "reaches", "gives", "takes", "makes", "comes", "gets", "looks",
    "sounds", "stands", "lives", "dies", "grows", "happens", "occurs",
    "exists", "runs", "works", "serves", "holds", "represents",
    "indicates", "shows", "tells", "says", "calls", "acts", "lies", "sits",
    "flows", "borders", "separates", "connects", "requires", "needs",
}


def is_verb(word: str) -> bool:
    """True if `word` is (probably) a verb -- an auxiliary, a past tense
    (irregular or regular "-ed"), or a present tense from the closed list
    above. Shared by `corpus_v2._verb_pos` (strips through it) and
    `adversary._head_noun_rule` (extracts the word before it)."""
    lowered = word.lower()
    bare = word.strip("\"'`,.").lower()          # keep the apostrophe: "'s"
                                                  # is itself an AUX_WORDS entry
    if lowered in AUX_WORDS or bare in AUX_WORDS:
        return True
    if bare in IRREGULAR_PAST_VERBS or bare in PRESENT_TENSE_VERBS:
        return True
    return bare.endswith("ed") and len(bare) > 3


__all__ = ["LABEL_NOUNS", "cue_words", "cue_pattern", "WH_OPENERS",
          "AUX_WORDS", "IRREGULAR_PAST_VERBS", "PRESENT_TENSE_VERBS",
          "is_verb"]
