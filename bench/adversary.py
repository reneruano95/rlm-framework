"""Two build-time adversaries for v2's linear-semantic tasks (spec §14).

WHY TWO ADVERSARIES. A task is only worth the RLM machinery if (1) no
deterministic program reading the *parsed* record fields can answer it more
cheaply than the register's own scoring, and (2) the window set an agent
would actually need to read is bigger than a root could plausibly just read
itself instead of delegating. `parser_adversary` checks (1); `self_read_adversary`
checks (2). Both run BEFORE freeze, in Task 18's builder, as gates: a task
that fails either is not emitted.

`parser_adversary` mirrors `bench/corpus.py:regex_at_chance` -- same idea
(score every deterministic strategy a program could run, report accuracy
against the majority-class floor), applied to v2's `Query:` field instead of
v1's `Status:` line. The wh-word rule set is the load-bearing strategy: TREC
coarse labels are famously wh-word-recoverable ("who" asks about a person),
so a raw TREC register is expected to fail this gate -- that failure is what
Task 16's paraphrase step exists to fix, not a bug in the adversary.

A leak-free k-NN over the OTHER items' texts would need labels it does not
have (every item in the corpus is drawn from the same labelled pool, so
"nearest other item" still requires knowing that item's label to vote with)
-- excluded, per the brief, rather than implemented as a strategy that reads
the label it is trying to predict.

`self_read_adversary` answers a different question: even a program that
CANNOT parse the fields could still be answered by a root that just reads
enough of the corpus itself. For `count_label`/`count_two_labels`, every
record changes the answer if removed, so the necessary window set is every
window touching a record -- and the task earns its keep only if that set is
bigger than a root would read on its own initiative (`k`, default 40).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from bench.corpus_v2 import LinearSemanticCorpus
from rlm.context.chunker import ChunkConfig, split

_RECORD_RE = re.compile(
    r"\[(?P<ref>ENT-\S+)\]\s+(?P<org>.+?)\n"
    r"Filed:\s*(?P<filed>.+?)\n"
    r"Query:\s*(?P<query>.+?)\n"
    r"Notes:\s*(?P<notes>.*)",
    re.DOTALL,
)


def _parse_queries(corpus: LinearSemanticCorpus) -> list[str]:
    """The `Query:` line of every record, in the same order as `corpus.labels`.

    Parsed from `corpus.text` rather than `corpus.items` on purpose: the
    adversary is supposed to be a program reading the RENDERED register, the
    same thing a real parser-based defeat would read, not the pre-render
    `Item` objects the builder happens to still have around.
    """
    blocks = [b for b in corpus.text.split("\n\n") if b.startswith("[ENT-")]
    queries: list[str] = []
    for block in blocks:
        m = _RECORD_RE.match(block)
        assert m, f"record did not parse: {block[:80]!r}"
        queries.append(m.group("query"))
    assert len(queries) == len(corpus.labels), (
        f"{len(queries)} parsed records vs {len(corpus.labels)} labels")
    return queries


def _majority_label(labels: list[str]) -> str:
    return Counter(labels).most_common(1)[0][0]


# --------------------------------------------------------------------------- #
# Strategy 1: wh-word rules -- the load-bearing one. TREC's coarse label is
# famously recoverable from the question's own wh-word/opener.
# --------------------------------------------------------------------------- #

def _wh_word_rule(query: str) -> str | None:
    q = query.lower()
    if "abbreviation" in q or "stand for" in q or "acronym" in q:
        return "ABBR"
    if "how many" in q or "how much" in q or "what year" in q or \
       "what percentage" in q or re.search(r"\bwhen\b", q):
        return "NUM"
    if re.search(r"\bwho\b", q) or re.search(r"\bwhom\b", q) or "whose" in q:
        return "HUM"
    if re.search(r"\bwhere\b", q):
        return "LOC"
    if re.search(r"\bwhy\b", q) or "how did" in q or "how does" in q or \
       "how do" in q or "what does" in q or "what is the difference" in q:
        return "DESC"
    return None


def _wh_word_rules(queries: list[str], labels: list[str]) -> float:
    fallback = _majority_label(labels)
    hits = sum(1 for q, l in zip(queries, labels)
              if (_wh_word_rule(q) or fallback) == l)
    return round(hits / len(labels), 4)


# --------------------------------------------------------------------------- #
# Strategy 2: keyword lexicon per label -- broader than the wh-word opener,
# fixed ahead of time out of general TREC/domain knowledge (never fit to
# this corpus, or the strategy would be measuring itself).
# --------------------------------------------------------------------------- #

_LEXICON: dict[str, tuple[str, ...]] = {
    "HUM": ("who", "whom", "person", "author", "president", "actor",
           "invented", "founder", "wrote", "married"),
    "LOC": ("where", "country", "city", "state", "capital", "river",
           "mountain", "continent", "island", "ocean"),
    "NUM": ("how many", "how much", "when", "what year", "what percentage",
           "distance", "population", "date", "how long", "how far"),
    "ENTY": ("what is the name", "what kind of", "what type of", "what animal",
            "material", "language", "what color", "what movie"),
    "DESC": ("why", "what does", "how did", "how does", "what causes",
            "define", "meaning of", "what happens"),
    "ABBR": ("abbreviation", "stand for", "acronym", "short for"),
}


def _label_lexicon(queries: list[str], labels: list[str]) -> float:
    fallback = _majority_label(labels)
    hits = 0
    for q, l in zip(queries, labels):
        ql = q.lower()
        scores = {lab: sum(ql.count(kw) for kw in kws)
                 for lab, kws in _LEXICON.items()}
        best = max(scores.values())
        winners = [lab for lab, s in scores.items() if s == best]
        predicted = winners[0] if best > 0 and len(winners) == 1 else fallback
        hits += predicted == l
    return round(hits / len(labels), 4)


# --------------------------------------------------------------------------- #
# Strategy 3: capitalised-token heuristic -- a proper noun elsewhere in the
# question (not counting its first word, which is always capitalised by
# sentence position) is weak but deterministic evidence of an ENTY/LOC/HUM
# question rather than a NUM/DESC/ABBR one.
# --------------------------------------------------------------------------- #

def _capitalised_tokens(queries: list[str], labels: list[str]) -> float:
    fallback = _majority_label(labels)
    named_labels = {"ENTY", "LOC", "HUM"}
    named_fallback = _majority_label([l for l in labels if l in named_labels]) \
        if any(l in named_labels for l in labels) else fallback
    other_fallback = _majority_label([l for l in labels if l not in named_labels]) \
        if any(l not in named_labels for l in labels) else fallback
    hits = 0
    for q, l in zip(queries, labels):
        words = q.strip().rstrip("?").split()
        has_cap = any(w[0].isupper() for w in words[1:] if w and w[0].isalpha())
        predicted = named_fallback if has_cap else other_fallback
        hits += predicted == l
    return round(hits / len(labels), 4)


def parser_adversary(corpus: LinearSemanticCorpus) -> dict[str, float]:
    """Every deterministic strategy a program could run over the parsed
    record fields. Returns `{strategy: accuracy, "__chance__": max class
    share}`; a task is rejected if any strategy beats `chance + 0.02`."""
    queries = _parse_queries(corpus)
    labels = corpus.labels
    counts = Counter(labels)
    chance = max(counts.values()) / len(labels)
    return {
        "wh_word_rules": _wh_word_rules(queries, labels),
        "label_lexicon": _label_lexicon(queries, labels),
        "capitalised_tokens": _capitalised_tokens(queries, labels),
        "__chance__": round(chance, 4),
    }


def self_read_adversary(corpus: LinearSemanticCorpus, chunk_cfg: ChunkConfig,
                        count_tokens: Callable[[str], int], *, k: int = 40) -> int:
    """The size of the minimal necessary window set: for `count_label` /
    `count_two_labels`, every record is necessary (dropping any one changes
    the count or its certainty), so the necessary set is every window that
    holds at least one record. A task is rejected if that set's size is
    `<= k` -- a root could then just read the windows itself."""
    windows = split(corpus.text, chunk_cfg, count_tokens)
    necessary: set[int] = set()
    for record_id in corpus.record_ids:
        for i, window in enumerate(windows):
            if window.find(record_id) != -1:
                necessary.add(i)
    return len(necessary)


__all__ = ["parser_adversary", "self_read_adversary"]
