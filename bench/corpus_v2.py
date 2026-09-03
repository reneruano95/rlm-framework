"""v2's vendored label source: TREC question classification.

WHY A VENDORED HUMAN-LABELLED CORPUS. v2's linear-semantic and interactive
tasks need a per-item label that no deterministic program can derive from the
item's text -- otherwise a regex defeats the task the same way §8 already
rules out for aggregation. TREC's `coarse_label` is that: six categories
(what kind of answer a question is asking for) assigned by human annotators,
not recoverable from surface string features. `bench/sources/trec/fetch.py`
is the one-shot fetcher; this module only reads and pins what it wrote.

`load_trec()` refuses if the vendored bytes drift from `trec_train.sha256` --
the manifest's `label_source` (`label_source_id()`) records exactly which
bytes every v2 answer was computed from, the same role `task_hash` plays for
v1's corpus documents (see `bench/manifest.py`).
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bench.vocab import SYL_A, organisation

TREC_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
_SRC = Path(__file__).resolve().parent / "sources" / "trec"

# What kind of answer each TREC coarse label asks a question for, in the
# register's own words -- never the label string itself, which is the
# property the label-absence test enforces.
LABEL_DESCRIPTION = {
    "HUM": "person or group of people",
    "LOC": "place or location",
    "NUM": "number, quantity, date or other numeric value",
    "ENTY": "thing, object or entity that is not a person or place",
    "DESC": "description, definition, reason or manner",
    "ABBR": "abbreviation or its expansion",
}

# The canonical one-word answer `most_common_label` grades against.
_LABEL_WORD = {
    "HUM": "person", "LOC": "place", "NUM": "number",
    "ENTY": "entity", "DESC": "description", "ABBR": "abbreviation",
}

# Coined month names, out of the benchmark's own syllable pool -- not a real
# calendar month, so a date can never leak a real-world clue about the item.
_MONTHS = [s.capitalize() for s in SYL_A[:12]]

_FILLER_WORDS = (
    "ledger tally warrant escrow tithe assay bond muniment chancery quire "
    "vellum foolscap counterfoil docket schedule annexe rubric errata"
).split()


@dataclass(frozen=True)
class Item:
    text: str
    label: str


def load_trec() -> list[Item]:
    data = (_SRC / "trec_train.jsonl").read_bytes()
    want = (_SRC / "trec_train.sha256").read_text().split()[0]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(f"vendored TREC moved: {got} != pinned {want}")
    return [Item(text=r["text"], label=TREC_LABELS[r["coarse_label"]])
            for r in (json.loads(l) for l in data.decode("utf-8").splitlines() if l)]


def label_source_id() -> str:
    return "CogComp/trec:train@sha256:" + (_SRC / "trec_train.sha256").read_text().split()[0][:16]


def _coined_date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(_MONTHS)} {rng.randint(1990, 2019)}"


@dataclass
class LinearSemanticCorpus:
    text: str
    items: list[Item]            # the sampled items, in placement order
    labels: list[str]            # items[i].label
    record_ids: list[str]        # "ENT-5xxxxx" per item
    question_kind: str           # "count_label" | "most_common_label" | "count_two_labels"
    target: tuple[str, ...]      # the label(s) the question is about
    answer: str                  # computed from labels
    checker: str                 # "int_exact" | "name_exact"
    seed: int
    measured_tokens: int
    counter_name: str
    retries: int = 0             # most_common_label tie-resample count

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _record(rng: random.Random, seed: int, idx: int, item: Item) -> str:
    """One record. The item's `text` is the Query the register files; the
    label that answers the question is never printed anywhere in the record."""
    org = organisation(rng)
    date = _coined_date(rng)
    filler = " ".join(rng.choice(_FILLER_WORDS) for _ in range(rng.randint(12, 26)))
    ref = f"ENT-5{seed % 10:01d}{idx:05d}"
    return (f"[{ref}] {org}\n"
            f"Filed: {date}\n"
            f"Query: {item.text}\n"
            f"Notes: {filler}.\n")


def _question_text(question_kind: str, target: tuple[str, ...]) -> str:
    if question_kind == "most_common_label":
        return ("Each record in this register files one Query. Which kind of "
                 "thing do the most Queries ask about: a person or group, a "
                 "place, a numeric value, an entity, a description, or an "
                 "abbreviation? Reply with exactly one of: person, place, "
                 "number, entity, description, abbreviation.")
    if question_kind == "count_two_labels":
        descs = " or a ".join(LABEL_DESCRIPTION[t] for t in target)
        return (f"Each record in this register files one Query. Count the "
                 f"records whose Query asks about a {descs}. Reply with the "
                 f"integer only, nothing else.")
    return (f"Each record in this register files one Query. Count the "
             f"records whose Query asks about a {LABEL_DESCRIPTION[target[0]]}. "
             f"Reply with the integer only, nothing else.")


def _sample_register(rng: random.Random, seed: int, items: list[Item],
                     target_tokens: int,
                     count: Callable[[str], int]
                     ) -> tuple[str, list[Item], list[str]]:
    """One grow-then-drop pass: sample every item without replacement in a
    fresh order, add records until the next one would overshoot, then drop
    whole records from the end until the exact count fits -- `corpus.build`'s
    established shape (see that function's docstring for why records are
    dropped rather than trimmed). `rng` continues rather than resetting, so
    repeated calls on a tie (see `build_linear_semantic`) sample a genuinely
    different subset each time.
    """
    pool = rng.sample(items, k=len(items))
    parts: list[str] = []
    sampled: list[Item] = []
    record_ids: list[str] = []
    running = 0
    for idx, item in enumerate(pool):
        rec = "\n" + _record(rng, seed, idx, item)
        n = count(rec)
        if running + n > target_tokens:
            break
        parts.append(rec)
        sampled.append(item)
        record_ids.append(f"ENT-5{seed % 10:01d}{idx:05d}")
        running += n
    text = "".join(parts).lstrip("\n")
    while parts and count(text) > target_tokens:
        parts.pop()
        sampled.pop()
        record_ids.pop()
        text = "".join(parts).lstrip("\n")
    return text, sampled, record_ids


def build_linear_semantic(seed: int, target_tokens: int,
                          count: Callable[[str], int], counter_name: str,
                          *, question_kind: str, items: list[Item],
                          paraphrase: bool = False) -> LinearSemanticCorpus:
    """Grow a register of human-labelled items until it fits the target.

    `paraphrase` is accepted and threaded now (PF-5) but does nothing yet;
    task 16 implements it and flips the default.

    Each item is sampled without replacement via `rng.sample`, one call per
    seed, so a different seed samples a genuinely different subset -- the
    property `test_two_seeds_sample_different_items_and_build_is_deterministic`
    checks.

    THE BUDGET IS THE REPORTED FIGURE. `measured_tokens` is `count(c.text)`,
    and `c.text` is the question plus the records -- so the records must be
    sampled against `target_tokens` *minus* the question (and its separator),
    not against `target_tokens` alone. The question's own token count is
    known before any record is sampled in every `question_kind`: for
    `count_label`/`count_two_labels` its `target` is drawn from `TREC_LABELS`
    independent of what gets sampled, and `most_common_label`'s question text
    is fixed regardless of which label turns out to be most common. Reserving
    that space up front, rather than trimming after the fact, keeps the bound
    structural: `count` is monotone and ceil-subadditive (`count(a) +
    count(b) >= count(a + b)`, true of `approx_tokens` by construction), so
    `count(question + sep) + count(records) <= target_tokens` implies
    `count(question + sep + records) <= target_tokens` too -- no corpus this
    function returns can be reported over budget.
    """
    rng = random.Random(seed)

    if question_kind == "count_two_labels":
        target: tuple[str, ...] = tuple(rng.sample(sorted(TREC_LABELS), 2))
    elif question_kind == "count_label":
        target = (rng.choice(TREC_LABELS),)
    else:  # most_common_label: the question text does not depend on target
        target = ()

    question = _question_text(question_kind, target)
    reserved = count(question + "\n\n")
    record_budget = target_tokens - reserved
    if record_budget <= 0:
        raise ValueError(
            f"target_tokens={target_tokens} cannot fit the {question_kind!r} "
            f"question alone ({reserved} tokens reserved for it)")

    text, sampled, record_ids = _sample_register(rng, seed, items, record_budget, count)
    labels = [i.label for i in sampled]

    retries = 0
    if question_kind == "most_common_label":
        while True:
            if not labels:
                raise ValueError(
                    f"target_tokens={target_tokens} leaves a record budget of "
                    f"{record_budget} tokens after reserving {reserved} for "
                    f"the question, which is too small to fit even one "
                    f"record -- most_common_label needs at least one")
            counts = Counter(labels)
            top_count = max(counts.values())
            top_labels = [l for l, c in counts.items() if c == top_count]
            if len(top_labels) == 1:
                target = (top_labels[0],)
                break
            # Tie: resample the target seed -- rng continues, does not reset.
            retries += 1
            text, sampled, record_ids = _sample_register(
                rng, seed, items, record_budget, count)
            labels = [i.label for i in sampled]
        answer = _LABEL_WORD[target[0]]
        checker = "name_exact"
    elif question_kind == "count_two_labels":
        answer = str(sum(1 for l in labels if l in target))
        checker = "int_exact"
    else:  # count_label
        answer = str(sum(1 for l in labels if l == target[0]))
        checker = "int_exact"

    full_text = question + "\n\n" + text

    return LinearSemanticCorpus(
        text=full_text, items=sampled, labels=labels, record_ids=record_ids,
        question_kind=question_kind, target=target, answer=answer,
        checker=checker, seed=seed, measured_tokens=count(full_text),
        counter_name=counter_name, retries=retries)
