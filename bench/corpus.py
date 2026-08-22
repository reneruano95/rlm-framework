"""Aggregation corpora for §8's frozen benchmark.

No existing generator can build one. All three (milestones/s1/make_fixtures,
milestones/s2/make_sweep_fixtures, milestones/s2/make_distance_fixtures) are single-needle shaped and
assert exactly-one-identifier as a VALIDITY condition -- the opposite of what an
aggregation task needs, which is many countable items whose total is the answer.

What this builds, and why it is two shapes and not one. §8 requires of the
aggregation category:

    "at least one aggregation task must defeat deterministic string matching --
     requiring semantic judgment per item, verified at authoring by showing a
     pure-regex solution scores at chance -- and at least one must be
     regex-solvable, so the benchmark rewards the root choosing code over leaf
     calls when code suffices."

so the two shapes are not variety, they are the measurement:

    REGEX_SOLVABLE   every record carries an explicit `Status: SEALED|OPEN`
                     field. A three-line REPL script answers it exactly. A root
                     that spends 300 leaf calls here is choosing badly, and the
                     benchmark should say so.

    REGEX_DEFEATING  the same corpus, but the question is about a free-text
                     `Disposition:` line whose meaning is the answer. The
                     phrasings are drawn from per-label pools that SHARE their
                     surface vocabulary, so no keyword separates them -- the
                     at-chance demonstration §8 asks for is then a fact about
                     the corpus, buildable offline, not a claim in prose.

SIZING. `max_wall_clock_s` and the ~130K cap were ruled jointly (spec §8,
2026-08-15): 130,464 tokens = 302 windows at the snap bound = 604 sub-calls,
inside `max_subcalls: 926`. Both bounds are asserted at build time rather than
checked later, because a corpus one window over the line makes every episode
using it a `budget_kill`, which §8 scores as a FAILURE for every arm.

Token counting is OFFLINE by default (`bench.tokens.approx_tokens`, the
repo's one stated 4-chars-per-token proxy) so a corpus can be built with no GPU.
The proxy is not the truth: chars-per-token is vocabulary-specific (4.058 on
s1's pool, 3.765 on s2's), so the manifest stamps which counter was used and
`--leaf-port` re-measures against the real tokenizer before a freeze.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from bench.vocab import coined_name, harvest_names, organisation

# The ruled ceiling (spec §8). Asserted, never assumed.
MAX_AGG_TOKENS = 130_464
SNAP_STRIDE = 432          # int(stride 480 * 0.9)
MAX_SUBCALLS = 926

Label = Literal["withheld", "released"]

# Disposition phrasings, as WORD-MULTISET-MATCHED PAIRS.
#
# The first draft merely overlapped vocabulary, and it was not enough: measured,
# a bare `\b(not|no|refus)\w*\b` scored 0.654 against a chance of 0.551, because
# negation words still landed on the withheld side more often. "Mostly
# overlapping" is not regex-defeating; it is regex-inconvenient.
#
# So each pair below is an exact permutation of the same words, and the label is
# carried ONLY by their order. Any classifier that looks at which words are
# present -- which is what a regex is -- is then at chance BY CONSTRUCTION, not
# by luck of phrasing, and `assert_pairs_are_permutations()` keeps it that way
# when someone edits the list. Reading one still takes a second, which is the
# point: §8 wants "semantic judgment per item".
_PAIRS: list[tuple[str, str]] = [
    # (withheld, released)
    ("the embargo outranks the release authorisation",
     "the release authorisation outranks the embargo"),
    ("the refusal postdates the grant",
     "the grant postdates the refusal"),
    ("withholding was upheld and release was denied",
     "release was upheld and withholding was denied"),
    ("the sealing order replaced the disclosure order",
     "the disclosure order replaced the sealing order"),
    ("custody remained with the office, not the requesting party",
     "custody remained with the requesting party, not the office"),
    ("the later ruling denied what the earlier ruling allowed",
     "the later ruling allowed what the earlier ruling denied"),
]


def _words(s: str) -> list[str]:
    return sorted(re.findall(r"[a-z]+", s.lower()))


def assert_pairs_are_permutations() -> None:
    """Each (withheld, released) pair must use exactly the same words.

    This is the property that makes the at-chance demonstration a fact about
    the corpus rather than a hope about the phrasing, so it is checked at import
    and not in a test that may not be run before a freeze."""
    for w, r in _PAIRS:
        if _words(w) != _words(r):
            raise AssertionError(
                f"disposition pair is not a word permutation, so a bag-of-words "
                f"regex could separate the labels:\n  withheld: {w!r}\n  "
                f"released: {r!r}")


assert_pairs_are_permutations()

_WITHHELD = [w for w, _ in _PAIRS]
_RELEASED = [r for _, r in _PAIRS]

_FILLER_WORDS = (
    "ledger tally warrant escrow tithe assay bond muniment chancery quire "
    "vellum foolscap counterfoil docket schedule annexe rubric errata"
).split()


@dataclass
class AggregationCorpus:
    text: str
    n_records: int
    sealed_count: int              # ground truth for the regex-solvable task
    withheld_count: int            # ground truth for the regex-defeating task
    seed: int
    counter_name: str
    measured_tokens: int
    labels: list[Label] = field(default_factory=list)

    @property
    def windows(self) -> int:
        """Snap-bound window count, the number §8 requires in the manifest."""
        return -(-self.measured_tokens // SNAP_STRIDE)

    @property
    def subcalls(self) -> int:
        return 2 * self.windows

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def assert_affordable(self) -> None:
        """The two bounds the 2026-08-15 ruling pinned. A corpus one window over
        the line turns every episode that uses it into a `budget_kill`, which §8
        scores as a failure for EVERY arm -- so this is checked at build time,
        where the fix is free."""
        if self.measured_tokens > MAX_AGG_TOKENS:
            raise AssertionError(
                f"corpus is {self.measured_tokens} tokens, over the ruled "
                f"aggregation cap of {MAX_AGG_TOKENS}")
        if self.subcalls > MAX_SUBCALLS:
            raise AssertionError(
                f"corpus needs {self.subcalls} sub-calls for full coverage, "
                f"over max_subcalls {MAX_SUBCALLS}")


def _record(rng: random.Random, idx: int, label: Label) -> str:
    """One record. The label is carried by a RELATION, not by any word.

    Word-permutation pairs (below) defeat a bag-of-words regex but not a regex
    that enumerates six fixed sentences -- that is a lookup table wearing a
    solution's clothes, and it would satisfy §8 only in appearance. So the
    disposition names the record's OWNER and a counterparty, and the label is
    the DIRECTION of custody relative to the owner:

        withheld  <=>  custody passed TO the owner   (it stayed with them)
        released  <=>  custody passed FROM the owner (it went to the other party)

    Both names are per-record coined names out of a 41,600-name space, so no
    fixed string identifies the owner, and "passed from ... to ..." appears in
    every record either way. Deciding the label means reading the header and the
    disposition and comparing them -- a join, which is exactly the "semantic
    judgment per item" §8 asks for and exactly what a regex cannot do.
    """
    org = organisation(rng)
    counterparty = organisation(rng)
    while counterparty == org:
        counterparty = organisation(rng)
    sealed = rng.random() < 0.4
    ref = f"ENT-{100000 + idx}"
    if label == "withheld":
        src, dst = counterparty, org
    else:
        src, dst = org, counterparty
    # Drawn INDEPENDENTLY of the label, from both halves of every pair. If it
    # tracked the label it would re-introduce exactly the leak the custody
    # direction was adopted to remove -- a six-way phrase lookup would score
    # 1.000 and the task would be regex-solvable after all. It is decoration,
    # and `regex_at_chance`'s `phrase-lookup` candidate exists to prove it.
    flourish = rng.choice(_WITHHELD + _RELEASED)
    filler = " ".join(rng.choice(_FILLER_WORDS) for _ in range(rng.randint(12, 26)))
    return (f"[{ref}] {org}\n"
            f"Status: {'SEALED' if sealed else 'OPEN'}\n"
            f"Disposition: custody passed from the {src} to the {dst}; "
            f"{flourish}.\n"
            f"Notes: {filler}.\n")


def build(seed: int, target_tokens: int,
          count: Callable[[str], int],
          counter_name: str) -> AggregationCorpus:
    """Grow the corpus record by record until it fits the target.

    Grown rather than binary-searched on filler length: the ANSWER depends on
    how many records exist, so the corpus cannot be resized after the ground
    truth is fixed. Building up and stopping at the bound keeps the two
    consistent by construction.
    """
    if target_tokens > MAX_AGG_TOKENS:
        raise ValueError(f"target {target_tokens} exceeds the ruled cap "
                         f"{MAX_AGG_TOKENS}")
    rng = random.Random(seed)
    parts: list[str] = []
    labels: list[Label] = []
    sealed = 0
    idx = 0
    running = 0
    # Counted INCREMENTALLY: count each new record, keep a running sum, and
    # reconcile once at the end. Re-counting the whole accumulated text every
    # iteration is O(n^2) in both calls and bytes -- against the real leaf
    # tokenizer that is ~1,500 HTTP round trips per corpus, each shipping up to
    # half a megabyte, which is the difference between a build that takes a
    # minute and one that takes an hour.
    #
    # A running sum can drift from the true count by tokenizer boundary
    # effects, so the exact count is taken once below and whole records are
    # dropped until it fits. Dropping RECORDS rather than trimming text is what
    # keeps the corpus and the ANSWER consistent: the ground truth is derived
    # from the records that survive, not from the ones that were generated.
    while running < target_tokens:
        label: Label = "withheld" if rng.random() < 0.45 else "released"
        rec = "\n" + _record(rng, idx, label)
        n = count(rec)
        if running + n > target_tokens:
            break
        parts.append(rec)
        labels.append(label)
        sealed += "Status: SEALED" in rec
        running += n
        idx += 1
    text = "".join(parts).lstrip("\n")
    while parts and count(text) > target_tokens:
        dropped = parts.pop()
        labels.pop()
        sealed -= "Status: SEALED" in dropped
        text = "".join(parts).lstrip("\n")
    return AggregationCorpus(
        text=text, n_records=len(labels), sealed_count=sealed,
        withheld_count=sum(1 for x in labels if x == "withheld"),
        seed=seed, counter_name=counter_name,
        measured_tokens=count(text), labels=labels)


# --------------------------------------------------------------------------- #
# §8's at-chance demonstration, as an artifact rather than a claim


# Candidates chosen to be the ones an author would actually reach for, plus the
# two strongest attacks on this corpus's specific shape. `phrase-lookup` is the
# important one: it enumerates all six withheld flourishes verbatim, which is
# the attack that beat the previous word-permutation design. It is at chance
# here only because the flourish no longer carries the label -- the direction of
# custody does, and that needs the header.
_CANDIDATE_REGEXES = {
    "withholding": r"withholding",
    "denied": r"denied",
    "embargo": r"embargo",
    "sealing": r"sealing",
    "negation-any": r"\b(not|denied|refusal)\b",
    "phrase-lookup": "|".join(re.escape(w) for w, _ in _PAIRS),
    "release (inverted)": r"release|disclosure|grant",
}


def regex_at_chance(corpus: AggregationCorpus) -> dict[str, float]:
    """Score every plausible pure-regex solution against the ground truth.

    §8 requires the regex-defeating task be "verified at authoring by showing a
    pure-regex solution scores at chance". This computes that verification and
    returns it, so the manifest can record numbers instead of an assurance.

    Accuracy is per-RECORD classification, not the final count: a regex could
    land on the right total by cancelling errors, and that would be luck rather
    than a solution. Chance here is max(p, 1-p) for the label prior -- always
    predicting the majority class -- so a regex only beats chance by actually
    separating the sentences.
    """
    records = [r for r in corpus.text.split("\n\n") if "Disposition:" in r]
    assert len(records) == len(corpus.labels), (
        f"{len(records)} parsed records vs {len(corpus.labels)} labels")
    out: dict[str, float] = {}
    for name, pattern in _CANDIDATE_REGEXES.items():
        rx = re.compile(pattern, re.I)
        hits = 0
        for rec, label in zip(records, corpus.labels):
            line = rec.split("Disposition:", 1)[1].split("\n", 1)[0]
            predicted: Label = "withheld" if rx.search(line) else "released"
            if name == "granted (inverted)":
                predicted = "released" if rx.search(line) else "withheld"
            hits += predicted == label
        out[name] = round(hits / len(records), 4)
    p = corpus.withheld_count / max(1, corpus.n_records)
    out["__chance__"] = round(max(p, 1 - p), 4)
    return out


def foreign_identifier_overlap(corpus: AggregationCorpus,
                               other_texts: list[str]) -> set[str]:
    """Literal half of the disjointness check: identifiers this corpus shares
    with any existing fixture. The structural syllable check in `bench.vocab`
    cannot see a UUID or ENT- code collision, and that is the collision that
    would actually let a leak score as a pass."""
    ours = harvest_names(corpus.text)
    theirs: set[str] = set()
    for t in other_texts:
        theirs |= harvest_names(t)
    return ours & theirs


__all__ = ["AggregationCorpus", "build", "regex_at_chance",
           "foreign_identifier_overlap", "MAX_AGG_TOKENS", "MAX_SUBCALLS",
           "coined_name"]


# --------------------------------------------------------------------------- #
# The other category shapes. Same vocabulary and same record grammar as the
# aggregation corpus, so a model cannot tell the categories apart by style --
# only the QUESTION differs, which is what makes the per-category table in §8's
# report mean something.


@dataclass
class NeedleCorpus:
    text: str
    entity: str
    key: str
    needle_offset: int
    seed: int
    measured_tokens: int
    counter_name: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def build_needle(seed: int, target_tokens: int,
                 count: Callable[[str], int], counter_name: str,
                 position: float = 0.68) -> NeedleCorpus:
    """One unique custody key, planted at a fraction of the way through.

    §8 lets needle tasks keep the 200K-1M range because "a needle does not
    require full coverage -- finding it by REPL prescan and targeted windows is
    not sampling, it is the thesis". So this corpus is deliberately LARGER than
    an aggregation one and is not bound by the 130K cap.

    The needle is unique by construction: its entity appears exactly once, and
    the key is a UUID drawn from this corpus's own stream, so `uuid_exact` can
    grade it and a leak from anywhere else scores as a miss rather than a pass.
    """
    rng = random.Random(seed)
    entity = organisation(rng)
    key = "%08x-%04x-%04x-%04x-%012x" % (
        rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(16),
        rng.getrandbits(16), rng.getrandbits(48))
    needle = (f"\n[ENT-{900000 + seed}] {entity}\n"
              f"Status: SEALED\n"
              f"Custody key of record: {key}.\n"
              f"Notes: this key is issued once and is not reproduced elsewhere.\n")

    parts: list[str] = []
    idx = 0
    running = 0
    while running < target_tokens:          # incremental -- see build()
        label: Label = "withheld" if rng.random() < 0.45 else "released"
        rec = "\n" + _record(rng, 200000 + idx, label)
        running += count(rec)
        parts.append(rec)
        idx += 1
    body = "".join(parts)
    cut = int(len(body) * position)
    cut = body.find("\n\n[", cut)          # snap to a record boundary
    if cut < 0:
        cut = len(body)
    text = (body[:cut] + "\n" + needle + body[cut:]).lstrip("\n")
    if text.count(entity) != 1:
        raise AssertionError(f"needle entity {entity!r} is not unique in the "
                             f"corpus ({text.count(entity)} occurrences)")
    return NeedleCorpus(text=text, entity=entity, key=key,
                        needle_offset=text.find(key), seed=seed,
                        measured_tokens=count(text), counter_name=counter_name)


@dataclass
class SynthesisCorpus:
    docs: list[str]
    answer: str                  # the organisation present in EVERY document
    n_docs: int
    seed: int
    measured_tokens: int
    counter_name: str

    @property
    def text(self) -> str:
        return "\n\n=== DOCUMENT BREAK ===\n\n".join(self.docs)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def build_synthesis(seed: int, n_docs: int, tokens_per_doc: int,
                    count: Callable[[str], int],
                    counter_name: str) -> SynthesisCorpus:
    """Several documents; the answer is the one organisation appearing in ALL.

    §8 asks for "multi-document synthesis with checkable claims", and the check
    has to be programmatic. An intersection is: no single document contains the
    answer, every document is needed, and the ground truth is a name that
    `name_exact` grades. A root that reads one document and guesses cannot pass,
    and a root that reads all of them can.
    """
    rng = random.Random(seed)
    shared = organisation(rng)
    docs: list[str] = []
    for d in range(n_docs):
        parts = [f"REGISTER {d + 1} of {n_docs}\n"]
        idx = 0
        running = 0
        while running < tokens_per_doc:     # incremental -- see build()
            label: Label = "withheld" if rng.random() < 0.45 else "released"
            rec = "\n" + _record(rng, 300000 + d * 10000 + idx, label)
            running += count(rec)
            parts.append(rec)
            idx += 1
        # The shared organisation appears once per document, at a varying depth,
        # so it cannot be found by position.
        at = rng.randrange(2, max(3, len(parts)))
        parts.insert(at, f"\n[ENT-{800000 + d}] {shared}\nStatus: OPEN\n"
                         f"Disposition: entered on this register.\n")
        docs.append("".join(parts))
    corpus = SynthesisCorpus(docs=docs, answer=shared, n_docs=n_docs, seed=seed,
                             measured_tokens=0, counter_name=counter_name)
    corpus.measured_tokens = count(corpus.text)
    for d in docs:
        if d.count(shared) != 1:
            raise AssertionError("the shared organisation must appear exactly "
                                 "once per document")
    return corpus
