"""R13's foreign-string detector -- the DETECTION half of the R13 mitigation
(spec §10 R13 (3), §5 C4 "slot discipline").

WHY THIS EXISTS. The leaf server returns content from documents previously
held on the same slot: measured under a paired control in one process with
byte-identical prompts, a shared slot leaked 24/54 and a virgin slot 0/54
(Fisher p = 4.4e-9, `milestones/s2/R13.md` §1). Prevention is a slot-allocation rule
(`rlm.dispatcher.SlotPool`); this module is the free check that runs beside
it, because the prevention is bounded rather than proven: **138 virgin-slot
calls with zero leaks give a 95% upper bound of 2.2%, not zero, and a 200K
episode is ~848 leaf calls -- so the evidence permits roughly 19 contaminated
answers per episode.** Nothing in this file, and nothing that reports on it,
may say "leak-free"; the bound is the claim.

WHAT MAKES IT FREE. The scaffold holds every chunk of the corpus (C2 owns the
chunker), so leakage is deterministically detectable at zero model cost: an
identifier-shaped token in a leaf answer that is ABSENT from what was sent and
PRESENT in another chunk cannot have come from this call's document. One set
membership test per candidate token, no model call, no second opinion. This is
strictly stronger than R5's evidence-span check (measured 11% catch rate)
because it targets this failure exactly, and it doubles as §8's in-benchmark
contamination monitor -- the per-arm hit count is part of the S4 verdict.

THE TWO STATED LIMITS (§10 R13 (3)). Both are properties of the method, not
bugs to be fixed later, and both are pinned by tests in
`tests/test_leakcheck.py`:

  1. **Paraphrased leakage is invisible.** An answer that carries another
     chunk's content in its own words carries none of that chunk's
     identifier-shaped strings, and no widening of the pattern set changes
     that.
  2. **A contaminated REFUSAL is invisible**, and this is a real observed
     case, not a hypothetical -- `milestones/s2/R13.md` §4 quotes an answer that
     correctly refuses (the fact really was absent) while enumerating four
     entity names belonging to two documents held earlier on the same slot.
     The names are not identifier-shaped, and a proper-noun class would flag
     ordinary capitalised English shared across chunks. The reproducer's own
     oracle could afford one (`_PROPER` in `milestones/s2/r13_repro.py`) only because its
     fixture corpus is procedurally generated coinages; a real corpus is not.

So a clean verdict is evidence, not a certificate. `detected is None`
(`NOT_CHECKED`) is the third value on purpose: no corpus means not checked,
and it must never be recorded as False, which reads as "checked and clean".

THE PATTERN SET is derived from the strings that actually leaked in
`milestones/s2/R13.md`, not from imagination:

  * UUIDs -- `8243843a-ecc2-4f29-9122-60b53028b36b` (§4, the true key),
    `1251d802-86aa-4e75-96be-aefc175c1e8e` (§4, the leaked one);
  * `ENT-#####` codes -- §10 R13's strongest artifact is "eight
    `ENT-#####:hex` pairs whose bindings are split verbatim between the
    current document and one previously held on that slot";
  * hex runs -- the other half of those pairs;
  * long alphanumeric tokens -- a mixed letter+digit run of 12+ characters,
    which is what an opaque key looks like when it is neither a UUID nor hex.
    The digit requirement is what keeps ordinary English words out.

This module is deliberately ISOLATED (`tests/test_import_rules.py`): it
imports nothing but `re` and the stdlib, so the trace/analysis side can import
it without pulling in an HTTP client, and C4 can call it on every answer.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

#: One alternation, tried in this order, so a UUID is captured whole rather
#: than as its constituent hex runs. Case-insensitive: the leaked strings and
#: their chunk-side originals differ in case in real answers (`milestones/s2/R13.md` §4).
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_ENT_CODE = r"\bENT-\d{4,6}\b"
_HEX_RUN = r"\b[0-9a-f]{8,}\b"
#: Long mixed alphanumeric: at least one digit AND at least one letter, so
#: "responsibilities" is not an identifier and "k7x2mq9v4zt1" is.
_ALNUM_RUN = r"\b(?=[0-9a-z]*\d)(?=[0-9a-z]*[a-z])[0-9a-z]{12,}\b"

IDENTIFIER_RE = re.compile(
    "|".join((_UUID, _ENT_CODE, _HEX_RUN, _ALNUM_RUN)), re.IGNORECASE)

#: `leak_detail` is a TEXT column read by humans, not a dump of every hit --
#: `LeakVerdict.hits` carries the full list for analysis.
DETAIL_CAP = 512


def identifier_tokens(text: str) -> set[str]:
    """Every identifier-shaped token in `text`, as written (case preserved).

    Pure and cheap: one regex pass. Ordinary prose yields the empty set, which
    is what keeps the false-positive floor at zero for English answers."""
    if not text:
        return set()
    return set(IDENTIFIER_RE.findall(text))


@dataclass(frozen=True, slots=True)
class LeakHit:
    """One identifier that was absent from what was sent and present in
    `source` -- another chunk of the same corpus."""

    token: str
    source: str


@dataclass(frozen=True, slots=True)
class LeakVerdict:
    """`detected` is tri-state: True (foreign identifiers found), False
    (checked, none found -- evidence, not a certificate; see the 2.2% bound in
    the module docstring), None (NOT checked -- no corpus was available)."""

    detected: bool | None
    detail: str | None = None
    hits: tuple[LeakHit, ...] = ()


#: The "no corpus, so nothing was checked" verdict. A singleton so a caller can
#: test identity, and so `detected=False` never stands in for it.
NOT_CHECKED = LeakVerdict(detected=None)
_CLEAN = LeakVerdict(detected=False)


@dataclass(frozen=True, slots=True)
class ChunkIndex:
    """token -> the chunk it lives in, built ONCE per corpus.

    This is what makes the check affordable on every leaf call. Detection then
    costs one regex pass over the ANSWER plus a dict lookup per candidate --
    O(answer), not O(corpus). Re-scanning the corpus per answer would cost
    O(corpus) on each of ~848 leaf calls in a 200K-token episode
    (`milestones/s2/R13-mitigations.md` §8.1), which is the difference between a free
    check and a check nobody leaves switched on.

    `by_token` maps the LOWERCASED token to the first chunk id (in corpus
    iteration order) that contains it. Chunk texts are not retained.
    `chunk_count` is kept so an index built from a real corpus that happens to
    contain no identifiers at all reports CHECKED-and-clean, while an index
    built from nothing reports NOT CHECKED -- the two are not the same claim.
    """

    by_token: Mapping[str, str] = field(default_factory=dict)
    chunk_count: int = 0

    @classmethod
    def from_chunks(cls, chunks: Mapping[str, str] | Sequence[str] | Iterable[str]) -> "ChunkIndex":
        """Build from C2's chunk list (ids become `chunk[i]`) or from an
        explicit {chunk_id: text} mapping."""
        items: Iterable[tuple[str, str]]
        if isinstance(chunks, Mapping):
            items = chunks.items()
        else:
            items = ((f"chunk[{i}]", text) for i, text in enumerate(chunks))
        by_token: dict[str, str] = {}
        count = 0
        for chunk_id, text in items:
            count += 1
            for token in identifier_tokens(text):
                by_token.setdefault(token.lower(), chunk_id)
        return cls(by_token=by_token, chunk_count=count)

    def foreign(self, answer: str, *, sent: str) -> LeakVerdict:
        """R13's oracle, verbatim in its definition (`milestones/s2/R13.md` §2): a leak is
        a string that is (a) absent from the document that was actually sent,
        (b) absent from the question, and (c) present in another document of
        the corpus. `sent` is therefore the WHOLE user segment -- chunk and
        question together -- so a model quoting the question back is never a
        hit."""
        if not self.chunk_count:
            return NOT_CHECKED
        own = {t.lower() for t in identifier_tokens(sent)}
        hits = tuple(
            LeakHit(token, self.by_token[token.lower()])
            for token in sorted(identifier_tokens(answer))
            if token.lower() not in own and token.lower() in self.by_token
        )
        if not hits:
            return _CLEAN
        return LeakVerdict(detected=True, detail=_format_detail(hits), hits=hits)


def foreign_identifiers(answer: str, *, sent: str,
                        corpus: Mapping[str, str] | Sequence[str]) -> LeakVerdict:
    """The detector as ONE pure function: identifier-shaped tokens in `answer`
    that are absent from `sent` and present in `corpus`.

    Equivalent to `ChunkIndex.from_chunks(corpus).foreign(answer, sent=sent)`,
    and tested against it. Use the index directly when the same corpus is
    checked more than once (C4 does, on every leaf call)."""
    return ChunkIndex.from_chunks(corpus).foreign(answer, sent=sent)


def _format_detail(hits: tuple[LeakHit, ...]) -> str:
    detail = (f"{len(hits)} foreign identifier(s): "
              + "; ".join(f"{h.token}@{h.source}" for h in hits))
    if len(detail) > DETAIL_CAP:
        detail = detail[:DETAIL_CAP - 3] + "..."
    return detail
