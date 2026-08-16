"""The benchmark corpus generator's build-time guarantees, as tests.

Everything here is a property §8 requires of an aggregation corpus and that is
cheaper to enforce than to discover: a corpus one window over the ruled cap
makes every episode using it a `budget_kill` (a FAILURE for every arm), and a
corpus whose names collide with a fixture's lets a leak score as a pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import corpus as bc
from bench.vocab import SYL_A, SYL_B, SYL_C, assert_disjoint_from_fixtures
from s1.make_fixtures import approx_tokens

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    c = bc.build(seed=7001, target_tokens=bc.MAX_AGG_TOKENS,
                 count=approx_tokens, counter_name="approx-offline")
    c.assert_affordable()
    return c


def test_the_name_space_is_disjoint_from_every_fixture_pool():
    """§8: S2's gates run on "dedicated non-benchmark fixtures so S2 cannot
    overfit the benchmark it is authoring". That holds only if the names cannot
    collide. This ran at import and found four collisions on its first run --
    keep it as a test too, because the pools are editable."""
    assert_disjoint_from_fixtures()


def test_the_disposition_pairs_are_exact_word_permutations():
    bc.assert_pairs_are_permutations()


def test_the_corpus_respects_both_ruled_bounds(built):
    """The 2026-08-15 ruling: <=130,464 tokens and <=926 sub-calls for full
    coverage. Asserted at build time; asserted again here so a change to the
    record shape cannot quietly push a corpus over."""
    assert built.measured_tokens <= bc.MAX_AGG_TOKENS
    assert built.subcalls <= bc.MAX_SUBCALLS
    assert built.windows == -(-built.measured_tokens // bc.SNAP_STRIDE)


def test_the_ground_truth_is_recoverable_from_the_text_alone(built):
    """The answer must be re-derivable by a reader of the corpus, not only by
    the generator that made it -- otherwise the benchmark grades against a
    number nobody can check."""
    records = [r for r in built.text.split("\n\n") if "Disposition:" in r]
    assert len(records) == built.n_records

    sealed = sum(1 for r in records if "Status: SEALED" in r)
    assert sealed == built.sealed_count

    withheld = 0
    for r in records:
        owner = r.split("]", 1)[1].split("\n", 1)[0].strip()
        disp = r.split("Disposition:", 1)[1]
        # withheld <=> custody passed TO the owner
        if disp.split(" to the ", 1)[1].startswith(owner):
            withheld += 1
    assert withheld == built.withheld_count


def test_no_pure_regex_beats_chance_on_the_semantic_question(built):
    """§8 requires the regex-defeating task be "verified at authoring by showing
    a pure-regex solution scores at chance". This IS that verification.

    `phrase-lookup` is the candidate that matters: it enumerates every
    disposition flourish verbatim, which defeated an earlier design where the
    flourish tracked the label. It is at chance now because the label is carried
    by the DIRECTION of custody relative to the record's own owner -- a join a
    regex cannot perform."""
    scores = bc.regex_at_chance(built)
    chance = scores.pop("__chance__")
    for name, acc in scores.items():
        assert acc <= chance + 0.02, (
            f"regex {name!r} scores {acc:.3f} against chance {chance:.3f} -- "
            f"the task is not regex-defeating and fails §8's aggregation rule")


def test_the_regex_solvable_question_really_is_regex_solvable(built):
    """The other half of §8's aggregation rule: at least one task must be
    solvable in code, "so the benchmark rewards the root choosing code over leaf
    calls when code suffices". A root that spends 600 leaf calls on this one is
    choosing badly and the benchmark should be able to say so."""
    assert built.text.count("Status: SEALED") == built.sealed_count


def test_building_is_deterministic_for_a_seed():
    a = bc.build(seed=99, target_tokens=20_000, count=approx_tokens,
                 counter_name="approx-offline")
    b = bc.build(seed=99, target_tokens=20_000, count=approx_tokens,
                 counter_name="approx-offline")
    assert a.sha256 == b.sha256
    assert (a.n_records, a.sealed_count, a.withheld_count) == \
           (b.n_records, b.sealed_count, b.withheld_count)


def test_no_identifier_is_shared_with_any_fixture_on_disk(built):
    """The literal half of disjointness. The syllable check cannot see a UUID or
    ENT- code collision, and that is the collision that would actually let a
    stale slot or a leaked answer score as a pass."""
    others: list[str] = []
    for pat in ("s1/tasks/*.txt", "s2/fixtures*/**/*.txt"):
        for p in REPO.glob(pat):
            others.append(p.read_text(encoding="utf-8", errors="replace"))
    assert others, "no fixture corpora found to compare against"
    overlap = bc.foreign_identifier_overlap(built, others)
    assert not overlap, f"benchmark shares identifiers with a fixture: {sorted(overlap)[:5]}"


def test_a_corpus_over_the_cap_is_refused_rather_than_truncated():
    with pytest.raises(ValueError, match="exceeds the ruled cap"):
        bc.build(seed=1, target_tokens=bc.MAX_AGG_TOKENS + 1,
                 count=approx_tokens, counter_name="approx-offline")
