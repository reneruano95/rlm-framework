"""The instruction-decay experiment (`s2/run_distance.py`,
`s2/make_distance_fixtures.py`).

Two things in this experiment can quietly make it measure nothing, and both are
pinned here rather than trusted:

  1. **The layouts.** FACTOR 1 is POSITION, so the three arms must differ in
     where `leaf-prefix.v1.md` sits and in NOTHING else — same bytes, same
     question, chunk still first in the user message.
  2. **The density control.** FACTOR 3 exists to break the confound between
     distance and distractor count; a matched-density cell that does not hold
     its entity budget flat across sizes separates nothing, and the experiment
     would report the confound as the result.
"""
from __future__ import annotations

import pytest

from s1.make_fixtures import approx_tokens
from s2.leafcall import LAYOUTS, PinnedLeafCaller
from s2.make_distance_fixtures import (
    DENSITIES,
    MATCHED_ENTITIES,
    SIZES,
    build_cell,
    n_bindings,
    neutral_paragraph,
    place_entities,
)
from s2.run_distance import (
    ARMS,
    LEAF_PREFIX,
    fisher_exact_two_sided,
    render_report,
    wilson_upper95,
)
from s2.run_sweep import CORRECT, FALSE_POSITIVE

PREFIX = "RULES: answer only from the excerpt."


def caller(layout: str) -> PinnedLeafCaller:
    c = PinnedLeafCaller(client=None, system_prefix=PREFIX, max_predict=512,
                         temperature=0.3, top_p=0.9, slot_capacity_tokens=2560,
                         layout=layout)
    c.prefix_tokens = 320
    c.prefix_body_tokens = 300
    return c


# --------------------------------------------------------------------------- #
# FACTOR 1 — the layouts
# --------------------------------------------------------------------------- #


def test_the_three_arms_are_three_layouts_of_ONE_prefix():
    assert {a[1] for a in ARMS} == set(LAYOUTS)
    assert LEAF_PREFIX == "prompts/leaf-prefix.v1.md"


def test_layout_A_is_the_shipped_layout_byte_for_byte():
    msgs = caller("A").compose(question="Q?", chunk="CHUNK")
    assert msgs == [{"role": "system", "content": PREFIX},
                    {"role": "user", "content": "CHUNK\n\nQ?"}]


def test_layout_B_repeats_the_prefix_after_the_chunk_and_keeps_the_head():
    msgs = caller("B").compose(question="Q?", chunk="CHUNK")
    assert msgs[0] == {"role": "system", "content": PREFIX}
    user = msgs[1]["content"]
    assert user.startswith("CHUNK")          # §4's cache contract: chunk first
    assert user.endswith("Q?")               # question LAST
    assert user.count(PREFIX) == 1           # the repeat, in the user segment


def test_layout_C_drops_the_leading_prefix_entirely():
    msgs = caller("C").compose(question="Q?", chunk="CHUNK")
    assert [m["role"] for m in msgs] == ["user"]
    assert msgs[0]["content"] == f"CHUNK\n\n{PREFIX}\n\nQ?"


def test_every_layout_carries_the_SAME_prefix_text():
    bodies = [ "".join(m["content"] for m in caller(l).compose(
        question="Q?", chunk="CHUNK")) for l in LAYOUTS]
    assert all(PREFIX in b for b in bodies)


def test_an_unknown_layout_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        caller("D").compose(question="Q?", chunk="CHUNK")


def test_layout_B_costs_slot_capacity_and_admission_knows_it():
    """The repeated prefix is not free: it is priced in the head, so a cell the
    shipped layout admits can be inadmissible under B. Reporting that as an
    exclusion is honest; silently sending an over-long prompt is not."""
    a, b, c = caller("A"), caller("B"), caller("C")
    assert a.head_tokens() == 320
    assert b.head_tokens() == 620          # rendered head + the prefix body again
    assert c.head_tokens() == 320          # no system head, same markup
    assert a.admits(2048) and not b.admits(2048)
    assert all(x.admits(1024) for x in (a, b, c))


# --------------------------------------------------------------------------- #
# FACTOR 3 — the density control
# --------------------------------------------------------------------------- #


def test_matched_density_is_flat_in_size_and_natural_is_not():
    for size in SIZES:
        assert n_bindings(size, "matched") == MATCHED_ENTITIES
    natural = [n_bindings(s, "natural") for s in SIZES]
    assert natural == sorted(natural) and natural[0] < natural[-1]


def test_the_natural_rate_reproduces_the_fixtures_the_AB_actually_ran():
    """3 bindings per 640, 6 per 1,024, 11 per 2,048 — counted in
    `s2/fixtures-refusal-640-s*`, `s2/fixtures-refusal-s*` and `s2/fixtures`."""
    assert [n_bindings(s, "natural") for s in (640, 1024, 2048)] == [3, 6, 11]


def test_neutral_filler_carries_no_entity_binding():
    import random
    rng = random.Random("neutral")
    for _ in range(50):
        assert "ENT-" not in neutral_paragraph(rng)


def test_entities_are_spread_not_clustered():
    filler = "\n\n".join(f"p{i}" for i in range(12))
    out = place_entities(filler, ["H1", "H2", "H3"])
    hit = [i for i, para in enumerate(out.split("\n\n")) if para.startswith("H")]
    assert len(hit) == 3
    assert max(hit) - min(hit) >= 6      # spread across the document, not adjacent


@pytest.mark.parametrize("density", DENSITIES)
def test_a_built_cell_holds_its_own_density_level(density):
    """Built with the offline proxy — the assertions under test are about entity
    COUNT and the questions, neither of which depends on the tokenizer."""
    cell = build_cell(approx_tokens, size_tokens=1024, density=density, seed=7)
    assert cell["entity_bindings"] == n_bindings(1024, density)
    assert cell["density"] == density
    assert set(cell["questions"]) == {"literal", "paraphrase", "absent"}
    assert cell["questions"]["absent"]["expected"] is None


def test_the_absent_organisation_is_absent_at_both_densities():
    for density in DENSITIES:
        cell = build_cell(approx_tokens, size_tokens=640, density=density, seed=7)
        text = (cell["text"]).lower()
        assert cell["questions"]["absent"]["entity"].lower() not in text
        assert cell["questions"]["literal"]["expected"] in cell["text"]
        assert cell["questions"]["paraphrase"]["expected"] not in cell["text"]


def test_a_cell_that_misses_its_density_level_is_refused():
    with pytest.raises(AssertionError):
        build_cell(approx_tokens, size_tokens=640, density="matched", seed=7,
                   matched_entities=99)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def test_fisher_matches_known_values():
    # The A/B's own headline table: 30/30 vs 0/21.
    assert fisher_exact_two_sided(30, 0, 0, 21) < 1e-13
    # A table with no difference at all.
    assert fisher_exact_two_sided(5, 5, 5, 5) == pytest.approx(1.0)
    # The v2 comparison the A/B reported as p = 1.0 (29/30 vs 30/30).
    assert fisher_exact_two_sided(29, 1, 30, 0) == pytest.approx(1.0)


def test_zero_observed_is_never_reported_as_zero():
    assert wilson_upper95(0, 21) > 0.10
    assert wilson_upper95(0, 0) is None


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def _rec(**kw):
    base = dict(phase="grid", arm="A-shipped", layout="A", status="ok",
                cell_uid="s2d-1024-matched#s11", cell_id="s2d-1024-matched",
                fixture_seed=11, size_target=1024, size_measured=1024,
                density="matched", entity_bindings=3, position=0.5,
                question_type="absent", question="q", expected=None,
                expected_kind="uuid", trial=1, call_idx=0, cold=True,
                raw_output="8243843a-ecc2-4f29-9122-60b53028b36b",
                tokens_in=1400, tokens_cached=0, tokens_out=33, wall_s=1.2,
                requested_slot=1, slot_ok=True, leak_detected=False)
    base.update(kw)
    return base


def test_the_report_prints_recall_beside_every_false_positive_rate():
    """An arm that buys 0% false positives with 0% recall must be visible as
    such: the report may never print the first number alone."""
    records = [
        _rec(),
        _rec(question_type="literal", expected="8243843a-ecc2-4f29-9122-60b53028b36b",
             raw_output="8243843a-ecc2-4f29-9122-60b53028b36b"),
        _rec(question_type="paraphrase", expected_kind="name", expected="Ann Bee",
             raw_output="NONE"),
    ]
    out = render_report(records)
    assert "FALSE-POS" in out and "RECALL" in out
    assert "DISTANCE vs DENSITY" in out


def test_the_report_re_derives_labels_rather_than_trusting_stored_ones():
    """A stored label that disagrees with the raw output must lose."""
    out = render_report([_rec(label=CORRECT)])
    assert FALSE_POSITIVE in out


def test_an_inadmissible_cell_is_reported_not_dropped():
    out = render_report([_rec(arm="B-repeated", layout="B", status="inadmissible",
                              size_target=2048, head_tokens=620,
                              slot_capacity_tokens=2560, label=None)])
    assert "INADMISSIBLE" in out


def test_slot_reuse_across_two_cells_is_audited():
    out = render_report([_rec(requested_slot=4),
                         _rec(requested_slot=4, cell_uid="other#s12")])
    assert "s2d-1024-matched#s11" in out.split("Never-reused-slot check:")[1][:400]
