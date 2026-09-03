import pytest

from pathlib import Path
from bench.corpus_v2 import load_trec, TREC_LABELS, build_linear_semantic
from bench.tokens import approx_tokens

REPO = Path(__file__).resolve().parents[1]


def test_the_vendored_label_source_is_pinned_and_shaped():
    items = load_trec()
    assert len(items) == 5452
    assert {i.label for i in items} == set(TREC_LABELS) == {"ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"}
    assert all(3 <= len(i.text.split()) <= 60 for i in items)
    sha = (REPO / "bench/sources/trec/trec_train.sha256").read_text().split()[0]
    import hashlib
    assert hashlib.sha256((REPO / "bench/sources/trec/trec_train.jsonl").read_bytes()).hexdigest() == sha


def test_linear_semantic_answer_is_computed_from_labels_and_the_label_never_appears():
    items = load_trec()
    c = build_linear_semantic(seed=9101, target_tokens=20_000, count=approx_tokens,
                              counter_name="approx-offline", question_kind="count_label", items=items)
    assert c.answer == str(sum(1 for l in c.labels if l == c.target[0]))
    for lab in TREC_LABELS:
        assert lab not in c.text                     # the class name is nowhere in the register
    assert c.text.count("Query:") == len(c.items) == len(c.labels) == len(c.record_ids)
    assert c.measured_tokens <= 20_000


def test_two_seeds_sample_different_items_and_build_is_deterministic():
    items = load_trec()
    a = build_linear_semantic(9101, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    b = build_linear_semantic(9102, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    a2 = build_linear_semantic(9101, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    assert a.sha256 == a2.sha256 and a.sha256 != b.sha256
    assert set(i.text for i in a.items) != set(i.text for i in b.items)


def test_most_common_label_answers_a_canonical_word():
    items = load_trec()
    c = build_linear_semantic(9111, 8_000, approx_tokens, "approx-offline", question_kind="most_common_label", items=items)
    from collections import Counter
    top = Counter(c.labels).most_common(1)[0][0]
    assert c.answer == {"HUM": "person", "LOC": "place", "NUM": "number", "ENTY": "entity",
                        "DESC": "description", "ABBR": "abbreviation"}[top]
    assert c.checker == "name_exact"


# Fix round 1: measured_tokens must be bounded by target_tokens INCLUDING the
# question, not just the bare records. These two seeds were reproduced as
# known-failing before the fix (8002 and 8012 tokens respectively, both over
# the 8000 budget) -- kept as regression material.
@pytest.mark.parametrize("seed,kind", [(9294, "count_label"), (9130, "count_two_labels")])
def test_measured_tokens_respects_target_tokens_on_known_failing_seeds(seed, kind):
    items = load_trec()
    c = build_linear_semantic(seed, 8_000, approx_tokens, "approx-offline", question_kind=kind, items=items)
    assert c.measured_tokens <= 8_000


@pytest.mark.parametrize("kind", ["count_label", "count_two_labels", "most_common_label"])
def test_measured_tokens_respects_target_tokens_at_a_tight_budget(kind):
    # Small enough that the question's own tokens are a meaningful fraction of
    # the budget, so an unaccounted question would overshoot every time
    # rather than by luck of a large margin.
    items = load_trec()
    for seed in range(20):
        c = build_linear_semantic(seed, 500, approx_tokens, "approx-offline", question_kind=kind, items=items)
        assert c.measured_tokens <= 500


def test_most_common_label_fails_clearly_when_no_record_fits():
    items = load_trec()
    with pytest.raises(ValueError, match="too small to fit even one record"):
        build_linear_semantic(1, 75, approx_tokens, "approx-offline", question_kind="most_common_label", items=items)
