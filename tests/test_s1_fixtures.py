import json
from pathlib import Path

S1 = Path(__file__).resolve().parents[1] / "s1"


def test_needle_fixture_is_at_least_64k_leaf_tokens():
    meta = json.loads((S1 / "tasks" / "needle.json").read_text())
    assert meta["tokenized_len"] >= 64_000  # asserted programmatically, spec §9
    assert len(meta["context"]) >= 250_000 or Path(meta["context_path"]).exists()


def test_control_truncation_rule_is_deterministic_and_drops_the_needle():
    from s1.make_fixtures import control_truncate
    meta = json.loads((S1 / "tasks" / "needle.json").read_text())
    text = Path(meta["context_path"]).read_text(encoding="utf-8")
    a = control_truncate(text, 28_000)
    assert a == control_truncate(text, 28_000)
    assert meta["answer"] not in a  # needle beyond any retained region


def test_paraphrase_needle_defeats_regex():
    meta = json.loads((S1 / "tasks" / "paraphrase.json").read_text())
    text = Path(meta["context_path"]).read_text(encoding="utf-8")
    assert meta["answer"] not in text  # must require a leaf call, not a scan


def test_fixtures_are_reproducible_from_seed():
    from s1.make_fixtures import build
    assert build(seed=1)["needle"]["answer"] == build(seed=1)["needle"]["answer"]
