from pathlib import Path
from bench.corpus_v2 import load_trec, TREC_LABELS

REPO = Path(__file__).resolve().parents[1]


def test_the_vendored_label_source_is_pinned_and_shaped():
    items = load_trec()
    assert len(items) == 5452
    assert {i.label for i in items} == set(TREC_LABELS) == {"ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"}
    assert all(3 <= len(i.text.split()) <= 60 for i in items)
    sha = (REPO / "bench/sources/trec/trec_train.sha256").read_text().split()[0]
    import hashlib
    assert hashlib.sha256((REPO / "bench/sources/trec/trec_train.jsonl").read_bytes()).hexdigest() == sha
