import pytest

from rlm.context.chunker import ChunkConfig
from rlm.context.interactive import (
    MAX_DOC_ID_CHARS, MAX_TITLE_CHARS, InteractiveIndex, Hit)

CFG = ChunkConfig(size_tokens=64, overhead_tokens=0, snap_to_boundary=True,
                  snap_tolerance=0.10, stride_tokens=48)
count = lambda s: (len(s) + 3) // 4

TEXT = ("=== DOCUMENT d-01: Alpha register ===\n" + "alpha " * 300 + "\n\n"
        "=== DOCUMENT d-02: Beta register ===\n" + "beta tithe barn " * 200)


def test_documents_are_split_on_the_header_and_windowed_with_c2():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    assert ix.n_docs == 2 and set(ix.docs) == {"d-01", "d-02"}
    assert ix.open("d-02") == {"doc_id": "d-02", "title": "Beta register",
                               "n_windows": len(ix.windows["d-02"]), "n_chars": len(ix.docs["d-02"])}
    assert ix.open("d-02")["n_windows"] > 3


def test_search_returns_locations_not_text_and_is_capped():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    hits = ix.search("TITHE", max_hits=5)
    assert len(hits) == 5 and all(isinstance(h, Hit) and h.doc_id == "d-02" for h in hits)
    assert ix.search("nowhere") == []


def test_window_is_range_checked_never_clamped():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    n = ix.open("d-01")["n_windows"]
    assert ix.window("d-01", n - 1)
    with pytest.raises(IndexError):
        ix.window("d-01", n)
    with pytest.raises(KeyError):
        ix.window("d-99", 0)


def test_a_pathologically_long_header_line_is_bounded_by_construction():
    """A well-behaved corpus never triggers this, but the guarantee has to
    hold regardless of corpus content -- v2's own adversarial tasks
    (int-05/int-06) put an injection record in a document. `open()` must
    never echo an unbounded, corpus-derived string back to the sandbox."""
    long_id = "y" * 5000
    long_title = "X" * 5000
    text = f"=== DOCUMENT {long_id}: {long_title} ===\nsome body text\n"
    ix = InteractiveIndex.from_text(text, CFG, count)
    (doc_id,) = ix.docs
    assert len(doc_id) <= MAX_DOC_ID_CHARS
    result = ix.open(doc_id)
    assert len(result["title"]) <= MAX_TITLE_CHARS


def test_search_raises_rather_than_mis_attributing_an_unlocatable_window():
    """C2's exact-substring invariant is assumed, not re-verified here -- if
    it is ever violated, a silently WRONG window attribution is worse than a
    loud failure: a search hit would point at a window that does not
    actually contain it."""
    ix = InteractiveIndex(docs={"d-01": "hello world"},
                          windows={"d-01": ["this text is not in the body"]})
    with pytest.raises(RuntimeError):
        ix.search("hello")
