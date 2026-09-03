from rlm.context.chunker import ChunkConfig
from rlm.context.interactive import InteractiveIndex, Hit

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
    import pytest
    with pytest.raises(IndexError):
        ix.window("d-01", n)
    with pytest.raises(KeyError):
        ix.window("d-99", 0)
