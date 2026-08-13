import pytest

from rlm.chunker import ChunkConfig, split


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def count(text: str) -> int:
    """Deterministic stand-in for the leaf server's /tokenize (1 token/word)."""
    return len(text.split())


CFG = ChunkConfig(size_tokens=100, overhead_tokens=20, snap_to_boundary=True,
                  snap_tolerance=0.10)


def test_covers_every_token_exactly_once_when_rejoined():
    text = words(1000)
    chunks = split(text, CFG, count)
    assert "".join(chunks) == text


def test_no_chunk_exceeds_target_plus_tolerance():
    text = words(1000)
    for chunk in split(text, CFG, count):
        assert count(chunk) <= CFG.size_tokens * (1 + CFG.snap_tolerance) + 1


def test_is_deterministic():
    text = words(997)
    assert split(text, CFG, count) == split(text, CFG, count)


def test_snaps_to_paragraph_boundary_within_tolerance():
    # a paragraph break sits at token 95 — inside the -5% tolerance of a 100 cut
    text = words(95) + "\n\n" + words(200)
    chunks = split(text, CFG, count)
    assert chunks[0].endswith("\n\n") or chunks[1].startswith("w95")


def test_snap_disabled_cuts_at_exact_target():
    cfg = ChunkConfig(size_tokens=100, overhead_tokens=20, snap_to_boundary=False,
                      snap_tolerance=0.10)
    text = words(95) + "\n\n" + words(200)
    chunks = split(text, cfg, count)
    assert count(chunks[0]) == 100


def test_earliest_boundary_wins_ties():
    # two equidistant boundaries around the target: the earlier one must win
    text = words(95) + "\n\n" + words(9) + "\n\n" + words(200)
    chunks = split(text, CFG, count)
    assert count(chunks[0]) == 95


def test_short_text_yields_one_chunk():
    assert len(split(words(10), CFG, count)) == 1


def test_empty_text_yields_no_chunks():
    assert split("", CFG, count) == []


def test_rejects_nonsense_config():
    with pytest.raises(ValueError):
        split(words(10), ChunkConfig(0, 20, True, 0.1), count)
