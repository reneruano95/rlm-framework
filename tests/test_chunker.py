import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlm.chunker import ChunkConfig, split


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def lines(n: int) -> str:
    """`words(n)` with a boundary between every token, so a snapped cut lands
    exactly on a token edge and the geometry can be checked token-exactly."""
    return "\n".join(f"w{i}" for i in range(n))


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


# --------------------------------------------------------------------------- #
# Window/stride overlap (spec §7 #2, v0.2.5).
#
# Retrieval does not degrade with chunk SIZE; it falls off a cliff at absolute
# DISTANCE from the needle to the question -- 38/39 correct within ~1,000
# tokens, 0/39 beyond it (Fisher p ~ 1e-21), with the bracket measured at
# [967 pass, 1022 fail]. A 32,768-token chunk whose needle sat 967 tokens from
# the end scored 6/6 while a 1,024-token chunk whose needle sat 1,022 from the
# end scored 0/3, so shipping `size_tokens: 1024` on a non-overlapping chunker
# would still leave the HEAD of every chunk outside the horizon.
#
# The property that replaces the byte-exact round trip is therefore geometric,
# and it has two clauses: every token appears in some window (coverage), AND
# every token appears within `stride` tokens of some containing window's END
# (the horizon). The second clause is the whole point -- a coverage-only test
# passes a geometry that still loses needles, which `test_the_distance_clause_
# has_teeth` pins by showing non-overlapping chunking satisfies coverage and
# fails the horizon.
# --------------------------------------------------------------------------- #

import re  # noqa: E402 -- kept next to the helpers that use it


def spans(text: str, chunks: list[str]) -> list[tuple[int, int]]:
    """(start, end) char offsets of each window, in order.

    Windows overlap, so `text.index` is anchored at the previous window's
    start; token names are unique in these fixtures, so the first hit at or
    after that anchor is the right one."""
    out: list[tuple[int, int]] = []
    at = 0
    for c in chunks:
        i = text.index(c, at)
        out.append((i, i + len(c)))
        at = i
    return out


def token_starts(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"\S+", text)]


def geometry_violations(text: str, chunks: list[str], stride: int) -> list[tuple]:
    """Every (token, why) that breaks the two-clause property.

    `(p, None)` = the token is in no window at all (coverage);
    `(p, d)` = its nearest containing window's end is d > stride tokens away
    (the horizon clause).
    """
    sp = spans(text, chunks)
    bad: list[tuple] = []
    for p in token_starts(text):
        containing = [(s, e) for s, e in sp if s <= p < e]
        if not containing:
            bad.append((p, None))
            continue
        nearest = min(count(text[p:e]) for _, e in containing)
        if nearest > stride:
            bad.append((p, nearest))
    return bad


OVERLAP = ChunkConfig(size_tokens=100, overhead_tokens=20, snap_to_boundary=True,
                      snap_tolerance=0.10, stride_tokens=75)


def test_stride_defaults_to_the_size_so_nothing_silently_changes():
    """Today's callers pass no stride and must keep today's chunker: a
    partition whose concatenation reproduces the source byte for byte."""
    text = lines(400)
    assert CFG.stride == CFG.size_tokens
    chunks = split(text, CFG, count)
    assert "".join(chunks) == text
    explicit = ChunkConfig(size_tokens=100, overhead_tokens=20,
                           snap_to_boundary=True, snap_tolerance=0.10,
                           stride_tokens=100)
    assert split(text, explicit, count) == chunks


def test_every_token_appears_in_at_least_one_window():
    text = lines(1000)
    assert not [b for b in geometry_violations(text, split(text, OVERLAP, count),
                                                OVERLAP.stride) if b[1] is None]


def test_every_token_lands_within_stride_tokens_of_some_windows_end():
    text = lines(1000)
    assert geometry_violations(text, split(text, OVERLAP, count), OVERLAP.stride) == []


def test_the_head_of_the_corpus_is_inside_the_horizon_too():
    """The naive geometry (windows starting at 0, stride, 2*stride, ...) leaves
    the first `size - stride` tokens at distance up to `size` from the only
    window that contains them -- 1,024 tokens on the shipped geometry, i.e.
    past the measured cliff. The head therefore needs a window whose END is
    within `stride` of token 0."""
    text = lines(1000)
    chunks = split(text, OVERLAP, count)
    first_token = token_starts(text)[0]
    sp = spans(text, chunks)
    nearest = min(count(text[first_token:e]) for s, e in sp if s <= first_token < e)
    assert nearest <= OVERLAP.stride


def test_the_distance_clause_has_teeth():
    """A coverage-only property would pass the non-overlapping chunker, which
    is exactly the geometry §7 #2 measured as losing needles. Asserting the
    horizon clause against it must FAIL, or the clause is decoration."""
    text = lines(1000)
    contiguous = split(text, CFG, count)
    assert not [b for b in geometry_violations(text, contiguous, CFG.size_tokens)
                if b[1] is None]                      # coverage holds ...
    violations = geometry_violations(text, contiguous, OVERLAP.stride)
    assert violations                                  # ... the horizon does not
    assert max(d for _, d in violations) > OVERLAP.stride


def test_windows_overlap_rather_than_partition():
    text = lines(500)
    chunks = split(text, OVERLAP, count)
    assert "".join(chunks) != text                     # the old property is gone
    assert sum(count(c) for c in chunks) > count(text)  # because text repeats


def test_no_window_exceeds_the_size_plus_tolerance():
    text = lines(1000)
    for window in split(text, OVERLAP, count):
        assert count(window) <= OVERLAP.size_tokens * (1 + OVERLAP.snap_tolerance) + 1


def test_overlap_is_deterministic():
    text = lines(997)
    assert split(text, OVERLAP, count) == split(text, OVERLAP, count)


def test_overlap_keeps_snapping_to_boundaries():
    """Boundary-snapping survives the geometry: with a boundary between every
    token, BOTH edges of every window must land on one (no half-token
    windows), and an unsnapped chunker splits mid-token here."""
    text = lines(600)
    chunks = split(text, OVERLAP, count)
    for start, end in spans(text, chunks):
        assert start == 0 or text[start - 1] == "\n"
        assert end == len(text) or text[end - 1] == "\n"
    unsnapped = ChunkConfig(size_tokens=100, overhead_tokens=20,
                            snap_to_boundary=False, snap_tolerance=0.10,
                            stride_tokens=75)
    assert any(text[e - 1] != "\n" and e != len(text)
               for _, e in spans(text, split(text, unsnapped, count)))


def test_the_production_geometry_costs_the_expected_window_count():
    """window 1,024 / stride 768 over 10,000 tokens: ceil(10000/768) = 14
    windows, the same arithmetic §7 #2 uses to price a 200K corpus at 261."""
    cfg = ChunkConfig(size_tokens=1024, overhead_tokens=1536,
                      snap_to_boundary=True, snap_tolerance=0.10,
                      stride_tokens=768)
    text = lines(10_000)
    chunks = split(text, cfg, count)
    assert geometry_violations(text, chunks, cfg.stride) == []
    assert 13 <= len(chunks) <= 15


def test_rejects_a_stride_that_cannot_hold_the_geometry():
    text = lines(50)
    with pytest.raises(ValueError):
        split(text, ChunkConfig(100, 20, True, 0.1, stride_tokens=0), count)
    with pytest.raises(ValueError):
        # A stride LONGER than the window would leave gaps -- tokens in no
        # window at all, which is the coverage clause, not merely the horizon.
        split(text, ChunkConfig(100, 20, True, 0.1, stride_tokens=101), count)


@settings(max_examples=25, deadline=None)
@given(n=st.integers(min_value=1, max_value=400),
       size=st.integers(min_value=4, max_value=60),
       ratio=st.floats(min_value=0.25, max_value=1.0),
       snap=st.booleans())
def test_the_geometry_holds_for_arbitrary_sizes_and_strides(n, size, ratio, snap):
    stride = max(1, int(size * ratio))
    cfg = ChunkConfig(size_tokens=size, overhead_tokens=8, snap_to_boundary=snap,
                      snap_tolerance=0.10, stride_tokens=stride)
    text = lines(n)
    chunks = split(text, cfg, count)
    assert chunks
    assert geometry_violations(text, chunks, cfg.stride) == []
