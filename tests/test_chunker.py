import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rlm.context.chunker import ChunkConfig, split


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
# tokens, 0/39 beyond it (Fisher two-sided p = 2.9e-21). The bracket is
# [989 pass, 1003 fail] (amended 2026-08-23; n=12 per side, p = 7.4e-07),
# superseding [967, 1022], which was one n=3 cell per side. A 32,768-token
# chunk whose literal needle sat 967 tokens from the end scored 3/3 while a
# 1,024-token chunk whose needle sat 1,022 from the end scored 0/3, so shipping
# `size_tokens: 1024` on a non-overlapping chunker would still leave the HEAD
# of every chunk outside the horizon.
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


def horizon_for(cfg: ChunkConfig) -> int:
    """The largest distance-to-window-end clause 2 can promise under `cfg`.

    `stride` on the OVERLAP path, and that is the guarantee the shipped 1024/768
    geometry rests on: `_window_ends` advances ends by at most a stride (its snap
    is backward-only), and window 0 is exactly one stride long, so no token is
    ever further than `stride` from the end of some window containing it.

    On the PARTITION path (`stride == size`, the legacy default) the snap is the
    one §5 C2 specifies — "the nearest boundary within a ±10% token tolerance" —
    and ± is symmetric, so a cut may move LATER and a window may hold up to
    `size * (1 + snap_tolerance)` tokens (already pinned by
    `test_no_chunk_exceeds_target_plus_tolerance`). The head of such a window is
    that far from its end, which is exactly the tolerance more than `stride`.
    Found by hypothesis at n=34/size=32/sparse17, pinned as a regression by
    `test_partition_snap_may_overshoot_the_stride_horizon_by_the_tolerance`.

    This is a limit of the geometry the sweep CONDEMNED, not of the one it
    recommended: at `stride == size` the horizon clause is nearly vacuous anyway
    (a token may sit a full window from the question), which is why §7 #2's fix
    is overlap rather than a smaller size.
    """
    if cfg.stride < cfg.size_tokens or not cfg.snap_to_boundary:
        return cfg.stride
    return int(cfg.size_tokens * (1 + cfg.snap_tolerance))


def geometry_violations(text: str, chunks: list[str], horizon: int) -> list[tuple]:
    """Every (token, why) that breaks the two-clause property.

    `(p, None)` = the token is in no window at all (coverage);
    `(p, d)` = its nearest containing window's end is d > `horizon` tokens away
    (the horizon clause). `horizon` is `horizon_for(cfg)`, which is the stride on
    every geometry that ships.
    """
    sp = spans(text, chunks)
    bad: list[tuple] = []
    for p in token_starts(text):
        containing = [(s, e) for s, e in sp if s <= p < e]
        if not containing:
            bad.append((p, None))
            continue
        nearest = min(count(text[p:e]) for _, e in containing)
        if nearest > horizon:
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
    """window 640 / stride 480 over 10,000 tokens (§7 #2, v0.3.0).

    The naive price is ceil(10000/480) = 21 windows. The chunker may need more
    and may never need more than the snap bound -- see
    `test_the_window_count_can_exceed_the_naive_formula_and_never_the_snap_bound`,
    which is why `max_subcalls` is derived from the bound and not the formula."""
    cfg = ChunkConfig(size_tokens=640, overhead_tokens=1920,
                      snap_to_boundary=True, snap_tolerance=0.10,
                      stride_tokens=480)
    text = lines(10_000)
    chunks = split(text, cfg, count)
    assert geometry_violations(text, chunks, horizon_for(cfg)) == []
    assert horizon_for(cfg) == cfg.stride      # the shipped geometry promises the stride
    naive = -(-(10_000 - 640) // 480) + 1
    bound = -(-10_000 // int(480 * 0.9))
    assert naive == 21 and bound == 24
    assert naive <= len(chunks) <= bound


def test_the_window_count_can_exceed_the_naive_formula_and_never_the_snap_bound():
    """`max_subcalls` is a coverage guarantee, so its derivation has to be the
    bound the chunker cannot beat -- not the formula the spec priced it with.

    `ceil((T - size) / stride) + 1` assumes every window end lands exactly one
    stride after the last. `_snap_back` moves an end BACKWARD to the nearest
    boundary within the +-10% tolerance, so a gap can be as short as
    `int(stride * (1 - snap_tolerance))` and the count rises. Measured on the
    real chunker over 200,000 tokens of S2 fixture prose: 424 windows at
    640/480 against the formula's 417, and 268 at 1,024/768 against 261 -- so
    the shipped `max_subcalls: 522` was already 14 calls short of its own
    geometry and would have truncated coverage silently.

    Constructed here with boundaries only every 90% of a stride, which forces
    every snap to take the full tolerance."""
    stride, size, tol = 480, 640, 0.10
    min_gap = int(stride * (1 - tol))          # 432
    # Tokens separated by spaces, with a newline every `min_gap` tokens: the
    # only boundary a backward snap can reach is a full tolerance early.
    groups = ["  ".join(f"w{i}_{j}" for j in range(min_gap)) for i in range(20)]
    text = "\n".join(groups)
    total = count(text)
    cfg = ChunkConfig(size_tokens=size, overhead_tokens=1920,
                      snap_to_boundary=True, snap_tolerance=tol,
                      stride_tokens=stride)
    chunks = split(text, cfg, count)
    naive = -(-(total - size) // stride) + 1
    bound = -(-total // min_gap)
    assert len(chunks) > naive          # the formula under-counts, as measured
    assert len(chunks) <= bound         # and the bound holds
    assert geometry_violations(text, chunks, horizon_for(cfg)) == []


def test_rejects_a_stride_that_cannot_hold_the_geometry():
    text = lines(50)
    with pytest.raises(ValueError):
        split(text, ChunkConfig(100, 20, True, 0.1, stride_tokens=0), count)
    with pytest.raises(ValueError):
        # A stride LONGER than the window would leave gaps -- tokens in no
        # window at all, which is the coverage clause, not merely the horizon.
        split(text, ChunkConfig(100, 20, True, 0.1, stride_tokens=101), count)


# --------------------------------------------------------------------------- #
# What windowing COSTS. Every `count_tokens` here is an HTTP round trip to the
# leaf's /tokenize (spec §5 C2: "measured in target-leaf tokens" means that
# counter and no other), and C2 runs inside the episode's wall clock, before a
# single leaf call. So the number of counts AND the amount of text handed to
# each one are correctness-adjacent, not merely tidiness.
# --------------------------------------------------------------------------- #


class CountingCounter:
    """`count`, instrumented: how many round trips, over how much text."""

    def __init__(self) -> None:
        self.calls = 0
        self.chars = 0

    def __call__(self, text: str) -> int:
        self.calls += 1
        self.chars += len(text)
        return count(text)


PRODUCTION = ChunkConfig(size_tokens=640, overhead_tokens=1920,
                         snap_to_boundary=True, snap_tolerance=0.10,
                         stride_tokens=480)


def test_windowing_cost_is_linear_in_window_count_not_in_corpus_size():
    """The boundary search must be bounded by a WINDOW, not by the tail.

    Searching to `len(text)` makes every probe tokenize up to half the
    remaining corpus, and the per-window tail check tokenizes all of it -- so
    windowing is O(n) round trips over an O(n) tail, i.e. quadratic work, all
    of it before the first leaf call. Measured on this geometry before the
    fix: 55x the corpus in tokenized characters at 2K tokens, 80x at 8K, 130x
    at 32K, growing ~7x for every 4x of corpus, with counts-per-window
    climbing 38.7 -> 62.7 as the corpus grew. On a 200K-token corpus that is
    hundreds of millions of characters over HTTP to build 424 windows.

    Both halves are asserted: the count per window has to be a constant (a
    function of the window, not of the corpus), and the WORK -- characters
    actually sent to /tokenize -- has to stay a bounded multiple of the corpus.
    The second is the one that was quadratic; a call-count-only assertion
    would have passed the old code at a single size.
    """
    assert_window_bounded(PRODUCTION)


def test_the_partition_path_is_window_bounded_too():
    """`stride == size` is B2's and B3's chunker as well as today's default
    (§8: they use the C2 chunker verbatim), so the same full-tail scan was
    costing both controls the same quadratic pre-episode work. Measured before
    the fix on this path: 1.69x the work per corpus character for 4x the
    corpus."""
    assert_window_bounded(ChunkConfig(size_tokens=640, overhead_tokens=1920,
                                      snap_to_boundary=True, snap_tolerance=0.10))


def assert_window_bounded(cfg: ChunkConfig) -> None:
    """4x the corpus must cost ~4x the windowing, not ~16x.

    Both instruments, because they fail differently. `chars` is the WORK --
    the text actually handed to `/tokenize`, i.e. the HTTP payload -- and it is
    the one that was quadratic; `calls` is the round-trip count, which is
    O(log window) per window by construction and grows only if the search is
    reaching past the window. Deterministic arithmetic on a fixed fixture, so
    the margins can be tight without being flaky.
    """
    small, big = lines(4_000), lines(16_000)
    cs, cb = CountingCounter(), CountingCounter()
    ws, wb = split(small, cfg, cs), split(big, cfg, cb)
    assert ws and wb

    work_small = cs.chars / len(small)      # tokenized chars per corpus char
    work_big = cb.chars / len(big)
    assert work_big <= work_small * 1.15, (
        f"windowing cost {work_small:.1f}x the corpus at {len(small)} chars and "
        f"{work_big:.1f}x at {len(big)}: the work per character grows with the "
        "corpus, which is the quadratic tail scan")

    per_window = (cs.calls / len(ws), cb.calls / len(wb))
    assert per_window[1] <= per_window[0] * 1.20, (
        f"round trips per window grew with the corpus: {per_window[0]:.1f} -> "
        f"{per_window[1]:.1f}; the search is not window-bounded")
    assert cb.calls / cs.calls <= 1.20 * (len(wb) / len(ws)), (
        f"{cb.calls / cs.calls:.2f}x the round trips for "
        f"{len(wb) / len(ws):.2f}x the windows: the count is not linear in "
        "window count")


def test_bounding_the_search_did_not_move_a_single_boundary():
    """The bound is a cost fix and must be nothing else: same windows, byte
    for byte, at both stride settings and with snapping on and off."""
    text = lines(3_000)
    for cfg in (CFG, OVERLAP, PRODUCTION,
                ChunkConfig(100, 20, False, 0.10, stride_tokens=75)):
        chunks = split(text, cfg, count)
        assert chunks
        assert geometry_violations(text, chunks, horizon_for(cfg)) == []
        assert split(text, cfg, count) == chunks       # still deterministic


# --------------------------------------------------------------------------- #
# TEXT SHAPES. Every overlap fixture above is `lines(n)` -- a boundary between
# every single token -- which is the one shape where snapping can barely move a
# cut at all. Boundary snapping is the only thing in the geometry that MOVES a
# window edge, and it moves it by up to the tolerance, so the shapes that can
# actually stress the horizon clause are the ones where boundaries are scarce
# (a snap has to travel) or clustered (a snap lands far from the nominal cut).
# The property holds on all of them; what was missing was any fixture able to
# show it.
# --------------------------------------------------------------------------- #


def sparse(n: int, every: int) -> str:
    """A newline only every `every` tokens: snapping must travel to reach one."""
    toks = [f"w{i}" for i in range(n)]
    return " ".join(
        tok + ("\n" if (i + 1) % every == 0 else "") for i, tok in enumerate(toks)
    ).replace("\n ", "\n")


def clumped(n: int) -> str:
    """Runs of boundary-dense text alternating with long unbroken runs -- the
    shape real corpora actually have (paragraph blocks, then a code block or a
    table with no breaks in it)."""
    out: list[str] = []
    i = 0
    dense = True
    while i < n:
        run = 12 if dense else 90
        block = [f"w{j}" for j in range(i, min(n, i + run))]
        out.append(("\n" if dense else " ").join(block))
        i += run
        dense = not dense
    return "\n".join(out)


def unbroken(n: int) -> str:
    """No boundary anywhere: every snap must fall back to the nominal cut."""
    return words(n)


SHAPES = {"lines": lines, "sparse5": lambda n: sparse(n, 5),
          "sparse17": lambda n: sparse(n, 17), "sparse60": lambda n: sparse(n, 60),
          "clumped": clumped, "unbroken": unbroken}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_geometry_holds_on_every_text_shape(shape):
    """The horizon clause on the SHIPPED geometry, over boundary layouts the
    overlap fixtures never had. `unbroken` is the control that the snapping
    itself is not what satisfies the property."""
    text = SHAPES[shape](2_000)
    chunks = split(text, OVERLAP, count)
    assert chunks
    assert geometry_violations(text, chunks, horizon_for(OVERLAP)) == []


@settings(max_examples=60, deadline=None)
@given(n=st.integers(min_value=1, max_value=400),
       size=st.integers(min_value=4, max_value=60),
       ratio=st.floats(min_value=0.25, max_value=1.0),
       snap=st.booleans(),
       shape=st.sampled_from(sorted(SHAPES)))
def test_the_geometry_holds_for_arbitrary_sizes_and_strides(n, size, ratio, snap,
                                                             shape):
    """Randomized over geometry AND text shape.

    The shape axis is the one that was missing: with a boundary between every
    token a snap moves a cut by at most one token, so no amount of size/stride
    randomization could ever show a snap pushing the effective distance past
    `stride`. Sparse boundaries (a snap travels up to the full tolerance),
    clumped ones (it lands in a burst far from the nominal cut) and none at all
    are where that could happen.
    """
    stride = max(1, int(size * ratio))
    cfg = ChunkConfig(size_tokens=size, overhead_tokens=8, snap_to_boundary=snap,
                      snap_tolerance=0.10, stride_tokens=stride)
    text = SHAPES[shape](n)
    chunks = split(text, cfg, count)
    assert chunks
    assert geometry_violations(text, chunks, horizon_for(cfg)) == []


def test_partition_snap_may_overshoot_the_stride_horizon_by_the_tolerance():
    """The one place clause 2 does NOT hold at `stride`, pinned rather than
    hidden — hypothesis found it, and rewriting the bound without a regression
    test would let the next widening of the tolerance pass unnoticed.

    §5 C2's snap is symmetric ("±10% token tolerance"), so on the partition path
    a cut may move LATER: here a 34-token text snaps to a single 34-token window
    against a stride of 32, and its head sits 34 tokens from the end. The bound
    is `size * (1 + snap_tolerance)` = 35 and it is tight. The OVERLAP path
    cannot do this — `_snap_back`/`_snap_start` are backward-only by
    construction, which is what makes the shipped geometry's promise the stride.
    """
    cfg = ChunkConfig(size_tokens=32, overhead_tokens=8, snap_to_boundary=True,
                      snap_tolerance=0.10, stride_tokens=32)
    text = SHAPES["sparse17"](34)
    chunks = split(text, cfg, count)
    assert [count(c) for c in chunks] == [34]
    assert geometry_violations(text, chunks, cfg.stride) == [(0, 34), (3, 33)]
    assert geometry_violations(text, chunks, horizon_for(cfg)) == []
    assert horizon_for(cfg) == 35

    # Same text, same tolerance, OVERLAPPING: the promise is the stride again.
    over = ChunkConfig(size_tokens=32, overhead_tokens=8, snap_to_boundary=True,
                       snap_tolerance=0.10, stride_tokens=24)
    assert horizon_for(over) == 24
    assert geometry_violations(text, split(text, over, count), over.stride) == []
