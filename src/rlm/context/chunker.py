"""C2 — the deterministic chunker (spec §5).

The root chunks ONLY through this utility: free-form chunking in model code
would make chunk_size advisory (a soft I1 violation) and render the §7 #2 sweep
uncontrolled. B2 and B3 (§8) use this verbatim.

The token counter is injected so this module never imports an LLM client
(dependency rule, §5).

WINDOW/STRIDE GEOMETRY (spec §7 #2) — why this is no longer a partition. The
chunk sweep found that retrieval does NOT degrade with chunk size: it falls off
a cliff at absolute DISTANCE from the needle to the question, 38/39 correct
within ~1,000 tokens and 0/39 beyond it (Fisher two-sided p = 2.9e-21). The
bracket is [989 pass, 1003 fail] -- amended 2026-08-23, n=12 per side, Fisher
p = 7.4e-07, superseding the [967, 1022] pair this module was written against
(one n=3 cell per side). Size is ruled out by the data's own counterexamples --
a 32,768-token chunk with the literal needle 967 tokens from the end scored 3/3,
a 1,024-token chunk with it 1,022 from the end scored 0/3. So shipping a small
`size_tokens` on a non-overlapping chunker fixes nothing: the HEAD of every
chunk still sits outside the horizon. The lever is window/stride geometry.

**The shipped geometry is window 640 / stride 480 (v0.3.0), not the 1,024/768
this module was written for.** ONE horizon governs two distances: the same
~1,000-token cliff that makes facts unfindable makes INSTRUCTIONS unobeyed, and
instruction-to-generation distance is (window + question) — so a 1,024 window
puts the system prefix past the measured fail point (30/30 false positives at
1,024, 0/45 with 45/45 literal recall at 640). No window between 640 and 1,024
has ever been sampled, so the geometry is justified by which SIDE of the
boundary it lands on, never by a margin in tokens.

**Window COUNT is not `ceil((T - size) / stride) + 1`.** That formula assumes
every end lands exactly one stride after the last; `_snap_back` moves ends
BACKWARD, so a gap can be as short as `int(stride * (1 - snap_tolerance))` and
the count rises — measured 424 windows against the formula's 417 over 200,000
tokens of fixture prose. Any budget derived for full coverage (`max_subcalls`)
must use the snap bound, `ceil(T / int(stride * (1 - tolerance)))`.

**The property this module now guarantees**, and the one its tests assert:

  1. every token of the source appears in at least one window, and
  2. every token appears within `stride` tokens of the END of some window
     that contains it.

Clause 2 is the whole point. A coverage-only guarantee is satisfied by the
non-overlapping chunker that was measured losing needles. Byte-exact
round-tripping (`"".join(chunks) == text`) is NOT a property under overlap —
text repeats by construction — and it is kept only on the default,
`stride == size` path, which is today's chunker unchanged.

**Clause 2 holds at `stride` on the OVERLAP path only.** On the partition path
(`stride == size`) the snap is the symmetric one §5 C2 specifies — the nearest
boundary within a ±10% token tolerance — so a cut may move LATER and a window
may hold up to `size * (1 + snap_tolerance)` tokens; its head is then that far
from its end, i.e. the tolerance further than `stride`. Found by hypothesis
(34 tokens, size 32, sparse boundaries → one 34-token window against a stride of
32) and pinned in `tests/test_chunker.py`
(`test_partition_snap_may_overshoot_the_stride_horizon_by_the_tolerance`,
`horizon_for`). It is a limit of the geometry §7 #2 CONDEMNED, not of the one it
recommends: at `stride == size` the horizon clause is nearly vacuous anyway. The
overlap path cannot do this — `_snap_back` and `_snap_start` are backward-only by
construction, which is exactly what makes the shipped promise the stride.

How clause 2 is obtained (and why the geometry is anchored on window ENDS):
window ends advance by at most `stride` tokens each, and every window reaches
back `size` tokens from its own end, so for any token p the FIRST window end
after p belongs to a window that contains p and sits within `stride` tokens of
it. Anchoring on starts instead (windows at 0, stride, 2·stride, …) fails at
the head: the first `size - stride` tokens are then only ever in window 0, at
distance up to `size`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Boundary classes, best first. A cut is snapped to the earliest boundary of the
# best available class inside the tolerance window; ties break to the earliest
# character offset (deterministic).
_BOUNDARIES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n```\n"),      # fenced code-block edge
    re.compile(r"\n\s*\n"),      # paragraph break
    re.compile(r"\n"),           # line break
)


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    size_tokens: int
    overhead_tokens: int
    snap_to_boundary: bool
    snap_tolerance: float
    #: Tokens between one window's end and the next one's (spec §7 #2).
    #: `None` means "same as `size_tokens`" — a partition, which is what every
    #: caller got before overlap existed, so a config that does not mention
    #: stride keeps today's behaviour exactly.
    stride_tokens: int | None = None

    @property
    def stride(self) -> int:
        return self.size_tokens if self.stride_tokens is None else self.stride_tokens


#: First guess at how many characters hold one token, used only to size the
#: initial search span. Any value is correct -- the span doubles until it
#: actually holds `target` tokens -- so this only decides how many probes the
#: doubling costs. 4 brackets the 3.77 chars/token measured over the S2 fixture
#: corpus (`milestones/s2/fixtures/manifest.json`).
_CHARS_PER_TOKEN_GUESS = 4


def _char_for_token_target(text: str, target: int,
                           count_tokens: Callable[[str], int]) -> int:
    """Smallest char offset whose prefix holds >= target tokens.

    BOUNDED BY THE WINDOW, not by the tail, and that is the whole point.
    Every `count_tokens` here is an HTTP round trip to the leaf's `/tokenize`
    (spec §5 C2), and this used to binary-search over `[0, len(text)]` -- so
    the first probe of every boundary tokenized half the remaining corpus, and
    the cost of windowing grew with the corpus rather than with the number of
    windows. Measured on the shipped 1024/768 geometry before the bound: 55x
    the corpus in tokenized characters at 2K tokens, 80x at 8K, 130x at 32K
    -- ~7x more work for every 4x of corpus, with round trips per window
    climbing 38.7 -> 62.7. That is quadratic work inside the episode's wall
    clock, all of it before the first leaf call.

    The span starts at a window-sized guess and DOUBLES until it holds
    `target` tokens, so the answer is still exact (the prefix count is
    monotone in length, and the doubling stops only once the target is
    reached or the text runs out) while both the probe count and the text
    each probe carries scale with `target`.
    """
    if target <= 0:
        return 0
    hi = min(len(text), max(1, target * _CHARS_PER_TOKEN_GUESS))
    while hi < len(text) and count_tokens(text[:hi]) < target:
        hi = min(len(text), hi * 2)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def split(text: str, cfg: ChunkConfig,
          count_tokens: Callable[[str], int]) -> list[str]:
    """Split `text` into windows of ~cfg.size_tokens whose ends advance by
    ~cfg.stride tokens, snapped to boundaries.

    With the default `stride == size_tokens` this is a partition and
    concatenating the result reproduces `text` byte-for-byte. With a shorter
    stride the windows overlap, that round trip is gone by construction, and
    what holds instead is the two-clause geometry in the module docstring.
    """
    if cfg.size_tokens <= 0:
        raise ValueError("chunk size_tokens must be positive")
    if not 0.0 <= cfg.snap_tolerance < 1.0:
        raise ValueError("snap_tolerance must be in [0, 1)")
    stride = cfg.stride
    if stride <= 0:
        raise ValueError("chunk stride_tokens must be positive")
    if stride > cfg.size_tokens:
        raise ValueError(
            f"chunk stride_tokens ({stride}) must not exceed size_tokens "
            f"({cfg.size_tokens}): a stride longer than the window leaves "
            "tokens in no window at all")
    if not text:
        return []
    if stride == cfg.size_tokens:
        return _split_partition(text, cfg, count_tokens)
    return _split_windows(text, cfg, count_tokens)


# --------------------------------------------------------------------------- #
# stride == size: the original chunker, byte-for-byte
# --------------------------------------------------------------------------- #


def _split_partition(text: str, cfg: ChunkConfig,
                     count_tokens: Callable[[str], int]) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        rest = text[start:]
        # "Does the whole tail fit in one chunk?" asked WITHOUT tokenizing the
        # whole tail: if no proper prefix of `rest` reaches `size_tokens`, the
        # tail holds at most that many and is the last chunk. The direct
        # `count_tokens(rest) <= size` this replaces was one full-tail round
        # trip per chunk -- on its own, O(n) round trips over an O(n) tail.
        cut = _char_for_token_target(rest, cfg.size_tokens, count_tokens)
        if cut >= len(rest):
            chunks.append(rest)
            break

        if cfg.snap_to_boundary:
            cut = _snap(rest, cut, cfg, count_tokens)
        cut = max(cut, 1)  # never make zero progress
        chunks.append(rest[:cut])
        start += cut
    return chunks


def _snap(rest: str, cut: int, cfg: ChunkConfig,
          count_tokens: Callable[[str], int]) -> int:
    """Move `cut` to the best boundary inside the ±tolerance token window."""
    lo_tokens = int(cfg.size_tokens * (1 - cfg.snap_tolerance))
    hi_tokens = int(cfg.size_tokens * (1 + cfg.snap_tolerance))
    lo = _char_for_token_target(rest, lo_tokens, count_tokens)
    hi = min(_char_for_token_target(rest, hi_tokens, count_tokens), len(rest))
    if lo >= hi:
        return cut
    window = rest[lo:hi]
    for pattern in _BOUNDARIES:
        hits = [m.end() + lo for m in pattern.finditer(window)]
        if hits:
            return min(hits)  # earliest boundary of the best class wins ties
    return cut


# --------------------------------------------------------------------------- #
# stride < size: overlapping windows, anchored on their ends
# --------------------------------------------------------------------------- #


def _split_windows(text: str, cfg: ChunkConfig,
                   count_tokens: Callable[[str], int]) -> list[str]:
    ends = _window_ends(text, cfg, count_tokens)
    # How far back a window's start can possibly lie, in ends: each gap is at
    # most `stride` tokens, so ceil(size/stride) gaps already cover `size`
    # tokens; one extra gap of slack absorbs a snap that shortened a gap. This
    # bound exists to keep the start search off the whole prefix -- every
    # count_tokens call is an HTTP round trip on the real leaf.
    lookback = -(-cfg.size_tokens // cfg.stride) + 1
    windows: list[str] = []
    for k, end in enumerate(ends):
        floor = ends[k - lookback] if k - lookback >= 0 else 0
        start = _start_for_window(text, floor, end, cfg.size_tokens, count_tokens)
        if cfg.snap_to_boundary:
            start = _snap_start(text, floor, start, end, cfg, count_tokens)
        windows.append(text[start:end])
    return windows


def _window_ends(text: str, cfg: ChunkConfig,
                 count_tokens: Callable[[str], int]) -> list[int]:
    """Window ends, each at most `stride` tokens after the previous one.

    "At most" is the load-bearing half, and it is why this snap is BACKWARD
    ONLY (`_snap_back`): a boundary beyond the nominal end would push the gap
    past `stride` and leave the tokens just after the previous end outside the
    horizon of every window that contains them -- the exact failure the
    geometry exists to prevent.
    """
    stride = cfg.stride
    ends: list[int] = []
    pos = 0
    while pos < len(text):
        rest = text[pos:]
        # Same window-bounded tail test as the partition path: no proper
        # prefix reaching `stride` tokens means the tail holds at most a
        # stride's worth, so `len(text)` is the last end. Asking
        # `count_tokens(rest) <= stride` instead tokenized the entire
        # remaining corpus once per window.
        cut = _char_for_token_target(rest, stride, count_tokens)
        if cut >= len(rest):
            ends.append(len(text))
            break
        if cfg.snap_to_boundary:
            cut = _snap_back(rest, cut, stride, cfg, count_tokens)
        cut = max(cut, 1)  # never make zero progress
        pos += cut
        ends.append(pos)
    return ends


def _snap_back(rest: str, cut: int, target: int, cfg: ChunkConfig,
               count_tokens: Callable[[str], int]) -> int:
    """The best boundary at or before `cut`, within the tolerance window.

    Deliberately the LATEST such boundary, not the earliest: within the
    backward half of the tolerance window every candidate is equally safe for
    the geometry, and the latest one loses the least ground (taking the
    earliest would shorten every gap by up to the full tolerance and buy ~11%
    more windows for nothing). The partition path keeps the spec's
    earliest-boundary tie-break, which is symmetric and has no such asymmetry
    to resolve.
    """
    lo_tokens = int(target * (1 - cfg.snap_tolerance))
    lo = _char_for_token_target(rest, lo_tokens, count_tokens)
    hi = min(cut, len(rest))
    if lo >= hi:
        return cut
    window = rest[lo:hi]
    for pattern in _BOUNDARIES:
        hits = [m.end() + lo for m in pattern.finditer(window)]
        if hits:
            return max(hits)
    return cut


def _start_for_window(text: str, floor: int, end: int, size: int,
                      count_tokens: Callable[[str], int]) -> int:
    """Smallest start >= `floor` whose window holds at most `size` tokens.

    Monotone in `start` (a later start can only hold fewer tokens), so a
    binary search is exact. Returning `floor` when even that window is short
    enough is the ordinary case at the head of the corpus.
    """
    if count_tokens(text[floor:end]) <= size:
        return floor
    lo, hi = floor, end
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[mid:end]) > size:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _snap_start(text: str, floor: int, start: int, end: int, cfg: ChunkConfig,
                count_tokens: Callable[[str], int]) -> int:
    """Move a window's start back to the nearest boundary within tolerance.

    BACKWARD ONLY, for the same reason `_snap_back` is: moving a start LATER
    can uncover a token that no other window covers inside the horizon, while
    moving it earlier only makes the window slightly longer than `size` (bounded
    by the tolerance, and priced into `chunk.overhead_tokens`).
    """
    if start <= floor:
        return start
    widest = _start_for_window(text, floor, end,
                               int(cfg.size_tokens * (1 + cfg.snap_tolerance)),
                               count_tokens)
    if widest >= start:
        return start
    window = text[widest:start]
    for pattern in _BOUNDARIES:
        hits = [m.end() + widest for m in pattern.finditer(window)]
        if hits:
            return max(hits)   # the latest safe boundary: least text added
    return start
