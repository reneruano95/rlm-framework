"""C2 — the deterministic chunker (spec §5).

The root chunks ONLY through this utility: free-form chunking in model code
would make chunk_size advisory (a soft I1 violation) and render the §7 #2 sweep
uncontrolled. B2 and B3 (§8) use this verbatim.

The token counter is injected so this module never imports an LLM client
(dependency rule, §5).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Boundary classes, best first. A cut is snapped to the latest boundary of the
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


def _char_for_token_target(text: str, target: int,
                           count_tokens: Callable[[str], int]) -> int:
    """Smallest char offset whose prefix holds >= target tokens (binary search)."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def split(text: str, cfg: ChunkConfig,
          count_tokens: Callable[[str], int]) -> list[str]:
    """Split `text` into chunks of ~cfg.size_tokens, snapped to boundaries.

    Concatenating the result reproduces `text` byte-for-byte.
    """
    if cfg.size_tokens <= 0:
        raise ValueError("chunk size_tokens must be positive")
    if not 0.0 <= cfg.snap_tolerance < 1.0:
        raise ValueError("snap_tolerance must be in [0, 1)")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        rest = text[start:]
        if count_tokens(rest) <= cfg.size_tokens:
            chunks.append(rest)
            break

        cut = _char_for_token_target(rest, cfg.size_tokens, count_tokens)
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
