"""C3 — OutputTruncator (spec §5).

The hard cap on everything the root sees from the REPL. Applied scaffold-side,
after execution, over the concatenated view as ONE unit. Nothing running inside
the sandbox can raise, lower, or bypass it (I1).
"""
from __future__ import annotations

from dataclasses import dataclass

_MARKER = "[truncated: showing {shown:,} of {total:,} chars]"
# The longest marker we could ever need to append, used to reserve room so the
# marker itself is never truncated (property-tested).
_MARKER_RESERVE = len(_MARKER.format(shown=10**12, total=10**12))

# Smallest cap that can hold the marker without corrupting it. Below this,
# truncate_view() emits no marker at all (config will later forbid this
# regime in production; it exists here only to keep the property total).
MIN_MARKER_CAP = _MARKER_RESERVE


@dataclass(frozen=True, slots=True)
class CellOutput:
    """Raw, untruncated result of one REPL cell. Stored in full by C6."""

    stdout: str = ""
    stderr: str = ""
    repr_: str = ""
    traceback: str = ""


def build_view(out: CellOutput) -> str:
    """Ordered, labeled concatenation. Empty sections are omitted."""
    parts: list[str] = []
    for label, body in (
        ("stdout", out.stdout),
        ("stderr", out.stderr),
        ("repr", out.repr_),
        ("traceback", out.traceback),
    ):
        if body:
            parts.append(f"[{label}]\n{body}")
    return "\n".join(parts)


def truncate_view(view: str, cap: int) -> str:
    """Truncate the assembled view to `cap` chars, marker included in the cap.

    Three regimes:
      cap <= 0                    -> "" (never rely on negative-index slicing).
      0 < cap < MIN_MARKER_CAP    -> head only, no marker. A truncated marker
                                      is corrupt and unparseable; an absent one
                                      is honest.
      cap >= MIN_MARKER_CAP       -> head + accurate marker (normal case).
    """
    if cap <= 0:
        return ""
    total = len(view)
    if total <= cap:
        return view
    if cap < MIN_MARKER_CAP:
        return view[:cap]
    budget = cap - _MARKER_RESERVE
    head = view[:budget]
    marker = _MARKER.format(shown=len(head), total=total)
    return head + marker


def observation_view(out: CellOutput, cap: int) -> str:
    """What the root actually sees. `steps.observation_view` in §6."""
    return truncate_view(build_view(out), cap)
