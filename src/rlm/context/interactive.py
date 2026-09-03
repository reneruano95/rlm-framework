"""The `interactive` category's corpus (benchmark v2 §14): a multi-document
index that lives SCAFFOLD-SIDE and never crosses the pipe.

An interactive episode keeps `context` and `chunks` empty in the sandbox --
the root navigates this index instead, through the `env.search` / `env.open`
/ `env.window` verbs, each served by `episode._on_env` and each return value
truncated scaffold-side exactly like any other observation (I2). `search`
returns locations only (document id, window index, offset -- never text);
`open` returns structure and length (window count, char count -- not
content); `window` returns one bounded slice.

Chunk geometry is unchanged: each document is windowed with the same C2
chunker (`rlm.context.chunker.split`) and the same `ChunkConfig` the ordinary
`chunks` view would use, so the ~1,000-token distance cliff this scaffold was
built around governs `env.window` too.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rlm.context.chunker import ChunkConfig, split

#: The builder's per-document header, e.g. "=== DOCUMENT d-07: <title> ===".
#: A blank line precedes every header but the first, matching
#: `rlm.context.loader.DOCUMENT_SEPARATOR`'s "\n\n" join.
DOC_DELIM = "\n\n=== DOCUMENT "

_HEADER = re.compile(r"^=== DOCUMENT (\S+): (.*?) ===\n", flags=re.M)


@dataclass(frozen=True)
class Hit:
    doc_id: str
    window: int
    offset: int


@dataclass(frozen=True)
class InteractiveIndex:
    docs: dict[str, str]
    windows: dict[str, list[str]]
    titles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, chunk_cfg: ChunkConfig,
                  count_tokens: Callable[[str], int]) -> "InteractiveIndex":
        docs: dict[str, str] = {}
        titles: dict[str, str] = {}
        parts = _HEADER.split(text)
        # parts = [preamble, id1, title1, body1, id2, title2, body2, ...]
        for j in range(1, len(parts), 3):
            docs[parts[j]] = parts[j + 2].strip("\n")
            titles[parts[j]] = parts[j + 1]
        if not docs:
            raise ValueError(
                "an interactive corpus needs at least one "
                "'=== DOCUMENT id: title ===' header")
        windows = {d: split(body, chunk_cfg, count_tokens) for d, body in docs.items()}
        return cls(docs=docs, titles=titles, windows=windows)

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def search(self, term: str, *, max_hits: int = 50) -> list[Hit]:
        """Case-insensitive substring search over every document's raw text.

        Each match is mapped to the FIRST window whose span contains it --
        windows can overlap (§7 #2), so a match near a boundary may belong to
        more than one; the earliest-starting one wins, deterministically.
        Capped at `max_hits` after sorting by `(doc_id, window)`, so the cap
        is stable regardless of match order within a document.
        """
        hits: list[Hit] = []
        pattern = re.compile(re.escape(term), re.I)
        for doc_id in sorted(self.docs):
            body = self.docs[doc_id]
            windows = self.windows[doc_id]
            starts: list[int] = []
            cursor = 0
            for w in windows:
                start = body.find(w, cursor)
                starts.append(start)
                cursor = start + 1
            for m in pattern.finditer(body):
                offset = m.start()
                for i, start in enumerate(starts):
                    if start <= offset < start + len(windows[i]):
                        hits.append(Hit(doc_id=doc_id, window=i, offset=offset))
                        break
        hits.sort(key=lambda h: (h.doc_id, h.window))
        return hits[:max_hits]

    def open(self, doc_id: str) -> dict:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return {"doc_id": doc_id, "title": self.titles[doc_id],
                "n_windows": len(self.windows[doc_id]),
                "n_chars": len(self.docs[doc_id])}

    def window(self, doc_id: str, i: int) -> str:
        """RANGE-CHECKED, NEVER CLAMPED -- like `episode.resolve_chunk_ref`. A
        window index the root did not get from `search`/`open` is a bug in
        the emitting cell, and answering about window 0 instead would be
        scored as an ordinary (wrong) answer instead of surfacing the bug."""
        if doc_id not in self.windows:
            raise KeyError(doc_id)
        windows = self.windows[doc_id]
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(windows):
            raise IndexError(
                f"window index {i!r} is outside {doc_id} (0..{len(windows) - 1})")
        return windows[i]
