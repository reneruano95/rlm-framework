"""C2/C3: the context a root sees -- loading, chunking, truncation.

`rlm.context` was a module before 2026-08-22 and is a package now. Everything
it exported is re-exported here, so `from rlm.context import ...` is unchanged.
The loader itself lives in `rlm/context/loader.py`; `chunker` and `truncate`
are siblings (they were `rlm.context.chunker` and `rlm.context.truncate`)."""
from __future__ import annotations

from rlm.context.loader import (  # noqa: F401
    DOCUMENT_SEPARATOR,
    _DECODE_ERRORS,
    _load_mapping,
    load_context,
    read_text,
)

__all__ = [
    "DOCUMENT_SEPARATOR",
    "_DECODE_ERRORS",
    "_load_mapping",
    "load_context",
    "read_text",
]
