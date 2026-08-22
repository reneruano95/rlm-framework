"""C2 — ContextLoader (spec §5).

Materializes an episode's input as ONE `str`. The episode runner then
`setvar`s that string into the sandbox under the name `context` and never
touches it again: **no part of it ever enters a message array** (I2). The
root only ever sees C3-truncated observations of code it wrote about
`context`, so this module's output has exactly two destinations — the
sandbox heap, and the C6 blob store as ground truth.

The chunker is the other half of C2 and lives in `rlm.context.chunker` (it has to,
so the injected token counter keeps this module free of any LLM client —
`tests/test_import_rules.py` lints both).

**`context` is the corpus; `chunks` is a set of VIEWS of it, and since §7 #2
those views OVERLAP.** At window 1,024 / stride 768 every token appears in one
or two windows, so `chunks` is no longer a partition: concatenating it repeats
text, `sum(len(c) for c in chunks) > len(context)`, and anything that counts
occurrences must count them over `context` (or de-duplicate by position),
never by summing over windows. This module's output is the single
non-repeating view, which is exactly why the sandbox gets both and why the
raw corpus — not the window list — is what the trace stores as ground truth.

**A `str` spec is always literal text, never a path.** Sniffing a string for
path-likeness would make a corpus that happens to contain a filename load a
different document than the one the task declared, silently and only
sometimes. Paths are requested explicitly: `Path(...)`, `{"path": ...}`, or
`{"paths": [...]}`.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlm.errors import ConfigError

# Documents in a multi-document spec are joined with a blank line. Deliberately
# a plain separator and not a header: a per-document banner would be scaffold
# text the root cannot distinguish from corpus text, and §8's adversarial
# tasks turn on exactly that distinction. Document identity, when a task needs
# it, belongs in the task's own instruction text.
DOCUMENT_SEPARATOR = "\n\n"

# Corpora are ground truth, not code: a single bad byte in a 500 MB document
# must not fail an episode, and "replace" is auditable (the blob store keeps
# the raw file, and the replacement char is visible in the trace).
_DECODE_ERRORS = "replace"


def read_text(path: str | os.PathLike) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors=_DECODE_ERRORS)
    except OSError as exc:
        raise ConfigError(f"context file {path} could not be read: {exc}") from exc


def load_context(spec: Any) -> str:
    """Resolve a context spec to the exact string the sandbox will hold.

    Accepted shapes:

      ``None``                     -> ``""`` (a task with no corpus)
      ``str``                      -> that text, verbatim
      ``bytes`` / ``bytearray``    -> UTF-8 decoded
      ``Path`` / ``os.PathLike``   -> that file's text
      ``{"text": ...}``            -> the text
      ``{"path": ...}``            -> that file
      ``{"paths": [...]}``         -> those files, joined
      ``{"documents": [...]}``     -> each element resolved, joined
      ``list`` / ``tuple``         -> each element resolved, joined

    Anything else raises `ConfigError` — an unrecognised spec is a task-file
    error and must refuse the episode, never quietly load an empty corpus.
    """
    if spec is None:
        return ""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (bytes, bytearray)):
        return bytes(spec).decode("utf-8", errors=_DECODE_ERRORS)
    if isinstance(spec, os.PathLike):
        return read_text(spec)
    if isinstance(spec, Mapping):
        return _load_mapping(spec)
    if isinstance(spec, Sequence):
        return DOCUMENT_SEPARATOR.join(load_context(item) for item in spec)
    raise ConfigError(
        f"unsupported context spec of type {type(spec).__name__}; expected str, "
        "bytes, a path, a list of documents, or a mapping with one of "
        "'text'/'path'/'paths'/'documents'"
    )


def _load_mapping(spec: Mapping) -> str:
    keys = [k for k in ("text", "path", "paths", "documents") if k in spec]
    if len(keys) != 1:
        raise ConfigError(
            "a context mapping must carry exactly one of 'text', 'path', "
            f"'paths' or 'documents'; got {sorted(spec)!r}"
        )
    key = keys[0]
    value = spec[key]
    if key == "text":
        return load_context(value)
    if key == "path":
        return read_text(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"context {key!r} must be a list; got {type(value).__name__}")
    if key == "paths":
        return DOCUMENT_SEPARATOR.join(read_text(p) for p in value)
    return DOCUMENT_SEPARATOR.join(load_context(item) for item in value)
