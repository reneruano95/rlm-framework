"""The narrow JSONL lifecycle log (spec §5).

NOT a second source of truth. Episode data belongs in the trace store (I4);
this file carries only the events the trace store structurally cannot record.
The S3 gate runs with this file deleted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

ALLOWED_KINDS = frozenset({
    "trace_write_failure",
    "config_refused",
    "handshake_refused",
    "server_health",
    "quiesce_wait",
    "recovery_action",
    "sandbox_spawn",
    "sandbox_death",
    "operator_abort",
    # v0.3.16: a root render that is not a byte-for-byte extension of the
    # previous one (scaffold.root.history_mode monitor); once per episode.
    "root_history",
})


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


class Lifecycle:
    def __init__(self, path: Path | None, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._fh = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:  # degrade to stream-only, never explode
                print(f"lifecycle: cannot open {path}: {exc}", file=self._stream)

    def event(self, kind: str, **fields: Any) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"{kind!r} is not a lifecycle kind; episode data goes to the "
                "trace store (spec §5, I4)"
            )
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
               **_scrub(fields)}
        line = json.dumps(rec, ensure_ascii=True)
        print(line, file=self._stream, flush=True)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
            except OSError as exc:
                print(f"lifecycle: write failed: {exc}", file=self._stream)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
