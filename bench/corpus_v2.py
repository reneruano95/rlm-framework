"""v2's vendored label source: TREC question classification.

WHY A VENDORED HUMAN-LABELLED CORPUS. v2's linear-semantic and interactive
tasks need a per-item label that no deterministic program can derive from the
item's text -- otherwise a regex defeats the task the same way §8 already
rules out for aggregation. TREC's `coarse_label` is that: six categories
(what kind of answer a question is asking for) assigned by human annotators,
not recoverable from surface string features. `bench/sources/trec/fetch.py`
is the one-shot fetcher; this module only reads and pins what it wrote.

`load_trec()` refuses if the vendored bytes drift from `trec_train.sha256` --
the manifest's `label_source` (`label_source_id()`) records exactly which
bytes every v2 answer was computed from, the same role `task_hash` plays for
v1's corpus documents (see `bench/manifest.py`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TREC_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
_SRC = Path(__file__).resolve().parent / "sources" / "trec"


@dataclass(frozen=True)
class Item:
    text: str
    label: str


def load_trec() -> list[Item]:
    data = (_SRC / "trec_train.jsonl").read_bytes()
    want = (_SRC / "trec_train.sha256").read_text().split()[0]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(f"vendored TREC moved: {got} != pinned {want}")
    return [Item(text=r["text"], label=TREC_LABELS[r["coarse_label"]])
            for r in (json.loads(l) for l in data.decode("utf-8").splitlines() if l)]


def label_source_id() -> str:
    return "CogComp/trec:train@sha256:" + (_SRC / "trec_train.sha256").read_text().split()[0][:16]
