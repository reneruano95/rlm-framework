"""One-shot: fetch CogComp/trec train split from the Hub and vendor it as jsonl.
    uv run --python 3.12 --no-project --with requests python bench/sources/trec/fetch.py
Writes trec_train.jsonl + trec_train.sha256 next to itself. Re-running must reproduce
the same bytes; if the source file changes, the sha moves and the manifest's
label_source records which bytes every v2 answer was computed from.

WHY THE CSV/loading-script URL, NOT PARQUET. The brief's candidate path
(`.../resolve/main/default/train/0000.parquet`) was checked against the live repo
first (`hf_fs ls hf://datasets/CogComp/trec --recursive`): the repo carries only
`trec.py` (a `datasets` loading script) + README, no parquet, and the Hub's
auto-conversion refuses it ("doesn't support this dataset because it runs
arbitrary Python code" -- 501 from the dataset viewer, checked 2026-09-03). So this
follows the brief's named fallback: `trec.py`'s own `_URLs["train"]`, the CogComp
source the loading script itself downloads from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

TRAIN_URL = "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label"
COARSE_LABELS = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
FINE_LABELS = [
    "ABBR:abb", "ABBR:exp",
    "ENTY:animal", "ENTY:body", "ENTY:color", "ENTY:cremat", "ENTY:currency",
    "ENTY:dismed", "ENTY:event", "ENTY:food", "ENTY:instru", "ENTY:lang",
    "ENTY:letter", "ENTY:other", "ENTY:plant", "ENTY:product", "ENTY:religion",
    "ENTY:sport", "ENTY:substance", "ENTY:symbol", "ENTY:techmeth", "ENTY:termeq",
    "ENTY:veh", "ENTY:word",
    "DESC:def", "DESC:desc", "DESC:manner", "DESC:reason",
    "HUM:gr", "HUM:ind", "HUM:title", "HUM:desc",
    "LOC:city", "LOC:country", "LOC:mount", "LOC:other", "LOC:state",
    "NUM:code", "NUM:count", "NUM:date", "NUM:dist", "NUM:money", "NUM:ord",
    "NUM:other", "NUM:period", "NUM:perc", "NUM:speed", "NUM:temp",
    "NUM:volsize", "NUM:weight",
]
HERE = Path(__file__).resolve().parent


def _parse_line(raw: bytes) -> dict:
    # trec.py's own parse: one non-ASCII byte in the upstream file
    # ("sister\xf0city") is replaced with a space before decoding.
    fine_label, _, text = raw.replace(b"\xf0", b" ").strip().decode("utf-8").partition(" ")
    coarse_label = fine_label.split(":")[0]
    return {
        "text": text,
        "coarse_label": COARSE_LABELS.index(coarse_label),
        "fine_label": FINE_LABELS.index(fine_label),
    }


def main() -> None:
    resp = requests.get(TRAIN_URL, timeout=30)
    resp.raise_for_status()
    rows = [_parse_line(line) for line in resp.content.splitlines() if line.strip()]

    out = HERE / "trec_train.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")

    data = out.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    (HERE / "trec_train.sha256").write_text(f"{sha}  trec_train.jsonl\n", newline="\n")
    print(f"wrote {len(rows)} rows, sha256={sha}")


if __name__ == "__main__":
    main()
