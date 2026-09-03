# TREC question classification -- vendored label source

`trec_train.jsonl` is the `train` split (5,452 rows) of the TREC Question
Classification dataset, vendored so benchmark v2's linear-semantic and
interactive tasks have a per-item label that no deterministic program can
derive from the item's text -- only a model reading for meaning can supply
it. Each row is `{"text": str, "coarse_label": int, "fine_label": int}`;
`coarse_label` indexes `TREC_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC",
"NUM")` in `bench/corpus_v2.py`.

## Source

Fetched by `fetch.py` from
`https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label` -- the same URL
the `CogComp/trec` Hugging Face dataset's own loading script
(`trec.py`, `_URLs["train"]`) downloads from. The brief's candidate path,
a Hub-hosted parquet mirror, does not exist: `CogComp/trec` ships only a
`datasets` loading script, and the Hub's auto-parquet-conversion refuses it
("doesn't support this dataset because it runs arbitrary Python code" -- 501
from the dataset viewer, checked 2026-09-03 via `hf_fs`). This follows the
brief's named fallback instead.

Citation: Li & Roth, "Learning Question Classifiers", COLING 2002;
Hovy et al., "Toward Semantics-Based Answer Pinpointing", HLT 2001.
Homepage: https://cogcomp.seas.upenn.edu/Data/QA/QC/

## Licence status

`unknown` on the `CogComp/trec` Hub dataset card (`license: unknown` in its
README front matter). CogComp distributes the file for research use; no
further grant is recorded. If that status becomes a blocker, the named
fallback is `PolyAI/banking77` (CC-BY-4.0, unambiguous licence), which the
same `Item(text, label)` shape can be re-pointed at without changing
`bench/corpus_v2.py`'s interface.

## Pinning

`trec_train.sha256` pins the exact bytes of `trec_train.jsonl`.
`bench.corpus_v2.load_trec()` refuses (raises) if the file's sha256 no
longer matches -- so `label_source_id()`'s manifest string
(`CogComp/trec:train@sha256:<first 16 hex chars>`) always names the bytes
every v2 answer was actually computed from.

## Never edit `trec_train.jsonl` by hand

This file is a vendored artifact, not source code. If the upstream data
needs to change, re-run `fetch.py` (or point it at a new source) and let it
regenerate both `trec_train.jsonl` and `trec_train.sha256` together --
never hand-edit either file. A hand-edited jsonl with a hand-computed (or
stale) sha256 breaks the guarantee this vendoring exists for.

## Re-fetching

```
uv run --python 3.12 --no-project --with requests python bench/sources/trec/fetch.py
```

Writes `trec_train.jsonl` and `trec_train.sha256` next to itself, LF line
endings, `ensure_ascii=True`, original row order. Re-running against an
unchanged upstream file reproduces the same bytes and the same sha.
