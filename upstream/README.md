# Upstream bug reports (llama.cpp)

Two defects found while building this project, both **re-verified on
`b10488-9d77fa172`** (2026-08-18, current at time of writing) as well as on the
project's pinned `b10375-ba360efe1`. Neither is fixed upstream; neither appears
to be reported in this form.

Everything here is self-contained: standard library plus `httpx`, no imports
from `rlm.*`, no repo checkout required beyond this directory.

| file | what it is |
|---|---|
| `ISSUE-slot-leak.md` | issue body — a reused slot injects a previous document's content |
| `ISSUE-concurrent-decode.md` | issue body — answer quality collapses under concurrent decode |
| `r13_repro.py` | reproducer for the slot leak |
| `make_fixtures.py` | deterministic corpus generator the slot-leak repro needs |
| `concurrent_decode_repro.py` | reproducer for the concurrent-decode collapse |

## Slot leak

```bash
# 1. build the corpus (deterministic from --seed; uses the server's /tokenize)
python make_fixtures.py --leaf-port 8081 --out ./fixtures

# 2. paired run: every prompt goes to an accumulating slot AND a virgin slot
python r13_repro.py --port 8081 --replay-fixtures ./fixtures --replay-trials 3 \
  --temperature 0.3 --n-predict 512 --slot 0 --paired-virgin --out results.jsonl
```

Server: `-c 327680 -np 8 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 -lm none
--no-kv-unified --cont-batching --slot-save-path <dir>`

Measured: shared slot 37/54, virgin slot 0/54 on b10488 (Fisher p = 8.3e-16);
35/54 vs 0/54 on b10375. Counting only full UUIDs from another document —
values that cannot appear by chance — 4/54 and 2/54 vs 0/54.

## Concurrent decode

```bash
python concurrent_decode_repro.py --base http://127.0.0.1:8081 --doc-tokens 640
```

Server: as above but `-np 128` (every call takes a slot no other call has used,
so slot reuse cannot contribute).

Measured, both builds: **32/32 correct serial, 2/32 at two requests in flight**,
with 30 of 32 outputs degenerate.

## A note on scoring, for anyone re-running these

Both reproducers ship a positive control, and both need one. While preparing
these reports, three separate instruments produced confident, wrong answers
until a control caught them:

* a from-scratch rewrite of the slot-leak repro scored 0/36 on a build where the
  bug is present — its probe question was ambiguous against the corpus;
* a first draft of the concurrency repro scored 0/32 *serially*, because it
  placed the answer ~1,076 tokens from the question, past this model's effective
  retrieval reach — nothing to do with concurrency;
* the slot-leak harness's own coarse verdict field counts proper nouns and does
  not exclude the entity named in the question, so it reports 37/54 where a
  strict identifier-only rule reports 2/54.

The numbers quoted in the two issue bodies exclude the question's own entity and
are the same rule under which the original investigation reported its figures.
If you re-run these, check the serial / virgin arm first: if the control arm
cannot answer correctly, the experimental arm's zeros mean nothing.
