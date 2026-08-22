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

## Corrections and new evidence, 2026-08-20

These were found while re-examining the bundle before filing. Two of them would
have been visible to a maintainer within minutes, which is why nothing was filed.

**The reproducer did not run.** `make_fixtures.py` raised
`NameError: _word_ends` on its first call — the helper was dropped when the
bundle was extracted to be standalone, so it never produced a single fixture.
Restored verbatim from `milestones/s1/make_fixtures.py` and verified with `--offline`.

**The slot-leak headline was scored with two different rules.** The draft led
with 37/54 shared vs 0/54 fresh. The 37 comes from a coarse scorer that counts
any proper noun shared with another document; applied to the *control* that same
scorer gives 5/54, not 0. The coined organisation names come from one
generator-wide pool and recur across documents, so that rule cannot separate the
arms. Only the per-document UUID does. Re-run at 6 trials on the strict rule:

| | shared slot | fresh slot | Fisher |
|---|---|---|---|
| b10488, 6 trials | **7 / 108** | 0 / 108 | **p = 0.014** |
| b10488, 3 trials (original) | 4 / 54 | 0 / 54 | p = 0.118 |
| b10375, 3 trials (original) | 2 / 54 | 0 / 54 | p = 0.495 |

The leak is real; the original n could not show it. `r13-b10488-ckpt32.jsonl`.

**Context checkpoints are a lead, not a cause.** Disabling them takes the shared
arm from 7/108 to 2/108, but that difference is p = 0.17 and 2 is not 0.
Separating 6.5% from 1.9% needs ~350 trials per arm.
`r13-b10488-ckpt0.jsonl`.

**`--no-kv-unified` is ruled out for the concurrent-decode collapse.** Unified KV
reproduces it exactly: 2/32 at concurrency 2 either way, p = 1.0.
`r14-b10488-kvu-off.jsonl`, `r14-b10488-kvu-on.jsonl`.

**The concurrent-decode collapse is a HIP backend defect, not a server defect.**
Same build (`b10375`), same box, same model, same flags, same reproducer:
ROCm 2/32 at concurrency 2, **Vulkan 32/32** (Fisher p = 2.0e-13) — and on Vulkan
concurrency is a real win, 177.0 s → 126.4 s wall. A dense model
(`gemma-4-12B-it`) also degrades on HIP but far less (11/32, 4 degenerate),
so severity is model-dependent rather than hybrid/recurrent-only.
`r14-vulkan-leaf.jsonl`, `r14-rocm-gemma.jsonl`.

This has a project consequence beyond the bug report: R14 is the reason
`scaffold.dispatch_concurrency` is pinned to 1 and the reason S2 gate (c) has
never been scored. On Vulkan the leaf answers 32/32 at concurrency 2, so the
gate is scoreable for the first time — at the cost of S0's measured
prefill difference between the backends (leaf 32K prefill 949.9 t/s ROCm vs
836.2 Vulkan, -11.9%). That is now a trade with numbers on both sides.

---

## Rebuilding the b10488 binary (2026-08-22)

`tools/llamacpp-rocm-b10488/` was **deleted on 2026-08-22**. It was 1.03 GiB,
referenced by no config — the only mention anywhere was a comment at
`config.yaml:89` — and it existed solely as the binary behind the
`r13-b10488-*.jsonl` and `r14-b10488-*.jsonl` results in this directory. The
results themselves are committed and untouched; only the binary is gone.

If a maintainer asks for a re-run on b10488, rebuild it in two steps:

1. Re-download `llama-b10488-bin-win-rocm-7.14-x64.zip` from the public
   ggml-org llama.cpp releases. The local archived copy was deleted on
   2026-08-22 along with `D:/ARCHIVE`; the release is the source of record.
   That gives 99 of the 354 files.
2. Copy the **255-file AMD-wheel graft** from `tools/llamacpp-rocm/` — the
   pinned b10375 build, which is live and carries the identical graft.
   Verified 2026-08-22: the graft file *sets* are equal and spot-checked
   sha256-identical. It is `hipblas.dll`, `hipblaslt/library/gfx1151/**`,
   `rocblas.dll`, `rocsolver.dll` and `.kpack/blas_lib_gfx1151.kpack`.

So the graft does **not** need re-deriving from AMD wheels; it is a copy from a
build that is still on disk. That is what made the deletion cheap enough to
take. If `tools/llamacpp-rocm/` is ever also removed, this recipe dies with it
and the graft must come from the wheels again — do not delete both.
