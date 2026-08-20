# HIP: answer quality collapses when requests are decoded concurrently (100% -> 6% at two in flight); Vulkan on the same GPU is clean

## Summary

With one request in flight, the server answers a trivial extraction question
correctly 32 times out of 32. With **two** in flight — nothing else changed, same
prompts, same slots, same sampler, same process — it answers correctly 2 times
out of 32, and 30 of the 32 outputs are degenerate: character loops, 3-token
stubs, and identifiers truncated mid-string.

**It is the HIP/ROCm backend.** The same model, the same build, the same flags
and the same reproducer on the **Vulkan** backend of the same GPU are completely
clean at concurrency 2 — and concurrency there does what it is supposed to do,
cutting wall-clock while keeping every answer:

| build `b10375-ba360efe1`, same box, same model, same flags | c=1 | c=2 | degenerate @ c=2 | wall c=1 → c=2 |
|---|---|---|---|---|
| **HIP / ROCm** | 32/32 | **2/32** | 30 | 48.2 s → 68.9 s (slower) |
| **Vulkan** | 32/32 | **32/32** | 0 | 177.0 s → 126.4 s (1.40× faster) |

Fisher exact on the c=2 row: **p = 2.0e-13**. Both arms are the *same build*
(`b10375`), so this isolates the backend and not a version difference — the
collapse was independently measured on `b10375` and `b10488` under ROCm.

A dense model degrades on HIP too, but far less: `gemma-4-12B-it` Q8_0
(no `ssm.*` keys) scores 11/32 at c=2 with 4 degenerate, against the hybrid
MoE leaf's 2/32 with 30 (p = 0.011 between them). So the defect is not confined
to hybrid/recurrent models, but severity depends strongly on the model.

| concurrency | correct | degenerate | wall (s) | correct/s |
|---|---|---|---|---|
| 1 | **32/32** | 0 | 48.2 | 0.664 |
| 2 | **2/32** | 30 | 68.9 | 0.029 |
| 4 | 2/32 | 30 | 123.2 | 0.016 |
| 8 | 2/32 | 30 | 148.4 | 0.013 |

`b10488-9d77fa172`. Identical on `b10375-ba360efe1` (32/32, 2/32, 2/32, 0/32).

Note the last column: because the failures are near-total, **serial dispatch
beats every concurrent setting on correct answers per second, by 23x**. There is
no throughput being bought here.

## Environment

```
build       b10488-9d77fa172 and b10375-ba360efe1 (collapse on both, under HIP)
affected    HIP / ROCm 7.14, Windows 11, gfx1151 (Radeon 8060S, Strix Halo)
clean       Vulkan (AMD proprietary driver), same GPU, same box, build b10375
models      Qwen3.6-35B-A3B UD-Q4_K_M (arch qwen35moe, hybrid attention + SSM)
            gemma-4-12B-it Q8_0 (arch gemma4, dense SWA, no ssm.* keys)
```

```
llama-server --host 127.0.0.1 --port 8081 -m <model.gguf> \
  -c 327680 -np 128 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 \
  -lm none --no-kv-unified --cont-batching
```

## Reproducer

`concurrent_decode_repro.py` (attached) is self-contained — standard library
plus `httpx`. It generates its own corpus, renders prompts through the server's
own `/apply-template`, and scores by substring against an identifier that is
literally present in the document.

```
python concurrent_decode_repro.py --base http://127.0.0.1:8081 --doc-tokens 640
```

32 documents of ~780 rendered tokens, each stating one UUID-shaped record
identifier near the top; one question per document, whose answer is verbatim in
the text. The same 32 calls are issued at concurrency 1, 2, 4 and 8.

**Every call is pinned to a slot no other call in the run has used** (hence
`-np 128`), so that slot reuse cannot contribute to the result.

## Expected

Decode of one sequence does not depend on how many other sequences are in
flight. The score should be the same at every concurrency.

## Actual

Serial answers are clean and exact:

```
c2a7af9e-ab79-b005-6dd1-77d2c700d84c        (stop=eos, 32 predicted tokens)
c2b9546e-0f02-20f3-edb7-f1d5cbf15150        (stop=eos, 33 predicted tokens)
```

At concurrency 2, the same prompts on fresh slots:

```
//////////////////////////////////////////////////////////////85   (stop=limit, 64 tok)
e638bca4-6bd7-0d89-987f-c91e855cds                                 (stop=eos,   31 tok)
30                                                                 (stop=eos,    3 tok)
bypass                                                             (stop=eos,    3 tok)
61                                                                 (stop=eos,    3 tok)
```

**The second line is the important one.** That document's true identifier is
`e638bca4-6bd7-0d89-987f-c91e855cdff8`. The reply reproduces the first 24
characters exactly and then loses the tail, ending on `eos` mid-identifier. The
model found the right answer; the decode broke while emitting it. This is not a
model getting the question wrong — it is a correct decode losing tokens.

Across the run the degeneracy flags are `EOSCUT` 55, `CHARLOOP` 43, `STUB` 39
(`eos` fired mid-phrase; a character repeated 6+ times; fewer than 12 characters
or <= 2 predicted tokens). All are ~0 in the serial condition.

## What was ruled out (in the investigation this came from)

* **The prompt cache** — `--cache-ram 0 --no-cache-idle-slots` changes nothing
  at equal n (concurrency 2: 10/32 -> 9/32; concurrency 4: 5/32 -> 6/32, both
  Fisher p = 1.0), and `cache_n` was 0 on every call in the cache-on arm.
* **Slot reuse** — every call in the reproducer above takes a slot that no other
  call in the run has touched.
* **Sampling** — temperature 0 gives 6/128 at concurrency 8.
* **Client-side stream handling** — draining the stream instead of closing it
  gives 7/128.
* **Batch geometry** — `-ub 2048` gives 24/128, `-ub 128` gives 8/128; every
  concurrency-8 condition tried lands between 1.6% and 18.8% against serial's
  96.9%.
* **Thermal and memory** — a serial run started 13 s after a long eviction storm
  scored 31/32; memory is identical at concurrency 1 and 2 while quality falls.
* **`--no-kv-unified`, i.e. the ubatch split mode.** With that flag,
  `llama_memory_hybrid::init_batch` splits with `sequential = !unified`, which
  admits only consecutive increasing seq_ids into a ubatch
  (`src/llama-batch.cpp`), so it was the obvious suspect. It is not the cause:
  dropping the flag reproduces the collapse exactly.

  | | concurrency 1 | concurrency 2 |
  |---|---|---|
  | `--no-kv-unified` | 32 / 32 | **2 / 32** |
  | unified KV | 32 / 32 | **2 / 32** |

  Fisher p = 1.0 at both levels. This is also a third independent reproduction
  of the collapse on the same build.

## The `--no-cont-batching` observation (predates the backend finding)

At concurrency 2, `--no-cont-batching` restores correctness from 9/32 to
**27/32** (Fisher p = 1.08e-05 against the batched arm), which is statistically
indistinguishable from serial's 31/32 (p = 0.196). It does **not** hold at
concurrency 8 (7/128).

My reading is that `--no-cont-batching` does not stop concurrent sequences
sharing a decode batch;
it stops *new* requests being inserted into a batch already running. At two in
flight that is nearly enough to serialise them (and the wall clock agrees: the
"concurrent" arm is *slower* than serial). At eight, sequences still share decode
batches and the corruption returns. That puts the defect in multi-sequence
decode rather than in batch admission -- consistent with the backend result
above, since the Vulkan arm shares decode batches too and stays clean.

One more detail that may or may not matter: the corruption is **bursty, not
per-call**. Correct answers arrive in contiguous runs of roughly 5-25 requests
(permutation test p < 1e-4 across four conditions), so the server appears to
enter and leave a healthy regime rather than failing independently per request.
