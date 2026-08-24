# The ~1,000-token horizon is a ROCm backend defect

**Date:** 2026-08-23 · **Status:** measured. ~1,750 leaf calls across seven arms, three model
families, two backends. **0 leaks, 0 slot mismatches** on every arm.
**Supersedes the mechanism, not the measurement, of `docs/research/2026-08-23-distance-cliff-audit.md`.**

---

## 0. The finding, in one table

Same GGUF. Same llama.cpp commit — **both builds report `b10375 (ba360efe1)`**. Same flags. Same
GPU. Byte-identical prompts: `chunk_sha256` and `prefix_sha256` match 84/84 across the two arms and
`tokens_in` agrees to the token. The only difference is which backend directory `llama-server.exe`
was launched from.

Literal-identifier recall, by needle→question distance, 12 independent facts per cell:

| arm | 750 | 925 | 950 | 975 | 1000 | 1025 | 1050 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROCm, `q8_0` K/V, `-fa on` — **shipped** | 4/12 | 12/12 | 12/12 | 12/12 | 4/12 | **0/12** | **0/12** |
| ROCm, `f16` K/V, `-fa on` | 4/12 | 12/12 | 12/12 | 12/12 | 6/12 | **0/12** | **0/12** |
| ROCm, `f16` K/V, `-fa off` | 4/12 | 12/12 | 12/12 | 12/12 | 5/12 | **0/12** | **0/12** |
| ROCm, `ROCBLAS_USE_HIPBLASLT=0` | — | — | — | — | — | **0/12** | **0/12** |
| **Vulkan**, `q8_0` K/V, `-fa on` | **12/12** | **12/12** | **12/12** | **12/12** | **12/12** | **12/12** | **12/12** |

Across the full 13-point sweep at 24 facts per point, Vulkan is **156/156** correct from 600 to
1,500 tokens, plus **84/84** on the boundary cells — **240/240**. ROCm is **0/144** at every
distance from 1,025 to 1,500.

**There is no ~1,000-token horizon. There is a ROCm backend that stops attending correctly to
content more than roughly a thousand positions behind the generation point.**

---

## 1. What was excluded, and how

| candidate | test | result |
|---|---|---|
| box drift since the earlier arms | ROCm re-run live, same three far cells | **0/36** — still fails |
| hipBLASLt | `ROCBLAS_USE_HIPBLASLT=0` | **0/36** — not the cause |
| KV cache quantization | `f16` K/V, doubling the cache | identical curve |
| flash-attention kernel | `-fa off` (with `f16`, since quantized V requires FA) | identical curve |
| Vulkan silently on CPU | server log | `Vulkan0 (AMD Radeon 8060S)`, **41/41 layers offloaded**, KV 3,400 MiB and RS 8,040 MiB — the same allocation ROCm makes |
| the model | two other families, below | not the model |
| the probe | the absent question, below | not the probe |

---

## 2. It is not the model, and not hybrid attention

Both control models were verified **UNIFORM GLOBAL ATTENTION** from their GGUF headers before
running — no `ssm.*` keys, no `sliding`/`window` keys. This also closes the R7 checklist item open
since v0.3.2, which needed a uniform-attention instruct model that was not on disk.

| model | attention | ROCm | Vulkan |
|---|---|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_M` (the leaf) | hybrid, 10 of 40 layers global | **0/144** past 1,025 | **240/240** |
| `Qwen2.5-7B-Instruct-Q8_0` | uniform global | **0/25**, plus **23 stream failures** | **48/48** |
| `Meta-Llama-3.1-8B-Instruct-Q8_0` | uniform global | **0/48, every one MALFORMED** | **48/48** |

Three model families, two attention architectures, three parameter scales, two vendors. Every one
correct on Vulkan; every one broken on ROCm.

### 2.1 The smallest form of the defect needs no fixtures at all

`upstream/prompt_length_repro.py`: a 12-hex marker generated from a seed is planted N tokens before
the question and must be repeated back verbatim. Repeating a string you were handed is the least a
model can do — nothing to reason about, nothing to count, no world knowledge — so a wrong answer is
a serving fault and not a capability limit. `Qwen2.5-7B-Instruct-Q8_0`, same flags, same GPU:

| marker tokens before the question | 80 | 584 | 1,200 | 1,760 | 2,936 |
|---|---|---|---|---|---|
| ROCm | ok | **WRONG** | **HTTP 500** | **WRONG** | **HTTP 500** |
| **Vulkan** | ok | **ok** | **ok** | **ok** | **ok** |

Both pass the 80-token positive control, so the model and the flags are not the problem.

**Two probes that do NOT work, recorded so nobody re-derives them.** *Counting* occurrences of a
repeated sentence: a 7B genuinely cannot do it, both backends fail, and the test discriminates
nothing — a first draft of this reproducer failed its own positive control on both backends, which
is the only reason its numbers were not reported as a difference. And *anything scored by eye on
repetitive filler*: "repeating because degenerate" and "repeating because the source repeats" are
the same string, a point a peer session established independently by nearly publishing a 23.5%
n-gram flag rate as a defect rate when the flagged answers were all correct quotations.

The earlier free-form probe is kept only as colour, with its limitation stated: at ~1,200 tokens
Llama-3.1-8B on ROCm answered `"The end of the end of the end of the end of the of the end of…"`
while on Vulkan it answered with a tool call — coherent, but not a scoreable answer, which is why
the marker probe replaced it. Qwen2.5-7B returned **HTTP 500** on ROCm at that length and `"20"` on
Vulkan.

---

## 3. It is not the probe: the bluff control

Every fixture carries **exactly one** UUID-shaped string by construction, so "emit the only UUID you
can see" scores 100% on the literal question without reading anything. The **absent** question asks
for the key of an organisation that owns none — the only correct answer is a refusal, and the
degenerate strategy scores zero. **Paraphrase** answers are assembled from two separate lines and
never appear verbatim.

2,048-token chunks, distances 950–1,500, 12 seeds:

| backend | literal | paraphrase | **absent (correct = refuse)** |
|---|---:|---:|---:|
| ROCm | 12/48 | 3/48 | **0/48** |
| **Vulkan** | **48/48** | **34/48** | **47/48** |

Vulkan emits a clean `NONE` 47 times out of 48 while getting every literal right. **Vulkan is
retrieving, not bluffing.** ROCm never refuses — it answers `'Cenregnfield'`, `'The key is 42.'`, or
echoes the question back.

*This control exists because a peer session pushed for it after a clean first sweep. It was the
right push: without the absent column, the literal column proves nothing.*

---

## 4. The shipped geometry is NOT affected

The question that decides whether the benchmark is implicated. A peer session measured the trace
store: **29,517 recorded leaf calls, median `tokens_in` 1,023, 95.3% at or above 1,000** — every
leaf call this project has made sits near the length where the Llama-3.1 sanity probe degenerates.

But that conflates two axes. E1 varied **distance inside a constant ~2,400-token prompt**; the trace
number is **total prompt length**. The shipped 640 geometry is the one cell where they disagree:
total ~1,000 tokens, needle under 640 back.

Measured on the archive's **own** 640-token fixtures (`fixtures-refusal-640-s{2,3,4,5,7,9,10}`,
recovered from `4e75b53` — the exact family the project's 640 claims rest on), needle 260–340 tokens
from the question, real pinned prefix, greedy, 7 seeds:

| backend | literal | paraphrase | absent (refuse) |
|---|---:|---:|---:|
| ROCm | 7/7 | 6/7 | **7/7 — all `NONE`** |
| Vulkan | 7/7 | 6/7 | **7/7 — all `NONE`** |

**Identical, cell for cell, including the single paraphrase both miss.** And the archive's own grid
says the same:

| chunk | rendered prompt | needle distance | ROCm |
|---:|---:|---:|---:|
| 640 | ~1,000 | 309–357 | 24/24 |
| 1,024 | ~1,400 | 453–566 | 24/24 |
| 2,048 | ~2,400 | 958–989 | 12/12 |
| 2,048 | ~2,400 | 1,003–1,091 | **0/12** |

**The axis is distance, not total length.** A 1,400-token prompt is clean when its needle is 500
tokens back. A 2,400-token prompt fails when its needle is 1,003 back. The shipped geometry keeps
every needle inside the horizon at both ends, so the benchmark's leaf calls are not degraded by this
defect. That is not a clean bill of health for S4 — nobody re-ran it — but the mechanism under
suspicion does not reach it.

**Read that table carefully: it licenses nothing at a 1,024 window.** Retrieval-clean and
instruction-clean are different cells at the same chunk size. The 1,024 row is 24/24 on *retrieval*,
where the needle sits 453–566 tokens back — while at that window the system PREFIX sits ~1,100 back,
past the bracket, which is exactly the 30/30 false-positive result §5 reattributes. And this spec's
own standing admission is unchanged: **"the margin is NOT measured: no window between 640 and 1,024
has ever been sampled"** (`config.yaml`). Nothing measured here or in the audit samples a *window*
between 640 and 1,024, so the largest safe chunk on ROCm remains unknown.

### 4.1 The shipped geometry is not corrupted. It is overpriced.

This is the consequence with product weight, and the invoice is itemised in the project's own
config. 640/480 against 1,024/768 over a 200,000-token corpus:

| | 1,024/768 | 640/480 | Δ |
|---|---:|---:|---:|
| windows | 268 | 424 | **+58%** |
| served prompt tokens | 515,903 | 646,130 | **+25.2%** |
| leaf wall, serial | ~884 s | ~1,180 s | **+33%** |

plus `max_subcalls` 522 → 926. The stated reason for paying it, verbatim from §7 #2: *"at 1,024 the
leaf answers a question it cannot see the instructions for, so the cheaper geometry buys answers R5
says are no evidence of anything."*

**On Vulkan that sentence is false.** At 2,048-token chunks with the needle 950–1,500 tokens back
the leaf refuses 47/48. The project is paying +33% leaf wall clock to avoid a defect that exists on
one of its two backends.

So 640 is not a floor to be defended — it is a **ceiling imposed by ROCm**. Moving the leaf to
Vulkan does not merely remove a correctness risk; it unlocks a geometry that is measurably cheaper
at the same correctness. That makes the backend question a performance decision as much as a
correctness one, and it is still the owner's: see §7.

---

## 4.2 A live trap: `-ctxcp 0` is a Vulkan-only lever

Gate 0 measured that disabling llama.cpp's context checkpoints cuts the ROOT's median turn from
13.88 s to 2.48 s — **−82%**, with the cached fraction identical to four decimals
(`docs/research/2026-08-22-gate0-soak.md` §6). The root is on Vulkan. **Do not carry that flag to the
leaf while the leaf is on ROCm.**

Shipped 640 geometry, ABSENT question on a cold virgin slot, 7 seeds, greedy. Correct = refuse:

| `-ctxcp` | `--cont-batching` | refused |
|---|---|---:|
| default | yes — **what ships** | **7/7** |
| **0** | yes | **0/7** |
| default | no | **7/7** |
| 0 | no | **0/7** |

Fisher two-sided with `--cont-batching` held constant: **p = 0.00058**. With `-ctxcp` held constant:
**p = 1.0**. So it is the checkpoint flag, not continuous batching. The seven failures are confident
wrong-entity UUIDs, byte-identical across two independent runs — intra-chunk misattribution on the
first call of a fresh slot, not a cross-call leak.

**On Vulkan the same flag is harmless**: `-ctxcp 0` scores 14/14, and it is also the configuration
that makes Vulkan fast (below).

*Two things this run also settles, both incidental and both worth having.* `servers.bench_leaf`
ships **without** `--cont-batching` (`config.yaml:252`) and served B1 and B3 in S4 — the p = 1.0 row
says that is harmless. And the first version of this 2×2 was invalid: the harness's
`--extra-flags` override replaces the flag list wholesale, so the first ROCm arm silently dropped
`--cont-batching` and varied two things at once. The isolating cell was run only after a review
caught it.

## 4.3 What the backends actually cost at the shipped geometry

Same fixtures, ~995-token prompts, 7 seeds. The checkpoint flag dominates everything:

| config | cold prefill | warm prefill | reuse | correct |
|---|---:|---:|---:|---|
| **ROCm** default — what ships | 1,020 ms | **581 ms** | 484 tok | 7/7, 7/7 |
| ROCm `-ctxcp 0` | 979 ms | — | 0 | **0/7** |
| Vulkan default | 3,627 ms | 3,163 ms | 484 tok | 7/7 |
| **Vulkan `-ctxcp 0`** | **897 ms** | — | 0 | 14/14 |

**Vulkan's apparent 5× slowness is the checkpoint tax, not compute.** With `-ctxcp 0` its cold
prefill (897 ms) is *faster* than ROCm's (1,020 ms). But the checkpoints are also what buy the leaf
its intra-window warm re-query — `-ctxcp 1` still pays the full tax and buys no reuse (3,620 ms,
cached 0); `-ctxcp 2` buys reuse and pays the tax (3,195 ms, cached 478). **There is no Vulkan
configuration that gets both.**

Per window of Q questions: ROCm `1.05 + (Q−1)×0.61` s against Vulkan-`ctxcp 0` `Q × 1.20` s. ROCm
wins **1.1× at Q=1 and 1.8× at Q=10** — far less than the 4× a naive reading of the default-flag
numbers suggests, and the gap is entirely the warm path.

## 4.4 THE DECIDING MEASUREMENT — Vulkan wins the leaf, and by 2x

§4.3 compared per-call latency and gave ROCm the warm re-query. That is the wrong unit. The question
is not which backend is faster per call — it is **which backend covers a corpus faster at equal
correctness** — because the backend constrains both dials throughput depends on: the window size
(ROCm caps it at 640 to keep the needle inside its horizon) and the concurrency (R14 pins
`dispatch_concurrency: 1` because HIP collapses at two calls in flight).

Instrument: the project's own `upstream/concurrent_decode_repro.py`, unmodified. It generates its own
corpus, applies the template through the server, **pins every call to a slot no other call has used**
— which is R13's production policy, so no warm re-query for either backend — and scores by substring
against a UUID verbatim in the document. 32 calls per cell. `-ctxcp 0` on the Vulkan arms and default
on ROCm: each backend's best *safe* configuration, per §4.2.

| arm | doc tokens | c=1 | c=2 | c=4 | c=8 |
|---|---:|---|---|---|---|
| **ROCm 640** — what ships | ~830 rendered | **32/32**, 0.669 correct/s | **2/32**, 30 degenerate | — | — |
| **Vulkan 640** | ~830 | **32/32**, 0.756 | 32/32, 0.864 | 32/32, 0.961 | 32/32, **0.977** |
| **Vulkan 1024** | ~1,129 | 32/32, 0.591 | 32/32, 0.656 | 32/32, 0.698 | 32/32, 0.732 |
| **Vulkan 2048** | ~2,200 | 32/32, 0.380 | 32/32, 0.402 | 32/32, 0.416 | 32/32, 0.423 |
| **ROCm 2048** | ~2,200 | **0/32** | — | — | — |

**Vulkan is faster than ROCm at the shipped geometry even serially** — 0.756 against 0.669 correct
answers per second, +13% — and 1.46x at concurrency 8, with 32/32 at every level. ROCm collapses to
**2/32** with two calls in flight, re-confirming R14 on this build. At 2,048 tokens, where the needle
sits ~2,000 back, ROCm scores **0/32**: the wall, in the instrument the project already trusted.

Normalised to one 200,000-token corpus (window counts at 640/480 and 1024/768 are §7 #2's **measured**
chunker output; 2048/1536 is arithmetic and marked):

| configuration | windows | s/call | corpus wall | vs what ships | correct |
|---|---:|---:|---:|---:|---|
| ROCm 640/480 serial — **ships today** | 424 | 1.50 | **635 s** | — | 32/32 |
| Vulkan 640/480, c=8 | 424 | 1.02 | 433 s | **1.46x** | 32/32 |
| Vulkan 1024/768, c=8 | 268 | 1.37 | 366 s | **1.73x** | 32/32 |
| **Vulkan 2048/1536, c=8** | 131 *(arithmetic)* | 2.36 | **309 s** | **2.05x** | 32/32 |
| ROCm 2048/1536, c=1 | 131 | 3.13 | 410 s | n/a | **0/32** |

**Vulkan at a 2,048-token window with eight calls in flight covers the corpus in 309 s against ROCm's
635 s at 640/480 serial — 2.05x, at 100% correctness on both sides.** Sub-calls fall from 424 windows
to 131, which also relieves `max_subcalls` (926 → ~290) and §8's corpus ceiling.

**So §4.3's conclusion is superseded by its own successor.** ROCm's per-call advantage was real but it
lived entirely in the intra-window warm re-query, and under R13's never-reuse policy the first call of
every window is cold — which is what this measures. On the unit that decides the question, Vulkan wins
outright and the margin grows with the window ROCm cannot use.

**Limits, stated rather than buried.** The documents are the reproducer's, not the project's chunker
output. One question per document — production re-queries within a window, which is the one place
ROCm's warm path still helps, and it is unmeasured here. The 2048/1536 window count is arithmetic; the
real chunker's snap-back raised the 640 count from 417 to 424, so expect a similar few percent. One
run per cell, one box. And Vulkan's concurrency scaling is modest — 1.29x from c=1 to c=8 at 640, not
linear — so most of the 2.05x comes from the larger window, not from fan-out.

## 4.5 R13 IS A ROCm DEFECT TOO — and this is the one with product weight

R13 is the cross-request leak: a slot that has held one document injects its content into the answer
to a later, unrelated one. `DIRECTION.md` calls it a **privacy defect** and **product-blocking**; the
2.2% upper bound is a number a customer has to be shown; and the entire `never_reuse` slot policy,
the 128-slot pool, the server rotations and `MAX_ROTATIONS_PER_CALL` exist to work around it.

Run on both backends with the repo's own `upstream/r13_repro.py`, unmodified — same fixtures, same
flags, same commit, 108 calls each, scored by the reproducer's own `FOREIGN` verdict:

| backend | shared slot | virgin slot | total |
|---|---:|---:|---:|
| **ROCm** | **32/54** | 3/54 | 35/108 |
| **Vulkan** | **0/54** | **0/54** | **0/108** |

**Fisher two-sided p = 2.15e-12.** The shared-slot arm — R13's entire subject — goes from 32/54 to
zero by changing which directory the server was launched from.

**Zero is not zero.** 0/108 gives a 95% upper bound of **2.8%** by the rule of three, which is not
better than ROCm's existing 2.2% virgin-slot bound in absolute terms; the contrast that matters is
the *shared* arm, where ROCm is 32/54. "Leak-free" is not written here for Vulkan any more than it
was for ROCm.

**How this went unseen for ten days:** every R13 measurement in the project was taken on ROCm,
including the control that "falsified the recurrent-state hypothesis" by showing a non-recurrent
model leaking *more*. That control was right about what it tested and wrong about the frame — both
arms were on the same defective backend, so it compared two models inside one fault instead of
isolating the fault.

**It also explains a smaller thing seen earlier today.** The long-context probe at `-np 2` had ROCm
answering with the marker from **two requests earlier**, twice — which at two slots is the same slot.
Not "degenerate output": R13, in the same probe that shows the reach limit. The two defects compose.

## 4.6 THE SWITCH, AND WHAT WAS DELIBERATELY NOT SWITCHED

`config.yaml` now serves both leaf profiles on Vulkan (v0.3.22). The decision was the owner's; the
evidence is §§4.4–4.5 plus the long-context validation below.

**`bench_leaf` could not be left behind** — `tests/test_config.py:619` asserts it shares the leaf's
backend and model — and it serves B1/B3 through a **262,144-token slot**, a regime nothing else here
exercises. Measured before following, marker planted near the top and asked for at the bottom:

| prompt tokens | 2,168 | 8,635 | 34,482 | 107,726 |
|---|---|---|---|---|
| **Vulkan** | ok | ok | ok | ok — 443 t/s |
| ROCm | WRONG | WRONG | WRONG | ok — 464 t/s |

Vulkan is 4/4 with its positive control passing and no long-context throughput regression. ROCm's row
is erratic and its own short-prompt control only passes below ~1,000 tokens; **the 100K cell passing
on ROCm is unexplained and is recorded as an anomaly, not explained away.**

**A consequence for §8 that follows directly, and is not this document's to settle:** B1's recorded
**0/30** was taken on ROCm through this profile, where a single-shot 256K prompt puts its answer far
past the ROCm reach limit. S4 described that 0/30 as *"consistent with the measured distance cliff"*.
It is now suspect as a **backend artifact rather than a baseline**, and RLM's **+30 margin over B1**
rests on it. Re-running the baselines on another backend is a §8 comparability event.

### 4.6.1 B1: what its outputs do and do not say

A peer session read all **117** of B1's archived outputs from the trace blobs, at prompts of
27,382–129,555 tokens. The distribution:

- **46× `NONE`** — an honest refusal in the format the `b1` prompt demands
- bare values: 6× `'14'`, 4× `'6'`/`'1'`/`'3'`/`'2'`, 3× `'15'`/`'5'`/`'9'`
- 6× `'=== FILE: rlm/bridge.py ==='`, 3× `'=== FILE: rlm/leak_verdict.py ==='`
- longest: *"The organisation's full name is the Glinfallowwardine Chancery Annexe."*

**B1's output is not degenerate.** No repetition, no runaway, no malformed text, nothing resembling
`"The end of the end of the end of"`. So the *degenerate-decode* half of the ROCm defect did not
happen on that arm.

**And that does not settle it, for a reason worth stating precisely.** A reach limit on RETRIEVAL
does not produce gibberish — it produces exactly this: coherent, correctly formatted answers that
are wrong, plus honest refusals when the model cannot see content that is in fact present. §3's own
ROCm cell shows the same shape: 12/48 literal, **wrong rather than malformed**. A model genuinely
unable to do single-shot retrieval at 100K produces an identical distribution. **39% `NONE` is
consistent with both readings and discriminates neither.**

*A false positive the peer chased and discarded before reporting, recorded because it is the shape
of error this whole document is about:* three answers were the identical entity
`'Xanthlammertofts Tithe Barn'`, which looks like the wrong-entity misattribution signature. All
three have `tokens_in` of exactly 27,395 — one benchmark item repeated across three episodes,
answered deterministically under greedy. Checked before claiming, not after.

**Evidence pointing the other way, from §4.6's own table:** ROCm answered the **107,726-token** cell
correctly. If ROCm could retrieve at 100K in that probe, B1's prompts are not automatically past its
reach. That cell is the anomaly noted above, and it cuts against the backend explanation for B1.

**So the question is open in both directions and is cheap to close.** The discriminating measurement
is not a re-run of S4: it is the marker probe at B1's actual prompt sizes — 27K, 80K, 130K, several
trials each, both backends. If ROCm retrieves reliably there, B1's 0/30 is the model and RLM's +30
stands. If it does not and Vulkan does, S4's largest margin was measuring a backend. The full B1
re-run (117 single-shot calls on the now-shipped Vulkan config, roughly six hours of box time) is
the confirmatory step *after* that, not instead of it.

### 4.6.2 B1's OWN CELL, MEASURED — and it goes against B1

The reconciliation took four probes, three of which tested the wrong cell. Recorded in order,
because the wrong ones are why the right one is trustworthy.

**What B1's serving condition actually is**, established from the trace store by a peer session and
from the code by me: **one call per episode, all 117 on slot 0, zero rotations**, and
`cli.py:1128`'s `_bring_up_leaf` returns 0.0 when the profile already matches — so
`swap_to(BENCH_PROFILE)` is a no-op and **all 117 episodes ran on one persistent process**. Worse
for the instrument: those 117 calls are only **20 distinct documents, repeated up to 22×**.

**The mechanism was in this repo ten days ago.** `upstream/ISSUE-slot-leak.md:156`: *"The restore
path also runs only when `n_past > 0`, which would explain why this never reproduced on a
high-entropy corpus: with no shared prefix `n_past` is 0 and no checkpoint is restored."* So the
gate is **shared prefix**, not entropy — and B1, sending the same 130K document through the same
slot up to 22 times, is the most extreme shared-prefix condition in the benchmark.

The 2×2 that had to be filled, with the cell each probe actually tested:

| | fresh process | persistent process + reuse |
|---|---|---|
| **repeated sentence** (no shared prefix at the marker) | ROCm **3/3** | ROCm **2/12** |
| **real prose + maximal shared prefix** | — | **ROCm 0/4, Vulkan 4/4** ← B1's cell |

**B1's cell: `agg-01.txt` prose, marker at 95% depth so four consecutive calls share ~95% of the
prompt byte-for-byte, persistent process, 130K tokens.**

| trial | ROCm | Vulkan |
|---|---|---|
| 0 | `'[ENT-100090] Rookenethergriff Tith…'` | **ok** |
| 1 | `'[ENT-100090] Rookenethergriff Tith…'` | **ok** |
| 2 | `'100000'` | **ok** |
| 3 | `'//////////////////////////////////'` | **ok** |
| | **0/4** | **4/4** |

The within-cell control fails on ROCm: the marker sits a few hundred tokens from the question, so
**this is not a reach limit** — it is corruption under shared-prefix reuse.

**AND HERE IS WHY B1's DETECTOR SAW NOTHING.** `[ENT-100090] Rookenethergriff Tithe Barn` is
**verbatim in `agg-01.txt`** — the document that was in the call's own input. A foreign-string check
flags content absent from the call's own input; content drawn from the right document is not
foreign and cannot be flagged. **The failure signature is invisible to the instrument by
construction**, and B1's dominant case — the same document repeated up to 22× through one slot — is
exactly where the blindness is total. The detector's power lived in ~20 first-sight calls, not 117,
so the "2.6% by the rule of three over 117 calls" bound offered earlier is much weaker than stated
and is withdrawn.

**This reconciles the two facts that would not sit together.** B1's 117 outputs were coherent,
correctly formatted and detector-clean *because* the corruption draws on the document already in
the prompt. Coherent, clean, and wrong are not in tension — they are the same event.

**What this licenses, and what it does not.** Measured: in B1's own serving condition, on B1's own
corpus, the backend B1 ran on returns wrong answers drawn from the corpus, 0/4, where Vulkan is 4/4.
**B1's recorded 0/30 is therefore plausibly a backend artifact, and RLM's +30 margin over B1 rests
on it.** Not measured: whether B1's actual questions would be answered on Vulkan — the marker task
is verbatim retrieval and B1's questions are not, and n=4 is four. **The confirmatory step is the
B1 re-run** — 117 single-shot calls on the now-shipped Vulkan config, ~6 hours — and the evidence
now points at it being worth the box time. Under §8's comparability rule that is the owner's call.

### 4.6.3 `never_reuse` did not hold, and the exposure is inverted

Established by a peer session from the trace store and **re-derived independently here** by joining
`steps` against `4e75b53:milestones/s4/results/ledger*.jsonl` (my join covers two of the seven
ledgers, so absolute counts differ from theirs; the shape does not).

`config.yaml:172` states the policy: *"one never-reused slot per window, both questions about a
window on that window's slot"* — so **two** calls per slot is by design and more than two is the
mitigation failing.

| arm | leaf calls | over-reused (>2/slot) | max on one slot | no `slot_id` recorded |
|---|---:|---:|---:|---:|
| `rlm` | 81 | **0.0%** | 1 | 4 (4.7%) |
| `b2` | 17,595 | **13.2%** *(peer: 13.4%)* | 4 | 0.6% |
| `rlm-restricted` | 3,157 | **81.9%** *(peer: 69.2%)* | 13 *(peer: 17)* | 24.8% *(peer: 34.5%)* |
| `b1` / `b3` | 96 / 95 | 0.0% *within* an episode | 1 | 0.0% |

**And the single-shot arms are the most exposed of all, on an axis the within-episode grouping
hides.** B1 makes one call per episode, so per-episode reuse is 1 by construction. Across episodes:

| arm | calls | distinct slots | distinct rotations | episodes |
|---|---:|---:|---:|---:|
| **b1** | 96 | **1** (slot 0) | **0** | 96 |
| **b3** | 95 | **1** (slot 1) | **0** | 95 |

**Every B1 call after the first sat on a slot that had already served up to 95 prior documents, on
one persistent process, with zero rotations.** That is the most extreme shared-prefix reuse in the
benchmark — more extreme than `rlm-restricted`'s worst within-episode slot — and it is precisely the
cell measured at **0/4 on ROCm and 4/4 on Vulkan** in §4.6.2.

**§8's predicted bias channel is inverted.** v0.2.6 recorded the risk as *"two-exposed (RLM, B2) vs
two-spared (B1, B3)"* and corrected the B1/B3 relaunch to `--parallel 2` so the baselines would stop
sharing slot 0. The measured exposure is the opposite: **`rlm` — the arm that won S4 — is the least
exposed arm in the benchmark**, and B1/B3 are the most.

**Stated at the strength the evidence supports, and no further.** `rlm`'s 0% is trivially achieved:
it made 81 leaf calls in its entire history, so an arm that barely delegates cannot be contaminated
through delegation. That is a fact about the arm's behaviour, not a clean bill of health. And this
is **exposure, not outcome** — no `rlm-restricted` or B1 answer has been shown corrupted, and per
§4.6.2 the detector cannot show it. What it establishes is that the S4 comparison ran a
near-zero-exposure arm against arms carrying 13%, 82% and single-slot-for-96-episodes exposure, on a
backend now measured to corrupt under exactly that condition.

**A second hole, instrumentation rather than serving:** 24.8–34.5% of `rlm-restricted`'s calls carry
**no `slot_id` at all**, and the detector's `None` verdicts are exactly those calls. For a third of
that arm there is neither a slot record nor a leak verdict. `b2`'s equivalent hole is 0.6%.

*Method note from the peer, kept because it is the same class of error this document is about: their
first pass reported 87.4% and a max of 155 calls on one slot, an artifact of grouping without
excluding `NULL slot_id` — the 4,268 slotless calls collapsed into one fake occupancy and the 155
was `slot=None`. Corrected before it was sent.*

**Not switched, each for its own reason:**

- **`size_tokens: 640`** — the window is no longer capped by the backend, but moving it is a §8
  comparability event and the 2.05× above is measured, not the geometry's own re-derivation.
- **`dispatch_concurrency: 1`** — Vulkan is 32/32 at c=8, but that was the reproducer's client, not
  `rlm.dispatcher`. The pin comes off after the scaffold's own dispatch path is measured.
- **`slot_policy: never_reuse`** — Vulkan is 0/54 on the shared arm, so reuse may become legal and
  the rotation machinery may become unnecessary. That is a larger change than a flag: it is a typed
  `Literal`, it removes the mitigation for a defect whose upper bound is still 2.8%, and it would
  retire the machinery that 15 of 30 `rlm-restricted` episodes died in.

**Live smoke on the shipped config**, launched through the project's own `launch_argv`: 7 seeds ×
{literal, paraphrase, absent} = **21/21 correct**, 1.07 s median wall, 0 leaks, 0 slot mismatches,
0 errors.

## 5. What this reattributes

| finding | recorded as | actually |
|---|---|---|
| §7 #2's distance cliff, `[989, 1003]` | a property of the leaf, forced by hybrid attention | the ROCm backend's reach limit. The measurement stands; the attribution does not. |
| v0.2.9 **instruction decay** — 30/30 false positives at a 1,024 window vs 0/21 at 640 | instructions decay with distance from generation, "the same horizon measured twice" | the same defect. On Vulkan at 2,048-token chunks the leaf refuses 47/48 — **there is no instruction decay on Vulkan.** The prefix was never disobeyed; on ROCm it sat in the corrupted region. |
| R5's **95% false-positive rate** | an architecture-level error model — "a leaf answer is no evidence the fact was present" | measured at windows of 1,024 and up, where the defect bites. At the shipped 640 window ROCm refuses 7/7 here, matching the project's own 0/45. **The headline number is an artifact of the window sizes it was measured at, not of every leaf call.** |
| the 640/480 geometry | forced by the measured leaf horizon | forced by a backend defect nobody had identified. It happens to be the largest shipped window that keeps every needle inside the ROCm horizon. It would also have been safe on Vulkan, at a lower cost. |
| gemma-4 "reproducing the threshold across two model families" | independent corroboration | a model whose SWA window is exactly 1,024, measured on the backend whose reach limit is also ~1,000. Two different causes, one number. |
| **R14**, concurrent-decode collapse (`upstream/ISSUE-concurrent-decode.md`) | a separate defect | the same backend, the same degenerate output, HIP broken and Vulkan clean. **One defect with two dials — concurrency and distance.** |

### 5.1 The project had this in hand twice and named it "the model"

`upstream/README.md`, on the first draft of the concurrency reproducer:

> scored 0/32 **serially**, because it placed the answer ~1,076 tokens from the question, past this
> model's effective retrieval reach — nothing to do with concurrency

That is this defect, observed at concurrency 1, at ~1,076 tokens, written down, and attributed to
the model. And `ISSUE-concurrent-decode.md` already reported HIP degenerate where Vulkan is clean.
Both times the backend was in frame and the model took the blame.

The deeper reason it survived: `ARCHITECTURE.md` carried, from v0.3.5, that the horizon is *"a
property of the prompt rather than of the serving path"*. That experiment excluded **slot state and
prefix reuse** — two things — and the sentence generalised to the whole serving path. It is the
sentence that stopped anyone looking at the backend, and it was already flagged as an overreach in
this morning's audit (D5) before anyone knew how badly.

---

## 6. Why the literature never had an analogue

`2026-08-23-distance-cliff-audit.md` §4 established that a near-100%-to-0% transition at ~1,000
tokens has no published counterpart: NoLiMa puts models at 94–98% of base at 1K and 2K, RULER's grid
bottoms out at 4K, and the nearest architectural relative — Qwen3-Next at the identical 3:1
Gated-DeltaNet layout — scores RULER 98.5 at 4K.

That gap was the strongest signal available that something was wrong, and it was read as novelty
instead. **A finding with no analogue anywhere is more likely to be an instrument fault than a
discovery**, and this project had the instrument in two backends the whole time.

---

## 7. What is owed

1. **An upstream issue.** Not a third report — an extension of `ISSUE-concurrent-decode.md`, since
   it is the same backend producing the same degenerate output on a second dial. Reproducer needs no
   fixtures and no model of ours: one ~1,200-token counting prompt, two backend directories, a
   positive control at 30 tokens. I searched: gfx1151 + ROCm has known issues for performance, VMM,
   loading hangs and crashes, but **no report of silent wrong output**. Discussion #20856's
   known-good stack is ROCm 7.2.0 with `GGML_HIP_NO_VMM=ON`, and its author moved to Vulkan anyway;
   this project's leaf is a ggml-org win-rocm zip with ROCm 7.14 wheel DLLs grafted in, which nobody
   upstream runs.
2. **A backend decision, which is the owner's.** S0 put the leaf on ROCm for +13.6% prefill. That
   trade now reads as buying 13.6% of prefill with a correctness ceiling at ~1,000 tokens. Moving
   the leaf to Vulkan overturns a gate decision, and under §8's comparability rule re-running the
   baselines on a different backend is a **benchmark event**, not a flag change. Neither is a
   documentation edit and neither is made here.
3. **A geometry question worth reopening on the other side of that decision.** 640/480 costs +25%
   in served tokens and ~+33% in wall against 1,024/768, and `max_subcalls` rose 522 → 926 to pay
   for it. If the leaf moves to Vulkan, the constraint that forced it is gone.
4. **Re-measurement, scoped honestly.** Every leaf-side number taken at a window of 1,024 or more is
   about a backend, not a model: the instruction-decay A/B, R5's false-positive rate, the layout
   arms, the arch ladder. The ones taken at 640 are unaffected — measured, not assumed, in §4.

---

## 8. Limits

- One box, one GPU (gfx1151), one ROCm graft, one llama.cpp commit. "ROCm is broken" here means
  *this* ROCm build on *this* device; a different ROCm version or GPU is untested.
- The Vulkan arms are clean everywhere measured, which is 600–1,500 tokens of distance at 2,048-token
  chunks plus the 640 geometry. Vulkan is not certified beyond that, and "no boundary found" is not
  "no boundary".
- The mechanism inside ROCm is **not identified**. Not hipBLASLt, not KV precision, not the FA
  kernel, not CPU fallback — but which kernel, and why the onset is near a thousand positions, is
  unknown.
- The 750-token dip recorded in the audit's E1 (§7.2 there) is a ROCm phenomenon too — Vulkan is
  12/12 at 750 — and remains unexplained. It is now a detail of the defect rather than a property of
  the model.
- `paraphrase` is imperfect on Vulkan too (34/48 at long distances, 6/7 at the shipped geometry).
  Some of the leaf's difficulty with assembled answers is real and is not the backend.
