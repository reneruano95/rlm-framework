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
