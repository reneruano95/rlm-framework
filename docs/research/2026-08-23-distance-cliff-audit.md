# The ~1,000-token distance cliff — audit, amendment, and the measurement that is actually owed

**Date:** 2026-08-23 · **Status:** audit complete (§§0–5, from records already on disk, no new calls);
**E1 and E2 MEASURED** (§§7–8, 638 fresh leaf calls the same day).
**Trigger:** an outside reading of `ARCHITECTURE.md` §7 #2 — *"the ~1,000-token cliff is more aggressive
than the published literature, and it is your finding, not an established fact."*
**Verdict:** the objection was correct, the repo's own source file said it first and louder than the spec
ever did, the audit found a **better bracket sitting unextracted in data taken eight days ago** — and the
sweep that could have removed the word "cliff" instead sharpened it, while falsifying a different sentence
the spec had been leaning on.

---

## 0. The short answer

Three things are true at once, and the spec was carrying only the first.

1. **The effect is real and the discipline behind it is good.** Fresh process per cell, 0/117 foreign
   identifiers under a detector validated on a positive control that fires 17/54, counterexamples that
   decouple chunk size from distance. Nothing here is contamination.
2. **The number the spec shipped was its weakest artifact.** `[967 pass, 1022 fail]` rests on **one n=3
   cell per side** — Fisher p = 0.10 taken alone — with a pass-side count (`6/6`) that pooled two needles
   679 tokens apart, at temperature 0.3 rather than greedy. `milestones/s2/RESULTS.md` states the limit
   plainly: *"the location of the cliff is worth one more sweep before anything is derived from its exact
   value."* That sentence never reached the working tree.
3. **A stronger bracket already exists in `distance.jsonl` and nobody scored it.** Re-scored by each
   cell's own needle distance: **[989 pass, 1003 fail], n = 12 per side, Fisher two-sided p = 7.4e-07**,
   four independent fixture families, chunk size held constant. It supersedes the shipped pair.

And one thing the audit removed from the table: **a sliding window cannot be the mechanism.** The leaf's
ten global attention layers see token 1,022 exactly as they see token 967. Whatever produces the boundary,
it is not an architectural window — which is the only mechanism the published record offers for a hard
threshold at a round number.

---

## 1. The amended bracket, re-derived

**Source:** `4e75b53:milestones/s2/results/distance.jsonl`, arm `A-shipped`, phase `grid`,
`question_type == literal` — 72 trials over 24 cells. Distance is each cell's own
`(1 − token_depth) × measured_tokens` from the four `fixtures-distance-s1{1,2,3,4}/manifest.json` files.
`DISTANCE.md` reported these rows pooled per size and read the 2,048 column as a 50% collapse. Disaggregated
by distance it is not a collapse. It is a step with **no exceptions in 72 trials**:

| chunk size | distance | seed | density | result |
|---:|---:|---:|---|---|
| 640 | 309 – 357 (8 cells) | 11–14 | both | **24/24** |
| 1,024 | 453 – 566 (8 cells) | 11–14 | both | **24/24** |
| 2,048 | 958 | 12 | matched | **3/3** |
| 2,048 | 983 | 14 | matched | **3/3** |
| 2,048 | 989 | 13 | natural | **3/3** |
| 2,048 | 989 | 14 | natural | **3/3** |
| 2,048 | 1,003 | 12 | natural | **0/3** |
| 2,048 | 1,025 | 11 | matched | **0/3** |
| 2,048 | 1,035 | 11 | natural | **0/3** |
| 2,048 | 1,091 | 13 | matched | **0/3** |

**Bracket [989, 1003] — 14 tokens wide.** At the boundary, 12/12 vs 0/12, Fisher two-sided **p = 7.40e-07**.
Whole grid, 60/60 below vs 0/12 above, **p = 6.5e-14**.

Why this is better controlled than `[967, 1022]` on every axis:

- **Chunk size is held at 2,048 across the boundary**, so total prefill is constant and size cannot be
  the variable. The old bracket compared a 32,768-token cell against a 1,024-token one.
- **Four independent fixture families** (seeds 11–14, four separate generator runs), not one.
- **Both density conditions appear on both sides** — `matched` at 958/983 (pass) and 1,025/1,091 (fail),
  `natural` at 989/989 (pass) and 1,003/1,035 (fail). Distractor density is ruled out as the carrier.
- **Seeds cross the boundary in both directions**: s12 passes at 958 and fails at 1,003; s13 passes at
  989 and fails at 1,091. A per-fixture difficulty confound would not do that.
- **Slot indices 36–43 interleaved**, one never-reused slot per call, under the R13 policy.

The 11 wrong answers on the fail side are **confabulations, not refusals** (1 malformed) — the same error
shape R5 records, and the reason "a leaf answer is no evidence the fact was present" is an architecture
statement rather than a chunking one.

### 1.1 Three limits the amendment does not remove

1. **The distance convention is undocumented and larger than the bracket.** Distance is measured from the
   needle's **first** token; the literal needle is ~50 tokens wide. The same data reads ~940 if measured
   from the needle's last token. The convention moves the number by more than the bracket's width.
2. **This is the LITERAL-identifier horizon, not one horizon per question type.** The paraphrase question
   scores 0/24 at 2,048 across distances 884–1,020 — it fails *inside* the bracket — while the sweep's
   generator scores 3/3 at 951. Two question types, two thresholds.
3. **Nothing between 640 and 1,024 has ever been sampled at fine grain**, in any instrument in the archive.
   Every "cliff" claim is a claim about two sampled points with an unsampled gap between them. The
   discontinuity-versus-steep-sigmoid question is **open**, and E1 below is what closes it.

---

## 2. What was wrong in the record, and is now fixed

| # | Claim as it stood | Correction | Sites amended |
|---|---|---|---|
| D1 | *"reproduced across two model families"* welded to the fact-cliff bracket | The two-family arm (`ARCH-LADDER.md`) measured the **refusal criterion only**; its literal recall is 4/4 at every size in both models because that probe puts the needle at the end of the chunk. It carries **zero** evidence about the fact-distance cliff. And gemma-4's SWA window is exactly 1,024, so it is a **positive control for the probe** — a model with a known reach boundary, measured at that boundary — not an independent replication. The primary itself grades the inference *"weak evidence"*. | `gate0-soak.md:25`, `avo-arc-agi-3-dossier.md:179`, `long-horizon-agent-design.md:100,177`, `rlm-paper-fidelity…:130` |
| D2 | *"a 32,768-token chunk whose needle sits 967 tokens from the end scores 6/6"* | **3/3.** The "6/6" pooled that cell's literal needle at 967 with its own paraphrase needle at **288** (`fixtures/manifest.json`: `token_depth` 0.9705 and 0.9912 on 32,765 tokens). Taken alone the old bracket is 3/3 vs 0/3, p = 0.10. | `ARCHITECTURE.md:279`, `chunker.py`, `test_chunker.py` |
| D3 | `[967 pass, 1022 fail]` shipped as the location of record, with its source's own limit absent from the tree | Superseded by `[989, 1003]`; the limits now travel with the number. | `ARCHITECTURE.md:126,279`, `config.yaml`, `chunker.py`, `test_chunker.py`, `test_config.py`, all four docs |
| D4 | *"a verified full-attention control"* (gemma-4-12B-it) | **Non-recurrent control.** gemma-4 is SWA-interleaved (`gemma4.attention.sliding_window = 1024`, 5:1 local:global). The R13 falsification needs only "no state that cannot be rewound", which holds — but the label collided with the *global-attention* sense used in `CHANGELOG.md` v0.3.2/v0.3.6, and three of the five sites were in shipped code. | `ARCHITECTURE.md:468`, `config.py`, `dispatcher.py`, `test_dispatcher.py` |
| D5 | *"characterized as a property of the prompt rather than the serving path"* | v0.3.5 excluded **slot state and prefix reuse**, on 20 trials with zero failures — "no detectable effect", ~15% upper bound by the rule of three, leaf only, single priming question. The rest of the serving path was never tested: `-ctk/-ctv q8_0` and `-fa on` are pinned on the only 10 of 40 layers that carry long-range KV. | `avo-arc-agi-3-dossier.md:179` |
| D6 | *"false-positive rate … flat across every size"* | Flat across **the sizes the sweep measured**, 1K–32K. `REFUSAL-AB.md` says it: *"the cliff is below 1,024, where the sweep never looked."* The correction had been sitting inside a block marked "(v0.2.8, superseded)". | `ARCHITECTURE.md:299` |
| D7 | `Fisher p ≈ 1e-21` | **2.9e-21** two-sided (point probability 1.5e-21), recomputed from `sweep.jsonl` with the repo's own Fisher implementation. No analysis script ever produced the "≈1e-21". | 5 sites |
| D8 | the sweep's hygiene, uncited | All 117 sweep records carry **temperature 0.3 / top_p 0.9** — not greedy, contrary to the rule adopted at v0.3.4. Now stated where the result is stated. | `ARCHITECTURE.md:279` |

Not amended, deliberately: the historical `CHANGELOG` entries. This project corrects the record in a new
dated entry rather than by editing old ones — see v0.3.17.

---

## 3. Mechanism: what the audit excluded, for free

Read offline from the GGUF header and the loader's own log (`4e75b53:milestones/s2/logs/gate-leaf.err`),
no model load, no server:

| evidence | value |
|---|---|
| `general.architecture` | `qwen35moe`, `block_count = 40`, `context_length = 262144` |
| keys containing `sliding` or `window` | **none** (54 KV pairs scanned) |
| `qwen35moe.full_attention_interval` | **4** — i.e. 10 of 40 layers are full attention (log line 65) |
| loader `n_swa` / `is_swa_any` | **0 / 0** (log lines 113–114) |
| `llama_kv_cache` | 3,400 MiB, **2,560 cells, 10 layers**, K/V `q8_0` (log line 199) |
| `llama_memory_recurrent` | 8,040 MiB, 128 cells, 40 layers → **62.8 MiB/slot**, context-independent (line 203) |
| rope | `freq_base 1e7`, scaling `linear`, `rope_finetuned unknown`, mrope sections `[11,11,10,0]` |

**A sliding window is excluded three independent ways.** The ten global layers hold every token of the
2,560-cell slot. Token 1,022 is as visible to them as token 967. This does not falsify the cliff — it
removes the one explanation the literature offers for a hard threshold at a round number, and it makes
the finding *harder* to explain, not easier.

It also kills the outside reading's own proposed mechanism ("Q4 + hybrid"): the corroborating gemma arm
ran at **Q8_0**, and gemma is SWA-interleaved, not hybrid-recurrent. Whatever this is, it is not those two
things acting together.

---

## 4. Where this sits against the published record

Checked against primary sources; the citations below were read, not recalled.

**The critic is right that the finding is unprecedented in shape.**

- *Lost in the Middle* (Liu et al., TACL 2024, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) is the
  **wrong comparator**, and not in the repo's favour: it varies needle **position at fixed length** and
  finds a U-shape — accuracy is *best* at the position farthest from the question — which contradicts a
  monotone distance-to-question law rather than supporting a gentler version of it.
- **NoLiMa** ([arXiv:2502.05167](https://arxiv.org/abs/2502.05167)) is the only benchmark found that
  resolves this scale. At 1K and 2K, models sit at **94–98% of base**, and collapse takes 8K–32K. It is
  the best comparator that exists and it disagrees with the repo.
- **RULER** ([arXiv:2404.06654](https://arxiv.org/abs/2404.06654)) reports *"almost linear degradation
  with input length on log scale within the max training context size"*, with abrupt drops **only** when
  extrapolating past the trained length. Its grid bottoms out at 4K, so a 1K effect is invisible to it
  by construction.
- **Same Task, More Tokens** ([arXiv:2402.14848](https://arxiv.org/abs/2402.14848)) does test 250–3,000
  tokens and finds real degradation starting in the high hundreds — 0.92 → 0.68 by 3,000. Gradual,
  monotonic, never zero. *Degradation beginning at these lengths is ordinary; a step to 0/39 is not.*

**And the defence that "a cliff is the expected signature for fixed-state models" does not apply to this
leaf.** Cliffs in that literature belong to models with **no attention at all** (Jelassi et al.,
[arXiv:2402.01032](https://arxiv.org/abs/2402.01032): copying *"drops to zero almost immediately"*) or
whose only attention is a window (Griffin, [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) §6.2:
perfect phone-book lookup *"up to a context length that matches its local attention window size of 1024"*
— zero global layers, and even there *"starts to degrade"*). Zoology/Based frame recall as a
**tradeoff in model dimension** (`d ≥ N`), not a threshold in distance. The leaf is neither of those
architectures, and the nearest published relatives at the identical 3:1 layout retrieve far past 1K:

| model | attention layers | published long-context recall |
|---|---|---|
| Qwen3-Next-80B-A3B (same 3:1 Gated-DeltaNet layout) | 12/48 | RULER 98.5 @4K · 98.7 @32K · 93.5 @256K |
| Jamba ([arXiv:2403.19887](https://arxiv.org/abs/2403.19887)) | 1:7 = 12.5% | *"excellent performance in the needle-in-a-haystack evaluation"* to 256K |
| MiniMax-01 ([arXiv:2501.08313](https://arxiv.org/abs/2501.08313)) | ~1 in 8 softmax | 100% NIAH at 4M |
| 72-model ratio study ([arXiv:2507.06457](https://arxiv.org/abs/2507.06457)) | 5 ratios swept | GatedDeltaNet at 3:1–6:1 *"achieves Transformer-level recall"* |

There is **no published long-context evaluation of `Qwen3.6-35B-A3B` itself** — its card carries no
RULER/NIAH table. So the architecture prior says this checkpoint should retrieve at 4K; nothing measures
whether it does.

**On quantization:** 4-bit does hurt long context ([arXiv:2505.20276](https://arxiv.org/abs/2505.20276),
EMNLP 2025: *"4-bit methods lead to substantial losses… drops of up to 59%"*) — but that paper's entire
regime is inputs **>64K**, it tests nothing at 1–2K, and it characterises nothing as a threshold. No
published work connects weight quantization to retrieval-head damage. That link is a hypothesis, not a
citation.

**Net:** the finding is not refuted by the literature; it is **unprecedented in shape and unexplained in
mechanism**, in an architecture whose relatives do not behave this way. That is a reason to measure
harder, not to soften the language and move on.

---

## 5. The defensible claim

*(Written before E1/E2 ran, and superseded by the sentence below it. Kept because the pre-registration
is only worth something if what it committed to is still legible afterwards.)*

> On this box, literal-identifier recall by the leaf `Qwen3.6-35B-A3B-UD-Q4_K_M`, served by llama.cpp
> b10375 under ROCm with `q8_0` K/V and flash-attention on, is all-or-nothing in the needle's distance to
> the question: **12/12 correct at 989 tokens and 0/12 at 1,003** (Fisher two-sided p = 7.4e-07, four
> independent fixture families, chunk size held at 2,048). A coarser 640-vs-1,024 step reproduces on a
> second, sliding-window model whose architectural window is itself 1,024. This is an on-box measurement
> of one serving stack — not a property of the model, of hybrid attention, or of long-context LLMs in
> general — and it has no published analogue at this scale.

**AMENDED AFTER E1/E2 (§§7–8), and this is the sentence to use:**

> On this box, the leaf `Qwen3.6-35B-A3B-UD-Q4_K_M` under llama.cpp b10375/ROCm has a **step boundary** in
> the needle's distance to the question: over 13 distances at a 25-token grid with 24 independent facts
> each, it scores **72/72 at 925–975 tokens and 0/144 at 1,025 and beyond** (Fisher two-sided
> p = 6.2e-14), with a single transition bin at 1,000. The boundary is **unmoved by `f16` K/V or by
> disabling flash attention**, and is **size-independent at the boundary** (0/12 at ~1,050 tokens at chunk
> sizes 2,048, 4,096 and 8,192 alike). Below the boundary, accuracy is **neither monotone in distance nor
> a function of distance alone** — the same fact at ~750 tokens scores 0/12 in a 2,048-token chunk and
> 11/12 in a 4,096-token one — and a distinct failure mode, the key retrieved and then mis-emitted,
> accounts for much of the sub-boundary loss. This remains an on-box measurement of one serving stack:
> llama.cpp b10375 is the common factor in every arm, no uniform-attention control has been run, and there
> is no published analogue at this scale.

---

## 6. Pre-registered measurement programme

Pre-registered **before** any of it runs, so the analysis cannot follow the data. Baselines: cold prefill
961 tok/s; median per-call wall 0.75 s @640, 1.15 s @1,024, 2.66–2.96 s @2,048; R13 gives 128 calls per
leaf process, then a restart. Every arm is **greedy** (`temperature 0`), one never-reused slot per call,
`id_slot` asserted, R13's foreign-identifier detector on every answer.

### E1 — fine distance sweep, size-crossed. **RUN FIRST.**

The measurement `RESULTS.md` said was owed and that has never been run: **no instrument anywhere in the
archive placed a cell between 640 and 1,024.**

- Chunk size fixed at **2,048** (the geometry that produced [989, 1003]), same generator, 8 fixture seeds
  × 3 trials = **n = 24 per point**.
- Distances: **600, 750, 875, 925, 950, 975, 1000, 1025, 1050, 1075, 1150, 1300, 1500** — 13 points, 312 calls.
- Size cross at d ≈ 950 and d ≈ 1,050, chunk sizes 1,024 / 4,096 / 8,192 — 144 calls.
- **Cost:** 456 calls ≈ 27 min compute, 4 process restarts, ~45–60 min wall.
- **Power:** n = 24 gives SE 0.10 at p = 0.5. A step predicts 24/24 then 0/24 across one 25-token bin. A
  logistic with 10–90% width of 50 tokens predicts ≥2 points strictly inside 0.15–0.85; width 100 predicts
  ≥4. Both separate from a step at binomial p < 0.001.
- **FALSIFIES the cliff if:** three or more consecutive points sit strictly inside 0.15–0.85 (a graded
  transition wider than ~75 tokens — "lost in the middle"-shaped); **or** accuracy at fixed distance
  differs by ≥0.3 across chunk sizes, which would put size back on the table and take distance off it.
- **Reports regardless:** the boundary under both distance conventions (first-token and last-token), and
  the paraphrase question scored separately — it does not share the literal threshold.

### E2 — serving-path 2×2, same fixtures. *(180 calls, ~35 min)*

`{q8_0 K/V, f16 K/V} × {-fa on, -fa off}` over 5 boundary distances at n = 12. (Quantized V requires FA,
so `-fa off` pairs with f16 only.) f16 KV costs ~6,400 MiB against 3,400 — +2.9 GiB on a 32.2 GiB leaf
residency; it fits. **Stated honestly:** q8_0 noise is position-*uniform*, so it cannot by itself
manufacture a distance threshold; the prediction is **displacement**, not disappearance.
**FALSIFIES "property of the model" if:** the boundary moves by more than one grid bin (25 tokens) in any
arm — which would make the cliff a serving-stack artifact and every downstream citation wrong as written.

### E3 — quant ladder. **On disk, no download.** *(120 calls, ~1 h)*

Q4_K_M → Q6_K → Q8_0 of the same weights. Use the **unsloth MTP** files, whose chat template is
byte-identical to the shipped leaf's (8,057 chars):
`D:\AI\models\unsloth\Qwen3.6-35B-A3B-MTP-GGUF\{Qwen3.6-35B-A3B-UD-Q6_K.gguf, Qwen3.6-35B-A3B-Q8_0.gguf}`.
**Do not use** the lmstudio-community Q6_K/Q8_0: same weights, **different chat template** (7,764 chars) —
that is precisely the mismatch that invalidated the Muse-Glimmer arm at v0.3.6. Confound to declare: both
MTP files report `block_count = 41` (the MTP head); assert `n_layer = 40` at the loader and launch without
`--spec-type` so the head is inert. Q8_0 residency ≈ 49.2 GiB against the 64 GiB carve — fits, but the
root cannot be co-resident. **F16 is out of scope**: ~70 GB, not on disk, does not fit the carve.
**FALSIFIES "Q4 causes the cliff" if:** all three quants land in the same 25-token bin — which the repo's
own data already predicts, since the corroborating gemma arm ran at Q8_0.

### E4 — backend replication on the same GGUF. *(60 calls, ~20 min)*

`tools/llamacpp-vulkan` is on disk beside `tools/llamacpp-rocm`. Same weights, same graph, **completely
different kernels**. The cheapest second-runtime test that exists here, and it needs no download.
**FALSIFIES a ROCm/gfx1151 flash-attention kernel fault if:** the boundary lands in the same bin.

### E5 — dose-response with a declared window, done properly. *(180 calls, ~30 min)*

`gemma-4-12B-it-Q8_0.gguf` is on disk with `gemma4.attention.sliding_window = 1024`. b10375 supports
`--override-kv KEY=TYPE:VALUE`, so run `gemma4.attention.sliding_window=int:512` and `=int:1536` — a
second and third dose **on the same weights, same template, same tokenizer**, which is exactly what the
invalid Muse-Glimmer arm was trying to buy. Verify the loader prints the overridden `n_swa` before
spending anything; if the key is not honoured the arm is invalid and costs five minutes to discover.
**FALSIFIES "gemma is an independent replication" if:** the boundary tracks the override — gemma then
becomes a window effect and stops being evidence about the leaf at all.

### E6 — the uniform-global-attention control, owed since v0.3.2. *(~1 h + download)*

Every GGUF in the library is hybrid, SWA or recurrent (v0.3.6). Closing this needs a download —
Llama-3.x-8B, Qwen2.5-7B or Mistral-7B at Q8_0, 5–16 GB. **The only test that separates "bounded-context
mechanism" from "property of this stack."** Ranked below E1–E5 only because each of those can kill the
finding for less.

### E7 — second runtime on the same weights. *(2–5 h + ~70 GB — LAST)*

vLLM is not supported on Windows; transformers under the Windows ROCm wheels needs the BF16 safetensors
(~70 GB), which do not fit the carve. CPU transformers on the 128 GB host is feasible at minutes per call.
**E4 buys most of the same information for 20 minutes and no download.** Run E7 only if E1–E6 agree and
the finding is going somewhere public.

### Order and stopping rule

**E1, then E2 the same afternoon over the identical fixtures.** Under two hours of box time answers both
*"what shape"* and *"is it the serving path"*. If E1 shows a graded transition, the word "cliff" comes out
of the spec and the geometry argument is re-derived from a curve instead of a threshold. If E2 moves the
boundary, every citation becomes a statement about a flag rather than about a model, and E3–E7 are run
before anything downstream is trusted.

---

## 7. E1 RESULT — the boundary is a step, and two things nobody expected

**Ran 2026-08-23, after §6 was written and committed.** 470 leaf calls, greedy,
one never-reused slot per cell, `id_slot` asserted on every call: **0 leaks, 0
slot mismatches, 0 errors**. Harness in the session scratchpad (`e1_fixtures.py`,
`e1_run.py`, `e1_analyse.py`), scored with `s2.run_sweep.classify` unchanged.

**One design change was forced before any call, and it is worth recording.**
`make_distance_fixtures.py` places the needle with `boundary_at_token_target`,
which snaps to a paragraph boundary — five different targets (925, 950, 975,
1000, 1025) all landed on the same achieved distance of 967. A generator
quantised at ~70 tokens cannot answer a 25-token question. E1 therefore builds
ONE base cell per (seed, size) and moves the distance by appending `k` tokens of
neutral filler to the END and trimming `k` from the HEAD: distance-to-question
rises by exactly `k`, total size is held, and every cell in a seed carries the
same needle, question and bindings. The distance axis is paired within fact,
which no grid in the archive was. Achieved distances land within ±2 of target.

### 7.1 The step is real — and sharper than the pre-registration asked for

24 independent facts per point, chunk size fixed at 2,048. Two curves: `strict`
is `classify`'s CORRECT; `located` also counts an answer sharing a ≥12-character
run with the true key (see §7.3 for why that column exists).

| distance | strict | located |
|---:|---:|---:|
| 600 | 14/24 | 23/24 |
| 750 | **0/24** | **8/24** |
| 875 | 11/24 | 20/24 |
| 925 | 24/24 | 24/24 |
| 950 | 23/24 | 24/24 |
| 975 | 24/24 | 24/24 |
| 1000 | 5/24 | 6/24 |
| 1025 | 0/24 | 0/24 |
| 1050 – 1500 (6 bins) | 0/144 | **0/144** |

**925–975 pooled: 72/72 located, 71/72 strict. 1,025 and above: 0/144.** The
transition occupies a single bin — 1,000, at 6/24 — so the 10–90% width is under
50 tokens. Fisher two-sided: 975 vs 1,025 **p = 6.2e-14**; 975 vs 1,000
p = 3.7e-08.

**The pre-registered falsification did not fire.** It required three or more
consecutive bins strictly inside 0.15–0.85; the boundary has one. **The cliff
survives the first measurement designed to break it,** and the location agrees
with the [989, 1003] re-derivation of §1 to within one bin.

### 7.2 The boundary is size-independent; everything BELOW it is not

Same 12 facts, the same distances built at three chunk sizes:

| distance | 2,048 | 4,096 | 8,192 |
|---:|---:|---:|---:|
| ~750 | **0/12** | 11/12 | 11/12 |
| ~950 | 12/12 | 11/12 | 6/12 *(located 12/12)* |
| ~1050 | **0/12** | **0/12** | **0/12** |

**At the boundary, distance rules absolutely** — 0/12 at 1,050 at every size,
0/144 above it in the main sweep. That is the claim §7 #2 has always made, now
measured at three sizes instead of inferred from two counterexamples.

**Below the boundary the claim is false as written.** The same fact, at the same
distance of ~750 tokens, scores 0/12 in a 2,048-token chunk and 11/12 in a
4,096-token one. E1's OTHER pre-registered falsification — accuracy at fixed
distance differing by ≥0.30 across sizes — **fires at 750 (spread 0.92) and at
950 (0.50), and does not fire at 1,050 (0.00)**. So:

> Retrieval below the horizon is not a function of distance alone. The HORIZON
> is; what happens inside it is not.

**The 750 dip is not an artefact of the shift mechanism, and three controls say
so.** Rebuilt off a lower base so the same distance carries a different tail
pad: pad 0 → 0/12 strict, pad ~150 → 0/12, pad ~300 → 1/12. Distractor
proximity is ruled out offline: the median token gap from the needle to the
nearest `ENT-` header is **190 tokens at every distance**, including the ones
that score 24/24. The dip tracks the (size, distance) pair, not the construction.
**Its mechanism is unidentified.** It is recorded, not explained.

### 7.3 A second failure mode: retrieved, then mangled

28 of 312 main-sweep answers carry a long verbatim run of the TRUE key and are
still wrong — a dropped prefix (`5692-4579-aa0e-de32ddc7e401` for
`1ee38678-5692-4579-aa0e-de32ddc7e401`), a repeated interior segment
(`…-980ab009d584-41cc-8328-980ab009d584-…`), a bare fragment. `classify` scores
them CONFABULATION, which is right — they are wrong answers — but they say the
key WAS reached and then mis-emitted, which is a different defect from "the
needle is out of reach". They cluster **below** the boundary (9 at 600, 8 at 750,
9 at 875, 1 at 950, 1 at 1000) and vanish above it, where nothing is retrieved
at all.

**This is R14's signature under strictly serial dispatch.** R14 is recorded as
concurrency-dependent — "degenerate text, not wrong answers", at two or more
calls in flight. E1 ran one call at a time, awaited, on a never-reused slot. The
same shape appears anyway. That does not refute R14's concurrency finding; it
adds a serial instance R14's account does not cover, and it means the
false-positive and confabulation rates recorded across S2 pool two mechanisms.

### 7.4 What E1 changes in the record

- **The word "cliff" survives**, and for the first time on evidence that could
  have removed it: 13 points, n=24, a 25-token grid through the transition.
- **"Retrieval does not degrade with chunk size" is now too strong.** It holds
  for the horizon and fails inside it. §7 #2 is amended to say which.
- **The shipped geometry is unaffected.** 640/480 puts every needle inside the
  horizon by a wide margin, and the sub-horizon dip is measured at 2,048 — a
  chunk size the scaffold does not ship.
- **Two new open items**: the 750-at-2,048 dip, and serial retrieved-then-mangled
  emission. Neither existed before E1 and neither is explained by it.

---

## 8. E2 RESULT — the serving path is excluded, for the two knobs that could plausibly have done it

D5 of §2 says the spec overreached in calling the horizon "a property of the
prompt rather than the serving path": v0.3.5 excluded slot state and prefix
reuse, and nothing had ever tested the two flags that sit directly on the only
long-range channel this model has — `-ctk/-ctv q8_0` and `-fa on`, both pinned
on the 10 of 40 layers that carry per-token KV. E2 tests them, over the same
fixtures, same 12 facts, same greedy sampling, one never-reused slot per cell.

`located` (CORRECT, or ≥12 verbatim characters of the true key), 12 facts per cell:

| arm | 750 | 925 | 950 | 975 | 1000 | 1025 | 1050 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q8_0` KV, `-fa on` — **shipped** | 4/12 | 12/12 | 12/12 | 12/12 | 4/12 | 0/12 | 0/12 |
| `f16` KV, `-fa on` | 4/12 | 12/12 | 12/12 | 12/12 | 6/12 | 0/12 | 0/12 |
| `f16` KV, `-fa off` | 4/12 | 12/12 | 12/12 | 12/12 | 5/12 | 0/12 | 0/12 |

**The boundary does not move by a single bin.** The pre-registered falsification
— a shift of more than one 25-token bin in any arm — **did not fire**. Perfect
below, dead above, in all three configurations; the transition bin at 1,000
wobbles 4/6/5 of 12, which is binomial noise at this n.

Two knobs are therefore excluded: **KV quantization is not the cause** (f16 K/V
doubles the cache and changes nothing) and **the flash-attention kernel is not
the cause** (turning it off changes nothing). The spec pre-registered a bf16-KV
fallback twice and never closed it; it is closed now, in the direction that says
the fallback would buy nothing here.

**And the 750 dip is identical in all three arms — 4/12 everywhere.** Whatever
it is, it is not KV precision and not the attention kernel either.

**What is NOT excluded**, and the honest statement of the residual: llama.cpp
b10375 itself is the common factor in every arm. E4 (the Vulkan build, same
weights, entirely different kernels, on disk, ~20 minutes) and E6 (a
uniform-global-attention model, one download) remain owed and are now the only
cheap ways left to separate "this model's weights" from "this runtime".

---

## 9. What this does not change

`size_tokens: 640` still ships, and E1 leaves it better justified than it was. Every geometry argument in
§7 #2 turns on which **side** of the boundary the window lands on: 685 is inside a boundary now measured
at 24 facts per point, and 1,069 is outside it in every arm. The sub-boundary dip does not touch the
shipped geometry either — it is measured at 2,048 tokens, a chunk size this scaffold does not ship, and at
640 the needle is never more than ~640 tokens from the question.

What changes is precision and scope. The "~47 tokens past the fail point" arithmetic is gone: no window
between 640 and 1,024 has ever been sampled, so the geometry is justified by which side it lands on and
never by a margin. And the claim is the LEAF's literal-identifier boundary on THIS stack — applied to the
root by same-family analogy, and to other question types not at all (the paraphrase question fails inside
the boundary).

**Where the evidence lives.** The E1/E2 harness (`e1_fixtures.py`, `e1_run.py`, `e1_analyse.py`, the
`s1`/`s2` modules recovered from `4e75b53` and an import shim for the C1–C6 regroup) and the 638 raw
records are in the session scratchpad, not in this repository — the same disposition the Gate 0 soak used,
and consistent with the 2026-08-22 decision to stop writing into an evidence archive. **They are therefore
session-local and will not survive it.** Every number in §§7–8 is reproducible from the recipe in §6 plus
the two design notes in §7 (the paragraph-snap defect and the append-and-trim fix); nothing here rests on
a file only this session can see.
