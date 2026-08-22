# The arch ladder — v1 was contaminated by its own design; v2 measures the model

**Date:** 2026-08-14 · **Script:** `milestones/s2/arch_ladder.py` · **Raw:** `milestones/s2/results/arch_ladder_*.jsonl`
**Supersedes** the 2026-08-14 "INCONCLUSIVE" version of this file, whose numbers
must not be quoted. Prompted by the question "so either gemma or qwen models
works, maybe we are doing it wrong?" — the answer was yes, and this is the fix.

---

## 1. What was wrong with v1

v1 reported that both models "collapse at 2,048" and called the comparison
inconclusive because the control was SWA-interleaved. Both statements were
premature: **the failures it measured were not model behaviour at all.**

Every one of its ABSENT-question failures returned the *correct custody key for
the entity it was asked about*, retrieved from **a different fixture's
document**:

| asked about (absent from its own chunk) | answered | that key's real owner |
|---|---|---|
| Marnwickstead Trust | `48e81295-…` | Marnwickstead Trust, in fixture 1024/t3 |
| Quinfennsted Trust | `d9f804c1-…` | Quinfennsted Trust, in fixture 1024/t1 |
| Selkdaleridge Trust | `7e41c11e-…` | Selkdaleridge Trust, in fixture 1024/t3 |

Established offline by `milestones/s2/replicate_fixtures.py`, which re-derives each
fixture's bindings from its seed without touching a server. Two independent
signatures make chance impossible: the wrong answers are **byte-identical across
two different model families**, and each is entity-*correct* for a 128-bit
random identifier the model could not guess.

Tracing the qwen run's slots against its own log (`milestones/s2/audit_archladder_slots.py`)
matched every leak to the prior occupant of **its own slot** — including the one
that leaked from a different donor because it ran on a different slot:

| leak row | slot | donor fixture | donor's slot |
|---|:--:|---|:--:|
| 2048/t0 ABSENT | 0 | 1024/t3 | 0 |
| 2048/t1 ABSENT | **1** | 1024/t1 | **1** |
| 2048/t2 ABSENT | 0 | 1024/t3 | 0 |

**The probe built R13's documented condition.** `R13-mitigations.md` §4.4
measures document *growth* down a reused slot as the leak condition ("six windows
ascending 159 → 1,934 tokens: 4/18"). An ascending 640 → 1,024 → 2,048 ladder on
server-chosen slots is that condition exactly — the gemma arm funnelled all 24
documents onto slot 3, the qwen arm 18 of 24. `ARCH-LADDER.md` v1 was also the
only report in `milestones/s2/` that recorded no launch line and no slot policy.

**Naming the condition is not naming the mechanism**, and §4 shows both obvious
mechanisms are excluded by measurement. The prompt cache is not the culprit: the
client sent `cache_prompt: false`, and the log join shows *zero* tokens skipped
on all 24 requests. Whatever crosses between requests is not tokens in a KV
cache, and it survives a full prompt re-evaluation.

## 2. The corrected instrument

Four changes, each closing a path by which the harness could produce the result:

1. **`--policy isolated` (new default): one question per never-used slot.**
   Nothing whatsoever precedes the request being measured. `virgin` (one slot per
   *fixture*) is not sufficient — see §4.
2. **Greedy decoding (`--temp 0.0`, was 0.3).** At `0.3`, two server geometries
   disagreed 4/4 vs 1/4 on identical prompts and seeds; at n=4 sampling noise
   swamps the effect.
3. **A leak oracle in the runner.** Every wrong answer is classified against the
   run's full binding registry: `MISATTRIBUTED` (key from its own chunk),
   `LEAKED` (key from another fixture), `FABRICATED` (matches nothing).
   v1 could not tell these apart, which is why it misread a leak as a refusal
   failure.
4. **Slot guards, up front and per request.** Both are needed — see §3.

## 3. The `id_slot` hazard — already in the spec, and this probe ignored it

**Not a new finding.** §5 of ARCHITECTURE.md (v0.2.6, R13) already states it and
already requires the fix: *"C4 must ASSERT the `slot_id` the server returns
equals the one requested — an out-of-range `id_slot` is silently reassigned with
HTTP 200 (measured: asked 200, got 72)."* The probe simply did not do what the
spec obliges the scaffold to do, and paid for it exactly as predicted.

Recorded here because it is now measured a second time, in a second way, and
because it shows the failure is silent enough to survive a deliberate control.
An `isolated` run asking for 25 slots on a 16-slot server had requests 16–24
served by slots 0–8, which had already held the 640-token documents — and leaked
2/24 *inside the arm built to prevent leaking*:

```
 #15 1024 ABSENT  req=16  served=0   REFUSED   <-- silently reassigned
 #19 2048 ABSENT  req=20  served=4   LEAKED    <-- silently reassigned
 #23 2048 ABSENT  req=24  served=8   LEAKED    <-- silently reassigned
```

The lesson is about *probes*, not about C4: every ad-hoc script in `milestones/s2/` that
pins slots is subject to the same rule as the shipped dispatcher, and none of
them inherited it. `arch_ladder.py` now checks the pool size against `/slots`
before the first call and asserts the returned `id_slot` on every call, refusing
to run otherwise.

## 4. Result — one question, one never-used slot, greedy

Retrieval (LITERAL, needle at the end of the chunk) is **4/4 at every size in
both models**. The refusal criterion is what moves:

| chunk tokens | Qwen3.6-35B-A3B (hybrid) ABSENT refused | gemma-4-12B-it (SWA-interleaved) ABSENT refused |
|---:|:---:|:---:|
| 640 | **4/4** | **4/4** |
| 1,024 | 0/4 | 0/4 |
| 2,048 | 0/4 | 0/4 |

`arch_ladder_np32_isolated.jsonl`, `arch_ladder_gemma-np32_isolated.jsonl`.
Cross-checked on a second geometry: the in-range subset of the `-np 16` isolated
run gives 0/3 refusals at 1,024, agreeing with `-np 32`.

Both models fail the same way and at the same place. Qwen hands over a
neighbouring entity's key from its own chunk; gemma does that at 2,048 and at
1,024 confabulates a decoding procedure instead ("we must decode the grid of
words provided… a 'Word Search' style cipher"), which is a failure of the same
criterion by a different route.

### What actually predicts a leak — and it is not slot policy alone

All 15 greedy runs, by server geometry and slot policy:

| server | policy | leaked | refused @640 | @1,024 |
|---|---|:--:|:--:|:--:|
| `-np 4 -c 65536` | auto, shared (×3 runs) | **2/24** | 4/4 | 4/4 |
| `-np 16 -c 65536` | auto, shared (×3 runs) | **2/24** | 4/4 | 4/4 |
| `-np 16 -c 65536` | **virgin** | 0/24 | 4/4 | 0/4 |
| `-np 32 -c 131072` | isolated, virgin, shared, auto | 0/24 | 4/4 | 0/4 |

Two things follow, and the second corrects this file's first draft.

**A never-reused slot does stop the leak where leaking happens** — on `-np 16`,
virgin is 0/24 against 2/24 for the same prompts on a reused slot.

**But nothing leaks on `-np 32 -c 131072` under ANY policy, including a slot
pinned to hold all twelve documents in ascending order.** The definitive isolated
runs in §4 were taken there, so their cleanliness is over-determined: geometry
alone would have sufficed. It is therefore **not established that the slot policy
is why they are clean**, and the earlier draft of this file claimed otherwise.

**The mechanism remains unidentified.** Both mundane candidates are excluded by
measurement: prefix-cache reuse (all 72 arch-qwen tasks and all 24 arch-gemma
tasks carried **zero** tokens across requests; `prompt eval` equals the full
rendered prompt every time) and the host prompt cache (all four arch-ladder
server processes log **zero** "making room for prompt cache entry" lines). What
crosses between requests is not tokens in a KV cache, and `-c 65536` versus
`-c 131072` at identical 4,096 tokens per slot separates leaking from non-leaking
runs for reasons this probe cannot explain. That is the open question, and it is
worth an upstream report — the entity-correct, byte-identical-across-models
evidence in §1 is not in doubt, only its cause.

### Contamination is not conservative

Leakage and refusal are perfectly anticorrelated across all 15 runs: **every arm
that leaked refused 4/4 at 1,024; every arm that did not leak refused 0/4.** A
leaking harness made the leaf look *better behaved* than it is, then leaked
outright one rung later. Two consequences worth carrying forward:

* A contaminated measurement is not conservatively wrong. It can mask a failure
  as easily as manufacture one, so "the leak would only have hurt us" is not
  available as a defence of any past number.
* ~~`virgin` (one slot per *fixture*) is not clean enough: the model answered the
  ABSENT question with the key it had just emitted for the LITERAL question on
  that slot in 6 of 8 cases, so the previous **question** carries.~~
  **WITHDRAWN — measured and does not reproduce (`milestones/s2/CARRYOVER.md`).** That
  reading was a confound of this probe's own making: its priming question asked
  about `present[0]`, which is also the leaf's preferred misattribution target
  (baseline picks it 4 times in 5), so "repeated the previous answer" and "chose
  its usual wrong answer" were the same string. Priming on `present[2]` instead,
  at production geometry with the warm path genuinely engaged (850 tokens
  reused): **40/40 byte-identical answers across solo / warm-reuse / reuse-
  without-prefix at the shipped 640 window**, and an identical failure rate in
  all three arms at 1,024. R13-mitigations §4.3 is right as written, and §5's
  slot discipline stands. `isolated` remains the correct policy for *measuring*
  the model, but not because same-document reuse corrupts anything.

## 5. What this settles, and what it does not

**Settles:** the v1 file's numbers and its "collapse at 2,048" are withdrawn. The
true threshold sits between 640 and 1,024, in both model families, measured with
every contamination path closed.

**Corroborates rather than undermines the spec.** The parallel audit of the
load-bearing runs (`milestones/s2/audit/`) found them clean under a detector validated on a
known-leaking positive control. Of `distance.jsonl`'s 153 wrong answers, **76
quote an identifier from the exact chunk that request sent, 9 quote a mangled
string present in no fixture, 68 carry no identifier at all, and 0 come from
another fixture** — resolved per record by its own `chunk_sha256` against all 14
fixture directories, a wider universe than the runner's own per-phase detector.
Of its 92 false positives, 73 quote that cell's own planted key and 0 quote
another cell's. `refusal-ab` is 356 identifiers with exactly one non-substring,
and that one exists in no fixture (fabrication, not leakage). The positive
control (`sweep-run1-shared-server.jsonl`, pre-R13) yields 17 foreign identifiers
in 54 calls, correctly naming their donor fixture, so the detector fires when
leakage is present.

Two caveats found by the adversarial pass, neither of which moves the verdict:
**85 of 91 slots** held exactly one document, not 91 — six slots (90–95) each
served two documents inside a 12-record cache-instrumentation phase whose records
omit `chunk_sha256`, which is why the first scan missed them. And the run
launched **without** `--cache-ram 0` (added to `config.yaml` three hours later),
so the host prompt cache was live. What holds the verdict up is physics rather
than bookkeeping: all **85 cold calls report `cache_n = 0`** and prefill at
954–1,005 tok/s against this build's independently measured 961 tok/s cold rate,
with none faster than 1,600 — every cold call genuinely processed its whole
prompt. The slot-index confound is killed by existing data too: `refusal-ab-640`
ran 640-token cells at slots 60–87 for **0/21** false positives while
`refusal-ab` ran 1,024-token cells at slots 12–54 for **30/30**, so high-slot 640
is clean and low-slot 1,024 is saturated — the confound points the wrong way. §4's cliff, the
false-positive rate and the instruction-decay model therefore stand — and this
probe, a different instrument on a different fixture generator, independently
reproduces the same ~1,000-token threshold.

**Validates the shipped geometry.** `size_tokens: 640` refuses 4/4 in both
models; 1,024 refuses 0/4 in both. The window sits on the correct side of the
threshold by measurement, not by inference.

**Does not settle the architecture question, and now cannot be settled locally.**
Every GGUF in the library was read (`milestones/s2/gguf_arch.py`, metadata only, no model
load) and **not one is uniform global attention**:

| model | `general.architecture` | mechanism | capacity |
|---|---|---|---|
| Qwen3.6-35B-A3B (leaf), Qwen3.6-27B (root), **Qwen3.8-27B** | `qwen35` | recurrent (`ssm.state_size 128`, `inner_size 6144`) | fixed, context-independent |
| gemma-4-12B / 31B / 26B-A4B / E2B / E4B | `gemma4` | sliding window | **1,024** |
| Muse-Glimmer-30B | `muse-glimmer` | sliding window | **2,048** |
| Nemotron-3-Nano-Omni-30B | `nemotron_h_moe` | recurrent (`ssm.state_size 128`) | fixed |

This **discharges R7 checklist item 2**: `Qwen3.8-27B` is hybrid, the same family
and the same 3:1 ratio as the current leaf — its model card gives the layer
pattern as `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`, and
the local file carries the matching `qwen35.ssm.*` keys. **It is therefore not a
control, and swapping to it for S5 would not change the attention story.**

*Noted for S5 sizing, arithmetic not measurement (same caveat §4 applies to the
current leaf's 62.8 MiB/slot):* Qwen3.8-27B's `ssm.inner_size` is **6,144**
against the current leaf's 4,096, over ~48 recurrent layers — about **144
MiB/slot**, so at `-np 128` roughly **18 GiB** of recurrent state against today's
8.04 GiB. That does not fit the current carve at `-np 128` and must be priced
before any S5 swap.

**A better test than the one originally planned, and why it still failed.** Since
the window sizes differ across models, the horizon can be probed by *dose
response* rather than by finding a full-attention control: gemma's window is
**exactly 1,024**, and gemma's refusal collapses at **exactly 1,024** chunk
tokens. Muse-Glimmer's 2,048 window predicts a clean 1,024 and a failure at
2,048. Run (`arch_ladder_muse-swa2048_isolated.jsonl`), the arm is **invalid**:
Muse-Glimmer emits `to=self<|message|>` and then echoes the document back, i.e.
it wants harmony-style channel control tokens that this ChatML path does not
produce, and it fails even LITERAL recall at 640 (2/4). That is a prompt-format
mismatch, not a measurement. Fixing it would characterise a creative-writing
merge rather than anything deployable, so it was not pursued.

`fallback_leaf` therefore stays neither confirmed nor killed, and closing it now
requires **downloading** a uniform-attention instruct model (Llama-3.x-8B,
Qwen2.5-7B and Mistral-7B all qualify) rather than anything on disk. What has changed is that the question is now less
urgent: the two families fail identically at the same threshold, which is weak
evidence that the effect is general instruction-following decay rather than
anything specific to Gated DeltaNet — and the geometric mitigation is measured to
work regardless.

**Next, in order:** ~~(1) measure the same-slot previous-question carryover~~
**done, `milestones/s2/CARRYOVER.md` — no effect at the shipped window;** (2) assert
returned `id_slot` in C4; (3) verify `Qwen3.8-27B`'s attention layout — still R7
checklist item 2 and the S5 swap target — and if it is uniform full-attention,
run this ladder against it as the clean discriminating arm; (4) the carry-over
probe primes with a **single** question, so a window asked five or ten questions
in sequence is still uncovered, and that accumulation case is what production
hits at high `max_subcalls`.
