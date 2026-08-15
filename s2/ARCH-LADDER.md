# The arch ladder — v1 was contaminated by its own design; v2 measures the model

**Date:** 2026-08-14 · **Script:** `s2/arch_ladder.py` · **Raw:** `s2/results/arch_ladder_*.jsonl`
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

Established offline by `s2/replicate_fixtures.py`, which re-derives each
fixture's bindings from its seed without touching a server. Two independent
signatures make chance impossible: the wrong answers are **byte-identical across
two different model families**, and each is entity-*correct* for a 128-bit
random identifier the model could not guess.

Tracing the qwen run's slots against its own log (`s2/audit_archladder_slots.py`)
matched every leak to the prior occupant of **its own slot** — including the one
that leaked from a different donor because it ran on a different slot:

| leak row | slot | donor fixture | donor's slot |
|---|:--:|---|:--:|
| 2048/t0 ABSENT | 0 | 1024/t3 | 0 |
| 2048/t1 ABSENT | **1** | 1024/t1 | **1** |
| 2048/t2 ABSENT | 0 | 1024/t3 | 0 |

**This is R13, and the probe built its own trigger.** `R13-mitigations.md` §4.4
measures document *growth* down a reused slot as the leak condition ("six windows
ascending 159 → 1,934 tokens: 4/18"). An ascending 640 → 1,024 → 2,048 ladder on
server-chosen slots is that condition exactly. `ARCH-LADDER.md` v1 was also the
only report in `s2/` that recorded no launch line and no slot policy.

Note that the prompt cache was **not** the culprit: the client sent
`cache_prompt: false`, and the log join shows *zero* tokens skipped on all 24
requests. The residue survives a full prompt re-evaluation.

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

The lesson is about *probes*, not about C4: every ad-hoc script in `s2/` that
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

### Slot policy changes the answer, and contamination is not conservative

Same server, same moment, same prompt bytes, greedy — only slot history differs:

| server | policy | 640 | 1,024 | 2,048 | leaked |
|---|---|:--:|:--:|:--:|:--:|
| `-np 4 -c 65536` | auto / shared | 4/4 | 4/4 | 2/4 | 2 |
| `-np 16 -c 65536` | auto / shared | 4/4 | 4/4 | 2/4 | 2 |
| `-np 16 -c 65536` | **isolated** | 4/4 | 0/3 | — | 0 |
| `-np 32 -c 131072` | all four policies | 4/4 | 0/4 | 0/4 | 0 |

The reused-slot arms **refuse more often at 1,024** than the clean arm does. A
leaking harness made the leaf look *better behaved* than it is, then leaked
outright one rung later. Two consequences worth carrying forward:

* A contaminated measurement is not conservatively wrong. It can mask a failure
  as easily as manufacture one, so "the leak would only have hurt us" is not
  available as a defence of any past number.
* `virgin` (one slot per *fixture*, both questions on it) is **not** clean enough
  for this measurement. Under it the model answered the ABSENT question with the
  key it had just emitted for the LITERAL question on that slot in 6 of 8 cases.
  R13-mitigations §4.3 calls same-document reuse legal, and for *leakage* it is —
  0 foreign keys — but the previous **question** still carries. Production asks
  multiple questions per window on one slot, so this is a live effect, not a
  probe artefact. It deserves its own measurement.

## 5. What this settles, and what it does not

**Settles:** the v1 file's numbers and its "collapse at 2,048" are withdrawn. The
true threshold sits between 640 and 1,024, in both model families, measured with
every contamination path closed.

**Corroborates rather than undermines the spec.** The parallel audit of the
load-bearing runs (`s2/audit/`) found them clean under a detector validated on a
known-leaking positive control: `distance.jsonl` emitted 278 identifiers, **278
in their own chunk, 0 from another fixture**, across 91 slots that each held
exactly one document, with 0 slot mismatches; `refusal-ab` 218/0/0 and 331/0/0
with `refusal-ab-640`; `sweep` 75 own / 0 foreign / 3 fabricated. The positive
control (`sweep-run1-shared-server.jsonl`, pre-R13) yields 17 foreign identifiers
in 54 calls, so the detector does fire when leakage is present. §4's cliff, the
false-positive rate and the instruction-decay model therefore stand — and this
probe, a different instrument on a different fixture generator, independently
reproduces the same ~1,000-token threshold.

**Validates the shipped geometry.** `size_tokens: 640` refuses 4/4 in both
models; 1,024 refuses 0/4 in both. The window sits on the correct side of the
threshold by measurement, not by inference.

**Does not settle the architecture question.** Both arms remain bounded-context
mechanisms (recurrent state vs sliding window), so a genuinely uniform
full-attention control is still absent from disk, and `fallback_leaf` is still
neither confirmed nor killed. What has changed is that the question is now less
urgent: the two families fail identically at the same threshold, which is weak
evidence that the effect is general instruction-following decay rather than
anything specific to Gated DeltaNet — and the geometric mitigation is measured to
work regardless.

**Next, in order:** (1) measure the same-slot previous-question carryover in §4,
since production depends on multi-question windows; (2) assert returned `id_slot`
in C4; (3) verify `Qwen3.8-27B`'s attention layout — still R7 checklist item 2
and the S5 swap target — and if it is uniform full-attention, run this ladder
against it as the clean discriminating arm.
