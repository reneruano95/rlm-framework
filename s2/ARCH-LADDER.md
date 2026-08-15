# Is the horizon architectural? — INCONCLUSIVE, and the control was confounded

**Date:** 2026-08-14 · **Script:** `s2/arch_ladder.py` · **Raw:** `s2/results/arch_ladder_*.jsonl`

## The question

The leaf (Qwen3.6-35B-A3B) is **hybrid**: ~25% of layers carry true KV attention,
the rest compress history into a fixed-size, context-**independent** recurrent
state (measured 62.8 MiB/slot regardless of context length). That is the shape
that produces a hard horizon, and §4 measured facts *and* instructions failing at
one shared threshold. If the horizon belongs to the architecture, then swapping
to a full-attention `fallback_leaf` fixes it — and the spec reserves a config
slot for exactly that. If it does not, that mitigation is dead and the horizon
must be managed geometrically forever.

## Result

Both models, same probe, size targets hit against **each model's own** `/tokenize`
(so "1024 tokens" means 1024 of that model's tokens in both arms), needle planted
at the END of the chunk so needle-distance cannot explain a failure, 3 planted
entity→UUID bindings, ABSENT question naming a fourth entity from the same family.

| chunk tokens | Qwen3.6-35B-A3B (hybrid) LITERAL / ABSENT-refused | gemma-4-12B-it (control) LITERAL / ABSENT-refused |
|---:|:---:|:---:|
| 640 | 4/4 · 4/4 | 4/4 · 4/4 |
| 1024 | 4/4 · 4/4 | 4/4 · 4/4 |
| 2048 | 4/4 · **1/4** | 4/4 · **2/4** |

Retrieval (LITERAL) is intact at every size in both models — expected, since the
needle sits at the end. The **refusal criterion** collapses at 2,048 in both.

## Why this does NOT settle the question

**The control is not a full-attention model.** `gemma-4-12B-it` was verified
during the R13 work as having no recurrent state (`arch gemma4`, no `ssm.*` keys,
loader allocates "non-SWA + SWA KV caches") — and that phrasing is the problem:
it confirms Gemma interleaves **sliding-window attention**. Gemma's local layers
carry a fixed attention window, so this model has *its own* limited-context
mechanism, just a different one from the leaf's recurrent state.

So the two arms compare **two different limited-context mechanisms**, not
"limited context" against "unlimited context". The result is therefore
*consistent with* the horizon being caused by a bounded-context mechanism, but it
cannot say which mechanism, and — critically — **it does not show that a
full-attention model would be free of the effect**. The `fallback_leaf`
mitigation is neither confirmed nor killed.

Additional limits, stated rather than buried:
- **n = 4 per cell.** 1/4 versus 2/4 is not a distinguishable difference; do not
  read the hybrid as "slightly worse".
- **This probe's threshold (2,048) is later than the validated harness's (1,024).**
  `s2/run_distance.py` reproduces the effect at 1,024 with 30/30 false positives;
  this probe does not. It is a weaker instrument, built for a two-model
  comparison the main harness could not do without a config fork. Its absolute
  thresholds should not be quoted against the main results.
- Both models are quantized (Q4_K/Q8_0) and differ in size and family.

## A methodological note worth keeping

The probe's **first** shape did not reproduce the phenomenon at all — one lone
planted entity plus an obviously-foreign ABSENT question ("Vandermoor Bureau" in
a document of nature words) made refusal trivial, and Qwen scored 4/4 refusals at
every size. Only after switching to three planted bindings and an ABSENT question
naming a **plausible fourth entity from the same family** did the collapse appear.

That is a finding in itself: the false-positive failure mode appears to need
*plausible neighbours to hand over*. It is consistent with the earlier
observation that 59/66 wrong answers quote a **real identifier from the same
chunk** — misattribution, not fabrication. A leaf with nothing to misattribute
does not misattribute.

## What a clean test needs

A model with **uniform global attention in every layer**, instruct-tuned, of
comparable size. None is currently on disk: the library holds gemma-4 (all
SWA-interleaved), Qwen3.6 (hybrid), Qwen3.8-27B (**attention layout unverified —
R7 anticipates exactly this and makes it day-one checklist item 2**),
Muse-Glimmer-30B and nemotron-3-nano-omni (both unverified).

Cheapest path forward, in order:
1. Verify `Qwen3.8-27B`'s attention layout from GGUF metadata (`ssm.*` keys) and
   the `-lv 4` load dump. It is on disk and it is already the S5 swap target.
2. If it is uniform full-attention, re-run this ladder against it — that is the
   discriminating arm.
3. If nothing on disk qualifies, the question stays open and the geometric
   mitigation (window inside the horizon) stands on its own merits regardless,
   since it is measured to work.
