# Does the previous question on a slot corrupt the next answer? — NO

**Date:** 2026-08-14 · **Script:** `s2/carryover.py`
**Raw:** `s2/results/carryover_prod.jsonl`, `carryover_prod640.jsonl`
**Server:** production geometry, `-c 327680 -np 128 -ctk q8_0 -ctv q8_0 -fa on
-ub 512 -b 2048 -lm none --no-kv-unified --cont-batching`
(`n_slots = 128, n_ctx_slot = 2560`, from `s2/logs/carryover.err`)

## Why this was asked

`R13-mitigations.md` §4.3 rules that asking a second question about the **same**
document on the **same** slot is legal and leak-free (0/72), and the project's
warm re-query economics depend on it: §5 slot discipline gives each window one
never-reused slot and puts every question about that window on it.

The corrected arch ladder appeared to contradict that. Under one-slot-per-fixture
reuse, the leaf answered the ABSENT question with the key it had just emitted for
the LITERAL question on that slot in **6 of 8 cases**, and `ARCH-LADDER.md` §4
recorded that as a live effect owed a measurement. This is that measurement, and
**it does not reproduce.**

## The confound that made 6/8 look real

In the arch ladder the priming question asked about `present[0]` — the **first**
binding in the document. That is also the leaf's preferred misattribution target:
measured here, baseline (unprimed) misattributions pick `present[0]` **4 times
out of 5** resolvable cases. So "repeated the previous answer" and "chose its
usual wrong answer" were the same string, and could not be told apart.

This probe primes on `present[2]`, the **last** binding, which the model does not
otherwise favour. An answer of `uid_C` is then evidence of carry-over rather than
of baseline preference.

## Design

One document per fixture planting A = `present[0]`, B = `present[1]`,
C = `present[2]`, and a fourth entity D that is genuinely absent. Three arms per
measured question, each on its **own never-used slot**, prompts byte-identical
across arms, greedy:

| arm | what runs on the slot |
|---|---|
| `solo` | the measured question alone, nothing before it |
| `after_cp1` | LITERAL about C first, `cache_prompt: true` — **what production does** |
| `after_cp0` | LITERAL about C first, `cache_prompt: false` — same slot, no prefix reuse |

`after_cp0` exists to separate "KV prefix reuse" from "slot state": if only
`after_cp1` degraded, the warm path would be the cause; if both did, something
about the slot would be.

Measured questions: **ABSENT** about D (correct answer `NONE`) and **LITERAL_B**
about B (correct answer `uid_B`, a binding the priming question never touched).

## The warm path really engaged

Not assumed — checked, because an arm that silently failed to reuse would make
this a null result about nothing:

| size | arm | rendered | evaluated (`prompt_n`) | reused |
|---:|---|---:|---:|---:|
| 640 | `solo` | 973 | 973 | 0 |
| 640 | `after_cp1` | 973 | **516** | **457** |
| 640 | `after_cp0` | 973 | 973 | 0 |
| 1,024 | `after_cp1` | 1,366 | **516** | **850** |

The priming question was answered correctly in **80/80** and **64/64** runs, so
the slot really did hold a correct prior answer before the measured question ran.

## Result — at the shipped window, no effect at all

**640 tokens (`size_tokens`, what ships), n = 20 per arm:**

| measured | `solo` | `after_cp1` | `after_cp0` |
|---|:--:|:--:|:--:|
| ABSENT refused | **20/20** | **20/20** | **20/20** |
| LITERAL_B correct | **20/20** | **20/20** | **20/20** |

**40 of 40 fixture-questions produced BYTE-IDENTICAL answers across all three
arms.** Not merely equal scores — the same characters.

**1,024 tokens, n = 8 per arm** (past the horizon, where ABSENT already fails):

| measured | `solo` | `after_cp1` | `after_cp0` |
|---|:--:|:--:|:--:|
| ABSENT refused | 1/8 | 1/8 | 1/8 |
| LITERAL_B correct | 8/8 | 8/8 | 8/8 |

The failure rate is **identical in all three arms** — priming neither causes nor
worsens it. 29 of 32 fixture-questions were byte-identical across arms; of the 3
that differed, one made the model refuse *correctly* under priming and one made
it fail, so the changes go in both directions and are consistent with a marginal
decision rather than a bias.

The classifier's `ECHO_PREV` count (1–2 of 8 at 1,024) **overstates the effect by
construction** and should not be quoted on its own: at 1024/t3 the `solo` arm
chose the same key the priming question had used, so the "echo" is baseline
misattribution that happens to coincide. Only the paired per-fixture comparison
is meaningful here.

## What this settles

* **Production's warm re-query is safe at the shipped geometry.** Multiple
  questions per window on that window's slot, `cache_prompt: true`, 850 tokens of
  prefix genuinely reused — and the answers do not change by a single character.
  §5's slot discipline and §4's prefix contract stand as written.
* **`ARCH-LADDER.md` §4's "6 of 8" is withdrawn.** It was the `present[0]`
  confound, not carry-over. The claim is corrected in that file.
* **The 1,024 failure is a property of the prompt, not of the serving path.** It
  is identical whether the slot is fresh, warm-reused, or reused without prefix
  reuse — which is one more independent line of evidence that the ~1,000-token
  horizon is real and not an artefact of how this project serves requests.

## Depth — ten questions down one slot

The depth-two result above left the accumulation case open: a residue invisible
at question two could still be obvious at question ten, and that is what
production hits at high `max_subcalls`. Measured with `s2/carryover_depth.py`,
same production geometry, at the shipped 640-token window.

Ten questions per fixture, alternating LITERAL and ABSENT so neither kind sits
only at the start or only at the end — a run that put every ABSENT last would
confound position with question difficulty. Two arms:

* **deep** — all ten questions down **one** never-used slot, in order
* **shallow** — the same ten questions, each on its **own** never-used slot

Question *i* is byte-identical in both arms, so any difference at position *i* is
the slot's history and nothing else.

| position | kind | deep | shallow | deep `prompt_n` |
|---:|---|:--:|:--:|---:|
| 0 | LITERAL | 8/8 | 8/8 | 974 |
| 1 | ABSENT | 8/8 | 8/8 | 516 |
| 2–8 | alternating | 8/8 each | 8/8 each | 515 |
| 9 | ABSENT | 8/8 | 8/8 | 514 |
| **total** | | **80/80** | **80/80** | |

**80 of 80 answers byte-identical between the deep and shallow arms.** No decay
with position, and none in the aggregate.

The `prompt_n` column shows the warm path doing real work rather than the arm
quietly degenerating into cold calls: 974 tokens evaluated for the first question
on the slot, then ~515 for every subsequent one — about **460 tokens of document
prefix reused per question**, which is the whole point of putting several
questions on one window's slot.

**So the accumulation case is clean too:** a window may be asked at least ten
questions on its slot at the shipped geometry with no measurable quality cost,
and the 2nd through 10th questions each cost roughly half the prefill of the
first. `max_subcalls` needs no depth cap on quality grounds.

## Limits, stated

* n = 20 per arm at depth two and n = 8 per cell at depth ten, both with **zero**
  failures — by the rule of three that bounds the true rate at roughly 15% and
  31% respectively. This is "no detectable effect", not "provably zero".
* Leaf only. The root's multi-turn path is a different mechanism (§4 R8) and is
  not covered here.
* **Depth was measured at 640 only.** At 1,024 the ABSENT criterion is already
  saturated at failure, so that cell cannot show accumulation on top of it; a
  depth run there would only test whether LITERAL recall decays, which is a
  different question and is untested.
* Ten is not a hundred. Nothing here says a slot asked 100 questions is safe.
