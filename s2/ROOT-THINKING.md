# Should the root think? Accuracy says yes, the cache says it is expensive

**Date:** 2026-08-15 · **Scripts:** `s2/root_thinking.py`, `s2/root_multiturn.py`
**Raw:** `s2/results/root_thinking.jsonl`, `s2/results/root_multiturn.jsonl`
**Server:** the shipped root config — Qwen3.8-27B-Q4_K_M, vulkan, `-c 32768
-np 1 --spec-type draft-mtp --spec-draft-n-max 2`

`config.yaml` sets `enable_thinking: false` for **both** roles. That was decided
on the LEAF, where thinking measurably burns the whole generation budget on a
preamble and never emits an answer. The root is a different job, so the setting
was **inherited, not measured**. This measures it, and the two halves disagree.

Qwen3.8-27B's template (read from the GGUF, `s2/gguf_compare.py`) exposes three
kwargs: `enable_thinking` (default **true**), `preserve_thinking` (default
**true**), and `reasoning_effort` — one of `xhigh` (**the default**), `medium`,
`low`. `xhigh` and `low` each inject an instruction into the system block;
`medium` injects none. So "turn thinking on" is two decisions, and the default
is the expensive one.

## 1. Accuracy — thinking wins, and `xhigh` is the CHEAPEST way to get it

8 filter-and-aggregate tasks over 24 records (the shape the root orchestrates),
ground truth computed in Python, exact match on an integer, greedy, one server:

| arm | correct | median tokens | median wall | tokens per correct |
|---|:--:|---:|---:|---:|
| **off** (production today) | **5/8** | 4 | 12.6 s | 6 |
| low | **8/8** | 353 | 24.3 s | 322 |
| medium | **8/8** | 364 | 24.9 s | 325 |
| **xhigh** | **8/8** | **224** | **20.6 s** | 199 |

The `off` arm's three failures are ordinary arithmetic slips — 213 for 138, 204
for 132, 299 for 289 — answered confidently and wrongly, which is this project's
recurring failure shape.

**`xhigh` is cheaper than `low`**, using fewer tokens on 7 of 8 tasks. Telling
the model to "validate key assumptions and prioritize correctness" appears to
send it straight to a systematic method, while "keep your thinking brief"
produces meandering shortcuts. Stated as suggestive, not established: n=8, one
rep, sign test p≈0.07 — but it is consistent and it points the opposite way to
intuition, so it should not be assumed away.

## 2. Multi-turn cost — thinking destroys the root's prefix reuse

§7 #3c and §9 S0 item 5(b) want a scripted 3-turn conversation showing per-turn
reuse covering all prior-turn tokens — "only the new observation prefills". That
pass was owed on the old root, and the S5 swap made it owed again. Measured, with
`cache_prompt: true` and no truncation (`n_predict 3072`):

| arm | turn | rendered | evaluated | reused | reuse |
|---|:--:|---:|---:|---:|---:|
| **think-off** | 1 | 369 | **55** | 314 | **85.1%** |
| **think-off** | 2 | 744 | **49** | 695 | **93.4%** |
| preserve-on | 1 | 2,209 | 2,106 | 103 | 4.7% |
| preserve-on | 2 | 3,332 | 1,127 | 2,205 | 66.2% |
| preserve-off | 1 | 2,057 | 1,954 | 103 | 5.0% |
| preserve-off | 2 | 3,371 | 1,318 | 2,053 | 60.9% |

**With thinking off the continuation property holds** — 85% then 93% reuse, 55
and 49 tokens evaluated per turn. That discharges S0 item 5(b) on the new root,
*for a non-thinking root*.

**With thinking on it collapses.** Turn 1 re-prefills the entire prior turn
(2,106 of 2,209 tokens); only the 103-token system prefix survives. The prior
assistant turn does not match the tokens the slot cached when it was generated —
the template's rendering of a reasoning turn differs from the raw generation — so
the prefix DIVERGES instead of extending, which is exactly the failure mode §4's
prefix contract exists to prevent.

**`preserve_thinking` is not the lever.** True and false give the same reuse
(4.7% vs 5.0%, 66.2% vs 60.9%) and near-identical prompt sizes. `enable_thinking`
is what moves it.

Two costs compound. Root prefill on this build is **~87 tok/s** (measured, vulkan,
27B dense), so re-prefilling 2,106 tokens costs **~24 s of the serial part of a
root turn** against ~0.6 s when reuse holds. And thinking grows the prompt
**4.5×** by turn 2 (3,332 vs 744 tokens), which spends the 32K root window faster
and brings `context_exhausted` closer on a long episode.

An earlier run of this probe capped generation at 1,024 tokens and every thinking
turn hit the cap, leaving unclosed `<think>` blocks in the history — a plausible
cause of the divergence. It is not the cause: re-run at 3,072 with nothing
truncated, the collapse is *worse* (4.7% against 8.7%).

**A probe bug worth recording, because it was already documented and I hit it
anyway.** This probe first read the response's top-level `tokens_cached` field
and treated it as a reuse count. It is not: a 107-token prompt reported 1,130,
which is 107 + 1,024 generated — the slot's occupancy *after* the call — and
dividing gave a nonsensical "1056% reuse". `rlm/dispatcher.py:437-442` already
says so in as many words: `tokens_cached` in a `steps` row comes from
**`timings.cache_n`**, "NOT the top-level `tokens_cached` field — they are NOT
interchangeable". So the shipped code reads the right field and this probe read
the wrong one. The reuse numbers above are unaffected: they are derived as
`rendered − evaluated`, which does not depend on either field.

## 3. What this does not settle

The accuracy win is measured **single-turn**; the cache cost is **multi-turn**.
They are not directly comparable, and no arm here measures accuracy *on a
multi-turn task with the cache cost included*. So this file deliberately does not
flip `enable_thinking` in `config.yaml`:

* the accuracy delta is large (5/8 → 8/8) and I5 does not allow trading
  correctness for speed;
* but the cost is paid on the serial, decode-and-prefill-bound part of **every
  root turn**, and it scales with the conversation, not with the task — on a
  realistic 20K-token root context, re-prefilling every turn at 87 tok/s is
  minutes per turn against a 900 s episode cap;
* and the root's real work is writing Python in a REPL loop, which is neither
  this file's arithmetic tasks nor its 3-turn code conversation.

**The decision needs one more measurement** — a full episode on the S1 fixtures
with thinking on and off, scored on task success and wall clock together. That
run is owed anyway: the S5 root swap invalidated S1's 3/3, which credited
root-as-programmer on the *old* root. Doing both in one pass answers both.

Until then `enable_thinking: false` stands for the root, recorded here as
**measured-and-contested** rather than inherited — which is already a better
position than it was in this morning.
