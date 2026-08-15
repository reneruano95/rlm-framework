# S2 gates (a) and (b) — SCORED, both PASS

**Date:** 2026-08-15 · **Script:** `s2/gate_ab.py` · **Raw:** `s2/results/gate_ab.jsonl`
**Leaf:** the shipped launch line, `-c 327680 -np 128 -ctk q8_0 -ctv q8_0 -fa on
-ub 512 -b 2048 -lm none --no-kv-unified --cont-batching --cache-ram 0`

`--cache-ram 0` is the stated precondition on all three gates, and it is in
`config.yaml`'s `servers.leaf.extra_flags`, so the gate runs against what ships.

## The re-query fixture

§9 names one as an S2 deliverable. It is this: each of the seven named
`s2/fixtures-refusal-640-s*` cells is prefilled ONCE on a **never-reused slot**,
then asked two further questions about the **same** document on that same slot.
That is the only warm call production makes — R13 forbids reusing a slot across
different documents, and `s2/CARRYOVER.md` measured same-document reuse as safe
to depth ten. The cells are non-benchmark fixtures, as §8 requires of every S2
gate. 7 cells × 3 questions = 21 calls, 7 cold and 14 warm.

## Result

| gate | criterion | measured | verdict |
|---|---|---|---|
| (a1) | `sha256(rendered_head)` constant on every call | 1 distinct hash over 21 calls | **PASS** |
| (a1) | rendered head is 311 tokens | **311**, every call | **PASS** |
| (a2)/(b1) | `cache_n` == the reuse law, exactly | **14/14** | **PASS** |
| (b2) | median warm prefill ≤ 0.62 × median cold | **0.566** (580.1 ms / 1024.5 ms) | **PASS** |

Reproduced on three independent slot ranges (base 1, 30, 60): 0.565, 0.572,
0.566. (a2) is an integer identity and hit exactly on every warm call in each.

**The 311/304 question is now settled by measurement, not inference.** The
rendered head — template markup and generation prompt included — is 311 tokens;
the raw prefix body is 304. They are different strings, and the gate is about
the head. The probes that printed "prefix tokens: 304" all week were tokenizing
the body.

## Two mistakes this gate made before it passed, both mine

**1. The reuse law is not `incoming − ub − 4`.** Scored that way, (a2) reported
**0/14 FAIL** against a system that was exactly right. `n_resident` in
`cache_n == n_resident − ub − 4` is the token length of the prompt that **last
occupied the slot**, not the incoming one, and the law also depends on the LCP
between them. By hand: slot 1's first re-query reused 476, and the previous
prompt was 992 tokens — 992 − 512 − 4 = 476. The runner now imports
`rlm.dispatcher.predicted_reuse` instead of restating the formula, so the gate
cannot drift from the code it scores.

**2. A gate runner must start from virgin slots, and only the cold call can
prove it.** Re-running against a live server without moving `--slot-base`
scored nothing: the "cold" calls came back warm (`cache_n` 494 on a call never
made in that run), (b2)'s denominator collapsed to a warm number, and the ratio
landed at **1.006 against a 0.62 bar** — a FAIL that describes the runner, not
the system. There is now a tripwire: a cold call reporting non-zero `cache_n`
aborts with instructions to restart the leaf or move the slot base.

Both failures are the same shape as the arch-ladder mistake earlier this week:
an instrument that measures its own configuration and reports it as a property
of the model.

## Field note kept deliberately

`timings.cache_n` is the reuse count the gate is written against. The response's
top-level `tokens_cached` is a **different number** — `rlm/dispatcher.py:437-442`
says so in as many words ("NOT interchangeable"). Both are recorded per row in
`gate_ab.jsonl` so the distinction stays visible in the data rather than only in
a comment.

## What remains in S2

Gate **(c)** — fan-out speedup — stays **BLOCKED**, and not on throughput: R14
measured concurrent leaf dispatch corrupting answers (31/32 correct serial, 7/32
at two in flight). It may not be scored, and `scaffold.dispatch_concurrency`
stays 1, until that defect has a reproducer and a mechanism. Nothing here
changes that.

Still open elsewhere in S2: the question-batching A/B (pre-registers as a likely
reject — batching pushes the prompt past the ~1,000-token instruction horizon
for a prize of ~1.15× at k=2), and benchmark authoring, which is blocked on the
§8 aggregation ruling priced in `s2/aggregation_options.py`.
