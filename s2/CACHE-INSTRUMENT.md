# S2 — CACHE INSTRUMENT: `timings.cache_n` is honest; the model of it was wrong

**Date:** 2026-08-14 · **Build:** llama.cpp b10375 (ROCm/HIP, gfx1151), leaf `Qwen3.6-35B-A3B-UD-Q4_K_M`
**Runner:** `s2/run_cache_instrument.py` · **Tables:** `s2/analyse_cache_instrument.py` · **Raw:** `s2/results/cache_instrument.jsonl`
**Scale:** 6 conditions, 6 server launches, **373 calls**, 0 slot mismatches, 0 foreign-identifier hits, 0 errors.
**Blocks:** §7 #2's INSTRUMENT WARNING and §7 #3 (a)/(b)/(c), none of which could be scored while the counter was under suspicion.

---

## 0. The one-paragraph answer

`timings.cache_n` was accused on the strength of a single number — 1,374 reported
against 3 tokens of genuinely shared prefix (`s2/DISTANCE.md` §5). **The accusation
does not survive contact with a truth model that drops one assumption.** §4 states
that llama.cpp's prompt cache is per-slot with no cross-slot sharing, and the
prior probe's "true shared prefix" was computed against *the previous prompt on
that slot*. It is not per-slot: b10375 ships a **host prompt cache** on by
default (`--cache-idle-slots`, `--cache-ram 8192`) that saves idle slot states to
host RAM and restores them **onto a different slot** — the same subsystem
`s2/OCCUPANCY.md` measured as the entire latency-versus-occupancy effect. Scored
against the best common prefix with *any* prompt the process has served,
`cache_n` **never over-reported once in 373 calls**. Scored against the per-slot
model §4 asserts, it over-reported by up to **+961 tokens**. Switch the host
cache off and reuse becomes an exact closed-form function of two numbers the
scaffold already holds — verified to the token on **239/239 calls at two `-ub`
values**. The instrument to trust is therefore not a server counter at all: it is
a **scaffold-side predicted-reuse model**, with `timings.prompt_ms` against a
cold baseline as the independent cross-check that catches the day the law
changes.

---

## 1. Method — what "true" means here, and why it is not `cache_n`

For any two prompts the number of tokens that *can* be reused is a property of
the token sequences, not of the server: the length of their longest common token
prefix. Both sequences are tokenized by `/tokenize` with `add_special=true` on
the same server that then serves the call, and the string tokenized is **the
exact rendered prompt that is POSTed**, control tokens already neutralised — not
a re-composition of it. (`add_special=true` is the setting measured to match what
`/completion` counts: pre-flight versus served 284/285, 474/475, 1274/1275, a
constant +1 that the flag removes — `rlm.dispatcher.ServerClient.tokenize`.)

Two truths are computed on every call and both are recorded:

* `truth_lcp_prev_same_slot` — the LCP with whatever that slot last held. **This
  is §4's model**, and it is the one the prior probe used.
* `truth_lcp_best_any_slot` — the LCP with the best-matching prompt the *process*
  has ever served, on any slot. This drops the per-slot assumption.

Everything else — composition, prefix, sampling, layouts A and C — comes from
production code (`rlm.dispatcher.compose_leaf_user`, the sha256-pinned
`leaf-prefix.v1.md`, `s2.leafcall`'s layout C), so this measures the prompt
production sends. Cases, each on its own never-reused slot (within a case the
slot *is* reused — that is the thing R13 forbids and the thing being measured;
every answer still goes through R13's foreign-identifier detector):
`identical`, `prefix-only`, `diverge`, `requery`, `virgin`, `three-docs`,
`cross-slot`, `cross-slot-lag`, `layoutC-elsewhere`, a 9-point `diverge-sweep`,
a 4-length `requery-len`, and a 4-turn `root-turn` conversation. Conditions:
`default` (shipped flags), `cram0` (`--cache-ram 0`), `nocacheidle`
(`--no-cache-idle-slots`), and `ub128` (`-ub 128`).

Rendered leaf head: **311 tokens**, identical in all six launches.

---

## 2. Reported versus true, in both directions

### 2a. Under the shipped launch line (`default`, host cache at its 8192 MiB default)

| case | step | n | prompt tokens | reported `cache_n` | true LCP (prev on slot) | true LCP (any slot) | err vs per-slot | err vs any-slot | `prompt_n` | prefill ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `calibration` | cold-baseline | 12 | 1,292 | **0** | 0 | 311–337 | 0 | −311…−337 | 1,292 | 1,328 |
| `identical` | repeat | 3 | 966 | **962** | 966 | 966 | −4 | −4 | 4 | 52.0 |
| `prefix-only` | newdoc | 3 | 970 | **0** | 311 | 337 | −311 | −337 | 966 | 1,010.5 |
| `diverge` | diverged | 3 | 962 | **456** | 693 | 693 | −237 | −237 | 506 | 542.9 |
| `requery` | requery | 3 | 979 | **457** | 954 | 954 | **−497** | −497 | 522 | 582.2 |
| `virgin` | firstsight | 3 | 972 | **0** | 0 | 335 | 0 | −335 | 972 | 1,014.6 |
| `three-docs` | newdoc (2nd doc) | 3 | 973 | **136** | 311 | 336 | −175 | −200 | 840 | 930.8 |
| `three-docs` | newdoc (3rd doc) | 3 | 962 | **0** | 335 | 335 | −335 | −335 | 956 | 954.1 |
| `cross-slot` | virgin-slot-repeat | 3 | 970 | **0** | 0 | 970 | 0 | −970 | 970 | 1,015.0 |
| `cross-slot-lag` | virgin-slot-lagged | 3 | 970 | **0** | 0 | 970 | 0 | −970 | 970 | 1,027.5 |
| `layoutC-elsewhere` | seen-elsewhere | 3 | 962 | **958** | 3 | 962 | **+955** | −4 | 4 | 49.6 |

The last row is `s2/DISTANCE.md`'s anomaly, reproduced on demand: **958 reported
against 3 tokens shared with what that slot last held, and against 962 shared
with a prompt the process served two calls earlier on a different slot.** Prefill
collapsed to 49.6 ms, so the reuse was real. The counter was right; the truth
model was wrong.

### 2b. The same cases with the host cache switched off

Each cell is `cache_n` / true-LCP-prev-on-slot / true-LCP-any-slot.

| case / step | `default` | `cram0` (`--cache-ram 0`) | `nocacheidle` (`--no-cache-idle-slots`) |
|---|---|---|---|
| `identical` repeat | 962 / 966 / 966 | 962 / 966 / 966 | 962 / 966 / 966 |
| `prefix-only` newdoc | 0 / 311 / 337 | 0 / 311 / 337 | 0 / 311 / 337 |
| `requery` requery | 457 / 954 / 954 | 457 / 954 / 954 | 457 / 954 / 954 |
| `cross-slot` virgin-slot-repeat | 0 / 0 / 970 | 0 / 0 / 970 | 0 / 0 / 970 |
| `cross-slot-lag` virgin-slot-lagged | 0 / 0 / 970 | 0 / 0 / 970 | 0 / 0 / 970 |
| **`layoutC-elsewhere` seen-elsewhere** | **958 / 3 / 962** | **0 / 3 / 962** | **0 / 3 / 962** |

One row moves, and it is the anomalous one. **Either host-cache flag removes
cross-slot reuse entirely**, which is the direct test of the mechanism. Note also
that cross-slot restore does *not* fire onto a virgin slot (`cross-slot`,
`cross-slot-lag`: 0 under every condition, including with an intervening task) —
it fires only onto a slot that already holds a state and is being replaced. That
asymmetry is why the prior probe saw it in layout C's second document and nowhere
else.

### 2c. Direction of error, over all 373 calls

| claim | count |
|---|---|
| `cache_n` **exceeds** the process-wide truth (`truth_lcp_best_any_slot`) | **0 / 373** |
| `cache_n` **exceeds** the per-slot truth (§4's model) | 3 / 373, all `layoutC-elsewhere` under `default`, max **+961** |
| `cache_n` **falls short** of the per-slot truth | 111 / 373, max shortfall **−497** |
| `cache_n + prompt_n == tokenized prompt length` | **373 / 373** |
| non-zero `cache_n` on a first-sight call (`truth_lcp_prev_same_slot == 0`) | **0 / 236** |

So the honest summary is: **`cache_n` never over-reports against what the process
has actually seen. It over-reports by up to the whole prompt against the per-slot
model the spec is written in, and it under-reports against the true shared prefix
on nearly a third of calls — by up to 497 tokens.** The under-reporting is the
larger and more systematic error, and §7 #3 (b) is written entirely inside it.

Also settled in passing: **`prompt_n` is not a second opinion.**
`cache_n + prompt_n` equalled the independently tokenized prompt length on every
one of 373 calls, so `prompt_n` is `cache_n` subtracted from a constant and
carries zero additional information.

---

## 3. Why it under-reports: the reuse law, measured

`cache_n` looked erratic because reuse is **quantised to a single rollback point
per slot**. The `diverge-sweep` holds prompt length fixed and walks the
divergence point across the document:

| divergence point (true LCP) | prompt tokens | `cram0` | `default` | `nocacheidle` |
|---|---|---|---|---|
| 376 | 989 | **0** | 0 | 0 |
| 435 | 969 | **0** | 0 | 0 |
| 507 | 990 | **452** | 452 | 452 |
| 605 | 1,004 | **456** | 456 | 456 |
| 629 | 964 | **446** | 446 | 446 |
| 691 | 978 | **446** | 446 | 446 |
| 752 | 962 | **450** | 450 | 450 |
| 821 | 964 | **448** | 448 | 448 |
| 882 | 963 | **451** | 451 | 451 |

Reuse is flat at ~450 while the true shared prefix more than doubles, and it is
*zero* below a threshold. Subtract from each slot's **resident prompt length**
rather than from the incoming one and the flatness becomes exact: `N_resident −
cache_n` was **516 on every same-slot divergence that reused anything at all —
42 of 42, with the other 28 correctly reusing nothing** — across the nine
divergence points above *and* across four prompt lengths spanning 3.5×:

| chunk tokens | resident prompt `N` | `cache_n` | `N − cache_n` | reuse as % of new prompt |
|---|---|---|---|---|
| 320 | 641 / 647 / 652 | 125 / 131 / 136 | **516** | 19.3–20.7% |
| 640 | 962 / 970 / 973 | 446 / 454 / 457 | **516** | 46.1–46.7% |
| 1,280 | 1,607 / 1,614 / 1,610 | 1,091 / 1,098 / 1,094 | **516** | 67.6–67.8% |
| 1,900 | 2,235 / 2,225 / 2,231 | 1,719 / 1,709 / 1,715 | **516** | 76.6–76.7% |

**516 is `-ub` + 4, and that is measured, not inferred.** Relaunching at `-ub 128`
moved the gap to **exactly 132** at all four lengths. The `+4` is the
generation-prompt markup, the same 4 tokens a byte-identical re-send still
re-evaluates (`identical`: 962 of 966).

### The law

Let `N` be the token length of the prompt that last occupied the slot and `L` the
longest common token prefix between the incoming prompt and that one:

```
predicted_reuse(N, L, ub) =
    N − 4               if L ≥ N          # byte-identical re-send
    L                   if L ≥ N − 4      # CONTINUATION: the prompt extends the slot
    N − ub − 4          if L ≥ N − ub − 4 # DIVERGENCE: the single rollback point
    0                   otherwise         # the shared prefix is NOT available, however long
```

Scored against `timings.cache_n` on every call:

| condition | calls | `cache_n` == predicted | max abs error |
|---|---|---|---|
| `cram0` (`--cache-ram 0`) | 90 | **90 / 90** | 0 |
| `cram0-len` | 44 | **44 / 44** | 0 |
| `nocacheidle` (`--no-cache-idle-slots`) | 90 | **90 / 90** | 0 |
| `ub128` (`--cache-ram 0 -ub 128`) | 15 | **15 / 15** | 0 |
| `default-len` | 44 | **44 / 44** | 0 |
| `default` (shipped) | 90 | 83 / 90 | **+964** |

**239/239 exact with the host prompt cache off, at two micro-batch sizes**;
366/373 overall. All seven disagreements are under the shipped `default`, and all
seven are over-reports caused by cross-slot restore. (`default-len` scored 44/44
not because the host cache was off but because that phase's slot sequence never
gave it an opportunity — which is itself the point: under the default, whether
the counter matches depends on run history.)

Independent corroboration from before this experiment existed: §7 #3 (d) recorded
`tokens_cached: 29641` for a re-queried ~30K chunk. `N − 516 = 29,641` puts that
chunk's rendered prompt at 30,157 tokens, which is exactly where a ~30K chunk plus
a 311-token head lands. The law reproduces a measurement taken nine days earlier
under a different script.

---

## 4. The recommended instrument, and what it costs

Three candidates were on the table. Two are rejected on evidence:

* **`prompt_n` versus the tokenized length — rejected.** Not independent:
  `cache_n + prompt_n == tokenized length` on 373/373 calls.
* **`timings.prompt_ms` against a cold baseline — kept, but as the cross-check,
  not the primary.** Resolution measured against the server's own `prompt_n`:
  median absolute error **14–20 tokens (1.5–2.0% of the prompt)**, p95 **57–69
  tokens**, in the fixed-length conditions; it degrades to a 47–51 token median
  in the length-varying conditions because the cold rate is itself
  length-dependent (**786–802 t/s at 320 chunk tokens versus 977–989 t/s at
  1,900**), so it needs a per-length-class calibration to hold its error bar.
  Cost: 12 cold calls, ~15 s, once per server launch. Pooled cold rate this run:
  **961 t/s**.
* **A scaffold-side model of what SHOULD have been reused — RECOMMENDED.** It is
  §3's law. Its inputs are two numbers the scaffold already holds: the token
  length of the prompt it last sent to that slot, and the LCP between that prompt
  and the one it is about to send — one `/tokenize` call it already makes for
  admission (§5 C4 pre-flight). Zero extra model calls, zero extra prefill.

**The recommendation, stated as the thing to assert:** compute
`predicted_reuse(N, L, ub)` scaffold-side and assert `cache_n == predicted_reuse`
exactly. **Error bars: 0 tokens on 239/239 calls**, provided two preconditions
hold:

1. **`--cache-ram 0` (or `--no-cache-idle-slots`) must be in the launch line.**
   Under the shipped default the host prompt cache adds unmodelled cross-slot
   reuse — the same call reported 0 or 958 depending on host-cache occupancy and
   eviction order, which also means `cache_n` under the default is **not a
   function of the prompts and not reproducible run to run**. `s2/OCCUPANCY.md`
   already recommends this flag on latency grounds (it removes 272 s of a 543 s
   run). This is a **second, independent reason for the same flag**, and it
   should be recorded as such: the flag is now load-bearing for correctness of
   measurement, not only for speed.
2. **`ub + 4` is a property of a build and a flag (R11).** It was measured at 512
   and at 128; it must be re-measured whenever `-ub` or the llama.cpp build
   changes. `s2/run_cache_instrument.py --condition <name> --extra "--cache-ram 0
   -ub <n>" --reps 0 --diverge-reps 0 --cal-reps 1 --len-reps 1` re-derives it in
   ~4 minutes and is the regression test.

`prompt_ms` is kept alongside precisely because it is independent of the law: if
llama.cpp changes its rollback behaviour, `cache_n` and the prediction will both
move together while `prompt_ms` will not. **A divergence between predicted reuse
and `prompt_ms × cold-rate` larger than 69 tokens (the measured p95) is the
signal that the law itself has changed.**

---

## 5. §7 #3's three gates, re-derived

Each is restated as an assertion that is *measurable on this stack* and *able to
fail*. Where a gate cannot be made to fail, that is said plainly rather than
dressed up.

### Gate (a) — prefix integrity / the R3 drift detector

**The original cannot be salvaged as a cache assertion, and the reason now has a
closed form.** `tokens_cached ≥ prefix_len` on a warm-slot call with a new chunk
requires the divergence point (which for a new chunk is the end of the prefix,
311) to be at or after the slot's only rollback point, i.e. `311 ≥ N − ub − 4`,
i.e. `N ≤ 315 + ub`. At `-ub 512` that caps the whole rendered prompt at 827
tokens (chunk ≲ 490); at `-ub 128`, at 443 tokens (chunk ≲ 107). And even at that
cap the reuse equals `N − ub − 4 ≤ 311`, so the inequality is satisfiable only at
the single value `N = 315 + ub`. **Gate (a) as written is unmeetable at every
window size and every `-ub` except a one-token coincidence** — which strengthens
§4's existing "not meetable for a first-sight chunk by ANY configuration" from an
observation into a derivation. Worse, under R13's never-reuse policy production
has *no* warm-slot first-sight call at all: `cache_n` was **0 on 236/236**
first-sight calls. An assertion that cannot fail is not a detector.

**Re-derived, keeping the intent (detect prefix drift), in two parts:**

> **(a1) — always on, zero cost, no model call.** On every leaf and root call,
> assert `sha256(rendered_system_head) == config_snapshot.prefix_render_sha256`
> **and** `len(tokenize(head, add_special=True)) == 311`. Fail the episode
> (`outcome_reason=prefix_drift`) on either. *Measured stable at 311 tokens
> across 6 server launches and 373 calls; it fails the moment a byte moves, which
> is exactly and only what R3 is about.*
>
> **(a2) — the cache-side residue that CAN fail.** On the intra-window
> **re-query** — the only warm call production makes — assert
> `cache_n == N_resident − ub − 4` exactly. *Measured exact on 42/42
> divergences that reused anything, with the other 28/28 correctly reusing
> nothing, under `--cache-ram 0`/`--no-cache-idle-slots`; 0 tolerance.* A non-zero tolerance would be a fudge: the
> quantity is an integer identity, not an estimate.

### Gate (b) — repeated-chunk reuse (the win case)

**The >80% target is reachable, and buying it is measured to be a net loss.** The
ceiling on a re-query is `(N − ub − 4) / N_new`; 80% requires `N ≥ 5·(ub + 4)`,
i.e. **N ≥ 2,580 tokens at `-ub 512`** (chunk ≳ 2,250) — outside the window §4's
instruction-decay result permits. At `-ub 128` the requirement drops to N ≥ 660
and the 640-token window clears it: measured **85.7% reuse**. But the purchase
is priced, and it is bad. Per window, production pays one cold prefill and at
most one re-query:

| | cold prefill | re-query prefill | per-window prefill | reuse fraction |
|---|---|---|---|---|
| `-ub 512` (pinned) | 1,006.5 ms | 587.1 ms | **1,593.6 ms** | 46.5% |
| `-ub 128` | 2,022.7 ms | 371.5 ms | **2,394.2 ms** | 85.7% |

`-ub 128` halves cold prefill throughput (**961 → 478 t/s**, the same direction and size as
§7 #2 v0.2.5's ~43%) and costs **+50.2% of prefill per window** — +336 s over a
417-window 200K corpus. **Keep `-ub 512` and re-derive the gate; do not buy the
80%.**

> **(b1) — the assertion.** On every intra-window re-query,
> `cache_n == N_resident − ub − 4` (identical to a2; one identity serves both
> intents). *Measured 42/42 exact.*
>
> **(b2) — the win, priced honestly.** Median re-query prefill ≤ **0.62 ×** median
> cold prefill for the same window. *Measured 0.583 (587.1 / 1,006.5) at window
> 640, i.e. **1.72×**, saving 419–440 ms of prefill and ~441 ms of wall per
> re-query.* **This replaces the 20–40× figure in §7 #3 (d), which was measured at
> 30K chunks and does not transfer**: the saving is `ub + 4` tokens of re-prefill
> no matter the window, so it shrinks as a fraction exactly as the window shrinks.
> At the 640-token window the re-query lever is worth 1.72×, not 20×.
>
> **(b3) — the contingency trigger.** If (b1) fails on >2% of re-queries in an
> episode, LCP mis-routing or an R8 checkpoint regression is real and §7 #3 (b)'s
> per-episode `{chunk_hash → slot_id}` map ships. *Currently 0%.*

**What gate (b) can no longer claim:** a token-reuse *fraction* target is not a
meaningful gate on this stack, because the fraction is fixed by geometry
(`1 − (ub+4)/N`) and carries no information about whether caching is working.
Only the identity does.

### Gate (c) — root-turn integrity (the root-turn monitor)

**Measurable, and it passes with equality.** The root is the same hybrid
architecture (§7 #3c), so its turn-by-turn growth was measured here directly: a
rendered `[system, user]` prompt is a strict token prefix of the rendered
`[system, user, assistant, user']` that follows it, which is what "only the new
observation should prefill" means. Four-turn conversations, both host-cache
settings and both `-ub` values:

| turn | prompt tokens | `cache_n` | true LCP | `prompt_n` (new tokens only) | prefill ms |
|---|---|---|---|---|---|
| 0 | 960 | 0 | 0 | 960 | 967 |
| 1 | 1,010 | **956** | 956 | 54 | 177 |
| 2 | 1,061 | **1,006** | 1,006 | 54 | 180 |
| 3 | 1,112 | **1,057** | 1,057 | 54 | 180 |

`cache_n` equalled the true LCP **exactly on 14/14 turns**, prefill fell 967 → 178
ms (**5.4×**), and `prompt_n` was exactly the new observation. **R8's feared
checkpoint invalidation does NOT occur for pure conversation growth on b10375** —
an extension is a continuation, not a divergence, so it never touches the
rollback point that costs the leaf its reuse.

> **(c) — the assertion.** On every root turn *k* > 0, assert
> `cache_n == LCP(rendered_turn_k, rendered_turn_{k−1})` **and**
> `LCP ≥ prior_request_tokens − 4`. Flag the episode on either. *Measured 14/14
> exact; the −4 is the generation-prompt markup and is an identity, not slack.*
> Because `root_view_hash` (§6) already stores the rendered request per turn, the
> LCP is computable from the trace alone — the check runs in `rlm replay` as well
> as live.

**Stated limitation:** these turns were served by the **leaf** process (port 8081,
ROCm, `-np 128`) because it is the same architecture and this run's server. §4's
"verified on both servers" still requires one pass on the root (port 8080,
Vulkan, `-np 1`) before gate (c) is treated as verified there; the assertion is
ready, the root measurement is not taken.

---

## 6. Consequences to carry into the spec

1. **§4's cache contract has a false sentence.** "The prompt cache is **per-slot**
   — there is no cross-slot sharing" is wrong on b10375 with default flags:
   `layoutC-elsewhere` reused 958 tokens of a document served on a different slot.
   It becomes true only with `--cache-ram 0` / `--no-cache-idle-slots`.
2. **`--cache-ram 0` earns a second, independent justification.** `s2/OCCUPANCY.md`
   wanted it for latency. It is also the precondition that makes every
   token-weighted gate deterministic. It should be pinned in `config.yaml`'s leaf
   `extra_flags`, not left to a runner.
3. **§7 #2's INSTRUMENT WARNING can be discharged, not carried.** The counter is
   sound; the sentence to replace it with is that reuse is quantised to one
   rollback point at `N − ub − 4` and that the per-slot truth model was the error.
4. **R13's "virgin slot" is safe from this mechanism.** Cross-slot restore never
   fired onto a virgin slot in 12 attempts, including with an intervening task —
   so the host prompt cache is not a route by which a fresh slot inherits a
   previous document's state. (It is also not R13's carrier, which §10 already
   records: R13 survives `--cache-ram 0` and `--no-cache-idle-slots`.)
5. **The re-query lever is worth 1.72×, not 20–40×, at the geometry that ships.**
   Any S2 plan pricing intra-window re-query at the 30K-chunk figure is wrong by
   an order of magnitude.

---

## 7. Audit

| condition | calls | slot mismatches | foreign-identifier hits | `cache_n+prompt_n != tokenized` |
|---|---|---|---|---|
| `default` | 90 | 0 | 0 | 0 |
| `cram0` | 90 | 0 | 0 | 0 |
| `nocacheidle` | 90 | 0 | 0 | 0 |
| `default-len` | 44 | 0 | 0 | 0 |
| `cram0-len` | 44 | 0 | 0 | 0 |
| `ub128` | 15 | 0 | 0 | 0 |
| **total** | **373** | **0** | **0** | **0** |

**R13 (§10):** every call requested a slot no other call in its run would ever
use again, and the returned `id_slot` was asserted equal to the requested one on
every call (0 mismatches). Within a case the slot *is* reused across steps — that
is the measurement — and those answers are cache records, not quality records.
R13's foreign-identifier detector ran on all 373 answers: **0 hits**. Per §10 a
clean verdict is evidence, never a certificate. Every server started by this
experiment was shut down by its own runner.

**Answer correctness is not this experiment's variable**, but it was recorded and
it replicates §7 #2's distance cliff independently: with the identifier placed
near the top of the document, first-sight extraction scored **9/11 at 320 chunk
tokens, 19/38 at 640, and 0/22 at 1,280 and 1,900** — correct inside the measured
~1,000-token horizon, dead outside it, on a corpus built for a different purpose.
