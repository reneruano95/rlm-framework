# R13 — pricing the mitigations

Measured 2026-08-13 with `milestones/s2/r13_mitigation_bench.py` on build `b10375-ba360efe1`
(ROCm), leaf `Qwen3.6-35B-A3B-UD-Q4_K_M`. The leak oracle (`foreign_strings`)
and the transport are imported unchanged from `milestones/s2/r13_repro.py`, so every
leak verdict below is produced by the same code that produced `milestones/s2/R13.md`.
Every server started here was shut down.

**Headline: the two mitigations the spec named as the cheap ones are not
mitigations at all. `cache_prompt: false` leaks at the same rate as the default
and costs ~2× wall-clock. `--parallel 1` does not avoid slot reuse — it makes it
mandatory, and it is the slowest leaky option available. The mitigation that
works is a scaffold-side rule (one never-reused slot per window) on a
high-`--parallel` server, and it costs 13 seconds on a 200K-token corpus.**

---

## 0. What was measured, at a glance

All at production geometry (§7 #2: window 1024 / stride 768) unless stated.
Episode = 20 windows × 2 questions, every answer pinned to exactly 96 decoded
tokens (`ignore_eos`) so the arms compare like with like.

| # | option | leak-free? (measured) | prefill t/s | decode t/s | per-window overhead | 20×2 episode |
|---|---|---|---|---|---|---|
| 1 | `cache_prompt: false`, shared slot | **NO — 15/18 (83%)** | 924 (never cached) | 55 → 17 (k=8) | none | 60.2 s |
| 1b | `cache_prompt: false`, virgin slot | yes — but the virginity does the work | 924 | 55 | none | 49.1 s |
| 2 | one PROCESS per window | **YES** | 842 → 968 | 54.9 | **6.72 s load** | **263.8 s** |
| 3 | `--parallel 1`, serial, one slot | **NO — 4/18 (22%)** | 837 | 54.9 | none | 95.2 s |
| 4 | full-attention fallback leaf | not priced — see §5 | — | — | — | — |
| 5 | `action=erase` / save-restore | **NO** (33/54, `milestones/s2/R13.md` §2) | — | — | — | — |
| ★ | **one virgin SLOT per window, `-np 128`, k=8** | **YES — 0/72** | 948 | 54.9 → 10 | **0.05 s** (amortised restart) | **53.7 s** |

Raw records: `milestones/s2/results/r13_mitigation.jsonl` and `milestones/s2/results/r13_mit_*.json`.

---

## 1. Option 1 — `cache_prompt: false`. Priced first and hardest. It is dead.

The instruction was: if the reproducer showed the bug does not occur under some
ordinary condition, that is the cheap mitigation. So this was tested directly,
on a virgin server, with the same corpus, probe and oracle that produce the
positive control.

| arm | n | leaked | rate |
|---|---|---|---|
| shared slot, `cache_prompt: true` (control) | 18 | 15 | 83% |
| **shared slot, `cache_prompt: false`** | 18 | **15** | **83%** |
| virgin slot per document (either setting) | 18 | 0 | 0% |

`milestones/s2/results/r13_mit_nocacheprompt_full.json`, `r13_mit_posctl_full.json`.

**Every single leaking call reported `cache_n = 0`.** The prompt cache was not
consulted; the whole prompt was prefilled from scratch; the answer still
enumerated the previous document's keys. Verbatim, from the `cache_prompt:false`
run — a chunk that has never contained either string:

```
0f3aac07-d1fe-460c-907f-53ddb57cc79a  (lives only in s2-2048-p50)
1251d802-86aa-4e75-96be-aefc175c1e8e  (lives only in s2-1024-p50)
Quinfennsted, Prylfennwick            (ditto)
```

This is the same conclusion `milestones/s2/R13.md` §2 row 2 reached from cold re-prefills,
now confirmed under the flag itself: **whatever carries the contamination is not
the prompt cache, so turning the prompt cache off cannot stop it.**

### And its cost, priced anyway — one chunk, three questions

`milestones/s2/results/r13_mit_pattern.json`. Same slot, same three questions, `cache_prompt`
the only variable.

| chunk | `cache_prompt: true` | `cache_prompt: false` | penalty |
|---|---|---|---|
| 32,768 tokens (the spec's headline case) | 56.9 s | 126.1 s | **+69.2 s (2.22×)** |
| **1,024 tokens (production geometry)** | 5.27 s | 4.56 s | **none measurable** |

The 30K row reproduces §7 #3(d): cold 41.9 s vs warm 3.9 s per repeat question.
The 1,024 row is the one that matters, and it retires a load-bearing assumption:
**at window 1024 the warm re-query lever is worth 0.72 s per repeat question**
(prefill 1,350 ms cold → 630 ms warm), **not the 20–40× recorded in §7 #3(d)**.
That 20–40× was a property of 32K chunks. Once §7 #2's window/stride geometry
lands, prefilling a window from scratch costs 1.2 s, so there is not much left
to save.

**Verdict: `cache_prompt: false` buys zero correctness and costs ~2× wall-clock
across an episode (49.1 s vs 25.4 s at natural stop; 60.2 s vs 53.7 s
decode-normalised). Rejected on both axes.**

---

## 2. Option 3 — `--parallel 1`. Worse than dead: it *guarantees* the bug.

The spec priced this as "no fan-out at all", to be weighed against S0's flat
aggregate-prefill finding. Two measurements change the verdict.

**(a) It is not a mitigation.** With `-np 1` the server has exactly one slot, so
every window after the first *must* reuse the slot that held the previous one.
That is precisely the condition that leaks. Measured on a real `-np 1 -c 40960`
server, ascending window ladder, one slot:

> **4/18 leaked** — identical cells, identical foreign entities, identical rate
> to the same ladder on a shared slot of an `-np 8` server (4/18).
> `milestones/s2/results/r13_mit_np1_shared.json`

Serialising chunks does not give the slot amnesia. `--parallel 1` should be
struck from R13's interim-options list; it is the only listed option that makes
slot reuse structurally unavoidable.

**(b) The "fan-out is free to give up" argument does not survive the geometry
change.** S0's flat ~950 t/s aggregate curve was measured on 32K chunks, where
prefill is 97% of the call. At window 1024 the call is decode-bound — 1.2 s
prefill against 1.7 s decode — and decode *does* batch:

| k concurrent 1024-windows | wall/window (128-token decode, `-np 64`) |
|---|---|
| 1 | 3.52 s |
| 2 | 2.66 s |
| 4 | **1.67 s** |
| 8 | 1.93 s |
| 16 | 1.52 s |

`milestones/s2/results/r13_mit_tput_np64.json`. End to end over a 20×2 episode the same
effect shows as 92.5 s (k=1) → 53.7 s (k=8): **fan-out is worth 1.72× at
production geometry.** S0's finding stands as written — *aggregate prefill* is
still flat, per-stream prefill degrades exactly in proportion (948 → 197 t/s from
k=1 to k=16) — but prefill is no longer the whole cost, so the conclusion drawn
from it no longer transfers to 1K windows.

**Verdict: leaks at 22%, and costs 1.77× (95.2 s vs 53.7 s). Rejected on both
axes.**

---

## 3. Option 2 — one leaf PROCESS per window. Clean, and 4.9× too expensive.

Model load measured directly, from `Start-Process` to a 200 on `/health`, with
the server's own loader timestamps as a cross-check.

| | measured | loader-reported |
|---|---|---|
| **cold** (weights evicted from the OS file cache first — 69.6 GB read through at 2.13 GB/s to displace them) | **10.40 s** | 9.88 s |
| **warm** (OS file cache holds the 20.6 GB GGUF) | **4.61 – 5.14 s** | 4.17 s |
| warm, inside a tight restart loop (includes teardown + settle) | **6.72 s** median | — |
| first-call penalty after a cold load | +0.2 s (3.70 s vs 3.48 s steady) | — |

Cold is only 2.3× warm because llama.cpp mmaps the GGUF; in a per-window restart
loop every load after the first is warm, so **6.72 s is the honest per-window
overhead**, not 10.4 s.

Measured end to end (`milestones/s2/results/r13_mit_episode96_freshproc.json`, a real
20-launch loop, not a composition of parts):

```
20 windows x 2 questions, one process each
  episode wall      263.8 s      (13.19 s per window)
  of which launch   134.4 s      51% of the episode is loading weights
  of which inference 105.9 s
  of which teardown  23.5 s
```

**Verdict: leak-free, but 4.9× the recommended option (263.8 s vs 53.7 s), and
half of that is spent loading the same 20.6 GB of weights twenty times. Rejected
on cost.**

---

## 4. The recommendation — one virgin SLOT per window on a dense-slot server

`milestones/s2/R13.md` §8 identified the shape of this. This section prices it and finds it
is not merely cheaper than a process — it is nearly free, for a reason the
reproducer did not test: **the number of slots is a free parameter, and raising
it raises how many windows one process can serve before it must restart.**

### 4.1 It is leak-free, and the rule scales

| test | n | leaked |
|---|---|---|
| R13 §1 paired control (9 calls per virgin slot) | 54 | **0** |
| this run, `-np 64`, 24 virgin slots, one process accumulating 72 calls | 72 | **0** |
| this run, `-np 128`, slots 100–105 | 12 | **0** |
| this run, `-np 128`, fifteen 20×2 episodes, `id_slot` asserted on every reply | 600 | slot mismatches: **0** |
| positive control, same script, same probe, shared slot | 18 | **15** |

The positive control is what makes the zeros meaningful: the identical harness,
probe and oracle detect the leak at 83% one command earlier.

### 4.2 The dense-slot configuration

`-np 128 -c 327680` → **2,560 tokens per slot, 128 virgin slots per process**,
at the same total KV budget as the currently pinned `-np 8 -c 327680`.
Launch measured at 5.14 s; throughput identical to `-np 8` (k=1: 3.58 vs 3.77
s/window; k=16: 1.60 s/window). There is no penalty for allocating 128 slots.

Two bonuses fall out of it:

* **The restart cadence collapses.** One process serves 128 windows, so a 261-window
  200K corpus needs 3 processes — **2 restarts, 13.4 s total.** Per window that
  is **0.05 s**, against 6.72 s for one process per window: a **134× reduction in
  the same overhead**.
* **The server becomes a structural guard.** At 2,560 tokens per slot any prompt
  larger than a window is rejected outright — `HTTP 400 exceed_context_size_error`,
  observed on a 4,189-token prompt. An oversized chunk can no longer silently
  slip through and re-create the growth condition of §4.4.

### 4.3 It keeps warm re-query; `cache_prompt: false` throws it away

The rule is *never reuse a slot for a **different** document*. Asking a second
question about the **same** document on the **same** slot is same-document reuse:
it is warm, and it is measured clean (0/72 here at 3 calls per slot; 0/54 in
R13 §1 at 9 calls per slot). So the recommended option is the only one that both
stops the leak and keeps the warm path — modest as that path now is (§1).

### 4.4 New finding: the trigger is document GROWTH, not document size

This matters for how the rule must be enforced, so it was isolated rather than
assumed. Same script, same probe, same shared slot, virgin server each time:

| ladder of documents down one shared slot | leaked |
|---|---|
| six windows all ~842–1,020 tokens (uniform, production-like) | **0/18** |
| six windows ascending 159 → 475 → 737 → 842 → **1,416** → **1,934** tokens | **4/18** |
| six full documents ascending 1,021 → 32,873 tokens | **15/18** |

`r13_mit_windows_np8.json`, `r13_mit_smallladder.json`, `r13_mit_posctl_full.json`.

The first leak in the small ladder appears at **1,416 tokens** — *inside* the
production window range — and only once a document exceeds one the slot has
already held. Two consequences:

1. **Production geometry is not incidentally safe.** The 0/18 uniform result is
   not a safety property: it holds only while every window is the same length,
   and real chunking produces a short final window, boundary snapping
   (`snap_tolerance: 0.10`), and variable question lengths. Any window longer
   than its slot's predecessor re-arms the bug.
2. It sharpens R13 §7's mechanism triangulation — consistent with stale cells
   remaining addressable beyond the previous document's extent — but this is
   still black-box evidence and **no llama.cpp code path is claimed.**

### 4.5 Operational hazard the scaffold must handle

**An out-of-range `id_slot` is silently reassigned.** Asking for `id_slot: 200`
on a 128-slot server returns **HTTP 200** with `"id_slot": 72` — a slot that has
held other documents. There is no error and no warning. In-range ids are honoured
correctly, including under contention (four concurrent requests all pinned to
slot 5 were queued onto slot 5, none reassigned).

Therefore C4/`LLMDispatcher` must:
* keep the slot allocator strictly within `[0, --parallel)`;
* **assert `response.id_slot == requested`** on every leaf call, and treat a
  mismatch as a contaminated answer (`status=error`), not a warning;
* never hand a slot that has held document A to document B — restart the process
  instead when the pool is exhausted.

The rule is enforceable only because the server tells the truth about which slot
answered. That field is now load-bearing and needs a test.

---

## 5. Option 4 (full-attention fallback) — deliberately not priced

The brief said to price this **only if** the reproducer's control showed
full-attention models are clean. It showed the opposite: `gemma-4-12B-it`
(arch `gemma4`, no `ssm.*` keys, no recurrent cache allocated) leaked **39/54
(72%)** against the hybrid's 34/54 (63%), and on 4 of 6 cold prefills against
the hybrid's 2 of 6 (`milestones/s2/R13.md` §3). Swapping to it would pay a different
model's quality and speed for a *higher* leak rate. **`fallback_leaf` is not an
R13 mitigation and must not be listed as one.** It remains whatever it was as an
R8 hedge.

## 6. Option 5 (API-level reset) — no working form exists on this build

`action=erase` returns 200 with a truthful `n_erased` and does not stop the leak
(33/54 with erase vs 34/54 without, `milestones/s2/R13.md` §2 row 5). `save`/`restore` were
only ever exercised on the synthetic two-prompt design, which does not reproduce
the effect at all, so those are null results on a null instrument and are not
claimed here either way. There is no measured API-level reset that works.
(Correction already recorded in `milestones/s2/R13.md` §6: `erase` is not 501 on b10375 —
501 only means `--slot-save-path` was unset.)

---

## 7. The 20-chunk × 2-question episode, all arms

20 windows of 1,024 tokens, 2 questions each, every answer pinned to exactly 96
decoded tokens. `milestones/s2/results/r13_mit_episode96_*.json`.

| arm | leak-free | episode wall | vs best |
|---|---|---|---|
| **virgin slot/window, `-np 128`, k=8** | **yes (0/72)** | **53.7 s** | **1.00×** |
| virgin slot/window, k=4 | yes | 57.5 s | 1.07× |
| virgin slot/window, k=2 | yes | 69.8 s | 1.30× |
| virgin slot/window, k=1 | yes | 92.5 s | 1.72× |
| shared slot, k=1 (what `--parallel 1` forces) ‡ | **no (4/18)** | 95.2 s | 1.77× |
| `cache_prompt: false`, shared, k=4 | **no (15/18)** | 60.2 s † | 1.12× † |
| one process per window | yes | 263.8 s | **4.91×** |

† natural-stop run; the decode-normalised arms are the first five rows plus
fresh-process. The `cache_prompt:false` arms were measured at natural stop
(49.1–60.2 s vs 25.4 s for virgin k=4 on the same basis, i.e. ~2×).

‡ measured as a single pinned slot driven serially, which is behaviourally what
`--parallel 1` gives. The substitution is checked: a real `-np 1 -c 40960`
server measures 3.67 s/window against 3.58 s/window for k=1 on the `-np 128`
server — within noise — and it leaks at the same 4/18
(`milestones/s2/results/r13_mit_tput_np1.json`, `r13_mit_np1_shared.json`).

---

## 8. What this does to the project's economics

### 8.1 The arithmetic, restated

§7 #2's geometry: **window 1,024 / stride 768.** For a 200,000-token corpus,

```
windows = ceil((200,000 - 1,024) / 768) + 1 = 261
```

Measured prompt length per call is **~1,130 tokens** (window + system prefix +
question), so one full pass costs **295K prompt tokens** — a **1.47×** prefill
amplification over the raw corpus, from 25% window overlap plus the per-call
prefix. At 2 questions per window: **522 subcalls, ~590K prompt tokens.**

### 8.2 What the mitigation costs on a 200K corpus

**Nothing in subcalls, and 13 seconds in wall-clock.** The virgin-slot rule does
not add a single call — it changes which slot a call lands on. Its only cost is
the restart when the slot pool runs out, and at `-np 128` that is 2 restarts
(13.4 s) across the whole corpus.

Full pass over 200K, 2 questions per window, projected from the measured
per-window figures:

| configuration | wall-clock | note |
|---|---|---|
| **virgin slot, `-np 128`, k=8** | **11.9 min** (702 s + 13 s restarts) | leak-free |
| virgin slot, k=4 | 12.5 min | leak-free |
| `--parallel 1`, serial | 20.7 min | **leaks** |
| one process per window | **57.4 min** | leak-free; 29.2 min of it is loading weights 261 times |

Reference points: B1's single-pass 200K prefill is **10.4 min** (§7), and C5's
`max_wall_clock` default is **15 min**.

### 8.3 Is full coverage of a 200K corpus still affordable?

**On wall-clock, yes — barely, and the mitigation is not what threatens it.**
11.9 min of leaf work is in the same class as the 10.4-min single-pass reference
and fits inside a raised `max_wall_clock`. The mitigation contributes 13 s of
that. Had the answer been one process per window, the same corpus would take
57.4 min and the RLM arm would lose its structural advantage over single-pass
long context outright. **That is the number that decides the option: 13 s versus
29 min of redundant model loading.**

**On budgets, no — not at the current defaults, and this was true before R13.**

```
max_subcalls default            32
windows needed for 200K        261   (522 calls at 2 questions each)
corpus covered by 32 windows:  1,024 + 31 x 768 = 24,832 tokens = 12.4%
```

**A 32-subcall budget covers 12.4% of a 200K corpus.** §7 #2 already flagged that
`max_subcalls` "must rise with it or coverage silently breaks"; the number is
**8× (261, one question per window) to 16× (522, two).** `max_total_tokens` is
fine — ~640K against a 1.5M default. `max_wall_clock`'s 15-min default survives
the leaf work at 11.9 min but leaves under 4 min for root turns at 12 t/s decode,
and C5's formula (`≈3 × task_leaf_tokens / 947`) predicts 15.6 min because it
assumes serial prefill; it needs re-deriving for a geometry where the work is
decode-bound and batches 1.72×.

**Verdict: the architecture survives R13 in its current shape.** The blocking
defect is contained by a scaffold-side slot-allocation rule that costs 13 s per
200K corpus and requires no change to the fan-out model, the chunker, or the
model choice. What does *not* survive unchanged is `max_subcalls: 32` — but that
was already condemned by §7 #2's window/stride result and is a budget constant,
not an architecture.

---

## 9. Corrections to the R13 record

1. **`--parallel 1` is not an interim mitigation.** It forces slot reuse and
   leaks at the same rate as a shared slot on an 8-slot server (4/18 both).
   Strike it from ARCHITECTURE.md §10 R13's options list.
2. **`cache_prompt: false` is not a mitigation.** 15/18 leaked, every call at
   `cache_n = 0`. It was never listed in the spec, but it is the first thing
   anyone will try.
3. **`fallback_leaf` is not an R13 mitigation** and should not be listed as one
   (falsified in `milestones/s2/R13.md` §3; not re-measured here, by design).
4. **§7 #3(d)'s 20–40× warm re-query lever is geometry-dependent.** It is 2.22×
   over three questions at 32K and unmeasurable at window 1,024 (0.72 s of
   prefill saved per repeat question). It should not be cited as an argument
   about the post-§7 #2 design.
5. **S0's flat-aggregate-prefill finding does not transfer to 1K windows.**
   Aggregate prefill is still flat, but at window 1,024 the call is decode-bound
   and fan-out is worth 1.72× end to end.
6. **The leak's trigger is document growth, not document size** (§4.4). Uniform
   ~1,000-token windows on a shared slot: 0/18; an ascending 256→2,048 ladder:
   4/18, first leak at 1,416 tokens.
7. **An out-of-range `id_slot` is silently reassigned** with HTTP 200 (§4.5).
   The dispatcher must assert the returned `id_slot`.

## 10. Recommendation

**Run the leaf at `-np 128 -c 327680` (2,560 tokens/slot) and enforce, in
C4/`LLMDispatcher`, one never-reused slot per window — restarting the process
when the 128-slot pool is exhausted. Both questions about the same window go to
that window's own slot.**

Leak-free: 0/72 measured, against 15/18 on the positive control run by the same
harness one command earlier. Cost on a 200K corpus: **2 restarts, 13.4 seconds**
— 0.05 s per window, versus 6.72 s per window for the process-per-chunk
alternative that is the only other clean option.

Required alongside it: `max_subcalls` 32 → ≥261, an `id_slot` assertion on every
leaf call, and the slot allocator kept strictly in range.

## Files

* `milestones/s2/r13_mitigation_bench.py` — the harness (`--mode pattern|throughput|windows|episode`).
* `milestones/s2/results/r13_mitigation.jsonl` — every call, with verdicts.
* `milestones/s2/results/r13_mit_posctl_full.json` — positive control, 15/18.
* `milestones/s2/results/r13_mit_nocacheprompt_full.json` — option 1, 15/18.
* `milestones/s2/results/r13_mit_np1_shared.json` — option 3, 4/18.
* `milestones/s2/results/r13_mit_smallladder.json`, `r13_mit_windows_np8.json` — size vs growth.
* `milestones/s2/results/r13_mit_virginscale_{1,7,13,19}.json`, `r13_mit_virgin_np128.json` — 0/72, 0/12.
* `milestones/s2/results/r13_mit_tput_np{1,8,64,128}*.json` — throughput curves.
* `milestones/s2/results/r13_mit_episode96_*.json`, `r13_mit_episode_*.json` — the episode table.
* `milestones/s2/results/r13_mit_pattern.json` — one chunk, three questions.

Reproduce the decisive pair:

```
llama-server --host 127.0.0.1 --port 8081 -m <leaf.gguf> \
  -c 327680 -np 128 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 \
  -lm none --no-kv-unified --cont-batching

# positive control: 15/18 on a shared slot
uv run --python 3.12 python milestones/s2/r13_mitigation_bench.py --mode windows \
  --window 40000 --trials 3 --arms shared_slot --cells \
  s2-1024-p50,s2-2048-p50,s2-4096-p50,s2-8192-p50,s2-16384-p50,s2-32768-p50

# the mitigation: 0/18 on virgin slots, same prompts
uv run --python 3.12 python milestones/s2/r13_mitigation_bench.py --mode windows \
  --window 40000 --trials 3 --arms virgin_slot --slot-base 1 --cells \
  s2-1024-p50,s2-2048-p50,s2-4096-p50,s2-8192-p50,s2-16384-p50,s2-32768-p50
```
