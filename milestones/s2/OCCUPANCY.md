# S2 — OCCUPANCY: where the per-call wall-clock goes as the slot pool fills

**Runner:** `milestones/s2/run_occupancy.py` · **conditions:** `milestones/s2/occupancy_conditions.ps1`,
`milestones/s2/occupancy_rest.ps1`, `milestones/s2/occupancy_conc.ps1`, `milestones/s2/occupancy_warm.ps1` ·
`milestones/s2/occupancy_ladder.ps1` · **analysis:** `milestones/s2/analyse_occupancy.py` ·
**raw:** `milestones/s2/results/occupancy.jsonl`
(1,416 calls, 24 conditions, one leaf process per condition) ·
**server logs:** `traces/logs/occ-*.log` · llama.cpp b10375 ROCm,
`ROCBLAS_USE_HIPBLASLT=1`, 2026-08-14.

## 0. The answer

`milestones/s2/DISTANCE.md` §4b reported per-call wall growing **1.15 s → 5.02 s** with
slot-pool occupancy while the server's own `timings.prompt_ms` and
`timings.predicted_ms` stayed flat. It reproduces, and larger: on an identical
128-call workload the median wall goes **2.18 s at occupancy 0–15 to 6.69 s at
112–127** (peak call 7.20 s) with prefill 1,391 → 1,389 ms and decode 758 →
593 ms — **flat to within noise**.

**The growing bucket is the one §6 defines for exactly this purpose:
`latency_queue_ms`.** Every millisecond of the growth lands between the
client's dispatch and the first byte of the response, and none of it is in
prefill, decode, streaming, or the client.

**The cause is llama-server's host prompt cache, and the server's own log names
it.** `--cache-idle-slots` is on by default and saves every idle slot to a
host-RAM prompt cache **on each new task**; `--cache-ram` defaults to 8,192 MiB
while one slot of this configuration is **202.80 MiB**, so the cache holds
**40.4 → 41 entries against a 128-slot pool**. Past 41 occupied slots the cache
cannot hold the working set and every request re-saves and re-evicts all of it:

    evictions logged between one `launch_slot_` and the next
      occupancy  0..40  ->   0
      occupancy  41     ->   1
      occupancy  42..127 ->  exactly the occupancy

**Two flags remove it completely and independently** — `--cache-ram 0` and
`--no-cache-idle-slots` — each taking the 128-call run from 542.9 s to 270.5 s
/ 274.7 s and the per-call wall to a **flat 2.05–2.16 s at every occupancy from
0 to 127**. Neither costs the intra-window re-query, which is the one cache
lever R13 leaves intact (§5 below: warm prefill 50 ms and `cache_n` 1,335 with
the host cache off, byte-identical to the default).

**For the geometry decision the effect is a red herring, and that is the
finding.** The overhead is priced by the **slot pool**, not by the window: it is
**37.5 ms per occupied slot** at window 1,024 and **38.3 ms** at window 640
(entry size 202.80 vs 198.77 MiB), and both geometries drain the same 128-slot
pool to the same ceiling. **A 417-window episode and a 261-window episode reach
identical peak occupancy (127) and therefore identical per-call wall — 6.87 s
against 6.35 s, the difference being the smaller window's cheaper prefill, not
the occupancy.** The occupancy effect does not discriminate between the
geometries; it taxes both, and with `--cache-ram 0` it taxes neither.

**A second blocker fell out of the controls and it is worse than this one:
concurrent leaf dispatch destroys the answer.** Serial, the leaf returns the
literal in-document answer 31/32 times; at 2 calls in flight, 7/32; at 4, 2/32;
at 8, 7/128 — degenerate text, not wrong answers. Reproduced with two
independent OS processes sharing one server, so it is not this runner's async
client. `scaffold.dispatch_concurrency: 8` is shipped config. See §7.

## 1. The decomposition — which bucket grows

Every call is split into buckets that sum to the wall:
`send` (dispatch → response headers) + `queue` (headers → first token, less
`prompt_ms` and one token of decode) + `prefill` (`timings.prompt_ms`) +
`decode` (`timings.predicted_ms`) + `tail` (streaming and teardown).
Two probes ride in front of every call: `GET /health`, answered by the HTTP
thread without touching the task queue, and `GET /slots`, answered by the
inference loop through the task queue.

**`baseline` — the shipped launch line, 128 calls, one virgin slot each.**

| occupancy | n | wall s | send ms | queue ms | prefill ms | decode ms | tail ms | residual ms | /health ms | /slots ms | server private MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-15 | 16 | **2.18** | **38** | -18 | 1391 | 758 | 19 | 39 | 2.8 | 1.9 | 35,813 |
| 16-31 | 16 | **2.10** | **41** | -18 | 1384 | 669 | 19 | 42 | 2.8 | 2.2 | 41,128 |
| 32-47 | 16 | **2.19** | **48** | -18 | 1391 | 602 | 19 | 49 | 2.8 | 2.7 | 46,443 |
| 48-63 | 16 | **4.01** | **2036** | -18 | 1392 | 616 | 19 | 2037 | 2.8 | 3.8 | 48,812 |
| 64-79 | 16 | **4.75** | **2637** | -18 | 1394 | 727 | 19 | 2638 | 2.9 | 3.7 | 50,876 |
| 80-95 | 16 | **5.41** | **3292** | -18 | 1390 | 597 | 19 | 3292 | 2.7 | 4.1 | 52,939 |
| 96-111 | 16 | **6.04** | **3972** | -18 | 1387 | 614 | 19 | 3974 | 2.8 | 4.8 | 55,005 |
| 112-127 | 16 | **6.69** | **4632** | -18 | 1389 | 593 | 19 | 4633 | 2.8 | 5.4 | 57,066 |

Read the row: **`prefill`, `decode`, `queue` and `tail` are constant to within
noise across a 3.07× change in wall.** `residual_ms` (= wall − prefill −
decode, the bucket-agnostic form) tracks `send_ms` exactly. The §6 quantity
`latency_queue_ms = (t_first_byte − t_dispatch) − timings.prompt_ms` is
`send + queue + one token`, so **the instrument the spec already defines is the
one that catches this**, and a real episode's `steps` rows would have shown it.

**It is not the client.** `/health` is 2.8 ms at every occupancy and `/slots`
— which is answered by the inference loop — is 1.9 → 5.4 ms, so the HTTP path,
the task queue and the loop are all responsive in the instant before the call.
The workload is serial (one call in flight), so nothing can queue behind
anything. What grows is work done *for this request*, on the server, before
prompt processing starts.

**Where `send_ms` sits, stated precisely.** On this build the SSE response
headers are not observed until the slot's pre-prefill work is finished: the
first byte then arrives exactly `prompt_ms` + one decode step later (`queue_ms`
is a constant −18 ms at every occupancy, the sign being the one-token
subtraction over-correcting). So `send_ms` is not client send time — it is
server-side per-request work ahead of prefill, and §4 locates it in the log to
the millisecond.

## 2. The mechanism, read off the server's own log

`traces/logs/occ-baseline.log`, one call at occupancy 100:

    6:10.283  slot launch_slot_: id 100 | task 4186 | processing task
    6:10.284  srv alloc: - making room for prompt cache entry, removing oldest entry (size = 202.827 MiB)
    ...       (100 such lines, ~37 ms apart)
    6:14.2x   slot ... prompt eval starts
    6:16.258  slot release: id 100 | task 4186 | n_tokens = 1386

Counted over the whole run, the number of eviction lines between one
`launch_slot_` and the next **equals the occupancy exactly**: 0 for occupancy
0–40, 1 at 41, then 42, 43, 44 … 127. The run logs **7,268 evictions** at a
mean entry size of **202.80 MiB** (min 202.37, max 203.43) — about **272 s of
the 543 s run**, which is the half of the wall clock that `timings` cannot see.

The arithmetic closes with no free parameters:

* `--cache-ram` defaults to **8,192 MiB**; one entry is **202.80 MiB**;
  8192 / 202.80 = **40.4**, so 41 entries fit — the measured knee is **42**.
* `--cache-idle-slots` defaults to **enabled** ("save idle slots to the prompt
  cache on new task"). With `--no-kv-unified` the slots are *not* cleared after
  saving, so under R13's never-reuse policy the cache is pure redundancy: every
  window is a distinct document on a virgin slot, and a saved entry can never
  produce a hit.
* Above capacity the access pattern touches every entry each cycle, which is
  the textbook LRU-thrash shape: **cost per request = occupancy × 37.5 ms.**

The model predicts the mean per-call wall of the whole baseline run as
`2.105 + 37.5 ms × mean(occupancy above the knee)` = **4.193 s**. Measured:
**4.233 s.**

## 3. The condition table — one factor at a time

Identical workload everywhere: 128 synthetic same-length documents (1,008 leaf
tokens, 1,339 rendered with the sha256-pinned `leaf-prefix.v1.md`; 622/953 in
the 640 arms), the same question, `temperature 0.3 / top_p 0.9 / seed 1` from
`config.yaml`, one never-reused slot per call, `n_predict 64`. "first/last 16"
are the first and last 16 calls of the run.

| condition | factor moved | np | order | conc | n | wall s, first 16 | wall s, last 16 | ratio | send ms first | send ms last | prefill ms first | prefill ms last | evictions/call, last |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | — (shipped launch line) | 128 | asc | 1 | 128 | 2.18 | **6.69** | **3.07x** | 38 | 4632 | 1391 | 1389 | 119 |
| `cram0` | `--cache-ram 0` | 128 | asc | 1 | 128 | 2.16 | **2.05** | **0.95x** | 3 | 4 | 1391 | 1452 | **0** |
| `nocacheidle` | `--no-cache-idle-slots` | 128 | asc | 1 | 128 | 2.22 | **2.05** | **0.92x** | 4 | 4 | 1436 | 1461 | **0** |
| `sps0` | `-sps 0` (LCP routing off) | 128 | asc | 1 | 128 | 2.23 | **6.69** | **3.00x** | 36 | 4628 | 1439 | 1388 | 119 |
| `shuffle` | slot order shuffled | 128 | shuffle | 1 | 128 | 2.25 | **7.02** | **3.12x** | 38 | 4919 | 1452 | 1383 | 119 |
| `np8` | pool 8, 8 slots in use | 8 | asc | 1 | 8 | 2.21 | 2.21 | 1.00x | 38 | 38 | 1373 | 1373 | 0 |
| `np32` | pool 32, 8 slots in use | 32 | asc | 1 | 8 | 2.21 | 2.21 | 1.00x | 37 | 37 | 1376 | 1376 | 0 |
| `np128` | pool 128, 8 slots in use | 128 | asc | 1 | 8 | 2.23 | 2.23 | 1.00x | 37 | 37 | 1391 | 1391 | 0 |
| `conc8-keepalive20` | `httpx.Limits(keepalive=20)` | 128 | asc | 8 | 32 | 14.32 | 7.46 | 0.52x | 194 | 139 | 4705 | 2956 | 0 |
| `conc8-keepalive1` | `httpx.Limits(keepalive=1)` | 128 | asc | 8 | 32 | 14.38 | 6.88 | 0.48x | 995 | 182 | 4440 | 1916 | 0 |
| `w640` | window 640, shipped cache | 128 | asc | 1 | 128 | 1.62 | **6.52** | **4.03x** | 37 | 4997 | 961 | 954 | 119 |
| `cram0-w640` | window 640 + `--cache-ram 0` | 128 | asc | 1 | 128 | 1.59 | **1.61** | **1.01x** | 3 | 3 | 961 | 1000 | **0** |

The two `conc8-*` rows are here only to close hypothesis 2 and their wall
column does not mean what the others' does: with 8 calls in flight each call's
wall includes the seven it is batched with, and 32 calls never reach the
41-slot knee (0 evictions), so the 0.5× "ratio" is a warm-up artefact, not a
trend. What those two rows are for is the `send_ms` comparison — and what they
also turned up is in §7.

## 4. What was ruled out, and by what

**Hypothesis 1 — slot scanning by longest-common-prefix (`-sps`). DEAD, twice.**
`-sps 0` is indistinguishable from the shipped default: 2.23 → 6.69 s, ratio
3.00× against the baseline's 3.07×, 119 evictions per call at the top band. And
the pool-size control settles it independently: holding **slots-in-use at 8**
and varying the pool over `-np` 8 / 32 / 128 gives `send_ms` **38 / 37 / 37 ms**
— identical. **The effect tracks slots IN USE, never pool size**, which is the
opposite of what a scan linear in slot count predicts. (These calls also pass
`id_slot` explicitly, which bypasses LCP routing outright.)

**Slot INDEX versus CALL ORDINAL. It is occupancy.** The `shuffle` run
decorrelates the two by handing calls slots in a seeded random order:
`send_ms` correlates with the call ordinal — i.e. the number of slots already
in use — at **r = 0.960**, and with the slot index at **r = −0.035**.

**Hypothesis 2 — client-side pooling. DEAD.** `rlm/dispatcher.py` builds
`httpx.AsyncClient(timeout=timeout)` with library defaults
(`max_connections=100`, `max_keepalive_connections=20`) against
`scaffold.dispatch_concurrency: 8`, so production's concurrency is below its
keepalive ceiling and nothing can queue in the client. Measured anyway at
concurrency 8: keepalive 20 versus keepalive 1 changes the first band's
`send_ms` (194 → 995 ms, a real but one-off connection-setup cost that decays
to 139/182 ms) and does not change the shape at all. More decisively, **the
whole primary result is measured with ONE call in flight on ONE connection**,
where client-side queueing is not available as an explanation.

**Hypothesis 3 — allocator / memory pressure. NOT the cause, but visible.**
The server's private bytes climb 35.8 → 57.1 GiB across the baseline run. With
`--cache-ram 0` they climb 34.5 → 47.8 GiB for the same 128 slots — the
**~8.2 GiB difference is the host prompt cache itself**, and removing it
removes both the memory and the time. The remaining ~13 GiB is the per-slot
state R13's `-np 128` already priced. Nothing here is a page-fault or
fragmentation story: the growth is a bounded allocation with a name.

**Hypothesis 4 — KV eviction or defragmentation as the KV pool fills. DEAD.**
`prefill_ms` is the direct observable for it and it is flat (1,391 → 1,389 ms
across the full pool), and the two prompt-cache flags remove the entire effect
while leaving the KV pool configuration untouched.

**Nothing is left unexplained.** The residual after `--cache-ram 0` is **4–5 ms
per call at every occupancy**, against 4,633 ms at the top of the baseline.

## 5. What the fix does not cost

The intra-window re-query — the second question about a window, on that
window's own slot — is the one cache lever R13 leaves intact and §7 #3 (d)
prices at 20–40×. It survives untouched. Eight cold windows on eight virgin
slots, then the same eight documents re-asked on their own slots:

| condition | cold prefill ms | cold `cache_n` | warm prefill ms | warm `cache_n` | warm wall s |
|---|---|---|---|---|---|
| `warm-default` (shipped) | 1,415–1,574 | 0 | **50–55** | 1,331–1,355 | 0.62–0.87 |
| `warm-cram0` (`--cache-ram 0`) | 1,410–1,559 | 0 | **49–56** | 1,331–1,355 | 0.63–0.85 |

`cache_n` is **identical element-for-element** between the two arms, and warm
prefill is 50 ms in both. Switching the host prompt cache off costs the warm
path exactly nothing, which is what the mechanism predicts: with
`--no-kv-unified` the slot keeps its own state and the host copy was never the
thing being reused. (Per §7 #2's R8 instrument warning, `cache_n` is reported
here as an observation, not as a certified token count.)

## 6. THE DECISION — projected per-call wall at 417 windows versus 261

Under R13 the pool drains monotonically and the leaf is rotated when it is
spent, so occupancy is a **sawtooth from 0 to `servers.leaf.parallel` = 128**,
not a ramp across the whole episode. Both questions about a window share that
window's slot, so occupancy is counted in windows. Model:
`wall(occ) = base + (0 if occ ≤ 41 else occ × 37.5 ms)`, with `base` measured
per geometry on the `--cache-ram 0` arms and the two constants measured in §2.
A 200K-token corpus:

| geometry | windows | sub-calls | rotations | peak occupancy | **per-call wall at peak** | mean per-call wall | episode, serial |
|---|---|---|---|---|---|---|---|
| **1024/768 shipped, `-cram 8192`** | 261 | 522 | 2 | 127 | **6.87 s** | 4.19 s (measured 4.23) | 2,189 s |
| **640/480 candidate, `-cram 8192`** | 417 | 834 | 3 | 127 | **6.35 s** | 3.55 s (measured 3.80) | 2,959 s |
| **1024/768 shipped, `--cache-ram 0`** | 261 | 522 | 2 | 127 | **2.11 s** | 2.11 s | 1,099 s |
| **640/480 candidate, `--cache-ram 0`** | 417 | 834 | 3 | 127 | **1.59 s** | 1.59 s | 1,324 s |

Three things follow, and the first is the one that unblocks the geometry edit:

1. **Occupancy does not discriminate between the geometries.** Both drain the
   same 128-slot pool to the same ceiling, and the per-call overhead is set by
   the slot's state size (202.80 MiB at window 1,024, 198.77 MiB at 640 — a 2%
   difference) rather than by the window. At peak occupancy the **640 geometry
   is the CHEAPER call** (6.35 s vs 6.87 s), because its prefill is 963 ms
   against 1,390 ms. The 4.4× that blocked the decision was never a reason to
   prefer 261 windows over 417.
2. **With the host prompt cache off, the geometry costs exactly what §7 #2's
   token arithmetic said it costs.** 834 calls × 1.59 s = 1,324 s against 522 ×
   2.11 = 1,099 s — a **+20% wall for +60% sub-calls**, because the smaller
   window prefills less per call. That is the honest price of 640/480, and it
   is far below what the confounded 4.4× implied.
3. **`max_wall_clock_s: 900` is the next binding constraint, not occupancy.**
   Every serial column above exceeds it. These are one-call-in-flight numbers,
   and §7 shows why they have to be: **concurrent leaf dispatch is not usable
   on this build at all.** That is a separate blocker and it does not block
   `scaffold.chunk`; it blocks the wall-clock re-derivation.

## 7. A SECOND BLOCKER, FOUND BY THE CONTROL: concurrent leaf dispatch destroys the answer

The concurrency conditions exist in this report only to close hypothesis 2.
They closed it, and they turned up something larger. Same corpus, same prompts,
same slots, same sampling, one never-reused slot per call — only the number of
calls in flight moves:

| condition | in flight | n | **correct answers** | median wall s |
|---|---|---|---|---|
| `conc1-ladder` | 1 | 32 | **31/32** | 2.12 |
| `conc2-ladder` | 2 | 32 | **7/32** | 3.14 |
| `conc3-ladder` | 3 | 32 | **3/32** | 3.48 |
| `conc4-ladder` | 4 | 32 | **2/32** | 6.04 |
| `conc8-keepalive20` | 8 | 32 | **1/32** | — |
| `conc8-keepalive1` | 8 | 32 | **1/32** | — |
| `conc8-full` | 8 | 128 | **7/128** | 26.24 |
| `conc8-full-cram0` | 8 | 128 | **10/128** | 13.73 |

The question has a literal answer in the document and the serial arms return it
122–128 times out of 128. Concurrent, the leaf returns degenerate text —
`'###/remainile'`, `'the previous previous previous previous\nThe user
previous previous)////…'`, `'Based/.dexевичandroideutti'` — with `stop_type`
`eos` or `limit` and **no** foreign identifiers. This is not a wrong answer, it
is a broken decode.

**It is the server, not this runner's async client.** Control: one leaf process,
`--cache-ram 0`, and **two independent OS processes** each dispatching serially
(one taking slots 0–7 ascending, the other 127–120 descending), overlapping in
time. Result **0/8 and 5/8 correct** — the corruption appears with two
single-threaded clients that share nothing but the server.

**Consequences.** `scaffold.dispatch_concurrency: 8` is shipped config and is
the value §7 #1 was written around; §7 #3 (e) already found that fan-out costs
more wall than it saves at chunk scale, and this says it also costs the
answers. The break is at **two**, not at eight. Nothing here characterises the
mechanism — the two candidates a next experiment must separate are
continuous batching under `--no-kv-unified` at `-np 128`, and the streamed-abort
pattern both this runner and `rlm/dispatcher.py` use (break on the final SSE
event, close the connection). **Every concurrent leaf measurement in the S2
record needs re-reading against this**, and R13's detector cannot see it: the
corrupted text carries no identifier at all, so a contamination-clean audit is
not an answer-quality audit.

## 8. R13 audit

Every call in this report went to a slot no other call in its run had ever
used, and every reply's `id_slot` was asserted against the requested one.
Across **1,416 calls in 24 conditions**: **0 foreign-identifier detector hits,
0 slot mismatches, 0 errors, 0 NOT CHECKED.** Per §10 R13 a clean verdict is
evidence, never a certificate — and §7 is the sharpest illustration yet of why:
128 calls of degenerate output audit perfectly clean, because degenerate output
contains no identifiers to be foreign. **A stated limit of this run's corpus:
the neutral filler is drawn from one shared sentence pool, so only the per-document
identifier is detectable and prose-level bleed would be invisible.**

Two by-products worth recording, neither of them this experiment's target:

* **Extraction correctness reproduces the instruction-decay result by
  accident.** The synthetic corpus asks one question with an answer literally
  present in the document. At window 1,024 the leaf returned the correct
  identifier in **122/128** calls in every serial arm that ran there (baseline,
  `cram0`, `nocacheidle`, `sps0`, `shuffle` — the same 6 documents fail in all
  five, so it is the document, not the run). At window 640 it returned it in
  **128/128**, twice (`w640`, `cram0-w640`). Same prefix, same sampling, same
  documents, shorter window.
* **The prompt cache is worth 8.2 GiB of host RAM under R13 and buys nothing**,
  because never-reuse guarantees a saved entry can never be hit.

## 9. What to change

1. **Add `--cache-ram 0` to `servers.leaf.extra_flags`.** It is the more
   targeted of the two flags (it removes the allocation as well as the copying)
   and it is measured not to cost the warm re-query. `--no-cache-idle-slots` is
   the equivalent alternative; both were measured, both work.
   The root server (`-np 1`) cannot reach the knee and needs no change, but its
   idle-slot save is equally useless and equally free to switch off.
2. **Then land §7 #2's window 640 / stride 480** and `max_subcalls` 522 → 834.
   This report removes the stated blocker: the occupancy effect is
   geometry-independent, is caused by a default flag, and is gone.
3. **Do not re-derive `max_wall_clock_s` against `dispatch_concurrency: 8`
   until §7 is characterised** — and do not treat 8 as a safe value. On the
   evidence here the only measured-clean dispatch concurrency is **1**. This is
   a blocker of its own and deserves the same treatment R13 got: a minimal
   reproducer, a mechanism, and an upstream issue before any config edit.
4. **Treat every wall-clock number recorded on this box before 2026-08-14 as
   confounded** unless its server ran with the prompt cache off and its
   occupancy is known. That includes every per-arm wall in `milestones/s2/DISTANCE.md`
   (which says so), `milestones/s2/REFUSAL-AB.md`, and the `-np` ladder in
   `milestones/s2/R13-slotcount.md` — the last of which priced `-np 128` on memory and
   aggregate prefill and could not have seen this.
5. **Upstream.** A default `--cache-ram` of 8,192 MiB against a default-enabled
   idle-slot save is a latency cliff for any server whose slot state is large
   and whose slot count is high: capacity in ENTRIES falls as slot state grows,
   and past that capacity the cost per request becomes linear in occupancy with
   no diagnostic other than a `W`-level log line. Worth the same treatment as
   `milestones/s2/R13-upstream-draft.md`.
