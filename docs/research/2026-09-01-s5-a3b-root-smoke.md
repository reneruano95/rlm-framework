# S5 pre-registered row: A3B-as-root — the smoke, the same-model result, and the slot-pool bug it exposed

**Status:** COMPLETE (smoke only — not scored, not the S5 row). Written as running notes during the
run in `runs/s5-a3b-root/` (gitignored) and moved here unchanged except for this header and the
"FIXED" paragraph. Companion files: `2026-09-01-s5-a3b-root-smoke/ledger.jsonl` (every decided cell)
and `smoke-calibration.txt` (the `--smoke` table as printed). The config that produced it is
`config.s5-a3b-root.yaml` at the repo root.

**Date:** 2026-09-01 · **Config:** `config.s5-a3b-root.yaml` (5-line derivative of `config.yaml` @ c0b4e7c: root model → the leaf GGUF, `dflash: false`, root `backend_dir` → plain Vulkan build, root log path, DFlash2 flags dropped) · **Command:** `rlm bench --config config.s5-a3b-root.yaml --smoke --ledger runs/s5-a3b-root/ledger.jsonl --report runs/s5-a3b-root/RESULTS.md` · **Smoke run_id:** `b650df33-e03d-4791-9eed-cfca19575f39` · **Servers:** root 8080 and leaf 8081 both `Qwen3.6-35B-A3B-UD-Q4_K_M`, both llama.cpp `b10375-ba360efe1` Vulkan.

This is the first run in the project's history where all five arms use the same model. Any
difference between arms is the scaffold's, not the model's.

## RESULT — smoke complete, 20/20 cells (4 tasks × 5 arms × 1 seed), 2026-09-01 14:22–17:40

Same model in every arm. From `ledger.jsonl` (final row per cell; one rerun) and `traces/rlm.duckdb`:

| task | rlm | rlm-restricted | b2 | b1 | b3 |
|---|---|---|---|---|---|
| agg-02 (544, int_exact) | **✓ 56 s** | ✗ 541 s (answered 542) | ✗ 810 s | ✗ 347 s | ✗ 691 s |
| needle-02 | **✓ 63 s** | ✓ 183 s (after 1 ERROR, see below) | ✗ 649 s | ✓ 112 s | ✓ 217 s |
| synth-01 | **✓ 43 s** | ✗ 1,259 s `context_exhausted/root_window` | ✓ 457 s | ✓ 34 s | ✓ 13 s |
| codeqa-01 | **✓ 99 s** | ✓ 533 s | ✗ 723 s | ✗ 244 s | ✗ 359 s |
| **tasks passed / 4** | **4** | 2 | 1 | 2 | 2 |
| median wall | ~60 s | ~537 s | ~686 s | ~178 s | ~288 s |

**Leaf calls per episode, from the trace store** (`actor='leaf'`):

| task | rlm | rlm-restricted | b2 | b1 | b3 |
|---|---|---|---|---|---|
| agg-02 | **0** (3 cells) | 348 (299 ok) | 285 | 1 | 1 |
| needle-02 | **0** (5 cells) | 135 (0 ok, killed) → rerun 61 | 140 | 1 | 1 |
| synth-01 | **0** (4 cells) | 343 (42 cells, window drowned) | 59 | 1 | 1 |
| codeqa-01 | **0** (7 cells) | 282 (28 cells) | 224 | 1 | 1 |

**The free root made zero leaf calls in all four tasks and won all four.** With the *same model* on
both ends of `llm_query`, the root still never delegates — exactly what the 27B did across S1, S2
and S4's 90 episodes. The model pair was never the reason. ~~The prompt's brake is the binding
constraint.~~ **CORRECTED 2026-09-02** — that sentence was inference by elimination with zero direct
tests, and the brake review (`2026-09-02-delegation-brake-review.md`) found the binding constraint is
**the tasks**: all 30 v1 tasks fall to a deterministic zero-model-call program (re-verified 30/30),
§8 designed the benchmark to reward code over leaf calls, the root's windows peaked at 10–17% of
32K, and none of its 19 cells so much as mentions `llm_query`. The brake is soft, unmeasured, and
accurate advice for these tasks; whether it holds delegation at zero on readable chunks is n = 0.

**Calibration** (printed by `--smoke`): pre-registered constants project the full 30 × 3 × 5 grid at
**40.9 h**; substituting these measurements gives **50.7 h**. The constants are badly off in both
directions — `rlm` cells ran at 0.05–0.22× projection, `rlm-restricted` at 3–21×.

**What the smoke does and does not establish.** n = 1 seed × 4 tasks; a smoke never scores. Within
that: the scaffold's margin over every baseline **survives the same-model control** on these four
tasks, so at least here it is not a model-swap artifact. Forced delegation with the same model lost to
free code on 3 of 4 tasks and was 5–20× slower on all of them. The A3B is a competent root ("weak
roots write bad Python" did not appear at n = 4).

**agg-02** (answer 544, `int_exact`): the free root ran `re.finditer(r'SEALED', context)` (corrected
2026-09-02 from "`context.count`" — no cell contains `.count(`), cross-checked
with `'Status: SEALED'`, answered 544 in 3 steps and zero leaf calls. The forced-delegation root
made ~330 leaf calls over 283 chunks; per-chunk leaf counts were inconsistent (6, 7, 7, 8, 7, 13…);
it summed to **542**. Off by two, 10× the wall-clock. Same model on both sides.

## FINDING — `slot_pool_error_drained` is misfired by arithmetic, not by a failing server

Episode `dd0783a8-a2ad-4ced-ba0d-b0b4ee613b8e` (needle-02, rlm-restricted, first attempt).

**What happened, from the blobs** (mtimes):
- 15:05:52.31 — step 1 observation: the root's cell raised `TypeError: chunk content is not readable
  in this arm; pass the handle to the sub-model instead: await llm_query(question, chunk=chunks[i])`.
  The root had tried to f-string the chunk handle into its prompt. Recoverable; the message says how.
- 15:06:03.09 → 15:06:04.08 — **136 leaf request blobs written inside one second** (steps 2–137,
  rendered with the leaf system prompt, `layout: chunk_question`, `prefix_tokens: 311`). Zero
  observations follow. The root had corrected itself and fanned out over all 139 chunks with
  `asyncio.gather`.
- Episode outcome: `error`, reason `slot_pool_error_drained`. The bench's §8 one-rerun rule re-ran
  the cell and it **succeeded in 183 s** — so the leaf server was healthy throughout.

**Why, from the code** (`src/rlm/serve/dispatcher.py`, `src/rlm/episode.py`):
1. Slots are consumed at **acquire time**, before any HTTP request is sent: `self.slots.acquire(window)`
   at `dispatcher.py:1517` runs *before* `async with self.semaphore` at `:1586`. With
   `dispatch_concurrency: 1` the sends are serialized, but acquisition is not.
2. `SlotPool.acquire` (`:414-427`) hands out `_next` until `_next >= size` (128), then raises
   `SlotPoolExhausted`.
3. `error_drained` (`:386-401`) is `self.restart_required and not self._answered` — *spent and nothing
   answered*. It does not require any window to have **failed**.
4. On `SlotPoolExhausted`, `episode.py:655-667` checks `pool_error_drained` and, if true, trips
   `Outcome.ERROR` with `SLOT_POOL_ERROR_DRAINED` and refuses rotation.

**Consequence: any fan-out of ≥129 windows on a virgin 128-slot pool dies instantly and
deterministically as "the server is failing", before the server has received a single request.**
`gather` launches all coroutines within milliseconds; the 129th acquire fires while the first HTTP
send is still queued behind the semaphore; `answered` is empty by construction; the episode is killed
and the 128 in-flight calls cancelled. The lifecycle log records `rotation_refused`, the trace says
`slot_pool_error_drained`, and both point at a server that never did anything wrong.

This is at least one concrete mechanism behind R16's *"known worsening `slot_pool_error_drained`
fault"*. Every one of the thirteen heaviest-delegating episodes on record (393–1,067 leaf calls) was a
wide fan-out. The 2026-08-20 smoke note describes a *different* path to the same reason (an inherited
drained pool); this is a second path, on a **virgin** pool, and it needs no prior failure at all.

**What this does and does not say.** It does not say delegation works — agg-02 above says the
opposite for that task. It says that the evidence previously marked *contaminated* was contaminated by
the scaffold's own accounting, and that "delegation quality is unmeasured" (R16) is now explained
rather than merely asserted. The `error_drained` predicate needs a third state — *spent, nothing
answered yet, nothing failed* — that means **wait or rotate**, not **kill**.

**VERIFIED after the smoke released the store** (`traces/rlm.duckdb`, read-only). Episode
`dd0783a8`'s 138 steps: **129 `cancelled`, 7 `error`, 2 `ok`**. All 7 errors carry the same
`error_detail`: *"leaf slot pool exhausted: all 128 slots have held a window, and reusing one is R13's
defect…"*. The one remaining detail is the sandbox death itself: *"bridge closed:
SandboxError('sandbox killed: slot_pool_error_drained')"*. Not one leaf HTTP error. The predicted shape,
exactly: 128 acquired-then-cancelled, the overflow rejected at `acquire`, the episode killed for a
"failing server" that never received a request.

**No code was changed during the run** — fixing it mid-grid would have broken the scaffold-sha pin.
**FIXED AFTER THE SMOKE, TDD, same day.** RED first:
`checks/test_episode.py::test_a_simultaneous_burst_wider_than_a_virgin_pool_rotates_instead_of_dying`
— five windows under one `gather`, pool of two, no stagger — failed against the unmodified scaffold
with exactly the production signature, `('rotation_refused', 'slot_pool_error_drained')`, in 2 s.
GREEN: the `pool_error_drained` judgment moved out of `_dispatch_leaf`'s exhaustion handler and into
`_rotate_leaf`, **inside `rotating()` and therefore after `quiesce()`**, so a generation is judged only
once every in-flight window has answered or failed. The `rotating` lifecycle event now follows the
judgment, so a refused generation never logs one. `test_a_pool_drained_by_errors_is_never_rotated`
(the genuine failed-server case) still passes: 41/41 in `test_episode`, 49/49 rotation-related in
`test_arms`/`test_dispatcher`/`test_delegation_arm`, and the full suite **888 passed** (`pytest -q`,
719 s, including the known-flaky concurrent-fanout test, green on this run).

**SECOND FIX, from the adversarial review of the first (2026-09-02, 8 agents, 4 lenses).** The review
reproduced a should-fix live: a settled window has THREE dispositions — answered, failed, or
**cancelled by its caller** — and a cancelled slot is marked neither answered nor failed. So a pool
spent by the root's own `asyncio.wait_for` timeouts read as "served nothing" and a healthy leaf was
refused as failed, with the same trace signature as the fan-out bug (N cancelled + exhausted + 0 leaf
errors). Pre-existing, not a regression — but the first fix's own docstring claimed "every window has
answered or failed", which was false. RED first:
`test_a_pool_spent_by_the_cells_own_cancellations_rotates_instead_of_dying` (two `wait_for` timeouts
on a pool of two, then a plain call) failed with `('rotation_refused', 'slot_pool_error_drained')`.
GREEN: `SlotPool.error_drained` now requires **at least one window the leaf actually failed** —
`restart_required and not _answered and bool(_failed)`. A third test pins the branch the first fix
created — `test_a_burst_whose_in_flight_windows_all_fail_is_still_refused_after_the_wait`: the judge
waits on two live windows, both fail, the generation is still refused. The burst test's `restarts >= 2`
was tightened to `== 2` (the review showed `>=` lets `_rotate_leaf`'s early-return guard regress
unnoticed) and given the sibling test's `stray == []` guard. 43/43 `test_episode`, 56/56
rotation/pool tests across `test_arms`/`test_dispatcher`/`test_delegation_arm`/`test_cli`.

Review findings left as **notes**, deliberately: (a) the bench arms' mirror `ArmEpisode._dispatch_leaf`
still judges before the quiesce — correct for its callers, which are all serial (B1/B3 one call, B2 a
plain `for`), and its docstring now says so and names the condition under which it must move (a
concurrent `call_leaf` caller); (b) a mixed drain (one answered, the rest failed) still rotates, by
`error_drained`'s documented design — ARCHITECTURE.md:162 states the never-restart rule more
absolutely than the code enforces it; (c) the `raise SlotPoolExhausted` after `_trip` in the refusal
path is unreachable in a live session (the kill's `CancelledError` arrives first) and stays as
belt-and-braces; (d) RED-on-old-code for the burst test is a strong timing bias, not a construction
guarantee — measured 0/22 false-greens by the reviewer.

## Leaf-log capture

`start()` truncates the leaf log on every relaunch (D27), and the restricted arm and B2 both relaunch
it, so the resident leaf log at any moment only shows the *current* generation. A truncation-proof
capture runs alongside: `tail -F traces/logs/leaf-server.log >> runs/s5-a3b-root/leaf-server-capture.log`
(started ~15:06). The B2/needle-02 generation shows 240 successful `print_timing` lines and zero
errors — the server serves.

## Caveats carried into interpretation

- Config-only means the root prompt still tells the root that `llm_query` *"reaches a small, fast,
  stateless model"*. In this run that sentence is false — it is the same model. The A3B root also
  reads the same delegation brake as the 27B did (*"Look before you delegate"*, *"Try code first"*,
  *"Fan out only if the scan fails outright"*).
- n = 1 seed, 4 tasks. A smoke is a calibration pass; it never counts toward scoring.
- The 27B-root single-shot baseline (a same-model control for the *current* config) is still not
  runnable without relaxing `checks/test_config.py:625`; not attempted here.
