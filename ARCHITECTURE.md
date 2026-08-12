# ARCHITECTURE.md — RLM Runtime (working name: `rlm-halo`)

**Spec version:** `rlm-runtime-spec-v0.2` (changelog: §14)
**Status:** Pre-implementation constitution. No code exists. §7 carries community-measured priors (off-box, mid-2026 gfx1151 data — §13); every number is still replaced by S0 on-box measurements.
**Amendment rule:** Invariants (§3) and gates (§9) change only with a version bump and a dated changelog entry. §7 numbers and §4 sizing update in place as measurements land, no bump required.
**Hardware target:** AMD Ryzen AI MAX+ 395 "Strix Halo", 128 GB unified LPDDR5X, 256 GB/s theoretical (~212–215 GB/s measured) bandwidth, Radeon 8060S iGPU (RDNA 3.5, gfx1151).
**Prime directive:** The LLM proposes; the scaffold disposes.

---

## 1. Purpose & Scope

A local Recursive Language Model (RLM) runtime: a deterministic scaffold that lets a *root* LLM inspect and decompose arbitrarily large contexts through a persistent Python REPL, delegating chunk-level work to cheap parallel *leaf* LLM calls. Context is passed **by reference** (a Python variable), never by value (never in a message array).

**This project is:**
- A serving topology (two resident `llama-server` processes with distinct roles).
- A deterministic scaffold (sandbox, dispatcher, budgets, truncation, tracing).
- A frozen benchmark and a measurement discipline.

**This project is not:**
- An agent framework, a chat product, or a general orchestrator.
- A training project. Learning loops are S6, explicitly unscheduled (§9).
- A simulator host. World models never enter the runtime loop (Invariant I3).

Primary reference: Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601. Full reference list: §13.

---

## 2. Thesis

This is Thesis B (author's framework — external reference, §12) applied to context management. Deterministic logic renders all control decisions: truncation caps, recursion budgets, routing, termination, and logging are scaffold configuration. The LLM's only powers are (a) writing code that runs in a sandboxed REPL and (b) requesting sub-calls through a dispatcher it does not control. Ground truth lives in the real REPL and in shared artifacts (traces), never in model output taken at face value.

Corollary for performance: on this hardware the binding constraint is **leaf prefill throughput** (compute-bound on the iGPU), not root decode speed. The architecture optimizes for many cheap parallel leaf calls over one smart slow context window.

---

## 3. Invariants

Violating any of these is a bug, regardless of benchmark results.

- **I1 — The LLM proposes; the scaffold disposes.** Truncation caps, budgets, routing, and termination live in `config.yaml` and scaffold code. No model output, prompt content, or REPL side effect may alter them at runtime.
- **I2 — Context by reference.** The full context never enters any model's message array. Models see only scaffold-truncated views (`print` output capped by C3).
- **I3 — Real environment only in the runtime loop.** No simulator or world model may sit between the root and the REPL, or between the scaffold and the final answer. World models are lifecycle tools (training data generation, S6) — never runtime components.
- **I4 — Every episode is a trajectory.** All runs are logged as `(state, action, observation)` steps with a terminal outcome (§6; the state-reconstruction rule is stated there). A run that is not logged did not happen.
- **I5 — Quality gates precede performance work.** No optimization is merged without the frozen benchmark (§8) showing task success unchanged or better. Tokens/sec alone never justifies a change.
- **I6 — Models are config.** Model identity, quant, context sizes, ports, and budgets live in one `config.yaml`. Swapping a model must require zero code changes (S5 gate).
- **I7 — Evaluate only against the real REPL.** Future training (S6) may consume synthetic or simulated environments; evaluation gates may not.

---

## 4. Serving Topology (Capa 0)

*("Capa" = layer: Capa 0 serving → 1 scaffold → 2 performance → 3 measurement. Each layer may depend only on the ones below it.)*

Two `llama-server` processes, both resident simultaneously. Not LM Studio — we need `--parallel`, continuous batching, explicit KV cache types, and prompt-cache control.

| | **ROOT** | **LEAF** |
|---|---|---|
| Model | Qwen3.6-27B — dense FFN, **hybrid attention** (16/64 layers carry KV, rest Gated DeltaNet; multimodal, mmproj unused) | Qwen3.6-35B-A3B — MoE ~3B active, **hybrid attention** (10/40 KV layers) |
| Quant | Q4_K_M (16.8 GB; 17.1 GB MTP — multi-token prediction — variant) | UD-Q4_K_M (~22 GB) |
| Port | 8080 | 8081 |
| Context | **32K per slot, hard cap** | per-slot: `chunk_size + overhead` (40K default) → `-np 8 -c 327680 --no-kv-unified` |
| Parallel | `--parallel 1` — 2nd slot reserved for Q3 self-calls only (mechanics below the table) | `--parallel 8` (continuous batching is default-on; keep `--cont-batching` for explicitness) |
| KV cache | q8_0 K+V, `-fa on` pinned | q8_0 K+V, `-fa on` pinned |
| Backend | Vulkan (Linux: Mesa RADV; Windows: AMD proprietary driver) — decode-bound role | ROCm/HIP + flash attention (prefill-bound role; flags per §7) |
| Variant | MTP GGUF — adopt only per the R4 criteria (§10) | standard |
| Role | plan, write REPL code, decide sub-calls, emit final | chunk-level extraction/summary/QA |

Quantized V-cache hard-requires flash attention; `-fa on` is pinned (not `auto`) so a failed FA probe fails loudly at startup instead of silently changing memory and performance behavior. Whether q8_0 KV actually engages on this hybrid architecture is an S0 check (bf16 KV is the fallback). Root 2nd-slot mechanics: enabling it requires `-c 65536` (llama-server splits `-c` across slots) plus re-priced KV, and is incompatible with the MTP variant, which is single-slot. The physical batch flags `-ub`/`-b` are pinned per server, recorded in `config_snapshot`, and stated next to every S0 number — they shape the concurrency-scaling and contention measurements, and a silent upstream default change (R11) must not move the baseline.

**Startup handshake** (scaffold entrypoint, ahead of C4): before accepting any task, the scaffold queries each server's `/props` and asserts model path, `n_ctx`, `n_parallel`, and cache types against `config.yaml`; any mismatch refuses to start. The `/props` responses (including server build) are recorded into `config_snapshot` (§6). The same assertion re-runs per episode at the C5 quiesce point — a server that crashed and relaunched with different flags mid-benchmark must be caught before it poisons a multi-hour run, not after.

**Why the root gets only 32K:** the root never sees the raw context (I2). It sees truncated prints and sub-call results. A large root window is wasted KV and slower prefill for zero benefit. If the root "needs" more context, that is a scaffold smell — the work belongs in the REPL or in leaves. The cap is enforced, not assumed: the root server runs with context shift disabled, C5 tracks the root conversation from server-reported usage, and crossing 90% of the window ends the episode deterministically as `context_exhausted` (§6). A multi-turn flail loop can fill 32K well inside the wall clock; overflow must be an outcome, never an accident.

**Chunk size is the parallelism lever.** Leaf slots are sized to the chunk budget, not to the model's 262K native maximum. Corrected for the hybrid architecture (only 10/40 leaf layers carry per-token KV): a 32K chunk costs ≈0.35 GB of KV at q8_0; the configured 40K slot (chunk + overhead) ≈0.45 GB, so 8 slots ≈ 3.5 GB versus ≈22 GB for 8 full-window slots (~12 vs ~90 GB on a classic all-layers-KV model). On this architecture KV memory alone no longer forces chunk-sizing — the binding rationale is per-call prefill latency and scheduling: a slot holding one chunk turns around in one chunk-prefill time, keeps the fan-out tail short, and makes per-call cost uniform. **Pre-registered S0 expectation: ≈0.45 GB per configured 40K slot (≈0.35 GB of that is the 32K chunk). A measurement near 1.5 GB/slot signals misconfiguration (dense-sized allocation, or KV quantization not engaging on the hybrid arch) — not confirmation.**

**Memory budget (128 GB total):**

| Item | GB |
|---|---|
| Root weights (Q4_K_M; 17.1 with MTP head) | 17 |
| Leaf weights (UD-Q4_K_M) | 22 |
| Leaf KV (8 × 40K slots, q8_0, 10/40 KV layers) + recurrent state | ~4 |
| Root KV (32K q8_0, 16/64 KV layers ≈ 1.1) + runtime buffers | ~3 |
| OS + scaffold + DuckDB | ~8 |
| **Committed** | **~54** |
| **Headroom** | **~74** |

Headroom is deliberate. It exists for: (a) a third resident server if S6 ever needs one, (b) A/B running a candidate model next to the incumbent during S5 swaps, (c) larger chunk experiments, (d) B1/B3's dedicated 256K single-slot profile (§8; ~2.8 GB KV, run non-concurrently anyway). Do not spend it by default.

**Host config (Linux path — see A2):** BIOS UMA carve-out at 512 MB; GTT raised via `ttm.pages_limit` / `ttm.page_pool_size` (~100 GiB; `amdgpu.gttsize` is deprecated on kernels ≥ 6.18); `--no-mmap` on both servers so weights land in GTT predictably.

**Prefix & cache contract (critical):** all leaf calls share one **byte-identical** system prefix. No timestamps, run IDs, task IDs, or counters anywhere in the prefix — run metadata travels in the trailing user segment or stays out-of-band in the trace. Prompt layout is fixed: `[system prefix][chunk][question]` — question **last**, so a re-queried chunk extends the cached prefix instead of invalidating it at token 0. Reuse mechanics (llama.cpp): the prompt cache is **per-slot** — there is no cross-slot sharing, so the prefix cold-prefills up to `--parallel` times before steady state; slot routing is longest-common-prefix (`--slot-prompt-similarity`), and `slot_id` is recorded per call (§6) to verify affinity. The monitored metric is **token-weighted** (`steps.tokens_cached`, from the server's `timings.cache_n`), never a boolean — see §7 #3 for the targets and the structural ceiling on unique-chunk calls. R8 applies: the hybrid Gated-DeltaNet architecture reuses cache through context checkpoints, with documented llama.cpp regressions — cache behavior is verified in S0, not assumed, **on both servers**: the root is the same hybrid architecture, and its turn-by-turn conversation growth is exactly the checkpoint-extend pattern the R8 issues describe (§7 #3c).

---

## 5. Scaffold Components (Capa 1)

Six numbered components — each lists what it must NOT do; those are the load-bearing lines — plus the section-level contracts that bind them (bridge, registry, dry-run, config schema, CLI, logging, testing). Python 3.12+ (assumption A1, §12).

**C1 — SandboxManager.** Spawns one isolated Python interpreter per episode (§6: one run of a task; `subprocess` + resource limits: CPU time, memory, no network by default). "No network" means **no AF_INET/AF_INET6** (Linux: network namespace; Windows: best-effort — AppContainer preferred, else WFP/firewall rules — documented as weaker, A2). Persistent across REPL turns within an episode (Jupyter-like: variables survive). Killed unconditionally at episode end or budget breach.
*Must not:* share state between episodes; grant network unless the task config explicitly enables it.

**C2 — ContextLoader.** Materializes the input as variable `context` inside the sandbox before the root's first turn. Handles str, bytes, file paths, and lists of documents. Also owns the **deterministic chunker**: `chunks = split(context)` (chunk size from config, measured in target-leaf tokens via the leaf server's `/tokenize`), injected read-only into the sandbox. **Cut rule:** fixed target size from config, cut point snapped to the nearest paragraph/newline/code-block boundary within a ±10% token tolerance, deterministic tie-break (earliest boundary). The rule is identical across the §7 #2 sweep (the sweep varies target size only); snap ships on/off behind one config flag and is A/B'd on a non-benchmark fixture before the S2 freeze. Full content-adaptive sizing is rejected (§11). The root chunks **only** via this utility — free-form chunking in model code would make `chunk_size` advisory (a soft I1 violation) and render the §7 #2 sweep uncontrolled. B2 and B3 (§8) use the same chunker verbatim.
*Must not:* place any part of the raw context into any message array (I2); allow model code to redefine or bypass the chunker.

**C3 — OutputTruncator.** Hard cap on all REPL output returned to the root (default 2,000 chars, config). `observation_view` is the ordered, labeled concatenation of stdout, stderr, last-expression repr, and formatted traceback, truncated **as one unit**. Appends a marker: `[truncated: showing 2000 of 184,203 chars]`.
*Must not:* be overridable by model output or by code running in the sandbox (I1). The cap is applied scaffold-side, after execution.

**C4 — LLMDispatcher.** Exposes `await llm_query(prompt, role="leaf")` inside the sandbox. Routes by role to the correct server. Owns an `asyncio.Semaphore` equal to the target server's `--parallel`. **Pre-flight check:** token count via the target server's `/tokenize`; prompts exceeding slot capacity are rejected without dispatch and logged as steps with `status=rejected`. **Retries:** every attempt is a logged step (shared `call_id`, incrementing `retry_idx`, per-attempt `status`); a retried call counts **once** against `max_subcalls`; every attempt's tokens count against `max_total_tokens`. Defaults (config): max_attempts 3, backoff 1 s / 4 s, per-call timeout 240 s. Requests are streamed so that cancellation (client disconnect) aborts server-side generation. **Server death:** retries exhausting against a dead/unreachable server → step `status=error` → episode `outcome=error, outcome_reason=server_unreachable`; the scaffold never restarts servers mid-episode. Returns the result as a Python object in the caller's REPL — never auto-injected into the root's messages.
*Must not:* let the model choose servers, ports, or semaphore size; exceed `--parallel` (queueing upstream hides latency pathologies from the trace).

**The C1/C4 bridge.** `llm_query` is a scaffold-injected stub, not importable library code: it serializes requests over a scaffold-owned AF_UNIX socketpair (Linux; duplex pipe on Windows) passed at spawn — the only channel crossing the sandbox boundary, and explicitly exempt from the no-network rule (which bans AF_INET/AF_INET6, not the scaffold's own pipe). Semaphore, routing, retries, timeouts, and step logging all execute **scaffold-side**; nothing the model runs in the sandbox can alter them (I1). The sandbox REPL compiles cells with top-level `await` against a persistent event loop so `await llm_query(...)` is natural. **The name `llm_query` is load-bearing:** it matches the RLM paper harness API (alexzhang13/rlm) that RLM-post-trained roots are conditioned on (S5 candidate track) — do not rename casually.

**C5 — BudgetEnforcer.** Per-episode limits from config. Termination is deterministic and unconditional.
*Budgets:* `max_depth` (default **1** — root + leaves, no grandchildren); `max_subcalls` (default 32); `max_wall_clock` (default 15 min; re-derived per task-size class after S0: ≈2–3× predicted prefill time at measured t/s); `max_total_tokens` (default 1.5M); `max_predict` per call, **per role** (root 1024; leaf 512 default, re-derived from the S1 answer-length distribution — a 1024-token reservation against ~100-token extraction answers inflates admission accounting by up to ~30K phantom tokens across 32 subcalls).
*Root window:* accounting from server-reported usage per turn; at ≥90% of the root window the episode terminates deterministically as `context_exhausted`.
*Admission:* the token budget is enforced at dispatch admission — `running_total + reservation ≤ cap`, reservation = pre-flight prompt tokens + role `max_predict`; worst-case overshoot is bounded by in-flight calls × `max_predict` and documented.
*On breach:* cancel all in-flight dispatcher tasks (closing streamed connections aborts server-side work), release the semaphore, kill the sandbox, mark the outcome, persist partial state and full trace; cancelled calls are logged as steps with `status=cancelled`. Operator Ctrl-C routes through this same path (`outcome_reason=operator_abort`) — never dies mid-write.
*Quiesce:* between timed episodes the scaffold verifies both servers report all slots idle before starting the clock, and re-runs the `/props` assertion (§4).
*Must not:* warn-and-continue; accept model requests for extensions. Raising `max_depth` above 1 requires S4 evidence that depth-1 is the binding constraint.

**C6 — TraceLogger.** Single writer task fed by an asyncio queue; **one transaction commit per step** — this is what "a crash loses at most the current step" means — with blobs written before their referencing row. `step_idx` is assigned in commit order; causality is carried by `parent_step_idx`/`call_id`, never by `step_idx` adjacency. Blobs go to a per-episode directory, referenced by episode-relative paths. Monitoring reads happen in-process (DuckDB is single-writer; external readers get an append-only export, not a live handle).
*Must not:* sample, summarize, or truncate the stored record (store full observation as a blob ref; only the *root's view* is truncated, per C3).

**Lifecycle log (narrow, and not a second truth).** Operator logs are forbidden as a channel for episode data — the trace store is the sole episode record (I4). One JSONL lifecycle log (stderr + file) exists for exactly the events the trace store structurally cannot record: its own write failures, config-validation refusals, startup/per-episode handshake refusals, server health transitions, quiesce waits, and recovery actions at restart. Nothing else. The S3 reconstructability gate runs with this log deleted, so two-sources-of-truth drift is structurally impossible.

**Prompt registry.** Root and leaf system prompts live as files under `prompts/` with monotonically versioned names (`root.v3.md`), referenced from config by path and pinned by sha256; inline prompt strings in config or scaffold code are forbidden. Each file opens with a one-line-per-version changelog header. **Strategy templates:** a config-owned library of five strategy blocks (needle: scan-with-code-then-leaf-verify; aggregation: full-coverage map over `chunks` with code-side reduction; synthesis: per-document leaf briefs then cited merge; code-QA; plus a generic default for ad-hoc tasks). The task's **declared category selects the block deterministically — the model never chooses** (I1). Adversarial-context tasks (§8) declare the category of their underlying shape — the injection lives in the corpus, not the category; R12 hardening (the evidence-span check) lives in all extraction-shaped templates. The selected template is appended to the root system prompt; its sha256 goes into `config_snapshot`. Templates for extraction-shaped categories include the evidence-span check (R5) and a REPL-prescan tip (regex/keyword scan of `context` before spending leaf calls — the paper's models did this unprompted). Authoring happens in S1 against non-benchmark fixtures; the library freezes at the end of S2 together with the benchmark, under the same no-overfit rule. Rationale: the strongest verified no-training lever against R1 — Prime Intellect showed a strategy hint flips harness results for untrained models, and minRLM's code-first scaffold (§13; self-reported, no component ablation) is directionally consistent: +22.3pp for a mini-class untrained model, −9.5pp for a nano-class one that cannot write reliable code.

**Dry-run mode.** `dispatcher: mock` in config runs a full episode with canned responses from a fixtures file keyed by (role, prompt-hash) — real sandbox, real C1/C2/C3/C5/C6. Episodes in this mode carry `dry_run=true` (§6) and `config_snapshot.dispatcher="mock"`; the bench runner and every scoring query refuse dry-run rows, keeping I7 airtight. The mock is canned-response-only — any "leaf simulation" realism would drift into I3 territory.

**Config schema.** `config.yaml` is a pydantic model with `extra="forbid"` (a typo'd `max_subcals:` silently running at defaults is an I1-grade hazard) and cross-field validators encoding the §4/§5 arithmetic: leaf slot ctx == chunk_size + overhead; root `-c` == root window; C4 semaphore == leaf `--parallel`; reservation ≤ slot capacity; MTP variant ⇒ root single-slot. `config_snapshot` is the canonical JSON dump of the validated model (stable field order ⇒ stable hashing). Sketch: Appendix A.

**Operator surface (CLI).** The only entrypoint, five verbs: `rlm validate` (config schema + `/props` probe), `rlm run <task-file>` (one episode; prints episode_id + outcome), `rlm replay <episode-id>` (§6 replay check + transcript render), `rlm bench --arm {rlm,b1,b2,b3} --seeds 1,2,3` (the frozen §8 protocol, including blocked scheduling and quiesce), `rlm export <run-id>` (run-id = one `rlm bench` invocation, stamped into every episode's `config_snapshot`; a single episode-id is also accepted; self-contained bundle: parquet export + blob directory + config snapshots — the external-reader export C6 promises, and the concrete mechanism behind R1's publish-a-negative-result commitment). **Non-goals, written down:** no daemon, no REST API, no web UI, no interactive chat mode.

**Dependency rule (lint-enforced, same pattern as Conduo — external reference, §12):** C1–C3, C5, C6 must not import C4 or any LLM client. Only C4 talks to model servers. The scaffold must be able to run a full episode with a mock dispatcher (used in S3's runaway test).

**Testing:** every C-component ships unit tests against the mock dispatcher (pytest, A1); each slice gate additionally requires the component suite green. **Property-based suites (hypothesis) are mandatory for the two components that carry I1/I2:** C3 — over arbitrary (stdout, stderr, repr, traceback) tuples including pathological unicode, NULs, and megabyte single lines: view ≤ cap always, truncation applied to the concatenated unit (never per-stream), marker counts accurate and the marker itself never truncated, deterministic; C5 — a stateful machine driving the mock dispatcher through arbitrary interleavings of dispatch/complete/retry/cancel/breach, with invariants: admitted reservations never exceed the cap, overshoot ≤ in-flight × max_predict, no admissions after breach, termination always reached, a retried call counts once against `max_subcalls` while every attempt's tokens count against `max_total_tokens`. Hypothesis profiles/seeds pinned in CI.

---

## 6. Data Layer — Trajectory Schema

The one decision made today that pays later: traces are stored **trajectory-shaped**, not log-shaped. The tuple `(state, action, observation, outcome)` is simultaneously the SFT format of the original RLM paper, the training format of Qwen-AgentWorld, and the raw material for a Voyager-style skill library. Zero runtime cost; three learning paths kept open.

**The state rule (discharges I4):** "state" is not a column. Root state at step *k* := deterministic replay of steps 0..*k−1*'s `(action, observation_view)` under the versioned system prompt (prompt registry, §5). The schema stores everything that replay needs — nothing else is required. **The rule has an instrument:** `steps.root_view_hash` stores, for every root turn, the sha256 of the exact chat-template-rendered request sent to the root server (canonical serialization pinned — the rendered string, not the message list). `rlm replay` re-derives the message array at every step from the trace alone, recomputes the hash, and asserts equality; this is the S3 gate check and the standing regression canary for prompt-assembly drift.

**The final-answer channel:** `final` is emitted only via the scaffold-injected `final_answer(value)` inside the sandbox — never parsed from root prose. (A prose-parsed answer would be the one channel able to smuggle untruncated context past C3, violating I2.) The value is stored to a blob; the call is the episode's terminal step.

**Terminology:** episode = one run of a task; `task_id` is stable across seeds and arms; `task_hash` = sha256 of the instruction text (corpus documents hashed separately, in the benchmark manifest referenced by `benchmark_version`).

**`episodes`**

| column | type | notes |
|---|---|---|
| episode_id | UUID | |
| task_id | TEXT | benchmark task or ad-hoc |
| task_hash | TEXT | sha256 of instruction text |
| tokenized_task_len | INT | via leaf `/tokenize`; identifies the B1-infeasible subset (§8; B3 never overflows by construction) |
| started_at / ended_at | TIMESTAMP | |
| outcome | ENUM | `success` \| `fail` \| `budget_kill` \| `context_exhausted` \| `error` |
| outcome_reason | TEXT | conventions: which budget breached; `no_final_emitted`; `operator_abort`; `orphaned_at_recovery`; `server_unreachable` |
| final_answer_ref | TEXT | parquet blob path (episode-relative) |
| dry_run | BOOL | mock-dispatcher episodes (§5); refused by all scoring queries |
| scaffold_instance_id | TEXT | which scaffold process wrote this episode |
| sandbox_pid | INT | for recovery reaping |
| superseded_by | UUID | nullable; the rerun that replaced this errored episode (§8 rerun rule) |
| avg_power_w / energy_j | REAL | nullable; 1 Hz package-power integral (§8 cost scorecard; enabled only if the S0 overhead check passes) |
| pkg_temp_c_start / pkg_temp_c_end | REAL | R9 per-episode temperature record; one-shot reads, not gated on the power-overhead check |
| config_snapshot | JSON | canonical dump of the validated config model (§5): models, quants, budgets, chunk size, truncation cap; per-role sampling (temperature/top_p/seed); sha256 of every prompt/prefix/strategy-template text; llama.cpp build (sha, backend, compile + launch flags incl. `-ub`/`-b`) per server; chat-template hash; `/props` responses; dispatcher real/mock |
| scaffold_git_sha | TEXT | |
| benchmark_version | TEXT | nullable |

**Outcome semantics:** `success` = final emitted and checker passes. `fail` = final emitted and checker fails, **or** root ends without emitting final (`outcome_reason` distinguishes). `budget_kill` = any C5 budget breach, including wall clock. `context_exhausted` = root-window accounting breach (C5). `error` = scaffold/server fault with no final.

**Crash recovery (tombstone, never resume):** on startup the scaffold scans for episodes with NULL `outcome`, kills any surviving `sandbox_pid`, waits for both servers to report all slots idle (draining orphaned generation), then tombstones: `outcome=error, outcome_reason=orphaned_at_recovery`. Resume is rejected as unsound — the sandbox interpreter heap is not stored, and §8's own caveat says continuous batching breaks bitwise reproducibility at fixed seed, so a resumed episode would violate the state rule. Benchmark treatment of `error` episodes is pre-registered in §8 (rerun-once, `superseded_by`).

**`steps`**

| column | type | notes |
|---|---|---|
| episode_id | UUID | FK |
| step_idx | INT | assigned by C6's single writer in commit order |
| parent_step_idx | INT | nullable; the `repl_exec` that spawned this call |
| call_id | UUID | stable across retries of one logical call |
| retry_idx | INT | 0 for first attempt |
| depth | INT | 0 = root turn, 1 = leaf; representable for Q3 self-calls |
| actor | ENUM | `root` \| `leaf` |
| action_type | ENUM | `repl_exec` \| `llm_call` \| `final` |
| status | ENUM | `ok` \| `error` \| `timeout` \| `cancelled` \| `rejected` |
| error_detail | TEXT | nullable |
| action_payload | TEXT | code or prompt (full) |
| root_view_hash | TEXT | root turns only: sha256 of the exact rendered request at dispatch (state-rule instrument) |
| observation_view | TEXT | what the root actually saw (post-C3) |
| observation_full_ref | TEXT | parquet blob path (pre-truncation, episode-relative) |
| tokens_in / tokens_out | INT | server-reported actuals |
| tokens_cached | INT | server `timings.cache_n`; hit ratio derived as cache_n/(cache_n+prompt_n) — never stored as a boolean |
| slot_id | INT | server slot that served the call (cache-affinity debugging) |
| t_dispatch / t_first_byte / t_end | TIMESTAMP | t_dispatch/t_end on every step (repl_exec durations and the §8 time split derive from these); t_first_byte on llm_call rows |
| latency_queue_ms | INT | dispatcher-side: (t_first_byte − t_dispatch) − timings.prompt_ms — llama-server does not report per-request queue wait |
| latency_prefill_ms / latency_decode_ms | INT | server `timings.prompt_ms` / `timings.predicted_ms` |
| cache_hit | — | **removed in v0.1** — replaced by `tokens_cached` |

Both `observation_view` and `observation_full_ref` are stored deliberately: the first reconstructs what the root knew (debugging, SFT); the second preserves ground truth (evaluation, world-model data).

---

## 7. Performance Model & Optimization Order (Capa 2)

**Numbers below are community-measured priors (off-box, mid-2026 gfx1151 data — §13). S0 replaces them with on-box measurements; this section is then updated in place.**

Decode ceiling (bandwidth-bound): `t/s ≈ BW_eff ÷ (active_params × bytes/param)`, with BW_eff ≈ 212 GB/s measured (256 theoretical).
- Root (27B dense-FFN, Q4): theoretical ceiling ~15 t/s, effective ~12.5. Community-measured: ~12 t/s fresh, 9–12 at 32K depth. Acceptable: root emits code, not prose.
- Leaf (~3B active, Q4): single-stream 50–60 t/s fresh, 44–49 at 32K depth. Aggregate decode across 8 continuous-batching slots is unmeasured — S0 measures it.

Prefill is compute-bound and is the real budget: a 32K-token leaf chunk is the unit cost of everything. Community-measured (Linux) leaf prefill at d=32K: 807 t/s (ROCm+rocWMMA FA) / 716 t/s (Vulkan RADV). The oft-cited 319 t/s AMDVLK figure is historical only — AMDVLK was discontinued Sept 2025 and is not a candidate.

**Contention model:** root decode and leaf prefill share one iGPU. Working assumption is phase alternation (root decodes → awaits fan-out → leaves prefill); S0 measures the contended case explicitly (root decode during an active 8-slot leaf prefill), and the R4/MTP comparison must state its contention condition.

**Backend is a per-server launch choice, not a late optimization** (both backends ship prebuilt, on Linux and Windows): leaf on ROCm/HIP with flash attention (prefill-bound; Linux datasets put ROCm ahead of RADV by +13% at d=32K — 807 vs 716 t/s — and by +41% at pp16K in a second dataset; the spread is build- and depth-dependent, which is why the pair is A/B'd on-box), root on Vulkan (decode-bound; Linux RADV +5–19% decode vs ROCm). Flag caveat (v0.1.1): rocWMMA flash-attention is reported *slower* than plain HIP FA at long context since ROCm 7.0.2+ — the S0 A/B uses the current known-good ROCm flags (`GGML_HIP_NO_VMM=ON`, `GGML_HIP_MMQ_MFMA=ON`, runtime `ROCBLAS_USE_HIPBLASLT=1`) rather than assuming rocWMMA. All cited numbers are Linux (Mesa RADV); the Windows pairings (ROCm vs AMD proprietary Vulkan) are publicly unmeasured. Chosen and pinned at S0.

Optimizations in strict order of expected return. Per I5, each ships only with benchmark-neutral-or-better evidence, and each has one owning metric:

1. **Parallel dispatch** — metric: measured speedup on the 8-leaf fan-out fixture vs serial. Target: **≥80% of the aggregate prefill scaling S0 measures at 8 slots.** (An a-priori "3×" was ungrounded: the achievable factor depends on batch-1 compute headroom, which S0 measures at 1/2/4/8 slots.)
2. **Chunk-size tuning** — metric: wall-clock per task at fixed quality, swept over {16K, 24K, 32K, 48K}. Note: at 16K, one full pass over a 1M-char task consumes about half of `max_subcalls` — sweep with budgets re-derived per §9 S0. **Locked until after the S4 verdict (§8 chunk-size lock);** a post-S4 direction-only sensitivity annex (16K/48K, 8-task stratified subset, seed 1) precedes the full sweep.
3. **Prefix caching** — metrics token-weighted (`steps.tokens_cached`; never a boolean):
   **(a) Prefix integrity (hygiene, always on):** `tokens_cached ≥ tokenized-prefix-length` on every warm-slot call. This is the R3 drift detector.
   **(b) Repeated-chunk reuse (the win case):** >80% token reuse when re-querying a chunk already resident in a slot. Requires the §4 prompt layout (chunk before question) and slot affinity. Contingency design (built only if S2 gate (b) fails under LCP routing): C4 keeps a per-episode map {chunk_hash → slot_id, last_used}; re-queries of a mapped chunk pass `id_slot` explicitly; new chunks take the least-recently-used slot and overwrite its entry (mirroring the server's one-prompt-per-slot cache). Diagnostic: warm-hit rate on re-queries (`tokens_cached ≥ chunk+prefix length`, grouped by whether the dispatcher predicted residency) — separates LCP mis-routing from R8 checkpoint invalidation in one query. Caveat, documented: pinning serializes concurrent calls targeting one slot; queue-latency spikes there are expected, not pathological.
   **(c) Root-turn integrity (always on):** every root turn expects `tokens_cached ≥` the prior conversation's token count — only the new observation should prefill, because the turn extends the slot's resident cache. Violations are flagged per episode. The root is the same hybrid architecture; a silent incremental-reuse failure here re-prefills the whole conversation every turn and reads as "root is slow" forever (R8 applies to port 8080 too).
   Structural ceiling for single-pass unique-chunk calls is `prefix_len / prompt_len` (a few %) — for pure map-reduce episodes this optimization **cannot** be the largest win; it earns its rank only where the root re-queries chunks. R8 applies (hybrid-arch cache fragility, verified in S0 on both servers).
4. **MTP on the root** — metric: root decode t/s (target ≥1.4×) *and* unchanged benchmark success (R4). Prerequisites: llama.cpp ≥ the 2026-05 MTP merge (PR #22673), `--spec-type draft-mtp`, draft-n ≤ 2, root at `--parallel 1` (MTP is single-slot today), measured under the stated contention condition. Ordered after chunk tuning deliberately: MTP is beta and single-slot (R4); chunk tuning is pure config.
5. **Build-flag fine-tuning** (FA-path variants — plain HIP FA vs rocWMMA — and per-backend flags incl. `-ub`/`-b`) — metric: raw prefill t/s. Residual after the S0 backend split; done last because it is fiddly and orthogonal.

---

## 8. Measurement (Capa 3)

**Primary metric:** wall-clock per task at fixed quality. Never t/s in isolation (I5).

**Secondary metrics:** leaf prefill t/s; root decode t/s (contended and uncontended); leaf `tokens_cached` ratio; per-episode time split (REPL exec / LLM wait / scaffold overhead) — all derivable from the `steps` table, by design. Package temperature logged per episode (R9).

**Sampling & seeds:** per-role `temperature`, `top_p`, `seed` live in `config.yaml` and are recorded in `config_snapshot`. "3 seeds" = sampling seeds {1, 2, 3} per task per arm. Stated caveat: continuous batching means a fixed seed does not guarantee bitwise reproducibility — seeds pin sampling identity, not numerics.

**Frozen benchmark:** the task count is chosen at freeze by a pre-registered rule and fixed thereafter (the scoring denominator depends on it): **20 tasks by default; 30 tasks (margin scaled to +3) if S0-derived cost projections keep a full S4 (4 arms × 3 seeds × N tasks + escalation reserve) under the wall-budget of 60 h — exact, fixed now, not restated after S0.** Programmatically verifiable (exact match with stated normalization, or checker function). **Authoring is an explicit S2 deliverable:** each task ships as (input ref, question, checker fn, expected answer) with unit-tested checkers and pinned corpus sources (repos at a fixed commit for code QA). Frozen and versioned at the **end of S2**; S2's own gates run on dedicated non-benchmark fixtures so S2 cannot overfit the benchmark it is authoring.

*Contamination & checker-validity preconditions (must pass before freeze):*
1. **Closed-book probe:** every benchmark question runs context-free against both the root and leaf models, 3 seeds each; any task answered correctly in ≥1/3 seeds without the corpus is rewritten or replaced (memorized answers differentially inflate the single-shot arms — B1/B3 lean hardest on parametric knowledge).
2. **Synthetic needles:** needle tasks embed generated facts (random entity–UUID pairings) that cannot exist in any training corpus.
3. **Corpus dating:** pinned commits/documents post-date the models' training cutoff where feasible; the benchmark manifest records each corpus date against the assumed cutoff.
4. **Checker near-miss suite:** each checker's unit tests include ≥3 authored plausible-but-wrong answers that must fail, plus normalization edge cases (permissive checkers convert R5 confabulation into false passes in every arm).

Categories:
- Needle retrieval in 200K–1M chars.
- Aggregation/counting across the full context (forces coverage, punishes sampling). **Rule:** at least one aggregation task must defeat deterministic string matching — requiring semantic judgment per item, verified at authoring by showing a pure-regex solution scores at chance — and at least one must be regex-solvable, so the benchmark rewards the root choosing code over leaf calls when code suffices. This keeps the S4 signal decomposable into root-as-programmer vs root-as-orchestrator-of-leaves, and stops strategy templates from quietly turning the category REPL-only while the leaf-reliability surface (R5, R12) ships unmeasured.
- Multi-document synthesis with checkable claims.
- Repo-level code QA.
- At least one adversarial-context task (R12: injection-shaped text in the corpus).

**Scoring & inference (pre-registered; changing any of this after runs exist is p-hacking):**
- **Primary artifact:** the per-task × per-arm × per-seed outcome grid, not two aggregate rates.
- **Decision rule (unchanged in kind):** a task passes for an arm if ≥2/3 seeds pass; success rate = tasks passed / N; "beats" = margin of **+2 tasks (N=20) or +3 tasks (N=30)** against each baseline. A tie with any baseline fails the S4 gate; a tie-or-loss to B2 additionally triggers the pivot-to-B2 rule; a tie-or-loss to B3 records a pivot-to-RAG finding of the same standing.
- **Inference layer (mandatory in the verdict, decision rule unchanged):** for each RLM-vs-baseline pair, the exact sign (McNemar) test over discordant tasks, and a 10,000-resample task-level paired bootstrap CI on the success-rate delta using fractional per-task scores (0, 1/3, 2/3, 1 over seeds). The S4 verdict states the p-value and CI next to the margin. (Context: under a no-difference null, +2/20 passes by luck roughly a third of the time — the verdict must carry its own evidential weight.)
- **Escalation (pre-registered at freeze):** if the net margin against any baseline lands in {+1, +2, +3} tasks, run seeds {4, 5} on the discordant tasks only for that pair and re-decide those tasks on ≥3/5 — removes threshold-flip noise exactly where the decision is made, at a bounded cost (typically 8–32 extra episodes). After escalation, escalated tasks score fractionally in fifths (0..5/5); the sign test and bootstrap are recomputed **once** on the final grid, and the verdict reports both pre- and post-escalation p/CI. No other recomputation is permitted.
- **Scheduling:** runs execute in (task, seed) blocks adjacent in time across all arms, so R9 thermal drift cancels within each paired comparison. "Interleaved" means this, not round-robin by arm. Within-block arm order is pre-registered: RLM then B2 on the resident topology, then one leaf relaunch serves B1 and B3 back-to-back — bounding server relaunches at two per block. Relaunch time is excluded from per-task wall-clock but included in the S0 cost projection; the quiesce + `/props` re-assertion covers correctness at each switch.
- `budget_kill` and `context_exhausted` episodes count as failures for **every** arm. `error` episodes are re-run once with the same seed (rows linked via `superseded_by`, §6); a second failure scores as a failure for that arm. Tokenized task length is recorded per episode.

**Cost scorecard (mandatory in the S4 report):** per-arm, per-task total tokens (tokens_in + tokens_out over all steps, retries included) and wall-clock, from the trace; energy in joules where available — the scaffold samples package power at 1 Hz (rocm-smi/hwmon, alongside the R9 temperature log) into `avg_power_w`/`energy_j`, enabled **only** if the S0 overhead check shows sampling does not perturb the numbers it measures (nullable otherwise; the named collectors are Linux-path — on a Windows host, `energy_j` stays null unless an equivalent is validated in the same S0 item). Interpretive rule: **any win claim states the cost multiple next to the margin** ("+3 tasks at N× median wall-clock and M× tokens vs B1/B2/B3") and includes a success-vs-cost Pareto chart. Deliberately no hard cost gate — the decision rule stays single.

**Per-category reporting:** the S4 report tables success per category per arm. **Zero-floor tripwire (pre-registered):** any category where RLM scores 0 while any baseline scores ≥3 of its tasks blocks a "clean pass" — the aggregate gate may still pass, but the verdict carries a mandatory named category-regression finding, and that category is the first post-S4 investigation. **Per-category margin gates are explicitly refused:** at 4–6 tasks per category any margin is noise, and a per-category gate would generate random vetoes of a valid aggregate result. This refusal is written down so the idea is not reinvented later.

**Chunk-size lock:** S4 runs RLM, B2, and B3 at the untouched config default (32K — Q1's admitted guess); the §7 #2 sweep is forbidden before the S4 verdict (sweeping RLM's main lever on benchmark-adjacent signal while B1 — the only chunk-independent arm — gets no analogous tuning would be test-set tuning); any post-S4 chunk-size change re-runs **all three chunked arms** (B2 and B3 share the C2 chunker verbatim, so the controls stay controlled) before any claim updates. The verdict line records `chunk_size`.

**Baselines (defined now, run in S4):**
- **B1:** leaf model, single shot, full 256K native context. Runs on a dedicated config profile (`--parallel 1 -c 262144`; ~2.8 GB KV at q8_0 on this arch), launched non-concurrently with the RLM topology. Overflow policy, pre-registered: tasks whose tokenized length exceeds the window are head+tail truncated to fit (50/50 split), and the truncation is recorded — the B1-infeasible subset must be identifiable in the results. (Does raw long context beat the scaffold?)
- **B2:** deterministic map-reduce — scaffold-only chunking (the C2 chunker, verbatim) → leaf summaries → root final. No RLM agency, no REPL.
- **B3:** deterministic BM25-RAG single shot — the C2 chunker verbatim; chunks indexed with DuckDB's FTS extension (BM25; DuckDB is already a dependency); chunks ranked against the task question; the B1 single-slot 256K profile filled to a pre-registered **80% of window**, restoring original document order; one call; same checkers. **No tunable k** — window-fill is the selection rule, fixed before freeze, so nothing is tuned on the benchmark. Pre-registered as a floor for deployable practice, not a ceiling on retrieval. (Is the scaffold worth more than the pipeline a practitioner would actually deploy?)

B2 is the honest control. If fixed map-reduce matches RLM on success rate, the root's agency is not earning its complexity and the project pivots to B2. B3 closes the blind spot B2 leaves open: without it, "beats B2 but loses to a free BM25 pipeline" would score as a project win.

---

## 9. Vertical Slices & Gates

Each slice has a verifiable exit criterion. No slice's **gate may be attempted** before the previous gate passes. Every gate additionally requires the component test suite green (§5 Testing). **Amendment (2026-08-12):** Capa-1 scaffold development against the mock dispatcher (§5 dry-run mode) may proceed before and in parallel with S0 — S0 gates hardware viability, not scaffold code; S1 and later still gate on S0.

**S0 — Kill gate (one afternoon, plus one 10-minute soak).**
Both servers up with §4 flags, each on its §7 backend; `-ub`/`-b` values stated next to every reported number. Measure and record:
1. Leaf prefill at a 32K-token prompt: median of 3 cold runs, single slot, per backend. **Gate: ≥ 500 t/s — exact threshold, no tilde, no rounding — on the chosen backend. Below: RLM is dead on this hardware; stop and write up why.** (Linux community prior: 716–807 t/s; Windows pairings publicly unmeasured — the gate judges on-box numbers either way.)
2. Concurrency scaling: aggregate prefill throughput at 1/2/4/8 slots (this sets the S2 fan-out target, §7 #1).
3. Decode on both servers, fresh and at 32K depth; root decode again **during** an active 8-slot leaf prefill (the contention number; also the stated R4 measurement condition).
4. Leaf KV cost per slot. Pre-registered expectation ≈0.45 GB per configured 40K slot (≈0.35 GB of that is the 32K chunk); ~1.5 GB indicates misconfiguration (§4). Verify q8_0 K/V engages on the hybrid arch; fall back to bf16 KV if not.
5. `timings.cache_n` sanity on the hybrid arch, **both servers**: (a) leaf — a warm-slot re-query of an identical prompt must show reuse; (b) root — a scripted 3-turn conversation on port 8080 must show per-turn `cache_n` ≈ total prior-turn tokens (only the new observation prefills; this arms the §7 #3c monitor). Document observed checkpoint behavior (R8). Moved here from S2 because R8 can kill §7 #3 outright.
6. 10-minute sustained 8-slot prefill: record clock and package-temperature drift (R9).
7. B1 shakedown: leaf relaunched once at `--parallel 1 -c 262144` — full-window prefill and decode spot-check, so the B1/B3 profile (§8) is not first exercised on benchmark day. (Restores v0's full-window measurement.)
8. Power-sampling overhead check: repeat item 1 with the 1 Hz package-power poller on vs off; the §8 joules columns are enabled only if the delta is noise.
After S0: §7 numbers replaced in place; C5 budget defaults re-derived (`max_wall_clock` ≈ 2–3× predicted prefill time per task-size class); the §8 benchmark task count (20 vs 30) is fixed by the pre-registered cost-projection rule.

**S1 — Minimal loop.**
Root + sandbox + C2 + C3 + one synchronous leaf call. Depth 1, no parallelism, no budgets beyond a wall-clock kill.
*Gate (operationalized):* needle task with context ≥64K tokens (asserted programmatically via `/tokenize`; ≈250K chars):
(a) control — root alone, document truncated to ≤28K tokens (the 32K window minus prompt overhead and the 90% C5 margin, so the request actually fits) by a stated deterministic rule, needle placed beyond any retained region: 0/3 attempts pass;
(b) RLM — ≥2/3 attempts pass the same checker.
Second task: a paraphrase-needle that regex cannot find, requiring ≥1 leaf call — this exercises the leaf path (otherwise untested until S2) and pre-tests the R5 confabulation surface.
*The R1 checkpoint is a controlled prompt A/B*, not ad-hoc tinkering: two pre-authored root-prompt variants from the registry (§5) — (a) tips-only, (b) tips + two compact worked REPL exemplars (~1.2K tokens: a needle-scan loop over `chunks`, and an `await llm_query` fan-out ending in `final_answer()`), using the exact injected API signatures and fixture-shaped data only, never benchmark corpora — 3 attempts each on the S1 fixtures; the winner is pinned, hashed into `config_snapshot`, and frozen with the S2 benchmark. Hard caps: at most two variants live at a time; non-benchmark fixtures only. Strategy-template authoring (§5) happens here under the same rules. If the root flails in both arms, record it — a negative result is a finding, not a failure to hide. S1 traces also re-derive the leaf `max_predict` default (C5).

**S2 — Parallelism + prefix caching + benchmark authoring.**
C4 complete with semaphore; byte-identical leaf prefix; chunk-first prompt layout (§4); `asyncio.gather` fan-out. Benchmark authored per §8 (including the contamination & checker-validity preconditions) and frozen at end of this slice, together with the prompt/strategy registry. The S2 fixtures are specified as part of the same deliverable:
- fan-out fixture (gate c; also carries the wave-tail metric — max−min completion within a gather — and a shuffled-vs-LPT admission-ordering A/B; if the fixture shows a real tail, LPT ordering enters §7 as a new numbered entry under I5, never earlier);
- re-query fixture (gate b; doubles as the priced cost measurement for same-slot revote — re-issuing an identical prompt with a different seed pinned to the warm slot, a possible post-S4 leaf self-consistency mechanism);
- question-batching A/B (the R8 fallback: k warm-cache single-question calls vs 1 batched `llm_query_multi(chunk_ref, [q1..qk])` call — a formatting-only wrapper — scored on latency **and** per-question answer quality);
- leaf-envelope A/B: JSON envelope `{"answer": str, "evidence": [str], "abstain": bool}` with format instructions in the byte-identical prefix, parsed and validated **scaffold-side** in `llm_query` (one retry on parse failure via existing C4 machinery, then a structured error the root can branch on); evidence spans verified by whitespace-normalized substring match against the chunk in the REPL (deterministic R5/R12 detector, zero model calls; normalization rules pinned). Server-side grammar/json_schema enforcement is a separate optional flag, A/B'd and **never trusted** (documented llama.cpp defects include silent fail-open on schema-parse failure). If the A/B shows format cost exceeds attribution benefit, the envelope stays off — that outcome is acceptable going in.
*Gate (all on named non-benchmark fixtures):*
(a) prefix integrity — `tokens_cached ≥ tokenized-prefix-length` on every warm-slot call of the fan-out fixture;
(b) repeated-chunk reuse — >80% token-weighted reuse on the re-query fixture;
(c) fan-out speedup ≥ 80% of S0's measured 8-slot aggregate scaling.

**S3 — Budgets + tracing.**
C5 and C6 complete. Adversarial self-test with the mock dispatcher: a task that infinite-loops in the REPL and one that requests unbounded sub-calls; plus a hard-kill mid-episode to verify the C6 durability promise (R10) **and** the post-restart tombstone (§6 crash recovery).
*Gate:* all terminate deterministically within budget, and `rlm replay` verifies the full trajectory of any episode from the trace store alone (DuckDB + referenced blob directory, episode-relative paths — per-step `root_view_hash` equality plus transcript render), **with the lifecycle log deleted** — no logs, no stdout.

**S4 — Benchmark + quality gate.**
Run frozen benchmark: RLM vs B1 vs B2 vs B3, 3 seeds each, in (task, seed) blocks per §8 scheduling.
*Gate:* RLM beats all three baselines per the §8 margin rule; the verdict states the sign-test p-value and bootstrap CI beside each margin, applies the pre-registered escalation if any margin lands in the noise band, reports the cost scorecard and per-category table, honors the zero-floor tripwire, and records `chunk_size`. A tie with any baseline fails the gate; B2/B3 tie-or-loss triggers the respective pivot rule. Only after this gate may anyone propose `max_depth > 1` or new optimizations.

**S5 — Models are config.**
*Gate:* swap the root to a new model (target: Qwen3.8-27B — open weights announced as imminent at v0.1 time; this slice may activate early) by editing `config.yaml` only, rerun S0 raw numbers + S4 benchmark, produce one comparison row. Day-one checklist for any new model: (1) dense or MoE? (2) **attention layout — how many layers carry per-token KV? linear-state size?** (3) MTP head present? (4) chat template diffs (template hash into config_snapshot)? (5) tokenizer identity? (6) license? (7) GGUF quality (community quants stabilize over ~2 weeks). (8) multimodal components (mmproj) present but unused? (9) **RLM-post-trained?** If so, which harness, system prompt, and conditioning flags was it trained against — align the scaffold's injected API and prologue or expect degradation.
*Pre-registered candidate rows (run nothing before S5):*
- **A3B-as-root** — the leaf GGUF loaded on the root server (`--parallel 1 -c 32768`), config-only. Rationale: root turns are decode-bound serial segments, and a 3B-active root decodes ~4× faster; the RLM paper's RLM(GPT-5-mini) beat plain GPT-5 on OOLONG, suggesting root capability requirements inside a good scaffold are lower than intuition says. Counter-evidence: minRLM's nano-class result (−9.5pp — weak roots write bad Python), which is exactly what this row measures. Either outcome purchases real evidence.
- **RLM-post-trained root** — `mit-oasys/rlm-qwen3-30b-a3b-v0.1`: an **official RLM-paper-author release** (mit-oasys = Zhang/Khattab's lab), Apache-2.0, LoRA r=32 on Qwen3-30B-A3B-Instruct-2507, RL-trained at depth-1 inside the alexzhang13/rlm harness (the `llm_query` API this scaffold deliberately matches, §5). Self-reported vs untrained base, in-harness: OOLONG +13.4pp, BrowseComp-Plus +18.1pp, LongBench-v2 CodeQA +20.0pp — not independently replicated. Path: merge LoRA offline → convert/quantize to GGUF → S5 row. Checklist item 9 applies with force (trained against a different system prompt and conditioning); base is full-attention MoE, so KV reprices via item 2. This is the direct answer to R1's root cause without opening S6.

**S6 — Learning loop. UNSCHEDULED. Do not build.**
Documented only so the trace schema decision (§6) has a stated payoff. Preconditions before this slice may even be *planned*: (a) ≥500 successful episodes logged, (b) a recurring failure mode demonstrably unfixable by prompting, (c) S4 gate passed. Contents if ever activated: SFT on own successful trajectories; Voyager-style skill library persisted between episodes; RL with AgentWorld-simulated environments **for training only** — evaluation stays on the real REPL (I7). Known hazard: template collapse (RAGEN-2 precedent) — any training data must pass **input-sensitivity / cross-input-distinguishability checks** (RAGEN-2 shows the collapse evades entropy and surface-diversity metrics).

---

## 10. Known Risks

| # | Risk | Response |
|---|---|---|
| R1 | Root degrades under the scaffold (precedent: Prime Intellect's RLMEnv evaluation, Jan 2026 — untrained models incl. INTELLECT-3 scored *lower* inside an RLM harness on several benchmarks, though supplying a strategy flipped some results; the RLM paper's own Qwen3-8B root needed dedicated post-training; minRLM's nano-class root lost 9.5pp, consistent with weak roots writing bad code — directional, no ablation) | S1 gate is early and explicit; scaffold-selected strategy templates + prompt registry (§5), decided by the S1 controlled prompt A/B; pre-registered S5 hedges (A3B-as-root, RLM-post-trained root); accept and publish a negative result rather than rationalize it |
| R2 | iGPU prefill insufficient for leaf economics | S0 is a kill gate, not a benchmark; threshold is exact (§9) |
| R3 | Prefix cache silently broken by prefix drift | `tokens_cached` continuously monitored via the prefix-integrity check (§7 #3a); prefix text hashed into `config_snapshot`, so **between-run** drift is detectable too (within-run monitoring alone cannot see it) |
| R4 | MTP path immature | MTP merged into llama.cpp 2026-05 (PR #22673) but beta: single-slot only, draft-n ≤ 2, spotty Vulkan history. Optional variant; adopt only if benchmark success unchanged AND decode ≥1.4×, measured on-box under the stated contention condition |
| R5 | Leaves confabulate on chunks (plausible-but-wrong extractions) | Aggregation tasks in the benchmark punish it; root cross-checks via code (regex/count verification in REPL) where the task allows; S1's paraphrase-needle task probes it early; evidence-span substring verification (S2 envelope fixture) detects it per-extraction at zero model cost |
| R6 | Sandbox escape / resource abuse | No-network default (defined in C1), rlimits/Job Objects, per-episode interpreter, unconditional kill (C1, C5). Local single-user threat model — documented, not overengineered |
| R7 | Qwen3.8-27B lands with different architecture (Qwen already switched to hybrid attention once between 3 and 3.6) | S5 checklist (now incl. attention layout); the incumbent stays resident during A/B (headroom, §4) |
| R8 | Hybrid linear-attention defeats llama.cpp prompt caching (checkpoint invalidation; llama.cpp issues #18497 / #19794 / #20225 / #24055) — on either server | `cache_n` sanity check in S0 covers leaf re-query **and** root multi-turn; §7 #3c monitors the root continuously; if leaf reuse is broken, demote §7 #3b and price the pre-registered fallback — question batching per chunk (`llm_query_multi`), measured by the S2 A/B fixture; full-attention MoE fallback leaf stays listed in config |
| R9 | Thermal throttling on a mini-PC drifts wall-clock — the primary metric — across a benchmark run | S0 10-minute soak; package temp logged per episode; S4 runs in (task, seed) blocks adjacent across arms so drift cancels within each paired comparison |
| R10 | Trace loss on crash | C6 per-step transaction commit, blob-before-row ordering; S3 hard-kill test verifies the "at most the current step" promise and the post-restart tombstone (§6) |
| R11 | Upstream llama.cpp regressions across updates | Build pinned S0→S4 and recorded in `config_snapshot` (incl. `-ub`/`-b`); upgrades are treated as optimizations under I5 (benchmark-neutral-or-better) |
| R12 | Injection-shaped text in the analyzed corpus steers leaf extractions (a benchmark-robustness risk under the local single-user threat model, not a security hole) | Leaf outputs treated as untrusted data by the root's cross-checks (R5); evidence-span verification means injected text cannot fabricate a span that verifies against the actual chunk; one adversarial-context task in the benchmark (§8) |

---

## 11. Out of Scope

- **vLLM on this hardware.** Historically broken for months on gfx1151; upstream support merged ~Q1 2026 and community containers now work. Still out of scope: llama.cpp wins single-stream latency and operational simplicity for this topology, and one serving stack is enough.
- **NPU.** llama.cpp has no XDNA2 backend, and the NPU stacks that do exist (Lemonade, FastFlowLM) share the same 256 GB/s memory — the NPU cannot raise the decode ceiling. Excluded. **NPU auxiliary offload** (embeddings/rerankers) likewise: no auxiliary model exists in the minimal system, and a second inference stack is a new failure surface with no workload to serve.
- **Runtime world models / simulators** (I3). Lifecycle-only, S6-only.
- **Recursion depth > 1** until S4 evidence demands it.
- **Browser/pyodide sandboxing.** `subprocess` + rlimits (or Job Objects, A2) is sufficient for the threat model.
- **Multi-machine RPC clustering.** One box.
- **Any training** until S6 preconditions are met.
- **Decision records (v0.2 — evaluated and rejected; do not re-litigate without new facts):**
  - *KV/slot save-restore persistence across restarts.* The endpoint exists (`--slot-save-path`, POST /slots/{id}?action=save|restore) but hybrid recurrent-state serialization sits on the R8-fragile surface, and the avoided cost is ~seconds of prefix re-prefill per server lifetime.
  - *Eager dispatch of leaf calls parsed from an incomplete root turn.* Executes model output before the model finished emitting it (I1) and creates steps whose parent `repl_exec` does not exist yet (I4). The in-code alternative — the root using `asyncio.as_completed` to process early results — is already available in the REPL and belongs in strategy-template content.
  - *Root slot-2 speculative next-turn prefill.* The next turn extends slot 0's resident cache — only the ≤2,000-char observation prefills (<1 s); slot 2 also conflicts with single-slot MTP. (This incremental-cache fact is what §7 #3c protects.)
  - *Mid-episode RLM→B2 arm switching.* Makes the RLM arm's measured success rate asymptotically max(RLM, B2) — the margin rule and pivot trigger become unfalsifiable. Episode-level B2 fallback is admissible post-S4 in an operational mode only (fresh episode, `fallback_of` linkage — column added to §6 if/when the mode is built; benchmark runner asserts the mode off), never during benchmark runs.
  - *Content-adaptive variable chunk sizing.* Destroys per-call cost uniformity — the stated scheduling rationale for chunk sizing — and confounds the §7 #2 sweep. Boundary-snapping within ±10% (C2) captures the quality benefit at none of the scheduling cost.
  - *Scaffold-side relevance prefiltering (embeddings/BM25 deciding which chunks the root sees).* Usurps the coverage-vs-sampling decision the RLM thesis assigns to the root and the aggregation category exists to score; it would quietly convert the runtime into a RAG pipeline wearing an RLM costume. B3 (§8) is where retrieval competes — as a baseline, not inside the loop. The only admissible future form is a root-owned, read-only `rank_chunks()` utility, post-S4-gated on traces showing the root attempts ranking in the REPL and does it badly.

---

## 12. Stated Assumptions & Open Questions

- **A1 — Scaffold language is Python 3.12+.** Rationale: the REPL is Python, `asyncio` maps to the dispatcher, servers are HTTP (language-agnostic). Tooling: pytest + hypothesis (§5 Testing), pydantic for the config schema (§5). If the deterministic-core-in-C# convention (external reference, below) should extend here, the scaffold is small enough to port later; do not block on it.
- **A2 — Host OS. DECISION REQUIRED BEFORE S0 — the tripwire has fired: this spec is being authored on a Windows 11 machine.** The choice is convenience vs headroom + comparability, not possible vs impossible (v0.1.1 correction: ROCm is officially on Windows for gfx1151 since HIP SDK 7.1.1, Feb 2026).
  **Linux path:** rlimits + network namespace (C1); dynamic GTT (raised to ~100 GiB per §4 host config; practical ceiling ~110–120 GB); Mesa RADV for the root; and — decisive for §7/§8 — every community benchmark corpus behind this spec's priors is Linux, so on-box numbers stay comparable to reference data.
  **Windows path (viable, with named caveats):** C1 uses Job Objects (also providing C5's process-tree kill); network isolation is best-effort — AppContainer, else WFP/firewall rules; all weaker than a netns (R6 weakens accordingly). llama.cpp ROCm ships prebuilt for gfx1151 (AMD repo.radeon.com, ggml-org win-rocm zips, lemonade nightlies). Vulkan is AMD's proprietary driver — RADV does not exist on Windows — and Windows-vs-Linux llama.cpp performance is publicly unmeasured (anecdotes: ~20–30% behind Linux/RADV). GPU memory: Adrenalin Variable Graphics Memory static carve, up to 96 GB dedicated on 128 GB (set before S0; the ~54 GB residency fits), plus the 50%-of-remaining shared pool; two open bugs bite this workload — hipMalloc spills to slow shared memory past ~32 GB (ROCm #5940) and Windows ROCm places KV cache in shared memory (llama.cpp #18011). Keep a 32–64 GB pagefile for load-time commit spikes.
  **WSL2 now runs ROCm on this chip (Adrenalin 26.2.2 + ROCm 7.2.1, Mar 2026) but remains invalid for measurement: unrepresentative performance, and no RADV inside WSL2.** Decision record: `Host OS: ________ — decided ____-__-__`.
- **Q1 —** Default chunk size 32K is a guess; §7 #2 sweeps it (post-S4, per the chunk-size lock).
- **Q2 —** Truncation cap 2,000 chars is a guess; revisit after S1 traces show what the root actually reads.
- **Q3 —** Whether the root should be allowed `role="root"` self-calls (recursion) is deferred to the `max_depth` gate in S4 — the dispatcher supports the parameter, the budget forbids it by default, and `steps.depth` (§6) can represent it.
- **External references:** "Thesis B" (§2), "Conduo" (§5), and the "deterministic-core-in-C# convention" (A1) are the author's private framework references. Nothing in this spec requires them: the operative content is stated inline wherever they appear.

---

## 13. References

- Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601 (method reference; depth-1 experiments, truncated REPL views, cheap-leaf sub-calls; released RLM-Qwen3-8B, +28.3% over untrained base).
- Prime Intellect — *Recursive Language Models: the paradigm of 2026* (blog, 2026-01-01, primeintellect.ai/blog/rlm) — RLMEnv evaluation behind R1, incl. the strategy-hint flip; INTELLECT-3 technical report is arXiv:2512.16144.
- minRLM — community RLM reimplementation + benchmark, Avi Lumelsky, Mar 2026 (github.com/avilum/minrlm; write-up at avilum.github.io/minrlm). Self-reported, not peer-reviewed, **no component ablation** — cited only as directional evidence: +22.3pp vs vanilla for an untrained mini-class root under a code-first scaffold with task-routed templates; −9.5pp for a nano-class root that cannot write reliable code; 1.3–2.6× per-query token reductions vs vanilla (its 3.6× headline is vs the official RLM implementation, a different comparator).
- mit-oasys/rlm-qwen3-30b-a3b-v0.1 (Hugging Face, May 2026) — official RLM-paper-author LoRA (Apache-2.0, r=32 on Qwen3-30B-A3B-Instruct-2507, RL via prime-rl at depth-1 in the alexzhang13/rlm harness); self-reported +13.4/+18.1/+20.0pp vs base in-harness; S5 candidate. Harness API reference: github.com/alexzhang13/rlm (`llm_query`/`rlm_query`).
- Qwen team — *Qwen-AgentWorld: Language World Models for General Agents*, arXiv:2606.24597 (trajectory format cited in §6; S6 simulation option).
- *RAGEN-2: Reasoning Collapse in Agentic RL*, arXiv:2604.06268 — template collapse, S6 hazard. (Original RAGEN: arXiv:2504.20073.)
- Wang et al. — *Voyager*, arXiv:2305.16291 — executable-code skill library, S6 option.
- Evaluation methodology: Dietterich 1998 (approximate statistical tests for classifier comparison; McNemar), Dror et al. 2018 (statistical significance in NLP), Sainz et al. 2023, arXiv:2310.18018 (LLM data contamination); BrowseComp-Plus, arXiv:2508.06600 (fixed-corpus BM25 baseline practice behind B3).
- Strix Halo performance priors (§7): kyuz0/amd-strix-halo-toolboxes results dataset (2026-05, llama.cpp b9187/b9193, benchmarks the exact §4 models); nabe2030/hip-vs-vulkan-evo-x2 (ROCm 7.2.2 vs Vulkan, FA on); llama.cpp discussion #20856 (known-good gfx1151 stack, q8_0 KV at 262K ≈ 2.7 GB on this arch).
- llama.cpp mechanics: server README (slot semantics, `timings.cache_n`, `--slot-prompt-similarity`, `id_slot`, `-b`/`-ub`, slot save/restore); PR #22673 (MTP speculative decoding, 2026-05-16); issues #18497, #19794, #20225, #24055 (hybrid/recurrent prompt-cache regressions, R8); issue #11681 / discussion #22658 (`-c` split across slots); json_schema/grammar defects #19051 (silent fail-open), #21571, #23775, #22314 (why server-side enforcement is never trusted, §9 S2).
- Model cards/configs: Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B (hybrid attention layouts, MTP heads); unsloth GGUF + MTP-GGUF repos (measured file sizes).
- Windows path (v0.1.1): AMD HIP SDK for Windows system requirements (gfx1151 since 7.1.1, Feb 2026); AMD prebuilt llama.cpp ROCm binaries for Windows (rocm.docs.amd.com radeon-ryzen llama.cpp page / repo.radeon.com); strixhalo.wiki "llamacpp with ROCm" (rocWMMA-slower-at-long-context caveat; known-good HIP flags); llama.cpp issue #18011 (Windows ROCm KV cache in shared memory); ROCm issue #5940 (hipMalloc >32 GB spill despite 96 GB VGM); AMD Variable Graphics Memory blog (96 GB dedicated on 128 GB systems); Phoronix, AMDVLK discontinued (Sept 2025); CPython gh-77589 (no AF_UNIX on Windows); ROCm-on-WSL2 for Strix Halo (Adrenalin 26.2.2 + ROCm 7.2.1) with ROCm issue #6022 (WSL2 VRAM mapping).

---

## 14. Changelog

- **v0.2 — 2026-08-12.** Applied the design-panel add-now set (4-lens panel + skeptic prioritization; both externally-cited sources verified before citation — minRLM re-framed as directional/self-reported with no ablation; the mit-oasys 30B LoRA confirmed as an official paper-author release). **Eval science (the heart of this amendment):** S4 scoring gains a mandatory inference layer (exact sign test over discordant tasks + 10k task-level paired bootstrap CI on fractional scores; verdict states p and CI beside the margin), pre-registered noise-band escalation (seeds {4,5} on discordant tasks when a margin lands in {+1..+3}), blocked (task, seed) scheduling, a pre-registered 20-vs-30 task-count rule, contamination & checker-validity freeze preconditions (closed-book probe, synthetic needles, corpus dating, near-miss checker suite), the **B3 BM25-RAG baseline** (DuckDB FTS, C2 chunks, 80%-window fill, no tunable k, symmetric margin + pivot-to-RAG rule), a mandatory cost scorecard (tokens + wall-clock; joules gated on the S0 overhead check), per-category reporting with a zero-floor tripwire (per-category margin gates explicitly refused), the S4 chunk-size lock, and an aggregation-category rule (≥1 regex-defeating + ≥1 regex-solvable task). **R1 hedges:** prompt registry (hash-pinned versioned files, no inline prompts) + scaffold-selected per-category strategy templates (frozen with the benchmark); the S1 R1 checkpoint becomes a capped controlled prompt A/B (tips-only vs tips+worked exemplars); pre-registered S5 rows: A3B-as-root and the mit-oasys RLM-trained LoRA (checklist item 9: harness-alignment; `llm_query` name declared load-bearing). **Measurement/systems:** S0 adds root-side multi-turn cache verification (new §7 #3c standing monitor — the root is hybrid too), the power-poller overhead check, and `-ub`/`-b` pinning; per-role `max_predict` (leaf 512, re-derived from S1); C2 cut rule (boundary-snap ±10%, deterministic tie-break); §7 #3b affinity-map contingency design + warm-hit diagnostic; question-batching pre-registered as the R8 fallback with an S2 pricing fixture; leaf JSON envelope + evidence-span verification as an S2 A/B (server-side grammar never trusted). **Operability:** `root_view_hash` + `rlm replay` as the S3 gate instrument; dry-run mode (`dispatcher: mock`, `dry_run` flag, scoring refusal) with a dated amendment allowing pre-S0 Capa-1 development; pydantic config schema (extra=forbid, cross-field validators) + Appendix A sketch; narrow JSONL lifecycle log (S3 gate runs with it deleted); crash recovery (tombstone-never-resume, `scaffold_instance_id`/`sandbox_pid`, `operator_abort`, `superseded_by` rerun rule); five-verb CLI surface with written non-goals; property-based suites for C3/C5; per-episode `/props` re-assertion + `server_unreachable` mapping. **§11 decision records** added for six evaluated-and-rejected paths (slot save/restore, eager dispatch, root slot-2 prefill, mid-episode arm switching, content-adaptive chunking, scaffold-side prefiltering). Eval-methodology and RLM-ecosystem references added to §13. Post-amendment verification (3 agents: consistency, 27-item coverage — all items present; regression vs v0.1.1 — zero substantive loss, invariants byte-identical) fixes folded in: chunk-size lock corrected to cover all three chunked arms (B3 is chunked too); escalation↔inference recomputation rule pre-registered (escalated tasks score in fifths, one recomputation, both pre/post p+CI reported); per-step timestamps (t_dispatch/t_first_byte/t_end) and per-episode temperature columns added to §6; within-block S4 arm order pre-registered (RLM, B2 resident, then one relaunch for B1+B3); adversarial-category → underlying-shape template mapping defined; 60 h wall-budget made exact; run-id defined as a bench invocation; aggregation regex-solvable half hardened to must; dangling references (LPT queue slot, same-slot revote, B3-infeasible, GTT figure) resolved; R1's minRLM claim softened to directional.
- **v0.1.1 — 2026-08-12.** Windows/backend corrections after a challenge to the "no ROCm on Windows" claim (challenge verified correct against Aug-2026 sources). A2 rewritten from "ROCm path gone on Windows" to viable-with-caveats: ROCm official on Windows for gfx1151 since HIP SDK 7.1.1 (Feb 2026), prebuilt llama.cpp ROCm zips from AMD/ggml-org/lemonade; GPU memory via Variable Graphics Memory up to 96 GB dedicated; named Windows bugs (ROCm #5940 hipMalloc spill, llama.cpp #18011 KV-in-shared-memory); AppContainer named as the stronger network best-effort. "Vulkan RADV" scoped to Linux — RADV does not exist on Windows (AMD proprietary driver there, publicly unmeasured vs RADV). "Never AMDVLK" dropped: AMDVLK discontinued Sept 2025; its 319 t/s figure kept as historical context only. rocWMMA demoted from recipe to A/B candidate — reported slower than plain HIP FA at long context since ROCm 7.0.2+; S0 uses known-good flags (GGML_HIP_NO_VMM=ON, GGML_HIP_MMQ_MFMA=ON, ROCBLAS_USE_HIPBLASLT=1). §7 priors labeled as Linux measurements; WSL2 rationale updated (runs ROCm since Adrenalin 26.2.2 + ROCm 7.2.1, still invalid for measurement). Windows-path references added to §13.
- **v0.1 — 2026-08-12.** Applied external review (research-verified against mid-2026 sources). Headline changes: (1) both models identified as hybrid linear-attention — KV math corrected (~4× cheaper; committed ~54 GB, headroom ~74 GB), attention layout added to the S5 checklist, R8 added; (2) prefix-caching metric redefined token-weighted (`tokens_cached` replaces `cache_hit BOOL`), prompt layout pinned chunk-first, §7 #1 demoted to #3 with an honest structural ceiling, S2 gate split into integrity/reuse/speedup; (3) fan-out gate re-based on S0-measured concurrency scaling instead of an a-priori 3×; (4) §4 "12 vs 48 GB / 2.5×" arithmetic replaced with a derivable comparison; (5) S0 protocol pre-registered (exact 500 t/s threshold, backend named, contention + soak + KV + cache_n checks added); (6) C1/C4 bridge, chunker ownership (C2), final-answer channel, retry/budget semantics, root-window policy (`context_exhausted`), and C6 write mechanics specified; (7) steps/episodes schema extended (parent linkage, status, slot, retries, depth, seeds, build pinning); (8) benchmark authoring made an S2 deliverable, freeze-date contradiction resolved (end of S2, gates on non-benchmark fixtures), scoring/margin rules added, B1 profile + overflow policy defined; (9) R1 attribution corrected to Prime Intellect's RLMEnv post; vLLM/NPU out-of-scope wording updated to 2026 reality; MTP status updated (merged, beta, single-slot); (10) A2 escalated to a pre-S0 decision record (authored on Windows 11); amendment rule, references (§13), and this changelog added. Post-rewrite verification (3-agent) fixes folded in: S1 control document sized to actually fit the root window (≤28K tokens); per-slot KV pre-registration re-based to the configured 40K slot (≈0.45 GB); S0 gains a B1 full-window shakedown (item 7), restoring v0's full-window measurement; §7 backend deltas reconciled to their sources (+13% at d=32K, +41% at pp16K); C1/C5/R6 lifetimes stated per-episode; benchmark pinned at exactly 20 tasks with an explicit B1 tie rule; optimization order rationale stated (MTP after chunk tuning: beta, single-slot); S6's v0 "format diversity checks" deliberately superseded by input-sensitivity checks (RAGEN-2). Numbers in §7 remain priors until S0 runs on-box.
- **v0 — original.** Pre-implementation constitution, all §7 numbers hypotheses.

---

## Appendix A — config.yaml sketch (v0.2)

Validated by the §5 pydantic schema (`extra="forbid"`; cross-field validators: leaf `ctx == parallel × (chunk.size + chunk.overhead)`, root `ctx ==` window, semaphore == leaf parallel, reservation ≤ slot capacity, `mtp: true ⇒ root parallel == 1`). `config_snapshot` (§6) is the canonical JSON dump of the validated model.

```yaml
servers:
  root:
    model: models/qwen3.6-27b-q4_k_m.gguf     # or the -mtp variant, per R4
    mtp: false
    port: 8080
    backend: vulkan                            # linux: radv
    ctx: 32768
    parallel: 1
    cache_type: q8_0                           # bf16 fallback per S0 item 4
    flash_attn: "on"                           # pinned, never auto
    ub: 512                                    # pinned; stated with every S0 number
    b: 2048
    extra_flags: ["--no-mmap", "--no-context-shift"]
  leaf:
    model: models/qwen3.6-35b-a3b-ud-q4_k_m.gguf
    port: 8081
    backend: rocm                              # HIP FA; flags per §7
    ctx: 327680                                # = parallel × (chunk.size + chunk.overhead)
    parallel: 8
    cache_type: q8_0
    flash_attn: "on"
    ub: 512
    b: 2048
    extra_flags: ["--no-mmap", "--no-kv-unified", "--cont-batching"]
  fallback_leaf: null                          # full-attention MoE candidate, R8

scaffold:
  dispatcher: real                             # real | mock (dry-run mode, §5)
  chunk: { size_tokens: 32768, overhead_tokens: 8192, snap_to_boundary: true, snap_tolerance: 0.10 }
  truncation_cap_chars: 2000
  budgets:
    max_depth: 1
    max_subcalls: 32
    max_wall_clock_s: 900                      # re-derived per task-size class after S0
    max_total_tokens: 1500000
    max_predict: { root: 1024, leaf: 512 }     # leaf re-derived from S1 traces
  retries: { max_attempts: 3, backoff_s: [1, 4], per_call_timeout_s: 240 }
  root_window_kill_fraction: 0.90
  sampling:
    root: { temperature: 0.7, top_p: 0.8, seed: 1 }
    leaf: { temperature: 0.3, top_p: 0.9, seed: 1 }
  prompts:
    root: { path: prompts/root.v1.md, sha256: "<pinned>" }
    leaf_prefix: { path: prompts/leaf-prefix.v1.md, sha256: "<pinned>" }
    strategy_templates:                        # scaffold-selected by declared task category (§5)
      needle: { path: prompts/strat-needle.v1.md, sha256: "<pinned>" }
      aggregation: { path: prompts/strat-aggregation.v1.md, sha256: "<pinned>" }
      synthesis: { path: prompts/strat-synthesis.v1.md, sha256: "<pinned>" }
      code_qa: { path: prompts/strat-codeqa.v1.md, sha256: "<pinned>" }
      default: { path: prompts/strat-default.v1.md, sha256: "<pinned>" }
  leaf_envelope: { enabled: false }            # decided by the S2 A/B fixture

trace: { db_path: traces/rlm.duckdb, blob_root: traces/blobs }
benchmark: { version: null, seeds: [1, 2, 3] } # escalation seeds {4,5} per §8
power_sampling: { enabled: false }             # flipped only if S0 item 8 passes
```
