# ARCHITECTURE.md — RLM Runtime (working name: `rlm-halo`)

**Spec version:** `rlm-runtime-spec-v0.1` (changelog: §14)
**Status:** Pre-implementation constitution. No code exists. §7 now carries community-measured priors (off-box, mid-2026 gfx1151 data — §13); every number is still replaced by S0 on-box measurements.
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
| Backend | Vulkan RADV (decode-bound role) | ROCm + rocWMMA flash-attention (prefill-bound role). Never AMDVLK. |
| Variant | MTP GGUF — adopt only per the R4 criteria (§10) | standard |
| Role | plan, write REPL code, decide sub-calls, emit final | chunk-level extraction/summary/QA |

Quantized V-cache hard-requires flash attention; `-fa on` is pinned (not `auto`) so a failed FA probe fails loudly at startup instead of silently changing memory and performance behavior. Whether q8_0 KV actually engages on this hybrid architecture is an S0 check (bf16 KV is the fallback). Root 2nd-slot mechanics: enabling it requires `-c 65536` (llama-server splits `-c` across slots) plus re-priced KV, and is incompatible with the MTP variant, which is single-slot.

**Startup handshake** (scaffold entrypoint, ahead of C4): before accepting any task, the scaffold queries each server's `/props` and asserts model path, `n_ctx`, `n_parallel`, and cache types against `config.yaml`; any mismatch refuses to start. The `/props` responses (including server build) are recorded into `config_snapshot` (§6).

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

Headroom is deliberate. It exists for: (a) a third resident server if S6 ever needs one, (b) A/B running a candidate model next to the incumbent during S5 swaps, (c) larger chunk experiments, (d) B1's dedicated 256K single-slot profile (§8; ~2.8 GB KV, run non-concurrently anyway). Do not spend it by default.

**Host config (Linux path — see A2):** BIOS UMA carve-out at 512 MB; GTT raised via `ttm.pages_limit` / `ttm.page_pool_size` (~100 GiB; `amdgpu.gttsize` is deprecated on kernels ≥ 6.18); `--no-mmap` on both servers so weights land in GTT predictably.

**Prefix & cache contract (critical):** all leaf calls share one **byte-identical** system prefix. No timestamps, run IDs, task IDs, or counters anywhere in the prefix — run metadata travels in the trailing user segment or stays out-of-band in the trace. Prompt layout is fixed: `[system prefix][chunk][question]` — question **last**, so a re-queried chunk extends the cached prefix instead of invalidating it at token 0. Reuse mechanics (llama.cpp): the prompt cache is **per-slot** — there is no cross-slot sharing, so the prefix cold-prefills up to `--parallel` times before steady state; slot routing is longest-common-prefix (`--slot-prompt-similarity`), and `slot_id` is recorded per call (§6) to verify affinity. The monitored metric is **token-weighted** (`steps.tokens_cached`, from the server's `timings.cache_n`), never a boolean — see §7 #3 for the two targets and the structural ceiling on unique-chunk calls. R8 applies: the hybrid Gated-DeltaNet architecture reuses cache through context checkpoints, with documented llama.cpp regressions — cache behavior is verified in S0, not assumed.

---

## 5. Scaffold Components (Capa 1)

Six components. Python 3.12+ (assumption A1, §12). Each lists what it must NOT do — those are the load-bearing lines.

**C1 — SandboxManager.** Spawns one isolated Python interpreter per episode (§6: one run of a task; `subprocess` + resource limits: CPU time, memory, no network by default). "No network" means **no AF_INET/AF_INET6** (Linux: network namespace; Windows: best-effort via Job Objects + firewall rules, documented as weaker — A2). Persistent across REPL turns within an episode (Jupyter-like: variables survive). Killed unconditionally at episode end or budget breach.
*Must not:* share state between episodes; grant network unless the task config explicitly enables it.

**C2 — ContextLoader.** Materializes the input as variable `context` inside the sandbox before the root's first turn. Handles str, bytes, file paths, and lists of documents. Also owns the **deterministic chunker**: `chunks = split(context)` (chunk size from config, measured in target-leaf tokens via the leaf server's `/tokenize`), injected read-only into the sandbox. The root chunks **only** via this utility — free-form chunking in model code would make `chunk_size` advisory (a soft I1 violation) and render the §7 #2 sweep uncontrolled. B2 (§8) uses the same chunker verbatim.
*Must not:* place any part of the raw context into any message array (I2); allow model code to redefine or bypass the chunker.

**C3 — OutputTruncator.** Hard cap on all REPL output returned to the root (default 2,000 chars, config). `observation_view` is the ordered, labeled concatenation of stdout, stderr, last-expression repr, and formatted traceback, truncated **as one unit**. Appends a marker: `[truncated: showing 2000 of 184,203 chars]`.
*Must not:* be overridable by model output or by code running in the sandbox (I1). The cap is applied scaffold-side, after execution.

**C4 — LLMDispatcher.** Exposes `await llm_query(prompt, role="leaf")` inside the sandbox. Routes by role to the correct server. Owns an `asyncio.Semaphore` equal to the target server's `--parallel`. **Pre-flight check:** token count via the target server's `/tokenize`; prompts exceeding slot capacity are rejected without dispatch and logged as steps with `status=rejected`. **Retries:** every attempt is a logged step (shared `call_id`, incrementing `retry_idx`, per-attempt `status`); a retried call counts **once** against `max_subcalls`; every attempt's tokens count against `max_total_tokens`. Defaults (config): max_attempts 3, backoff 1 s / 4 s, per-call timeout 240 s. Requests are streamed so that cancellation (client disconnect) aborts server-side generation. Returns the result as a Python object in the caller's REPL — never auto-injected into the root's messages.
*Must not:* let the model choose servers, ports, or semaphore size; exceed `--parallel` (queueing upstream hides latency pathologies from the trace).

**The C1/C4 bridge.** `llm_query` is a scaffold-injected stub, not importable library code: it serializes requests over a scaffold-owned AF_UNIX socketpair (Linux; duplex pipe on Windows) passed at spawn — the only channel crossing the sandbox boundary, and explicitly exempt from the no-network rule (which bans AF_INET/AF_INET6, not the scaffold's own pipe). Semaphore, routing, retries, timeouts, and step logging all execute **scaffold-side**; nothing the model runs in the sandbox can alter them (I1). The sandbox REPL compiles cells with top-level `await` against a persistent event loop so `await llm_query(...)` is natural.

**C5 — BudgetEnforcer.** Per-episode limits from config. Termination is deterministic and unconditional.
*Budgets:* `max_depth` (default **1** — root + leaves, no grandchildren); `max_subcalls` (default 32); `max_wall_clock` (default 15 min; re-derived per task-size class after S0: ≈2–3× predicted prefill time at measured t/s); `max_total_tokens` (default 1.5M); `max_predict` per call (default 1024).
*Root window:* accounting from server-reported usage per turn; at ≥90% of the root window the episode terminates deterministically as `context_exhausted`.
*Admission:* the token budget is enforced at dispatch admission — `running_total + reservation ≤ cap`, reservation = pre-flight prompt tokens + `max_predict`; worst-case overshoot is bounded by in-flight calls × `max_predict` and documented.
*On breach:* cancel all in-flight dispatcher tasks (closing streamed connections aborts server-side work), release the semaphore, kill the sandbox, mark the outcome, persist partial state and full trace; cancelled calls are logged as steps with `status=cancelled`.
*Quiesce:* between timed episodes the scaffold verifies both servers report all slots idle before starting the clock.
*Must not:* warn-and-continue; accept model requests for extensions. Raising `max_depth` above 1 requires S4 evidence that depth-1 is the binding constraint.

**C6 — TraceLogger.** Single writer task fed by an asyncio queue; **one transaction commit per step** — this is what "a crash loses at most the current step" means — with blobs written before their referencing row. `step_idx` is assigned in commit order; causality is carried by `parent_step_idx`/`call_id`, never by `step_idx` adjacency. Blobs go to a per-episode directory, referenced by episode-relative paths. Monitoring reads happen in-process (DuckDB is single-writer; external readers get an append-only export, not a live handle).
*Must not:* sample, summarize, or truncate the stored record (store full observation as a blob ref; only the *root's view* is truncated, per C3).

**Dependency rule (lint-enforced, same pattern as Conduo — external reference, §12):** C1–C3, C5, C6 must not import C4 or any LLM client. Only C4 talks to model servers. The scaffold must be able to run a full episode with a mock dispatcher (used in S3's runaway test).

**Testing:** every C-component ships unit tests against the mock dispatcher (pytest, A1); each slice gate additionally requires the component suite green.

---

## 6. Data Layer — Trajectory Schema

The one decision made today that pays later: traces are stored **trajectory-shaped**, not log-shaped. The tuple `(state, action, observation, outcome)` is simultaneously the SFT format of the original RLM paper, the training format of Qwen-AgentWorld, and the raw material for a Voyager-style skill library. Zero runtime cost; three learning paths kept open.

**The state rule (discharges I4):** "state" is not a column. Root state at step *k* := deterministic replay of steps 0..*k−1*'s `(action, observation_view)` under the versioned system prompt. The schema stores everything that replay needs — nothing else is required.

**The final-answer channel:** `final` is emitted only via the scaffold-injected `final_answer(value)` inside the sandbox — never parsed from root prose. (A prose-parsed answer would be the one channel able to smuggle untruncated context past C3, violating I2.) The value is stored to a blob; the call is the episode's terminal step.

**Terminology:** episode = one run of a task; `task_id` is stable across seeds and arms; `task_hash` = sha256 of the instruction text (corpus documents hashed separately, in the benchmark manifest referenced by `benchmark_version`).

**`episodes`**

| column | type | notes |
|---|---|---|
| episode_id | UUID | |
| task_id | TEXT | benchmark task or ad-hoc |
| task_hash | TEXT | sha256 of instruction text |
| tokenized_task_len | INT | via leaf `/tokenize`; identifies the B1-infeasible subset (§8) |
| started_at / ended_at | TIMESTAMP | |
| outcome | ENUM | `success` \| `fail` \| `budget_kill` \| `context_exhausted` \| `error` |
| outcome_reason | TEXT | e.g. which budget breached; "no final emitted" |
| final_answer_ref | TEXT | parquet blob path (episode-relative) |
| config_snapshot | JSON | models, quants, budgets, chunk size, truncation cap; per-role sampling (temperature/top_p/seed); sha256 of every prompt/prefix text; llama.cpp build (sha, backend, compile + launch flags) per server; chat-template hash; `/props` responses |
| scaffold_git_sha | TEXT | |
| benchmark_version | TEXT | nullable |

**Outcome semantics:** `success` = final emitted and checker passes. `fail` = final emitted and checker fails, **or** root ends without emitting final (`outcome_reason` distinguishes). `budget_kill` = any C5 budget breach, including wall clock. `context_exhausted` = root-window accounting breach (C5). `error` = scaffold/server fault with no final.

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
| observation_view | TEXT | what the root actually saw (post-C3) |
| observation_full_ref | TEXT | parquet blob path (pre-truncation, episode-relative) |
| tokens_in / tokens_out | INT | server-reported actuals |
| tokens_cached | INT | server `timings.cache_n`; hit ratio derived as cache_n/(cache_n+prompt_n) — never stored as a boolean |
| slot_id | INT | server slot that served the call (cache-affinity debugging) |
| latency_queue_ms | INT | dispatcher-side: (t_first_byte − t_dispatch) − timings.prompt_ms — llama-server does not report per-request queue wait; both timestamps recorded |
| latency_prefill_ms / latency_decode_ms | INT | server `timings.prompt_ms` / `timings.predicted_ms` |
| cache_hit | — | **removed in v0.1** — replaced by `tokens_cached` |

Both `observation_view` and `observation_full_ref` are stored deliberately: the first reconstructs what the root knew (debugging, SFT); the second preserves ground truth (evaluation, world-model data).

---

## 7. Performance Model & Optimization Order (Capa 2)

**Numbers below are community-measured priors (off-box, mid-2026 gfx1151 data — §13). S0 replaces them with on-box measurements; this section is then updated in place.**

Decode ceiling (bandwidth-bound): `t/s ≈ BW_eff ÷ (active_params × bytes/param)`, with BW_eff ≈ 212 GB/s measured (256 theoretical).
- Root (27B dense-FFN, Q4): theoretical ceiling ~15 t/s, effective ~12.5. Community-measured: ~12 t/s fresh, 9–12 at 32K depth. Acceptable: root emits code, not prose.
- Leaf (~3B active, Q4): single-stream 50–60 t/s fresh, 44–49 at 32K depth. Aggregate decode across 8 continuous-batching slots is unmeasured — S0 measures it.

Prefill is compute-bound and is the real budget: a 32K-token leaf chunk is the unit cost of everything. Community-measured leaf prefill at d=32K: 807 t/s (ROCm+rocWMMA) / 716 t/s (Vulkan RADV); 319 on AMDVLK (disqualifying).

**Contention model:** root decode and leaf prefill share one iGPU. Working assumption is phase alternation (root decodes → awaits fan-out → leaves prefill); S0 measures the contended case explicitly (root decode during an active 8-slot leaf prefill), and the R4/MTP comparison must state its contention condition.

**Backend is a per-server launch choice, not a late optimization** (both backends ship prebuilt): leaf on ROCm + rocWMMA flash-attention (prefill-bound; community datasets put ROCm ahead of RADV by +13% at d=32K — 807 vs 716 t/s — and by +41% at pp16K in a second dataset; the spread is build- and depth-dependent, which is why the pair is A/B'd on-box), root on Vulkan RADV (decode-bound; +5–19% decode vs ROCm). Chosen and pinned at S0.

Optimizations in strict order of expected return. Per I5, each ships only with benchmark-neutral-or-better evidence, and each has one owning metric:

1. **Parallel dispatch** — metric: measured speedup on the 8-leaf fan-out fixture vs serial. Target: **≥80% of the aggregate prefill scaling S0 measures at 8 slots.** (An a-priori "3×" was ungrounded: the achievable factor depends on batch-1 compute headroom, which S0 measures at 1/2/4/8 slots.)
2. **Chunk-size tuning** — metric: wall-clock per task at fixed quality, swept over {16K, 24K, 32K, 48K}. Note: at 16K, one full pass over a 1M-char task consumes about half of `max_subcalls` — sweep with budgets re-derived per §9 S0.
3. **Prefix caching** — two metrics, both token-weighted (`steps.tokens_cached`; never a boolean):
   **(a) Prefix integrity (hygiene, always on):** `tokens_cached ≥ tokenized-prefix-length` on every warm-slot call. This is the R3 drift detector.
   **(b) Repeated-chunk reuse (the win case):** >80% token reuse when re-querying a chunk already resident in a slot. Requires the §4 prompt layout (chunk before question) and slot affinity (verified via `slot_id`; pin with `id_slot` if LCP routing proves insufficient).
   Structural ceiling for single-pass unique-chunk calls is `prefix_len / prompt_len` (a few %) — for pure map-reduce episodes this optimization **cannot** be the largest win; it earns its rank only where the root re-queries chunks. R8 applies (hybrid-arch cache fragility, verified in S0).
4. **MTP on the root** — metric: root decode t/s (target ≥1.4×) *and* unchanged benchmark success (R4). Prerequisites: llama.cpp ≥ the 2026-05 MTP merge (PR #22673), `--spec-type draft-mtp`, draft-n ≤ 2, root at `--parallel 1` (MTP is single-slot today), measured under the stated contention condition. Ordered after chunk tuning deliberately: MTP is beta and single-slot (R4); chunk tuning is pure config.
5. **Build-flag fine-tuning** (rocWMMA variants, per-backend flags) — metric: raw prefill t/s. Residual after the S0 backend split; done last because it is fiddly and orthogonal.

---

## 8. Measurement (Capa 3)

**Primary metric:** wall-clock per task at fixed quality. Never t/s in isolation (I5).

**Secondary metrics:** leaf prefill t/s; root decode t/s (contended and uncontended); leaf `tokens_cached` ratio; per-episode time split (REPL exec / LLM wait / scaffold overhead) — all derivable from the `steps` table, by design. Package temperature logged per episode (R9).

**Sampling & seeds:** per-role `temperature`, `top_p`, `seed` live in `config.yaml` and are recorded in `config_snapshot`. "3 seeds" = sampling seeds {1, 2, 3} per task per arm. Stated caveat: continuous batching means a fixed seed does not guarantee bitwise reproducibility — seeds pin sampling identity, not numerics.

**Frozen benchmark:** 20 tasks (count fixed exactly at freeze — the scoring denominator depends on it), programmatically verifiable (exact match with stated normalization, or checker function). **Authoring is an explicit S2 deliverable:** each task ships as (input ref, question, checker fn, expected answer) with unit-tested checkers and pinned corpus sources (repos at a fixed commit for code QA). Frozen and versioned at the **end of S2**; S2's own gates run on dedicated non-benchmark fixtures so S2 cannot overfit the benchmark it is authoring. Categories:
- Needle retrieval in 200K–1M chars.
- Aggregation/counting across the full context (forces coverage, punishes sampling).
- Multi-document synthesis with checkable claims.
- Repo-level code QA.
- At least one adversarial-context task (R12: injection-shaped text in the corpus).

**Scoring:** a task passes for a given arm if ≥2/3 seeds pass. Success rate = tasks passed / 20. `budget_kill` and `context_exhausted` episodes count as failures for **every** arm. Tokenized task length is recorded per episode.

**Baselines (defined now, run in S4):**
- **B1:** leaf model, single shot, full 256K native context. Runs on a dedicated config profile (`--parallel 1 -c 262144`; ~2.8 GB KV at q8_0 on this arch), launched non-concurrently with the RLM topology. Overflow policy, pre-registered: tasks whose tokenized length exceeds the window are head+tail truncated to fit (50/50 split), and the truncation is recorded — the B1-infeasible subset must be identifiable in the results. (Does raw long context beat the scaffold?)
- **B2:** deterministic map-reduce — scaffold-only chunking (the C2 chunker, verbatim) → leaf summaries → root final. No RLM agency, no REPL.

B2 is the honest control. If fixed map-reduce matches RLM on success rate, the root's agency is not earning its complexity and the project pivots to B2. **Margin rule:** "beats" means +2 tasks (10 pp) or more on success rate against each baseline. A tie with either baseline fails the S4 gate; a tie with B2 additionally triggers the pivot-to-B2 rule.

---

## 9. Vertical Slices & Gates

Each slice has a verifiable exit criterion. No slice starts before the previous gate passes. Every gate additionally requires the component test suite green (§5 Testing).

**S0 — Kill gate (one afternoon, plus one 10-minute soak).**
Both servers up with §4 flags, each on its §7 backend. Measure and record:
1. Leaf prefill at a 32K-token prompt: median of 3 cold runs, single slot, per backend. **Gate: ≥ 500 t/s — exact threshold, no tilde, no rounding — on the chosen backend. Below: RLM is dead on this hardware; stop and write up why.** (Community prior: 716–807 t/s; AMDVLK fails at 319.)
2. Concurrency scaling: aggregate prefill throughput at 1/2/4/8 slots (this sets the S2 fan-out target, §7 #1).
3. Decode on both servers, fresh and at 32K depth; root decode again **during** an active 8-slot leaf prefill (the contention number; also the stated R4 measurement condition).
4. Leaf KV cost per slot. Pre-registered expectation ≈0.45 GB per configured 40K slot (≈0.35 GB of that is the 32K chunk); ~1.5 GB indicates misconfiguration (§4). Verify q8_0 K/V engages on the hybrid arch; fall back to bf16 KV if not.
5. `timings.cache_n` sanity on the hybrid arch: a warm-slot re-query of an identical prompt must show reuse; document observed checkpoint behavior (R8). Moved here from S2 because R8 can kill §7 #3 outright.
6. 10-minute sustained 8-slot prefill: record clock and package-temperature drift (R9).
7. B1 shakedown: leaf relaunched once at `--parallel 1 -c 262144` — full-window prefill and decode spot-check, so the B1 profile (§8) is not first exercised on benchmark day. (Restores v0's full-window measurement.)
After S0: §7 numbers replaced in place; C5 budget defaults re-derived (`max_wall_clock` ≈ 2–3× predicted prefill time per task-size class).

**S1 — Minimal loop.**
Root + sandbox + C2 + C3 + one synchronous leaf call. Depth 1, no parallelism, no budgets beyond a wall-clock kill.
*Gate (operationalized):* needle task with context ≥64K tokens (asserted programmatically via `/tokenize`; ≈250K chars):
(a) control — root alone, document truncated to ≤28K tokens (the 32K window minus prompt overhead and the 90% C5 margin, so the request actually fits) by a stated deterministic rule, needle placed beyond any retained region: 0/3 attempts pass;
(b) RLM — ≥2/3 attempts pass the same checker.
Second task: a paraphrase-needle that regex cannot find, requiring ≥1 leaf call — this exercises the leaf path (otherwise untested until S2) and pre-tests the R5 confabulation surface.
Also the R1 checkpoint: if the root flails at REPL use even with environment tips in its system prompt, record it — a negative result here is a finding, not a failure to hide.

**S2 — Parallelism + prefix caching + benchmark authoring.**
C4 complete with semaphore; byte-identical leaf prefix; chunk-first prompt layout (§4); `asyncio.gather` fan-out. Benchmark authored per §8 and frozen at end of this slice; the S2 gate fixtures (fan-out, re-query) are specified as part of the same deliverable.
*Gate (all on named non-benchmark fixtures):*
(a) prefix integrity — `tokens_cached ≥ tokenized-prefix-length` on every warm-slot call of the fan-out fixture;
(b) repeated-chunk reuse — >80% token-weighted reuse on a re-query fixture;
(c) fan-out speedup ≥ 80% of S0's measured 8-slot aggregate scaling.

**S3 — Budgets + tracing.**
C5 and C6 complete. Adversarial self-test with the mock dispatcher: a task that infinite-loops in the REPL and one that requests unbounded sub-calls; plus a hard-kill mid-episode to verify the C6 durability promise (R10).
*Gate:* all terminate deterministically within budget, and the full trajectory of any episode is reconstructable from the trace store (DuckDB + referenced blob directory, episode-relative paths) alone — no logs, no stdout.

**S4 — Benchmark + quality gate.**
Run frozen benchmark: RLM vs B1 vs B2, 3 seeds each, arms interleaved (R9).
*Gate:* RLM beats both baselines per the §8 margin rule (+2 tasks / 10 pp; a tie with either baseline fails the gate, a tie with B2 triggers the pivot). Wall-clock recorded per task. Only after this gate may anyone propose `max_depth > 1` or new optimizations.

**S5 — Models are config.**
*Gate:* swap the root to a new model (target: Qwen3.8-27B — open weights announced as imminent at v0.1 time; this slice may activate early) by editing `config.yaml` only, rerun S0 raw numbers + S4 benchmark, produce one comparison row. Day-one checklist for any new model: (1) dense or MoE? (2) **attention layout — how many layers carry per-token KV? linear-state size?** (3) MTP head present? (4) chat template diffs (template hash into config_snapshot)? (5) tokenizer identity? (6) license? (7) GGUF quality (community quants stabilize over ~2 weeks). (8) multimodal components (mmproj) present but unused?

**S6 — Learning loop. UNSCHEDULED. Do not build.**
Documented only so the trace schema decision (§6) has a stated payoff. Preconditions before this slice may even be *planned*: (a) ≥500 successful episodes logged, (b) a recurring failure mode demonstrably unfixable by prompting, (c) S4 gate passed. Contents if ever activated: SFT on own successful trajectories; Voyager-style skill library persisted between episodes; RL with AgentWorld-simulated environments **for training only** — evaluation stays on the real REPL (I7). Known hazard: template collapse (RAGEN-2 precedent) — any training data must pass **input-sensitivity / cross-input-distinguishability checks** (RAGEN-2 shows the collapse evades entropy and surface-diversity metrics).

---

## 10. Known Risks

| # | Risk | Response |
|---|---|---|
| R1 | Root degrades under the scaffold (precedent: Prime Intellect's RLMEnv evaluation, Jan 2026 — untrained models incl. INTELLECT-3 scored *lower* inside an RLM harness on several benchmarks, though supplying a strategy flipped some results; the RLM paper's own Qwen3-8B root needed dedicated post-training) | S1 gate is early and explicit; environment tips in root system prompt; accept and publish a negative result rather than rationalize it |
| R2 | iGPU prefill insufficient for leaf economics | S0 is a kill gate, not a benchmark; threshold is exact (§9) |
| R3 | Prefix cache silently broken by prefix drift | `tokens_cached` continuously monitored via the prefix-integrity check (§7 #3a); prefix text hashed into `config_snapshot`, so **between-run** drift is detectable too (within-run monitoring alone cannot see it) |
| R4 | MTP path immature | MTP merged into llama.cpp 2026-05 (PR #22673) but beta: single-slot only, draft-n ≤ 2, spotty Vulkan history. Optional variant; adopt only if benchmark success unchanged AND decode ≥1.4×, measured on-box under the stated contention condition |
| R5 | Leaves confabulate on chunks (plausible-but-wrong extractions) | Aggregation tasks in the benchmark punish it; root cross-checks via code (regex/count verification in REPL) where the task allows; S1's paraphrase-needle task probes it early |
| R6 | Sandbox escape / resource abuse | No-network default (defined in C1), rlimits/Job Objects, per-episode interpreter, unconditional kill (C1, C5). Local single-user threat model — documented, not overengineered |
| R7 | Qwen3.8-27B lands with different architecture (Qwen already switched to hybrid attention once between 3 and 3.6) | S5 checklist (now incl. attention layout); the incumbent stays resident during A/B (headroom, §4) |
| R8 | Hybrid linear-attention defeats llama.cpp prompt caching (checkpoint invalidation; llama.cpp issues #18497 / #19794 / #20225 / #24055) | `cache_n` sanity check moved into S0; if reuse is broken, demote §7 #3b and keep a full-attention MoE fallback leaf listed in config |
| R9 | Thermal throttling on a mini-PC drifts wall-clock — the primary metric — across a benchmark run | S0 10-minute soak; package temp logged per episode; S4 interleaves arms so drift hits all three equally |
| R10 | Trace loss on crash | C6 per-step transaction commit, blob-before-row ordering; S3 hard-kill test verifies the "at most the current step" promise |
| R11 | Upstream llama.cpp regressions across updates | Build pinned S0→S4 and recorded in `config_snapshot`; upgrades are treated as optimizations under I5 (benchmark-neutral-or-better) |
| R12 | Injection-shaped text in the analyzed corpus steers leaf extractions (a benchmark-robustness risk under the local single-user threat model, not a security hole) | Leaf outputs treated as untrusted data by the root's cross-checks (R5); one adversarial-context task in the benchmark (§8) |

---

## 11. Out of Scope

- **vLLM on this hardware.** Historically broken for months on gfx1151; upstream support merged ~Q1 2026 and community containers now work. Still out of scope: llama.cpp wins single-stream latency and operational simplicity for this topology, and one serving stack is enough.
- **NPU.** llama.cpp has no XDNA2 backend, and the NPU stacks that do exist (Lemonade, FastFlowLM) share the same 256 GB/s memory — the NPU cannot raise the decode ceiling. Excluded.
- **Runtime world models / simulators** (I3). Lifecycle-only, S6-only.
- **Recursion depth > 1** until S4 evidence demands it.
- **Browser/pyodide sandboxing.** `subprocess` + rlimits (or Job Objects, A2) is sufficient for the threat model.
- **Multi-machine RPC clustering.** One box.
- **Any training** until S6 preconditions are met.

---

## 12. Stated Assumptions & Open Questions

- **A1 — Scaffold language is Python 3.12+.** Rationale: the REPL is Python, `asyncio` maps to the dispatcher, servers are HTTP (language-agnostic). Test framework: pytest (see §5 Testing). If the deterministic-core-in-C# convention (external reference, below) should extend here, the scaffold is small enough to port later; do not block on it.
- **A2 — Host OS. DECISION REQUIRED BEFORE S0 — the tripwire has fired: this spec is being authored on a Windows 11 machine.** The Linux path assumes: rlimits + network namespace (C1), ROCm + rocWMMA leaf backend (§4/§7), GTT tuning (§4). If the box stays Windows: C1 uses Job Objects (which also provide the process-tree kill C5 needs), network isolation is best-effort (R6 wording weakens accordingly), the ROCm path is gone — Vulkan RADV on both servers (community prior still clears the S0 gate: 716 t/s at d=32K), and shared GPU memory is driver-capped near 50% of RAM unless raised. **WSL2 is not a valid measurement environment for gfx1151.** Decision record: `Host OS: ________ — decided ____-__-__`.
- **Q1 —** Default chunk size 32K is a guess; §7 #2 sweeps it.
- **Q2 —** Truncation cap 2,000 chars is a guess; revisit after S1 traces show what the root actually reads.
- **Q3 —** Whether the root should be allowed `role="root"` self-calls (recursion) is deferred to the `max_depth` gate in S4 — the dispatcher supports the parameter, the budget forbids it by default, and `steps.depth` (§6) can represent it.
- **External references:** "Thesis B" (§2), "Conduo" (§5), and the "deterministic-core-in-C# convention" (A1) are the author's private framework references. Nothing in this spec requires them: the operative content is stated inline wherever they appear.

---

## 13. References

- Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601 (method reference; depth-1 experiments, truncated REPL views, cheap-leaf sub-calls).
- Prime Intellect — *Recursive Language Models: the paradigm of 2026* (blog, 2026-01-01, primeintellect.ai/blog/rlm) — RLMEnv evaluation behind R1; INTELLECT-3 technical report is arXiv:2512.16144.
- Qwen team — *Qwen-AgentWorld: Language World Models for General Agents*, arXiv:2606.24597 (trajectory format cited in §6; S6 simulation option).
- *RAGEN-2: Reasoning Collapse in Agentic RL*, arXiv:2604.06268 — template collapse, S6 hazard. (Original RAGEN: arXiv:2504.20073.)
- Wang et al. — *Voyager*, arXiv:2305.16291 — executable-code skill library, S6 option.
- Strix Halo performance priors (§7): kyuz0/amd-strix-halo-toolboxes results dataset (2026-05, llama.cpp b9187/b9193, benchmarks the exact §4 models); nabe2030/hip-vs-vulkan-evo-x2 (ROCm 7.2.2 vs Vulkan, FA on); llama.cpp discussion #20856 (known-good gfx1151 stack, q8_0 KV at 262K ≈ 2.7 GB on this arch).
- llama.cpp mechanics: server README (slot semantics, `timings.cache_n`, `--slot-prompt-similarity`); PR #22673 (MTP speculative decoding, 2026-05-16); issues #18497, #19794, #20225, #24055 (hybrid/recurrent prompt-cache regressions, R8); issue #11681 / discussion #22658 (`-c` split across slots).
- Model cards/configs: Qwen/Qwen3.6-27B, Qwen/Qwen3.6-35B-A3B (hybrid attention layouts, MTP heads); unsloth GGUF + MTP-GGUF repos (measured file sizes).

---

## 14. Changelog

- **v0.1 — 2026-08-12.** Applied external review (research-verified against mid-2026 sources). Headline changes: (1) both models identified as hybrid linear-attention — KV math corrected (~4× cheaper; committed ~54 GB, headroom ~74 GB), attention layout added to the S5 checklist, R8 added; (2) prefix-caching metric redefined token-weighted (`tokens_cached` replaces `cache_hit BOOL`), prompt layout pinned chunk-first, §7 #1 demoted to #3 with an honest structural ceiling, S2 gate split into integrity/reuse/speedup; (3) fan-out gate re-based on S0-measured concurrency scaling instead of an a-priori 3×; (4) §4 "12 vs 48 GB / 2.5×" arithmetic replaced with a derivable comparison; (5) S0 protocol pre-registered (exact 500 t/s threshold, backend named, contention + soak + KV + cache_n checks added); (6) C1/C4 bridge, chunker ownership (C2), final-answer channel, retry/budget semantics, root-window policy (`context_exhausted`), and C6 write mechanics specified; (7) steps/episodes schema extended (parent linkage, status, slot, retries, depth, seeds, build pinning); (8) benchmark authoring made an S2 deliverable, freeze-date contradiction resolved (end of S2, gates on non-benchmark fixtures), scoring/margin rules added, B1 profile + overflow policy defined; (9) R1 attribution corrected to Prime Intellect's RLMEnv post; vLLM/NPU out-of-scope wording updated to 2026 reality; MTP status updated (merged, beta, single-slot); (10) A2 escalated to a pre-S0 decision record (authored on Windows 11); amendment rule, references (§13), and this changelog added. Post-rewrite verification (3-agent) fixes folded in: S1 control document sized to actually fit the root window (≤28K tokens); per-slot KV pre-registration re-based to the configured 40K slot (≈0.45 GB); S0 gains a B1 full-window shakedown (item 7), restoring v0's full-window measurement; §7 backend deltas reconciled to their sources (+13% at d=32K, +41% at pp16K); C1/C5/R6 lifetimes stated per-episode; benchmark pinned at exactly 20 tasks with an explicit B1 tie rule; optimization order rationale stated (MTP after chunk tuning: beta, single-slot); S6's v0 "format diversity checks" deliberately superseded by input-sensitivity checks (RAGEN-2). Numbers in §7 remain priors until S0 runs on-box.
- **v0 — original.** Pre-implementation constitution, all §7 numbers hypotheses.
