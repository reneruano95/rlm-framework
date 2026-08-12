# ARCHITECTURE.md — RLM Runtime (working name: `rlm-halo`)

**Spec version:** `rlm-runtime-spec-v0`
**Status:** Pre-implementation constitution. No code exists. Every number in §7 is a hypothesis until S0 replaces it with a measurement.
**Hardware target:** AMD Ryzen AI MAX+ 395 "Strix Halo", 128 GB unified memory, ~256 GB/s bandwidth.
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

Reference: Zhang, Kraska, Khattab — *Recursive Language Models*, arXiv:2512.24601.

---

## 2. Thesis

This is Thesis B applied to context management. Deterministic logic renders all control decisions: truncation caps, recursion budgets, routing, termination, and logging are scaffold configuration. The LLM's only powers are (a) writing code that runs in a sandboxed REPL and (b) requesting sub-calls through a dispatcher it does not control. Ground truth lives in the real REPL and in shared artifacts (traces), never in model output taken at face value.

Corollary for performance: on this hardware the binding constraint is **leaf prefill throughput** (compute-bound on the iGPU), not root decode speed. The architecture optimizes for many cheap parallel leaf calls over one smart slow context window.

---

## 3. Invariants

Violating any of these is a bug, regardless of benchmark results.

- **I1 — The LLM proposes; the scaffold disposes.** Truncation caps, budgets, routing, and termination live in `config.yaml` and scaffold code. No model output, prompt content, or REPL side effect may alter them at runtime.
- **I2 — Context by reference.** The full context never enters any model's message array. Models see only scaffold-truncated views (`print` output capped by C3).
- **I3 — Real environment only in the runtime loop.** No simulator or world model may sit between the root and the REPL, or between the scaffold and the final answer. World models are lifecycle tools (training data generation, S6) — never runtime components.
- **I4 — Every episode is a trajectory.** All runs are logged as `(state, action, observation)` steps with a terminal outcome (§6). A run that is not logged did not happen.
- **I5 — Quality gates precede performance work.** No optimization is merged without the frozen benchmark (§8) showing task success unchanged or better. Tokens/sec alone never justifies a change.
- **I6 — Models are config.** Model identity, quant, context sizes, ports, and budgets live in one `config.yaml`. Swapping a model must require zero code changes (S5 gate).
- **I7 — Evaluate only against the real REPL.** Future training (S6) may consume synthetic or simulated environments; evaluation gates may not.

---

## 4. Serving Topology (Capa 0)

Two `llama-server` processes, both resident simultaneously. Not LM Studio — we need `--parallel`, continuous batching, explicit KV cache types, and prompt-cache control.

| | **ROOT** | **LEAF** |
|---|---|---|
| Model | Qwen3.6-27B (dense) | Qwen3.6-35B-A3B (MoE, ~3B active) |
| Quant | Q4_K_M (~17 GB) | Q4_K_M (~21 GB) |
| Port | 8080 | 8081 |
| Context | **32K hard cap** | per-slot: `chunk_size + overhead` (~40K default) |
| Parallel | `--parallel 1` (2 max) | `--parallel 8 --cont-batching` |
| KV cache | q8_0 | q8_0 (`--cache-type-k q8_0 --cache-type-v q8_0`) |
| Variant | MTP GGUF if it passes R4 check | standard |
| Role | plan, write REPL code, decide sub-calls, emit final | chunk-level extraction/summary/QA |

**Why the root gets only 32K:** the root never sees the raw context (I2). It sees truncated prints and sub-call results. A large root window is wasted KV and slower prefill for zero benefit. If the root "needs" more context, that is a scaffold smell — the work belongs in the REPL or in leaves.

**Chunk size is the parallelism lever.** Leaf slots are sized to the chunk budget, not to the model's 256K maximum. At ~32K chunks with q8_0 KV, a leaf slot costs ≈1.5 GB (hypothesis; measure in S0), so 8 slots ≈ 12 GB instead of ~48 GB at full window. Same total prefill work, ~2.5× more usable concurrency.

**Memory budget (128 GB total):**

| Item | GB |
|---|---|
| Root weights | 17 |
| Leaf weights | 21 |
| Leaf KV (8 × 32K slots, q8_0) | ~12 |
| Root KV + overhead | ~4 |
| OS + scaffold + DuckDB | ~8 |
| **Committed** | **~62** |
| **Headroom** | **~66** |

Headroom is deliberate. It exists for: (a) a third resident server if S6 ever needs one, (b) A/B running a candidate model next to the incumbent during S5 swaps, (c) larger chunk experiments. Do not spend it by default.

**Prefix caching contract (critical):** all leaf calls share one **byte-identical** system prefix. No timestamps, run IDs, task IDs, or counters anywhere in the prefix — run metadata travels in the trailing user segment or stays out-of-band in the trace. Cache hit rate is a monitored metric with an alert threshold (S2), not a one-time check, because prefix drift fails silently.

---

## 5. Scaffold Components (Capa 1)

Six components. Python 3.12+ (assumption A1, §12). Each lists what it must NOT do — those are the load-bearing lines.

**C1 — SandboxManager.** Spawns one isolated Python interpreter per RLM task (`subprocess` + resource limits: CPU time, memory, no network by default). Persistent across REPL turns within a task (Jupyter-like: variables survive). Killed unconditionally at task end or budget breach.
*Must not:* share state between tasks; grant network unless the task config explicitly enables it.

**C2 — ContextLoader.** Materializes the input as variable `context` inside the sandbox before the root's first turn. Handles str, bytes, file paths, and lists of documents.
*Must not:* place any part of the raw context into any message array (I2).

**C3 — OutputTruncator.** Hard cap on all REPL output returned to the root (default 2,000 chars, config). Appends a marker: `[truncated: showing 2000 of 184,203 chars]`.
*Must not:* be overridable by model output or by code running in the sandbox (I1). The cap is applied scaffold-side, after execution.

**C4 — LLMDispatcher.** Exposes `await llm_query(prompt, role="leaf")` inside the sandbox. Routes by role to the correct server. Owns an `asyncio.Semaphore` equal to the target server's `--parallel`. Retries with backoff; per-call timeout. Returns the result as a Python object in the caller's REPL — never auto-injected into the root's messages.
*Must not:* let the model choose servers, ports, or semaphore size; exceed `--parallel` (queueing upstream hides latency pathologies from the trace).

**C5 — BudgetEnforcer.** Per-task limits from config: `max_depth` (default **1** — root + leaves, no grandchildren), `max_subcalls` (default 32), `max_wall_clock` (default 15 min), `max_total_tokens`. On breach: kill sandbox, mark episode `budget_kill`, persist partial state and full trace. Termination is deterministic and unconditional.
*Must not:* warn-and-continue; accept model requests for extensions. Raising `max_depth` above 1 requires S4 evidence that depth-1 is the binding constraint.

**C6 — TraceLogger.** Writes every step to DuckDB/Parquet in the trajectory schema (§6), synchronously enough that a crash loses at most the current step.
*Must not:* sample, summarize, or truncate the stored record (store full observation as a blob ref; only the *root's view* is truncated, per C3).

**Dependency rule (lint-enforced, same pattern as Conduo):** C1–C3, C5, C6 must not import C4 or any LLM client. Only C4 talks to model servers. The scaffold must be able to run a full episode with a mock dispatcher (used in S3's runaway test).

---

## 6. Data Layer — Trajectory Schema

The one decision made today that pays later: traces are stored **trajectory-shaped**, not log-shaped. The tuple `(state, action, observation, outcome)` is simultaneously the SFT format of the original RLM paper, the training format of Qwen-AgentWorld, and the raw material for a Voyager-style skill library. Zero runtime cost; three learning paths kept open.

**`episodes`**

| column | type | notes |
|---|---|---|
| episode_id | UUID | |
| task_id | TEXT | benchmark task or ad-hoc |
| task_hash | TEXT | sha256 of task text |
| started_at / ended_at | TIMESTAMP | |
| outcome | ENUM | `success` \| `fail` \| `budget_kill` \| `error` |
| final_answer_ref | TEXT | parquet blob path |
| config_snapshot | JSON | models, quants, budgets, chunk size, truncation cap |
| scaffold_git_sha | TEXT | |
| benchmark_version | TEXT | nullable |

**`steps`**

| column | type | notes |
|---|---|---|
| episode_id | UUID | FK |
| step_idx | INT | |
| actor | ENUM | `root` \| `leaf` |
| action_type | ENUM | `repl_exec` \| `llm_call` \| `final` |
| action_payload | TEXT | code or prompt (full) |
| observation_view | TEXT | what the root actually saw (post-C3) |
| observation_full_ref | TEXT | parquet blob path (pre-truncation) |
| tokens_in / tokens_out | INT | |
| latency_queue_ms / latency_prefill_ms / latency_decode_ms | INT | from server timings |
| cache_hit | BOOL | leaf prefix cache |

Both `observation_view` and `observation_full_ref` are stored deliberately: the first reconstructs what the root knew (debugging, SFT); the second preserves ground truth (evaluation, world-model data).

---

## 7. Performance Model & Optimization Order (Capa 2)

**All numbers below are hypotheses. S0 replaces them with measurements; this section is then updated in place.**

Decode ceiling (bandwidth-bound): `t/s ≈ 256 GB/s ÷ (active_params × bytes/param)`.
- Root (27B dense, Q4): ceiling ~15 t/s, expect 10–13. Acceptable: root emits code, not prose.
- Leaf (3B active, Q4): expect 60–90 t/s.
Prefill is compute-bound and is the real budget: a 25K-token leaf chunk on this iGPU is the unit cost of everything.

Optimizations in strict order of expected return. Per I5, each ships only with benchmark-neutral-or-better evidence, and each has one owning metric:

1. **Prefix caching** — metric: leaf `cache_hit` rate (target >80%). Largest single win; see contract in §4.
2. **Parallel dispatch** — metric: measured speedup on an 8-leaf fan-out task vs serial (target ≥3×; will be <8× because prefill saturates compute).
3. **MTP on the root** — metric: root decode t/s (target ≥1.4×) *and* unchanged benchmark success (R4).
4. **Chunk-size tuning** — metric: wall-clock per task at fixed quality, swept over {16K, 24K, 32K, 48K}.
5. **ROCm/Vulkan build flags** (gfx1151, rocWMMA flash-attention vs Vulkan) — metric: raw prefill t/s. Worth 20–30%; done last because it is fiddly and orthogonal.

---

## 8. Measurement (Capa 3)

**Primary metric:** wall-clock per task at fixed quality. Never t/s in isolation (I5).

**Secondary metrics:** leaf prefill t/s; root decode t/s; leaf cache hit rate; per-episode time split (REPL exec / LLM wait / scaffold overhead) — all derivable from the `steps` table, by design.

**Frozen benchmark:** ~20 tasks, programmatically verifiable (exact match or checker function), frozen and versioned **before S2** so optimization can never overfit a moving target. Categories:
- Needle retrieval in 200K–1M chars.
- Aggregation/counting across the full context (forces coverage, punishes sampling).
- Multi-document synthesis with checkable claims.
- Repo-level code QA.

**Baselines (defined now, run in S4):**
- **B1:** leaf model, single shot, full 256K context. (Does raw long context beat the scaffold?)
- **B2:** deterministic map-reduce — scaffold-only chunking → leaf summaries → root final. No RLM agency, no REPL.

B2 is the honest control. If fixed map-reduce matches RLM on success rate, the root's agency is not earning its complexity and the project pivots to B2.

---

## 9. Vertical Slices & Gates

Each slice has a verifiable exit criterion. No slice starts before the previous gate passes.

**S0 — Kill gate (one afternoon).**
Both servers up with §4 flags. Measure raw prefill and decode on both, at chunk-sized and full-window prompts. Record leaf KV cost per slot.
*Gate:* leaf prefill ≥ ~500 t/s at 32K chunks. **Below that, RLM is dead on this hardware — stop and write up why.** §7 numbers replaced with measurements.

**S1 — Minimal loop.**
Root + sandbox + C2 + C3 + one synchronous leaf call. Depth 1, no parallelism, no budgets beyond a wall-clock kill.
*Gate:* solves a needle-in-haystack at 200K chars that the root alone (32K window) provably cannot. Also the R1 checkpoint: if the root flails at REPL use even with environment tips in its system prompt, record it — a negative result here is a finding, not a failure to hide.

**S2 — Parallelism + prefix caching.**
C4 complete with semaphore; byte-identical leaf prefix; `asyncio.gather` fan-out. Benchmark frozen at end of this slice.
*Gate:* measured cache hit >80% and ≥3× speedup on an 8-leaf fan-out vs serial.

**S3 — Budgets + tracing.**
C5 and C6 complete. Adversarial self-test with the mock dispatcher: a task that infinite-loops in the REPL and one that requests unbounded sub-calls.
*Gate:* both terminate deterministically within budget, and the full trajectory of any episode is reconstructable from DuckDB alone (no logs, no stdout).

**S4 — Benchmark + quality gate.**
Run frozen benchmark: RLM vs B1 vs B2, 3 seeds each.
*Gate:* RLM beats both baselines on success rate. Wall-clock recorded per task. Only after this gate may anyone propose `max_depth > 1` or new optimizations.

**S5 — Models are config.**
*Gate:* swap the root to a new model (target: Qwen3.8-27B when weights land) by editing `config.yaml` only, rerun S0 raw numbers + S4 benchmark, produce one comparison row. Day-one checklist for any new model: (1) dense or MoE? (2) MTP head present? (3) chat template diffs? (4) license? (5) GGUF quality (community quants stabilize over ~2 weeks).

**S6 — Learning loop. UNSCHEDULED. Do not build.**
Documented only so the trace schema decision (§6) has a stated payoff. Preconditions before this slice may even be *planned*: (a) ≥500 successful episodes logged, (b) a recurring failure mode demonstrably unfixable by prompting, (c) S4 gate passed. Contents if ever activated: SFT on own successful trajectories; Voyager-style skill library persisted between episodes; RL with AgentWorld-simulated environments **for training only** — evaluation stays on the real REPL (I7). Known hazard: template collapse (RAGEN-2 precedent) — any training data must include format diversity checks.

---

## 10. Known Risks

| # | Risk | Response |
|---|---|---|
| R1 | Root degrades under the scaffold (INTELLECT-3 precedent: RLM harness *hurt* untrained models) | S1 gate is early and explicit; environment tips in root system prompt; accept and publish a negative result rather than rationalize it |
| R2 | iGPU prefill insufficient for leaf economics | S0 is a kill gate, not a benchmark |
| R3 | Prefix cache silently broken by prefix drift | `cache_hit` is a continuously monitored metric with alert threshold, never a one-time check |
| R4 | MTP GGUFs immature | MTP is an optional variant; adopt only if benchmark success is unchanged AND decode ≥1.4× |
| R5 | Leaves confabulate on chunks (plausible-but-wrong extractions) | Aggregation tasks in the benchmark punish it; root cross-checks via code (regex/count verification in REPL) where the task allows |
| R6 | Sandbox escape / resource abuse | No-network default, rlimits, per-task interpreter, unconditional kill (C1, C5). Local single-user threat model — documented, not overengineered |
| R7 | Qwen3.8-27B lands with different architecture (MoE? no MTP head? new layout?) | S5 checklist; the incumbent stays resident during A/B (headroom, §4) |

---

## 11. Out of Scope

- **vLLM on this hardware.** Documented multi-month failure mode on Strix Halo; llama.cpp is the stack.
- **NPU.** Inference stacks route LLM work to the iGPU; TOPS figures are irrelevant here.
- **Runtime world models / simulators** (I3). Lifecycle-only, S6-only.
- **Recursion depth > 1** until S4 evidence demands it.
- **Browser/pyodide sandboxing.** `subprocess` + rlimits is sufficient for the threat model.
- **Multi-machine RPC clustering.** One box.
- **Any training** until S6 preconditions are met.

---

## 12. Stated Assumptions & Open Questions

- **A1 — Scaffold language is Python 3.12+.** Rationale: the REPL is Python, `asyncio` maps to the dispatcher, servers are HTTP (language-agnostic). If the deterministic-core-in-C# convention should extend here, the scaffold is small enough to port later; do not block on it.
- **A2 — Linux host.** rlimit-based sandboxing and the ROCm/gfx1151 build path assume Linux. **If the box stays on Windows/LM Studio, C1 changes (Job Objects) and Vulkan becomes the default backend — flag before S0.**
- **Q1 —** Default chunk size 32K is a guess; §7 step 4 sweeps it.
- **Q2 —** Truncation cap 2,000 chars is a guess; revisit after S1 traces show what the root actually reads.
- **Q3 —** Whether the root should be allowed `role="root"` self-calls (recursion) is deferred to the `max_depth` gate in S4 — the dispatcher supports the parameter, the budget forbids it by default.
