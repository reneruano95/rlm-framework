# S6 — Long-horizon agent: design

**Date:** 2026-08-22 · **Status:** design, awaiting owner review. Nothing here amends `ARCHITECTURE.md` or `DIRECTION.md`; §6 and §9 list what would.
**Trigger:** the owner read NVIDIA's AVO announcement (2026-08, "100.00 RHAE on ARC-AGI-3") and said "that is the kind of thing i want to build."
**Inputs:** the AVO blog post and arXiv:2603.24517; ARC Prize docs, the ARC-AGI-3 agents repo, the community leaderboard; Duck Harness, Polyphony, VISTA, Tycho; the harness literature listed in §3.3; `ARCHITECTURE.md` v0.3.16, `DIRECTION.md` (2026-08-16), `docs/research/2026-08-20-rlm-paper-fidelity-and-next-steps.md`; `traces/rlm.duckdb` at 614 episodes; the working tree at `f19fdca`.
**Evidence discipline:** figures marked **[V]** were verified in this repo or its store during this session and are reproducible from the commands in §10. Figures marked **[R]** come from the research sweep and are cited to a source but were **not** independently re-measured. No figure is presented as measured that was not.

---

## 0. The verdict

**Build a constraint-shaped long-horizon harness for local models, proven on ARC-AGI-3 offline — conditional on a one-night measurement (§5.1) that can end the programme before it costs a decision.**

### 0.1 What is explicitly not being built

| Not this | Why |
|---|---|
| **"AVO"** | Not a well-formed target. Its supervisor's trigger threshold, mechanism (model call? heuristic?), invocation count, prompt, and ablated contribution are undisclosed in every source **[R]**. Its context management — the mechanism a 32K local window most needs — is described nowhere, though seven days of compiler output cannot literally sit in a window, so it must exist. Its entire ARC tool layer is undisclosed. Building it means inventing it and calling it a reimplementation. |
| **100.00 RHAE** | A saturated ceiling (RHAE caps at 115.0/level), already reported by VISTA and Tycho on all 183 public levels, both on Opus 5 **[R]**. AVO's claim carries no scorecard URL and appears on neither leaderboard; the `arcprize/ARC-AGI-Community-Leaderboard` repo has 21 submission dirs and 47 PRs, none AVO **[R]**. ARC Prize states it will never report public-set scores officially and that evaluating on that set "is emphatically not a valid measure of progress towards AGI" **[R]**. Reproducing it would not be a new result. |
| **A general agent framework** | `ARCHITECTURE.md` §1's exclusion stands and is still correct. What follows is one environment channel and one supervision shape, not an orchestrator. |
| **A second, unsandboxed agent surface** | The capability-first path (shell + git + filesystem, the true analogue of AVO's documented kernel half) requires exactly what C1 forbids by kernel token. It is deferred to §5.5 and is a governance decision, not a refactor. |

### 0.2 What is being built, and why it is defensible

1. **The empty space is verified.** Every published open-weight ARC-AGI-3 result uses vLLM. Zero use llama.cpp or GGUF. Zero use consumer unified-memory hardware. Nothing below 27B has any published result at all **[R]**. This is the same shape of gap the 2026-08 prior-art sweep found for serving co-design (`ARCHITECTURE.md` §13).
2. **The winning local harness is already this architecture.** Duck Harness — 1st place, ARC-AGI-3 Milestone 1, $25K — is Qwen3.6-27B FP8 served locally, driving a Python REPL that exposes `current_frame.ascii`, `current_frame.segmentation`, `transitions`, `last_action_result` as variables and **deliberately hides the raw numeric grid from the model** **[R]**. That is I2 ("context by reference: the full context never enters a message array") arrived at independently. Polyphony (19.8% public, $115 — cheapest on the community board by 12×) grows a world model as plain editable Python files and accepts an edit only if replaying it reproduces observed transitions exactly **[R]** — that is §5.4's learned registry, applied to ARC.
3. **The literature's split favours this project.** The reported divide is by mechanism, not parameter count. Harnesses that **instruct** (prose skills, playbooks the model may ignore) fail below ~32B: Qwen3-32B loads an available skill 25.1% of the time against ~96% frontier, and harness-following decays 0.52 → 0.13 within one episode (arXiv:2605.30621); skill libraries give Qwen3-4B/8B/14B ~zero gain on AppWorld while Qwen3-235B gains +28.1pp (arXiv:2608.20274). Harnesses that **constrain** (control externalised, state held outside the model) lift 8–35B models most: Self-Harness +104/113/132% on Qwen3.5-35B-A3B; HarnessX's *weakest* agent (Qwen3.5-9B) gains most, +44.0pp ALFWorld / +18.2pp SWE-bench; LivePlan gives weak executors +12 to +15pp against +5.45pp for a strong one **[R]**. I1 — "the scaffold owns truncation, budgets, routing, termination; no model output may alter them at runtime" — makes rlm-halo constraint-shaped by constitution.
4. **The environment is free, offline, deterministic, and I3-legal.** `Arcade(operation_mode=OperationMode.OFFLINE)` runs the real `ARCEngine` in-process at ~2,000 FPS with no API key and no rate limits, and each game's `metadata.json` ships per-level human `baseline_actions`, so RHAE computes locally **[R]**. It air-gaps, which is the appliance's own story. It also supplies graded long-horizon trajectories by the thousand at zero authoring cost — S6's "≥500 successful episodes" precondition solved without hand-authoring, against v1's 30 corpora which took three designs and two documented failures.
5. **Everything expensive is already built.** Hard budgets, replay-from-store-alone (S3: 12/12, zero `rlm/` changes), slot discipline, the sandbox, the trace store. Every ARC harness in the prior art improvises these badly **[R]**.

### 0.3 Why the verdict is conditional

The longest episode in this project's entire history is **146 root turns** **[V]**. AVO ran seven days. §3 sets out three blockers whose severity is currently unknown rather than merely unfavourable, and §5.1 buys all four deciding numbers in one unattended night with **zero `rlm/` changes**. The asymmetry is the point: a "no" ends the debate before it costs a decision, and `DIRECTION.md` is left undisturbed.

---

## 1. What the AVO post actually supports

The **GPU-kernel half is credible and documented** (arXiv:2603.24517: 7 days continuous, 500+ optimisation directions explored, 40 committed kernel versions, +3.5% over cuDNN and +10.5% over FlashAttention-4 on B200) **[R]**. Its formalism is `Vary(P_t) = Agent(P_t, K, f)` — replacing FunSearch's sampling operator with an agent — where `P_t` is the full lineage of solutions and scores, `K` is a knowledge base, and `f` is a **vector-valued, correctness-gated scorer** (one score per test config; correctness failure ⇒ score 0 regardless of throughput). The commit policy persists a version only if it passes correctness **and** matches-or-improves the best committed score; failed attempts stay in the agent's trajectory but never enter the lineage.

**The load-bearing observation:** that loop tolerated ~460 failures to bank 40 successes. That ratio *is* the architecture. Persistent memory and the supervisor exist to make a 12:1 failure ratio survivable over a week — they are not the point, they are what the point requires.

The **ARC half is a press release** (§0.1). It was also silently amended 8 hours after publication to correct public/semi-private wording **[R]**, and NVIDIA itself writes that the numbers "should not be interpreted as a direct measurement of the performance contribution of AVO."

### 1.1 The metric, and why it is hostile to this box

`level_score = min(((baseline_actions / actions_taken) ** 2) * 100, 115.0)`, zero for an incomplete level, with a **hard cutoff at 5× the human baseline per level**, levels played in sequence — so exceeding 5× zeroes that level *and every level after it in that game*. Only **environment actions** count: "tool calls, reasoning steps, or retries within the model itself, are not counted" **[R]**.

That is a metric purpose-built for the frontier regime: think enormously, act rarely. A local 27B is on the wrong side of both terms at once — it cannot think enormously (§3.2) and cannot see enough per action (§3.1), so it takes more actions, and the penalty is squared in exactly that quantity. Take 3× human actions and score 11%; take 5× and score zero.

**This is a design constraint from the first line of code, not a scoring detail.** Every milestone below is scored on *actions per level relative to `baseline_actions`*, never on level completion. A harness that clears levels at 6× human cost has scored zero and will look like progress.

---

## 2. What rlm-halo already is

### 2.1 Store census **[V]**

`traces/rlm.duckdb`: 614 episodes, 36,031 steps.

| arm | episodes | outcomes | leaf calls |
|---|---|---|---|
| `rlm` | 207 | **205 success**, 1 `context_exhausted`, 1 `budget_kill` | **85** |
| `rlm-restricted` | 30 | 5 success, 15 `error`, 5 `budget_kill`, 5 `fail` | 12,367 |
| `b1` | 117 | 0 success | 117 |
| `b2` | 117 | 59 success, 55 fail, 3 error | 21,220 |
| `b3` | 116 | 4 success, 111 fail, 1 error | 115 |
| pre-bench | 27 | 27 success | 7 |

Longest episodes by root turn: 146 (`rlm-restricted`), 116 (`rlm`), 89 (`rlm-restricted`) **[V]**.

**Two conclusions follow, and both are load-bearing.**

- **Benchmark v1 has no headroom.** 205/207 with 85 leaf calls across 207 episodes. The mechanism a supervisor or a memory layer would improve — delegation — is essentially unexercised, and the score is saturated. **v1 cannot score any part of this design.** This is not a cost of the pivot; it was already true on 2026-08-20.
- **The arm that delegates is broken, not merely unmeasured.** 5 successes in 30 episodes, 15 of them `error`. §5.2 of the 2026-08-20 brief already owns this.

### 2.2 Config, verified **[V]**

`window_tokens: 32768` (`config.yaml:506`) · `root_window_kill_fraction: 0.90` (`:410`) ⇒ **29,491 usable tokens** · `truncation_cap_chars: 2000` (`:294`) · `max_identical_turns: 3` (`:401`) · `max_wall_clock_s: 1300` (`:362`), `restricted_max_wall_clock_s: 2100` (`:387`).

### 2.3 The AVO mechanisms against what exists

| AVO mechanism | rlm-halo today | Distance |
|---|---|---|
| **Persistent memory** — disclosed as transcript + git-committed lineage with scores; no embedding store, no retrieval **[R]** | Within episode: `RootConversation` append-only (`rlm/rootclient.py:227-231`, D26). Across episodes: **nothing semantic.** Fresh uuid4, fresh AppContainer, fresh `BudgetEnforcer`, fresh conversation per episode. The DuckDB store is write-only from the runtime's perspective — its only readers are `rlm replay`, `rlm export`, and the bench ledger **[R]** | The git-lineage-with-scores has **no counterpart**. This is the single largest missing piece, and §3.4 explains why it cannot simply be added |
| **Supervisor** — "reviews the overall evolutionary trajectory and steers the search toward several candidate optimization directions"; named triggers are exhausted exploration and "unproductive cycles of edits that repeatedly fail to improve scores" **[R]** | ~24 mechanisms that watch a signal and act; **21 terminate or record, 2 redirect, 1 passively observes** **[R]**. `BudgetEnforcer._breach()` is irreversible and single-valued by spec (`ARCHITECTURE.md:143` — must not warn-and-continue, must not accept model requests for extensions) | **This is the sharpest gap, and it is a gap in *content*, not plumbing.** `note_turn` (`rlm/budget.py:253-280`) already detects AVO's second named trigger — consecutive turns with an identical (cell, observation) pair. AVO's response is to steer. rlm-halo's is `raise BudgetBreach(BUDGET_KILL, 'max_identical_turns')`. The one existing redirect (`repetition_observation`, `rlm/episode.py:320-336`) is a **fixed config-generated string with zero episode-specific content**, fired at cap−1, followed by a kill one turn later |
| **Environment tools** | `llm_query`, `final_answer`, `AnswerGuard`, `context`/`chunks`; child has zero egress by kernel token | New bridge verb needed; §5.3 keeps it cheap by executing the effect scaffold-side |
| **Loop surviving one invocation** | Episode is the unit; §6's crash rule **tombstones rather than resumes**, explicitly because "the sandbox interpreter heap is not stored" — correct for a REPL whose observable surface is wholly in the transcript, structurally wrong for an environment with hidden state | Deferred to §5.4; the prototype does not need it |

**The reframe that makes this tractable:** rlm-halo does not need a supervisor built. It needs 21 kills converted into redirects, starting with the one whose detector already fires.

---

## 3. The blockers, with numbers

### 3.1 The window

29,491 usable tokens **[V]**, append-only by D26, with **no compaction, summarisation, or trimming anywhere in the codebase** **[R]**. Measured burn on document QA is ~354 tokens/turn **[R]**; one real episode ended `context_exhausted / root_window` **[V]**.

One ARC observation is a 64×64 grid = 4,096 cells. The official harness renders it as one Python int list per row and estimates ~1 character per token, putting a single frame at **~4K tokens at the floor and ~12K as actually rendered** **[R]** — four to twelve times the LEAF's independently measured fact-cliff bracket of **[989 tokens pass, 1,003 fail]** (12/12 vs 0/12, Fisher p = 7.4e-07; amended 2026-08-23, superseding [967, 1,022]) **[R]**. The instruction-side companion — 30/30 false positives at a 1,024-token window against 0/21 at 640, p = 8.7e-15 — is a different quantity on the same leaf, and the second model family reproduces only the coarse 640-vs-1,024 refusal step (gemma-4, whose own SWA window is exactly 1,024). Applied to the ROOT here by same-family analogy, not by measurement **[R]**.

Turns available, as a function of tokens per observation (assuming ~60 tokens of reply and scaffolding per turn):

| tokens/frame | at 32K window (29,491 usable) | at 262K window (235,930 usable) |
|---|---|---|
| 400 | ~64 turns | ~513 turns |
| 1,000 | ~27 turns | ~222 turns |
| 2,000 | ~14 turns | ~114 turns |
| 4,000 | ~7 turns | ~58 turns |
| 12,000 | ~2 turns | ~19 turns |

**Two facts keep this from being fatal.** The 32K window is a config choice, not a memory limit: root KV is 34.0 KiB/token, so the model's full 262,144-token native window costs **8.5 GiB of 128 GB**, and the official ARC harness itself assumes `MAX_CONTEXT_LENGTH: 175,000` with front-trimming **[R]**. And the correct handling of a 64×64 frame is **never to show it to the model** — put it in a REPL variable and compute over it, where a diff between consecutive frames is a five-token observation. That is not speculation; it is what the Milestone-1 winner does **[R]**, and it is I2 restated.

### 3.2 The per-turn cost

Median root turn is **15.76 s** (n=264 episodes), of which **~8.3 s is llama.cpp creating two 189.65 MiB Gated-DeltaNet context checkpoints at 45.6 MiB/s** on a ~215 GB/s box — 0.02% of memory bandwidth, smelling like per-tensor synchronous device→host readback. In the whole logged session: **18 created, 15 erased as redundant, zero ever restored** **[R]**.

`-ctxcp` and `-cms` are live flags on the pinned build and are **untested** **[R]**. If the tax is removable, per-turn cost falls 27–53% — the difference between a 500-turn run costing 2.2 h and 1.1 h. One community summary asserts `-ctxcp 0` forces full prompt reprocessing every request; unconfirmed against primary source **[R]**, which is exactly what §5.1 measures.

Related and free: **drop DFlash2 for short-turn loops.** It buys ~3× decode for +1.85 s fixed overhead — a net loss below ~10 output tokens, and an ARC-style turn emits one action **[R]**.

### 3.3 Process endurance

Upstream llama.cpp issue **#23181**: 1,500–2,000 sequential chat-completion requests crash the server in 30 min–3 h, the crash always immediately follows a "created context checkpoint" line, and `--cache-ram 0` explicitly does **not** disable checkpoints **[R]**. `--cache-ram 0` is set on both servers (`config.yaml:200,252`) **[V]**.

**And no single `llama-server` process in this project's history is provable to have run more than ~35 minutes.** The 21-hour flat-throughput result comes from a bench that relaunches the leaf per block; `traces/lifecycle.jsonl` records no `server_start`/`server_stop`, and `root-server.log` truncates on launch **[R]**. This is not merely unmeasured — it is unprovable from existing data, and it gates every path.

### 3.4 Why cross-episode memory is not a feature you can just add

Blocked by I1 (model output must never alter scaffold behaviour at runtime) and I4's state rule (`ARCHITECTURE.md:173`). The only I1-clean shape is the one §5.4 of the 2026-08-20 brief already names: propose offline from traces, apply **between** episodes as versioned sha-pinned config, roll back by refinement id, gate on a held-out split.

**The under-appreciated consequence, and it must be pre-registered:** this destroys §8's blocked (task, seed) design. Once episode *n* is influenced by episodes 1..*n*−1, cells stop being exchangeable, and the sign test and paired bootstrap CI stop meaning what they say. Any memory layer arrives with a new statistical design or it does not arrive.

`sandbox_bootstrap/` (`rlm/sandbox/manager.py:99-139`) is the one existing cross-episode, sandbox-readable, RX-only channel — the natural home for scaffold-written skills, provided its bytes are hashed into `config_snapshot` **[R]**.

---

## 4. Decisions taken (decision record)

Taken by the owner in session on 2026-08-21/22. Re-litigate only with new facts, per `ARCHITECTURE.md` §11.

| # | Decision | Consequence |
|---|---|---|
| **D-A1** | All three of AVO's aspects — harness thesis, long-horizon capability, external benchmark — pursued **sequenced**, with a kill gate per step | No step begins before its predecessor's gate resolves |
| **D-A2** | **The root is local, always.** No frontier model in the scored loop, at any stage | The ablation axis moves from model to harness: supervisor on/off on a fixed local root. This is a *better* ablation than AVO's — they varied neither |
| **D-A3** | The second domain (the GPU-kernel analogue) is **deferred** until the loop is proven. The loop is proven on ARC-AGI-3, which supplies environment and scoring for free | §5.5 stays a stub. §0.1's governance question is not asked yet |
| **D-A4** | **Single track.** §5.2 of the 2026-08-20 brief is shared foundation; benchmark v2 is **parked, not dropped** | §6.3 records the standing argument for revisiting this after §5.1 |
| **D-A5** | Gate 1 is endurance **and** supervisor, run as **control and treatment arms of one experiment** — not as a conjunction | A failure is always attributable: control degrades ⇒ endurance failed; control holds and the delta is ~0 ⇒ the supervisor is what is wrong |
| **D-A6** | *(this document)* The target is **not** 100.00 RHAE. The available new result is the first llama.cpp / GGUF / unified-memory entry on ARC-AGI-3 at any score, and the Kaggle track where local air-gapped single-GPU is mandatory and 1.21% is the number to beat **[R]** | Every milestone is scored on actions-per-level against `baseline_actions` (§1.1) |

---

## 5. The programme

### 5.0 Free repairs, riding along

Independent of every gate; do them first because they are hours, not days. These *do* touch `rlm/` — the "zero `rlm/` changes" claim in §5.1 is about the soak script itself, which imports nothing from the runtime and can run before, during, or after these land.

1. **`rlm/cli.py:808`** calls `registry.render_root(category)` without `restricted=`, while the live path (`rlm/episode.py:942`) passes `restricted=self.restrict_chunks` and `render_root` (`rlm/config.py:891`) takes `restricted: bool = False` **[V]**. So replay re-derives `rlm-restricted` episodes against `root.v3` — a substitution `render_root`'s own docstring describes as a loud-refusal case in the live path. **Accurate severity:** this is a defect in the *verification* path only. Live runs are unaffected, and S3's PASS covered the `rlm` arm, which re-derives correctly. The correct statement is that **the restricted arm sits outside the replay canary**, not that I4 has stopped being verified.
2. **Re-run `rlm replay` against the current 614-episode store** after (1). S3's PASS is from a 12-episode store on 2026-08-16.
3. **§5.2 items from the 2026-08-20 brief**: rotation-duration as a logged health metric with a refuse-to-continue trend rule; verdict pairing for `rlm-restricted` (`rlm/verdict.py` computes no pair, margin, p-value or gate for it); the stale `config.yaml:370-372` comment the plan itself retracts.

### 5.1 Gate 0 — the soak (one unattended night, zero `rlm/` changes)

**A 300+ turn null-agent soak of the root server on real ARC frames.** One standalone script under `s6/`. No sandbox, no dispatcher, no episode runner, no bridge verb, no model reasoning required.

**Procedure.**
1. Pull real 64×64 frames from `Arcade(operation_mode=OFFLINE)` on one public game. Serialise each three ways — (a) raw int-list rows, (b) ASCII glyph rows, (c) a segmentation summary in Duck's shape (connected components, object hashes, containment, adjacency). Tokenise every one with the root's own `/tokenize`.
2. Drive the root server with a scripted, strictly append-only conversation of 300+ turns using the cheapest serialisation that clears the horizon, with a fixed short reply per turn.
3. Record per turn: `tokens_in`, `tokens_cached`, `timings.prompt_ms`, wall clock, RSS, and the turn at which 0.90 × window would trip.
4. Run three times: `-ctxcp 32`, `-ctxcp 1`, `-ctxcp 0`. Continue past 2,000 sequential requests and record whether the process survives.

**Pre-registered thresholds.**

| id | measures | PASS | notes |
|---|---|---|---|
| **T-1** | tokens per observation | at least one serialisation ≤ **1,000 tokens/frame** | the threshold is not arbitrary — it is the leaf's measured fact-cliff bracket **[989 / 1,003]** (12/12 vs 0/12, p = 7.4e-07), rounded down to 1,000 and transferred to the root by same-family analogy **[R]** |
| **T-2** | checkpoint tax | some `-ctxcp` setting cuts median turn wall by ≥ **30%** without forcing full re-prefill | full re-prefill is declared when that setting's `tokens_cached/tokens_in` falls below **0.5** at turn 100, or when `timings.prompt_ms` scales with total history rather than with the appended segment |
| **T-3** | process endurance | one process completes ≥ **2,000** sequential requests, no crash, RSS growth < 2× | directly tests upstream #23181 |
| **T-4** | prefix cache under append-growth | `tokens_cached/tokens_in` ≥ **0.9** at turn 300 | if this fails, per-turn cost is prefill-dominated and every estimate here is ~10× optimistic |

**Programme kill criterion (pre-registered).** If **T-1 finds no serialisation under 2,000 tokens/frame** *and* **T-3 crashes before 2,000 requests**, stop. Write the negative result as an `s6/` results doc, leave `DIRECTION.md` untouched, and return to v2 and the appliance. This is a real possible outcome and it is not a failure of the exercise — it is the exercise.

**What Gate 0 buys.** Frames per window (is the horizon a constraint or a wall); whether per-turn cost is 16 s or 8 s (29 h vs 15 h per public-set pass **[R]**); whether `llama-server` survives a long-horizon session at all, which nothing in this project currently establishes; and the tokens-per-frame figure every downstream estimate depends on.

### 5.2 Gate 1 — endurance and supervisor, as two arms (per D-A5)

Same synthetic environment, same local root, same seeds. `loop-nosup` (control) and `loop-sup` (treatment).

- **Endurance** is read off the *control arm alone*: answer quality and per-turn latency against turn index. It extends Gate 0's null-agent soak to a real reasoning loop.
- **The supervisor's contribution** is the delta between arms.

**Scored on repetition-attractor entry and recovery rate, never on task score** — because §2.1 proves v1 cannot score anything. Pre-registered milestone: **entry rate falls by ≥50% with zero new outcome types**. The DFlash2 entry-rate anomaly already logged in `milestones/s2/REPLAY-LOOP-AB.md` (11/88 vs 1/90 episodes, unexplained) is free signal and should be folded into the design rather than run separately.

**What the treatment arm changes, minimally.** `note_turn`'s detector is untouched. `repetition_observation` stops being a fixed config string and becomes **episode-specific content** derived from the trace — what was tried, what the observation was, what has not been tried. Still scaffold-composed, still I1-clean: the model never chooses it. The kill at cap remains as the backstop.

**What this would prove.** That a redirect beats a kill on this model, on this hardware, under a pre-registered design with a working replay gate — a result nobody in the literature has, because no supervisor or monitor paper tests an executor below ~30B **[R]**.

### 5.3 Gate 2 — the `env_step` spike (time-boxed, one day)

One new bridge verb against `Arcade(OFFLINE)` on one public game.

**Design that keeps it cheap.** The effect executes **scaffold-side** (the child has zero egress by kernel token); the observation returns into an ordinary Python variable; the root sees only that cell's stdout through C3. **I1, I2, C3 and the replay path are untouched**, and `rlm/cli.py:_rederive_messages` needs no change at all. The `steps.action_type` ENUM migration, reward columns, `env_id`/`env_seed`, `terminated`/`truncated` and `state_ref` belong to *scoring* (§5.4) and are deliberately deferred.

**Two hard rules.** Never edit `prompts/root.v3.md` — it is S4's pinned arm prompt; add a new versioned file and update **all four** prompt-slot enumerations, or every subsequent episode replays as `PromptDrift`. And the budgets need arm-specific values before this runs: `max_identical_turns: 3` will kill a legitimately repeated action (a poll, a wait, an idempotent probe), and `max_wall_clock_s: 1300` allows ~3 s/step at 100 steps on a box whose median turn is 15.76 s.

**Success criterion is not score.** It is: *the root clears level 1 of one public game without a `PromptDrift`, a `budget_kill`, or a bridge desync.*

**Precondition, one afternoon, before the spike:** can the 27B read an ARC observation at all? Twenty frames, ask for object segmentation plus a transition description, score by hand. Zero published results exist below 27B, and the two open-weight data points are Qwen3.6-27B and Gemma-4-31B **[R]**. If this fails, Gate 2 is over and §5.5 becomes the only live path.

### 5.4 Gate 3 — scoring, RHAE, and the trace vocabulary

Only after Gate 2. Local RHAE from the shipped `baseline_actions`; the schema work deferred at Gate 2 (`env_step_idx`, `tool_name`/`tool_args`, `reward`, `terminated`/`truncated`, `state_ref`, `observation_format`); and three trace defects the sweep surfaced that must be fixed here rather than papered over **[R]**: `steps.status` tracks *scaffold health* rather than action success (32 `repl_exec` rows contain a traceback and all 32 are `status='ok'`); `parent_step_idx` is NULL on all 1,878 `repl_exec` rows, so there is no monotone env-step index to build a timeline from; and `t_dispatch` is NULL on 100% of 2,120 non-`llm_call` rows, so REPL execution latency is unrecoverable and §8's time split is unimplemented.

Also here: **storage shape.** Every root turn writes its own complete rendered request as a blob (`rlm/episode.py:999-1005`), so *n* turns store ~*n*²/2 tokens of redundant prefix. Today that is 322 MB across 614 episodes; one 1,000-turn episode would be ~700 MB — roughly twice the entire existing blob store **[R]**. Base render plus deltas, before any long run, not after.

### 5.5 Gate 4 — the second domain (stub, per D-A3)

Deferred. Recorded so its shape is not forgotten: the truest local analogue of AVO's *documented* half is optimising the machine this runs on — llama.cpp behaviour on gfx1151, where `f` is real and cheap (t/s plus correctness against a reference), the artifact is committable, and §3.2 has already surfaced a documented unexploited defect. Two blockers stand in front of it: it needs the shell/git/filesystem surface C1 exists to forbid, which is a governance decision; and Co-Harness measured harness-only evolution on Qwen3-8B/32B at ~4.3pp, **plateauing after round one**, against ~20.4pp when co-evolved with weight updates **[R]** — a local-only loop may bank one improvement and stop.

---

## 6. What this does to RLM, to v1/v2, and to DIRECTION.md

### 6.1 The RLM thesis is validated, not abandoned

Duck Harness is an RLM. Qwen3.6-27B driving a Python REPL, frames exposed as variables, the raw grid deliberately hidden from the model **[R]** — I2 arrived at independently by the team that won Milestone 1. The **engine** transfers almost completely: root plus REPL, context by reference, budgets, replay, sandbox, slot discipline.

### 6.2 What does not transfer

The **evidence base**. Benchmark v1 is saturated (§2.1) and its 30 tasks are all code-solvable (2026-08-20 brief §2). This is not a cost of this design — it was already true, and §8's comparability rule plus `benchmark.manifest_sha256` mean no v2 or ARC number can be compared to a v1 number regardless of what gets built.

### 6.3 The delegation question is unchanged, and this design does not answer it

ARC-AGI-3 does not price delegation, and no claim here should suggest otherwise. `rlm-restricted` is 5 successes in 30 episodes **[V]** — the arm that delegates is broken, not merely unmeasured. **v2, or one interactive v2 category, remains the only instrument.**

**Standing argument for revisiting D-A4 after Gate 0** (recorded, not acted on): v1 having no headroom means it cannot score *any* path, so a v2 is a prerequisite rather than a follow-up; and making **one v2 category interactive** would buy the delegation measurement and the long-horizon trajectory supply from a single authoring effort instead of two. `DIRECTION.md`'s own rule — re-litigate only with new facts — is satisfied by §2.1. The decision is not due until Gate 0 resolves.

### 6.4 DIRECTION.md moves less than it appears to

The appliance's pitch is private, on-prem, air-gapped, auditable, hard-budgeted, replayable. An offline ARC-AGI-3 run is that story *executing* — it is the real engine, deterministic, with no network. What it does not validate is the **corpus** half: no private documents, no admission control, no multi-model co-residency. So S0–S4 keep their meaning and the appliance gains an endurance story it never had.

**But decision #1 of the 2026-08-20 brief is still owed**, and this is the second time the same unmade decision has surfaced. Recommendation: **pay it after Gate 0, not before.** If Gate 0 fails, no amendment is needed. If it passes, amend minimally — name long-horizon autonomy as an S6-onward destination and change §1's "not a training project" to "training is S6, scheduled behind the training-stack kill gate" — rather than tearing up the appliance. Allowing the contradiction to persist by drift is the one option this document rules out.

---

## 7. Risks, pre-registered

| risk | severity | how it shows up | response |
|---|---|---|---|
| Frames do not fit under the horizon at any serialisation | fatal | T-1 | Programme kill (§5.1) |
| `llama-server` dies mid-run (upstream #23181) | fatal as built | T-3 | Programme kill, plus an upstream reproducer — this would be a genuinely useful negative result |
| Checkpoint tax not removable | serious | T-2 | 16 s/turn stands; ARC full-set passes become ~29 h **[R]** and only single-game milestones are affordable |
| RHAE's 5× cliff zeroes runs that look successful | serious | any scored run | §1.1's rule: score actions-per-level from the first spike |
| The supervisor's redirect is unmeasurable on a saturated benchmark | serious | Gate 1 | §5.2 scores entry/recovery rate, not task score |
| Memory layer destroys §8's exchangeability | serious | Gate 3+ | §3.4: no memory layer ships without a new statistical design |
| Drift instead of decision on `DIRECTION.md` | manageable, corrosive | none — that is the problem | §6.4: pay it immediately after Gate 0 |
| Local-only leaves harness and model failures confounded | manageable | throughout | D-A2's answer: ablate the harness, not the model |
| Harness-only self-improvement plateaus after one round | unknown | Gate 4 | Co-Harness's ~4.3pp plateau **[R]** was measured on short-horizon math; transfer to agentic work is untested in either direction |

---

## 8. What this design does not settle

- **AVO's supervisor is unspecifiable.** No threshold, mechanism, invocation count, prompt, or ablation exists in any source **[R]**. §5.2 does not reimplement it; it generalises `max_identical_turns`, which is a different and better-grounded thing.
- **No parameter threshold for harness benefit exists in any paper.** Every reported threshold is behavioural (25.1% skill activation; 0.52 → 0.13 adherence decay). "~32B" in §0.2 is inferred from a handful of models **[R]**.
- **The literature's central contradiction is unresolved.** Qwen3-32B is classified weak-tier at +4.4pp (arXiv:2605.30621); Qwen3.5-35B-A3B gains +113% on the same SWE-bench Verified (arXiv:2606.09498). Model generation, harness shape, and domain are not separated by any experiment **[R]**, and the strongest positive result is a ~3B-active MoE off an 18–22% baseline, which is not scale-comparable to a dense 27B.
- **Whether the 27B can read an ARC frame at all** — §5.3's precondition, unanswered.
- **`f` for the second domain.** AVO's architecture is a *consequence* of having a cheap correctness-gated scorer and a committable artifact. §5.5 has neither yet.

---

## 9. Amendments this design would require if adopted

None before Gate 0 — that is the point of §5.1. If Gate 0 passes:

1. `ARCHITECTURE.md` §1: an environment channel is in scope; a general agent framework still is not.
2. `ARCHITECTURE.md` §5: `env_step` as a bridge verb; arm-specific budget values (§5.3).
3. `ARCHITECTURE.md` §6: the trace vocabulary of §5.4.
4. `ARCHITECTURE.md` §8: a statistical design that survives non-exchangeable cells (§3.4), before any memory layer.
5. `DIRECTION.md`: §6.4's minimal amendment.

---

## 10. Reproducing the [V] figures

```
# store census, per-arm outcomes, leaf calls, longest episodes
.venv/Scripts/python.exe  # duckdb.connect('traces/rlm.duckdb', read_only=True)
#   episodes/steps counts; group episodes by config_snapshot->'bench'->>'arm'
#   join steps for action_type='llm_call'; count action_type='repl_exec' per episode

# the replay-path defect
grep -n "render_root" rlm/cli.py rlm/episode.py rlm/config.py

# config values
grep -n "window_tokens\|root_window_kill_fraction\|truncation_cap_chars\|max_identical_turns\|max_wall_clock_s\|cache-ram" config.yaml
```
