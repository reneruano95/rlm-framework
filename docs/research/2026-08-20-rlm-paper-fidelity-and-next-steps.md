# Are we following the RLM paper? — fidelity audit and next steps

**Date:** 2026-08-20 · **Status:** research brief + proposal (nothing here amends ARCHITECTURE.md or DIRECTION.md; §6 lists the decisions that would)
**Inputs:** arXiv:2512.24601 v1 (HTML) and **v3 (2026-05-11, adds post-training)**; the live alexzhang13/rlm harness (PyPI `rlms` 0.1.3); mit-oasys model cards; PrimeIntellect-ai/prime-agent (v0.7.4) and its launch post; ARCHITECTURE.md v0.3.15, DIRECTION.md, `milestones/s4/RESULTS.md`, the delegation-arm plan, every file under `prompts/`, `rlm/`, `bench/`; the live ledger of run `9588328d` before it was stopped.
**Reframing supplied today:** the project's purpose is *a self-improving RLM agent on local models*, in the spirit of prime-agent. §4–§6 are written against that goal.

---

## 0. The short answer

**Yes on the thesis, deliberately no on the recipe — and every divergence is measured rather than drifted.** rlm-halo implements the paper's core idea exactly (prompt as a REPL variable, root writes ```` ```repl ```` cells, truncated observations, variables as buffers, `llm_query` at depth 1, big root / cheap leaf). It departs from the paper's *mechanism* in seven places (§1), each with an on-box measurement behind it.

**But the benchmark does not follow the paper, and that is the finding that matters.** The paper's own ablation ("RLM, no sub-calls") *beats* the full RLM on CodeQA and BrowseComp+ with Qwen3-Coder; sub-calls only earn 10–59 % on information-dense semantic tasks (OOLONG, OOLONG-Pairs). Benchmark v1's 30 tasks are **all** solvable by code over the raw text (verified this session, §2). So S4's "RLM 30/30 with zero leaf calls" is not an anomaly — it is the paper's Observation 2 reproduced on a benchmark that only contains the regime where code wins. The delegation question cannot be answered on v1, which is why the restricted arm was stopped (§3).

**On self-improvement:** prime-agent's "self-improving" is *harness* learning (prompt notes, memories, skills, sub-agent specs, CRUD-edited by the model from its own trajectory, reversible) — **no weight training**; its authors say no model has been trained around it. The paper v3, by contrast, *did* train: rejection-SFT of Qwen3-8B on 1,072 teacher trajectories (48 H100-hours) and RLVR of Qwen3-4B. rlm-halo already has what both need and neither has: trajectory-shaped replayable traces (§6 of the spec), hard budgets, a sandbox, and a pre-registered benchmark. What it lacks is (a) tasks on which delegation can earn anything, (b) a held-out split to measure learning without overfitting, and (c) the loop itself. §5 sequences those.

---

## 1. Fidelity matrix — mechanism

Legend: **same** · **divergence (measured)** · **extension** · **gap**. Citations are to the working tree at `ff6aee3`.

| # | Paper (v1/v3) | rlm-halo | Verdict | Why, and what it costs |
|---|---|---|---|---|
| 1 | `context` variable in a persistent Python REPL; root told type / total chars / per-chunk chars | `context: str` + read-only `chunks: list[str]`; root told **nothing** about sizes — tip 1 says measure them (`prompts/root.v3.md:23-24,59`; user message = task + `[turn N; sub-calls remaining: M]`, `rlm/episode.py:319-335`) | divergence | Keeps the user message hash-stable; costs one root turn. The live harness puts "Your context is a {type} of {N} chars" in the user message — cheap to add if a trained root expects it. |
| 2 | `llm_query(prompt: str) -> str`, free-form; sub-LM "handles ~500K chars", "feed 10 documents per call" | `await llm_query(question, *, chunk=None, role="leaf")`; scaffold composes `[pinned prefix][chunk]\n\n[question]` (`rlm/dispatcher.py:255-265`); single-string form still accepted; **hard cap 2,560 rendered tokens** (`327680 // 128`, `dispatcher.py:1456-1466`) | divergence | Forced by the measured ~1,000-token distance cliff on this leaf (38/39 correct within 1K tokens, 0/39 beyond, p≈1e-21; ARCHITECTURE.md:256) and by R13's one-slot-per-window rule. Consequence: a 130K corpus is ~300 windows / ~600 sub-calls where the paper's root makes ~10. **This is the structural divergence; it changes the economics of delegation by two orders of magnitude.** |
| 3 | Model chunks freely in code | Scaffold chunker C2, 640/480 overlapping windows; prompt forbids re-chunking (`root.v3.md:24`) | divergence | I1 + uniform per-call cost + sweep control. **Gap:** enforced only by prompt — `context` is a full `str` and `chunks` a mutable list; any substring can be passed as `chunk=` (`rlm/episode.py:543-547`, `rlm/sandbox/child.py:721-736`). Spec §5 C2 says "read-only"; code does not make it so. |
| 4 | `FINAL(text)` / `FINAL_VAR(var)` in prose; paper calls it brittle (16 % / 13 % of teacher turns misused them, v3 App. A) | `final_answer(value)` called **inside** the cell, never parsed from prose (`ARCHITECTURE.md:175`; `child.py:435-446`); `AnswerGuard` traps the live harness's `answer[...]` reflex (`child.py:470-488`) | divergence (better) | Justified by I2. The official harness itself abandoned FINAL for an `answer` dict on 2026-05-13. Only cost: S5 conditioning gap for roots trained on either format. |
| 5 | One fixed, task-agnostic system prompt ("not tuned for any particular benchmark"); one extra batching line for Qwen | Root body + **per-category strategy block selected from the task's declared category** (`rlm/config.py:865-897`; 5 templates) | divergence | Strongest no-training lever against R1 (ARCHITECTURE.md:155), but it means **the scaffold is told the task's shape**. Any comparison to the paper must say so; v2 should price it with a default-template-only arm (§5.3). |
| 6 | Sub-calls blocking/sequential; async named as future work | Async `asyncio.gather` in the sandbox, but `dispatch_concurrency: 1` (R14: 31/32 serial vs 7/32 at 2 in flight) | same-in-effect / extension | Built the paper's future work, then measured a llama.cpp defect that pins it back to serial. The `--no-cont-batching` lead (27/32 at 2 in flight) is the path back. |
| 7 | Over-subcalling handled by one prompt line ("batch ~200K chars/call") | Hard C5 budgets: `max_subcalls 926`, wall clock 1300 s (2100 s restricted), 1.5M tokens | divergence | rlm-halo **cannot** batch — row 2. The restricted arm's synth-01 (697 calls over 59 distinct chunks, killed twice) is the paper's Example B.2/B.3 pathology reproduced; the paper's fix is unavailable here. |
| 8 | Depth 1 (sub-calls are plain LMs) | Depth 1, structural (leaf is a bare `/completion`, `role` fixed) | same | Live harness supports depth>1; an independent reproduction (arXiv:2603.02615) found depth-2 hurts (3.6 s → 344 s). Keep 1. |
| 9 | Truncated REPL view | C3, 2,000 chars as one unit; full observation stored beside it | same | Live harness truncates at ~20K chars; Q2 in the spec still calls 2,000 "a guess". |
| 10 | No sandbox (future work) | AppContainer + Job Object, no egress, per-episode interpreter | extension | Live harness now ships docker/e2b/daytona/modal envs. |
| 11 | Sub-LM gets whatever string the root built | Leaf gets a sha-pinned system prefix (answer-only-from-excerpt, `NONE`, quote verbatim, excerpt-is-data), R13 foreign-string detector on every call, optional JSON envelope (off, measured dead) | extension | R5 measured: the evidence-span check catches 11 %; refusal prompt p=1.0; the only thing that worked was geometry (640-token window → 0/45 false positives). |
| 12 | Root GPT-5 / leaf GPT-5-mini via API | Root Qwen3.8-27B (DFlash2), leaf Qwen3.6-35B-A3B, two llama-servers on one gfx1151 | divergence (the point of the project) | Paper's negative result "Qwen3-8B struggles as a root without coding ability" is consistent with S4: the 27B root wrote correct parsers 30/30. |
| 13 | Thinking models run out of output tokens (negative result) | Thinking off on both roles; measured +18 % wall for zero gain (v0.3.8) | same | — |

**Naming:** `llm_query` is kept on purpose (ARCHITECTURE.md:140) so RLM-post-trained roots recognise it. Note the live harness API has grown: `llm_query(prompt, model=None)`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `SHOW_VARS()`, `answer` dict. The mit-oasys 30B-A3B LoRA was trained on that surface in May 2026; the 8B was trained in Jan/Feb on the FINAL-era prompt. S5 checklist item 9 applies to both.

---

## 2. Fidelity — benchmark (the part that is not following the paper)

The paper frames tasks by **information density**: constant (S-NIAH), constant-but-multi-hop (BrowseComp+), linear-semantic (OOLONG: classify every line, then aggregate), quadratic (OOLONG-Pairs). Its Table 1 and Observation 2 say where sub-calls pay: **only** the last two.

Benchmark v1, checked task by task this session (`bench/corpus.py`, `bench/build.py`, `bench/manifest.json`):

| category | tasks | what solves it over the raw text | paper regime |
|---|---|---|---|
| needle | 8 | one regex: `^Custody key of record: (.+)\.$` — 8/8 | S-NIAH (constant) |
| aggregation, regex-solvable | 3 | `grep -c '^Status: SEALED'` — 541/544/514 | linear, **non-semantic** |
| aggregation, "regex-defeating" | 4 | header-org == custody-`to`-org: a parse + string equality reproduces 594 on agg-04 | linear, **non-semantic** |
| synthesis | 8 | set intersection of `^\[ENT-\d+\] (.+)$` across documents — 8/8 | constant-ish |
| code QA | 7 | `^\s*(async\s+)?def NAME\(` inside `=== FILE:` sections — 7/7 (one repo: this project's own `rlm/`) | constant |

**30/30 are code-solvable. None has the OOLONG property (a per-item label only an LM can produce). No task is quadratic.** §8's rule "≥1 aggregation task must defeat deterministic string matching — requiring semantic judgment per item" was satisfied in letter — `regex_at_chance` scores seven bag-of-words regexes on the *Disposition line alone* (`bench/corpus.py:265-305`) — but not in spirit: it never tests a parser that reads the header, and the S4 root wrote exactly that parser (`milestones/s4/RESULTS.md` §"What the root actually did"). The rule's stated purpose ("stops strategy templates from quietly turning the category REPL-only while the leaf-reliability surface ships unmeasured", ARCHITECTURE.md:333) was not achieved by construction.

Two smaller mismatches with the paper's baselines: the paper's "base model" is the **root** model single-shot; B1 is the **leaf** model (no root-single-shot arm exists). And the paper's key ablation, "RLM no sub-calls", is not a named arm — S4's `rlm` arm was it by accident.

Consequence: **the restricted arm on v1 measures what the leaf subtracts on tasks code already solves.** That is a real number, but its sign is already known from the paper, and it is the most expensive cell in the grid.

---

## 3. The run that was stopped today (`9588328d`, 5 arms, started 16:23 local)

Stopped at the user's instruction after 6 of 90 blocks (5.4 h) on this evidence:

- **Restricted-arm fault, worsening.** Leaf rotations took 6–80 s for rotations 1–18, then **152 s and 222 s**; after each, the new 128-slot generation answered zero windows → `slot_pool_error_drained` (the dispatcher's own definition of a failed server, `rlm/dispatcher.py:349`). This happened *with* the waves guidance restored (`ff6aee3`), so the plan's "waves fix it" conclusion is contradicted on agg-02 seed 3. Restricted cells so far: 4 `checker_failed`, 1 `budget_kill` (2,113 s), 2 `error`; 0 successes.
- **Pace:** 0.89 h/block → 65–80 h remaining against a 60 h budget (aggregation-heavy head; it would have come down, not enough).
- **Redundant cells:** the only config changes since S4 are the root's DFlash2 swap (`config.yaml`, commit `14de8e8`) and the restricted arm's wall clock. B1 and B3 never touch the root — ~950 s per block of leaf-only cells re-measuring an unchanged system.
- **Known-sign measurement** (§2).

Stop method: `taskkill /F /T` on the tree (bench 5284, root server 15564, leaf server 9748, sandbox 3816). The S3-verified hard-kill path applies: one episode row is open (`agg-02`, started 01:51:14 UTC) and will be tombstoned `error / orphaned_at_recovery` at the next `rlm` start. The ledger holds 30 rows (6 complete blocks) and `--resume` exists if the grid is ever wanted.

Defects to file from this run regardless of what runs next: (a) the post-rotation drained generation (rotation duration is the leading indicator — the config note that DFlash2 thinned GPU headroom to ~16.5 % and "`-np 128` on the leaf is unchanged, so the floor is the thing to re-check first" is the first hypothesis); (b) `rlm/verdict.py` computes no pair, margin, p-value or gate for `rlm-restricted` (RLM_ARM/BASELINES are v1 constants) while the plan says restricted-vs-rlm is the comparison that carries meaning; (c) `config.yaml:370-372` still asserts "reverting the fan-out guidance did not move it" — the argument the plan itself retracts.

---

## 4. What "self-improving RLM agent" means elsewhere, and what this repo already has

### 4.1 The paper v3 (what *training* an RLM looked like)
- **RLM-Qwen3-8B:** 750 LongBenchPro tasks → 2,250 trajectories of RLM(Qwen3-Coder-480B) with Qwen3-8B sub-calls → drop score-0 and one-turn trajectories → **1,072** → one SFT sample per root turn (history → turn), drop turns >100K chars, **programmatically patch template mistakes** (16 % misused FINAL, 13 % misused FINAL_VAR) → prime-rl, batch 64, 300 steps, **48 H100-hours**. Result: median +28 %, >3× faster trajectories. Key insight (App. A): *"the leaf sub-calls are essentially general-purpose LLM requests and the major hurdle is learning to operate as the root model."*
- **Qwen3-4B via RLVR** on MRCRv2 32–64K/2 needles (150 steps, batch 128, 4 rollouts, 4,096 max output tokens, 20 iterations) generalises to 1M/8 needles.
- Official `training/` dir (2026-05-27): verifiers-compatible env at depth 1, subprocess REPL, sub-calls proxied to the policy itself.

### 4.2 prime-agent (what *harness* self-improvement looks like)
- TypeScript fork of pi-mono; **one tool: a persistent IPython kernel**; file ops, shell, MCP, skills and sub-agents all happen as Python. `await rlm("task", ...)` spawns a *full child agent session* and returns a handle immediately — results come back via messages or files. Default depth 1. Not a sandbox.
- **Continual Harness** (arXiv:2605.09998): state H = (prompt notes, memories, skill descriptions, sub-agent specs). `/refine` feeds the last 80K chars of the trajectory + harness overview + last 20 refinement events to the *same* model, gets a JSON list of create/update/delete edits, applies them, logs each with trigger and outcome, rollback by ID; base system prompt immutable. Auto-triggers on compaction / turn interval behind a review call.
- **No weights change.** Launch post: "currently no model has been trained around Prime Agent or its core feature set"; co-learning is "future work". Evals used Opus 5, GPT-5.6 Sol, GLM-5.2; **no small-local-model results.**
- Ecosystem check: harness-only self-improvement is reported to work from Qwen3.5-9B up, with "mid-tier models benefiting most" (Weng, 2026-07; Self-Harness on Qwen3.5-35B-A3B, 4×H200). **No published closed loop (agent → own trajectories → weight update → better agent) exists for a 3B–30B model on one Strix Halo / Mac Studio / DGX Spark.** On Strix Halo specifically: Qwen3 4B–8B SFT/LoRA works via HF Trainer on ROCm 7.x; Unsloth, bitsandbytes and torchao are broken on gfx1151; prime-rl requires NVIDIA.

### 4.3 What rlm-halo already holds toward that goal
- §6 trajectory schema, designed for this: `(state, action, observation, outcome)` per step, `observation_view` (what the root saw → SFT) and `observation_full_ref` (ground truth), `root_view_hash` replay verified at S3. The spec itself says the format is "simultaneously the SFT format of the original RLM paper … and the raw material for a Voyager-style skill library".
- A prompt registry that is already the shape of a learned harness layer: versioned files, sha-pinned, selected by the scaffold, never by the model at runtime.
- A final-answer channel that removes the paper's largest class of training-data noise (the 16 %/13 % FINAL errors cannot occur).
- Store today: 523 episodes, 211 successes; **116 successful `rlm`-arm episodes** (all of them) + 27 pre-bench successes ≈ 143 toward S6's "≥500 successful episodes" precondition. S6's other two preconditions: S4 passed (yes); "a recurring failure mode demonstrably unfixable by prompting" (not yet claimed — the restricted arm's thrash is a candidate, but it is a benchmark-shape problem as much as a model one).

### 4.4 The honest gap
The spec's §1 says this project is **not** an agent framework and **not** a training project, and DIRECTION.md (2026-08-16) points at an open-core library + on-prem appliance. The goal stated today — a self-improving RLM agent on local models — is compatible with the appliance (it is *the engine that learns from its own traces*) but it is not what the documents say. That needs a written decision, not drift (§6).

---

## 5. Proposed sequence

Ordered so that each step is cheap relative to what it settles and none requires re-deciding a frozen thing.

### 5.1 Pay the DFlash2 debt, root-only (≈2.5 h)
`rlm bench --arm rlm` over frozen v1, 3 seeds. This is the I5/R4 re-validation the config says is "owed, not edited". B1/B3 are root-independent and need nothing; B2's reduce step uses the root but B2's score is not the claim under test — state that and skip it. Gate: 30/30 or a named regression.

### 5.2 Fix what the stopped run exposed (before any delegating arm runs again)
Rotation-then-drained fault (start from headroom; log rotation duration as a health metric and refuse to continue when it trends); verdict pairing for `rlm-restricted`; the stale config comment. Add the rotation-duration series to the lifecycle log's health events so the next occurrence is caught at rotation 1, not block 6.

### 5.3 Benchmark v2 — the tasks the paper says delegation is for (the measurement that unblocks everything)
Authored before run, frozen separately (`benchmark.version: v2`, its own manifest sha; the v1 freeze stands).
- **Linear-semantic category (OOLONG-shaped):** items each carrying a label only an LM can assign, then an aggregate question over all items. Two sources are admissible under the existing contamination rules: (i) real human-labelled data as OOLONG does (TREC trec_coarse: 6 labels, public; the aggregate over a *sampled* subset is not memorisable and the closed-book probe will show it); (ii) a generator that emits paraphrased natural-language items *from* known labels. Either way the at-chance artifact must test a **parser**, not bag-of-words regexes — close the agg-04 loophole explicitly.
- **Quadratic category (Pairs-shaped):** a small one; it is where the paper's gains are largest and where the REPL's buffer-stitching matters.
- **Keep code-solvable controls** (2–3 tasks) so the root is still scored for choosing code when code suffices — §8's original intent, now actually enforced.
- **Arms:** `rlm`, a named `rlm-nosubcalls` (the paper's ablation, made explicit), `rlm-restricted` properly paired, `rlm-default-template` (prices divergence #5), B2 (the honest control). Drop B1/B3 unless a size class needs them.
- **Train / held-out split from day one** (e.g. 60/40 by task), because §5.4 and §5.5 cannot be measured without it.
- **Price it first** with `milestones/s2/aggregation_options.py`: per-item classification fits 640-token windows naturally (items are short); at 2.78 s/window serial a 300-window corpus is ~14 min/episode — inside 1,300 s. If R14's `--no-cont-batching` lead holds, this is also where fan-out would first pay.

### 5.4 S6-lite — harness-level self-improvement under rlm-halo's discipline (no weights)
prime-agent's mechanism, made I1-clean: a **learned registry layer** (Python helpers injected into the sandbox as skills; prompt notes; per-category strategy deltas) that a `rlm refine` step proposes from traces and the **scaffold applies between episodes** — versioned, sha-pinned into `config_snapshot`, rollback by refinement id, gated by the held-out split (I5: kept only if held-out success is unchanged or better). Never at runtime, never by the model's own choice mid-episode. This is the Voyager option S6 already names, and it is the only self-improvement path that works with frozen local models today. Hazard to pre-register: refinement overfitting the train split — the held-out gate is the whole point.

### 5.5 S6-full — rejection-SFT into a *smaller* root (the paper's recipe at this box's scale)
Preconditions: ≥500 successful episodes (v2 will supply them), the held-out split, and a **new kill gate**: one LoRA training step on a Qwen3-class 4B–8B dense model under ROCm 7.x on this box, measured (HF Trainer path; Unsloth/bnb/torchao are known broken on gfx1151). Recipe, copied from v3 App. A and corrected for this scaffold: teacher = the 27B root's successful trajectories (rejection sampling; drop one-turn episodes; one SFT sample per root turn; cap turn length) → student = 4B–8B LoRA → swap in as an S5 row ("models are config") → score on v2 held-out against the untrained student, the 27B teacher, and the mit-oasys 30B-A3B LoRA (harness-conditioning gap noted). The template-patching step the paper needed is unnecessary here (`final_answer` + `AnswerGuard`). RL on-box is out of reach (prime-rl is NVIDIA-only; GRPO on gfx1151 unverified); if RLVR is ever wanted it is a cloud one-off, which contradicts "local" and should be said so.

### 5.6 Carried regardless
R13 reproducer + upstream issue (privacy defect in the multi-user story); R14 `--no-cont-batching` reproducer (fan-out economics matter 10× more once delegation is real); the leaf horizon — v0.3.5 says it is "a property of the prompt rather than of the serving path" and both models tested show it, so the lever, if any, is prompt-side, and a self-improving loop would be the natural place to search for it.

---

## 6. Decisions this brief needs from the owner

1. **Direction.** Amend DIRECTION.md to name the destination as a self-improving local RLM engine (appliance/library as packaging of it), and amend §1's exclusions ("not a training project" → "training is S6, scheduled behind v2 and the training-stack kill gate"). Or keep DIRECTION.md as is and treat §5.4/§5.5 as research spikes. Either is defensible; doing neither leaves the documents contradicting the goal.
2. **v2 ground truth.** Real labelled data (TREC, as in OOLONG) vs synthetic-from-labels. Real data is faster and paper-comparable; synthetic keeps the "answers computed, never typed" rule and the coined-vocabulary contamination guarantee.
3. **Whether the restricted arm runs again at all.** On v1 it prices a known-sign quantity; on v2 it is informative. Recommendation: v2 only.
4. **S5 candidate order.** A3B-as-root (config-only, cheap) before the mit-oasys LoRA (needs a merge + GGUF conversion and harness alignment).

---

## 7. Hygiene found on the way (small, do when touching the files)
- `rlm/cli.py:2792-2793` help text says "default: all four"; ARM_ORDER has five.
- `milestones/s4/RESULTS.md:210-211` says code QA "used real repos"; it is one repo — this project's own `rlm/` at `45597d43`.
- ARCHITECTURE.md Q1 (:476) still calls the 32K chunk default "a guess" after 640/480 landed (:257); Appendix A is labelled v0.2 and shows `parallel: 8`, `max_wall_clock_s: 900`.
- `prompts/strat-aggregation.v2.md:3` cites the 1,024/768 geometry; body text still holds at 640/480.
- `config-thinkon.yaml` is pre-DFlash2 and pre-S4 in more than the thinking flag.
- The synthesis strategy template talks about briefs, quotes and disagreements that no synthesis task or checker exercises.
- C5 admission counts the unrendered user segment (`add_special=False`) while the slot pre-flight counts the rendered prompt; the reservation under-counts each call by the ~311-token prefix plus template markup.

---

## Postscript (2026-08-21)

- **§5.1 done.** Root-only DFlash2 re-validation (`milestones/s4/RESULTS-dflash2-rlm-only.md`, run `c1740386`): 30/30 tasks, median wall 0.80× S4, energy 0.84×, tokens 1.05× — R4 met, DFlash2 kept. Two new `synth` episodes looped on one byte-identical cell (70× and 111×) until killed; three `rlm` episodes delegated for the first time (one with 83 leaf calls, correct, ~3× wall).
- **Replay A/B done and adversarially reviewed** (`milestones/s2/REPLAY-LOOP-AB.md`): repetition *given* the loop state is the root model's, not the drafter's (≈64 % at the first repeat, ≈92 % once established, identical across DFlash2 / MTP / no speculation); *entry* into the state is not settled and leans against DFlash2 (11/88 vs 1/90 episodes with the entry condition, all code QA; a replicated shorter-reply bias). Owed: an entry-rate A/B (code QA × 3 seeds, `dflash` vs `mtp`, interleaved) and a spec amendment bundling a C5 repetition guard (`repetition_loop` outcome), per-turn seed derivation (the root is currently sampled with the same seed on every turn), and a fix for the doubled empty think block every past assistant turn is rendered with. §5.2's list grows by these three items; the rest of the sequence is unchanged.
