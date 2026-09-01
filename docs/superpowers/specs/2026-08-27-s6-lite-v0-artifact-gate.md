# S6-lite v0 — the artifact gate

**Date:** 2026-08-27 · **Status:** APPROVED by the owner 2026-08-27. Implementation plan: `docs/superpowers/plans/2026-08-27-s6-lite-v0-plan.md`.
**Decision this implements:** the owner chose **option C** on 2026-08-27 — stop building the agent, build what governs it. rlm-halo supplies the corpus, the splits, the proposer, the gate, the artifact store and the audit record; prime-agent supplies the agent loop.
**Inputs:** `docs/research/2026-08-27-prime-agent-spike.md` (the measurement this design consumes); `ARCHITECTURE.md` §3 (I1, I4, I5, I7), §8 (margin rule, cost scorecard, blocked design), §9 S6 (the S6-lite gate, verbatim below), §10 R16; `docs/superpowers/specs/2026-08-22-long-horizon-agent-design.md` §3.4 (the exchangeability constraint); `docs/superpowers/specs/2026-08-25-benchmark-v2-design.md` §5 (the three streams); `bench/manifest.json`; prime-agent v0.8.1 sources (extension API, `refinement.ts`, `agent-session.ts`) — all read 2026-08-27 [R] except in-repo facts marked [V].
**Evidence discipline:** every threshold below is fixed before any run. Where this design deviates from a rule already in force, the deviation is named, quoted and argued in §7 rather than left for a reader to discover.

---

## 0. What this is, in one paragraph

Prime-agent's Continual Harness applies whatever the model proposes. Measured on this box on 2026-08-27: given a free hand, the local root wrote **8 of 8 artifacts as `memory`, containing the answers to the tasks it had just been shown**, and the held-out task it was then evaluated on lost a run and cost 2.2× the wall. S6-lite v0 is the missing half — a gate that **proposes from traces rather than from the model's own conversation**, **screens candidates mechanically** against the held-out split and against I1, **accepts only what demonstrably helps on tasks it was not derived from**, and **records every decision so it can be replayed and rolled back**. Nothing about the agent loop changes.

## 1. The evidence this consumes

Four measured facts from the spike do the work; each maps to a design choice.

| # | measured [V] | consequence for this design |
|---|---|---|
| **E1** | 8/8 refinements were `kind: memory`; 0 `prompt`, 0 `skill`, 0 `subagent`, across 23 entries in the committed snapshots | The model does not know what to save. **The scaffold proposes** (§3), and `memory` is not the kind v0 starts with |
| **E2** | The one held-out regression happened with the correct answer visible in the model's own output (`other: 576`), while its memory carried a verified `re.M` parse it did not use | Artifacts must be **evaluated, not trusted**. §4's gate is the whole point |
| **E3** | Phase C wrote two independent counters and cross-checked seven files **because the goal asked**; Phase A, asked nothing, shipped `1` against 1,323 records | The valuable artifact is a **`prompt` or `skill`** — a procedure — not a fact. §3.2 starts there |
| **E4** | Every one of the four held-out/re-test tasks cost **more** tokens with artifacts loaded (1.05×–2.56×) | The gate needs a **cost** criterion, not only a quality one. §4.2 |

## 2. Topology and the seam

```
rlm-halo (Python, this repo)                  prime-agent v0.8.1 (vendored, pinned)
┌────────────────────────────────┐            ┌─────────────────────────────────┐
│ corpus + frozen splits         │            │ agent loop, IPython kernel,     │
│ runner (train / held-out)      │───runs────▶│ subagents, daemon               │
│ proposer  (from traces)        │            │                                 │
│ screens   (mechanical)         │            │  .prime/agent/extensions/       │
│ gate      (held-out A/B)       │            │    gate.ts  ──┐                 │
│ artifact store + sha ledger    │───writes──▶│  harness/     │ enforcement     │
│ audit record (DuckDB)          │            │    harness_state.json           │
└────────────────────────────────┘            └─────────────────────────────────┘
```

**Three seams, all verified in the v0.8.1 sources [R]:**

1. **`session_before_refine`** (`types.ts:521-549`, call site `agent-session.ts:8091-8117`). Returning `{skip: true}` throws `RefineSkippedError` and **vetoes the whole round**; returning `{proposal}` **replaces the built-in planner**. There is **no per-edit veto** of the built-in planner's output — if the handler returns `undefined`, the model's proposal goes to disk unsupervised. *Therefore the gate must be the proposer.* This is not a workaround; it is the same conclusion E1 forces.
2. **`before_agent_start`** (`types.ts:626-637`, result `{systemPrompt?}` at `:932-936`, chained across handlers). Receives the fully assembled system prompt and may replace it. This is the **enforcement** seam: it strips any harness entry not in the accepted set, which catches direct `rlm.harness.*` writes from the kernel — a path `tool_call` cannot see, because `refine.run` is a host request, not a tool (`agent-session.ts:3013-3024`).
3. **The state file.** `harness_state.json` under `getAgentDir()/harness/` is read on every system-prompt rebuild (`_loadMergedHarnessState`, `agent-session.ts:4376`) and is safely written out of band between sessions — verified by doing it in the spike's Phase B promotion step [V].

**One hazard this design must own, not inherit.** Entries reach the system prompt **alphabetically by `(path, title, id)`**, capped at 6 per kind and 180 chars of content (`refinement.ts:26-28, 466-481`). There is no recency or relevance ranking. So *which* accepted artifacts the model actually sees is decided by strings the gate writes. The gate therefore assigns `path` deterministically (§3.3) and the run record states how many entries were in-window — a silently truncated artifact set would make a gate verdict meaningless.

## 3. The proposer

### 3.1 It reads traces, not the conversation

The candidate generator runs **offline, between episodes**, over the train split's session JSONLs — the trajectory, its tool calls, its results and its outcome. It never runs inside a scored episode. This satisfies `2026-08-22-long-horizon-agent-design.md` §3.4's first demand verbatim: *"propose offline from traces, apply **between** episodes as versioned sha-pinned config, roll back by refinement id, gate on a held-out split."*

### 3.2 It proposes `prompt` and `skill` only

`memory` and `subagent` are **out of scope for v0**, for different reasons:

- `memory` is where E1's answer-memorization lived. It is not forbidden forever — it is simply not what v0 tries to learn, because E3 says the value is in procedure.
- `subagent` entries specify *when to invoke* a delegation. That is a **route**, and I1 says routes are never writable by an artifact (`ARCHITECTURE.md:50`: *"it may never rewrite a budget, a cap, a route, or a termination rule"*). Excluding it is the I1-clean reading, and the exclusion is enforced by the screen in §3.4, not by convention.

### 3.3 What a candidate is

A candidate is a `HarnessEntry` (schema per `refinement.ts:34-48`) with the gate's own fields fixed:

| field | value | why |
|---|---|---|
| `kind` | `prompt` \| `skill` | §3.2 |
| `id` | `rlmh-<kind>-<8 hex of content sha256>` | content-addressed; re-proposing the same text is the same artifact |
| `path` | `00-gate/<kind>/<nn>` | forces accepted entries into the alphabetical 6-per-kind window (§2 hazard); `nn` is the acceptance ordinal |
| `scope` | `global` | the store the runner promotes into |
| `metadata` | `{rlmh: {version, proposed_from: [episode_ids], content_sha256, gate_run_id}}` | provenance, and the join key to the audit record |
| `content` | ≤ 900 chars | 180 reach the prompt (`refinement.ts:28`); the rest is only visible when the model inspects the entry. The gate records both the full text and what was in-window |

### 3.4 The screens — mechanical, deterministic, zero model calls

A candidate that fails **any** screen is rejected before it costs a single held-out episode. Each screen is a test, not a judgement.

| screen | test | what it catches | known blind spot |
|---|---|---|---|
| **S-kind** | `kind ∈ {prompt, skill}` | §3.2's exclusion | — |
| **S-I1** | the content contains no token from the closed set of budget/cap/route/termination key names, generated from `src/rlm/config.py`'s own schema (`max_wall_clock_s`, `max_subcalls`, `max_total_tokens`, `max_identical_turns`, `truncation_cap_chars`, `dispatch_concurrency`, `slot_policy`, `root_window_kill_fraction`, …) | an artifact that tries to raise its own budget — I1's exact failure mode | paraphrase ("allow yourself more time") is invisible; §6 R3 |
| **S-answer** | normalized content contains no held-out task's expected `answer`, normalized by that task's own checker (`rlm.measure.checkers`) | the E1 failure aimed at the evaluation set | a *train* answer is legal and passes — correctly, since train-derived is the rule |
| **S-corpus** | `rlm.serve.leakcheck` over a `ChunkIndex` built from the **held-out corpora**: no identifier-shaped token (UUID, `ENT-#####`, hex run, long alphanumeric) from a held-out corpus may appear | a memorized needle UUID or entity code | **integers are not identifier-shaped** — `514` and `589` would pass. This is why S-answer exists separately, and why neither screen replaces the gate |

**Stated plainly:** the screens are cheap structural filters. They would **not** have rejected the spike's memories, which held only train answers and were legal by provenance. What rejects those is §4 — they did not help. Both layers are load-bearing and neither substitutes for the other.

## 4. The gate

### 4.1 Design: one frozen A/B, blocked in time

For a candidate set that passed §3.4, the gate runs the **held-out split twice** — artifacts **ON** and **OFF** — as two arms of one blocked `(task, seed)` design, in the order §8 fixes (*"runs execute in (task, seed) blocks adjacent in time across all arms, so R9 thermal drift cancels within each paired comparison"*, `ARCHITECTURE.md:380`). Seeds are v1's pinned `[1, 2, 3]`.

**Why this is a legitimate statistical design, given §3.4's warning.** §3.4 states: *"Once episode n is influenced by episodes 1..n−1, cells stop being exchangeable, and the sign test and paired bootstrap CI stop meaning what they say. Any memory layer arrives with a new statistical design or it does not arrive."* The hazard it names is **within-run adaptation**. This gate evaluates a **frozen artifact set**: the artifacts are fixed before the held-out run begins and nothing mutates during it — `autoRefine.enabled: false`, and `session_before_refine` returns `{skip: true}` for every round the gate did not initiate (§5). So cells remain exchangeable and the ON/OFF arms are two arms of the same blocked design the repo already uses. Three things that argument does **not** cover are pre-registered here as constraints:

- **C-1 — no online variant.** Any future design in which the artifact store mutates during a scored run is outside this spec and needs its own statistical design. v0 will not build one.
- **C-2 — the pairing is tighter than RLM-vs-baseline.** The two arms share a root, a corpus and a seed and differ only in the artifact set. The paired test is therefore the right test and the unpaired one would be wrong; the verdict reports the paired statistic only.
- **C-3 — the held-out split is spendable once per decision.** Repeated evaluations of successive artifact versions against the same held-out set are multiple comparisons. Every gate decision is recorded with its ordinal, and the report states how many decisions that split has absorbed. A split that has absorbed more than **5** decisions is retired and the reason is recorded.

### 4.2 The accept rule (pre-registered; changing it after seeing results is p-hacking)

A candidate set is **accepted** iff **both** hold:

- **Q — quality non-inferiority.** No held-out task that passes in the OFF arm fails in the ON arm. A task passes an arm at ≥2/3 seeds, per §8's rule (`ARCHITECTURE.md:377`). A single lost task rejects the set outright, with no cost trade permitted.
- **K — cost improvement.** On the per-task **total tokens** (`tokens_in + tokens_out` over all steps, §8's cost scorecard definition, `ARCHITECTURE.md:384`), the median per-task ON/OFF ratio is **≤ 0.90**, and the upper bound of a 10,000-resample paired bootstrap 95% CI on that median ratio is **< 1.00**. The resample is at task level, mirroring §8's inference layer (`ARCHITECTURE.md:378`).

Otherwise the set is **rejected**. Rejection is a result, not a failure: the run record keeps the candidate, its screens, its grid and its statistic, so a later proposer can be measured against it.

**Wall-clock is recorded but does not gate**, because the box's thermal drift (R9; +9.9% measured across C1b) is of the same order as the effect being sought. Tokens are the gated quantity; wall and energy travel as annotations, per §8's rule that *"any win claim states the cost multiple next to the margin"*.

### 4.2b PROPOSED — a reliability design, because v1 has almost no dynamic range (raised 2026-08-27, NOT ADOPTED)

Measured across decisions pc-01 and pc-02: **seven of the nine held-out tasks pass 3/3 in both arms, every time.** codeqa-04/05/07 and needle-05/06/07/08 have never produced a discordant cell. They contribute no quality signal at all, and on the cost statistic they contribute per-task ratios clustered at 1.0 that pull the median toward "no effect" regardless of what an artifact does elsewhere.

The only dynamic range in v1 is **agg-06 and agg-07**, which fail intermittently in both arms — the `749` and `0` answers, a mis-specified custody predicate — at roughly 15–20% of episodes. That is where an artifact could show an effect, and it is exactly where the design has the least support: 2 tasks × 3 reps.

**The design error is using one shape of experiment for two different questions.** A *cost* gate wants breadth: many tasks, few reps, median over tasks. A *reliability* gate wants depth: the tasks that actually fail, many reps, a failure-rate comparison. §4.1's blocked A/B is right for both; the sampling is not.

**Proposed.** For a reliability question, run a declared subset of the held-out side at high rep count — e.g. agg-06 and agg-07 at 10 reps per arm, 40 episodes, comparable in cost to a 9-task decision — and score it on discordant-episode counts (an exact McNemar test over ON/OFF pairs, matching §8's own inference layer) rather than on the ≥2/3 task rule, which cannot resolve a change from 20% to 5%.

`gate/run_decision.sh` now permits a held-out **subset** only when the caller sets `RLMH_SUBSET_REASON`, and records that reason with the decision. A subset with no declared design is still refused, as is any list that is not a subset. The 9-task split remains the set for cost decisions and is unchanged.

**Status: proposed, not adopted as a replacement for §4.2.** It changes what a decision measures. Under the owner's 2026-08-27 instruction to keep going and act on design ideas, it is run as a **declared, separately-scored sub-experiment** alongside §4.2 rather than instead of it; adopting it as the gate's rule remains the owner's call.

**The reliability reading, fixed here before decision pc-03 runs and not restated after:**

- **Unit is the episode, not the task.** 2 tasks × 10 reps × 2 arms = 40 episodes, blocked as `(task, rep)` with ON and OFF adjacent, arm order alternating by rep, exactly as §4.1.
- **Statistic: discordant `(task, rep)` pairs.** For each block, the ON/OFF outcome pair. `b` = pairs where OFF passed and ON failed; `c` = pairs where ON passed and OFF failed. Exact one-sided binomial test on `b + c` (McNemar), matching §8's own inference layer (`ARCHITECTURE.md:378`).
- **ACCEPT-reliability iff `c > b` and the exact one-sided p ≤ 0.05.** Cost is recorded beside it and does **not** gate this reading.
- **A tie or `b > c` is a REJECT**, and so is `c > b` at p > 0.05 — a direction without evidence is not a result.

**Stated power limitation, before the run rather than after.** At the ~15–20% per-episode failure rate measured on agg-06/agg-07, 20 pairs yields roughly 3–4 discordant pairs under a real effect. Reaching p ≤ 0.05 needs something like 5–0 or 6–1. **This design can confirm a large effect and cannot rule out a moderate one**; a REJECT here means "not demonstrated at this n", never "no effect". Raising n is the only fix and costs ~150 s per episode on these two tasks.

### 4.2a PROPOSED AMENDMENT — a quality-improvement path (raised 2026-08-27, NOT ADOPTED)

**Recorded before the positive control's verdict was known, and deliberately so.** Within the first ON/OFF pair of decision `pc-01` the artifact behaved exactly as designed — it reached the model (`in_window=1,0`), the answer was right, and the episode cost 116.0 s against the OFF arm's 58.6 s, because the artifact *asks for more work*: two independently written methods, reconciled.

That exposes a structural gap in §4.2 as written. **Q is non-inferiority, not improvement, and K requires cost to fall. So the rule can only ever accept an artifact that makes the agent cheaper** — an artifact that makes it *more reliable* is invisible to Q and penalised by K. The rule therefore cannot accept the very behaviour E3 identifies as the valuable one.

The spike's own reading did not have this gap. Its B2 was *"pass where Phase A failed, **or** median wall ≤0.8× and still passing"* — two paths. v0 kept only the cost path, and that was an error in drafting rather than a decision.

There is real quality headroom to gate on, despite v1 being nearly saturated: in step 2's verification `agg-07` scored **2/3 with an empty harness**, so a held-out task can fail without any artifact present.

**Proposed rule.** Accept iff **(Q ∧ K)** — no task lost and cost falls — **or** **(Q⁺ ∧ K⁰)** — some held-out task that fails OFF passes ON, no other task is lost, and cost does not worsen beyond a pre-registered multiple.

**Status: proposed, not adopted.** `pc-01` runs and is reported under §4.2 exactly as pre-registered; the threshold is not moved to make a control pass (plan step 4's stop rule). Adopting this is the owner's call, and if adopted the multiple in K⁰ must be fixed before any decision is re-scored under it.

### 4.3 Why cost, when §8 says there is no cost gate

`ARCHITECTURE.md:384` says: *"Deliberately no hard cost gate — the decision rule stays single."* That sentence governs the **S4 decision** — RLM against its baselines, where the question is whether the architecture wins on tasks. This gate answers a different question about a saturated instrument, and the repo's own primary metric is already the one used here: **`ARCHITECTURE.md:345` — "Primary metric: wall-clock per task at fixed quality."** Condition Q is "at fixed quality"; condition K is the cost half, moved from annotation to gate because on a benchmark where the agent already scores 8/8 there is no quality headroom to gate on. The deviation is deliberate, is scoped to this gate, and does not touch §8's S4 rule. §7 records it as a decision.

## 5. Enforcement — I1 structurally, not by policy

`ARCHITECTURE.md:472` requires: *"I1 is enforced structurally, not by policy: the writable surface is prompts, memories and skills; budgets, caps, routes and termination rules are not writable by any artifact, and a run in which one changed is a bug regardless of the score."* Four mechanisms, in order of how hard they are to bypass:

1. **The proposer is the only writer.** `autoRefine.enabled: false` (`settings.json`), and the gate extension's `session_before_refine` handler returns `{skip: true}` for any round it did not itself initiate — a hard `RefineSkippedError`, not a prompt instruction. In the spike the model was merely *asked* not to self-refine; here it cannot.
2. **The system prompt is filtered per turn.** `before_agent_start` rebuilds the `# Continual Harness State` block from the **accepted set** on disk, so an entry written directly by kernel code (`rlm.harness.create_*`) never reaches the model even if it lands in the file.
3. **The screens** (§3.4) reject a candidate naming a budget, cap, route or termination key before it is ever applied.
4. **The audit.** Every gate decision writes a row: candidate id, content sha256, the screens' verdicts, the ON/OFF grid, the statistic, the accept/reject, and the sha256 of the resulting `harness_state.json`. A held-out run whose harness sha does not match the accepted one is void.

**Rollback.** `/refine rollback <id>` restores the recorded `before` snapshots and is itself recorded as a new refinement with `rollbackOf` (`refinement.ts:813-845`) — and **rollbacks bypass the extension gate** by design (`agent-session.ts:8091` guards on `!options.rollbackId`), so the trusted path stays open when the gate is refusing everything. Note it is **not a byte-exact restore**: `version` is re-incremented and `updated_at`/`created_at` are re-stamped (`refinement.ts:769-785`). The gate's own store keeps the byte-exact prior state; prime-agent's rollback is the convenience path, rlm-halo's ledger is the record.

## 6. The split — and the trap in v1

**v0 evaluates on v1, and this is explicitly not the S6 gate.** `2026-08-25-benchmark-v2-design.md:105` reserves v2's held-out stream — *"not until S6's gate", "the S6 gate's only legal evaluation set"*. v0 must not spend it. v0 is the **dress rehearsal that proves the machine**; the S6-lite gate of `ARCHITECTURE.md:472` runs on v2's held-out when v2 exists, unchanged. Nothing in this spec amends that gate.

**The trap, measured from `bench/manifest.json` [V].** `regex_solvable` is **perfectly confounded with the question text**: agg-01/02/03 share `question_sha256 = e6576e21fa…`; agg-04/05/06/07 share `67c4d0898a…`. Splitting aggregation the natural way trains on one question shape and evaluates on another — a different distribution, not a held-out sample. Any v1 split must be drawn **within** a question shape.

**The v0 split, fixed now, before any run:**

| category | train | held-out | why it is exchangeable |
|---|---|---|---|
| code QA | codeqa-01, 02, 03, 04 | **codeqa-05, 06, 07** | one shared corpus (`code-bundle.txt`), one question shape, different symbols |
| needle | needle-01..04 | **needle-05, 06, 07, 08** | one question shape, distinct corpora drawn the same way |
| aggregation (regex-defeating only) | agg-04, agg-05 | **agg-06, agg-07** | same `question_sha256`; the other shape is excluded |
| aggregation (regex-solvable) | — | — | **excluded from v0.** agg-01/02/03 are a different question shape and agg-01 is one of only two adversarial tasks |
| synthesis | — | — | **excluded from v0.** Held for a later round so v0's held-out is not the whole benchmark |

**Totals: 10 train, 9 held-out.** A gate decision costs 9 tasks × 3 seeds × 2 arms = **54 episodes**; at the spike's measured 47–105 s median that is roughly 50–75 minutes.

`needle-01` carries the `adversarial` flag and sits in train; `agg-01`, the other adversarial task, is excluded. So no adversarial task is in held-out — recorded as a known limitation of v0's split, not an oversight.

## 7. Decisions taken

| # | decision | consequence |
|---|---|---|
| **D-S1** | **The gate is the proposer.** Candidates are generated offline from train traces and returned through `session_before_refine` as `{proposal}`; the built-in planner never runs | Forced by the seam (no per-edit veto) and by E1. The model's role becomes producing trajectories, not deciding what to keep |
| **D-S2** | **v0 learns `prompt` and `skill` only.** `memory` is deferred; `subagent` is excluded as a route under I1 | E3 says procedure is the valuable artifact. Enforced by S-kind, not by convention |
| **D-S3** | **The accept rule is quality non-inferiority AND a token-cost improvement** (Q ∧ K, §4.2), deviating from §8's "no hard cost gate" for this gate only, on the authority of §8's own primary metric | Recorded as a deviation, scoped to S6-lite, leaving the S4 rule untouched |
| **D-S4** | **v0 runs on a v1 split drawn within question shapes; v2's held-out is not spent** | v0 proves the machine. `ARCHITECTURE.md:472`'s gate is unamended and still runs on v2 |
| **D-S5** | **The artifact store is rlm-halo's own, with its own sha ledger.** `config_snapshot` integration is deferred | Verified [V]: `config_snapshot` cannot pin an arbitrary artifact today — every config model is `extra="forbid"` (`config.py:33-36`) and the pinnable slot set is closed and hand-enumerated in four places (`config.py:607-635`, comment at `:266-270`). Wiring artifacts in is a four-site schema change with a replay-compatibility question; it is real work and belongs to v1 of this slice, not v0 |
| **D-S6** | **The held-out split is spendable at most 5 times**, after which it is retired (C-3) | Multiple comparisons are the quiet way a gate stops meaning anything |
| **D-S7** | **The gate restores C5's `max_identical_turns` as a `tool_call` budget** (default 3: the run may reach 2, the next identical cell is blocked) | Added 2026-08-27 during plan step 2, on measurement rather than principle. One `agg-07` episode emitted **29 byte-identical cells over 51 turns, burned 88.7K tokens and 1,384 s and produced no answer** — R15's verbatim-repetition attractor inside prime-agent, which ships no identical-turn guard (its own issue #1326 reports 20–50 identical calls on a local Qwen). I1 makes budgets the scaffold's to own, and the scaffold's `config.yaml` already sets this one. It applies to both arms identically so it cannot bias an A/B. **What it does cost is comparability:** the spike's Phase A ran without it, so any Phase A comparison in a v0 report carries this note. It also narrows the spike's "A-loop: not reproduced" — true of those 24 runs, not a general claim; `docs/research/2026-08-27-prime-agent-spike.md` is a dated record and is not retro-edited |

## 8. Risks, pre-registered

| # | risk | why it is real | response |
|---|---|---|---|
| **R1** | **Nothing passes the gate.** v1 is saturated and the spike measured every artifact set costing *more* (E4). A gate that only ever rejects is a correct instrument and a useless product | The spike is the direct evidence: 4 of 4 tasks got more expensive | Rejection is a recorded, publishable result: *"a competent local root's self-proposed artifacts do not survive a held-out cost gate"* is a finding. But if **3 consecutive proposer rounds** produce nothing that passes the screens **and** nothing that passes the gate, v0 stops and the proposer's design — not the gate — is what gets revisited |
| **R2** | **The proposer is just the model again.** If candidate generation is an LLM pass, it may reproduce E1's answer-logging in a different wrapper | E1 was measured on this exact root | The screens are mechanical and run first; the proposer's prompt forbids naming any count, path, identifier or entity from a corpus; and every rejected candidate is kept, so the failure rate of the proposer is itself measured |
| **R3** | **S-I1 is keyword matching.** "Allow yourself more time before giving up" names no config key and passes | Stated as a property of the method, in the style of `leakcheck`'s own two documented limits | The screen's limit is written into its docstring and into every report. The real enforcement is that artifacts cannot alter config **at all** — they are prompt text, and budgets live in `config.yaml` and scaffold code (I1's actual mechanism). A persuasive artifact can only ask the model to behave differently; it cannot raise a cap |
| **R4** | **The alphabetical prompt window silently truncates.** More than 6 accepted entries of a kind and the gate's verdict is about a set the model never fully saw | Verified: no ranking, `slice(0, 6)`, 180-char content cap (`refinement.ts:466-481`) | `path` is assigned deterministically (§3.3) and every gate report states, per arm, how many entries were in-window and how many characters of each reached the prompt. More than 6 accepted entries of one kind triggers a pre-registered consolidation round rather than a silent drop |
| **R5** | **Vendored prime-agent moves.** v0.8.1 shipped the day the spike ran; the extension API is not a stability contract | Release cadence measured: 13 releases in 5 weeks | The version is pinned in the plan and the extension is one file. Each of the three seams has a startup assertion; a missing hook fails loudly at launch rather than silently degrading to "the model refines itself again" |
| **R6** | **Thermal drift contaminates the cost measurement.** +9.9% wall drift measured across C1b, R9's plateau on record | Measured [V] | The blocked `(task, seed)` design puts ON and OFF adjacent in time (§4.1), tokens rather than wall are gated (§4.2), and the report carries the per-block temperature deltas the trace already records |
| **R7** | **Everything derived from the split goes stale when the split changes, silently.** Added 2026-08-27 after it happened three times in one afternoon | Measured, all three during plan steps 2–4: (a) the first draw let two held-out code-QA answers duplicate train answers, so a memorised train answer would have scored a held-out task — caught by the screens on their first run against real material; (b) the screens' own fixture suite asserted a task was held-out after the redraw moved it to train; (c) a `heldout.txt` staged before the redraw still named `codeqa-06`, and a decision **started** on it — it would have evaluated on a training task with nothing to say so | **The split file is the only source of truth; everything else is a cache of it.** Three mechanisms, all now in code: `gate/make_split.py` refuses to write a split whose answers are not disjoint across sides or that straddles a question-shape cluster; the fixture suite reads task ids from the split rather than naming them; and `gate/run_decision.sh` refuses to run when its task list does not equal the split's held-out side, printing both. A fourth is owed and is not built: the split's sha256 is recorded per decision but no episode asserts it, so a split edited mid-decision would still be caught only after the fact |

## 9. What this does not do

- **It does not touch delegation.** prime-agent's subagents call the same model; there is no leaf. R16 stands, benchmark v2 remains the instrument, and nothing here advances or retires the RLM two-model thesis.
- **It does not amend the S6-lite gate** at `ARCHITECTURE.md:472`. That gate, on v2's held-out, is unchanged and unspent.
- **It does not satisfy S6's preconditions.** (a) is at 237/500 and is *"a content requirement before it is a count"* (`ARCHITECTURE.md:465`); (b) is a candidate, not demonstrated. v0 is a slice that builds the gate, not a claim that S6 has begun.
- **It does not sha-pin artifacts into `config_snapshot`** (D-S5). Until it does, an rlm-halo episode's snapshot does not record which artifacts were live — so v0's audit lives in the gate's own ledger, and that limit is stated wherever a v0 result is cited.
- **It does not decide the product.** DIRECTION.md's D4 publication question is untouched.

## 10. Open questions for the owner

1. **The proposer's model.** Local root (consistent with "everything local", and the thing under test), or a stronger model offline (better candidates, but then the artifact is not something the local agent could have produced alone)? v0 assumes **local root**; say if that should be an arm instead of an assumption.
2. **Failure appetite.** R1 says the honest expected outcome is that nothing passes. Is *"the gate works and rejects everything"* an acceptable v0 deliverable, or should v0 also hand-author one artifact known to help (e.g. E3's cross-check instruction) as a **positive control** that proves the gate can say yes? This spec's recommendation is **yes, add the positive control** — a gate never observed to accept is not yet known to work.

---

## 11. The proposer's contract, preserved from `gate/propose.py` before its deletion

**Added 2026-09-01.** `gate/propose.py` was deleted (reorganization group D8): it never
ran — `git log --all --diff-filter=A` finds zero `round-*.json` across the repo's whole
history, and §6 of the 08-28 results says so directly, *"The proposer does not exist.
Step 5 has not begun"*, in the same commit that added the file. Its output format was
prime-agent's `HarnessEntry` and its input was prime-agent session JSONL, so both ends
died with that harness.

Its SYSTEM prompt did not die with it. It is genuine design derived from a measured
failure: in pc-02's agg-07 trace the local root wrote **8 of 8 artifacts as memorised
answer logs**, which is what every constraint below exists to refuse. Any future
proposer — pointed at `src/rlm/`'s own scaffold rather than at prime-agent — starts here.

```
You read the execution trace of an agent that solved (or failed) one task and you
propose ONE reusable operating rule that would make the next attempt at a task of this
KIND go better.

You are not answering the task. You are not summarising what happened. You are writing
a rule for a future agent that will never see this trace.

HARD CONSTRAINTS, and a proposal that breaks any of them is discarded unread:
- Name no number, count, file path, identifier, organisation or entity that appears in
  the corpus or in the answer. A rule that mentions a specific value is a memorised
  answer wearing a rule's clothes.
- Name no configuration setting, budget, timeout, cap, route or termination rule. You
  cannot change those and asking for more of them is not a rule.
- Do not write a generic virtue. "Be careful", "double-check your work" and "think step
  by step" are worthless: the agent in this trace was already trying. A rule earns its
  place by naming the SPECIFIC MECHANISM that went wrong and the SPECIFIC OBSERVABLE
  that would have revealed it.

Reply with ONE JSON object and nothing else:
{"kind": "prompt" | "skill", "title": "<short imperative>", "content": "<the rule>"}

`kind` is "skill" only when the rule is a concrete procedure a future agent could follow
mechanically; otherwise "prompt". Keep `content` under 900 characters.
```

Two constraints are worth reading as measurements rather than style. The first — no
value that appears in the corpus — is the direct answer to the 8-of-8 failure. The
third — no generic virtue — is what separates a rule from the class of artifact this
project has already gated three times and rejected three times.

**What does NOT carry over:** `kind` is drawn from prime-agent's harness entry
taxonomy. A proposer aimed at `src/rlm/` takes its kinds from the 13 prompt slots
`Config._prompt_refs()` returns, and nothing in this scaffold has a "skill".
