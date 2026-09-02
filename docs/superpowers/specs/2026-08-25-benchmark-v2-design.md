# Benchmark v2 — the instrument that prices delegation, and supplies S6

**Date:** 2026-08-25 · **Status:** design, approved in session by the owner 2026-08-25; **amended §14 on 2026-09-02 and approved by the owner the same day**; next step is the implementation plan.
**Trigger:** `ARCHITECTURE.md` v0.4.0 (2026-08-25) schedules S6 and makes benchmark v2 a precondition of it under I5. The owner then asked to build v2.
**Supersedes:** decision **D-A4** of `2026-08-22-long-horizon-agent-design.md` ("single track; benchmark v2 parked, not dropped"). That document made the revisit due at Gate 0's resolution (§6.3); Gate 0 resolved PASS on 2026-08-23 (`docs/research/2026-08-22-gate0-soak.md`, commits `af80fa4`, `954d2ca`). See §12.
**Inputs:** `ARCHITECTURE.md` v0.4.0 §§1, 3, 8, 9, 10; `DIRECTION.md` amendment §0; `docs/research/2026-08-20-rlm-paper-fidelity-and-next-steps.md` §§2, 5.3; `docs/superpowers/specs/2026-08-22-long-horizon-agent-design.md` §§0, 4, 6; `docs/research/2026-08-22-gate0-soak.md`; `config.yaml`; `bench/`; `src/rlm/sandbox/child.py`; `src/rlm/measure/arms.py`; `traces/rlm.duckdb` at 707 non-superseded real episodes.
**Evidence discipline:** figures marked **[V]** were verified in this repo or its store during this session. Figures marked **[R]** are cited to a source and were not independently re-measured. Nothing is presented as measured that was not.

---

## 0. What is being built

**Sixteen frozen tasks in three categories, each existing as three separately generated corpora (train / held-out / practice), scored by three arms — designed so that the answer cannot be reached without delegating to leaf calls, and so that the design's own failure is detectable from the results.**

v2 does not replace v1. v1 stays frozen and keeps its S4 verdict; `benchmark.manifest_sha256` already makes cross-version comparison impossible, which is correct — they measure different things (`2026-08-22-long-horizon-agent-design.md` §6.2).

---

## 1. The mechanism: why v2 forces delegation, and why v1 could not

This is the load-bearing section. Everything else is consequence.

**The scaffold's own numbers bound how much text a root can ever read** [V], all from `config.yaml`:

| knob | value | consequence |
|---|---|---|
| `truncation_cap_chars` | 2,000 | every `print` observation is capped at ~530 tokens |
| `servers.root.ctx` | 32,768 | the root's window, and the conversation accumulates *every* prior observation |
| `budgets.max_predict.root` | 1,024 | the root's own output per turn, which also accumulates |
| `budgets.max_wall_clock_s` | 1,300 | the episode ceiling |

A read-turn costs roughly 530 observation tokens plus the root's code and reasoning. The conversation is retained as raw completions (`history_mode: raw`, v0.3.16), so it grows monotonically. **The root therefore reaches `context_exhausted` after on the order of 30–40 read-turns — an absolute ceiling near 20,000 tokens of corpus per episode, no matter how patient it is.** That figure is **arithmetic from the four knobs above, not a measurement**; the knobs are [V] and the conclusion is a derivation. It is stated as a bound rather than a point because the per-turn cost of the root's own code varies.

`docs/research/2026-08-22-gate0-soak.md` reached the same arithmetic independently from the other direction: a 32K window holds **70 turns at 467 tokens each**, and the loop "does not fail — it finishes, having taken seventy actions."

**This is exactly why benchmark v1 does not force delegation, and the diagnosis is not that its tasks were too easy.** v1's tasks never required *reading*. A regex reduces 490,000 characters to one integer, and that integer fits in 2,000 characters. The root never had to look at the text, so the 20,000-token ceiling never bound. The 2026-08-20 audit found all 30 tasks code-solvable this way [V]: needle is `^Custody key of record: (.+)\.$`; the "regex-defeating" aggregation falls to a header parse plus a string equality; synthesis is a set intersection; code QA is a `def` regex.

**So the requirement for v2 is precise: a per-item label that no deterministic program can produce from the item's text.** Once that holds, *something* must read every item. The root can read ~20,000 tokens. Everything beyond that must be read by leaves. Delegation stops being a stylistic choice and becomes the only reader that fits the budget.

**Sizing follows directly.** Static corpora are built at **~60,000 tokens** — three times the root's ceiling, a margin wide enough that no clever paging strategy closes it, and cheap to run (~139 windows at the 640/480 geometry).

---

## 2. Where the labels come from

**Decision: real human-labelled items inside a synthetic wrapper.** [owner, §11 D-B1]

The label source is the one thing v1 got wrong twice, and the failure mode is structural rather than careless: **a generator that emits item text *from* a label is invertible by construction.** Whatever template produced the text can be inverted by a parser, and the S4 root wrote exactly that parser (`milestones/s4/RESULTS.md`). `bench/corpus.py`'s `regex_at_chance` guard tested seven bag-of-words regexes against the Disposition line alone and never tested a parser reading the header — satisfied in letter, defeated in spirit [V].

The fix is to stop generating labels at all:

1. **Items and their labels come from a human-labelled corpus** — TREC coarse question classification (6 labels), the same source OOLONG uses [R].
2. **The builder wraps them in a synthetic register** with coined vocabulary, matching v1's contamination discipline: entity names, identifiers and structure are generated and cannot exist in any training corpus.
3. **Each task samples a different subset**, so the aggregate answer is not memorisable even for a model that has seen TREC.
4. **The builder computes the answer from the labels it sampled.** Answers are computed, never typed — v1's rule, preserved exactly.

**Build-time validation, mandatory before freeze:** the at-chance artifact is extended from bag-of-words regexes to a **parser adversary** — a program given the full item text and the register structure, allowed to parse fields, that must fail to reproduce the aggregate. A category whose parser adversary succeeds is not frozen. This closes the agg-04 loophole by name.

**Fallback, pre-registered so it is not a decision taken under pressure:** if TREC's licence or distribution proves unusable at build time, substitute any human-labelled natural-language classification set with ≥4 classes and ≥5,000 items. The design does not change; only the source string in the manifest does.

---

## 3. The sixteen tasks

| category | n | shape | why it exists |
|---|---|---|---|
| **linear-semantic** (static) | 6 | items carrying a human label; the question aggregates over them | the paper's linear-semantic regime — one of the two where sub-calls pay [R] |
| **interactive investigation** | 6 | corpus reachable only through a query interface; many turns | prices delegation *and* supplies long-horizon trajectories (§4) |
| **code-solvable control** | 4 | deterministic code suffices | keeps the root scored for choosing code when code suffices — without these, v2 is rigged in the opposite direction |

**Quadratic (Pairs-shaped) is a flag, not a category** — two linear-semantic and two interactive tasks carry it. **Adversarial injection (R12) is a flag** on two interactive tasks. This follows v1's own precedent, where adversarial context is a flag and "tasks declare the category of their underlying shape" (`ARCHITECTURE.md` §5).

Three categories rather than four is deliberate: at N=16 a fourth category leaves rows of three, and the §8 per-category table stops distinguishing anything from noise.

**Per-category split is pre-registered here and fixed at freeze: 6 · 6 · 4 = 16.** The scoring denominator depends on it.

---

## 4. The interactive category, specified

**What it is not.** It is not ARC-AGI-3. ARC does not price delegation — `2026-08-22-long-horizon-agent-design.md` §6.3 states this plainly — and pricing delegation is this category's entire reason for existing. The Gate 0 measurements stand and Gate 1 remains available; it is simply not next.

**What it is.** In the static categories the root receives `chunks` as a variable: the whole corpus is present from turn one. In this category **it does not.** The root receives a stateful query object, and the corpus lives behind it:

- `env.search(term) -> list[hit]` — hits carry a document id and an offset, never bulk text
- `env.open(doc_id) -> summary` — structure and length, not content
- `env.window(doc_id, i) -> str` — one bounded slice
- every return value passes through the same `truncation_cap_chars` ceiling as any other observation

**The corpus behind the interface is built at ~200,000 tokens** — ten times the root's ceiling, and the cap v1 already ruled workable for full-coverage categories. Corpus size costs nothing at run time here: the agent pays for the operations it issues, not for what it never opens. A task requires many rounds of act-observe, and its answer requires labels that only a model can assign — so the agent must both **navigate** (many turns) and **delegate** (leaves read what the root cannot).

**This is not a simulator, and I3 is not strained.** `env` is a deterministic index over real text — the same class of object as the REPL itself. No world model sits between the root and the REPL or between the scaffold and the final answer. The distinction that matters under I3 is *modelled vs. real*, and an index over stored text is real.

**Secondary metric, reported and never gating:** the builder knows the corpus it generated, so it records a **reference action count** — the number of `env` operations an optimal path needs. Episodes report actions-taken against it. This gives the long-horizon story a number without inventing a gate for it. Pass/fail remains the checker.

---

## 5. Corpora, splits and freeze

**Each of the 16 task shapes exists as three separately generated corpora**, differing only in generation seed and sampled subset:

| stream | frozen? | scored? | purpose |
|---|---|---|---|
| **train** | yes, sha-pinned | yes | the measurement run (§6) |
| **held-out** | yes, sha-pinned | **not until S6's gate** | the S6 gate's only legal evaluation set |
| **practice** | no — regenerable on demand | never | trajectory supply for S6 |

**Why three and not two.** S6 needs volume — its precondition is ≥500 successful episodes, and today the store holds 237 successful RLM episodes of which 6 delegate substantively [V]. Sixteen tasks × 3 seeds yields 48 episodes per pass, so volume must come from re-running. Re-running the *train* corpora would overfit the scored set; re-running *held-out* would destroy the gate. The practice stream is regenerable without limit and is never scored, so trajectories accumulate at zero cost to either frozen set.

**Practice counts as train-side.** `ARCHITECTURE.md` §9 S6 already reads: *"Artifacts may be derived only from the v2 train split; an artifact traceable to a held-out task voids the run."* Practice is part of that side, so the rule stands unmodified.

**Freeze:** `benchmark.version: v2`, its own `manifest_sha256`, its own manifest file. The v1 freeze is untouched. Splitting is done at freeze, before any result exists — splitting after seeing results is choosing a test set to taste, and it would void the S6 gate before that gate exists.

---

## 6. Arms, scoring, and the reading pre-registered against this design

**Arms: `rlm`, `rlm-nosubcalls`, `B2`.** [owner, §11 D-B2]

`rlm-nosubcalls` is the paper's own ablation ("RLM, no sub-calls"), made an explicit named arm rather than something S4 was by accident.

**`rlm-nosubcalls` is simultaneously the ablation and the instrument that audits this design.** Pre-registered reading, written before any run:

- **`rlm-nosubcalls` scores near zero on linear-semantic and interactive, `rlm` scores well** → the tasks force delegation, and the margin is the price of delegation. Design worked.
- **`rlm-nosubcalls` passes ≥3 of the 6 linear-semantic tasks, or ≥3 of the 6 interactive tasks** — half a category — → **the tasks failed to force delegation for that category. That is a finding about the benchmark, not about the model,** and the category is re-authored rather than reported. The threshold is fixed here, before any run, because a reading chosen after seeing the numbers is not a reading.
- **`rlm-nosubcalls` beats `rlm` on the code-solvable controls** → expected and healthy; it is what those four tasks are for.

**Pre-registered asymmetry: B2 does not run the interactive category.** A fixed map-reduce pipeline requires a static corpus to partition; behind a query interface there is nothing to partition. B2 runs the 10 static tasks. Stating this at design time is mandatory — discovering it at run time would be a hole in the grid, and quietly scoring B2 as failing six tasks it structurally cannot attempt would manufacture a margin.

**Scoring** inherits §8 unchanged: a task passes for an arm if ≥2/3 seeds pass; the primary artifact is the per-task × per-arm × per-seed grid; the exact sign test and the 10,000-resample paired bootstrap CI accompany every margin; `budget_kill` and `context_exhausted` count as failures for every arm; `error` episodes re-run once via `superseded_by`.

**Margin rule at N=16: +2 tasks**, by the same scaling v1 used (+2 at N=20, +3 at N=30). Fixed now.

**Escalation** is inherited: a net margin in {+1, +2} runs seeds {4, 5} on discordant tasks for that pair only, re-decided on ≥3/5.

**Checkers: no new ones.** On inspection `int_exact`, `uuid_exact`, `set_exact` and `name_exact` cover every v2 answer shape [V] — an earlier statement in session that v2 needed new checkers was wrong. Their existing near-miss suites apply unchanged, and `bench/manifest.py`'s `MIN_NEAR_MISSES` check keeps guarding them.

---

## 7. Cost

At the shipped 640/480 geometry, ~1.8 s per leaf call on Vulkan, and ~139 windows for a 60K-token corpus:

| | tasks | arms | episodes | ~min/ep | hours |
|---|---|---|---|---|---|
| static (linear-semantic + control) | 10 | 3 | 90 | ~11 | ~16.5 |
| interactive | 6 | 2 | 36 | ~18 | ~11 |
| **total (train split, first full run)** | | | **126** | | **~27 h** |

Inside the 36 h the owner approved. The held-out split costs the same again but is not spent until S6's gate; the practice stream is spent deliberately and incrementally.

**Geometry stays at 640/480.** [owner, §11 D-B4] Vulkan would support 2,048-token windows at roughly twice the throughput (v0.3.21), but changing the benchmark and the chunker geometry in the same step confounds two variables. A geometry sweep *on* v2 is a clean follow-up once v2 exists.

---

## 8. §8's preconditions, applied

All four are inherited and none is weakened:

1. **Closed-book probe** — every v2 question runs context-free against both models, 3 seeds; any task answered correctly in ≥1/3 seeds is rewritten or replaced. This matters more for v2 than v1: TREC is public and old, so the probe is load-bearing rather than ceremonial. `bench/closed_book.py` runs as-is.
2. **Synthetic identifiers** — the register wrapper, entity names and keys are generated; only the item text and its label are human-sourced.
3. **Corpus dating** — the manifest records each corpus's build date against the assumed cutoff, as `bench/build.py` already does.
4. **Checker near-miss suite** — ≥3 authored plausible-but-wrong answers per checker, unchanged.

**Added for v2, because v1's failure was not covered by any of the four:** the **parser adversary** of §2, run at build time over every linear-semantic and interactive task. It is a precondition of freeze with the same standing as the other four.

---

## 9. Implementation surface

| area | change | risk |
|---|---|---|
| `bench/corpus_v2.py` | new — register wrapper, TREC ingestion, sampling, interactive corpus + reference action count | medium |
| `bench/build_v2.py` | new — emits 16 shapes × 3 streams, computes answers, writes tasks | low |
| `bench/parser_adversary.py` | new — the §2 build-time guard | medium; it is the guard that must not be weak |
| `bench/manifest.py` | extend for `version: v2`, three streams, the new precondition | low |
| `src/rlm/sandbox/` | **one new bridge verb** for `env`, beside `llm_query` and `final_answer`, following the same `_RESERVED` pattern | **highest in the plan** |
| `src/rlm/measure/arms.py` | new `rlm-nosubcalls` arm | low — it is `rlm` with the dispatcher refused |
| `config.yaml` | `benchmark.version`, v2 manifest sha | low |

**The sandbox verb is the risky item and is called out as such.** `src/rlm/sandbox/child.py` documents that scaffold-side control must live outside any namespace a cell can edit, and enumerates the reachable hijack routes it deliberately does not try to close in the namespace. A new verb inherits that contract and must inherit its tests — including the analogue of `test_hijacked_llm_query_cannot_alter_scaffold_side_control`.

---

## 10. Risks, pre-registered

| # | risk | response |
|---|---|---|
| **V2-R1** | The parser adversary is too weak, and the interactive/linear tasks are code-solvable after all — v1's failure, repeated a third time | `rlm-nosubcalls` scoring well is the detector (§6), and its reading is fixed before the run. The adversary is authored by a different pass than the generator |
| **V2-R2** | TREC labels are memorised, so leaves label from parametric knowledge rather than from reading | Harmless for the aggregate — the leaf is *supposed* to label — and the sampled aggregate is not memorisable. The closed-book probe bounds the residue |
| **V2-R3** | The `env` verb widens the sandbox's attack surface | Same `_RESERVED` contract and the same test class as `llm_query`; the verb returns text only and holds no scaffold control |
| **V2-R4** | Interactive episodes hit `context_exhausted` as the *intended* failure and the category scores zero for every arm, measuring the window rather than the agent | The reference action count (§4) separates the two: an episode that exhausts far above the reference path is the agent's failure; one that exhausts near it is the budget's, and the corpus is re-sized before freeze |
| **V2-R5** | Practice-stream trajectories drift from the frozen shapes, so S6 learns something v2 cannot score | Practice is generated by the same builder from the same shapes; only the seed differs. Drift would be a builder bug and is testable |
| **V2-R6** | ~27 h is a projection from v1-era per-call timings, and interactive episodes have no precedent at all | Smoke-calibrate before the scored run, as S4 did (its 41.9 h projection held). If the projection breaches 36 h, the owner decides before the run, not during it |

---

## 11. Decisions taken (decision record)

Taken by the owner in session on 2026-08-25. Re-litigate only with new facts, per `ARCHITECTURE.md` §11.

| # | decision | consequence |
|---|---|---|
| **D-B1** | Labels come from **real human-labelled items inside a synthetic wrapper** | The generator never emits text from a label, so the invertibility failure of v1 cannot recur by construction. External dependency accepted (§2 fallback) |
| **D-B2** | Arms are **`rlm`, `rlm-nosubcalls`, `B2`** — three, not five | `rlm-restricted` and `rlm-default-template` are deferred; they refine rather than decide, and the five-arm shape is what breached budget on 2026-08-20 |
| **D-B3** | Split **by corpus, not by task** — 16 shapes × separate streams | Measurement keeps all 16 tasks of power and so does the S6 gate; the cost is generator time, not GPU time |
| **D-B4** | **One v2 category is interactive** | Delegation measurement and long-horizon trajectory supply come from one authoring effort. Revisits and supersedes D-A4 (§12) |
| **D-B5** | Geometry stays **640/480** for v2 | One variable at a time; a geometry sweep on v2 is a clean follow-up |

---

## 12. What this does to D-A4, and to the spec

**D-A4 is superseded, on the terms its own document set.** `2026-08-22-long-horizon-agent-design.md` §6.3 recorded a standing argument for revisiting D-A4 and stated that "the decision is not due until Gate 0 resolves." Gate 0 resolved PASS on 2026-08-23. The argument it recorded — that ARC cannot price delegation, and that one interactive v2 category buys both measurements from a single authoring effort — is the design above. The long-horizon programme is **not cancelled**: Gate 0's measurements stand, and Gate 1 remains available.

**`ARCHITECTURE.md` needs one edit, deferred to the implementation plan rather than taken here:** §8 is written for v1's four categories and N=30. v2's categories, N=16, +2 margin rule, three streams, the parser-adversary precondition and B2's pre-registered abstention must land as a §8 amendment with a version bump, per the amendment rule. **No gate or invariant changes** — §8's rules are inherited, not rewritten.

**`DIRECTION.md` needs nothing.** Amendment §0 already names the destination and already states that v2 precedes the learning loop.

**One correction owed to today's v0.4.0 entry:** it was written without knowledge of `2026-08-22-long-horizon-agent-design.md` and therefore does not mention the long-horizon track or D-A4. A line naming both belongs in the §8 amendment above.

---

## 13. Explicitly not in this design

- **ARC-AGI-3, Gate 1, and the supervisor arms.** Available, not next (§4).
- **`rlm-restricted` and `rlm-default-template`.** Deferred by D-B2. If v2's margin lands in the escalation band, they are the first thing added, on discordant tasks only.
- **B1 and B3.** Single-shot arms over corpora sized past any window measure truncation policy, not the thesis. v1 already priced them, and the B1 re-run (v0.3.25) showed what that price was worth.
- **`max_depth > 1`.** §9 forbids proposing it before the S4 gate, which passed on v1; v2 does not reopen it.
- **Any change to `size_tokens`, `dispatch_concurrency` or `slot_policy`.** Still pinned, still owed, still unrelated to this work.
- **The learning loop itself.** v2 is its precondition, not its first step.

---

## 14. Amendments of 2026-09-02 — what one week of measurement forced into this design

**Approved by the owner in session on 2026-09-02.** Inputs: `docs/research/2026-09-01-s5-a3b-root-smoke.md` (the same-model smoke; the `slot_pool_error_drained` mechanism, fixed in `5be9a41`), `docs/research/2026-09-02-delegation-brake-review.md` (the tasks, not the prompt, hold delegation at zero on v1), `docs/research/2026-09-01-three-papers-verified.md` §5(a) (S4's margins are cross-model). Everything in §§0–13 stands unless a line below says otherwise. Two owner decisions were taken before the amendments and are recorded first.

### 14.0 Decisions taken 2026-09-02 (extends §11)

| # | decision | consequence |
|---|---|---|
| **D-B6** | **All sixteen tasks are built in one pass**, interactive included — no staging | D-B4 is confirmed, not re-litigated. The `env` verb (§9) is built alongside the static categories, and v2 freezes once, at N=16, exactly as §3 pre-registers. The owner chose this over a static-first stage knowing the verb is the design's highest-risk item |
| **D-B7** | **v2 runs with one model in every arm**: the leaf GGUF (`Qwen3.6-35B-A3B-UD-Q4_K_M`) as root and as leaf, the S5 "A3B-as-root" configuration | `rlm`, `rlm-nosubcalls` and `B2` share a model, so every margin in the v2 verdict is the scaffold's. The 27B-root configuration becomes a later S5 row and is not part of the v2 verdict |

### 14.1 Configuration (amends §6, §7)

v2 runs from **`config.v2.yaml`**, derived from `config.s5-a3b-root.yaml` (itself five lines from `config.yaml`: root model → the leaf GGUF, `dflash: false`, plain Vulkan `backend_dir`, own root log path, DFlash2 flags dropped), with `benchmark.version: v2`, the v2 `manifest_sha256`, and the prompt pins of §14.3. `config.yaml` is not modified. `checks/test_config.py:625`'s assertion that `bench_leaf.model == leaf.model` holds trivially. Why this is not a §11 re-litigation: on 2026-08-25 the project believed its baselines were same-model; `2026-09-01-three-papers-verified.md` §5(a) showed S4 compared the 27B root carrying the scaffold against the A3B leaf carrying none. v2 does not inherit that.

**Cost consequence.** The A3B root decodes at ~59 t/s against the 27B's ~13 (35 with DFlash2), and the smoke's free-root cells ran at 0.05–0.22× the pre-registered projection while forced-delegation cells ran at 3–21×. §7's ~27 h is therefore **unreliable in both directions** and is superseded by §14.6.

### 14.2 Second build-time adversary: root self-read (amends §2, §8)

The parser adversary closes v1's failure. It does not close the one the brake review found: the free root solved needle-02 by **printing whole chunks into its own window and reading them** — lossy, lucky, and enough. Any task whose answer sits in a few locatable chunks is answerable by a root that never delegates, however semantic the label.

**Rule, pre-registered:** a task is rejected at build time if its answer is reachable by locating **≤ K = 40 chunks** by any deterministic locator (regex, keyword set, header or field parse — the same family the parser adversary is allowed) and reading those chunks in full. K is derived, not chosen: §1's ~20,000-token root reading ceiling ÷ ~530 tokens per capped observation ≈ 38 read-turns, rounded up for margin. The builder computes, from the labels it actually sampled, the **minimal necessary window set** — the smallest set of chunks whose contents determine the answer — and requires its size to exceed K. For a linear-semantic aggregate over every item this is automatic at ~139 windows; the check makes it explicit, and it is what stops an author from writing "how many HUM questions mention Russia" (locatable by one keyword). For interactive tasks the same rule applies over `env.window` slices: the minimal necessary slice set must exceed K, so no ≤ 40-window navigation reaches the answer.

**Standing:** a precondition of freeze, equal to the four inherited from §8 and to the parser adversary. Both adversaries are authored by a different pass than the generator (V2-R1's response, now covering both). `rlm-nosubcalls` remains the run-time detector of both adversaries' weakness (§6): if it passes half a category, the category is re-authored.

### 14.3 Truthful prompts per arm (amends §6, §9)

The brake review found the pinned prompts telling arms things that are false in their environment: `root.v3.md:37` says `llm_query` *"reaches a small, fast, stateless model"* (false under D-B7 — it is the same model), and `config.py` appends the unforked strategy blocks — *"Scan in code first ... zero sub-calls"* — to a root whose arm cannot read chunks. Neither changed today's outcomes measurably; both violate the arms' own rule that a prompt must not lie about the environment, and both would contaminate `rlm-nosubcalls`, whose whole point is to be `rlm` with delegation removed and nothing else.

Three new pinned files, sha-pinned as always; **no existing prompt file is edited**:

- **`root.v4.md`** — `root.v3.md` with :37 corrected to what is true under D-B7 (*"`llm_query` reaches a sub-model — in this configuration the same model as you, but with no REPL, no memory between calls, and no knowledge of your task beyond the string you hand it"*), and nothing else. Pinned for `rlm` in `config.v2.yaml`. The v3→v4 changelog line records the one sentence.
- **`root-nosubcalls.v1.md`** — `root.v4.md` with the `llm_query` API entry, the "sub-model" section, and the sub-call-referring tips removed (tips renumbered), so the prompt describes a REPL with `context`, `chunks` and `final_answer` and nothing more. Pinned for `rlm-nosubcalls`. The arm's dispatcher refusal (§9) stays: the prompt and the runtime agree.
- **`-nosubcalls` variants** of each strategy block that mentions sub-calls (needle, aggregation, code_qa; synthesis has none), with only those sentences removed.

The `rlm` arm's block set is unchanged. The brake itself (`root.v3.md:53,59,60`) is **kept** in v4 — the brake review found it accurate advice and unmeasured, and v2's code-solvable controls exist precisely to keep it honest.

**Name collision, resolved here.** `2026-09-02-delegation-brake-review.md` §4 describes an *optional* brake-removal ablation prompt and calls it `root.v4.md`. That ablation was never built and gates nothing. **`root.v4.md` is reserved by this spec for the :37 correction above**; if the ablation is ever built it is `root-nobrake.v1.md`, so the two cannot be confused in a `config_snapshot`.

### 14.4 The slot-pool precondition is met (amends §7, §10)

§7's cost model assumes a linear-semantic episode drives ~139 windows through a 128-slot leaf pool via rotation. Until 2026-09-01 that assumption was false in the scaffold: any `asyncio.gather` wider than the pool on a virgin generation was killed as `slot_pool_error_drained` before the leaf received a request (129 cancelled / 7 exhausted / 0 leaf errors in the smoke's own trace), and a pool spent by the root's own cancellations read the same way. Both are fixed (`5be9a41`; the judgment now follows `quiesce()` and requires a leaf-failed window; three RED→GREEN tests in `checks/test_episode.py`). **V2-R7, added:** the bench arms' mirror `ArmEpisode._dispatch_leaf` still judges before the quiesce — correct while its callers are serial, and B2 is. If a v2 arm ever fans out through `call_leaf`, that judgment moves first (its docstring names the condition).

### 14.5 Trace archiving is a precondition of the practice stream (amends §5)

The 707-episode store §5 cites as [V] no longer exists: `traces/` is gitignored, it was reset between 2026-08-25 and 2026-09-01, and R16's 237 / 15 / 6 now rest on documents alone. S6's practice stream is worthless if the same happens to it. **Rule:** every v2 run — smoke, train, held-out, practice — is exported with `rlm export <run_id> --dest D:\AI\rlm-halo-archive\<date>-<name>-<run_id8>\` before any `traces/` reset, and the research record of that run carries the archive path and the bundle manifest's `config_snapshot_sha256`. First instance: `2026-09-01-s5-a3b-root-smoke-b650df33` (`9238f263…`). The export reads a closed store, so this is a post-run step, never concurrent with the bench.

### 14.6 Label source, verified (amends §2)

`CogComp/trec` on Hugging Face: 5,452 train + 500 test human-labelled questions, 6 coarse / 50 fine labels, ~10 words per item, downloads 0.36 MB. **Licence on the card: `unknown`**; the CogComp homepage distributes it for research use. Decision under §2's pre-registered rule: **TREC stays primary** — the same source as OOLONG, so v2's linear-semantic regime is comparable with the paper's, and 6 coarse classes is a labelling task a 3B-active leaf can do without drowning the aggregate in label noise. The manifest records the licence status verbatim. **Fallback resolved to a named set:** `PolyAI/banking77` (CC-BY-4.0, 13,083 expert-labelled queries, 77 intents), verified 2026-09-02; it is the substitute if TREC's status is ruled unusable at build time, at the cost of a much harder per-item label. §8.1's closed-book probe is load-bearing for either source.

### 14.7 Smoke before score (strengthens V2-R6)

The S5 smoke showed the pre-registered projection constants (`450 s` chunked non-aggregation, `60 s` single-shot, `2.78 s/window` aggregation) off by 5–20× in both directions on this configuration. **v2's scored run does not start until a `--smoke` pass on the frozen train split has printed its projection**, and if that projection exceeds the owner's 36 h the owner decides before the run. The smoke also validates, on real episodes, that `rlm-nosubcalls` refuses `llm_query` and that the `env` verb survives the same hijack tests `llm_query` does (§9).

### 14.8 Unchanged, restated so no one wonders

Sixteen tasks, 6 · 6 · 4; three streams; three arms; B2 abstains from interactive; +2 margin at N=16; §8's inference layer; no new checkers; geometry 640/480; the `env` verb as §4 specifies it; the ARCHITECTURE.md §8 amendment deferred to the implementation plan (§12).
