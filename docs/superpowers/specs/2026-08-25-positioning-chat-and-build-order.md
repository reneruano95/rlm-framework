# Positioning, the chat surface, and the build order — session decisions of 2026-08-25 (evening)

**Date:** 2026-08-25 · **Status:** decisions taken by the owner in session; this document is their record.
**Trigger:** the owner asked what benchmark v2 has to do with an agent one can talk to, then asked whether the project makes sense at all, then commissioned adversarial research on differentiation before authorizing any build. Three research passes ran in parallel (ecosystem survey, self-improvement prior art, adversarial claim review); their findings are summarized here and govern the decisions below.
**Inputs:** `DIRECTION.md` §0; `ARCHITECTURE.md` §1, §5, §9 S6, §10 R16; `docs/superpowers/specs/2026-08-25-benchmark-v2-design.md`; `src/rlm/cli.py`; `src/rlm/episode.py`; web sources cited inline as [R] (reported by the cited source, not independently re-measured). In-repo statements are marked [V] where checked against the tree this session.
**Evidence discipline:** unchanged — nothing is presented as measured that was not. External figures carry their source; invariants and gates are quoted from the spec, not paraphrased.

---

## 0. Why this document exists

The owner's questions exposed a real gap between the documents and the felt direction: the spec forbids chat modes outright (`src/rlm/cli.py:12`, "no daemon, no REST API, no web UI, no interactive chat mode"), §1 excludes being a chat product, and yet the destination — set by the same owner five days ago — is a self-improving agent. Left unrecorded, the next session rediscovers this contradiction from scratch. This document resolves it, fixes the positioning that arbitrates all future build/no-build calls, and registers what the adversarial review actually found, including the findings that hurt.

---

## 1. The compass (the positioning adopted)

> **A local agent that learns from *your* corpus, on *your* box, where every improvement carries auditable proof and no data ever leaves — built on the only consumer-class hardware where that is economically coherent.**

Three claims compose it, each surviving adversarial review:

1. **The bandwidth wall makes context-by-reference an economic necessity, not a benchmark curiosity.** Local prefill on Strix Halo-class parts is on the order of ~170 t/s for a 27B model [R], so a message-array agent pays minutes per query while a reference-probing agent pays per operation. On hosted APIs this trade is a trick; here it is the difference between usable and unusable.
2. **Replayable provenance as a compliance product.** Every mainstream harness treats budgets and gates as runtime plumbing. None ships an auditable change-control record: deterministic replay of every episode (`root_view_hash` re-derivation, [V] `src/rlm/cli.py` replay checks), sha-pinned artifacts, pre-registered gates. For the on-prem private-org buyer this is the reason to buy, not hygiene.
3. **The impossible quadrant.** Dense aggregation over corpora far beyond any local window cannot be solved by a bigger quantized model (does not exist locally) nor by a cloud API (the corpus cannot leave). That quadrant is empty across the surveyed ecosystem.

**The feature test this imposes — every proposed build must answer yes to at least one:**
(a) does it improve quality-per-token/joule on local unified-memory hardware?
(b) does it thicken the auditable provenance story?
(c) does it attack the aggregation-massive-private quadrant?

If none: not built, regardless of how standard it would make the product feel.

---

## 2. What the adversarial research found (recorded unsoftened)

### 2.1 The three damages

| # | Finding | Evidence |
|---|---|---|
| **F1** | **Prime Intellect shipped the destination label first.** `prime-agent` (Aug 2026, MIT, ~18k★) is titled "a Self-Improving RLM harness": TUI, daemon with resume, `/refine` over prompts/memories/skills/subagent specs with snapshots+rollback, autonomous mode under turn/token/time budgets. The Continual Harness paper (arXiv:2608.23552, arXiv:2605.09998) is published prior art — co-authored by the RLM paper's own lead author [R] | https://github.com/PrimeIntellect-ai/prime-agent |
| **F2** | **Quantized small-code roots sit in the regime where RLM scaffolds collapse.** Replications 2026: nano-class roots −9.5pp inside a REPL (minRLM); qwen3-coder:30b located the right spans yet scored 0/7 formal; Kimi K2 native long-context 86.6% → 60.0% inside the scaffold; depth-2 uniformly degrades; the paper author's own post-mortem concedes prompt-steered fragility [R]. Our root is a Q4_K_M 27B. The thesis is therefore *unproven for our configuration* — which is precisely what benchmark v2 exists to measure | arXiv:2603.02615; avilum.github.io/minrlm; alexzhang13.github.io/blog/2026/longcot-rlm |
| **F3** | **Adjacent layers commoditized within the year.** Hard budgets are table stakes (OpenHands `max_budget_per_task`, LangGraph recursion/middleware limits, smolagents breakers, and the MIT reference library itself ships `max_budget`/`max_tokens`/`max_errors` with typed exceptions). Held-out-gated skill libraries are published (**GRASP**, arXiv:2605.29668 — balanced held-out probe, hard regression budget, evaluated on gpt-oss-120b-class models; **SKILLGEN**, arXiv:2605.10999). And 100B+ MoE quants now run on one Strix Halo box at 18–55 t/s with zero scaffold [R]. "Scaffold-owned budgets" alone is not a moat; neither is "self-improvement" alone; neither is "local inference" alone | github.com/OpenHands/OpenHands; gepa-ai; AMD developer playbooks |

### 2.2 The three survivors

Stated in §1. Nothing else reviewed survived contact: novelty of mechanism is gone everywhere; what remains open is the **conjunction** — closed loop (diagnose → mutate → gate → rollback), mechanically held-out-gated acceptance, traces replayed as a regression corpus, hard invariants the learner structurally cannot touch, all on-prem over a private corpus, on hardware where the design is an economic necessity. No surveyed system combines these. The two nearest neighbors stop short deliberately: context-labs/HALO ends at a diagnostic report and hands mutation to a human + coding agent [R]; prime-agent lets the model apply its own refinements with only the base prompt fenced.

### 2.3 Prior art the learning loop must cite and beat

GEPA (external optimizer, Pareto admission, train/val/test discipline), ACE (deterministic non-LLM merge — the right enforcement primitive), Meta-Harness (held-out-model transfer evaluation — the right evidence template), GRASP/SKILLGEN (per-edit held-out gating), Letta Context Repositories (git-versioned memory). S6-lite's gate predates none of these; its claim is integration and enforcement hardness, not mechanism novelty. Any pitch deck that says "novel" in front of someone who reads arXiv loses.

---

## 3. Decision record

Taken by the owner, 2026-08-25 evening session. Re-litigate only with new facts, per the standing rule.

| # | decision | consequence |
|---|---|---|
| **D-C1** | **The §1 compass is adopted as the arbitration instrument**, with the three-part feature test | Every future proposal is judged by (a)/(b)/(c) above. Positioning supersedes feature-parity instincts |
| **D-C2** | **A sixth CLI verb, `rlm chat`, is authorized under a written firewall** | Amends, by exception, §5's "five verbs are the whole thing" and the "no interactive chat mode" non-goal. The amendment entry is owed in `ARCHITECTURE.md` §14 when the verb lands. Firewall specified in §4 |
| **D-C3** | **The v2→S6 build order stands, with the collapse finding attached rather than ignored** | v2 is now double-duty: it prices delegation *and* it is the experiment that decides whether F2 applies to us. If the root collapses, the recorded exit is the mit-oasys RLM-post-trained LoRA row (S5 candidate track), not abandonment |
| **D-C4** | **HALO name collision managed, not merged**: internal name stays; external rename happens before the first public release | context-labs/HALO ("Hierarchical Agent Loop Optimizer", MIT, ~1.1k★) occupies the adjacent "RLM optimizes your harness" niche [R]. README must state the distinction at publication time; searchability/confusion cost accepted until then |
| **D-C5** | **Adopt-vs-fork of prime-agent is resolved: build.** The four-pillar test decides: prime-agent lacks Capa 0 entirely (models are OpenAI endpoints; slot pools, KV arithmetic, measured serial dispatch and chunk geometry do not transfer), lacks deterministic replay and mechanical gates, is unproven on quantized local roots, explicitly runs model code unsandboxed as the user, installs on macOS/Linux only (our sandbox work is Windows AppContainer/Job Object specific), is TypeScript (ours Python; S5/S6-full hinge on roots conditioned to the paper's `llm_query` API, not `pi`'s), and its `/refine` is model-applied, contradicting I1 | We take instead: their H=(ρ,G,K,M) state model and snapshot/rollback UX as prior art for S6-lite; their documentation honesty as a standard. **Contingency registered in §5** |

---

## 4. The `rlm chat` firewall (normative for the implementation)

Design shape (agreed): **one user message = one ad-hoc episode.** Conversation history lives as a sandbox variable (the pattern the upstream reference library ships as `persistent=True` sessions [R]); only the new message enters the message array. No changes to `_turn_loop`, no mid-episode injection, no schema migration. Roughly 200–250 lines: `src/rlm/chat.py` new, a subparser in `src/rlm/cli.py`, plus passing `benchmark_version=None` for chat/ad-hoc episodes (today `cmd_run` stamps live `"v1"` onto ad-hoc runs — cli.py:516, [V]) — a one-line correction that benefits `run` too.

Hard rules — violating any is a bug regardless of usefulness:

1. **Observation surface only.** Chat proposes hypotheses; it never accepts changes. Insights become candidates for pre-registered gates, never gates themselves.
2. **Every message is an episode.** Full trajectory into DuckDB, replayable, `benchmark_version=NULL`. A chat reply is not scored and must never be citable as a benchmark result.
3. **No config reach.** Chat code may not write config, prompts, budgets, or any file the scaffold reads. It reads config like any other client.
4. **Budgets apply per message** (wall clock, subcalls, tokens, window kill). Idle waiting for the human costs nothing because nothing accumulates between messages.
5. **Scope bound:** ship inside roughly a week including mock-dispatcher tests (the `fixtures` key already supports GPU-less end-to-end tests). If it grows past that, it stops being this decision and becomes a new one.
6. Known limitation accepted for v0: cross-message coherence comes only from the history variable read under the truncation cap. Rich multi-turn memory is future work and out of scope here.

---

## 5. Contingency ladder (pre-registered so it is not chosen under pressure)

| Trigger (measured, not felt) | Response |
|---|---|
| v2 runs and `rlm` delegates well on train; margin holds on controls | Proceed to trajectory accumulation toward S6 preconditions. No change |
| v2 shows root collapse on delegation tasks (the F2 scenario) | Activate the S5 candidate track: merge the official mit-oasys RLM-post-trained LoRA offline, convert/quantize, re-price KV per the swap checklist, rerun v2. This was already the recorded answer to R1's root cause; F2 raises its priority, not its validity |
| Collapse persists even with the post-trained root | **The thesis is falsified on this hardware.** Then the rational exits are, in order: (1) publish v2 and the negative result — the measurement discipline is itself the deliverable; (2) contribute local-harness learnings upstream (prime-agent or the MIT library) rather than sinking further cost; adoption/fork stops being heresy and becomes the recorded plan. Re-litigating D-C5 requires exactly this trigger |

---

## 6. Explicitly not decided here

- Whether the library wedge (DIRECTION.md D4) publishes before or after the agent has a result — still the owner's call, untouched by this session.
- Benchmark v2's remaining design questions — governed by `2026-08-25-benchmark-v2-design.md`, unchanged.
- The external product name (D-C4 fixes only the *when*, not the *what*).
- Anything about S6-full weights training — scheduled slice, gated, untouched.

## 7. References (external, all [R])

prime-agent: github.com/PrimeIntellect-ai/prime-agent · arXiv:2608.23552 · arXiv:2605.09998. Reference library: github.com/alexzhang13/rlm · docs (budget primitives, persistent sessions). Third-party TUIs: github.com/viplismism/rlm-cli · github.com/Trampoline-AI/fractal. Optimizers/gates: GEPA arXiv:2507.19457 · ACE arXiv:2510.04618 · GRASP arXiv:2605.29668 · SKILLGEN arXiv:2605.10999 · Meta-Harness arXiv:2603.28052. Collapse evidence: minRLM (avilum.github.io/minrlm) · arXiv:2603.02615 · alexzhang13.github.io/blog/2026/longcot-rlm. Adjacent systems: context-labs/halo · NVIDIA AVO arXiv:2603.24517 · Letta (letta.com). Hardware context: AMD developer playbooks (Lemonade, GAIA) · community Strix-Halo perf repos. RLM base paper: arXiv:2512.24601.
