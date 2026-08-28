# S6-lite v0 — building the gate, and what it measured

**Date:** 2026-08-28 · **Status:** COMPLETE for steps 1–4. Three decisions run (148 episodes); all three REJECT. §5 is the finding and it stops the artifact-iteration line.
**Spec:** `docs/superpowers/specs/2026-08-27-s6-lite-v0-artifact-gate.md` · **Plan:** `docs/superpowers/plans/2026-08-27-s6-lite-v0-plan.md`
**Consumes:** `docs/research/2026-08-27-prime-agent-spike.md`.
**Evidence discipline:** every threshold was fixed in the spec before its run. Where a rule was changed or a new reading added, the change is dated, its status recorded, and the run it governs stated. Nothing here was re-scored after the fact.

---

## 0. What was built

prime-agent applies whatever the model proposes to its Continual Harness. This slice builds the missing half: a gate that proposes, screens, evaluates on held-out evidence, and records every decision. Four steps landed.

| step | what | how it was verified |
|---|---|---|
| 1 | the extension: veto, filter, budget, notify | four adversarial checks + 22 unit assertions |
| 2 | the frozen split and the blocked runner | reproduces the spike's Phase A numbers |
| 3 | four mechanical screens | 21 fixtures, all from committed spike material |
| 4 | the accept rule, the audit row, three decisions | 148 episodes across pc-01, pc-02, pc-03 |

**I1 is now a mechanism.** In the spike the model was *asked* in its prompt not to self-refine. Now `session_before_refine` returns `{skip:true}` for any round the gate did not initiate: the model called `await refine.run(...)`, received `{'scheduled': True}` and believed it worked, and the harness state did not change [V]. An entry written straight into `harness_state.json` by kernel code never reaches the model's prompt, and the per-kind count lines are rewritten so it is not even told something exists [V]. Breaking the API guard stops the launch with a named error rather than degrading silently.

---

## 1. The measured result: the gate acquired resolving power

Two decisions on the frozen held-out split, 54 episodes each, **the same artifact and the same sha256** — differing only in how the gate delivered it.

| decision | delivery | median ON/OFF token ratio | 95% CI | reading |
|---|---|---|---|---|
| **pc-01** | left where prime-agent puts it: one line in `# Continual Harness State`, capped at 180 chars, under a preamble calling those entries *"routing/context hints"* | 1.015 | **[0.932, 1.060]** | straddles 1.0 — indistinguishable from having no artifact |
| **pc-02** | re-emitted in full by the gate under `# Operating rules (scaffold-enforced, non-negotiable)` | 1.148 | **[1.104, 1.645]** | **entirely above 1.0** — a reliable, measurable effect |

Both REJECT. Q passed in both (no held-out task lost); K failed in both.

Two things follow, and the second is the useful one.

**The delivery channel decides whether an artifact does anything at all.** In pc-01 the artifact reached the model on all 27 ON episodes (`in_window=1`, every one) and changed nothing measurable. The same bytes, delivered as a rule instead of a hint, moved the cost distribution clear of the null. An artifact that is not obeyed is not a weak artifact; it is an undelivered one, and only the second measurement can tell those apart.

**The gate can now tell an artifact's effect from noise.** It has rejected a human's confident artifact twice, for two *different* measured reasons — first because nothing happened, then because something did and it cost 15% more tokens for no quality gain. An instrument that only ever says no is not known to work; one that says no for reasons that change with the input is.

### 1.1 The design change this forced

**The gate is also the delivery mechanism.** Accepted `prompt` artifacts are re-emitted in full under a directive heading rather than left in a block the harness has told the model to treat as a hint. This is the scaffold owning prompt assembly — what rlm-halo's own `root.v3` does with its strategy blocks.

It opens no new writable surface: only what the gate has **accepted** is emitted, the model still cannot write one, and an empty accepted set adds not one character — pinned by a test, because OFF-arm fidelity is what the whole A/B rests on.

---

## 2. Why the artifact failed, from a trace rather than a guess

`agg-07` produced an answer of `749` or `0` where `576` is correct, repeatedly. At the time this was read as task instability at roughly 15–20% of episodes; **§5 shows that reading was wrong — those failures were almost entirely in ON arms, and the baseline agent gets these right 31 times in 32.** The trace below is still worth reading, because it is a precise record of what an artifact does to an agent that was doing fine without one.

**The model followed the artifact's rule exactly.** It built two methods, printed both results, reconciled the buckets against the record count with an assertion, and scanned for malformed records:

```
A: (0, 749, 576)   B: (0, 749, 576)
assert a == b and sum(a) == len(recsA)      # passed
records with !=1 Disposition line: 0
```

And it answered `0`.

Two things went wrong, and neither is laziness:

1. **It varied the wrong axis.** Its two "independent" methods differed in how they *split records* and shared one predicate extractor. That extractor's greedy capture — `from the (.+) to the (.+)` — swallowed the rest of the sentence into the compared value, so the comparison never matched anything. Two methods sharing the deciding step are one method.
2. **It narrated the answer and discarded it.** `576` sat in its residual bucket at 43% of all records. The artifact's own text says a large residual means the predicate is mis-specified. The model wrote the number out loud, in prose, and answered `0`.

`gate/artifacts/control-v2.json` names both axes explicitly and is derived from this trace. **That is artifact iteration informed by a trace — the proposer's job in step 5, done by hand as a dry run — not a fresh independent control**, and no reading of pc-03 may present it as one.

---

## 3. Defects the work found, all of one shape

Every substantive defect this slice uncovered was **something derived from a source of truth going stale, or a predicate that never changed**. None were found by reading code; all were found by running it against real material.

### 3.1 The split was contaminated, and the screens caught it

The first frozen split put `codeqa-06` (`rlm/dispatcher.py`) and `codeqa-07` (`rlm/budget.py`) on the held-out side while `codeqa-01`, `codeqa-03` and `codeqa-04` — which answer the *same strings* — sat in train. A model that memorised a train answer would have scored two of the three held-out code-QA tasks without reading anything. That is exactly the contamination this gate exists to prevent, and the split walked into it.

Answer disjointness is now an invariant: `gate/make_split.py` refuses to write a split where any answer appears on both sides.

### 3.2 Provenance is relative to a split

Two memories the local root wrote during the spike were **legal under the spike's own split and contaminating under v0's** — one names `rlm/budget.py` (codeqa-04), one names `589` (agg-06), both moved to held-out by the redraw. Neither entry changed; the split did. That is why a split is frozen and sha-pinned before artifacts exist, and why re-drawing one invalidates everything derived under the old.

### 3.3 Three staleness bugs in one afternoon (spec R7)

- the crossed answers above;
- the screens' fixture suite asserting a task was held-out after the redraw moved it to train;
- a `heldout.txt` staged before the redraw still naming `codeqa-06` — and a decision **started** on it. It would have evaluated on a training task with nothing to say so.

All three are now closed in code, and `run_decision.sh` refuses to run when its task list does not equal the split's held-out side. A fourth mechanism is owed and not built: no episode asserts the split's sha, so an edit mid-decision is still caught only afterwards.

### 3.4 Two predicates that never changed

- **The stdin bug.** `run_episode.sh` inherited the task loop's stdin and swallowed the list, so the first run executed one task three times instead of three tasks. In a decision that would have silently run only the first task of 54 episodes.
- **The settle poll.** It grepped for `running|executing|active` and prime-agent's *idle* output is the sentence `No active agents.` — which contains "active". Every wait timed out: 42 s an episode, 38 minutes a decision, proving nothing. Matching the idle string took it to 0–1 s, and pc-02 ran 54 episodes with **zero** voids where pc-01 had one.

### 3.5 R15 reproduced inside prime-agent

One `agg-07` episode emitted **29 byte-identical cells over 51 turns**, burned 88.7K tokens and 1,384 s and produced no answer. prime-agent ships no identical-turn guard (its own issue #1326 reports 20–50 identical calls on a local Qwen); the scaffold's C5 has `max_identical_turns: 3` and I1 makes budgets the scaffold's to own, so the gate restored it. This narrows the spike's "A-loop: not reproduced" — true of those 24 runs, not a general claim.

---

## 4. The instrument's own limitation, stated plainly

**v1 has almost no dynamic range for this question.** Across pc-01 and pc-02, seven of the nine held-out tasks pass 3/3 in **both arms, every time**. They carry no quality signal, and their cost ratios cluster at 1.0 and drag the median toward "no effect" whatever an artifact does elsewhere.

The only headroom is `agg-06` and `agg-07` — and that is exactly where the §4.2 design has the least support, at 2 tasks × 3 reps.

**The design error was using one shape of experiment for two questions.** A *cost* gate wants breadth: many tasks, few reps, median over tasks. A *reliability* gate wants depth: the tasks that fail, many reps, a failure-rate comparison. §4.1's blocked A/B is right for both; the sampling is not.

Hence spec §4.2b and `gate/reliability.py`: discordant ON/OFF pairs scored by an exact one-sided McNemar test, run on a **declared** subset of the held-out side (`run_decision.sh` requires `RLMH_SUBSET_REASON` and records it). Its threshold and its power limit were both fixed before pc-03 ran: **20 pairs can confirm a large effect and cannot rule out a moderate one**, so a REJECT there means "not demonstrated at this n", never "no effect".

### 4.1 A gap in the accept rule, raised before the verdict that exposed it

§4.2 as written can only ever accept an artifact that makes the agent **cheaper**: Q is non-inferiority, not improvement, and K requires cost to fall. An artifact that makes the agent more *reliable* is invisible to Q and penalised by K — so the rule cannot accept the behaviour E3 identifies as valuable.

The spike's own reading had two paths ("pass where Phase A failed, **or** wall ≤0.8×"); v0 kept only the cost path, which was a drafting error rather than a decision. §4.2a proposes restoring the quality path. It is recorded as **proposed, not adopted**, was written down before pc-01's verdict was known, and would not have changed that verdict — no held-out task failed in the OFF arm, so Q⁺ had nothing to fire on.

---

## 5. The result: three careful artifacts, all three made the agent worse

Decision **pc-03** ran artifact v2 — the one derived from pc-02's trace — on `agg-06` and `agg-07` at 10 reps per arm, scored by §4.2b's pre-registered reliability rule.

**REJECT.** 20 pairs, 0 voids. OFF failed 1/20, ON failed 2/20; discordant pairs `b=2, c=1`, exact one-sided **p = 0.875**. Cost, not gating: ON 8,034 tokens against OFF 7,136.

That verdict is not the finding. This is:

| every episode ever run on `agg-06` + `agg-07`, all three decisions | correct | failure rate |
|---|---|---|
| **OFF** (no artifact) | 31/32 | **3.1%** |
| **ON** (an artifact loaded) | 27/32 | **15.6%** |

| all nine held-out tasks, pc-01 + pc-02 | correct | failure rate |
|---|---|---|
| **OFF** | **54/54** | **0.0%** |
| **ON** | 51/54 | **5.6%** |

**The instability I built v2 to fix did not exist. My artifacts were causing it.**

Every failure I diagnosed across this slice — the `749`, the `0`, the `1` — happened in an **ON** arm. The one OFF failure in 32 episodes on the aggregation tasks was `577`, off by one from `576`: a boundary slip, not the predicate collapse I designed v2 against. The baseline agent, given no artifact at all, answers these correctly essentially always.

**A self-correction, stated plainly.** §4.2b's premise — "agg-06 and agg-07 fail intermittently at roughly 15–20%" — was wrong, and wrong in a way that matters. It came from counting failures without separating the arms, over about six observations. With 32 clean OFF episodes the rate is 3.1%. So the reliability design had even less power than §4.2b claimed, and worse, it was built to detect an improvement in a quantity that had no room to improve.

### 5.1 Why this is the strongest argument the gate has

Three artifacts were written across this slice. Each was derived from measured evidence rather than intuition — v1 from the spike's Phase C, where the same instruction produced two cross-checked counters on the first try; v2 from pc-02's own failing trace, naming the exact axis the model had failed to vary. **Each passed all four mechanical screens.** I was confident about all three.

**All three made the agent worse.** Not neutral — measurably worse, 0.0% → 5.6% failure across nine tasks and 3.1% → 15.6% on the two hardest. Without a held-out A/B, all three would have shipped, and each would have looked like an improvement to anyone reading the artifact text.

That is the case for this project, and it is now measured rather than argued: **a competent operator, writing careful evidence-derived instructions and screening them mechanically, degraded a working agent three times in a row.** prime-agent's Continual Harness would have applied all three without a word. The gate rejected all three, each time for a reason that changed with the input.

**A hypothesis for the mechanism, offered as a hypothesis.** All three artifacts ask the agent to do *more* — a second method, a reconciliation, a residual inspection. More work is more places to write a wrong predicate, and the baseline approach on these tasks is short and usually right. An instruction that adds steps to a process that already works is a net negative even when every step is individually sound. This is testable — an artifact that asks for *less* would decide it — and it is not tested here.

### 5.2 What this says about v1

**v1 cannot support this gate, and no artifact iteration fixes that.** There is no headroom: 54/54 in the OFF arm across nine tasks, 31/32 on the two that looked unstable. Quality cannot improve because it is already at ceiling; cost differences run 5–15% in the wrong direction. An instrument with no dynamic range cannot show an improvement, only a degradation — which is exactly what it showed, three times.

This lands on the conclusion the repo already held on other grounds. `ARCHITECTURE.md:52`, I5: *"benchmark v2 a **precondition** of S6, not an enhancement to it"*, because *"all 30 tasks are solvable without delegating, and S4's RLM arm won 30/30"*. That was argued from delegation; this is the same conclusion reached empirically from saturation. **The gate is built, tested and discriminating, and the next thing it needs is not a better artifact — it is benchmark v2.**

---

## 6. What is not settled

- **The gate has still never accepted anything**, and on v1 it now looks unlikely to: §5.2 shows there is nothing to improve. "The gate works" means "the gate discriminates" — demonstrated three times, on inputs that a screen pass and an author's confidence both cleared. The accept path stays unexercised until an instrument with headroom exists.
- **§5.1's mechanism is a hypothesis, not a result.** That all three artifacts asked the agent to do *more* is true; that this is *why* they hurt is untested. An artifact asking for less would decide it.
- **The proposer does not exist.** Step 5 has not begun; v2 was hand-written from a trace as a dry run of what the proposer must do.
- **§4.2a and §4.2b are proposed, not adopted.** Both change what a decision means and both are the owner's call.
- **Artifacts are not sha-pinned into `config_snapshot`** (D-S5). Until they are, an rlm-halo episode's snapshot does not record which artifacts were live, and v0's audit lives only in the gate's own ledger.
