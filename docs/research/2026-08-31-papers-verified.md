# Four harness papers, verified — what they give this project, and what they take back

**Date:** 2026-08-31 · **Status:** COMPLETE. Recuris verified against raw arXiv LaTeXML and against its source repository. The other three verified to abstract/table level only.
**Consumes:** `docs/research/2026-08-31-papers.md` (four X/Twitter posts, no numbers of their own).
**Bears on:** `docs/research/2026-08-28-s6-lite-v0-results.md` §5, `docs/superpowers/specs/2026-08-25-positioning-chat-and-build-order.md` (F2), and the intraday-trading environment being designed.

**Evidence discipline.** Two passes were run. The first read the papers through WebFetch's summarizing model and produced numbers that were real but stripped of the row, baseline or caveat that reversed their meaning — every strip in the flattering direction. This document is the second pass. A claim is marked [V] only where it was read in raw fetched text or source code; [S] marks anything still resting on a summarizer or a secondary source; NOT-FOUND means no route that reached raw text located a measurement. Two claims from the first pass are **inverted** here and are marked as such.

**Provenance limit, stated once and applying to everything below.** Five verification routes were run against Recuris. All five resolved to **one document**: four saved copies of the arXiv HTML share md5 `24b0ec3cc03d0249c81aa0558788c7d9`, 556,363 bytes, `arXiv:2608.24876v1 [cs.AI] 25 Aug 2026`, LaTeXML output. v1, no v2, not peer reviewed. A sixth check searched for third-party corroboration and found none — no blog, no review, no citing paper quotes these tables. **"5/5 routes" means five readings of one document.** Every CONFIRMED below means *the paper says this*. It never means *this is true*.

The HTML also silently drops six figures (1, 4, 5, 6, 7, 9 carry no `<img>` element at all); the PDF at `arxiv.org/pdf/2608.24876v1` was reachable and carries their text layer. Two claims recorded as unsupported in the first reconciliation are recovered from it in §3.

---

## 1. The headline: the artifact class we kept gating is not load-bearing

Three legs, all from Table 2/3, τ²-Retail, **456 episodes = 114 tasks × k=4** [V].

| component | effect | reading |
|---|---|---|
| Experiential Memory alone — prompt/skill **text** | **+2.0 [−4.0, +7.9]**, no dagger | indistinguishable from nothing |
| the same skills added **on top of** a working state | **+1.5 [−2.4, +5.7]** | the paper refuses an additive decomposition, and this is why |
| Working Memory alone — the ledger | **+23.9† [+17.5, +30.3]** | of a combined **+25.4†** |
| the harness/plumbing alone, 4 models (Appendix B) | +5.23, −0.28, −0.58, +0.87 | every interval contains zero |

Appendix B's caption, verbatim [V]: *"The harness on its own contributes nothing measurable… Every interval contains zero."* Note the label trap: the first pass called Table 2 "the harness ablation." It is not. Anyone chasing "the harness ablation" lands on Appendix B — the opposite finding.

§3.3.4, verbatim [V]: *"the skills do not carry capability on their own."*

**The separation is entirely on the write path** [V], Table 3 rows:

```
                                   Base | WM only | Model-controlled | Recuris
Task success (%)                   58.1 |  82.0   |      65.6        |  83.6
Required-write recall (%)          55.7 |  80.9   |      61.1        |  82.4
Omitted required writes/episode   0.596 |  0.145  |     0.417        |  0.121
Agent tokens per success (k)        116 |   102   |       147        |   101
```

So: **not the scaffold, and not the text. The mechanism inside the ledger.** This is the finding that bears on `2026-08-28-s6-lite-v0-results.md` §5 — three careful artifacts, all measuring worse. They were the shape of column E. Independent work at ~3× our episode count puts that column's effect inside its own noise.

### What this does NOT license

Three corrections to the first pass, all against the flattering reading:

- **A fifth row exists that all five routes missed** [V]: `Model-controlled invocation | 65.6 (299/456) | +7.5† [+1.5, +13.4]`. Daggered. Model-controlled invocation is **significantly better than base** — it is only worse than the ledger. The clean sentence is the paper's: *"Putting the whole library in context and delegating invocation to the model scores below the identical configuration carrying no skills at all, 82.0."*
- **The "second pair" 58.1 vs 82.0 is not a same-library pair** [V]. 58.1 is Base with no memory at all; 82.0 is WM-only with the skills removed. Only 65.6/83.6 shares a byte-identical library, per the caption.
- **The library-in-prompt cost is +2,642 tokens against Recuris, not +3,111** [V]. Table 12 first-call prompt: bare 5163, WM-only 5636, model-controlled 8274, Recuris 5632. 8274 − 5632 = 2,642; 8274 − 5163 = 3,111. The paper says "3,111 more tokens than Recuris" twice, in §3.2 and the Table 12 caption. **The error is the paper's own.** Cite +2,642 vs Recuris, +3,111 vs the bare agent.
- The paper contradicts itself inside one paragraph on **8 vs 10 skills** [V]. Appendix D and the repo both say ten: 8 `action_result` cards + 2 procedure cards.

---

## 2. The noise floor, and the discipline of applying it evenly

Appendix A, verbatim [V]:

> Re-running one τ²-Retail package unchanged, **three days apart on the same tasks, moves task success by +0.00 points with an interval of [−6.98, +7.27]**, so on that domain we treat differences of a few points as within run-to-run variation **regardless of their interval**.

Consequences the paper accepts and the first pass did not:

- **Only 12 of 37 Table 1 cells carry a dagger** [V] — counted programmatically from the raw table block, and independently by all five routes. So 25 of 37 "improvements" have intervals containing zero. The abstract's "improves 35 of 37" counts **sign**, not significance.
- Exactly four cells are +0.3 or negative [V]: Qwen3.5-4B Retail +0.3, Qwen3.6-35B Retail +0.3, Granite-4.1-3B SkillFlow −0.3, Gemini 3.7 Flash Airline −1.5.
- τ²-Retail is not τ²-Bench. On τ²-**Airline** the same package gives Opus 5 **+1.00 [−3.00, +4.00]** and Gemini 3.7 Flash **−1.50 [−5.50, +2.00]** [V].

**And the rule has to cut both ways.** Applied evenly, EM's +2.0, the +1.5 increment and model-controlled's +7.5† all sit at or inside the ±7 band. The honest reading of §1 is therefore narrower than "the text is a measured null":

> **WM's +23.9 is the only effect in the ablation clearly above the paper's own floor. EM's +2.0 is not evidence of a null; it is an absence of evidence at this instrument's resolution.**

Do not quote +2.0 as a measured zero. Our own three REJECTs remain the stronger local evidence, because they were measured on our own instrument with its own null floor characterised (median 0.9670, CI [0.9212, 1.0670] — see §9).

---

## 3. Two claims recovered from the PDF

The first reconciliation marked both "must not be cited." That was wrong; both are plotted, measured values in figures the HTML dropped [V, from `arxiv.org/pdf/2608.24876v1`].

- **+32.2** is the `>31 median turns` bin of a **six-bin** horizon split: `+12.5 +20.0 +27.9 +28.7 +25.7 +32.2`. The series is **non-monotone**, which corroborates §3.3.1's four-quartile analysis (+17.0 to +44.7, "no monotone decline") **against** the abstract's "the advantage widens as the horizon grows." Cite the series, not the abstract's gloss.
- **The failure-mode reduction** is a six-bar panel — `86% 62% 44% 24% 20%` and `80%` — annotated **"Cut by 20–86%"**. The abstract's "up to 80%" understates its own figure. No CIs on any bar, so cite without an interval.

---

## 4. The retry inversion — the single most transferable correction [V]

Terminal-Bench 2.1, Table 8. The caption is the whole story: **"Δ and p compare each row with the row above it on the same tasks."**

| | budget | solved | Δ vs row above | p |
|---|---|---|---|---|
| Terminus-2 (baseline) | 1 | 30/87 (34.5) | — | — |
| + seed memory | 1 | 28/87 (**32.2**) | **−2.3** | 0.824 |
| + seed memory, retry | 4 | 51/87 (58.6) | **+26.4** | <10⁻⁴ |
| + test-time adaptation | 4 | 53/87 (60.9) | +2.3 | 0.774 |

Four corrections, each load-bearing:

1. **The claim's chain deletes the row that goes down.** The memory layer alone moved 34.5 → 32.2.
2. **+26.4 is measured against 32.2, not 34.5.** Baseline → retry is +24.1, a number the paper never prints. A coincidence invites the error: 60.9 − 34.5 also equals 26.4.
3. **There is no retry-only arm.** The 58.6 row carries the seed memory. Retrying without it was never run.
4. **The +2.3 for adaptation is not significant** (p = 0.774; 7 tasks won, 5 lost).

The paper's own verdict, verbatim: *"That headline is **not a learning effect, and the table says so**… the attempt budget therefore explains the headline and both memory terms sit within run-to-run variation."* It then disowns its own metric: solved-within-budget is *"the wrong instrument for a learning effect… two runs of the identical seed memory differ by 5.7 points with nothing changed"* — larger than the +2.3 it would have to detect. Every per-attempt fallback contains zero: avg@4 learned +4.5 [−0.9, +9.8]; avg@4 all 87 +2.9 [−0.6, +6.3]; pass@4 +2.3 [−5.7, +10.3].

Two genuine positives survive: seven tasks (`compile-compcert`, `qemu-alpine-ssh`) solved **only** under adaptation, and one deep probe at 16 rollouts — bare 8, seed memory 7, adapted memory **15, p = 0.006**. And the context that kills any transplant story: *"cross-task evolution on Terminal-Bench admitted no patch in thirteen runs of the evolution loop."*

**Obligation on us, and it is the largest single transferable lesson in the set:**

> **Every gated comparison carries a matched-budget control.** If a change adds attempts, replays, symbols, candidate rules, wall-clock or context, the winning arm may have won on budget. Budget is a controlled variable, pre-registered, held equal across arms.
>
> **Report the adjacent-row delta.** Each accepted change is measured against the immediately preceding accepted state, never against the original baseline.

---

## 5. F2 — inverted in the first pass, and the real line sits below ~20B

The first pass described F2 as "the load-bearing claim that a scaffold rescues weak local models." **F2 is a damage finding, not a claim.** Verbatim at `docs/superpowers/specs/2026-08-25-positioning-chat-and-build-order.md:42` [V]:

> **F2** — Quantized small-code roots sit in the regime where RLM scaffolds collapse… Our root is a Q4_K_M 27B. The thesis is therefore *unproven for our configuration* — which is precisely what benchmark v2 exists to measure.

### The three-paper "contradiction" does not survive

It dissolves into different baselines. All three measure a **recovery** effect against three differently-broken references.

- **Recuris does not claim what was attributed to it.** Table 1's caption [V]: *"its value does not follow model scale."* §3.4.6: *"It is not a crutch for weaker models"* — an absolute-ceiling argument (87.9 > 81.4), never a delta-monotonicity claim. The largest gain in the table is **+23.3 on Doubao-2.0-Pro**, which §3.1 calls *"deliberately mid-sized rather than frontier."* No Terminal-Bench cell is significant for any model: +2.5 to +3.8 from Granite-3B to Opus 5, mean +3.03, sd 0.39 — a flat, model-independent offset.
- **AutoDesign's "monotone in weakness" partitions on a different axis** [S]. Native agent (GPT-5.5/Codex, Claude 4.8/Claude Code): +5.59, +5.01. Foreign agent (five non-Anthropic models inside Claude Code): +11.87 … +19.56. **Perfect separation, no overlap.** The apparent capability gradient is an **agent-fit** gradient. It does not equalise: post-harness spread is still 27.17 points, and DeepSeek with the best harness (54.29) stays below Claude 4.8's unimproved baseline (69.55).
- **Prime Agent's table caption disclaims the premise** [S]: *"Bold is not statistical significance, and uncertainty intervals are unavailable."* So Opus .900 vs Claude Code .920 is not a measured loss. GLM's headline advantage lives in three rows where the comparison harness collapses (.420, .556, and **.000**); drop those and GLM's remaining six median +.030 against Opus's +.007. *(The GLM-5.2 pairing and its ~744B-total/~40B-active MoE size come from vendor and blog sources, not the arXiv text, which never defines "Pi-mono." [S])*
- **The observation that closes it** [V]: Recuris' τ²-Retail *baseline* column is not a capability ordering. `Qwen3.5-9B 77.6 > Gemini 3.7 Flash 73.5 > Claude Opus 5 72.4`; `Qwen3.5-4B 68.0 > GPT-5.6 Sol 58.3`. A 4B model out-baselines GPT-5.6 Sol by ~10 points. That is a reference tool-calling agent mis-fitting frontier reasoning models. And the gains land exactly on the deficit: **Spearman(baseline, Δ) = −0.65**.

### The real residual — F2 measured

SkillFlow is the only procedural-by-design benchmark in the set, and there the pattern is a **floor, not a gradient** [V]:

| model | SkillFlow Δ |
|---|---|
| Granite-4.1-3B | **−0.3** |
| Qwen3.5-4B | +1.1 |
| GPT-OSS-20B | +2.6† |
| Qwen3.5-9B | +3.4 |
| **Qwen3.6-27B** | **+16.6†** |
| **Qwen3.6-35B** | **+13.5†** |
| Doubao-2.0-Pro | +16.8† |

Pearson(baseline, Δ) = **+0.979**, with a discontinuity between ~20B and ~27B. Below it the artifact buys +1 to +3; at and above it, +13 to +17. **The smallest model loses.**

**Conclusions for this box.** `config.yaml` pins root `Qwen3.8-27B-Q4_K_M` (line 38) and leaf `Qwen3.6-35B-A3B-UD-Q4_K_M` (lines 188, 274) [V]. Recuris measured Qwen3.6-27B at +16.6† and Qwen3.6-35B at +13.5† — the two largest significant open-weight gains in its table, both above the discontinuity.

1. **F2 as written is not supported for a ~27B-class root** on the best-controlled table available.
2. **F2 *is* supported as a floor effect somewhere below ~20B.** If the appliance ships a 4B–9B root, or if S6-full distils into "a 4B–8B model" as `ARCHITECTURE.md` contemplates, that is the configuration where the published evidence says the scaffold stops paying. **Re-register F2 against S6-full, not against the current root.**
3. **The appliance's honest claim is not parity.** AutoDesign compresses but leaves 27 points; Prime Agent's GLM beats Opus on 1 of 9 rows with the harness held constant. The defensible claim is the one already in `DIRECTION.md`: private, on-prem, auditable, and cheaper per answer — quality at a known cost.
4. **One methodological differentiator falls out of this.** Our B1/B2/B3 baselines are **the same model as the RLM arm**. That makes our margins immune to the confound that hollows out AutoDesign and Prime Agent, which report a harness swap and a model swap together and call it a harness gain. This belongs in the measurement record.

---

## 6. The mechanism, written to reimplement [V, from source]

Read from `schema.py`, `ledger.py`, `grounding.py`, `confirmation.py`, `render.py`, `selfmaintain.py`, `checkpoint.py`. The paper gives it one sentence; the code gives the whole thing.

Two corrections to the paper first: the code has **four** states, not three (`OBSOLETE` is real, audit-only), and an **`auth` half-state field the paper never mentions**. And Working Memory is **not a durable on-disk store** — on disk live the schema (`manifest.yaml`) and the skill cards; the ledger is a live per-conversation object whose `snapshot()` is serialized into the structured trace the Meta-Agent later reads.

### The record

```python
LedgerEntry:
  id: str                 # "req-1", "req-2", … monotonic per conversation
  description: str        # one line, MODEL-written
  state: EntryState       # NOT_YET | DONE | BLOCKED | OBSOLETE
  evidence: list[str]     # literal tool call_ids that grounded DONE
  blocked_reason: str
  fields: dict            # domain payload
  auth: dict | None       # {"quote": <verbatim>, "src_user_msg": idx}
```

An ordered list of atomic work items for one conversation, **capped at 8**. `visible()` filters OBSOLETE out; `snapshot()` keeps it. **The engine is domain-free**: retail → airline changes `binding_key: order_id` → `reservation_id` **in data, not code**.

### Who may write what — enforced structurally, not by prompt

| field | writer |
|---|---|
| `description`, `tool`, `params` | MODEL |
| `state.done` | **HARNESS** (from real tool receipts) |
| `state.blocked` | ORACLE |
| `state.obsolete` | MODEL |

`mark_done` raises `WritePermissionError("state.done requires Writer.HARNESS … the model can never mark its own work done")`. The incident in the module header: a model's belief that it had finished used to land in the ledger verbatim, which **disabled the termination gate, the truth protocol and the status board in a single chain.**

The model's only write channel is `apply_model_update()` — a **replace, not a patch**. DONE/BLOCKED entries kept verbatim; previous NOT_YET entries become OBSOLETE; a malformed proposal is **dropped, not stored**; a bad JSON parse **keeps the previous ledger** ("one bad parse must never nuke the account book"). The model re-derives its whole pending set every turn.

### What a verified commit requires

`grounding.py` is *"the **ONLY** path from evidence to ledger state."* Evidence is a frozen `ToolReceipt(call_id, tool, args, error, synthetic, content)`.

1. **Relevance** — only state-changing tools ground anything.
2. **Consume-once, BEFORE admissibility** — otherwise a rejected synthetic receipt re-counts every turn and the rejection counter inflates linearly.
3. **Admissibility**, decided by the kernel and never by package code: reject if `synthetic` (runtime registry) **or** `is_synthetic_content(content)` **or** `error`. The recorded incident: a synthetic ToolMessage with `error=False` landed as a real DONE, bringing back fabrication, early termination and dropped safeguards **while the run still looked healthy.**
4. **Binding** by integer score — tool mismatch or conflicting `binding_key` is −1 hard exclude; +1 tool match; +2 exact binding key; +2 collection overlap; admit only if best ≥ `min_score`. The runtime **refuses** `receipt_binding_match` without an explicit `binding_key`, because a missing key degrades to "first same-tool entry wins" *with harness authority*.
5. **Commit** — `mark_done(entry_id, eid, writer=HARNESS)`; the `call_id` is appended to `entry.evidence`.
6. **NO-FALLBACK** — an executed write matching no pending entry is counted and logged, never fabricated into the ledger.

> A goal flips pending → done **only** when a non-synthetic, non-error tool receipt scores ≥ `min_score` against *that specific* pending entry under a domain-declared binding key, written through the HARNESS principal, with the `call_id` recorded. **The model's assertion that it finished has no reachable code path to DONE.**

### Worth porting alongside

- **Authorization by verbatim quote.** The model proposes `{"quote": …}`; the harness verifies it against genuine user turns. `MIN_QUOTE_LEN = 8` ("bare 'yes' is too ambiguous"). Entries are re-keyed on every rewrite, so the model must re-assert authorization each turn. One loophole left in-source: a sentence authorizing ≥2 entries is logged but not blocked.
- **Bounce, not block.** A rejected draft is rewritten, not errored — and every fabricated ToolMessage id **must** be registered as synthetic.
- **Anti-anchoring in rendering.** Params matching `.*_ids?` render with values; every other param renders **name-only**, because echoing raw values anchored the model to unnormalized input ("New York" vs "NY") and *caused* wrong-arg failures.

### Build order here

1. A **receipt type** emitted by the runner, not the model.
2. An **entry ledger** with the four states and an `evidence` list of call_ids.
3. A **grounding kernel** that is the only path to DONE. Admissibility in the kernel; matchers decide *which* entry, never *whether*.
4. **Writer permissions that raise.** Not a prompt instruction.
5. **An invocation-reach counter per artifact** — see §7.

**Not read**, so turn ordering is inferred from docstrings: `builtin/entrykinds.py`, `board.py`, `fingerprint.py`, and the runtime loop that calls `ground()`.

---

## 7. What Recuris says about its own gate — read this before defending ours

`src/recuris/metaagent/gates.py`, character for character [V]:

```python
accept = (lo > 0) and (n_dn <= reg_cap)
```

Bootstrap **over items, not trials** — *"trials within an item are not independent, and resampling them would report an interval far narrower than the evidence supports."* `reg_cap` defaults to 0: one regressed item rejects. Module header: *"four gates, all of them code… A model proposes; whether the proposal is kept is decided by arithmetic over held-out outcomes."* It also carries three screens the summary omitted: `leakage_check`, `fingerprint_verify` ("False unless the prescribed carrier fired at least once"), and a `Ledger` do-not-repeat record keyed on `(cluster_id, carrier, primitive)`.

**Two things temper it, and both matter to us.**

**There are two gate files with different rules, and the evidence cannot say which produced the paper's results.** `gate.py` — the CLI the README documents — computes significance and **prints it without requiring it**: `accept = repaired and net > 1e-9 and dn <= args.reg_cap`, with `--reg-cap` defaulting to 1. Appendix E's `driver.log` excerpt matches `gate.py`'s printf format exactly; the paper's *stated* rejection reason matches `gates.py`'s `lo > 0`. Cite the rule as **`gates.py`'s library gate**, never as "the gate Recuris used."

**The paper says its own gate does not work as a filter.** §3.4.4 [V]: *"every one of the 18 rejected candidates has a dev interval containing zero, and 17 of the 18 sit inside their own size-matched band."* Appendix E records a **12-task split at k=4 producing an interval more than thirty points wide**, and a rejected candidate that later clears M₀ by **+11.92** with an interval excluding zero. A fourth run that ignored the dev verdict took no harm. Its own summary:

> **"It is not a filter that separates good patches from bad ones; on this evidence budget it cannot be."**

That is a warning aimed squarely at any gate operating on thin evidence, ours included. It does not invalidate our instrument — ours has a characterised null floor and has rejected for two *different* measured reasons — but it is the strongest external statement available on what a gate at this sample size can and cannot claim, and it should be quoted in full whenever ours is described as a filter.

**And the diagnostic that actually earned its keep** [V], §3.4.5: the single package invoked on **zero** held-out tasks was the only one failing to clear M₀ (Reach 0/86, +3.78 [−2.33, +9.88]), diagnosed as *"a broken binding, not a diminishing return."* Invocation reach is what separated a dead memory from a plateau. **Build the counter.**

The stated transfer precondition, verbatim: memory transfers *"where the held-out tasks still contain the kind of failure the memory repairs, and not where they do not."* Two τ²-Airline lineages, evaluated on 25 and 29 unseen tasks, neither separating from zero, both held-out sets already starting at 73.0% and 81.0%.

---

## 8. Fault localization needs a structured trace — Table 4 [V]

Not a method progression. A fault is injected into one component of a working package — a skill (ℰ), the working-memory spec (𝒲), the invocation policy (ρ) or the checkers (𝒞) — and a **fixed judge** names the component at fault from one of three evidence formats. Nine admitted cases per class, two repeats, 54 verdicts per condition.

```
Evidence given to the judge |  ℰ   |  𝒲   |  ρ   | Macro | Macro-F1
Outcome only               |  0.0 | 38.9 |  0.0 | 13.0  | 10.4
Raw trajectory             | 61.1 | 50.0 |  0.0 | 37.0  | 31.2
Structured trace Γ         | 72.2 | 83.3 | 38.9 | 64.8  | 63.4
```

Three caveats the first pass dropped: **outcome-only sits below the 33.3% constant-answer floor**; n = 54 per condition with **no CIs**; and only 3 of the 4 components are in the ground truth (*"removing the checkers changes the pass rate by zero on τ²-Retail"*).

**Why this is ours already.** `src/rlm/trace/schema.sql` distinguishes `observation_view` (what the root SAW, capped) from `observation_full_ref` (what HAPPENED), with causality in `parent_step_idx`/`call_id`. That is the Γ column. Neither Recuris nor Prime Agent makes this distinction. Table 4 is the argument for why it pays — and note that a raw trajectory scores **0.0 on invocation-policy faults**, which is exactly the class our delegation work needs to localize.

---

## 9. What this changes here

**Convergence, stated once.** Eight mechanisms in this repo have independent-research analogues; in five, ours is the stronger version — state-triggered rather than model-chosen delivery (`config.py:762`, `episode.py:128-142`); a CI-based admission gate with a zero-regression cap (`gate/decide.py:51-55`) against AutoDesign's two scalar means and Prime Agent's turn-boundary "gate"; **no LLM judge on the scoring path** (AutoDesign's judge agrees with humans **51.9% at 0–3-point gaps**, and 54 accepted updates were chained through it); context by reference; a least-privilege action interface; and a repetition guard (C5 `max_identical_turns`) that Prime Agent ships none of — its issue #1326 is R15.

**Untouched by all four papers:** they measure artifact **consumption** — a fixed, externally-authored artifact dropped onto a model. **None measures artifact authorship**, which is what the 2026-08-27 spike measured here, and which failed cleanly: 8 of 8 refinements were `kind: memory`, content was answer logs, zero `skill`, zero `prompt`, zero `subagent`. The consumption experiment has already been run here three times (`2026-08-28-s6-lite-v0-results.md:139-140`: OFF 54/54, 0.0% failure; ON 51/54, 5.6%; on the two hardest, 3.1% → 15.6%).

**Owed, and free.** Three findings were computed from data already on disk while reviewing these papers, and two defects were verified against files today:

- The gate's **null floor already ran twice, unread**: `run_episode.sh:36-42` writes `EMPTY` before every OFF episode, so pc-01-OFF and pc-02-OFF are byte-identical replications. Through `decide.py`'s own bootstrap: **median 0.9670, CI [0.9212, 1.0670]** — so `K_CI_UPPER_MAX = 1.00` sits inside the noise band.
- **`K_CI_UPPER_MAX` has never bound a verdict.** `decide.py:221` is a conjunction; pc-01's median 1.0151 misses the *median* conjunct by 0.115 and would reject with the CI test deleted.
- **Compliance is anti-correlated with value**, and it falsifies §5.1's stated mechanism: pc-01 2/27 fired, pc-02 14/27, pc-03 13/20, OFF 0/74 — and pc-01's two damaged cells are ones where the probe did **not** fire. The damage channel is context presence, not instruction execution.
- **[V, verified on disk 2026-08-31]** `gate/audit.jsonl` and `docs/research/2026-08-27-s6-lite-v0/decisions/audit.jsonl` both hold **2 rows for 3 spent decisions**; pc-01 has `decision.json` and `decision.log` but reached neither ledger. Root cause: `run_decision.sh` never calls `decide.py`, and `decide.py:252` makes `--audit` optional with `default=None`. **A decision can be scored and leave no trace, silently.** This is an I4 hole with the shape of a default value.
- **[V]** `max_decisions: 5` is read by nothing — two references in the repo, `bench/splits/s6lite-v0.json:19` that declares it and `gate/make_split.py:165` that writes it. A counter built today would report 2 of 5 spent when 3 are.
- **[V]** `gate/artifacts/negative-control.json` has **zero** references in repo content. The gate's ability to reject a known-bad input has never been exercised in the domain it was built for.

**Open decisions this document does not settle:** whether to backfill pc-01's audit row (and whether a backfilled row is marked as such); whether benchmark v2 proceeds now that a replay environment would supply a better programmatic verifier; and which environment validates the general machinery first.

---

## 10. What remains unverified

- Everything attributed to **AutoDesign, Prime Agent and Code World Model** [S] — read to abstract and table level, not against raw full text or source. Their headline numbers should not be quoted here without the same treatment Recuris received.
- **Which gate produced Recuris' published admissions** (§7). Unresolved by the available evidence.
- **Whether Table 1's GPT-OSS-20B +10.2† is the shipped single-source package or a model-specific rebuild.** README lines 431–433 say *"On GPT-OSS-20B a rebuilt package gained +10.2, while the general-purpose package transferred negatively"*; §3.1 says every model *"receives that same memory unchanged."* **Direct paper-vs-repo conflict, and the sharpest in the set.** A listing of `skill_memories/` would likely settle it and was not run.
- **A same-model, same-benchmark inconsistency inside the paper.** Table 1 gives Doubao-2.0-Pro τ²-Retail `58.1 → 81.4 (+23.3†)`; Tables 2/3 give the same model, same benchmark, same 58.1 baseline → `83.6 (+25.4†)`. 2.2 points apart on the identical intervention, inside the paper's own ±7 floor. Appendix A's "each analysis draws its own evaluation population" is the likely explanation but is never applied to this pair.
- **Recuris v1 only, no peer review, no third-party corroboration of any table.**
