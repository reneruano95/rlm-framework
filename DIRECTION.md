# DIRECTION.md — Product Direction (decision record)

**Decided:** 2026-08-16
**Status:** Standing decision. Re-litigate only with new facts (same rule as ARCHITECTURE.md §11 decision records).
**Relationship to the spec:** This document governs *where the project is going*; `ARCHITECTURE.md` governs *what is being built and how it is judged*. Nothing here amends an invariant, a gate, or a sizing. The spec's §1 exclusions ("not an agent framework, a chat product, or a general orchestrator") still hold for the runtime the spec governs — the appliance below is a *deployment* of that runtime, not a new scope for it.

---

## 1. The decision

**Open-core, both wedges, sequenced. Target buyer: private organizations (on-prem, data never leaves the building).**

1. **Entry — the orchestration layer as a standalone open library.** The C4/C5 surface (dispatcher slot discipline, admission control, pre-flight, prefix pinning, `predicted_reuse`, the R13 foreign-string detector, budgets) is extracted and published once it survives S3 hardening. **The library is the appliance's internals published as-is — never a second product.** It exists because the appliance needed it; it contains only what the appliance needed first.
2. **Destination — an on-prem appliance for private orgs** on Strix Halo-class unified-memory hardware: a private, flat-cost, long-context engine over corpora that cannot leave the building. The frozen benchmark (`bench/`, v1) is its evidence base; an S4 verdict is the substantiation behind any quality claim made to a buyer.
3. **Explicitly not chosen as a first move: a vertical app** (single-domain document QA). Admissible later as a monetization layer on the appliance, or as something third parties build on the runtime. Building it first would shift all effort to domain ingestion, domain evaluation, and distribution while freezing the infrastructure this project exists to get right.

## 2. The three wedges considered

- **Appliance** (full vertical stack, end user installs and gets a private assistant): slowest to first user, every bug at every layer is ours — but differentiation *compounds*, because the moat is stack coherence (tuned serving split + contention data + scaffold control + benchmark evidence), which cannot be copied one component at a time.
- **Library** (the co-residency/orchestration machinery standalone): shippable soonest, into documented demand — but it is the least defensible piece precisely because it is simple; upstream (llama.cpp router mode, Lemonade) could absorb the idea. It earns adoption and trust, not a moat.
- **Vertical app**: best revenue-per-user story, but the work becomes domain execution and sales; the infrastructure advantage is invisible to the buyer.

The wedges pull apart only in *marginal-effort allocation* (the library rewards generality, the appliance rewards opinionation, the app rewards neither). They are not mutually exclusive; the open-core pattern reconciles the first two by making them the same code.

## 3. Demand evidence (verified, 2026-08 prior-art sweep — ARCHITECTURE.md §13)

- Lemonade issue **#1547**: open, active request for exactly this layer ("subagent router mode").
- Lemonade issue **#1804**: a user OOMing a 128 GB Strix Halo box because nothing does admission control.
- **llama-swap** (~5.3k stars) doing something adjacent but simpler — the adoption channel exists.
- LucRoot's 7-model gfx1151 fleet (serving-only tooling); Apple WWDC26 session 232 (orchestrator + subagents on unified memory) — platform vendors see the same shape.
- The sweep's three unoccupied gaps (serving co-design on unified memory; scaffold-owned control; on-box gfx1151 contention data) are the differentiation this direction monetizes.

## 4. Discipline rules (what keeps "both" from becoming "neither")

- **D1 — The library only ever contains what the appliance needed first.** A feature request that does not serve the appliance is a PR someone else writes, or it does not happen.
- **D2 — Hard support statement in the library README:** supported = Strix Halo-class unified-memory boxes with llama.cpp; everything else = PRs welcome, no promises.
- **D3 — 0.x API, "API follows the appliance."** No stability promise before the appliance itself stabilizes. Public surface stays minimal: admission/scheduling interface, nothing else.
- **D4 — Publication gate:** not before the dispatcher/admission layer survives the S3 gate (budget kills, hard-kill durability, replay-from-store-alone). Publishing an unhardened admission layer under this project's name spends the credibility the appliance needs.
- **D5 — The library ships with its measurement record.** R13 and R14 are documented in the library's own docs, bounds and all. The measurement discipline *is* the differentiator; a library that hides known defects of the stack it schedules would spend the exact trust it exists to build. Never write "leak-free" (existing R13 rule, now with a customer-facing reason).
- **D6 — C4/C5 boundary stays cleanly separable.** It is today; keep it so. Extraction is a packaging exercise, not a refactor.

## 4a. S4 result and what it does to this direction (2026-08-18)

The S4 gate PASSED (RLM 30/30; +30/+13/+29 vs B1/B2/B3, all p ≤ 0.0002) — **and the RLM arm never called the leaf** (`milestones/s4/RESULTS.md`, ARCHITECTURE.md §9 S4). Three consequences bear directly on this document:

- **The evidence base for a product claim is now real, and cheaper than forecast.** On the frozen benchmark the scaffold answered at 0.78×/0.10×/0.43× the wall-clock and 0.19×/0.06×/0.13× the tokens of the three baselines. "Private, on-prem, *and* cheaper per answer than the obvious alternatives" is a stronger pitch than the quality-at-a-cost story §2 predicted.
- **Differentiator #1 is not load-bearing for this result.** §3 names serving co-design on unified memory as the first unoccupied gap, and S0's two-resident contention data as its proof. A single resident root plus a REPL would have produced the same 30/30. The co-design work is not invalidated (the baselines needed the leaf; delegation is unmeasured), but **the appliance's minimum viable shape may be one model, not two** — smaller, cheaper, and easier to support. Do not re-plan around this until the delegation measurement lands; do not sell the two-resident topology as the differentiator until it earns a scored win.
- **The library wedge is unaffected but re-aimed.** D1–D6 stand. The admission/scheduling layer's value proposition rests on multi-model workloads this benchmark did not produce, so the honest packaging is "the co-residency layer for people running two or more models on one box", not "the layer that made our benchmark win".

**Nothing in §1–§4 is amended by this.** Open-core, library-after-S3, appliance-for-private-orgs all stand; what changes is which technical claim carries the marketing weight, and that stays open until delegation is priced.

## 5. Consequences for the engineering plan

- **R13 upstream fix is promoted to product-blocking.** In a multi-user org deployment, cross-request slot leakage is a *privacy defect* — one user's document bleeding into another query's answer — not only a benchmark-validity threat. The interim mitigation (never-reuse-a-slot + deterministic detector, 95% upper bound 2.2%) remains the shipping posture and its bound is a number a customer must be shown. The upstream reproducer/issue (`milestones/s2/R13-upstream-draft.md`) moves from "worth filing" to scheduled work.
- **R14 gains commercial weight.** Serial-pinned dispatch means the appliance's throughput story is serial until upstream continuous batching is fixed; the `--no-cont-batching` parity lead at concurrency 2 is the path back to fan-out economics. The reproducer and upstream issue matter commercially, not just scientifically.
- **The trust stack is a sales asset, not overhead:** auditable open orchestration layer + replayable traces (I4) + hard budgets (I1) + published measurement record answer the on-prem buyer's first question — "how do I know what this thing does with my data?" — in a way a closed vendor cannot.
- **Revenue model (working assumption, not measured):** support and deployment, not licenses. Orgs at this trust level pay for accountability.
- **The S-ladder is unchanged.** No new slices, no gate changes. S3 is next regardless of this decision.

## 6. What the org deployment adds later (recorded now, built later)

Multi-user access and some access-control story; an install/update path an IT admin can run; a support relationship. None of this enters the current milestones; it is recorded so the appliance's definition of done is not mistaken for the runtime's.
