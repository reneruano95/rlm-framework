# The trading replay environment's statistical capacity — what ~416 sessions can and cannot decide

**Date:** 2026-08-31 · **Status:** ANALYSIS COMPLETE, no code written. The environment does not exist yet; this is the constraint it must be built inside.
**Consumes:** `docs/research/2026-08-31-papers-verified.md` §4 (the matched-budget obligation) and §7 (what a gate on thin evidence can claim).
**Bears on:** the intraday-trading RLM environment; and `bench/splits/s6lite-v0.json`'s `decision_budget` idea, which turns out to be the right primitive applied to the wrong dataset size.

**Why this document exists.** The trading direction was designed on the assumption that a replay environment supplies unlimited, objectively-verified episodes. It supplies objectively-verified episodes. It does not supply unlimited ones. The number of *decisions* the data can support is small, fixed, and consumed by use. That number is the binding constraint on the whole design and it was never computed before this.

**Evidence discipline.** Every formula below was implemented from primitives (erf-based Φ, bisected Φ⁻¹) and **verified against the source papers' own published worked examples before any project number was produced**:

| check | computed | published |
|---|---|---|
| maxZ(100) | 2.5306 | 2.530603 |
| DSR worked example (SR 2.5, N=100, T=1250, γ₃=−3, γ₄=10) | 0.9004 | 0.9004 |
| MinBTL, N=45, E[max]=1 | 4.998 yr | "5 years" |
| MinBTL, N=7, E[max]=1 | 1.923 yr | 1.92 yr |
| MinTRL daily (SR 2 vs 1, α=.95) | 2.7312 yr | 2.73 yr |
| MinTRL monthly (γ₃=−0.72, γ₄=5.78) | 4.9913 yr | 4.99 yr |

Claims marked **[assumed]** are inputs nobody measured and must be replaced before anything is frozen. Claims marked **[open]** are unresolved defects in this analysis, listed again in §7.

---

## 0. The dataset, corrected

Local SQL Server, 1-minute bars, **2025-01-02 → 2026-08-31**, 500+ symbols, mostly S&P 500.

Counting NYSE weekdays minus holidays gives **416 sessions** (250 in 2025, 166 Jan–Aug 2026). An earlier pass said 417 and an earlier one than that said ~165 from a mistaken "8 months." One session of disagreement is immaterial to every conclusion below, but **the split must be cut from the actual bar count, not from this arithmetic.** Numbers stated at T = 417 are kept as computed; date blocks use true counts.

**500 symbols add zero degrees of freedom.** Intraday P&L is cross-sectionally correlated within a day, so the independent unit is the **session**. Symbols buy diversification inside σ, not sample size: K_eff = K/(1+(K−1)ρ̄). At K = 500, ρ̄ = 0.05 gives 19.3 independent bets per day; ρ̄ = 0.20 gives 5.0; ρ̄ = 0.35 gives 2.8.

**One P&L observation per session.** Not per minute: 1-minute bars make T ≈ 162,000, √(T−1) ≈ 403, and every Deflated Sharpe rounds to 1.0000 — the instrument stops discriminating. Minute increments within a session are one trade decision decomposed, not new evidence.

---

## 1. The governing constant

**SE(SR_annualized) = √(252/417) = 0.777.**

A true annualized Sharpe of 1.5 has a 95% CI of roughly [−0.03, +3.03] on this dataset, and will fail a one-sided 5% test about half the time.

**Sessions needed to detect an edge at 80% power**, one-sided α = 0.05, using Lo (2002) eq. 9 with the Mertens correction — V = 1 − γ₃·SR + ((γ₄−1)/4)·SR², **γ₄ raw kurtosis, Normal = 3**:

| SR_ann | sessions (Normal) | years | with γ₃=−1, γ₄=8 |
|---|---|---|---|
| **0.5** | 6,235 | 24.7 | 6,439 |
| **1.0** | 1,561 | 6.2 | 1,667 |
| **1.5** | 696 | 2.8 | 769 |
| **2.0** | 393 | 1.6 | 449 |
| 2.5 | 252 | 1.0 | 299 |
| 3.0 | 176 | 0.7 | 217 |

Inverted — **what a block of n sessions can see** (MDE = √(K/(n − K/2))·√252):

| block | 1-sided MDE (SR_ann) | 2-sided |
|---|---|---|
| 417 | **1.940** | 2.188 |
| 311 | 2.249 | 2.542 |
| **250** | **2.512** | 2.835 |
| 146 | 3.297 | 3.718 |
| 84 | 4.388 | 4.970 |

Two corollaries that bite: serial correlation costs sample (ρ₁ = 0.10 → n_eff = 341 of 417; ρ₁ = 0.20 → 278), and annualizing with bare √252 instead of Lo's η(q) **overstates** the annual Sharpe by 10.5% at ρ₁ = 0.10.

**The implementation bug to avoid, stated once:** the SR in these formulas is **per-observation**, i.e. daily. Plugging an annualized SR into a formula whose n counts days inflates the t-statistic by ≈ √252 ≈ 16×. It is the single most common error in public reimplementations of this family, along with passing `scipy.stats.kurtosis`'s *excess* value where raw is required (`fisher=False`).

---

## 2. The decision budget

**Two instruments; use the stricter.**

**MinBTL** (design-time, looser). y = 417/252 = 1.6548, √y = 1.2864; maxZ(N) = (1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e)), γ = 0.5772. Largest N with maxZ(N) ≤ E[max]·√y:

| E[max] you refuse to be fooled by | N_max |
|---|---|
| 1.00 | 5.9 |
| 1.50 | 21.4 |
| 2.00 | 112.8 |
| 2.50 | 871.9 |

Read as its mirror — **what a skill-less search finds by luck on 417 sessions**: N = 6 → 1.011; N = 21 → 1.494; N = 113 → 2.000; N = 872 → 2.500.

> **Any gate whose accept threshold sits below the E[max] row for its own trial count is a noise generator.**

**Deflated Sharpe** (operative, 20–90× stricter, because MinBTL's N_max is only the point of *indifference* while DSR demands the winner clear it by 1.645 SE). Solving DSR = 0.95 for N at T = 417:

| observed SR_ann | N_max (Normal) | N_max (γ₃=−1, γ₄=8) |
|---|---|---|
| 1.0 | **fails at N = 1** | fails at N = 1 |
| 1.5 | 2 | 1 |
| 2.0 | 3 | 3 |
| **2.5** | **10** | **7** |
| 3.0 | 41 | 26 |

Bonferroni cross-check agrees from another direction: M = 10 → required SR_ann 2.18; M = 100 → 2.71; M = 1000 → 3.15.

### The number

> **On 417 sessions, at an accept bar of annualized Sharpe 2.5 on the gate statistic, the data supports 9–10 independent gate decisions. At a bar of 2.0, three. At a bar of 1.0, none — not one.**
>
> On the recommended split (Block A = 250 sessions): **3 at θ = 2.5**, 6 at θ = 3.0. The sealed holdout supports **one** look.

**Monte-Carlo check on the small-N regime**, because the Gumbel form is asymptotic in N and the whole answer sits at N ≈ 6–10. 200k draws: Gumbel **overstates** E[max] by +2.5% at N=6, +2.2% at N=10, +1.7% at N=21, +0.9% at N=100. Direction is conservative; magnitude ≈ one trial. The conclusion holds, and it held unverified until this check — the same omission with the sign reversed would not have been benign.

### The one lever that converts this into a usable number

Raw proposals ≠ independent trials: N = ρ̄ + (1−ρ̄)·M, with ρ̄ the mean off-diagonal correlation of the trials' P&L vectors.

| independent budget N | ρ̄ = 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|
| 3 | M = 11 | 21 | 41 | 201 |
| **10** | M = 46 | **91** | **181** | 901 |

So ~180 raw proposals are affordable at N = 10 **if ρ̄ = 0.95 and ρ̄ is measured from the actual P&L correlation matrix**. Two caveats that must not be waved away:

- With M > T ≈ 417 the correlation matrix is ill-conditioned and ρ̄ is itself overfit. Above ~400 proposals, cluster (ONC) and count clusters.
- **[open] Deflation does not touch the feedback channel.** Every raw proposal that returns a P&L number to the agent transfers information regardless of how correlated it is with its siblings. Deflation fixes the max-of-N statistic; it does not fix the leak. §4(a) is the mitigation.
- **[open] The variance input is inconsistent with the deflation.** The design deliberately maximizes proposal correlation to buy paired power, then deflates N accordingly — while holding √V[{SR_n}] at the *independent-null* 1/√(T−1) = 0.777. The sample variance of SR̂ across highly correlated trials is biased **downward**, which lowers SR*₀ and **inflates DSR**. You cannot assume near-duplicates for the trial count and independence for the dispersion. Resolve before pre-registering: either estimate V[{SR_n}] empirically from the logged trial vectors, or state the assumption and bound its direction.

### The scaling law worth internalising

N_max ≈ exp(y·E[max]²/2) — **exponential in years of data**, only logarithmically-inverse in the threshold. At θ = 2.5: y = 1.65 → ~10²; y = 5 → 6.1×10⁶; y = 10 → 3.7×10¹³.

> **Extending the sample backward buys decisions. Better statistics do not.** If a many-decision self-improvement loop is the product goal, extending the sample is a prerequisite, not an optimization. (An Alpaca subscription exists; see §6 for what to pull and the one thing it probably cannot give.)

---

## 3. The split

### Regime structure — [assumed], and this is the weakest input in the document

| label | assumed window | sessions |
|---|---|---|
| R1 low-vol grind / early-year rotation | 2025-01-02 → 2025-03-31 | 60 |
| R2 **policy/tariff volatility shock** | 2025-04-01 → 2025-05-30 | 42 |
| R3 recovery / trending | 2025-06-02 → 2025-12-31 | 148 |
| R4 2026, character unknown | 2026-01-02 → 2026-08-31 | 166 |

**The effective regime sample size of this dataset is 3–4, not 416.** Nothing in DSR, PBO, MinBTL, CSCV or the bootstrap measures regime diversity; all of them pass cleanly on a rule fitted to one regime.

**Replacement procedure, to run before the split is frozen:** define the partition causally and mechanically — realized-vol terciles of SPY over a trailing 20 sessions measured at the **prior close**, or VIX terciles at prior close — freeze the labels, then check the split against them and record per-regime session counts as a split artifact.

### The split

```
BLOCK A — DEVELOPMENT        2025-01-02 → 2025-12-31     250 sessions
   TRAIN and DEV are NOT a fixed cut inside A. They are the rotating
   CSCV partition: IS = OOS = 120, S = 16 blocks of 15 sessions,
   C(16,8) = 12,870 splits, purge + embargo at every seam.

EMBARGO                      2026-01-02 → 2026-01-30      20 sessions
   Read by nothing. Trained on by nothing. Scored by nothing.

BLOCK B — SEALED HOLDOUT     2026-02-02 → 2026-08-31     146 sessions
   Opened exactly once, against a pre-registered rule.
                                                  total  416 sessions
```

**Why this boundary.** A calendar year is the least gameable and the cheapest to enforce physically. And the arithmetic says Block A's capacity is nearly insensitive to where you cut (DSR budget at θ = 2.5 is 4 at A = 311 vs 3 at A = 250) while the holdout's resolving power is very sensitive. **The marginal session is worth more in B.**

| option | A | embargo | B | A paired MDE | A DSR budget @θ=2.5 | B paired MDE |
|---|---|---|---|---|---|---|
| A ends 2026-03-31 | 311 | Apr-26 (21) | 84 | 2.25 | 4 | 4.39 |
| A ends 2026-01-30 | 270 | Feb-26 (19) | 127 | 2.42 | 3 | 3.55 |
| **A ends 2025-12-31 (recommended)** | **250** | **Jan-26 (20)** | **146** | **2.51** | **3** | **3.30** |

**Enforcement, not convention.** Separate store; the replay harness refuses any date ≥ 2026-01-02 without an `--unseal` flag that writes to an append-only log. **If the seal is policy, it leaks, and there is no statistical repair.**

**Holdout resolving power, computed before sealing.** MinTRL at SR_ann 2.5 = 113 observations ≤ 146 ✓; at SR_ann 2.0 = 175 > 146 ✗. **Block B can resolve a claimed paired SR_ann ≥ ~2.19 and nothing weaker.** Pre-register the claim at or above that, or the holdout is decorative.

**[open] The stress regime sits entirely inside Block A.** Under the assumed partition the holdout contains no stress episode and cannot falsify regime-dependence. An earlier draft asserted "no re-cut of the calendar fixes this"; that is **false and was never examined** — a *non-contiguous* or regime-stratified holdout (matched stress days withheld from both blocks) is the obvious alternative. It costs contiguity, which matters for path-dependent reporting, and it must be decided before the mechanical partition is frozen.

**[open] CSCV block arithmetic does not close.** S = 16 × 15 = 240 ≠ 250, with no drop rule stated. Fix the drop rule in the pre-registration; do not leave it to the implementation.

### Purging and the embargo — and the trap

**Purging is a near no-op here, and that is the danger.** An intraday strategy flat at the close has labels that never span a session boundary, so López de Prado's three overlap conditions fire on nothing across a seam. The book's own `getTrainTimes` takes only the *label* interval and is therefore blind to the real leak in this design: **feature lookback.** A training row three sessions after the test block, whose feature reads a 20-session trailing volatility, literally reads test-block bars, and purging will happily keep it.

**Only the one-sided right-edge embargo catches that.** Size it as h = L_max + τ, with τ = ln(2/√T)/ln(λ), λ = 2^(−1/half-life). At T = 417 the ACF noise band is 2/√417 = 0.09794, giving **τ = 3.352 × half-life in sessions**. *([open] this constant is derived at T = 417 but applied to Block A at 250, where it should be ≈ 2.98 × half-life. Recompute at the block being split.)*

| feature half-life | τ | % of T |
|---|---|---|
| 1 session | 3.35 | 0.8% |
| 5 sessions | 16.76 | 4.0% |
| 20 sessions | 67.04 | 16.1% |

The customary 0.01·T rule = 4.17 sessions, and is correct **only if the slowest feature has a ~1-session half-life.**

**And the embargo caps the CSCV design**, because a block must not be swallowed: ⌊T/S⌋ ≥ 2h.

| h | usable S on T = 250 |
|---|---|
| 5 | 8, 10, 12, 16, 20 |
| 10 | 8, 10, 12 |
| 15 | 8 |
| **≥ 20** | **none — CSCV is unavailable** |

> **This is a hard design constraint, not a preference.** Cap every feature's trailing window and memory half-life at ≤ 1 session → h = 5 → S = 16 stays available. **Admit a single 20-day realized-vol feature and h = 53.5, and CSCV collapses entirely.** Enumerate every feature's lookback in bars and take the max; that one number decides whether the process-level overfitting check exists at all.

Assert ⌊T/S⌋ ≥ 2h in code and log the realized training-set size per split — an embargo that silently eats a neighbouring block raises no error.

---

## 4. The honest verdict, and the design that follows

> **A self-improvement loop that needs many gated decisions is NOT buildable on this data.** It supports a handful — 3 on the development block, ~10 gating on all 417 — and that is the whole statistical capital of the dataset, for the life of the project.

Three independent lines converge, none a matter of taste: **power** (the 80%-power MDE on Block A is SR_ann 2.51, so a real improvement worth 0.5–1.0 Sharpe is invisible and the gate will reject it — a Type II problem to accept explicitly, not tune away); **selection** (DSR permits 3 decisions on Block A at an observed 2.5); and the **noise floor** (two candidates correlating below 0.8 and differing by less than ~1.2 annualized Sharpe are indistinguishable).

**Calibration against the gate we already ship.** The text gate's measured null-floor CI is [0.9212, 1.0670], half-width 0.0729 = 7.3% of the metric's own scale, against an effect of interest around 10–15%. The trading gate's MDE is 1.94 against a plausible improvement of ~0.3. *([open] this ratio mixes an 80%-power MDE (2.49×SE) with a 95% CI half-width (1.96×SE), overstating the trading gate's relative coarseness by ≈1.27×; the "~0.3 plausible improvement" denominator is asserted with no source. The direction survives; the factor should be restated on matched definitions.)* The qualitative statement stands and is what matters:

> **It is not the same instrument with worse variance. It is an instrument that cannot see the thing it is pointed at.**

### (a) Two tiers — this is what makes a loop exist at all

- **Inner loop, unlimited**, on Block A's IS half. Score on things that are **not the gated statistic**: feature-causality assertions, turnover, fill realism, cost sensitivity, breakeven cost in bps, code-level invariants, and at most a PASS/FAIL against a deliberately lenient in-sample threshold (Harvey & Liu's own recommendation: p = 0.2 in-sample, then intersect with what survives out-of-sample).
- **Return PASS/FAIL or a 3-bucket quantized score, never a raw Sharpe.** Information per trial is the rate at which the dataset is consumed. At the noise floor, an agent optimizing against a returned real number finds SR_ann 2.4 in ~500 tries **on pure noise**.
- **Outer gate:** ≤ 3 firings against Block A, exactly 1 against Block B.

Per CSCV split, the agent's replay tool is restricted to the IS half, the proposal is scored once on the OOS half, and **the score is not shown to the agent.** If instead the agent sees all of Block A and CSCV runs on its final rule, the path distribution measures sequence-sensitivity of a rule that already memorised the block, and only DSR and PBO still say anything true.

### (b) Exploit the paired structure — the only thing that rescues this

Gate on d_t = P&L_ON,t − P&L_OFF,t, exactly as `gate/decide.py` already gates on a paired per-task ratio. σ_d = σ√(2(1−ρ)), so the smallest detectable **standalone** improvement is MDE_paired · √(2(1−ρ)):

| corr(candidate, incumbent) | Block A (250) | Block B (146) |
|---|---|---|
| 0.50 | 2.51 | 3.30 |
| 0.80 | 1.59 | 2.09 |
| 0.90 | 1.12 | 1.47 |
| 0.95 | 0.794 | 1.044 |
| **0.99** | **0.355** | **0.467** |

> **Instruct the agent to propose small, high-correlation modifications. Have the harness report ρ on every proposal and refuse to grade anything below ρ = 0.9 as an incremental change** — a rewrite correlating 0.5 with the incumbent must be judged as a fresh strategy against the full 2.51 bar.

The double benefit: at ρ = 0.99 a 0.355 standalone improvement becomes a paired θ of 2.5, which is exactly the accept bar whose trial budget is 10 rather than 3.

**[open] Two problems this creates, both unresolved.** A sequential accept-then-rebaseline loop measures trial k+1 against a *different* incumbent than trial k, so the trials are **not exchangeable** and the False Strategy theorem's E[max_N] is misestimated in an unknown direction. And **there is no incumbent for the first decision** — the day-one accept must clear the unpaired 2.51 bar, the hardest in the design, and nothing here says what the day-one baseline is. Both must be settled before pre-registration.

### (c) Pick E[max] first, publish it, derive the threshold from it

Do not pick a trial count and back into a threshold. Recommended: **E[max] = 2.5 on the paired statistic → budget 10 independent trials on 417 sessions.**

### (d) Say in advance what happens when the holdout fails

The honest answer is **the project stops**, not "we adjust and re-test." The sealed holdout is the only defense against the agent conditioning its next proposal on the gate's verdict, and no formula here corrects for that. **It works exactly once.**

---

## 5. Pre-registration — write it, hash it, commit it, before the first replay

The trial ledger and the counter are the two items that cannot be audited from outside, and are therefore the two that will be wrong.

**Units and statistic**
1. q = 252; one P&L observation per session; SR per-observation everywhere; γ₄ **raw** (`fisher=False`). Annualize for display only, at the very end.
2. Primary statistic: paired d_t = P&L_ON,t − P&L_OFF,t over Block A.
3. SE: Lo eq. 9 with the Mertens non-normal correction on d_t's own skew and kurtosis; cross-checked against a stationary-bootstrap CI.
4. Serial correlation: Ljung-Box Q on d_t at 10 lags; if it rejects at 5%, switch to HAC/GMM and report both. Annualize with Lo's η(q), reporting η(q) and √252 side by side so the gap is visible.
5. Bootstrap block length: Politis–White automatic **with the Patton–Politis–White (2009) correction, D_SB = 2g²(0)** — the 2004 D_SB is wrong and still circulates in code. **Do not accept `arch`'s √T default:** √417 = 20 sessions leaves ~21 effective blocks.

**The counter — the item most likely to be corrupted**

6. **Append-only trial ledger, written at the tool-call boundary, not the decision boundary.** One row per replay evaluation: timestamp, git SHA, full rule diff, date range replayed, and **the full per-session P&L vector**. N counts every proposal ever scored against replayed P&L — rejects, errors, exploratory runs, re-runs after a bug fix in the replay engine. **Never a number a human types.**

   `gate/audit.jsonl` is the right precedent; the P&L vector is the addition, and it is what makes V[{SR_n}], ρ̄ and CSCV's matrix computable at all. **Note the precedent's own failure mode, verified 2026-08-31:** that ledger holds 2 rows for 3 spent decisions, because `run_decision.sh` never calls `decide.py` and `--audit` is optional with `default=None`. **A trading trial ledger must be written by the path that runs the trial, not by a later optional step.**

   > **This project's decisive advantage over the entire published literature is that it can observe its own N.** Harvey–Liu–Zhu had to estimate hidden tests structurally (M ≈ 1,300 against 316 observed). We can just count. Discarding failed proposals throws that away.

7. **The budget as a hard counter:** E[max] = 2.5 → ≤ 10 effectively-independent trials against Block A, exactly 1 against Block B. Pre-register the consequence of overrun: DSR is recomputed at the higher N, which mechanically raises the bar. **`max_decisions: 5` in `bench/splits/s6lite-v0.json` is the right primitive and is currently read by nothing — do not repeat that here.**
8. ρ̄ estimation: raw correlation matrix while M ≤ ~250, ONC clustering above that, N = cluster count. Pre-register that N comes from ONC, not a hand-count.

**Accept rules**

9. **Primary: DSR > 0.95**, with its seven inputs named explicitly — SR̂ of the selected d-series, T, γ₃, γ₄, V[{SR_n}] across all trials, N (deflated), and the constants. *The input that lies is N: it is the only one nobody outside the process can verify, and understating it inflates DSR monotonically.* Verify the Gumbel term against Monte Carlo at the actual N (see §2).
10. **Secondary, process-level: PBO ≤ 0.05** via CSCV on the Block A trial matrix (250 × M), S = 16, C(16,8) = 12,870 splits, blocks of 15 sessions, purge + embargo at every seam. **Report two companions or do not report PBO at all:** the performance-degradation slope (OLS of OOS on IS across splits — **negative b is a reject even if PBO passes**) and Prob[R̄ < 0] (**can be high while PBO ≈ 0**; reporting PBO alone is the standard misuse). Gate on Sharpe under CSCV, never on drawdown or Calmar — the recombined path is not a real path. Cost: replay each trial **once** over all sessions to build the matrix; CSCV is then pure arithmetic. You pay M replays, not M × 12,870. **CSCV has no purging at its own submatrix seams — add it.**
11. **Worst-regime rule**, since regime narrowness is the declared #1 risk: on the mechanical partition frozen before the split is cut, accept on **min-over-regimes paired Sharpe > 0, reported as a vector, never pooled.** State the cost honestly: ~4 regimes over 250 sessions is ~60 each, MDE ≈ 5.2. **You can detect a regime breaking. You cannot certify one working.**
12. Holdout rule and MinTRL computed and recorded **before** sealing: the claim must be θ ≥ 2.19 or Block B cannot resolve it. Pre-register what happens if it fails (§4d).

**Structure and enforcement**

13. The split dates, hashed, frozen, never re-drawn. Seal enforced in code, not policy.
14. The embargo h = L_max + τ×half-life, with **every feature's lookback enumerated in bars** in the pre-registration, plus the assertion ⌊T/S⌋ ≥ 2h.
15. **Cost model frozen** — commissions, spread, a slippage model in bps, and the **breakeven cost at which SR → 0**. For 1-minute intraday on 500 names this dominates everything and must not be tunable after the fact. Related: fills cannot be modelled from 1-minute OHLCV because there is no book — replace a point estimate with a **fill-cost sensitivity band** (next-bar-open +0/+1/+2 ticks adverse under a participation cap) and pre-register the reading: *a strategy whose edge dies between 0 and 1 tick is not an edge, it is a fill assumption.*
16. **Look-ahead audit as a CI test suite, not a checklist.** The one test worth more than the rest combined: **recompute every feature on a truncated prefix and assert it equals its value in the full run at every shared timestamp.** Any feature that fails is non-causal, and this catches most normalization leaks in one assertion. Plus: point-in-time index membership; as-of price adjustment, not back-adjusted; no `bfill`, no bidirectional `interpolate`; and a **planted-leak test** proving the purge/embargo splitter recovers a null on a pure-noise target while shuffled K-fold reports a strong spurious score.
17. A **stopping rule**, committed in advance. Committing to it is what keeps N finite and therefore keeps the hurdle finite.

**The sentence at the top of the pre-registration:**

> *A perfect DSR on a leaky backtest is a perfectly precise wrong answer.* Every instrument here is a multiple-testing-and-non-normality correction applied to a return series whose correctness you have separately asserted. **None of them detects look-ahead, survivorship, missing costs, capacity, or regime change.**

---

## 6. Instruments considered and rejected, and what to pull from Alpaca

**Haircut Sharpe (Harvey–Liu) — for reporting, never for gating.** Four traps verified in the authors' own code: `Haircut_SR.m` assumes **360 days/year** on the daily branch, so passing 417 trading days understates the sample by 35%; the `ind_an==0 && ind_aut==1` branch references an undefined lowercase `sr` and throws; Holm and BHY haircuts are **stochastic** medians over 2,000 draws (the paper prints 0.616% and 0.621% for the same parameters), so an automated gate comparing against a re-simulated threshold flips decisions on reruns; and the simulated "other tests" are hard-coded to a US equity **factor** null. **[open] An earlier pass worked around the 360-day bug by entering 417 sessions as "20 monthly-equivalent observations" and reported conclusions from it. That forces T = 20 into formulas whose own caveat cites ~30 as the CLT floor. Those conclusions are withdrawn.** Since this project logs every p-value, **compute Holm and BHY directly on the logged vector** — strictly better than the paper's simulation, because the authors never had all N p-values and we do.

*And the counterintuitive one:* **BHY on a single candidate is harsher than Bonferroni.** Its leniency comes from a step-up structure needing many significant results; its first rung is α/(M·c(M)). At M = 100 the BHY t-hurdle is 3.900 against Bonferroni's 3.481.

**SPA (Hansen), not White's Reality Check**, when there are finalists and one incumbent. Report **all three p-values (lower / consistent / upper)** — if p_l and p_u straddle α the answer is *"undetermined,"* not "significant." Estimate ω̂ with the closed-form kernel, not the variance across bootstrap resamples. Fifty batches at α = 0.05 gives ~92% chance of at least one false accept. **Do not drop poor candidates before running it** — Hansen explicitly corrects that reading.

**What none of them cover:** regime narrowness and the feedback channel. A block bootstrap resamples within the observed regime distribution and cannot manufacture regimes it never saw.

**Alpaca — extending the sample is the highest-leverage action available** (§2, scaling law). Three things to decide before downloading:

1. **How far back.** As far as the subscription allows. Crossing 2020 introduces a very different regime — that is a feature, not a problem, but it must be labelled by the mechanical partition, not by hand.
2. **Adjustment must be as-of the session, not as-of today.** If the feed serves prices adjusted to the present, that is look-ahead baked into the data and no runtime validator can see it.
3. **Point-in-time index membership is the likely gap.** A universe of "symbols in the S&P 500 today" carries survivorship bias decided upstream of the replay, invisible to every statistic in this document. Expect 20–40 constituent changes over 20 months alone. Either source the historical composition, or record it as a **declared, unfixed limitation carried on every result** — and never write "leak-free."

---

## 7. Open defects in this analysis

Listed so they are not lost. Each is flagged **[open]** at its point of use.

1. **DSR's variance input is inconsistent with the correlation deflation** (§2) — holds V[{SR_n}] at the independent null while deflating N for correlation; biases DSR **upward**.
2. **Exchangeability is violated by the sequential accept-then-rebaseline loop** (§4b); E[max_N] misestimated in an unknown direction.
3. **No day-one incumbent is defined** (§4b); the first accept must clear the unpaired 2.51 bar and nothing says against what.
4. **The stress-regime-in-holdout problem was declared unfixable without examination** (§3); a non-contiguous or regime-stratified holdout was never considered.
5. **The regime partition is assumed, not measured** (§3), yet the split recommendation depends on it.
6. **CSCV block arithmetic does not close** (§3): S = 16 × 15 = 240 ≠ 250, no drop rule.
7. **The embargo constant τ = 3.352 is derived at T = 417 and applied at 250** (§3), where it should be ≈ 2.98.
8. **The text-gate/trading-gate noise comparison mixes MDE with a CI half-width** (§4), overstating the ratio by ≈1.27×; and its "~0.3 plausible improvement" denominator has no source.
9. **The haircut-table conclusions are withdrawn** (§6).
10. **Deflation does not close the feedback channel** (§2); §4(a)'s quantized inner score is a mitigation, not a proof.
11. **What happens when a split's decision budget is exhausted is unaddressed.** A fresh split resets the trial ledger, which is exactly what DSR forbids. Decide the policy before the budget is spent, not after.
