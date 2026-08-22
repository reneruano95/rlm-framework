# The verbatim-repetition loop: repetition is the model's; entry may not be

**Revised 2026-08-21 after adversarial review** (two independent refuters over the data; their objections and the re-analysis are §3b–§3c). The first draft's headline, "the drafter is exonerated", overreached: the replay settles *repetition given the state* and says nothing about *entering* it — and the traces show entry was ~10× more frequent under the DFlash2 run, in one category.

**Date:** 2026-08-21 · **Script:** `milestones/s2/replay_loop_ab.py` · **Raw:** `milestones/s2/results/replay-loop-ab/{dflash4,mtp2,base,mtp2-b10375,dflash4-r}.jsonl`, `run.log`, `*.server.log`, `stimuli/`
**Answers:** `milestones/s4/RESULTS-dflash2-rlm-only.md` § Findings, item 1 ("Replay A/B … owed before this finding is closed either way")

## 1. The question

In the DFlash2 root-only re-validation (run `c1740386`, 2026-08-21) two `rlm`
episodes re-emitted one byte-identical REPL cell **70×** (`9d9e47fb`, synth-01
seed 2, `context_exhausted`) and **111×** (`0c1c397d`, synth-07 seed 2,
`budget_kill`). S4's 90 episodes under the MTP root had no such loop (longest
identical run: 3), and every loop on record post-dated the drafter swap. The
suspicion was natural: a block-diffusion drafter that copies from context
being accepted wholesale. R4's adoption criterion is "benchmark success
unchanged", so the question had to be settled, not argued.

## 2. Design

**Stimuli (6).** For each loop, the exact chat-template-rendered root request
stored in the trace (`step-NNNNNN.root_request_ref.blob`, `rendered` stream,
sha256-verified into `stimuli/manifest.json`) at three points:

| stimulus | what the root saw | the reply it gave in the episode |
|---|---|---|
| `onset` | history up to the turn that first produced the loop cell | the loop cell, first instance |
| `repeat1` | … plus that cell and its observation | the loop cell, first repeat |
| `established` | … plus several more repeats (turn 15 / turn 12) | the loop cell again |

**Arms (5), run in this order, one `llama-server` each on port 8080 with
`config.yaml servers.root`'s exact launch line (`-c 32768 -np 1 -ctk/-ctv q8_0
-fa on -ub 512 -b 2048 -lv 4 -lm none --no-context-shift`), differing only in
binary and speculative flags:**

| arm | binary | speculation |
|---|---|---|
| `dflash4` | `tools/llamacpp-vulkan-dflash2` (`b1-5ecbe1a`, PR #27342) | `-md …DFlash2… --spec-type draft-dflash --spec-draft-n-max 4` — **production** |
| `mtp2` | same | `--spec-type draft-mtp --spec-draft-n-max 2` — S4's speculation on today's build |
| `base` | same | none |
| `mtp2-b10375` | `tools/llamacpp-vulkan` (`b10375-ba360efe1`) | `draft-mtp 2` — **S4's actual root binary and flags** |
| `dflash4-r` | as `dflash4` | as `dflash4` — order/thermal replicate, run last |

**Per completion:** `POST /completion` with the bench's own root sampling
(`temperature 0.7`, `top_p 0.8`, `n_predict 1024`, `cache_prompt true`), seeds
1..20, parsed with the bench's own `strip_reasoning` + `extract_cell`
(`rlm/rootclient.py`). **Metric:** the reply's cell equals the loop cell
(exact; a whitespace-normalised variant agreed on every one of 600 replies).
Secondary: prose outside the cell, `final_answer` present, tokens, `stop_type`.
600 completions, 0 errors, 0 truncations.

## 3. Result

Repeat count out of 20 (p = replies with prose, f = replies calling `final_answer`):

| arm | 9d9e-onset | 9d9e-repeat1 | 9d9e-established | 0c1c-onset | 0c1c-repeat1 | 0c1c-established | **pooled /120** |
|---|---|---|---|---|---|---|---|
| `dflash4` | 0 (p6 f4) | 8 (p5 f4) | 17 (p3 f0) | 2 (p0 f0) | 14 (p0 f0) | 20 (p0 f0) | **61** |
| `mtp2` | 2 (p6 f4) | 6 (p5 f1) | 17 (p3 f0) | 5 (p0 f0) | 16 (p0 f0) | 19 (p1 f0) | **65** |
| `base` | 2 (p6 f4) | 6 (p5 f1) | 17 (p3 f0) | 7 (p1 f0) | 17 (p0 f0) | 18 (p1 f0) | **67** |
| `mtp2-b10375` | 2 (p6 f4) | 6 (p5 f1) | 17 (p3 f0) | 5 (p0 f0) | 16 (p0 f0) | 19 (p1 f0) | **65** |
| `dflash4-r` | 0 (p6 f4) | 8 (p5 f4) | 17 (p3 f0) | 2 (p0 f0) | 13 (p0 f0) | 20 (p0 f0) | **60** |

Exact two-sided Fisher, pooled: `dflash4` vs `mtp2` **p = 0.70**; vs `base`
**p = 0.52**; replicate `dflash4` vs `dflash4-r` p = 1.0.

What the table settles:

1. **Given a loop-prone history, repetition does not depend on the drafter.**
   The unspeculated root repeats the cell as often as the DFlash2 root —
   slightly *more*, if anything (67 vs 61 of 120; paired McNemar p = 0.35).
   S4's own binary and flags (`mtp2-b10375`) repeat 65/120. At onset DFlash2
   repeats *less* (2/40 vs 9/40, p = 0.048).
2. **The loop is an attractor of model + prompt state, and it steepens with
   each repeat:** pooled over arms, onset ≈ 6 %, first repeat ≈ 64 %,
   established ≈ 92 %. At the established prompts `final_answer` is 0/20 in
   every arm — once five repeats are in the history the root effectively never
   exits on its own. At the `9d9e47fb` first-repeat prompt the root *could*
   still finish (1–4 of 20 replies submit the correct
   `final_answer("Glincrowetofts Bonding Yard")`), and ≈30 % already repeat.

### 3b. What the review added: the arms are two bodies of draws, not five, and DFlash2 is not sample-neutral

Same-seed raw-reply identity between arms (out of 120):

| pair | identical replies |
|---|---|
| `mtp2` vs `mtp2-b10375` (two binaries) | **120** |
| `mtp2` vs `base` | **102** |
| `dflash4` vs `dflash4-r` | 110 |
| `dflash4` vs `base` | **55** |
| `dflash4` vs `mtp2` | 54 |

MTP speculation is exactly distribution-preserving on this stack (the same
seed yields the same tokens with or without it, across two builds — token-
matching verification of the target's own samples). DFlash2 is **not**: it
diverges from the unspeculated path on 65 of 120 same-seed replies, and on
those discordant replies it is systematically **shorter** — 44 shorter / 21
longer (sign p = 0.006), replicated by `dflash4-r` (43 / 23, p = 0.019), where
`mtp2` vs `base` is symmetric (10 / 6). So "five arms agree" was an
overstatement: the evidence is one DFlash2 body and one MTP/base body of ~120
draws each, powered (80 %) only for a relative risk ≥ 1.3 pooled and ≥ ~2.5 at
onset. And DFlash2 is a materially different realisation of the sampling
process with a replicated directional bias, which the first draft called
"numerics, not bias" without checking.

### 3c. Entry into the state — the traces, not the replay

The replay conditions on histories the DFlash2 run produced; it cannot
measure how often a root *reaches* a loop-prone state. The trace store can.
For every `rlm`-arm episode of S4 (MTP root, b10375) and of the re-validation
(DFlash2 root), on **non-terminal** root turns (the final `final_answer` turn
produces an empty observation by construction and was excluded — the first
count of this forgot to, and inflated both runs to ~100 %), excluding the two
loop episodes themselves:

| | S4 MTP (90 eps) | DFlash2 (88 non-loop eps) |
|---|---|---|
| episodes with ≥ 1 empty observation | 4 | 12 |
| episodes with an **onset-like state** (prose-free cell → empty observation) | **1** | **11** (Fisher p ≈ 0.002) |
| total empty observations | 6 | 20 |
| … of which outside code QA | 0 | 0 |
| code QA: prose-bearing turns | 42/89 (47 %) | 33/110 (30 %) |
| aggregation / needle / synthesis prose | 55 % / 31 % / 56 % | 55 % / 23 % / 57 % |

Every empty observation in both runs is a genuine code-QA cell whose
conditional prints matched nothing (stdout/stderr/repr/traceback all 0 bytes),
so this is the root writing different cells, not a scaffold regression; the
turn-1 system and user messages are identical across the two runs. The
loop-entry state was therefore **not equally likely in the two runs**: it was
~10× more frequent under the DFlash2 root, confined to the category where
"grep for a name, print matches" cells can come back empty, and accompanied
by a drop in prose on exactly those turns. The code-QA episodes did not loop
because onset repeats only ~6 % of the time; the two loops happened in
synthesis, where entry is rare under either root. Caveat: the two runs are a
day apart, one is four interleaved arms and the other `rlm`-only, and this is
one comparison among several — a chance explanation is weakened, not
excluded. But "S4's 0/90 was chance" is no longer a defensible reading.

## 4. What this changes

- **R4's ruling in `milestones/s4/RESULTS-dflash2-rlm-only.md` stands on its stated
  criteria, but the finding attached to it is now two findings, not one.**
  (i) *Repetition given the state* is the root model's and is fixed in the
  scaffold. (ii) *Entry into the state* is open, with evidence — a 10×
  category-specific excess in the traces and a replicated shorter-reply bias
  in the replay — pointing at the DFlash2 path. Closing (ii) needs an
  **entry-rate A/B on fresh episodes**: the seven code-QA tasks × 3 seeds
  under `dflash: true` and under `mtp: true` (the distribution-preserving
  arm), same day, blocks interleaved, scored on prose-free-turn rate, empty-
  observation rate and turn count — ~42 cells, ~1.5 h. If DFlash2 raises
  entry, R4's "benchmark success unchanged" was met by the 2/3 rule and the
  swap should be re-decided with that cost priced, not assumed away.
- **Two scaffold-side contributors the review found, independent of the
  drafter and present in both runs:** (a) every past assistant turn is
  rendered with **two** empty think blocks (`<think>\n\n</think>\n\n<think>\n\n</think>`)
  because `rootclient.py` stores the assistant message as
  `assistant_prefix(rendered) + raw` and the chat template then prepends its
  own block on re-render — a prompt artifact the model never emitted, and part
  of the state the attractor lives in; (b) the root is sampled with the **same
  seed on every turn** of an episode (`config.yaml scaffold.sampling.root.seed`
  via `rootclient.py:160`), so when the history makes two consecutive turns'
  distributions near-identical the sampler makes the same choices — the replay
  with varied seeds repeats 64–92 %, production with a fixed seed repeated
  70/70 and 111/111. Deriving the per-turn seed from the episode seed and the
  turn index keeps reproducibility and breaks the lock-step; both are spec-
  level changes (§5 C5 / D26) and belong in the same amendment as the guard.
- **The fix that stands regardless of cause: a repetition guard in C5.** N identical consecutive cells
  (the data say N = 2 is already a 64 % repeat signal and N = 3 is near-certain)
  → one scaffold observation naming the repetition and pointing at
  `final_answer`; if the next cell is identical again → terminate as a named
  outcome `repetition_loop`. This is I1-clean (scaffold-owned termination,
  deterministic, no model input), costs a loop ~30 s instead of 1,154–1,308 s,
  and must land as a spec amendment (§5 C5, §6 outcome enum, a version bump)
  with a unit test that feeds the two recorded histories through the guard.
  *(Shipped 2026-08-21 as `budget_kill / max_identical_turns` — the existing
  outcome and the existing reason convention, so no schema, verdict or bench
  change; see ARCHITECTURE.md v0.3.16.)*
  The one design question it leaves is whether the corrective observation
  should be tried at all: at the established prompts the root never recovered
  on its own in 100 replies, but a scaffold message is new input the replay
  did not test — one correction before termination is cheap and measurable.
- **A prompt-side lever is visible but unmeasured:** every loop turn is
  prose-free (`<think>\n\n</think>\n\n` straight into a cell), and the passing
  seeds of the same tasks wrote a sentence before most cells. Whether asking
  for one line of intent before every cell lowers entry into the attractor is
  a controlled prompt A/B on non-benchmark fixtures, not a guess to ship.

## 5. Limits

Two loops, six prompts, both from the synthesis category, both with the
Qwen3.8-27B root at the bench's sampling. Arms share seeds, so they are paired
rather than independent, and MTP arms are the same draws as `base`: the
effective comparison is ~120 DFlash2 draws vs ~120 MTP/base draws, powered for
a relative risk ≥ 1.3 pooled and only ≥ ~2.5 at onset. The replay conditions
on DFlash2-generated histories and cannot speak to entry; the entry evidence
in §3c is observational across two runs a day apart with different arm
composition. The production seed (2 for both loops) does not reproduce the
production onset cell under the production arm, and `dflash4` vs `dflash4-r`
differ on 10/120 same-seed replies where MTP is 120/120 across two binaries —
the DFlash2 path has run-to-run nondeterminism MTP lacks, which the replay
does not explain. Same box, same day, arms back-to-back; the replicate arm
bounds order effects on repetition at zero.
