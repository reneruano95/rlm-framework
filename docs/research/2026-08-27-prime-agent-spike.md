# Prime Agent v0.8.1 on the local 27B root — spike results

**Date:** 2026-08-27 · **Status:** COMPLETE — Phases 0, A, B, C and C1b run; all pre-registered readings resolved.
**Plan:** `docs/superpowers/plans/2026-08-26-prime-agent-local-spike.md` (all readings pre-registered there before any run).
**Question:** the Prime Agent paper (arXiv:2608.23552) evaluated only frontier APIs. Can a Q4_K_M 27B on this Strix Halo box drive the same self-improving RLM harness — and at what cost?
**Evidence grades:** [V] measured on this box in this session; [R] reported by a cited source.

---

## 0. Setup as built (deviations from the plan recorded)

| Item | Plan | As built | Why |
|---|---|---|---|
| Root server | hand-launched, `-c 131072 -np 2 -a qwen3.8-27b --jinja --metrics` | as planned, DFlash2 drafter kept | — |
| `-ctxcp` | decided by Phase 0 gate 4b | **`-ctxcp 0`** (see §2) | measured 1.6× faster end-to-end on this workload |
| WSL user | `spike`, no `/mnt/*`, interop off | as planned, all four checks pass [V] | — |
| `sudo` | assumed available | **not available** (password required); all privileged steps run as `wsl -u root` | box has no passwordless sudo |
| Node | installer fetches standalone Node | **Node v24.20.0 LTS installed system-wide** to `/usr/local/lib/nodejs`, npm prefix `~/.npm-global` for `spike` | `spike` had no Node (rene's v22.22.2 is a per-user install); the installer refuses to auto-install Node without a terminal |
| Installer prompts | expected interactive | ran **non-interactive** (`< /dev/null`): "No terminal detected; continuing without confirmation" | tarball sha-verified by the installer itself |
| Kernel venv | `PRIME_AGENT_KERNEL_VENV` override | worked — venv at `~/prime-spike/kernel-venv`, **Python 3.11.16**; no `~/.prime` tree created [V] | the plan's v2 correction was right: the venv otherwise ignores the agent dir |

Isolation verified after `wsl --shutdown` [V]: `ls /mnt/c` → Permission denied; `ls /mnt/d` → Permission denied; `cmd.exe` → command not found (interop off); `curl :8080/health` → `{"status":"ok"}`; `spike` reads its own corpora. Corpora sha256 match `bench/corpora` byte-for-byte.

Server sanity [V]: `/props.default_generation_settings.n_ctx = 65536`, `total_slots = 2`, `model_alias = qwen3.8-27b`; `/v1/models` id `qwen3.8-27b`; `/metrics` exposes the `llamacpp:*` counters.

---

## 1. Phase 0 gates — all pass

| Gate | Result | Detail |
|---|---|---|
| **1 — raw tool call** | **PASS** [V] | `tool_calls[0].function.name == "ipython"`, `arguments = {"code":"print(2+2)"}` (parseable), final streamed chunk carries `usage` with `prompt_tokens_details.cached_tokens`. 9.75 s, 301 prompt / 32 completion tokens. The local serving stack serves prime-agent's tool contract without modification |
| **2 — harness smoke** | **PASS** [V] | `prime-agent -p --offline --thinking off "…compute 17*23…"` → `391`, exit 0, 55.2 s cold. One `ipython` tool call, one text reply |
| **3 — kernel** | **PASS** [V] | `prime-agent doctor` clean (daemon 0.8.1, current); kernel venv Python 3.11.16 at the isolated path |
| **4a — per-turn cost** | measured [V] | prime-agent's system prompt is **4,813 ± 3 tokens**. Three identical back-to-back runs: 47–51 s each under default `-ctxcp`, 29–31 s under `-ctxcp 0` |
| **4b — `-ctxcp` decision** | **`-ctxcp 0`** [V] | see §2 |
| **5 — corpus read** | **PASS** [V] | 474 KB `corpus.txt` → `6619` lines, correct, 46.3 s |
| **6 — gate mechanism** | **PASS** [V] | with `--autonomous-gate "test -s answer.txt"`: `answer.txt` written, exit 0, and **0 host-continuation messages** in the session JSONL. The plan's v2 correction is validated — without the gate every run would have been scored on a forced second pass |
| **7 — artifact location** | confirmed [V] | with `--session-dir <D>`, artifacts land in `$(dirname <D>)/session-artifacts/<session-id>/`, exactly as the v2 correction stated |

---

## 2. The `-ctxcp` measurement (gate 4b) — and a correction to the plan's rule

Two conditions, server relaunched between, same build and model. Probe: four requests sharing one 6,549-token system prompt, differing only in the user message (the shape every prime-agent run has).

| | cold prefill (6,549 tok) | 2nd–4th request (divergent user msg) | prime-agent run, end to end |
|---|---|---|---|
| default `-ctxcp` | 45.2 s | **7.7 s**, 6,525 / 6,549 cached | 47.4 – 51.4 s |
| `-ctxcp 0` | **27.0 s** | 27.0 s, **0** cached | **29.4 – 30.8 s** |

Two facts, both measured [V]:

1. **Context checkpoints are what makes divergent-prefix reuse possible on this hybrid model.** With `-ctxcp 0` a request that shares a long prefix but diverges at the tail re-prefills everything (0 cached, 3.5× slower). This *narrows* Gate 0 §6's finding rather than contradicting it, and it is the caveat Gate 0 itself asked to be tested: "measure a real episode both ways before pinning it."
2. **prime-agent never gets that benefit.** Across three back-to-back runs with identical cwd and identical user message, the first assistant message reported `cacheRead = 0` under **both** conditions, and the server processed the full ~4,834-token system prompt fresh each time. Its system prompt varies per session, so consecutive runs share no reusable prefix. What prime-agent *does* reuse is the within-run continuation (turn 2 read ~4,843 cached tokens under both conditions).

So the checkpoint machinery costs ~18–20 s per prime-agent run and buys it nothing.

**Deviation from the pre-registered rule, stated plainly.** The plan's rule was: "use `-ctxcp 0` if `cache_n` on the divergent request is equal under both settings; otherwise keep the default." Read literally, the rule selects the default, because `cache_n` is *not* equal (6,525 vs 0). But the rule's premise — that divergent-prefix reuse is what prime-agent's runs depend on — is falsified by fact 2 above. Chosen: **`-ctxcp 0`**, on the measured 1.6× end-to-end speedup on the workload actually being run. Both conditions' numbers are recorded here so any reader can adjust.

**Consequence for A-cost, stated before the numbers are read:** the S4 and DFlash2 baselines were measured on a root running the *default* `-ctxcp`. A-cost ratios below therefore compare prime-agent-on-`-ctxcp 0` against scaffold-on-default. The measured factor between the two server conditions is 1.6× on prime-agent's own workload; the scaffold's own append-only workload was measured by Gate 0 at 13.88 s → 2.48 s per turn. Neither correction is applied to the ratios; both are stated.

**Open item for Phase C:** compaction produces exactly the divergent-prefix case that `-ctxcp 0` cannot reuse. If compaction fires in the long session, each post-compaction turn re-prefills the whole retained context (~50–60K tokens ≈ 3.5 min at the measured 243 t/s). This is recorded as a hazard to watch, not a prediction.

---

## 3. Instrument notes (things a later reader will need)

- **prime-agent's own `usage.cacheRead` disagrees with the server.** In every cross-run probe the harness reported `cacheRead = 0` on the first assistant message while `llamacpp:prompt_tokens_cached_total` showed the within-run reuse. Token accounting in this spike therefore uses the **`/metrics` deltas as the [V] source** and treats the harness's numbers as [R], exactly as the plan required.
- **Session JSONL shape, confirmed against a real file** [V]: entries are `{type, id, parentId, timestamp, message?}`; `message.role ∈ user | assistant | toolResult`; assistant content blocks are `{type:"toolCall", id, name, arguments:{code}}` or `{type:"text", text}`; assistant carries `usage{input,output,cacheRead,cacheWrite,totalTokens,cost}`, `stopReason ∈ stop|toolUse|length|error|aborted`, `model`, `provider`, `responseId`; toolResult carries `toolName`, `content[].text`, `details{durationMs,status,stdout,stderr,result,kernelRestarted}`, `isError`. Non-message entry types seen: `session`, `model_change`, `thinking_level_change`, `service_tier_change`, `session_state`.
- **`.gitignore` hides the tooling.** Line 2 is an unanchored `tools/`, so `docs/research/2026-08-26-prime-agent-spike/tools/` is invisible to `git status`. It needs `git add -f` or a negation before these files can be committed.

---

## 4. Phase A — parity on v1: **A1**

24 runs (8 tasks × 3), sequential, 26.9 min wall, 2026-08-27 09:33–09:59. Server: `-ctxcp 0`. Every run scored by `rlm.measure.checkers` on the frozen v1 answer key.

**23 of 24 runs correct; 8 of 8 tasks pass (≥2/3 runs) → A1: the Q4_K_M 27B drives prime-agent's RLM harness.** F2 — the reported collapse of RLM scaffolds on small/quantized roots — does not apply to this model in this harness on these tasks. A′ was not run: the pre-registered trigger is a failing *category*, and no category failed.

| task | category | pass | median wall | vs DFlash2 wall | vs S4 wall | median tokens [V] | vs DFlash2 tokens |
|---|---|---|---|---|---|---|---|
| agg-03 | aggregation | 3/3 | 49.3 s | 1.33× | 0.97× | 5,720 | 0.75× |
| agg-04 | aggregation | 2/3 | 80.8 s | 1.05× | 0.83× | 6,600 | 0.34× |
| agg-07 | aggregation | 3/3 | 105.3 s | 1.10× | 1.24× | 8,262 | 0.38× |
| codeqa-01 | code QA | 3/3 | 49.0 s | 0.74× | 0.71× | 5,265 | 0.46× |
| codeqa-03 | code QA | 3/3 | 46.4 s | 0.43× | 0.64× | 5,301 | 0.22× |
| codeqa-05 | code QA | 3/3 | 47.1 s | 0.48× | 0.54× | 5,294 | 0.31× |
| needle-02 | needle | 3/3 | 69.1 s | 1.59× | 0.88× | 6,449 | 0.73× |
| synth-02 | synthesis | 3/3 | 58.2 s | 0.41× | 0.61× | 6,207 | 0.17× |

**A-cost.** Wall is a wash — 4 of 8 tasks faster than the scaffold's DFlash2 run, 4 slower, none outside 0.41×–1.59×. Tokens are not a wash: **prime-agent processed 0.17×–0.75× the scaffold's tokens on every task.** Over all 24 runs the server processed 144,161 prompt tokens and predicted 11,393, with 518,070 served from cache. Read this with §2's caveat — the baselines ran on a root with default `-ctxcp`, and the token columns compare `/metrics` non-cached prompt tokens against the scaffold's total step tokens. The direction is large enough to survive either correction; the exact multiple is not.

**A-loop: not reproduced.** `max_identical_streak = 1` in all 24 runs — no repetition attractor, against R15's ≈6% onset rate in the scaffold and prime-agent's own #1326 (20–50 identical calls on a local Qwen). Zero errors, every run `stopReason = stop`, every exit code 0.

**A-refine: clean.** Zero model-initiated `refine.run` calls across 24 runs — the prompt's prohibition held, so no Phase A run mutated harness state.

**A-continuations: zero.** No run received the host's autonomous-continuation nudge, confirming the gate mechanism ended every run at the model's own stop.

Turns per run: median 5 (4–12). `ipython` tool calls per run: median 3 (2–10). No subagents were spawned at depth 1.

### 4.1 The one failure, in full

`agg-04 run1` answered `1` against an expected `594`. The transcript shows the model loading the file, extracting 1,323 records with one regex, then writing a second pattern to join header against disposition:

```python
pat = re.compile(r'^\[ENT-\d+\]\s+(.+?)\nStatus: (\w+)\nDisposition: custody passed from (.+?) to (.+?);',
                 flags=re.DOTALL)
```

`re.DOTALL` without `re.MULTILINE`: `^` matches only at the start of the string, so `finditer` yielded exactly one match. The model repaired a `the`-prefix mismatch on that single record, printed `withheld: 1`, wrote `FINAL: 1`, and explained in prose that "only one record matches" — **without ever reconciling 1 against the 1,323 records its own previous cell had counted.**

The scaffold's `strat-aggregation.v2` prompt states that check explicitly ("Cross-check the count with a second, independently written pattern before you trust it"; "Report coverage before answering"). The spike's generic P1 prompt does not. Runs 2 and 3 of the same task, same prompt, got it right — so this is within-task variance in whether the model checks its own coverage, not a capability ceiling. It is the strongest single argument in this spike for what a learned `prompt`/`skill` artifact should encode first.

---

## 5. Phase B — self-improvement: **B1 passes decisively, B2 fails, and the content is the finding**

Protocol as run (a correction to the plan, recorded): the plan's v2 note said `/refine` is interactive-only. **It is not** — `prime-agent -p --continue "/refine"` works and returns "Refined continual harness state: N edit(s) applied" [V]. Phase B therefore ran entirely in print mode, scripted, with the same caps as Phase A, instead of in the TUI. `--resume` accepts the session file path or the header `id`; note the **filename is not the session id** (`01a0435b-dc68-…jsonl` holds id `01a0435b-e080-…`).

Per category: global harness archived and cleared → one train session, four tasks with an operator `/refine` (no instructions) after each → final local state promoted to global **by file copy** and re-scoped to `global` → held-out and re-test in fresh print-mode sessions with the Phase A invocation. Train turns swapped `./corpus.txt` inside one cwd so held-out runs are byte-identical in shape to Phase A. 34 min wall, 10:03–10:37.

### 5.1 B1 — the JSON contract holds: **8/8**

Every one of the eight operator `/refine` calls returned an applied edit and left a valid `harness_state.json` (threshold was ≥3/4 per category). Not one "Refiner did not return a JSON object". Issue #1143 reports that failure for local Ollama models; **this root does not exhibit it.** Train task answers were 8/8 correct as well.

### 5.2 B2 / B3 — measured against Phase A

| task | role | Phase A | Phase B (memories loaded) | wall Δ | token Δ |
|---|---|---|---|---|---|
| agg-07 | held-out | 3/3, 105.3 s, 8,262 tok | **2/3**, 233.3 s, 21,169 tok | **2.22×** | **2.56×** |
| agg-03 | re-test | 3/3, 49.3 s, 5,720 tok | 3/3, 56.0 s, 6,034 tok | 1.14× | 1.05× |
| codeqa-05 | held-out | 3/3, 47.1 s, 5,294 tok | 3/3, 55.1 s, 5,957 tok | 1.17× | 1.13× |
| codeqa-01 | re-test | 3/3, 49.0 s, 5,265 tok | 3/3, 46.7 s, 5,881 tok | 0.95× | 1.12× |

**B2 (transfer) is NOT met.** No held-out task passed where Phase A failed (none had failed), and neither held-out median wall came near the ≤0.8× threshold. **B3 as written does not fire either** — agg-07 still passes at 2/3 — but the measured direction on both held-out tasks is a regression: agg-07 lost a run and cost 2.2× the wall and 2.6× the tokens. **Every one of the four tasks cost more tokens with the artifacts loaded** (1.05×–2.56×). Eleven of twelve held-out runs answered correctly.

The plan's thresholds were written for a n=2×3 signal and this is one; nothing here is a verdict. What the *content* shows is stronger than what the counts show.

### 5.3 B-content — what the root actually wrote (the load-bearing result)

All **8 of 8 refinements are `kind: memory`. Zero `skill`, zero `prompt`, zero `subagent`** — three of the four surfaces prime-agent's Continual Harness offers went unused. And the memories are, overwhelmingly, **answer logs**:

- aggregation: *"SEALED count on ./corpus.txt … History: 514 on the 474,218-char version"*; *"WITHHELD … = 589; RELEASED = 731 … Historical answers in this session: 594 (473,807), 600 (473,758), 589 (474,564)"*.
- code QA, all four: *"Result: exactly one `def in_flight` at corpus offset 167524, in file `rlm/dispatcher.py`. Final answer written to ./answer.txt as `FINAL: rlm/dispatcher.py`. Completed successfully."*

Two qualifications, both in the model's favour, and one against:

- **It did record real method, twice.** The aggregation memory carries a verified parse (`re.compile(r'\[ENT-\d+\] (.+)\nStatus: \S+\nDisposition: custody passed from the (.+) to the (.+);', re.M)`) — with the `re.M` whose absence caused §4.1's only failure. The first code-QA memory carries a genuine procedure ("read file once, regex for FILE headers and `def X\b`, take last header before the definition offset"). The three later code-QA memories drop even that and record only the answer: **procedure degrades over the sequence, answers do not.**
- **It defended itself against its own memorization.** Both aggregation memories say "ALWAYS re-parse rather than trusting a cached number." That instruction held: the agg-03 re-test, whose answer `514` sits verbatim in the memory, **re-parsed the file** (`Counter({'OPEN': 810, 'SEALED': 514})`) rather than reciting. So the contamination did not manifest as recitation, and the re-test remains interpretable.
- **It did not use its own method when it mattered.** In the failing `agg-07` run the model's memory carried the proven `re.M` pattern; it wrote a different, worse one (`custody passed (from|to) the (.+?) to the (.+)`), printed `WITHHELD: 0 RELEASED: 749 other: 576` — **576, the correct answer, sitting in the "other" bucket, 43% of all records** — wrote `FINAL: 0`, re-ran the identical cell, and never questioned the imbalance. 368 s and ~29K cached context tokens spent to get further from the answer than the memory-free Phase A run did in 105 s.

**This is the S6-lite gate's justification, observed rather than argued.** Given a free hand, a competent local root writes memories that memorize *answers to the tasks it was just shown*. ARCHITECTURE §9 S6 already forbids exactly this ("artifacts may be derived only from the v2 train split — an artifact traceable to a held-out task voids the run"), and prime-agent has no equivalent rule: its `/refine` accepts what the model proposes, and only the base system prompt is fenced. Had the held-out task been one of the four train tasks, the memory would have supplied the answer directly.

### 5.4 A flaw in this Phase B design, recorded

Swapping `./corpus.txt` between train turns inside one cwd taught the model something false about the world — that "the file regenerates between questions (same path, different bytes)" — which it wrote into both aggregation memories. It happened to make the memory self-correcting ("always re-parse"), but a repeat should give each train task its own filename or its own session, and treat the "regenerates" line as an artifact of this harness, not a fact.

---

## 6. Phase C — the long autonomous session: **C2 passes 7/7, and the session was not long**

One `--goal` session, depth 2, global harness archived and cleared first (Phase C measures the harness, not Phase B's memories). Budget: 2 h wall, 1.5M tokens, 400 turns, 20 continuations, gate `len(results.json) == 7`.

**It finished in 112 seconds, exit 0, with all seven counts correct** [V]:

```json
{"agg-01": 541, "agg-02": 544, "agg-03": 514, "agg-04": 520,
 "agg-05": 519, "agg-06": 539, "agg-07": 546}
```

Byte-identical to the §2-registered key derived by `grep -cE '^Status: SEALED$'`. **C2 passes (7/7 against a ≥5/7 threshold).** This also settles the plan's v2 answer-key correction empirically: had the task-JSON answers been used (594/600/589/576, which answer a different question about custody direction), a perfect run would have scored 3/7.

**C3 — no delegation.** Zero subagents were spawned at depth 2. Nothing queued; the second slot was never needed.

**The interesting part is what the model did**, because it is the exact behaviour §4.1's failure lacked. Turn 1 inspected the format. Turn 2 defined **two independently written counters** — a line scan (`line.strip() == "Status: SEALED"`) and a record parse (split on blank lines, read each record's Status line) — ran both over all seven files and printed `MATCH` for every one. Turn 3 wrote the results with an atomic temp-file merge.

The goal prompt said "Verify each count by a second, independently written method." It did exactly that, first try, on all seven files. In Phase A, where the prompt did not ask, the same model on the same category shipped a count of `1` against 1,323 records without reconciling them. **The cross-checking behaviour is available in this root and is elicited by asking for it** — which is precisely what a learned `prompt` artifact is for, and precisely the artifact kind the root never produced in Phase B (0 of 8).

**C1a — the pre-registered endurance reading does not apply.** The plan expected a 2-hour session; the task took 112 s because the root writes one program and runs it seven times, so no endurance conclusion can be drawn from Phase C itself. What *is* recorded: at the moment Phase C ended, the single `llama-server` process had been up **75.7 minutes** across Phases 0, A, B and C, serving **7,910 decodes / 390,842 processed prompt tokens / 1.22M cached tokens**, RSS 9.77 GB. That already more than doubles the project's previous provable maximum of ~35 minutes (`docs/superpowers/specs/2026-08-22-long-horizon-agent-design.md` §3.3).

**C1b is therefore the real Q3 test**, run as a separate 2,000-request loop against that same process.

### 6.1 C1b — endurance: **PASS, and it is the first such measurement this project has**

2,000 sequential, prefix-disjoint chat completions (distinct ~200-token user messages, `max_tokens: 64`, `temperature: 0`, thinking off) against the same `llama-server` process that had already served Phases 0, A, B and C.

| | |
|---|---|
| requests attempted | **2,000** |
| 2xx / non-2xx / exceptions | **2,000 / 0 / 0** |
| total wall | 9,177 s (153.0 min) |
| wall_ms median / p95 / max | 4,617 / 4,808 / 9,048 |
| `/health` after the run | `{"status":"ok"}` |
| process uptime at the end | **230 min (3 h 50 min)** |
| RSS at launch → after 2,000 requests | 9,070 MB → **8,375 MB** |

**Zero failures, and RSS ended 7.7% *below* the launch figure** (it peaked near 10,056 MB mid-run and was released), against a pre-registered threshold of "alive and RSS < 2×". Cumulative server totals at the end: 880,807 processed prompt tokens, 1.22M cached, 33,274 predicted, 11,910 decodes.

This crosses upstream #23181's reported failure window (1,500–2,000 sequential chat completions, crash in 30 min–3 h) without an error, and it raises this project's provable single-process runtime from **~35 minutes to 3 h 50 min** — the number `docs/superpowers/specs/2026-08-22-long-horizon-agent-design.md` §3.3 called unprovable from existing data, and which gates every long-horizon path.

**Two caveats travel with it.** (1) This server runs `-ctxcp 0`, so context checkpoints are disabled, and #23181's crash always follows a "created context checkpoint" line. C1b establishes the endurance of **the configuration this spike would ship**; it does not test the upstream trigger, which needs a separate run under the default `-ctxcp`. (2) Mean wall drifted **+9.9%** across the run (first 200 requests 4,232 ms → last 200 4,650 ms), consistent with R9's thermal plateau (+3.5% in the Gate 0 soak) but larger; it is a drift, not a degradation, and no request failed.

---

## 7. What this spike settles, and what it changes

### 7.1 The three questions

| | verdict |
|---|---|
| **Q1 — does a Q4_K_M 27B drive an RLM harness?** | **Yes.** 8/8 tasks, 23/24 runs, on fewer tokens than the scaffold in all eight (0.17×–0.75×) and comparable wall (0.41×–1.59×). No repetition loops, no errors. The F2 collapse reported for small/quantized roots does not appear in this model, this harness, these tasks |
| **Q2 — does self-improvement work on local weights?** | **The mechanism yes; the learning no.** 8/8 `/refine` calls produced valid applied edits — the JSON contract that #1143 reports as broken for local models held every time. But nothing transferred: no held-out task improved, every task cost more tokens, and the one held-out run that regressed did so while *ignoring the verified method its own memory carried* |
| **Q3 — what does a long run cost and break?** | **Nothing broke.** The autonomous goal finished 7/7 in 112 s; the endurance loop ran 2,000 requests over 153 min with zero failures on a process that stayed up 3 h 50 min with no memory growth |

### 7.2 The result that changes the build order

Given a free hand, this root wrote **8 of 8 artifacts as `memory`, and their content is answer logs** — `514`, `594/600/589`, `"exactly one def in_flight at offset 167524, in rlm/dispatcher.py"`. Zero `skill`, zero `prompt`, zero `subagent`. Three of the four surfaces the Continual Harness offers went unused, and the surface it did use, it used to memorize what it had just been shown.

Two measured facts frame what that means:

1. **The behaviour that would have helped is available, and is elicited by asking.** Phase C's goal said "verify each count by a second, independently written method"; the model wrote two independent counters and cross-checked all seven files on the first try. Phase A's generic prompt did not ask, and `agg-04 run1` shipped a count of `1` against 1,323 records without reconciling them. The missing artifact was a **`prompt`** — exactly the kind never produced.
2. **The gate is not optional, and this is the evidence, not the argument.** ARCHITECTURE §9 S6 already forbids what happened here in writing ("artifacts may be derived **only** from the v2 train split — an artifact traceable to a held-out task voids the run"). prime-agent has no equivalent: `/refine` applies what the model proposes, and only the base system prompt is fenced. Had a held-out task been among the four train tasks, the memory would have handed over the answer.

**Consequences for S6-lite v0, on evidence rather than hypothesis:**

- The **scaffold** should author the artifact from traces; the model proposing from its own conversation produced answer logs, not method.
- Start with **`prompt` and `skill`**, not a memory of facts. The one genuinely reusable thing the root recorded (a verified `re.M` parse; a "read once, regex headers, take the last header before the offset" procedure) appeared early and **degraded over the sequence** — later refinements recorded only outcomes.
- The **held-out gate** is the part that carries the design. Without it, this loop would have shipped four memorized answers as an improvement.
- Any local long-horizon slice can now assume a **provable 3 h 50 min server**, subject to §6.1's `-ctxcp` caveat.

### 7.3 What this does not settle

- **D-C5 (build vs adopt) is not re-litigated here.** The spike strengthens the recorded verdict but for a different reason than the one on file: prime-agent works well on local weights — what it lacks is the gate. §5.3 is the datum; the decision stays the owner's.
- Delegation is untouched: prime-agent's subagents call the same 27B, there is no leaf, and zero subagents were spawned. Benchmark v2 remains the instrument for that.
- n is small throughout (3 runs per task; 2 held-out tasks per category). Every count here is a signal, not a verdict.
- `-ctxcp 0` versus the default is measured for prime-agent's workload only; whether it belongs in `config.yaml` for the scaffold's own root remains a §8 comparability event and the owner's call.
