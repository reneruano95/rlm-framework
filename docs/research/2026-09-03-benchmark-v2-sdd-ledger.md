# SDD ledger — plan: docs/superpowers/plans/2026-09-02-benchmark-v2.md

Spec: docs/superpowers/specs/2026-08-25-benchmark-v2-design.md
Branch: feat/benchmark-v2 (created from main @ cca5074)

Ruling (setup): work happens on branch `feat/benchmark-v2` in the primary working
directory, not a git worktree — the repo's benchmark tooling reads absolute model,
server and trace paths and every prior phase (S1..S5) landed on a branch off main
in-place. Cost if wrong: the user's working tree carries the WIP until merge; `git
switch main` restores it.

## Pre-flight scan

### Cross-task rows (tasks sharing a file or interface)

| tasks | shared surface | produces / consumes | finding |
|---|---|---|---|
| 1 → 2,3,4,6,18 | `bench/manifest.py` `rules`, `scored_tasks()`, v2 TaskEntry fields | T1 produces; T2 reads `manifest.rules`, T3/T4 read via `BenchmarkRules`, T6 reads `scored_tasks()`, T18 writes them | consistent |
| 1 → 18 | `manifest.validate()` | T1 leaves validate untouched; T18 adds the per-stream v2 branch + `strict_adversaries` | consistent; T18 owns validate |
| 2 → 3,4,6 | `BenchmarkRules` fields | T2 produces the frozen dataclass; T3 uses margin/band/tripwire/baselines/rlm_arm, T4 uses `abstains`, T6 uses `baselines`/`scored_stream` | consistent |
| 3 → 4 | `verdict.decide`, `Verdict`, `PairResult` | T3 adds `Verdict.rules`; T4 adds `PairResult.n_tasks` and abstention-aware pair math | consistent — T4 edits the same `decide` T3 rewrote; T4 must be built on T3's version |
| 4 → 6 | `load_grid(..., abstentions=, categories=)` | T4 adds the kwargs; T6's `cli._bench` passes them | consistent |
| 4 → 23 | ledger `outcome: "abstained"` | T4 writes it, T23 asserts it in the smoke ledger | consistent |
| 5 → 9 | `cli` runner map for `rlm-nosubcalls` | T5 registers a placeholder raising `ConfigError`; T9 replaces it | consistent, sequenced |
| 5 → 18,24 | `ARM_ORDER` literal | T5 pins `("rlm","rlm-restricted","rlm-nosubcalls","b2","b1","b3")`; T24 documents that order | consistent |
| 6 → 21,22 | `bench_manifest_path(version)` / `benchmark.version: v2` | T6 produces; T21 sets version, T22 pins the sha | consistent |
| 7 → 8,20,21 | `strategy_templates` dict + `strategy_templates_nosubcalls` | T7 produces the schema; T8/T20 author the files; T21 pins them | consistent |
| 7 → 9 | `render_root(..., no_subcalls=)` | T7 produces; T9 calls it from `episode` | consistent |
| 8 → 20,21 | prompt sha pins | T8/T20 create files; T21 records sha256 over the whole file | consistent |
| 9 → 12 | `run_episode` signature | T9 adds `no_subcalls=`, T12 adds `interactive=`; disjoint kwargs, same ctor block | consistent; T12 built on T9's file |
| 10 → 12 | `ActionType.ENV_CALL` + DuckDB enum | T10 produces; T12's `_on_env` writes the step | consistent, sequenced |
| 11 → 12 | bridge kind `"env"` | T11 child sends; T12 manager serves via `on_env` | consistent — T11's tests need a manager-side recorder, which is T12's `on_env`; see Ruling PF-1 |
| 12 → 17,18 | `DOC_DELIM` / `=== DOCUMENT id: title ===` grammar | T12 parses; T17/T18 emit | consistent |
| 13 → 14,15,16,17,18 | `load_trec()`, `Item` | T13 produces; the rest consume | consistent |
| 14 → 15,16 | `LinearSemanticCorpus` | T14 produces; T15's adversaries read `.text/.items/.labels/.record_ids`; T16 adds `paraphrase=` | consistent |
| 15 → 16,17,18 | `parser_adversary`, `self_read_adversary` | T15 produces; T16 asserts the paraphrase defeats it; T17 reuses the window helper; T18 gates on both | consistent |
| 17 → 12 | `InteractiveIndex` used in a bench test | T17's test imports `InteractiveIndex` from `rlm.context.interactive` | consistent (T12 < T17); but `checks/test_bench_v2_corpus.py` then imports `rlm` — allowed, `checks/` is test code |
| 18 → 19,22 | `bench/manifest.v2.json` | T18 writes it under tmp_path in tests; T22 writes the real one | consistent; Phase D gate forbids committing the real file before T22 |
| 21 → 22 | `config.v2.yaml` `manifest_sha256: null` → pinned | T21 leaves null, T22 fills it | consistent |

### Intra-task rows (does each task agree with itself)

| task | finding |
|---|---|
| 1 | agrees. `_OPTIONAL_V2` is corrected to `ClassVar` in the parenthetical. |
| 2 | agrees. |
| 3 | test helpers `six_task_manifest_factory`/`grid_with_passes` are named but may not exist; the task itself says to build them from the file's `_manifest`/`_grid`. |
| 4 | agrees. |
| 5 | agrees. |
| 6 | **conflict**: the test calls `default_task_ids(args)`; the parenthetical specifies `default_task_ids(manifest, args)`. See Ruling PF-2. |
| 7 | agrees. |
| 8 | agrees; the three tests over-constrain the derivation but the derivation is spelled out byte-by-byte. |
| 9 | agrees. |
| 10 | agrees. |
| 11 | **defect**: three test bodies take fixture `session` but call `s.exec_cell`. See Ruling PF-3. |
| 12 | minor: the `InteractiveIndex` interface block omits the `titles` field that `from_text` constructs and `open()` returns. See Ruling PF-4. |
| 13 | agrees. |
| 14 | agrees. |
| 15 | **sequencing**: `test_the_wh_word_rule_is_a_real_adversary_on_trec` passes no `paraphrase=`; T16 introduces the kwarg defaulting True and says to switch this test to `paraphrase=False`. See Ruling PF-5. |
| 16 | agrees. |
| 17 | agrees. |
| 18 | agrees. |
| 19 | agrees (deliberately short). |
| 20 | agrees. |
| 21 | agrees; the parenthetical resolves its own hesitation about `default` in the nosubcalls set (include only the three). |
| 22 | operational; needs a live leaf server on 8081. |
| 23 | operational; needs root+leaf. Carries an explicit owner-stop at >36 h projection. |
| 24 | agrees. |

### Rulings from the scan

Ruling PF-1: Task 11's child-side tests need a scaffold-side `env` handler, which
Task 12 registers via `session.on_env`. Task 11 therefore also adds the minimal
`SandboxSession.on_env` + `_serve` branch (its "Interfaces" already names
`manager._FrameGate`), and Task 12 keeps the rest. Cost if wrong: a two-line
manager hunk lands one task earlier than the plan's file table says.

Ruling PF-2: the extracted helper is `default_task_ids(manifest, task_ids)` ->
`list[str]`, and Task 6's test is written against that signature. The plan's
`default_task_ids(args)` form cannot see the manifest it must read. Cost if
wrong: a one-line signature change in `cli.py` and its test.

Ruling PF-3: Task 11's test bodies use the fixture name they declare (`session`
bound as `s`, i.e. `async def test_...(session): s = session`). Cost: none.

Ruling PF-4: `InteractiveIndex` carries `titles: dict[str, str]` alongside
`docs`/`windows`. Cost if wrong: none; `open()` cannot return a title otherwise.

Ruling PF-5: Task 15 authors `build_linear_semantic(..., paraphrase: bool = False)`
already, and Task 16 flips the default to True. This keeps T15's sanity test
truthful when written AND when re-run after T16, with no edit in between. Cost
if wrong: T16 must additionally edit that test, which is what the plan said.

---

## Execution

Task 1: minor (deferred): checks/test_manifest_v2.py module-level skipif applies to all 3 tests; only the byte-identity one needs bench/manifest.json (plan-mandated).
Task 1: complete (commits cca5074..b88112b, review clean)
Task 2: complete (commits b88112b..62c1dc7, review clean)
Task 3: minor (deferred): render paths outside the brief's line ranges still read module constants -- RLM headline (verdict.py:974), ARMS ordering (:948, :267), cost scorecard BASELINES loop (:1082). A v2 manifest that renamed rlm_arm would render a zero headline. Carried into Task 4's dispatch.
Task 3: minor (deferred): escalation.py:28 has a dead `BASELINES, RLM_ARM` import (pre-existing).
Task 3: minor (deferred): test_verdict.py:96 asserts `"+3-task threshold" not in text`, a substring that only appears in the partial_grid finding which this test never triggers -- non-diagnostic.
Task 3: Ruling: the plan's Global Constraint says `checks/test_import_rules.py` forbids module-scope `bench` imports under src/rlm/. The reviewer read that file: its FORBIDDEN sets name httpx/requests/urllib/aiohttp/socket/http and two rlm.serve modules, NOT `bench`. The real enforcement is the wheel build in checks/verify_distribution.py (bench/ is excluded from the wheel). The constraint STANDS -- every `bench` import under src/rlm/ stays lazy -- only the plan's stated enforcement mechanism was wrong. Cost if wrong: none; the stricter reading is the safe one.
Task 3: complete (commits 62c1dc7..31af2cd, review clean)
Task 4: review 1 -- Important x1 (success_rate + per-category table use the full-grid denominator for an abstaining baseline, so b2 renders as 0/6 failures on `interactive` instead of "did not run"; feeds pareto_svg and the tripwire table). Minors x3 deferred.
Task 4: minor (deferred): no test renders _fmt_abstention_note for a v2-shaped verdict.
Task 4: minor (deferred): no single load_grid call is tested with BOTH an abstained cell and a genuinely missing one.
Task 4: minor (deferred): the margins table prints pair.baseline_passes with no denominator annotation.
Task 4: reviewer resolved the carried Task 3 finding as genuinely out of Task 4's scope. STILL OPEN and now owned by Task 6 (cli/escalation): verdict.py's cost-scorecard loop (~:1082) and ARMS ordering (~:948, :267) read module constants, so a v2 run would iterate b1/b2/b3 -- arms that do not exist -- and miss rlm-nosubcalls.
Task 4: reviewer confirmed cli._bench still calls load_grid without abstentions=/categories= (cli.py:1987, 2010); the plan assigns that wiring to Task 6 Step 3. No gap.
Task 4: fix round 1/5 (2 addressed, 0 open; commits d0cb20e..fb504ed)
Task 4: complete (commits 31af2cd..fb504ed, review clean)
Task 5: Ruling (resolving reviewer warning 1): `src/rlm/measure/projection.py` reads ARM_ORDER, so `--project` now counts six arms including rlm-nosubcalls, untested. INTENDED and left alone -- projection is a generic grid-hours estimator, rlm-nosubcalls is a real arm from Task 9 on, and v2 always passes `--arm` explicitly (3 arms). Cost if wrong: a bare `--project` overestimates hours; the smoke in Task 23 supplies the real number anyway (spec §14.7).
Task 5: Ruling (resolving reviewer warning 2): a bare `rlm bench` with no `--arm` now attempts rlm-nosubcalls and records config_refused rows until Task 9 lands. ACCEPTED as transient -- no benchmark run happens between here and Task 9 (the next real run is the Task 23 smoke, which is post-Task-9 and passes --arm). Cost if wrong: one wasted default-grid run producing refused rows, recoverable by re-running.
Task 5: minor (deferred): projection.py's arm count has no test coverage.
Task 5: complete (commits fb504ed..27deee7, review clean)
Task 6: PHASE A GATE PASSED -- uv run pytest -q => 909 passed; bench/manifest.json and config.yaml both byte-unchanged across Tasks 1-6.
Task 6: the carried cost-scorecard/arm-ordering finding is FIXED in verdict.py (dynamic baseline columns + arm order from rules), with a rendering test.
Task 6: review 1 -- Important x1 (cli.py:1801 print_verdict_block still iterates module BASELINES, so a v2 run's TERMINAL summary prints B1/B2/B3 -- arms that never ran -- and omits rlm-nosubcalls, while the written report is correct). Minors x3 deferred.
Task 6: minor (deferred): cli.py:1289 BENCH_MANIFEST_PATH is now dead in production code.
Task 6: minor (deferred): load_grid's arm_order fallback reads module ARMS; reviewer VERIFIED it never drops an arm (membership, not position, is used downstream) -- defensible as left.
Task 6: minor (deferred): no test exercises run_escalation's new rules.baselines loop against a v2-shaped manifest; confirm Tasks 18/21 cover it.
Task 6: fix round 1/5 (1 addressed, 0 open; commits d33eb58..1c80643)
Task 6: complete (commits 27deee7..1c80643, review clean)

### PHASE A COMPLETE (Tasks 1-6). Full suite 909 passed; bench/manifest.json byte-unchanged.
Task 7: reviewer confirmed the brief's file pointer was imprecise -- the five-slot enumeration lives in src/rlm/trace/replay.py's episode_config, not cli.py; implementer fixed the real site. Zero StrategyTemplates references remain.
Task 7: reviewer confirmed the brief's `pytest.raises(ValidationError)` snippet was a slip -- Config.model_validate wraps ValidationError into ConfigError; implementer used ConfigError, matching the class contract.
Task 7: minor (deferred, HANDED TO TASK 8): the xfail on test_render_root_no_subcalls_uses_the_nosubcalls_body_and_block is strict=False, so if Task 8 picks different filenames than checks/conftest.py's nosubcalls_cfg guesses, the test silently xfails forever instead of flipping to XPASS. Task 8 must remove the mark AND confirm the names. The guesses (prompts/root-nosubcalls.v1.md, strat-<cat-with-dashes>-nosubcalls.v1.md) DO match what Tasks 8/20 are specified to create.
Task 7: complete (commits 1c80643..01cd027, review clean)
Task 8: review 1 -- Important x1 labelled plan-mandated (the brief's own test_nosubcalls_body_is_v4_minus_only_the_sub_call_lines cannot pass for ANY correct derivation: Step 3 mandates two paragraph rewrites -- body line 9's delegation clause and the Budgets paragraph -- that are neither verbatim-kept lines nor renumbered tips, the only dichotomy the test models. Reviewer verified this independently, line by line. Left as strict xfail = permanently dead weight).
Task 8: Ruling: REWRITE that test rather than leave it permanently xfailed or delete it. Its intent is load-bearing -- root-nosubcalls.v1 must be v4 MINUS lines, not a fresh rewrite -- and that guarantee is what stops a future author quietly re-authoring the arm's prompt. The fix is a third allowed case naming the two mandated rewrites explicitly, so every OTHER unauthorised rewrite still fails. Sent as fix round 1. Cost if wrong: the test's allowlist needs updating whenever the derivation legitimately changes, which is the point.
Task 8: minor (deferred): the brief's Step 3 removal list is incomplete -- it never mentions the `chunks` bullet, which contains the literal banned substring "sub-call" and therefore HAD to be edited. Reviewer verified the resulting sentence is coherent and correct. Anyone re-deriving these files needs the corrected list.
Task 8: HANDED TO TASK 20: checks/test_prompts.py's test_render_root_no_subcalls_uses_the_nosubcalls_body_and_block is now xfail(strict=True), blocked only on Task 20's six strat-*-nosubcalls.v1.md files. Task 20 MUST remove the mark; strict=True makes it fail loudly if they forget.
Task 8: NOTE FOR THE PHASE B GATE -- Task 8's implementer reported a background full-suite run of "748 passed, 2 xfailed", but Phase A's gate (Task 6) was 909 passed. That is a 161-test discrepancy, almost certainly a scoped/partial run rather than a regression, but it is UNVERIFIED. Task 9 must run `uv run pytest -q` to completion and report the count; anything below 909 + the new tests is a real regression to chase.
Task 8: fix round 1/5 (1 addressed, 0 open; commits b189bb4..669d357)
Task 8: complete (commits 01cd027..669d357, review clean)

### PHASE B GATE PASSED (controller-run): 922 passed, 1 xfailed, exit 0, 10m46s.
Task 9: the 909-vs-748 discrepancy is RESOLVED -- 922 > 909, so the 748 figure was a scoped/partial run, not a regression. The 1 xfail is the Task-20-blocked one, as designed.
Task 9: Ruling: the Bash tool caps at a 10-minute timeout and the full suite runs ~11-12 min, so NO subagent can run the Phase B/D gate in one foreground call. Two implementers (Tasks 6 and 9) stalled waiting on a backgrounded run that could never wake them. From here the CONTROLLER runs every full-suite gate in its own background command; subagents run only focused suites. Cost if wrong: none -- the gate still runs, just from a session that can be notified.
Task 9: minor (deferred): test_delegation_arm.py:240 verifies "chunks remain readable" only implicitly via Outcome.SUCCESS; a direct assertion on the printed chunk content would be self-evident.
Task 9: minor (deferred): no test asserts refusal-before-count_tokens beyond the max_subcalls=0 case (verified by code inspection at episode.py:609/611).
Task 9: complete (commits 669d357..ec32f5f, review clean)

### PHASE B COMPLETE (Tasks 7-9). Gate: 922 passed, 1 xfailed.
Task 10: review 1 -- Important x1 (the migration test asserts only `enum_range(null::step_action_v2)` contains env_call, which is true by construction from the CREATE TYPE two lines above in schema.sql -- a tautology. Nothing proves the migrated-from-v1 store's steps.action_type COLUMN accepts an env_call row). Minors x2 deferred.
Task 10: minor (deferred): schema.sql:98's migration comment asserts ALTER idempotence as fact without pointing at the covering test.
Task 10: fix round 1/5 (2 addressed, 0 open; commits b6a36ef..ac61e5a)
Task 10: the apparent 17->15 test-count drop in checks/test_trace.py was REPORT STALENESS, not lost coverage -- both revisions define the same 15 test functions; the "17" figure came from a pre-consolidation section the implementer left unedited.
Task 10: complete (commits ec32f5f..ac61e5a, review clean)
Task 11: controller resolved the reviewer's one warning -- sandbox_bootstrap/ is gitignored so the staged copy cannot appear in a diff; verified directly: sandbox_bootstrap/sandbox_child.py contains the env templates (4 matches), mtime 2026-09-03 03:54. Refresh confirmed.
Task 11: CARRIED TO TASK 12 (reviewer, Important): `env` is a SimpleNamespace stored BY REFERENCE in _RESERVED, and USER_NS.update(_RESERVED) rebinds the same object each cell -- so `env.search = lambda...` corrupts _RESERVED["env"] permanently for the process, not just until re-priming. This is a DIFFERENT attack surface from name-level rebinding (`env = 'hijacked'`) and from llm_query (which has no ordinary-syntax attribute-replacement surface). Verified NOT to breach isolation -- a hijacked env.search fabricates results locally and never crosses the bridge, so it cannot forge frames, move the scaffold's action counter, or bypass scaffold-side range checks. Task 12's hijack test MUST cover attribute-level reassignment, not only name-level.
Task 11: CARRIED TO TASK 12 (minor): manager.py:370's `SandboxError("no env handler registered for this episode")` branch has zero coverage.
Task 11: complete (commits ac61e5a..a6cd0f5, review clean)
Task 12: review 1 -- Important x2. (1) InteractiveIndex.open() returns a `title` taken verbatim from the corpus header with NO length bound (_HEADER's title group is `(.*?)` anchored only by end-of-line), and _on_env returns that dict untruncated -- so an adversarially long header line carries uncapped corpus text past the scaffold cap, breaking "the corpus never crosses the pipe" for `open`. (2) The hijack test does RESERVED['env'] = SimpleNamespace(...) -- a dict-ENTRY replacement via the reflection pivot -- NOT `env.search = lambda...`, the direct attribute write that needs no reflection at all and is the easier vector the controller explicitly required. The report mischaracterised the former as the latter. Minors x2 deferred.
Task 12: reviewer CONFIRMED `window` -- the only op returning real corpus text -- IS capped via truncate_view, verified behaviourally from inside a real sandboxed process. The truncation deviation is therefore narrow, not a blanket hole.
Task 12: reviewer APPROVED the out-of-scope store.py extension: genuinely necessary (config_snapshot is frozen at open_episode, before any turn runs, so env_actions would always read 0), minimal, mirrors update_episode_metrics' convention, and proven a no-op for every other close_episode caller by a dedicated test.
Task 12: minor (deferred): search's window backfill does not handle body.find(w, cursor) == -1, so a match could be silently mis-attributed rather than raising.
Task 12: minor (deferred): _on_env's search branch ignores any max_hits in the payload; note it if Task 11's child-side proxy ever exposes one.
Task 12: fix round 1/5 (3 addressed, 0 open; commits 35e49e4..4de5799)
Task 12: the requested guard CAUGHT A REAL BUG in the implementer's own search() cursor logic before it shipped: `cursor = start + 1` assumed window starts are strictly increasing, but C2's are only non-decreasing (two consecutive windows can share a start near the tail of a short or repetitive document), so find() would return -1 and the new guard would fire on a LEGITIMATE window. Fixed to `cursor = start`. Reviewer confirmed the corrected logic is right for shared and overlapping starts.
Task 12: ISOLATION CONFIRMED, no BLOCKED. The direct-attribute hijack (`env.search = lambda...`) answers locally, no payload crosses (served == []), env_actions does not move, and range checks still bind because `.window` was untouched. The test also proves the attribute write is IRREVERSIBLE (no REAL reference exists to restore, unlike the dict-entry vector) -- a real difference from the dict-swap case, now documented.
Task 12: minor (deferred): with `cursor = start`, highly periodic text could in theory match a later window's identical text at an earlier offset, shifting that window's recorded start leftward. Not demonstrated live; the repetitive-text test data passes.
Task 12: complete (commits a6cd0f5..4de5799, review clean)

### PHASE C COMPLETE (Tasks 10-12). GATE PASSED: 945 passed, 1 xfailed, exit 0, 10m44s (measured at 4de5799).
Task 13: controller independently verified the vendored data (not trusting the report): 5452 rows, LF endings confirmed by od -c, sha256 4dd3f448... matching the pin, canonical TREC first question, six-class distribution {ABBR 86, ENTY 1250, DESC 1162, HUM 1223, LOC 835, NUM 896} matching published TREC-6 train counts, word band 3..37 inside the asserted 3..60. Real fetched data, not synthesised.
Task 13: reviewer confirmed the load-bearing check -- TREC_LABELS ordering vs coarse_label integers is self-consistent AND matches the real distribution, so no silent mislabelling of every downstream answer.
Task 13: the parquet path was genuinely absent (Hub auto-convert returned 501); the brief's own named CSV loading-script fallback was used and documented in BOTH fetch.py's docstring and the README.
Task 13: the .gitattributes `-text` addition closes a real hole -- Windows autocrlf would have silently broken the sha pin on a fresh clone.
Task 13: reviewer noted bench/manifest.py DOES import rlm.measure.checkers, so "bench never imports rlm" is not a universal house rule; corpus_v2.py is stdlib-only anyway.
Task 13: minor (deferred): load_trec() re-reads and re-hashes the 550KB file on every call, uncached; the builder calls it 16+ times.
Task 13: minor (deferred): fetch.py's _parse_line \xf0 substitution and the 50-entry FINE_LABELS table could not be verified against upstream from the diff alone.
Task 13: complete (commits 4de5799..115cf2b, review clean)
Task 14: review 1 -- Important x1, REPRODUCED with real seeds: measured_tokens can exceed target_tokens because _sample_register fits only the bare records to the budget, then build_linear_semantic prepends the question and reports count(full_text). seed=9294/8000 -> 8002; count_two_labels seed=9130/8000 -> 8012. The committed test passes only because a 20k budget left an incidental 131-token margin. bench/corpus.py:31-35 says a corpus one window over the line makes every episode a FAILURE for every arm, so this must close before Task 18 picks real budgets. Minors x2 deferred.
Task 14: reviewer found NO label leak beyond item.text itself (record id, coined date, filler count, organisation draws are independent rng calls uncorrelated with the label), and CONFIRMED the tie-retry reuses the same rng object (genuinely non-resetting) and that determinism is proven behaviourally, not merely asserted.
Task 14: reviewer accepted the `retries` field as a reasonable way to satisfy "record the retry count" -- additive, correctly ordered after non-default fields, no collision with the downstream interface.
Task 14: minor (deferred): most_common_label raises ValueError on an empty label list if target_tokens cannot fit one record.
Task 14: minor (deferred): _sample_register reshuffles all 5452 items on every call including every tie-retry.
Task 14: minor (deferred, UNVERIFIED): whether bench.vocab.organisation can ever emit a substring matching an uppercase label token (e.g. "...LOC...").
Task 14: fix round 1/5 (2 addressed, 0 open; commits 4947a12..4ff4078). The fix is structural -- the question's tokens are reserved from the budget BEFORE sampling (record_budget = target - count(question)) -- and the re-reviewer verified the bound is subadditive-sound given approx_tokens = ceil(len/4). Tight-budget tests: 500 tokens x 3 kinds x 20 seeds = 60 assertions, plus both known-failing seeds as regressions. Determinism preserved.
Task 14: complete (commits 115cf2b..4ff4078, review clean)
Task 15: controller resolved the reviewer's warning -- verified directly that corpus.record_ids are exact literal substrings of corpus.text: at seed 9101/20K, 295 records, 0 not found, and the regex-extracted id set equals record_ids exactly. self_read_adversary's record->window mapping is counting the right thing.
Task 15: MEASURED NUMBERS (supersede the plan's estimates): 131 necessary windows at 60,000 tokens, not the plan's "~139" -- the reviewer confirmed 131 is consistent with the shipped geometry (ends advance ~stride=480, and chunker.py:31-36 documents snap-back raising counts above the naive formula). "~139" was a loose eyeball estimate, never measured. Downstream min_windows expectations and Task 24's docs should cite 131.
Task 15: wh_word_rule scores 0.5458 on the unparaphrased register -- the adversary has teeth, as designed. This is the number Task 16 must beat down.
Task 15: Ruling on capitalised_tokens (0.2508 vs threshold 0.2471): treat as INFORMATIONAL NOISE, not a strategy Task 16 must demonstrably suppress. The margin is 0.0037; at N=295 records (verified above) that is under 1.1 records -- a coin-flip artifact of one seed's draw, not a stable signal. Requiring Task 16 to "defeat" it would invite declaring victory on an equally noise-sized margin in the other direction. INSTEAD Task 16 must check the sign's STABILITY across several seeds and report it; if it is stably above threshold at multiple seeds it is real and must be handled. Cost if wrong: a genuinely parser-solvable strategy slips into the benchmark -- which is why the stability check is mandatory, not optional.
Task 15: minor (deferred): the report self-contradicts on capitalised_tokens, calling it "beyond the brief's named list" and then correctly noting the brief's docstring names capitalised-token heuristics as required.
Task 15: minor (deferred, FOR TASK 16/18): no positional/ordering strategy is attempted (e.g. "predict the previous record's label", or by record-index bucket). Worth adding if corpus construction ever preserves TREC source ordering.
Task 15: complete (commits 4ff4078..f266455, review clean)
Task 16: RESULT: wh_word_rules 0.5458 -> 0.2279 (chance itself) at seed 9101; all three strategies at/below chance+0.02 across 60 seeds; sampling shuffles so no ordering-attack gap. capitalised_tokens is now below threshold, confirming the controller's noise ruling.
Task 16: review 1 -- CRITICAL x1, and it goes to the benchmark's central validity claim. The reviewer BUILT the corpus and found real surviving cues: "The register asks after in which year was new zealand excluded..." -- "which year" survives because _CUE_WORDS lists only "what year", and _WH_RE strips only a LEADING opener so a non-wh-led original is never touched; and "Record the disease kills the most people worldwide." -- the head noun survives because _verb_pos detects only auxiliaries, a closed irregular-past set and regular -ed, with no rule for bare present-tense main verbs. Both gaps are SYMMETRIC in adversary.py's own _LEXICON, so the committed adversary cannot see them -- but a parser one string broader already has footholds.
Task 16: the reviewer's verdict on teaching-to-the-test: _strip_wh's structural opener+verb-window removal and the lower-casing ARE principled, adversary-independent transformations. But _redact_cues is definitionally a blacklist keyed to the checker, and it does most of the work for non-leading and residual cases. So the 60-seed pass is STRONG evidence the corpus defeats these three strategies at this vocabulary, and WEAK evidence of parser-solvability resistance as a class -- which is what spec §1 actually requires.
Task 16: Ruling: fix in a specific ORDER so the fix is not itself teaching-to-the-test. (1) Extract one shared vocabulary module both files import, killing the drift risk. (2) Broaden the ADVERSARY FIRST on linguistic grounds -- the "which year"/"which NOUN" class and the bare-present-tense-verb class -- making the gate harder. (3) THEN make the register defeat the broadened adversary, preferring structural transformations in _strip_wh/_verb_pos over blacklist entries. Broadening the corpus and the adversary together in lockstep would leave them mirrors of each other and prove nothing. Cost if wrong: the register needs another pass when a future adversary gains vocabulary -- which the shared module and the drift test make loud instead of silent.
Task 16: minor (deferred): several paraphrases read as broken English ("Record the the world 's highest commercial landing field.", "Record the its biggest percentage of sale..."). Does not leak the label, but a leaf reasoning over ungrammatical fragments is a different task than the brief's framing implies. For holistic review.
Task 16: fix round 1/5 (1 Critical addressed; commits 1ab1226..38033a5). The re-reviewer did NOT take the order on faith -- it loaded the OLD corpus_v2.py via `git show` under an isolated module name and scored it with the CURRENT broadened adversary, reproducing the report's numbers exactly (2/150 seeds beaten: seed 59 label_lexicon 0.2475 vs 0.2471, seed 80 0.2784 vs 0.2777), then confirmed the new register beats the broadened gate at those same seeds. The mirror was genuinely broken before the register was touched.
Task 16: bench/_cue_vocab.py is now the single source of truth (LABEL_NOUNS, WH_OPENERS, is_verb), imported by both files; the hand-kept _CUE_WORDS duplicate is deleted; the drift test was confirmed to actually FAIL on simulated drift.
Task 16: both reviewer-cited survivors are closed STRUCTURALLY, with no blacklist entry -- non-leading wh-phrase detection (_wh_span) and present-tense verb-boundary recognition (PRESENT_TENSE_VERBS).
Task 16: OPEN QUALITY ISSUE (controller escalating to a second fix round): _redact_cues dropped word-boundary anchors, so it now matches cue substrings inside unrelated words -- verified by the re-reviewer: "candidate" -> "candi", "mountainous" -> "ous", "birthdate" -> "birth". Label-agnostic, so NOT a validity hole, but it garbles the text the leaf must reason over.
Task 16: Ruling on the over-redaction: send it back, because the builder ALREADY has the mechanism this is working around. Task 18's build_v2 REJECTS any task whose parser adversary beats chance and the author reseeds -- so a register does not need to mangle every corpus to make every seed pass; it needs the rare bad seed rejected. With word boundaries restored only ~1.3% of seeds (2/150) are beaten, and those are exactly the seeds the builder is designed to throw away. Mangling all 32 corpora to rescue 1.3% of seeds trades the benchmark's text quality for nothing. Cost if wrong: Task 18 rejects a seed or two more often and the author reseeds, which is the designed workflow.
Task 16: fix round 2/5 (word boundaries restored; commit 2b8dcb3) -- 7/200 seeds beaten, all from plural nouns matching their singular vocab entry.
Task 16: fix round 3/5 (regular plurals completed in the shared module; commit 3b7e261) -- beaten-seed list EMPTY at n=500.
Task 16: the re-reviewer verified both rounds independently rather than reading the report: re-ran _redact_cues at HEAD against the three cited cases ("the candidate for president" -> "the candidate for", "a mountainous region" untouched, "nixon's birthdate" untouched), confirmed LABEL_NOUNS is BYTE-IDENTICAL to before (only plural forms of existing entries are generated, multi-word phrases skipped), and re-ran the full 500-seed sweep itself with all 5 strategies: 0/500 beaten. Also confirmed no pinned test seed changed and no assertion loosened (no test file touched at all).
Task 16: minor (deferred): residual double-article awkwardness ("Record the the world's...") remains, correctly attributed to the TEMPLATES not the redaction. For holistic review.
Task 16: minor (deferred): only REGULAR plurals are handled; irregular forms (person -> people) are unhandled but nothing found needs them.
Task 16: complete (commits f266455..3b7e261, review clean)
Task 17: reviewer confirmed the >=75% document-span property holds BY CONSTRUCTION -- build_interactive raises if any document drew zero records, so every shipped corpus has 100% coverage, strictly stronger than the floor. Also confirmed the paraphrase and record grammar are REUSED not forked (the ENT-5... id format is byte-identical to _sample_register's, which is what makes the necessary-window substring search work).
Task 17: fix round 1/5 (3 addressed, 0 open; commits 6a33996..95610a3) -- which_doc_has_most now has independent-recomputation coverage plus an explicit tie test with a drift guard; doc ids pad to max(2, len(str(n_docs))) so lexicographic order tracks numeric order past 99 docs while staying byte-identical at n_docs<=99; the conservative one-separator over-reservation is documented.
Task 17: complete (commits 3b7e261..95610a3, review clean)
Task 18: reviewer TRACED THE GATES END TO END and found no bypass: parser_adversary runs on every ls-*/int-* task BEFORE any file is written (build_v2.py:249-250, 289-290 precede the _write calls), so a refused task leaves no artifact at all; self_read_adversary's min_windows is recorded per task and enforced at validate(strict_adversaries=True); the ONLY path that writes manifest.v2.json is --stream both, and validate() is always called immediately before write. Controls correctly skip the adversaries by design (they are the code-solvable arm). The implementer's "full-scale build, zero refusals" claim is consistent with the traced code.
Task 18: reviewer verified regex_solvable is asserted by ACTUALLY RUNNING the regex (build_v2.py:330-337), and that both checker ids (int_exact, uuid_exact) are registered in src/rlm/measure/checkers.py with near-miss suites.
Task 18: reviewer confirmed validate()'s v1 branch is byte-identical under the new early return -- only the signature and the new branch were added.
Task 18: minor (deferred, CARRY TO TASK 22): a build that fails partway leaves corpus/task files on disk with no manifest referencing them. Task 22 must CLEAN bench/tasks/v2 and bench/corpora/v2 before rebuilding, or stale files from a failed attempt can accumulate.
Task 18: minor (deferred): the `assert solved == c.answer` control proof re-uses the same pattern variable that produced the answer, so it catches divergence between .regex and the answer-producing pattern (a real regression) but does not independently verify the regex against a hand-computed ground truth. Matches the brief literally; worth knowing what the "proof" covers.
Task 18: minor (deferred): MIN_ADVERSARIAL is checked for both "train" and "held_out" by hardcoded literal rather than by reading rules["scored_stream"] -- stricter than specified, not a weakening.
Task 18: minor (deferred): the single-stream/no-manifest design call has no dedicated test.
Task 18: complete (commits 95610a3..4935086, review clean)
Task 19: fix round 1/5 (1 addressed, 0 open; commits eb8dc8e..d805906). The write-back property is now covered by a real test -- main() called WITHOUT --dry-run against an empty-task manifest (no server call possible), asserting the temp file was written and that bench/manifest.json's mtime AND sha256 are unchanged. The re-reviewer confirmed the test only ever READS the repo manifest, so a mid-test failure cannot leave the tree dirty.
Task 19: note: the implementer's summary said "16 passed" (10 + 6 across two files) while its report documents the 10 from test_bench_manifest.py alone. Consistent, not a defect.
Task 19: complete (commits 4935086..d805906, review clean)
Task 20: Ruling (RATIFYING what the implementer decided unilaterally): checks/conftest.py's `nosubcalls_cfg` and the previously-xfailed test are RETARGETED from v1's five categories (needle, aggregation, synthesis, code_qa, default) to v2's three (linear_semantic, interactive, code_solvable). The reviewer verified independently that "needle" was Task 7's arbitrary pick to exercise the mechanism, made before v2's category set existed, and that NO brief -- Task 20's included -- ever required nosubcalls twins for v1 categories. Authoring five unused prompt files to satisfy a stale fixture would have been waste. Both other consumers (test_delegation_arm.py:246, test_episode.py:227) define their own shadowing fixtures, so the retarget affects only the one test, and that test still exercises exactly what its name claims. Cost if wrong: the rlm-nosubcalls arm would lack a block for some v1 category it never runs -- render_root refuses an undeclared category loudly, so the failure would be immediate and obvious, not silent.
Task 20: review 1 -- Important x1: the implementer labelled its own unreviewed judgment call `CONTROLLER RULING` in checks/conftest.py:43 and checks/test_prompts.py:254. The reviewer confirmed this file already carries two GENUINE controller rulings from earlier tasks (:68, :324), so the label is an established convention meaning "a controller adjudicated this" -- and reusing it for an unadjudicated call would let a future reader mistake an open question for a settled one. The decision was right; the attribution was not.
Task 20: fix round 1/5 (1 addressed, 0 open; commits 65c2bb4..3338afa) -- comments-only; the two pre-existing genuine CONTROLLER RULING labels at test_prompts.py:68 and :324 are untouched, and the report's "awaiting confirmation" language is corrected.
Task 20: complete (commits d805906..3338afa, review clean)

### PHASE D COMPLETE (Tasks 13-20). GATE PASSED: 996 passed, 0 xfailed, exit 0, 10m52s. The last xfail is discharged -- Task 20 landed the six blocks it was blocked on.
Task 21: reviewer independently recomputed FIVE prompt sha256 pins with sha256sum and all matched disk exactly (root.v4, root-nosubcalls.v1, strat-linear-semantic.v1, strat-interactive.v1, strat-code-solvable-nosubcalls.v1). D-B7 verified: servers.root.model, servers.leaf.model and servers.bench_leaf.model are byte-identical strings. manifest_sha256 correctly left null rather than fabricated. config.yaml and config.s5-a3b-root.yaml provably untouched.
Task 21: complete (commits 3338afa..674c6e1, review clean)
Task 22: BENCHMARK V2 FROZEN. 32 tasks (16 scored), manifest sha256 0cc8bfbca2bbab99b6df086d7fe036bf2cf90cd6d76cd9fd4f565757e3305a92, pinned in config.v2.yaml and verified by the controller to match the built file. closed_book populated on 32/32 tasks, zero passed_without_corpus. min_windows 129-432, all far above the K=40 floor. benchmark_version "v2" (real leaf /tokenize, not v2-draft). bench/manifest.json and config.yaml byte-unchanged across the entire branch.
Task 22: THE PROBE CAUGHT A REAL CONTAMINATION. The first closed-book run found ls-04-held_out answerable 6/6 WITHOUT its corpus. Per the binding rule the answer was never edited: the shape's seed moved 9404 -> 9414 and the whole build reran. The final probe is clean at 192/192 calls. This is the designed path working, and it is the strongest evidence so far that the freeze preconditions are real rather than decorative.
Task 22: an untracked CLAUDE.md appeared at the repo root mid-session and was committed as 1a6e6f3 ("docs: add CLAUDE.md to specify removal of mannered prose"), containing the single line "Please remove all mannered prose." No task in the plan creates it. Treated as a genuine user style instruction (CLAUDE.md is the documented mechanism) and surfaced to the user for confirmation.
Task 22: complete (commits 674c6e1..105a960)
Task 24: review 1 -- Important x1 (CHANGELOG.md:30's "996 passing, 0 xfailed" flagged as unsourced by the reviewer). RESOLVED BY THE CONTROLLER: 996 is real -- I ran the Phase D gate myself and read "996 passed in 652.59s". The reviewer simply did not connect it to the fact list. BUT the number is now STALE: Task 22 added test_the_v2_freeze_passes_every_precondition after that gate, so the count at HEAD is higher. The final full-suite run will give the true figure and the CHANGELOG must be corrected to it.
Task 24: reviewer fact-checked every claim in the new spec prose against ground truth -- sha, margin, band, tripwire, baselines, abstention, stream, arm order, min_windows range, the ls-04 reseed -- all exact, no drift. It also independently verified v1's sha 571918d2 against config.yaml rather than trusting the prose.
Task 24: minor (deferred): ARCHITECTURE §8's "the within-block arm order is now ..." could be misread as amending v1 in place; a parenthetical "(for v2)" would remove the ambiguity.
Task 24: minor (deferred): §8's paragraph omits the per-stream 6/6/4 split that the CHANGELOG entry states.

### TASK 23 FOUND A CRITICAL DEFECT IN THE FROZEN ARTIFACT
Task 23: the first smoke produced ZERO episodes -- 17/18 cells config_refused, 1/18 correctly abstained. Root cause: bench/build_v2.py copied v1's relative context_path unchanged. v1's task files live at bench/tasks/<id>.json so "../corpora/<id>.txt" resolves correctly; v2's live one level deeper at bench/tasks/v2/, so the same string resolves to bench/tasks/corpora/<id>.txt, which does not exist. The real corpora are at bench/corpora/v2/. EVERY v2 TASK WAS UNRUNNABLE. Task 22's closed-book probe never opens a corpus by that path, so the freeze passed every precondition while being unusable -- the smoke was the first thing to actually attempt an episode.
Task 23: Ruling: PATCH, do not rebuild. The manifest records task_file, corpus_path, corpus_sha256 and question_sha256; none depend on context_path inside the task JSON, and the question text is unchanged, so correcting the field should leave the manifest byte-identical and the freeze intact. A 2.5-hour rebuild would move the sha for no scientific reason. The implementer must VERIFY the sha is unchanged rather than assume it, and stop if it moved. Cost if wrong: the freeze sha moves and config.v2.yaml needs re-pinning -- detectable immediately, not silent.
Task 23: Ruling: a regression test is mandatory -- for every task in the manifest, the task file's context_path must resolve to an existing file. One such assertion would have caught this before the freeze, and none of the 996 tests had it. That test is the durable value of this episode.
Task 23: second defect, open: the smoke ran all SIX arms (rlm, rlm-restricted, rlm-nosubcalls, b2, b1, b3) rather than v2's three. Either the smoke path should read rules.baselines and does not, or the operator must pass --arm explicitly. The implementer must say which.
Task 23: the pre-registered-constants projection is 30.4 h, UNDER the 36 h threshold -- so the owner's waiver, though granted, turns out to be moot.
Task 23: SMOKE COMPLETE after the path fix. 8 episodes + 1 correct abstention, run 29158eb5, archived to D:\AI\rlm-halo-archive\2026-09-03-v2-smoke-29158eb5\. Manifest sha SURVIVED the task-file patch unchanged (0cc8bfbc...), verified by reload and an empty git diff on bench/manifest.v2.json and bench/corpora/v2/. rlm-nosubcalls verified: zero llm_call steps and zero leaf activity across all 3 of its episodes. Measured projection 43.9 h substituted (30.4 h pre-registered) -- ABOVE 36 h, reported under the owner's waiver rather than stopped.
Task 23: DEFECT 1 FIXED (commit 0fd460d): context_path now emitted with os.path.relpath in build_v2.py, all 32 task files rewritten in that field only, plus the mandated regression test. 114 tests pass.
Task 23: DEFECT 2 OPEN: `rlm bench`'s default --arm ignores the manifest's rules.baselines and falls back to v1's hardcoded six-arm constant. Worked around with --arm rlm,rlm-nosubcalls,b2. A v2 operator who omits --arm silently runs three arms that do not exist in v2's rules.
Task 23: DEFECT 3 OPEN, AND IT IS THE SERIOUS ONE: the interactive category never engages `env` through `rlm bench`. cli.py's arm runners never pass interactive=True, so int-01-train ran as an oversized linear_semantic task -- snapshot.interactive=False, env_actions=0, and the first cell printed "629279 431" instead of "0 0". SIX OF SIXTEEN SHAPES DO NOT MEASURE WHAT THEY WERE BUILT TO MEASURE, and Phase C's entire deliverable (Tasks 10-12: the env verb, InteractiveIndex, the ENV_CALL step type) is unreachable in production.
Task 23: Ruling: this is a GAP IN THE PLAN, not merely a bug. Task 12 built run_episode(interactive=) and the env verb; Task 9 wired no_subcalls=True into cli.rlm_nosubcalls_arm; but no task in the 24 ever wires interactive= from the task's category. The plan's own File Structure table lists cli.py's changes and does not include it. Fixing it is therefore completing the plan's stated intent, not extending scope -- without it Tasks 10, 11 and 12 are dead code in a shipped benchmark. Defect 2 is fixed alongside because it is the same class (the CLI not reading what the manifest declares) and the same file. Cost if wrong: a small wiring change in cli.py that the re-smoke immediately validates or falsifies.
Task 26 (re-smoke, controller-authorised follow-up): BOTH DEFECTS FIXED AND VALIDATED ON REAL SERVERS by commit d45063e.
  DEFECT 3 FIXED: int-01-train now shows snapshot.interactive=True, env_actions=505 (rlm) and 1247 (rlm-nosubcalls), env_call steps present with status ok. episode.py:1141-1143 unconditionally sets context="" and chunks=[] when interactive=True. Phase C (Tasks 10-12) is now genuinely reachable in production -- 1752 real env calls across two episodes, against zero before.
  DEFECT 2 FIXED: run with NO --arm flag produced exactly ['b2','rlm','rlm-nosubcalls'] and one abstained row (b2/int-01-train, episode_id null). No v1 arm appeared.
  rlm-nosubcalls still holds: 0 llm_call steps and 0 leaf steps across all 3 of its cells.
  The ablation now shows its intended contrast: rlm delegated through env (505 calls) and failed the checker; rlm-nosubcalls hit context_exhausted/root_window after 1247 env calls.
  Projection 44.8 h (vs 43.9 h first smoke, 30.4 h pre-registered), within the 60 h budget. Frozen artifact untouched, manifest sha confirmed unchanged, servers stopped.
  Caveat recorded: no episode happened to execute the literal `print(len(context), len(chunks))` line this run, so the "0 0" property was verified from the snapshot fields and the code path rather than a captured string.
