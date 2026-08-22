# Pre-S6 cleanup — plan

**Date:** 2026-08-22 · **Status:** **EXECUTED 2026-08-22 — Stages 0–5, plus two structural moves this plan did not contain.**

The two moves: `rlm/` → `src/rlm/`, and `s0/ s1/ s2/ s3/ s4/ upstream/` → `milestones/`. The second **reverses §0.1's verdict below**, which is left in place as the record of a conclusion that was wrong — see §0.1a. Top level went from 14 directories to 8. Everything temporary is now deleted rather than tracked: `sandbox_bootstrap/`, `**/__pycache__/`, `.pytest_cache/`, `.hypothesis/`, `milestones/s3/results/store/`.

Still refused, and still for the reason given: the **331-site `rlm/` prose sweep** (three classes of false positive, one of them a functional break in the sandbox destination strings). Stage 6 (spec-version event) and Stage 7 (six owner decisions) remain open.
**Goal (owner, 2026-08-22):** retire finished milestones · delete what is genuinely dead · navigability for S6. Disk reclaim explicitly **not** a goal.
**Method:** nine read-only agents (seven area inventories, a layout proposal, a destruction audit). Findings marked **[V]** were re-verified in-session; **[R]** are agent-reported and cited but not re-measured.

> **Standing rule for the whole cleanup.** Do not run any episode while the tree is dirty. `rlm/cli.py:244-257` stamps `episodes.scaffold_git_sha` as `<sha>-dirty` whenever `git status --porcelain` is non-empty, and that value cannot be resolved back to a commit. **[R]**

---

## 0. The finding that changes the goal

**The tracked tree is not disorganised.** 640 tracked files; `rlm/` has no dead module (`tests/test_import_rules.py` enforces the C1–C6 layering as a checked invariant); `tests/` has no orphan; `bench/` is 6 source files plus a frozen artifact; 17 of 18 `prompts/` files are config-pinned, replay-pinned, or hash-asserted; `traces/blobs/` has **zero orphans and zero dangling refs** (67,136 files against 67,136 distinct refs) **[R]**.

Two of the three stated goals are therefore mostly unavailable, and saying so is the main output of this inventory.

### 0.1 "Retire finished milestones" — blocked, verified **[V]**

`milestones/s1/` and `milestones/s2/` are **live importable packages that the frozen benchmark and the test suite depend on at import time**:

```
bench/build.py:34        from s1.make_fixtures import approx_tokens
bench/build.py:234       from s1.make_fixtures import leaf_counter
bench/vocab.py:80-81     import s1.make_fixtures as s1m ; import s2.make_sweep_fixtures as s2m
tests/test_bench_corpus.py:17, test_distance.py:19-37, test_refusal_ab.py:16-25,
tests/test_s1_fixtures.py:14, test_s2_sweep.py:17-19        (7 test modules)
```

`bench/vocab.py` is reached by `bench/corpus.py`, which is reached by `tests/test_bench_corpus.py` — so moving `milestones/s1/` or `milestones/s2/` makes **`pytest` fail at collection**, not at assert time. Beyond imports: `tests/test_bench_corpus.py:111-112` globs their fixture directories from the repo root; ~53 scripts under `milestones/s2/audit/` hardcode the absolute path and ~60 more depend on nesting depth via `parents[1]`/`parents[2]`, so even an archive that *changes depth* breaks the half that currently works **[R]**.

`ARCHITECTURE.md` cites into the milestone tree ~55 times against 5 citations into `docs/` **[R]**. `docs/` is not this project's documentation tree; the `sN/` directories are, and each record sits beside the runner and JSONL that make it reproducible under I4/I5.

**Verdict at the time: s0–s4 stay where they are.** ~~The navigability problem they cause is real and is solved by a README, not by moving 414 files.~~

### 0.1a That verdict was wrong — the move was done the same day

Every obstacle above is real and every one is *mechanical*. Measuring instead of asserting produced the numbers that made it tractable:

| obstacle | measured | resolution |
|---|---|---|
| 7 external modules `import s1` / `import s2` | 7 | one line: `dev-mode-dirs = [".", "src", "milestones"]` puts `milestones/` on the dev path, so the import statements never change |
| depth anchors `parents[N]` | 54 sites | **41 bumped +1**; the other 13 resolve *inside* `s2/` and survive a move untouched. Resolved each anchor rather than bumping blindly |
| hardcoded `D:\PROJECTS\…` absolute paths | 70 | rewritten. These were **already** welded to this one checkout, so the move added no new fragility |
| repo-relative `s2/…` references | 936 across 220 files | rewritten, then **every resulting `milestones/…` path verified to exist on disk** — a stronger check than the test suite, because the ~113 audit scripts are not covered by tests |
| path-component form `REPO_ROOT / "s4"` | 32 | rewritten separately; the two `tmp_path / "s1"` sites in tests were deliberately spared |

One real defect was introduced and caught: the absolute-path rewrite inserted a **single** backslash where JSON and escaped-string contexts need `\\`, producing `milestones\s1`. Three files, 19 sites, caught by `tests/test_s1_fixtures.py` failing. Fixed.

The honest lesson is not "the obstacles were imaginary." It is that **"blocked" was a claim about cost stated as a claim about possibility**, and the cost was never measured before the verdict was written. The frozen set in §1 is a different matter — those are pinned by sha256 or by 614 episode snapshots, and they genuinely cannot move.

### 0.2 "Delete what's genuinely dead" — nearly empty, and two candidates were wrong **[V]**

- `prompts/leaf-envelope.v1.md` was flagged "no live reference, delete candidate" by one agent. It is referenced at `tests/test_envelope_wiring.py:83, 107, 272`. Deleting it fails the suite.
- `milestones/s2/results/sweep-run1-shared-server.jsonl` looks like contaminated garbage. It is the **known-leaking positive control** ("17 foreign in 54 calls") that validated the leak detector which certified every other run in the milestone clean **[R]**. Deleting it makes every clean verdict in S2 unfalsifiable.
- `milestones/s2/OVERLAP-CONTROLS.md` is the only tracked file nothing references — and it is a *pre-registration* of what overlapping windows do to B2's reducer and B3's BM25 IDF, and S4 shipped B2/B3 **[R]**. Whether its warnings were honoured is recorded nowhere else.

In a repo with this reproducibility discipline, "dead" and "evidence behind a gate verdict" are indistinguishable from the outside. What genuinely is dead: 28 zero-byte tracked `.log` files **[V]**, build caches, and one duplicated binary blob (§2 Stage 1).

### 0.3 "Navigability for S6" — real, and the whole prize

| Problem | Measure |
|---|---|
| **No README.md at all** | Confirmed absent **[V]**. A person or subagent landing here cannot tell which directories are live code, frozen evidence, machine-bound one-offs, or gitignored-but-load-bearing — `tools/`, `sandbox_bootstrap/` and `traces/` are all three at once |
| `ARCHITECTURE.md` §14 changelog | 65,123 B = **30.6% of the 212 KB spec**, a near-complete second copy of the body (`31/32 correct serial` appears at 11 separate line numbers) **[R]** |
| `ARCHITECTURE.md` navigation | **16 h2 headings, zero h3**; 91 KB (43%) in 35 single lines over 1,500 chars — so ~55 inbound citations have no anchor finer than a whole section **[R]** |
| Appendix A | A "config.yaml sketch" diverging from the real file in **14 places** including the wrong root model and a leaf `parallel` count off by 16× — and `plans/2026-08-13-capa1-scaffold.md:11` instructs implementers to read it first **[R]** |
| §4 topology table (`:63`) | Still names the root as Qwen3.6-27B at `--parallel 8` / 40K per slot; ships as Qwen3.8-27B + DFlash2 at `parallel: 128` / 2,560. The correction sits three lines *below* the table **[R]** |
| §5:161 | Lists **four** benchmark arms; five ship **[R]** |
| `docs/` convention | Three coherent genres exist (`research/` = evidence, `specs/` = design, `plans/` = execution) and nothing states it. Two of five "plans" are not plans; **171 checkboxes across three executed plans are all unchecked** **[R]** |

---

## 1. What must not move (the frozen set)

Abbreviated from the full audit; each row is a thing whose move or edit breaks something.

| Frozen | Why | Error surfaced |
|---|---|---|
| `prompts/**` — all 18 | 614 snapshots store the path; `rlm/cli.py:769` rebuilds the registry from the **snapshot's** path. Hashes cover **content only**, so a move of a byte-identical file still breaks it | `ConfigError: prompt … could not be read` → exit 3. An edit gives `PromptDrift` → exit 3. `cp` is safe; `mv` is not |
| `bench/manifest.json`, `bench/tasks/`, `bench/corpora/` | Repo-relative path strings live *inside* the JSON whose sha256 is `benchmark.manifest_sha256` (`config.yaml:584`). **No move is both correct and pin-preserving** | "the frozen benchmark moved…" (`rlm/bench.py:181-186`) |
| `rlm/dispatcher.py`, `budget.py`, `serverproc.py`, `trace.py` — the **names** | These four paths are the literal ground-truth answers of 7 of 30 frozen codeqa tasks | **No error at all.** Silent scoring corruption — the one hazard with zero failure signal |
| `rlm/` at exactly one level below root | `rlm/bench.py:63`, `rlm/cli.py:113` define `REPO_ROOT = parents[1]`. A `src/` layout silently repoints it | `FileNotFoundError` under a directory that does not exist |
| `milestones/s1/`, `milestones/s2/` as top-level packages | §0.1 **[V]** | `pytest` fails at collection |
| `milestones/s4/RESULTS.md`, `milestones/s4/results/ledger.jsonl` | Hardcoded `DEFAULT_REPORT_PATH` / `LEDGER_PATH`; two deliberate tripwire tests. Worse without them: `rlm bench` silently starts a fresh ledger instead of resuming | `test_bench.py:884`, `test_cli.py:1316` |
| `traces/` (db + blobs together) | 67,136 refs are episode-relative paths resolved against `blob_root`; sole episode truth under I4 | All 67,136 break at once |
| `sandbox_bootstrap/` the **directory** | Carries a one-time `icacls /grant *S-1-15-2-1` ACE; absolute path at `config.yaml:498` | AppContainer child cannot read its own script, no obvious cause |
| `.gitattributes` | `*.md text eol=lf` + `bench/corpora/** -text` are what make every content hash reproducible across clones | Breaks 13 prompt pins and 24 corpus hashes on the **next fresh clone** — invisibly on the machine where it changed |
| `tools/llamacpp-vulkan-dflash2/` | `config.yaml:106` root; **no zip exists**; only build that loads the DFlash2 drafter; source is an unmerged PR at `5ecbe1a` that can be force-pushed | Root server will not start |
| The checkout's absolute path `D:\PROJECTS\rlm-halo-framework` | 63+ scripts, both configs, 614 snapshots **[R]** | Silent `FileNotFoundError`s. **This checkout cannot be relocated** — state it in the README |

---

## 2. Execution stages

Every stage leaves the repo working. Run the verification before proceeding.

### Stage 0 — baseline (no changes)

```
git status --porcelain
uv run pytest -q                                  # record the green count
uv run rlm validate --no-server-probe
python -c "from rlm.config import load_config; load_config('config.yaml'); load_config('config-thinkon.yaml')"
```
Pick and run **three replay canaries** before anything moves: one pre-root-swap S1 episode carrying `prompts\root.v1.md`; one carrying `prompts\strat-aggregation.v1.md`; one of the 208 `dflash: true` episodes. All must exit 0.

**Gate:** 614 episodes / 36,031 steps **[V]**, three replays exit 0, pytest green. Write the numbers down.

### Stage 1 — zero-risk deletes (nothing in git)

| item | size |
|---|---|
| `**/__pycache__/` (7 dirs) | 4,037,066 B |
| `.pytest_cache/` | 85,075 B |
| `.hypothesis/` (`examples/` is 284 B — no shrunk counterexamples held) | 222,911 B |
| `tools/.kpack/blas_lib_gfx1151.kpack` — third byte-identical copy (sha256 `67e3109…`), outside every `backend_dir`, unloadable by any launch | 42,891,257 B |
| `tools/llama-b10375-bin-win-vulkan-x64.zip` — verified exact 1:1 with `tools/llamacpp-vulkan/` (52 entries, 0 extra, no graft) and publicly re-downloadable | 34,203,191 B |

**≈ 77.7 MiB.** Note the two **ROCm** zips are *not* redundant the same way — each ROCm build directory carries ~124.7 MB of AMD-wheel graft absent from its zip **[R]**, so a ROCm zip cannot rebuild a pruned ROCm directory. They move to cold store rather than being deleted.

**Verify:** `uv run pytest -q` (regenerates caches; must match Stage 0), `uv run rlm validate --no-server-probe`.

### Stage 2 — cold-store moves (still no git)

`D:/ARCHIVE/rlm-halo-framework/toolchain-zips/` ← the two ROCm zips (393,316,879 B). `…/sdd-2026-08-13-capa1/` ← `.superpowers/` (988,406 B) — already declared scratch by `.gitignore:10`, but it is the only record of the capa1 per-task reports and the review diffs that are provenance for `config.yaml`'s original server block.

**Verify:** `uv run rlm validate` **with** server probe — root and leaf both start; root `build_info` still reports commit `5ecbe1a`.

### Stage 3 — tracked deletions and one move (one commit)

Delete the 28 zero-byte `milestones/s2/logs/*.log` and `milestones/s0/logs/*.log` **[V]** — `llama-server` writes everything to stderr, so every stdout capture is empty and only the `.err` siblings are ever cited. Then `git mv milestones/s2/logs/s1-{leaf,root}.err milestones/s1/logs/` — two S1 artifacts misfiled under S2, and grep across the tree returns zero references to either filename.

**Precheck:** `grep -rn "s1-root\.err\|s1-leaf\.err" --include='*.py' --include='*.md' --include='*.yaml' --include='*.ps1' .` must be empty. **Verify:** pytest green, tree clean.

### Stage 4 — source and config one-liners (one commit)

`pyproject.toml:5` description (pins spec v0.2.2; ships v0.3.16) · `rlm/cli.py:2788-2789` and `rlm/arms.py:532` "all four" → five (text only; validation reads `ARM_ORDER`) · `tests/test_sandbox_manager.py:45` → `REPO_ROOT`-derived (the one absolute machine path in the suite) · `.gitignore` +3 lines (`.venv/`, `.pytest_cache/`, `.hypothesis/` are currently ignored only by tool-authored inner files) · a one-line sync note in each of `milestones/upstream/r13_repro.py` and `milestones/s2/r13_repro.py` (byte-identical duplicates with **independent live citations** — dedup in either direction breaks one).

**Verify:** pytest green, canary 1 replays. None of these renames one of the four codeqa-answer modules.

### Stage 5 — the legibility layer (the actual deliverable)

1. **`README.md`** — one line per top-level directory tagged *live code / frozen artifact / evidence record / machine-bound one-off / gitignored-but-load-bearing*; the three facts that cost the most to rediscover (`milestones/s1/` and `milestones/s2/` are importable packages pytest depends on; `traces/` is gitignored **and** is sole episode truth under I4; `tools/llamacpp-vulkan-dflash2/` has no zip and cannot be rebuilt from a release); the five commands; a "before you move anything" block naming §1's frozen set; and the stated constraint that **this checkout cannot be relocated**.
2. **`docs/README.md`** — the three-genre convention, plus the note that `plans/` checkboxes are not a done/not-done signal (ground truth is §9's gate status lines and the changelog).
3. **`CHANGELOG.md`** — §14's body extracted; §14 becomes a 3-line stub so line 5's amendment rule, line 3's "(changelog: §14)" and the three plan citations still resolve.
4. **Appendix A** → a pointer to `config.yaml` plus the §5 cross-field validator list; heading kept so `plans/2026-08-13-capa1-scaffold.md:11` still resolves.
5. **§4 table, §12 Q1** corrected in place (line 5's amendment rule makes §7 numbers and §4 sizing update-in-place, no version bump).
6. **h3 subheadings + a TOC**, with **no renumbering**.
7. Five **plan status headers** (`LANDED <commit>` / `EXECUTED, S1 gate passed …` / `IMPLEMENTED, see …`).

**Verify:** `grep -c '^## ' ARCHITECTURE.md` must still be 16; every `§14` and `Appendix A` pointer must still resolve; pytest green; all three canaries exit 0. **If any section number moved, revert** — the ~55 citations are the cost.

### Stage 6 — spec-version event (only if §5:161 / §8:355 / the `rlm-restricted` definition are corrected)

These sit inside pre-registered gate text, which line 5's amendment rule makes version-bump-only. §8's own words: *"changing any of this after runs exist is p-hacking."* The change is a **correction of a stale statement, not a revision of the pre-registration**, and the changelog entry must say so and quote what the text said before.

### Stage 7 — owner decisions, one at a time

| # | Decision | Note |
|---|---|---|
| 1 | `tools/llamacpp-rocm-b10488/` — **1,108,556,858 B, the largest single reclaim** | Referenced by **no config**; the only mention anywhere is a *comment* at `config.yaml:89` **[V]**. It is the binary behind the committed `milestones/upstream/r13-b10488-*.jsonl` and `r14-b10488-*.jsonl` bug reports; deleting costs a re-extract **plus** an AMD-wheel re-graft to answer a maintainer question |
| 2 | `config-thinkon.yaml` + `milestones/s1/run_thinking_ab.py` | Loads cleanly, but its root block predates the DFlash2 swap: MTP + `--spec-type draft-mtp` + `llamacpp-vulkan` against `config.yaml`'s DFlash2 + `draft-dflash` + `llamacpp-vulkan-dflash2`, ~8 further deltas. The driver's docstring asserts "differ in exactly one key (verified)" with **no runtime assertion**, and `assert_props` compares only `model_path`/`total_slots`/`n_ctx` — all identical. **A thinking A/B run today would attribute an entire serving-stack difference to one boolean.** Refresh, or retire both and close the question with `milestones/s2/ROOT-THINKING.md`. Either way, add the assertion |
| 3 | `traces/rlm.duckdb` compaction (~203.5 MiB) | 814 of 1,634 blocks free. `VACUUM` is a **no-op** in DuckDB 1.5.5 — it needs `ATTACH` + `COPY FROM DATABASE` + swap, which changes the file identity of the I4 truth. Free blocks get reused, so this is a one-time tidy, not a leak. **Recommend deferring past S6** |
| 4 | `prompts/strat-aggregation.v2.md:3` stale 1,024/768 comment | The loader strips the comment so it never reaches the model; the sha256 covers it anyway. Ship a v3, or accept it. **Do not silently edit v2** — it breaks byte-level comparability with frozen S4 run `1cbafb8f` |
| 5 | `docs/research/2026-08-22-avo-arc-agi-3-dossier.md` + `docs/superpowers/specs/` | The only untracked paths in the tree. Coupled by `[R]` citations — commit together or not at all. Committing the design makes an S6 slice that `ARCHITECTURE.md` §9 marks "UNSCHEDULED. Do not build." look scheduled unless a status header travels with it |
| 6 | `rlm-restricted` undefined in §8 | A fifth arm ships in code, config and a sha-pinned prompt; §8's pre-registration is incomplete for any bare `rlm bench` |

---

## 3. Not proposed, and why

**`milestones/s2/` is not moved, renamed, split or archived** (§0.1) — the single most inviting target and a live package. **`milestones/s2/audit/` is not archived**: 113 re-runnable falsification scripts (~440 KB), collectively cited by `ARCH-LADDER.md:184`, and ~53 hardcode the absolute path so the move breaks them anyway; the `_*_out.txt` beside them are the only written record of the refutation round's verdicts. **`milestones/s2/results/sweep-run1-shared-server.jsonl` and `ub.jsonl` are not deleted** (§0.2 — the second documents a trap; deleting it removes the documentation, not the trap). **Nothing is pruned from `traces/logs/`** — two audit scripts iterate the whole directory. **`milestones/upstream/r13_repro.py` is not deduplicated** against its s2 twin. **The `sN/*.md` measurement records are not moved into `docs/`.** **No `src/` layout.** **The benchmark is not re-frozen** to normalise `bench/manifest.json`'s path strings — no move is both correct and pin-preserving, and §8's comparability rule makes a re-freeze incomparable with the v1 grid behind S4's +30/+13/+29. **The 171 plan checkboxes are not retroactively ticked** — that would fabricate a per-task history. **The absolute-path contamination is not fixed**: 63+ scripts, a project with its own testing burden and no S6 payoff; it belongs in the README as a stated constraint.

---

## 4. Reclaim, for the record

Disk was not a stated goal; these are recorded so the numbers exist. Stage 1: 81,439,500 B. Stage 2 (out of tree): 394,305,285 B. Stage 7 row 1: 911,776,824 B net — the biggest single win. Stage 7 row 3: 213,383,168 B, recommended deferred. **Ceiling 1.49 GiB** of the 3.64 GB untracked today; `tools/` would end at 1.20 GiB, all of it the three live builds.
