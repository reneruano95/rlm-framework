# Benchmark v2 smoke: real episodes at last, and three defects the smoke exists to find

**Status:** COMPLETE (smoke only -- not scored, no verdict). The first attempt today (run
`1fa499f1-fd9a-4943-a49d-142b12765a2a`) was BLOCKED before a single episode ran: every frozen v2
task file had a `context_path` that resolved to a nonexistent directory (Defect 1, below). That was
fixed under a controller ruling scoped narrowly to the path field, verified not to move the freeze's
sha, and re-run. This entry keeps both: the defect that blocked the first attempt, and the real
8-episode-plus-abstention smoke that followed it once the path was fixed.

**Date:** 2026-09-03 · **Config:** `config.v2.yaml` · **Manifest sha:**
`0cc8bfbca2bbab99b6df086d7fe036bf2cf90cd6d76cd9fd4f565757e3305a92` (frozen at `105a960`,
**confirmed unchanged** by the path fix -- see Defect 1). **Real smoke run_id:**
`29158eb5-b18f-4a05-91c0-09e2f39eb613` · **Servers:** root :8080, leaf :8081, `bench_leaf` :8081 profile,
all `Qwen3.6-35B-A3B-UD-Q4_K_M`, llama.cpp `b10375-ba360efe1`, Vulkan. **Archived:**
`D:\AI\rlm-halo-archive\2026-09-03-v2-smoke-29158eb5\` (8 episodes, `bundle-manifest.json`
`config_snapshot_sha256 = ab6c580c312060b875df1b14ee7397cfab5b3ddc955bf9cb375f51c5bf86c931`).

---

## Defect 1 (fixed here) -- every v2 task file pointed at a directory that does not exist

**Symptom, first attempt:** 17 of 18 smoke cells (`--arm` defaulted to all six v1 arms -- see Defect
2) errored `config_refused`; the 18th correctly abstained. Zero episodes, zero rows in
`traces/v2/rlm.duckdb`.

```
ConfigError("context file ...\bench\tasks\corpora\ls-01-train.txt could not be read: ...")
```

**Root cause:** `src/rlm/episode.py:183` resolves a task's `context_path` relative to the task JSON
file's own directory. Every frozen `bench/tasks/v2/*.json` carried
`"context_path": "../corpora/<id>.txt"` -- copied unchanged from v1's `build.py`, correct there
because `bench/tasks/` and `bench/corpora/` are siblings, wrong here because v2 nests both one level
deeper under `v2/`. The real corpus is at `bench/corpora/v2/<id>.txt`; the string resolved to the
nonexistent `bench/tasks/corpora/<id>.txt`. Task 22's freeze never caught this because
`bench/closed_book.py` (the probe that gated it) does not go through this resolution path at all --
this smoke is the first thing in the project's history to load these 32 corpora through a real
episode.

**Fix (controller-authorized, narrow scope: the `context_path` field only, no rebuild):**
1. `bench/build_v2.py` gained a `_context_rel(tasks_dir, corpora_dir, task_id)` helper that computes
   the relative path with `os.path.relpath` instead of a hardcoded literal, so a future rebuild
   reproduces the corrected form for both the real build (`bench/tasks/v2/` -> `bench/corpora/v2/`,
   `"../../corpora/v2/<id>.txt"`) and `--practice` (siblings under `--out`, `"../corpora/<id>.txt"`,
   unchanged). All three call sites (`build_linear_semantic`, `build_interactive`, the code-solvable
   builder) now call it instead of the old f-string.
2. All 32 files in `bench/tasks/v2/` had their `context_path` value rewritten to the corrected form,
   nothing else touched. Verified: every one of the 32 now resolves (`(task_file.parent /
   context_path).resolve().exists()` is `True` for all 32).
3. **Manifest sha verified unchanged**, exactly as the controller's hypothesis predicted (the sha
   covers `bench/manifest.v2.json`, and `corpus_path`/`corpus_sha256`/`question_sha256` there never
   depended on the task file's own `context_path` string):
   ```
   BenchmarkManifest.load('bench/manifest.v2.json').sha256
     == 0cc8bfbca2bbab99b6df086d7fe036bf2cf90cd6d76cd9fd4f565757e3305a92
   ```
   `git diff --stat bench/manifest.v2.json bench/corpora/v2/` is empty; only the 32 task files and
   `bench/build_v2.py` changed.
4. Regression test added: `test_every_v2_task_files_context_path_resolves_to_a_real_corpus` in
   `checks/test_manifest_v2.py` -- for every task in the manifest, asserts the task file exists and
   its `context_path` resolves to a real file. `checks/test_manifest_v2.py checks/test_config.py
   checks/test_episode.py`: 114 passed.

## Defect 2 -- the smoke's default arm set ignores the manifest's declared arms

**Symptom:** the first attempt's grid was "3 task(s) x 1 seed(s) x 6 arm(s)" even though v2's
`rules.baselines` is `["rlm-nosubcalls", "b2"]` (plus `rlm_arm: "rlm"`) -- three arms, not six.

**Diagnosis: this is a scaffold defect, not an operator-workflow gap.** `src/rlm/cli.py`'s `_bench`
computes `rules = _rules_for(manifest)` at the top of the function and then **never consults it** for
the default arm set:

```python
rules = _rules_for(manifest)
arms = tuple(_csv(args.arm) or ARM_ORDER)
```

`ARM_ORDER` (`src/rlm/measure/bench.py:91`) is a module-level constant,
`("rlm", "rlm-restricted", "rlm-nosubcalls", "b2", "b1", "b3")` -- v1's six arms, unconditionally,
regardless of which manifest is loaded. `rules` is used later in the same function (`rules.abstentions`
for scoring), so it is not unused entirely -- it is simply never read for the one thing that decides
what a bare `rlm bench --config config.v2.yaml --smoke` actually runs. `bench/rules.py:31` already
defines exactly the right value (`rules.arms == (rules.rlm_arm, *rules.baselines)`, i.e.
`("rlm", "rlm-nosubcalls", "b2")` for v2) -- it is simply not wired into the default.

**Not fixed here** -- out of this task's authorized scope (the controller's ruling covered the
`context_path` field, the regression test, and diagnosing this one, not patching `cli.py`'s arm
defaulting). **Worked around** for this smoke with an explicit `--arm rlm,rlm-nosubcalls,b2`, which
produced the correct 3 task x 3 arm grid.

## Defect 3 (found during Step 2's checks, not pre-registered) -- the `interactive` category never actually uses `env` when run through `rlm bench`

**This is the load-bearing finding of the real smoke.** The brief's Step 2 pre-registered checks
*assumed* an interactive episode would show `env_call` steps, `snapshot.env_actions` recorded, and
`print(len(context), len(chunks))` == `"0 0"` in the first cell. **None of that happened.** From the
real `int-01-train` episodes (`traces/v2/rlm.duckdb`, `config_snapshot`):

```
rlm            / int-01-train: snapshot.interactive = False, snapshot.env_actions = 0
rlm-nosubcalls / int-01-train: snapshot.interactive = False, snapshot.env_actions = 0
```

Both episodes' first `env_call` attempt (the model trying to follow its interactive-category system
prompt) was refused:

```
rlm.bridge.RemoteError: RlmError: env is not available for this task
```

...and the root's very next cell measured `context length: 629279`, `chunks length: 431` -- the
full, ordinary chunked view, not the `0 0` the interactive design promises. **`context`/`chunks` were
fully populated and `env` was unavailable, the exact opposite of §14's design for this category.**

**Root cause, traced through the code:** `Episode.__init__`'s `interactive: bool = False` parameter
(`src/rlm/episode.py:447,456`) gates everything -- `self._index` (the `InteractiveIndex` that serves
`env`) is only built `if self.interactive` (`episode.py:1069`), and it is a **constructor argument**,
never derived from `task.category` anywhere in `episode.py` itself. Grepping the whole repository for
`interactive=True` finds it in exactly two places: `checks/test_episode.py` (unit tests that call
`run_episode` directly) and the design plan doc that wrote those tests
(`docs/superpowers/plans/2026-09-02-benchmark-v2.md`). **It does not appear anywhere in
`src/rlm/cli.py`.** `rlm_arm` and `rlm_nosubcalls_arm` (`cli.py:1457,1585`) both call `run_episode`
without an `interactive=` argument at all, so it defaults `False` regardless of the task's category.
`b2_arm` calls `run_b2` (`src/rlm/measure/arms.py:1350`), a separate map-reduce implementation that
never touches `episode.py`'s `Episode` class or its `interactive` flag at all -- and b2 abstains from
the interactive category by `rules.abstentions` in any case, so even if `run_b2` supported it, it
would never run on `int-01-train`.

**Consequence: as currently wired, no arm in the real `rlm bench` command ever exercises `env` on the
interactive category.** The category runs, but it runs as an oversized `linear_semantic` task --
`rlm` still delegates heavily (356 leaf steps, 298 ok / 58 error) because chunks are readable and
`llm_query` is not restricted, but the mechanism the category exists to test (context withheld,
`env` the only route to content) never engages. The unit tests that exercise
`interactive=True` directly (`checks/test_episode.py`) pass and are correct -- the machinery works in
isolation. The wiring from `task.category == "interactive"` to that constructor argument, inside the
actual bench command, is simply missing.

**Not fixed here** -- discovered during Step 2's checks, squarely past this task's authorized scope
(context_path + regression test + Defect 2 diagnosis). Reported for the controller to route.

## Step 2: the real ledger and store observations (run `29158eb5-b18f-4a05-91c0-09e2f39eb613`)

Grid: `--arm rlm,rlm-nosubcalls,b2`, 3 tasks (`ls-01-train`, `int-01-train`, `ctl-01-train`) x 3 arms
= 9 cells, minus b2's declared interactive abstention = **8 episodes + 1 `abstained` row**, exactly
as the brief predicted:

| task | arm | outcome | reason | wall (s) | leaf steps | llm_call (ok/err) | env_call |
|---|---|---|---|---:|---:|---|---:|
| ls-01-train | rlm | fail | checker_failed | 479.2 | 130 | 129/1 | 0 |
| ls-01-train | rlm-nosubcalls | context_exhausted | root_window | 590.9 | 0 | absent | 0 |
| ls-01-train | b2 | fail | checker_failed | 647.4 | 130 | 130/1 | 0 |
| int-01-train | rlm | fail | checker_failed | 1161.0 | 356 | 298/58 | 1 |
| int-01-train | rlm-nosubcalls | context_exhausted | root_window | 559.3 | 0 | absent | 12 |
| int-01-train | b2 | **abstained** | abstained:interactive | 0.0 | -- | -- | -- |
| ctl-01-train | rlm | **success** | -- | 44.0 | 0 | absent | 0 |
| ctl-01-train | rlm-nosubcalls | **success** | -- | 23.5 | 0 | absent | 0 |
| ctl-01-train | b2 | fail | checker_failed | 633.9 | 131 | 131/1 | 0 |

**`rlm-nosubcalls` (nosubcalls check, confirmed):** all three episodes show **zero** `llm_call`
rows -- not "rejected", genuinely absent from the trace, which satisfies the brief's "rejected or
absent" either way -- and **zero** `leaf`-actor steps in every one of them. The leaf server's own
completion count is unaffected as a direct consequence: nothing was ever sent to it.

**Interactive checks (confirmed NOT to hold, see Defect 3):** `env_call` steps ARE present (1 for
`rlm`, 12 for `rlm-nosubcalls`) but every one is a refused attempt
(`status='error'`, `"env is not available for this task"`); `snapshot.env_actions` is `0` for both;
`print(len(context), len(chunks))` in the first cell prints `"629279 431"`, not `"0 0"`.

**The `rlm` interactive episode: delegated, did not hit `context_exhausted`.** 356 leaf-actor steps
(298 ok, 58 error), spanning two leaf-pool rotations (128-slot pool exhausted twice, `rotation: 1`
and `rotation: 2` in the lifecycle log) -- heavy, sustained delegation over the readable chunks. It
ultimately failed the checker (`checker_failed`), not from running out of context. Contrast
`rlm-nosubcalls` on the same task, which cannot delegate and predictably ran the root out of window
(`context_exhausted/root_window`) trying to read a ~629K-character corpus directly. **Both are data,
per the brief -- but read together with Defect 3, the more important fact is that this comparison was
never the interactive-vs-env comparison §14 designed; it is linear-semantic-at-larger-scale.**

## Step 3: calibration table and projection

```
arm   task         category      measured s  projected s   ratio  outcome
rlm   ls-01-train  linear_semantic       479.2        450.0   1.06x  fail (checker_failed)
rlm-nosubcalls ls-01-train  linear_semantic       590.9         60.0   9.85x  context_exhausted (root_window)
b2    ls-01-train  linear_semantic       647.4        450.0   1.44x  fail (checker_failed)
rlm   int-01-train interactive       1161.0        450.0   2.58x  fail (checker_failed)
rlm-nosubcalls int-01-train interactive        559.3         60.0   9.32x  context_exhausted (root_window)
b2    int-01-train interactive          0.0        450.0     n/a  abstained (abstained:interactive)
rlm   ctl-01-train code_solvable        44.0        450.0   0.10x  success
rlm-nosubcalls ctl-01-train code_solvable        23.5         60.0   0.39x  success
b2    ctl-01-train code_solvable       633.9        450.0   1.41x  fail (checker_failed)

full grid = 32 tasks x 3 seeds x 6 arms, plus §8's escalation allowance (0.0 h, up to 32 extra episodes)
  from the pre-registered constants:    30.4 h grid + 0.0 h =   30.4 h
  with these measurements substituted:  43.9 h grid + 0.0 h =   43.9 h
  the measured figure is the one to judge: 43.9 h against §8's 60 h budget -- WITHIN
```

**PROJECTION: 43.9 h with these measurements substituted (30.4 h from the pre-registered constants
alone). This exceeds the 36 h figure named in this task's original brief.** Per the owner's
standing decision relayed by the coordinator mid-task, that no longer halts the task: recorded
prominently here, and this run proceeded straight to export and this note rather than stopping. Both
figures are within §8's separate 60 h budget line the calibration table itself judges against. Two
caveats on reading "43.9 h" as a real estimate of the full grid: (1) it is a 6-arm-grid figure printed
by the tool (`full grid ... x 6 arms`) even though this smoke ran only 3 arms -- the tool's own
full-grid math did not pick up `--arm`'s restriction, which may be a fourth minor wrinkle worth a
one-line look separately, not investigated further here; (2) `int-01-train`'s "interactive" cells ran
as oversized linear-semantic cells (Defect 3), so their wall-clock is not necessarily representative
of what a real env-driven interactive cell would cost once Defect 3 is fixed -- it could be faster
(no giant `context` string) or slower (many small `env` round trips instead of few large chunk reads).

## Archive

`rlm export 29158eb5-b18f-4a05-91c0-09e2f39eb613 --config config.v2.yaml --dest D:\AI\rlm-halo-archive\2026-09-03-v2-smoke-29158eb5\`
-- 8 episodes, `episodes.parquet` + `steps.parquet` + `blobs/<episode_id>/` for all 8,
`bundle-manifest.json` `config_snapshot_sha256 = ab6c580c312060b875df1b14ee7397cfab5b3ddc955bf9cb375f51c5bf86c931`.

## Servers

Both processes torn down by `rlm bench`'s own shutdown on exit -- `Get-Process llama-server` returned
zero processes immediately after, no manual stop needed.

## What's still owed before a real v2 grid

- Defect 2 (arm default ignores `rules.arms`) -- a `cli.py` fix, low risk, not attempted here.
- Defect 3 (interactive category never uses `env` through `rlm bench`) -- wiring `interactive=
  (task.category == "interactive")` into `rlm_arm`/`rlm_nosubcalls_arm` (and deciding what, if
  anything, changes for `b2`, which currently abstains from the category entirely and uses a
  different code path). This is the one that changes what the benchmark actually measures for 6 of
  its 16 shapes and should be fixed and re-smoked before any scored run.
- The "43.9 h" figure should be re-read once Defect 3 is fixed, since `int-01-train`'s wall-clock
  will likely move.
