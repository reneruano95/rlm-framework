# Benchmark v2 smoke: BLOCKED -- every real cell hits `config_refused` because the frozen task files resolve to the wrong corpus directory

**Status:** BLOCKED. The smoke ran to completion (servers up, calibration table printed, both
processes exited cleanly) but produced **zero real episodes**. 17 of 18 cells errored with
`config_refused`; the 18th correctly abstained (`b2`/interactive). The store (`traces/v2/rlm.duckdb`)
has 0 rows in `episodes` and `steps`. None of Step 2's substantive checks (nosubcalls rejection
behaviour, `env_call`/`snapshot.env_actions`, the `rlm` interactive delegate-vs-exhaust question) can
be answered from this run because no episode ever started.

**Date:** 2026-09-03 · **Config:** `config.v2.yaml` (unmodified) · **Command:**
`uv run rlm bench --config config.v2.yaml --smoke --ledger runs/v2-smoke/ledger.jsonl --report runs/v2-smoke/RESULTS.md`
· **Smoke run_id:** `1fa499f1-fd9a-4943-a49d-142b12765a2a` · **Servers:** root :8080, leaf :8081 (plus
`bench_leaf` :8081 profile for the b1/b3 arms), all `Qwen3.6-35B-A3B-UD-Q4_K_M`, llama.cpp
`b10375-ba360efe1`, Vulkan. Manifest sha `0cc8bfbca2bbab99b6df086d7fe036bf2cf90cd6d76cd9fd4f565757e3305a92`
(frozen at commit `105a960`, unchanged by this run).

## What happened

Grid: 3 smoke tasks (`ls-01-train`, `int-01-train`, `ctl-01-train`, one per category) x 6 arms
(`rlm`, `rlm-restricted`, `rlm-nosubcalls`, `b2`, `b1`, `b3`) = 18 cells. From `runs/v2-smoke/ledger.jsonl`:

- **17 cells:** `outcome: error`, `reason: config_refused`, `episode_id: null`.
- **1 cell:** `b2`/`int-01-train`, `outcome: abstained`, `reason: abstained:interactive`,
  `episode_id: null` -- this one behaved exactly as specified (b2 does not run the interactive
  category) and is the only cell in the run that matches the brief's expectation.

Every `config_refused` error carries the identical shape, e.g. for `ls-01-train`:

```
ConfigError("context file D:\PROJECTS\rlm-halo-framework\bench\tasks\corpora\ls-01-train.txt
could not be read: [Errno 2] No such file or directory: '...\bench\tasks\corpora\ls-01-train.txt'")
```

The path the loader tried is `bench/tasks/corpora/<task_id>.txt` -- a directory that does not exist.
The real corpus is at `bench/corpora/v2/<task_id>.txt` (confirmed present, confirmed matching the
manifest's `corpus_path` field and `corpus_sha256`).

Both servers loaded cleanly and sat idle the whole run (`update_slots: all slots are idle`
throughout `traces/logs/leaf-server-bench.log` and `traces/logs/root-server-s5-a3b.log`; the smoke's
own `server_health` events show `state: ok` end to end) -- the failure is entirely in path resolution
before any dispatch, not a server or model problem. The bench command's own shutdown routine stopped
both processes cleanly on exit (`Get-Process llama-server` returns nothing afterward); no cleanup was
needed from this agent.

## Root cause

`src/rlm/episode.py:183` resolves a task's corpus relative to the task JSON file's own directory:

```python
raw["context"] = {"path": str((p.parent / raw.pop("context_path")).resolve())}
```

Every frozen v2 task file (e.g. `bench/tasks/v2/ls-01-train.json`) carries
`"context_path": "../corpora/ls-01-train.txt"` -- the exact same relative string v1's `build.py`
writes. That string is correct for v1's layout (`bench/tasks/*.json` next to `bench/corpora/*.txt`,
one `..` apart) but wrong for v2's, where both `tasks/` and `corpora/` gained an extra `v2/` level
(`bench/tasks/v2/*.json`, `bench/corpora/v2/*.txt`). One `..` from `bench/tasks/v2/` lands in
`bench/tasks/`, and `corpora/<id>.txt` from there is `bench/tasks/corpora/<id>.txt` -- confirmed by
direct resolution:

```python
>>> (Path('bench/tasks/v2/ls-01-train.json').parent / '../corpora/ls-01-train.txt').resolve()
D:\PROJECTS\rlm-halo-framework\bench\tasks\corpora\ls-01-train.txt   # does not exist
```

The bug is in `bench/build_v2.py`'s `build_linear_semantic`/`build_interactive`/(the code-solvable
builder), which all write `f"../corpora/{task_id}.txt"` unchanged from `build.py` (`bench/build_v2.py:240,282,326`)
without accounting for the extra nesting. `bench/manifest.v2.json`'s own `corpus_path` field is
correct (`bench/corpora/v2/ls-01-train.txt`) -- it is a separately-computed field, not derived from
`context_path`, and it is not what `episode.py` reads. **Task 22's freeze never caught this** because
`bench/closed_book.py` (the probe that gated the freeze) does not go through `episode.py`'s
`context_path` resolution at all; it is the only code path in this project that has ever read these
32 corpora before today.

## Why this is reported, not fixed

Fixing it means changing `context_path` in all 32 frozen task JSON files under `bench/tasks/v2/*`
(or rebuilding via a patched `build_v2.py`) -- both explicitly forbidden by this task's constraints
("Do not modify the frozen artifact ... Do not rebuild or reseed"). Task 22 pinned
`bench/manifest.v2.json`'s sha into `config.v2.yaml`; any edit to the task files would silently
invalidate that pin without changing the recorded sha (the manifest's sha covers the manifest JSON,
not the task files' `context_path` strings), which would be worse than leaving it broken and visible.
This is exactly the brief's escape valve ("If the smoke reveals a real defect in the scaffold, that
is a finding worth reporting, not a thing to work around") -- except the defect is in the frozen data
files themselves, not the scaffold code, which makes it not fixable within this task's remit at all.

## Consequence for Steps 2-4

None of the substantive checks could be performed:
- nosubcalls rejection behaviour: not observed (0 episodes reached the `llm_call` step).
- `env_call`/`snapshot.env_actions`/`len(context), len(chunks)`: not observed.
- Whether the `rlm` interactive episode delegates or hits `context_exhausted`: **undetermined** --
  the `rlm`/`int-01-train` cell never got past corpus loading.
- The store has 0 episodes, so there is nothing to `rlm export`; no archive bundle was created.

## Calibration table and projection (Step 3) -- read for the record, not load-bearing

The printed table is all `error (config_refused)` / one `abstained`, so the "measured" wall times
(0.0-12.2 s per cell) are load times and validation failures, not real episode durations, and the
projection built from them is **not a real projection of the v2 grid**:

```
full grid = 32 tasks x 3 seeds x 6 arms, plus S8's escalation allowance (0.0 h, up to 32 extra episodes)
  from the pre-registered constants:    30.4 h grid + 0.0 h =   30.4 h
  with these measurements substituted:   0.5 h grid + 0.0 h =    0.5 h
  the measured figure is the one to judge: 0.5 h against S8's 60 h budget -- WITHIN
```

**Per the owner's standing decision relayed by the coordinator mid-task, a >36 h projection would
not have blocked this task even if it had been real** -- but it is not real, so no projection number
from this run should be used to plan the full grid either way. The only trustworthy number here is
the **pre-registered constant projection, 30.4 h**, which was computed from `milestones/s2/aggregation_options.py`'s
fixed constants and does not depend on this run's broken measurements at all.

## What unblocks this

Someone with authority to touch the frozen v2 task files needs to either (a) patch
`context_path` in all 32 `bench/tasks/v2/*.json` files to `../../corpora/v2/<task_id>.txt` (or
equivalent) and re-verify the manifest sha is unaffected (it should be -- the sha covers
`bench/manifest.v2.json`, not the task files), or (b) fix `bench/build_v2.py` and do a controlled
rebuild + re-freeze (new sha, new Task-22-equivalent). Either path is a decision, not something this
task's remit covers.

## Files

- `runs/v2-smoke/ledger.jsonl` -- all 18 rows (gitignored, kept locally).
- `runs/v2-smoke/smoke.log` -- full run log including the calibration table (gitignored).
- `traces/logs/leaf-server-bench.log`, `traces/logs/root-server-s5-a3b.log` -- server logs showing
  clean load and idle slots throughout (truncated on next relaunch, not archived elsewhere).
- No `RESULTS.md` was written (the smoke path never reaches the report writer on an all-error grid)
  and no `rlm export` bundle exists (0 episodes in the store).
