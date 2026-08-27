# prime-agent local spike — tooling and results

Companion artefacts for `docs/research/2026-08-27-prime-agent-spike.md`, which is the
findings document. The plan that pre-registered every reading is
`docs/superpowers/plans/2026-08-26-prime-agent-local-spike.md`.

Measured 2026-08-27 on the target box: prime-agent v0.8.1 in WSL2 Ubuntu 24.04 driving
the shipped Qwen3.8-27B Q4_K_M over `llama-server` (Vulkan, DFlash2 drafter,
`-c 131072 -np 2 -ctxcp 0 --jinja --metrics`) at `127.0.0.1:8080`.

## What is here, and what is not

`results/` holds the **derived** artefacts — one CSV row per run, the harness states the
agent wrote, and the endurance log. It does **not** hold the raw episode material: 60
session JSONL files, per-run stdout/stderr, the 3.6 MB of task corpora (already in
`bench/corpora/`), and the kernel `.dill` snapshots stayed on the box under
`D:\spike\out\` and `/home/spike/` in WSL. Everything §4–§6 of the findings doc asserts
is reproducible from what is committed here.

## `tools/`

Written for this spike, stdlib only, Python 3.12. They run on the host (the sandboxed
`spike` user has no repo access), over a copy of the run tree taken out of WSL.

| file | what it does |
|---|---|
| `score.py` | scores one run directory: reads `answer.txt`, else the **first** assistant message in the session JSONL matching `^FINAL:`; checks it with `rlm.measure.checkers` against `bench/tasks/<id>.json`. `--dump-shape` prints a JSONL's distinct key paths |
| `usage.py` | one CSV row per run: turns, `ipython` calls, harness token accounting, `/metrics` deltas, longest identical-call streak, `refine.run` calls, host continuations, errors, subagents |
| `compare.py` | joins scored runs with the S4 / DFlash2 per-task medians and prints the A-cost ratios and the A1/A2/A3 reading |
| `endurance.py` | the C1b driver: N sequential prefix-disjoint chat completions, one CSV row each, flushed per row |
| `snap` | bash, run inside WSL: sha256 + copy of the global and session-local harness stores before and after a run (the A-refine guard) |

`score.py` and `usage.py` carry a resilient JSONL walker with the handled shapes
documented in their docstrings; the shape prime-agent v0.8.1 actually emits is recorded
in §3 of the findings doc.

## `results/`

### `phase0/` — the `-ctxcp` decision (findings §2)

`probe4b-cond{A,B}-*.json` are four requests sharing one 6,549-token system prompt and
differing only in the user message, under each server condition. `argv-cond{A,B}-*.txt`
are the two server command lines verbatim. Condition A reuses 6,525 tokens on a
divergent request (7.7 s); condition B reuses none (27.0 s) but is 1.6× faster end to
end on prime-agent's own workload, which never gets cross-run reuse. `-ctxcp 0` was
chosen; both conditions are kept so the choice can be re-argued.

### `phase-a/` — parity on v1 (findings §4)

`sweep.log` is the run-by-run record of 8 tasks × 3 runs. `usage.csv` has one row per
run; `compare.csv` the per-task medians against the scaffold baselines. Reading: **A1**,
8/8 tasks, 23/24 runs.

### `phase-b/` — self-improvement (findings §5)

`sweep.log` covers both categories: four train tasks each with an operator `/refine`,
promotion to the global store by file copy, then held-out and re-test runs.
`usage.csv` / `compare.csv` cover the 12 held-out runs only.

`harness/` is the load-bearing evidence for §5.3 — the state after each refinement, so
the accumulation can be read directly:

- `{agg,codeqa}-harness_state.after-{1..4}.json` — the session-local store after each
  of the four `/refine` calls in that category.
- `{agg,codeqa}-refine-{1..4}.out` — what `/refine` printed each time.
- `archived-global-before-codeqa-*.json` — aggregation's promoted state, archived when
  the code-QA category cleared the global store (the isolation step).
- `archived-global-before-C-*.json` — code QA's promoted state, archived before Phase C.

All eight refinements are `kind: memory`; there is no `skill`, `prompt` or `subagent`
entry in any of these files. Their `content` fields are where the memorized answers
live.

### `phase-c/` — the long autonomous session (findings §6)

`results.json` is the agent's own output, all seven counts correct;
`goal-stdout.txt` and `wall.txt` are the run record (112 s, exit 0);
`metrics-sampler.csv` is the 60-second server sampler, which caught only the first tick
because the session ended before the second.

### `endurance/` — C1b (findings §6.1)

`endurance.csv` is one row per request for 2,000 sequential prefix-disjoint completions
(index, status, wall, tokens, `timings`); `endurance.csv.metrics.csv` snapshots the
`llamacpp:` counters every 50 requests; `summary.log` is the verdict — 2,000/2,000,
zero failures, 153 min, server up 3 h 50 min, RSS ending below its launch figure.

## Reproducing

The server argv is in `phase0/`. Setup, the sandboxed WSL user, the `models.json` and
`settings.json`, and every Phase 0 gate are in §3 of the plan. Then, from a copy of the
run tree:

```
python tools/score.py   --run-dir <run> --task <id> --repo <repo>
python tools/usage.py   --runs-root <root> --out usage.csv
python tools/compare.py --runs-root <root> --usage-csv usage.csv
```

Note `.gitignore`'s `tools/` rule was anchored to `/tools/` in commit `6b2df45` so this
directory is visible to git at all.
