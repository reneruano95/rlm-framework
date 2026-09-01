# Reorganization: rlm-halo as a private library distributed by copying the package

**Date:** 2026-09-01 · **Status:** EXECUTED, with corrections recorded below. Steps 1-4 and all nine deletion
groups are committed (`ef85d78` … `abf3e9e`). Where the tree and this document disagree, the tree is right and the
disagreement is written into §10.
**Goal, in the owner's words:** a private library, distributed by copying the folder — professional, scalable, modular.
**Standard of comparison:** `stanfordnlp/dspy`, cloned and read, not described.

**How this document was produced, and why that matters.** A first plan was built by survey and killed by a hostile critic (12 findings, several fatal). A second plan was looped four times against a fresh critic and a blind comparison. The fatal count went 3 → 1 → 2: it stopped converging, because each round answered findings by *elaborating* — six gate modules, four new CLI verbs, two package roots — and the critic correctly killed the elaboration as speculative generality three rounds running. **The measurements say the job is small.** This document is scoped to what the measurements demand and names what it defers. Everything below marked **[M]** was measured in this session against the real tree; everything marked **[C]** is a critic finding I verified myself.

---

## 1. The current map

309 tracked files.

| area | files | what it is |
|---|---|---|
| `docs/` | 78 | research records, specs, plans |
| `bench/` | 64 | frozen corpora, task definitions, split, manifest, build code |
| `src/rlm/` | 39 | **the package** |
| `gate/` | 37 | the artifact gate: 4 Python tools, 20 shell scripts, 1 TS extension, 8 artifacts |
| `tests/` | 35 | one flat directory |
| `upstream/` | 18 | llama.cpp bug reproducers + result JSONL |
| `prompts/` | 18 | frozen `.md` prompts, sha-pinned in `config.yaml` |
| root | 11 | `ARCHITECTURE.md` 170 KB, `CHANGELOG.md` 115 KB, `config.yaml` 45 KB, + 8 |

**The package's real shape [M].** 38 modules, 78 internal edges, **zero import cycles** — verified by AST walk with DFS cycle detection. `cli.py` imports 19 of the 38 (it is the composition root and the operator surface). `errors` is imported by 15, `config` by 9. `episode` is the centre of gravity: 11 imports out, 4 in.

**What the package reaches outside itself [M].** Exactly two lines: `src/rlm/cli.py:113` and `src/rlm/measure/bench.py:63`, both `REPO_ROOT = Path(__file__).resolve().parents[N]`. Plus `--config` defaulting to the CWD-relative string `"config.yaml"` at `cli.py:2146` and `:2207`.

**The public API [M].** `src/rlm/__init__.py` is 7 lines and exports nothing but `__version__`. Seven submodules already declare `__all__` (`context/`, `trace/`, `measure/arms.py`, `measure/bench.py`, `measure/checkers.py`, `measure/verdict.py`) — a partial convention to build on rather than invent.

**One module already does package data correctly [M].** `src/rlm/trace/store.py:36` — `_SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")`, and `schema.sql` travels inside the package. **This is the pattern the rest must copy.**

**`gate/` is not library code [M, one half CORRECTED 2026-09-01].** Nothing anywhere imports it — zero references from `src/`, `bench/` or `tests/`, and that half holds. **The other half was wrong**: this document originally said `gate/` imports nothing from `rlm`. It does, in three places, all function-local — `gate/decide.py:133` (`from rlm.measure.checkers import check`), `gate/screens.py:146` (`ChunkIndex`) and `:202` (`check`). The measurement used `grep "^import \|^from "`, anchored to line start, which cannot see an indented import. The conclusion survives for the surviving reason — nothing imports `gate`, and it is invoked by shell — but the dependency direction below is corrected: `gate/` sits at the same layer as `bench/`, not below it. Three rounds of the loop proposed shipping them as `rlm/gate/` with six modules and four CLI verbs; the critic killed it every time as "zero production callers", and the measurement above is why it was right.

**prime-agent is concluded.** It was a one-off probe that local models can drive a recursive harness. `D:\spike` is deleted. The evidence that lived only in WSL was rescued to `docs/research/2026-08-27-s6-lite-v0/runtime/` in `3eca663`, verified byte-for-byte. Nothing lives outside the repo.

---

## 2. The measurable half, measured

**The test:** copy the package directory into an empty project outside the repo, import it, run its suite with zero edits.

**Result today, in a clean venv holding only what `pyproject.toml` declares (24 packages):**

> **The suite does not run.** Four test modules fail to collect with `ModuleNotFoundError: No module named 'bench'`, which aborts the session. Excluding those four: **250 passed, 64 failed, 25 skipped, 354 errors.**

An earlier measurement in this session reported 265/853. That number was taken with the repo's own `.venv`, whose `_rlm_halo.pth` puts the repo root on `sys.path`, so `import bench` resolved. **It was optimistic and is withdrawn.** [C — the critic caught this in my own instrument, and was right.]

**Three causes, all local, none about dependencies:**

| # | cause | effect | scope |
|---|---|---|---|
| 1 | `bench` imported at module scope by tests | **aborts collection** | 5 files: `test_bench.py:20`, `test_bench_corpus.py:15-16`, `test_bench_manifest.py:14`, `test_verdict.py:20`, and `test_dispatcher.py` (function-local, does not abort) |
| 2 | `conftest.py` reads `REPO_ROOT/"config.yaml"` | 355 errors | 3 lines: `conftest.py:33, 90, 256` |
| 3 | prompts read from repo-root `prompts/` | ~30 errors | `root.v1/v2/v3.md`, `leaf-prefix.v1.md`, `leaf-envelope.v1.md`, `strat-*.v1.md` |

**Cause 1 has a written rationale already in the repo.** `pyproject.toml:41-45`: *"`bench/` is deliberately NOT in the wheel — `rlm/bench.py`, `rlm/cli.py` and `rlm/verdict.py` take `bench.manifest` as a function-local import precisely because…"*. **The package obeys this rule. The tests do not.**

**A critic finding I refute [M].** The round-2 critic asserted "three undeclared third-party test dependencies". There are none. An AST walk over `tests/` finds exactly five third-party modules at any scope — `duckdb`, `httpx`, `hypothesis`, `pytest`, `yaml` — and all five are declared. The clean-venv run confirms it empirically: nothing failed on a missing package.

**The other two criteria are already met or nearly so.** Zero import cycles: **already true**. Zero absolute repo paths in the package: two lines.

---

## 3. What dspy actually does, and does not

Read from the clone, not from memory.

**What it does better than us today:** it *runs* its guarantees. `run_tests.yml:279-285` builds a wheel, installs it into a clean venv and imports it on Python 3.10–3.14 on every push. We have no CI at all — no `.github/`, and `ARCHITECTURE.md:192`'s "Hypothesis profiles/seeds pinned in CI" is false. This is the one axis where we are behind and it is not a layout problem.

**What it does worse, measured [M via the blind comparison, verified against the clone]:**

- **33 module-scope import cycles.** `dspy/__init__.py` is six wildcard re-exports; leaf modules then import the package root back. `dspy/primitives/module.py:6` imports `dspy.predict.parallel` while `dspy/predict/chain_of_thought.py:6` imports `dspy.primitives.module`. 113 function-local `from dspy…` imports exist purely to dodge the mesh.
- **The top-level API is undeclared.** 18 of 22 subpackage `__init__`s declare `__all__`; `dspy/__init__.py` declares none, so its ~128-name surface mutates silently whenever a subpackage's `__all__` grows.
- **Its only structural import rule has never executed.** `pyproject.toml` configures `[tool.ruff.lint.flake8-tidy-imports] ban-relative-imports = "all"` but `TID` is not in `select`.
- **Copying the package yields zero tests.** `tests/` is a sibling root package and `tests/conftest.py` does `from tests.test_utils.server import …`.
- **Non-library content ships inside the package:** `datasets/` (HotPotQA, GSM8K), `dsp/` (a dead pre-rename namespace re-exported by three wildcards), `experimental/` (a 7-line alias shim), `utils/__init__.py` shipping a network `download(url)`, `react.py` beside `react_v2.py`.

**What dspy does NOT have at its root, and we do:** experiment outputs, vendored binaries, run logs, large data corpora, machine-specific configuration, and a 115 KB changelog. Its root is 6 files and 3 directories.

**The lesson we take, and the one we reject.** Take: a declared, pinned top-level surface, and a dependency direction something enforces. Reject: their package layout as a model for ours — a barrel `__init__` is precisely what produced their cycle mesh, and our zero-cycle graph is the thing most worth not losing.

---

## 4. The target map

Scoped to the measurements. The package moves as little as possible; what leaves it, leaves because it was measured to not belong.

```
rlm-halo-framework/
├── src/rlm/                    ← THE COPY UNIT. This directory is the library.
│   ├── __init__.py             lazy façade: __all__ + PEP 562 __getattr__, ZERO module-scope imports
│   ├── _data/
│   │   ├── config.default.yaml complete, machine-free, dispatcher: mock
│   │   └── prompts/            the 18 frozen .md files, moved here
│   ├── errors.py  bridge.py  budget.py  episode.py  cli.py  power.py
│   ├── config.py   context/   serve/   sandbox/   trace/   measure/
│   └── _tests/                 the portable contract suite; reads nothing outside the package
│
├── bench/                      frozen benchmark data + build code. Imports rlm. Never in the copy unit.
├── gate/                       stdlib-only CLI scripts. Imports nothing. Never in the copy unit.
├── tests/                      repo-level tests: bench, gate, citations, import rules
├── docs/                       research records and specs
├── upstream/                   llama.cpp reproducers
└── config.yaml, pyproject.toml, ARCHITECTURE.md, CHANGELOG.md, DIRECTION.md, README.md
```

**Allowed dependency direction, one total order:** `rlm/` → stdlib + declared deps only. `bench/` → `rlm/`. `gate/` → `rlm/` (function-local, in three places; see §1). `tests/` → all three. Nothing may point left.

### 4.1 The public API boundary

`rlm/__init__.py` gets an explicit `__all__` and a PEP 562 `__getattr__` that imports on first attribute access. **Module-scope imports stay at zero.**

This is not a style preference; it is forced by a constraint the loop's first plan got fatally wrong and I then measured [M]:

`sandbox/manager.py:99-104` stages four files into the AppContainer — `sandbox/child.py`, `rlm/__init__.py`, `rlm/errors.py`, `rlm/bridge.py` — and `sandbox/child.py:151` runs `from rlm.bridge import BridgeEndpoint, encode_frame` inside it. That import executes the *staged* `__init__.py`, in a tree where only three modules exist. An eager façade raises `ModuleNotFoundError` on every episode. Staging the import closure is not an option either: `episode.py` pulls `serve/dispatcher.py`, which pulls `httpx` — an HTTP client inside the one process the architecture exists to keep clients out of.

**Measured resolution.** I built a three-module tree identical to what the AppContainer sees and ran it:

```
1. from rlm.bridge import BridgeEndpoint, encode_frame   → OK
2. import rlm                                            → 1 module loaded, httpx False
3. rlm.<lazy name>                                       → ModuleNotFoundError on ACCESS only
4. modules loaded                                        → rlm, rlm.bridge, rlm.errors
```

And against the real package: bare `import rlm` costs 1 module and no `httpx`; after resolving every public name, 22 modules and `httpx` present — correct for a consumer, never paid by the sandbox. `rlm.bridge` is a submodule import, so `__getattr__` never fires for it.

**The surface** is the names `cli.py` and the experiments actually consume: `run_episode`, `Task`, `EpisodeResult` from `episode`; `Config`, `load_config`, `config_snapshot` from `config`; `LLMDispatcher` from `serve`; `TraceLogger` from `trace`; `SandboxManager` from `sandbox`; the exception and enum set from `errors`. Pinned by a test asserting `rlm.__all__` equals a literal list, so any change to the surface fails until the list is edited.

**Internal marking:** no new `_`-prefix churn. The rule is the dependency order plus the `__all__` list; a name not in `rlm.__all__` and not in a subpackage `__all__` is internal.

### 4.2 Package data

Three kinds of file must travel inside the copy unit, following `trace/store.py`'s existing pattern:

- `_data/config.default.yaml` — a complete, machine-free config with `dispatcher: mock`. The repo's 45 KB `config.yaml` stays at the root as this box's override. `load_config()` resolves the default via `Path(__file__)`, never CWD.
- `_data/prompts/` — the 18 `.md` files. **The hard constraint [C, verified]:** `trace/replay.py:124` resolves prompts by the snapshot's literal relative path, unpinned, so 614 stored snapshots name `prompts/…`. The move needs a two-step resolver — try the recorded path, fall back to the package copy — and `config.yaml:510-559`'s 13 pinned path strings must be updated in the same commit.
- `trace/schema.sql` — already correct; declare it in the wheel's package-data so it is not silently absent from a built artifact.

### 4.3 Seams: what we build, and what we refuse

**Build one:** `Dispatcher` — it has two real implementations (`LLMDispatcher` and a mock the suite needs), duck-typed today. Give it a Protocol.

**Refuse the rest.** `Backend` has one implementer and selection is a three-line if-chain. `Environment` has zero implementers, and the gate — the seam's only claimed consumer — reads `bench/tasks/` and `bench/splits/` by path. `Arm` contradicts itself: three of four arms are functions, so `ArmRunner = Callable[...]` is the honest type and a Protocol is ceremony. `ProcessManager` already exists as a deliberate one-method Protocol at `serve/serverproc.py:50` and needs nothing.

Recorded plainly because it was earned: three rounds of the loop proposed five seams; three had zero or one implementation. Speculative generality is the failure mode this project has already paid for once.

---

## 5. The move table

| # | from | to | why | what breaks |
|---|---|---|---|---|
| M1 | `prompts/**` (18 files) | `src/rlm/_data/prompts/` | package data; a copied package cannot read the repo root | `config.yaml:510-559` (13 pinned paths); `tests/test_prompts.py:8`; `test_envelope_wiring.py:21-22`; `conftest.py:259-260, 873-880, 1039-1046`; **and** `trace/replay.py:124`'s resolver — see 4.2 |
| M2 | *new* | `src/rlm/_data/config.default.yaml` | the suite must run with no repo | nothing; additive |
| M3 | 5 test files' `bench` imports | function scope | cause 1 of the copy test; the package already obeys this rule | nothing — same rule `pyproject.toml:41-45` already states |
| M4 | `tests/**` | split: `src/rlm/_tests/` (package contract) vs `tests/` (repo-level: bench, gate, citations, import rules) | the copy unit must carry its own suite | **[C]** every `parents[1]` walk in moved files re-resolves. `test_citations.py:38` fails **open** — its `rglob` root shrinks to the test tree and the ~176 milestone citations stop being checked silently. That file stays at repo level. |
| M5 | `gate/run_decision.sh`, `run_episode.sh` | `docs/research/2026-07-27-s6-lite-v0/runtime/` | they are the record of how the 148 episodes ran; their `ledger.jsonl` and `SHA256SUMS` already live there | nothing runs them; `decide.py:139-160` still parses the layout they wrote — lift that cell contract into its docstring |
| M6 | `gate/artifacts/control-v2.screens.json` | `docs/research/2026-08-27-s6-lite-v0/decisions/pc-03/screens.json` | it is the **only** machine-readable record that artifact v2 passed the four screens: pc-03 has no `decision.json`, and its audit row parses to `screens:False` | nothing |
| M7 | `docs/…/spike/tools/endurance.py` | `bench/` | stdlib + `urllib`, zero prime-agent coupling; it produced the +9.9% drift figure `gate/decide.py:31` cites as the reason wall-clock is not gated | nothing; scrub the `D:\spike\rss.csv` default first |
| M8 | `src/rlm/cli.py:113`, `measure/bench.py:63` | constructor arguments / config fields | the two repo paths in the package | **[C]** `cli.py:1337` reads `BENCH_MANIFEST_PATH` as a module constant inside `load_benchmark_manifest()`, and `BenchmarkCfg` has no path field. `cli.py:1279-1281` states the current design as an anti-p-hacking measure — *"a scoring run an operator can point at a different manifest is a scoring run they can point at a friendlier one"*. **Add the field to `BenchmarkCfg`; do not add a CLI flag.** |

**Not moving:** `bench/` (frozen data, imports `rlm`, correctly outside the copy unit), `gate/`'s four Python tools (§1: stdlib, no importers, not library code), `upstream/`, `docs/`, `config.yaml`, `ARCHITECTURE.md`.

**`ARCHITECTURE.md` is not split by this plan.** It is 170 KB and that is a real problem, but it is cited by path-and-line across the research record, and splitting it is a change to the evidence trail, not to the library. Deferred, and owed as its own decision.

---

## 6. Deletions

Nine commits, least to most contentious. Each becomes one commit with its reason in the message, so each reverts alone.

| id | group | paths | why it dies | what is lost |
|---|---|---|---|---|
| **D1** | Duplicated gate inputs | `gate/artifacts/{positive-control,control-v2,negative-control}.{json,sha256}`, `positive-control.screens.json`, `gate/audit.jsonl` | live input directory for a runner that is gone; every byte is a verified duplicate of an archived record. `positive-control.json bd3c1958…` = `pc-01/candidate.json` = `pc-02/candidate.json`; `control-v2.json 1afadf52…` = `pc-03/candidate.json`; `gate/audit.jsonl 94615b5b…` = `decisions/audit.jsonl`, byte-identical | nothing |
| **D2** | Four dead symbols | `episode.py:204` `_normalise`; `winproc.py:46` `advapi32`; `winproc.py:63-66` `SECURITY_ATTRIBUTES`; `screens.py:279` `_iter_entries` | whole-tree grep returns exactly one line each — the definition | nothing; drops a DLL load from every `import rlm.sandbox.winproc` |
| **D3** | A stale build comment false in both halves | `pyproject.toml:46-47` | claims `bench/vocab.py` imports `milestones/s1,s2` at module time "which is why those directories cannot be archived". They were archived in `ff6c8ea` and the guard removed in the same commit; `bench/vocab.py:29` says it imports nothing from `milestones/` | nothing |
| **D4** | The WSL bridge | `gate/checks/{stage_gate,stage_tasks,step1_setup,collect_dec}.sh` | these four *are* the deploy step the owner forbade. `stage_gate.sh` is 14 out-of-repo absolute paths; both `/d/spike` referents are already gone, so two are unrunnable before deletion | the record of what was staged where — already in `runtime/README.md`. **Stated honestly:** the 30 per-task question `.txt` files these staged were never in git, so byte-identical re-runs were already impossible; deletion stops hiding that rather than causing it. The question text survives in each `bench/tasks/<id>.json` |
| **D5** | Thirteen spent one-off drivers | `gate/checks/{launch_pc01,launch_pc03,resume_pc03,smoke_pc02,step1_checks,step1_recheck,step2_guard,step2_verify,diag_stall,diag_agg07,profile_agg07,profile_ep,probe_list}.sh` | twelve of thirteen take zero arguments and hardcode one decision or task id. They are the record of one invocation each, of a concluded harness. Every one sources `$HOME/.spike_env`; `diag_agg07.sh:8` calls a script never tracked | the reproduction recipes. Every verdict they produced is recorded in the results doc and the three `decision.log`s |
| **D6** | The prime-agent extension | `gate/extension/rlmh-gate.ts`, `gate/checks/test_gate_unit.mjs`, `gate/checks/run_unit.sh` | 317 lines against prime-agent's `ExtensionAPI`; its four handlers are that host's hooks. No runtime can load it | **the largest real loss here.** It is the only executable statement of the I1 veto, the prompt filter and the identical-turn budget. **Precondition:** lift its header block (`:1-45`) and filter semantics into the s6-lite spec first, so the `src/rlm/` reimplementation has a written contract |
| **D7** | prime-agent session parsers | `docs/research/2026-08-26-prime-agent-spike/tools/{snap,usage.py,score.py,compare.py}` (104 KB) | all four parse prime-agent's session JSONL over a run tree that lived in WSL and `D:\spike\out\`, both gone; the inputs were never committed. `score.py:106-113` is already re-implemented line-for-line at `gate/decide.py:133-160` | the ability to re-derive the CSVs from raw sessions — already impossible. Every derived output stays committed |
| **D8** | `gate/propose.py` | 257 lines | never ran (`git log --all --diff-filter=A` finds no `round-*` across 224 commits), output format is prime-agent's `HarnessEntry`, input is prime-agent session JSONL. Declared not built by its own commit: results §6 *"The proposer does not exist"* | the SYSTEM prompt at `:47-70`, genuine design from a measured failure. **Precondition:** lift it into the spec first |
| **D9** | `tests/fixtures/repetition/extract.py` | 48 lines | self-labelled "One-off"; not collected by pytest; `:21-23` still walks `parents[3]` from before the `src/` move and resolves only because of the venv `.pth`; its input is gitignored | the SQL that selected the turns. **Precondition:** paste the two annotations (synth-01 looped 70× into `context_exhausted`, synth-07 111× into `budget_kill`) into the fixture JSONs first |

**Rule applied throughout, and never bent:** no deletion of evidence of a measurement actually run, of the benchmark corpora/tasks/split, or of a document recording a measured finding.

### KEEP — rejected candidates, and why the reason is stronger than caution

- **`gate/artifacts/control-v2.screens.json`** — a sweep called it a duplicate. It is not: pc-03 has no `decision.json` and its audit row has no `screens` key. Sole machine-readable record of a check actually run. → moves (M6).
- **`gate/run_decision.sh`, `run_episode.sh`** — `run_episode.sh` writes the exact cell layout `decide.py:139-160` still parses; `run_decision.sh` is the only written form of the blocked ON/OFF schedule that makes the 148 episodes valid against R9 thermal drift. Delete the writer and the surviving reader has no contract. → move (M5).
- **`endurance.py`** — produced the measurement `decide.py:31` cites as the justification for wall-clock not being gated. Deleting it makes a live decision-rule premise unreproducible. → moves (M7).
- **`serve/leakcheck.py:180` `foreign_identifiers`** — reference-counting flags it, but `test_leakcheck.py:197` asserts it agrees with `ChunkIndex.foreign`. It is a differential oracle for R13, and R13 is an open unfixed upstream defect.
- **`dispatcher.py:402 predicted_reuse`, `arms.py:219 SNAPSHOT_KEYS`** — "test-only", but their tests encode measurements: 5 parametrised cases pinning llama.cpp prefix-cache behaviour on this box, and a guard against a real config-key collision.
- **`docs/research/2026-08-31-papers.md`** — zero inbound but for its successor's `Consumes:` line. The successor's finding is that two first-pass claims inverted under verification; the raw text is the control that verification was measured against.
- **Both statistics implementations** — `decide.py:bootstrap_ci` vs `measure/stats.py:paired_bootstrap_ci`, `reliability.py:exact_mcnemar_one_sided` vs `sign_test_p`. They look like duplicates. They are not interchangeable: median-of-ratios vs mean-of-deltas, seed `20260827` vs `8`, one-sided vs two-sided. Collapsing them would change verdicts already in `audit.jsonl`. Carried as a consolidation note, never a deletion.
- **`.claude/skills/gauntlet-loop/`** — zero inbound and unrelated to the framework, but it is user tooling misfiled, not repo dead weight. Flagged, not touched; the owner decides.

---

## 7. Order of execution

Each step ends with a verification that can fail. Steps 1–3 are the copy test; 4–6 are the boundary; 7 is the pruning.

1. **Repair the lint before relying on it.** `tests/test_import_rules.py:7` resolves `parents[1]/"src"/"rlm"`; in a copied project that path is absent, all 25 cases `pytest.skip`, and the coverage guard passes vacuously against `set()`. Resolve the package with `importlib.util.find_spec("rlm")`, and **fail** rather than skip when a listed module is missing. **[C]** Its coverage guard matches by **basename** with 8 exempt names — any rename must update that list in the same commit. *Verify: delete a module from `ISOLATED` and confirm the suite goes red.*
2. **Move the `bench` imports to function scope** in the 5 test files. *Verify: clean-venv collection completes.*
3. **Ship `_data/config.default.yaml` and `_data/prompts/`**, repoint `conftest.py:33,90,256` and the prompt readers, add the two-step prompt resolver in `replay.py`, update `config.yaml:510-559`. *Verify: clean-venv run, and replay one stored episode from each of the four snapshot vintages `replay.py:105-121` enumerates — not one episode.*
4. **The lazy façade** in `__init__.py`, plus the `__all__`-equals-literal test. *Verify: a live episode on this box (the sandbox path is not covered by the copy test), plus `import rlm` loading exactly one module with `httpx` absent.*
5. **Split the suite** into `src/rlm/_tests/` and repo-level `tests/`. `test_citations.py` stays at repo level. The shipped conftest must not set `filterwarnings = ["error"]` — **[C]** that would reconfigure the consumer's whole test session. *Verify: the citations test still sees ~176 milestone paths, i.e. it did not go vacuously green.*
6. **The two repo paths** (M8), via a `BenchmarkCfg` field, not a CLI flag. *Verify: `grep -rn "parents\[" src/rlm/` returns nothing.*
7. **The nine deletion commits**, D1 → D9, each with its precondition satisfied first. *Verify after each: full repo suite green, and the gate's recorded decisions still reproducible from `decisions/` + `runtime/`.*

**Pass criterion for the copy test — the assertion, not the command.** `pytest --pyargs rlm` in a stripped temp dir with only declared dependencies, **and**: collection completes with zero errors; the pass count equals the count from the same suite run in-repo; and the run reports **zero skips attributable to a missing repo file**. On a non-Windows consumer the sandbox tests skip legitimately, so the criterion is checked on this box, where nothing may skip for a path reason.

---

## 8. Acceptance criteria

| criterion | today | target |
|---|---|---|
| Copy + import + run suite, zero edits | **aborts** | collection clean, pass count equals in-repo, zero path-caused skips |
| Zero import cycles | **already true** (38 modules, 78 edges) | held, and asserted by the repaired lint |
| Zero absolute repo paths in the package | 2 lines | 0, asserted by grep in the suite |
| Public API declared and pinned | exports nothing | explicit `__all__`, list-equality test, zero module-scope imports in `__init__` |
| A live episode still runs on this box | yes | yes — checked after step 4, the one thing the copy test cannot see |
| Gate + benchmark reproducible after move and prune | yes | yes — checked after every deletion commit |

---

## 9. What this document does not do

Named rather than quietly dropped:

- **No `rlm/gate/` subpackage.** §1 measured why: `gate/` is stdlib-only with zero importers. It stays where it is.
- **No second package root (`rlmx/`).** Its stated justification was moving "30 frozen task checkers" out of the library. **There are 6 checkers**, all generic format logic (`measure/checkers.py:126-133`), and the split has nothing to move.
- **No `ARCHITECTURE.md` split.** Real problem, cited by line across the research record, owed as its own decision.
- **No CI.** This is the one axis where dspy is genuinely ahead, and it is not a layout change. Until it exists, every rule here is enforced by a suite someone must remember to run — state that honestly rather than claim the guarantee.
- **Re-pointing the gate's decision machinery at `src/rlm/`'s own scaffold** — the owner's stated requirement. Four rounds of the loop failed to address it, and it is not a reorganization: it is the design of what replaces prime-agent as the thing being gated. It gets its own spec.
