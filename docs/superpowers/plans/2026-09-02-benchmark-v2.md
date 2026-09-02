# Benchmark v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build benchmark v2 — sixteen frozen tasks in three categories, two frozen corpus streams plus a regenerable practice stream, three arms (`rlm`, `rlm-nosubcalls`, `b2`), two build-time adversaries, one new sandbox verb — so the project can, for the first time, price delegation and supply S6 with trajectories where delegating was the right move.

**Architecture:** v2 is a *new benchmark version beside v1*, not a rebuild: its own manifest file, corpora, task files, config (`config.v2.yaml`, derived from the S5 same-model config) and `manifest_sha256` pin. Everything the scorer hardcoded for v1 (arms, margin, categories, baselines, abstentions) becomes a `rules` block read from the manifest, with defaults that reproduce v1 byte-for-byte so the frozen v1 record and its 890 tests stand untouched. The interactive category adds one scaffold-side verb, `env`, built on exactly the contract `llm_query` uses (rebuilt over a two-name namespace; the corpus never crosses the pipe; every return value truncated scaffold-side). Labels come from a vendored human-labelled corpus (TREC) wrapped in the project's synthetic register; the builder computes every answer from the labels it sampled and rejects any task a parser or a ≤40-window self-read can solve.

**Tech Stack:** Python 3.12, `uv`, pytest (`checks/`, `src/rlm/_tests/`), pydantic v2 (`_Strict` models), DuckDB (trace store; ENUM migration), llama.cpp `/tokenize` for build-time counts, the existing C2 chunker (`src/rlm/context/chunker.py`). No new third-party dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-benchmark-v2-design.md` (§§0–13 approved 2026-08-25; §14 amendments approved 2026-09-02). Read §1, §2, §3, §4, §6, §14 before starting any task.

## Global Constraints

- **Nothing under `src/rlm/` imports `bench` at module scope** — `checks/test_import_rules.py` enforces it; `load_benchmark_manifest` imports lazily inside the function. Keep every new `bench` import lazy.
- **No existing pinned prompt file is edited.** New versions are new files, pinned by sha256 **over the whole file, header included** (`config.py:958`). Every prompt file starts `<!-- changelog` and carries a `vN | YYYY-MM-DD | …` line (`checks/test_prompts.py:34-38`).
- **`config.yaml` is not modified** except where a task says so explicitly. v2 runs from `config.v2.yaml`. `checks/test_prompts.py:179` and `:207` read `config.yaml` and must stay green.
- **v1's manifest sha `571918d24bd848b8cb4122b7226882d65b929aa4dff0cb80a578d2eb04603c91` must not move.** Any schema addition to `bench/manifest.py` must serialize v1 to the identical bytes (Task 1 pins this).
- **Files written by the builder use `encoding="utf-8", newline="\n"`** — LF is load-bearing for every sha on Windows (`build.py:60-63`).
- **`_RESERVED` plumbing names are never settable by the parent** (`child.py:558`); a new verb's callables are rebuilt over `{"BRIDGE", "LOOP"}` only (`child.py:449-467`); `checks/test_sandbox_child.py:103` asserts the injected globals are exactly `['BRIDGE', 'LOOP']`.
- **Every step the scaffold writes gets its index from `_alloc()` and is written with `_put()`** (`episode.py:494-501`); a new `action_type` needs `errors.ActionType`, the DuckDB ENUM migration in `schema.sql`, and the transcript branch in `replay.py:219-243`.
- **Model configuration for every v2 run:** root and leaf are both `Qwen3.6-35B-A3B-UD-Q4_K_M` (spec D-B7). `config.v2.yaml` derives from `config.s5-a3b-root.yaml`.
- **Chunk geometry stays `size_tokens: 640, stride_tokens: 480, snap_tolerance: 0.10`** (spec D-B5). The self-read adversary's `K = 40` is derived from it and from `truncation_cap_chars: 2000` (spec §14.2).
- **Pre-registered numbers, fixed before any run:** N = 16 (6 linear-semantic · 6 interactive · 4 code-solvable), margin **+2**, escalation band {+1, +2}, seeds [1, 2, 3], escalation seeds [4, 5], `rlm-nosubcalls` re-authoring threshold ≥3 of 6 per category, B2 abstains from `interactive`, static corpora ~60,000 tokens, interactive corpora ~200,000 tokens.
- **Commit after every task** with a message in the repo's voice (lowercase prefix, then the claim). Run the named tests before each commit; run the full suite (`uv run pytest -q`, ~12 min) at the end of each phase.
- **Test invocation:** `uv run pytest <path>::<name> -q -p no:cacheprovider`. Sandbox tests are Windows-only and spawn real processes; expect 1–3 s each.

---

## File Structure

**Created**

| path | responsibility |
|---|---|
| `bench/rules.py` | `BenchmarkRules` dataclass: arms, margin, band, tripwire floor, abstentions, scored stream; `for_manifest()` with v1 defaults |
| `bench/sources/trec/{README.md, trec_train.jsonl, trec_train.sha256, fetch.py}` | vendored label source + licence note + the one-shot fetcher |
| `bench/corpus_v2.py` | register wrapper over labelled items; `LinearSemanticCorpus`, `InteractiveCorpus`, `ControlCorpus`; answer computation; token sizing |
| `bench/adversary.py` | the two build-time guards: `parser_adversary()`, `self_read_adversary()` |
| `bench/build_v2.py` | 16 shapes × {train, held_out} (+ `--practice`), tasks/corpora/manifest emission, adversary gates, `rules` |
| `bench/tasks/v2/*.json`, `bench/corpora/v2/*.txt`, `bench/manifest.v2.json` | the frozen artifact |
| `src/rlm/context/interactive.py` | `InteractiveIndex`: deterministic index over a multi-document corpus; `search/open/window`; reference-path helper |
| `src/rlm/_data/prompts/root.v4.md`, `root-nosubcalls.v1.md` | the two root bodies (spec §14.3) |
| `src/rlm/_data/prompts/strat-linear-semantic.v1.md`, `strat-interactive.v1.md`, `strat-code-solvable.v1.md` and their `-nosubcalls.v1.md` twins | v2 strategy blocks, per arm |
| `config.v2.yaml` | the v2 run configuration |
| `checks/test_rules.py`, `checks/test_bench_v2_corpus.py`, `checks/test_adversary.py`, `checks/test_interactive_index.py`, `checks/test_manifest_v2.py` | new test modules |

**Modified**

| path | change |
|---|---|
| `bench/manifest.py` | optional fields (`stream`, `shape_id`, `interactive`, `min_windows`, `reference_actions`, `label_source`), `rules`, None-stripping serializer, per-stream validation, `scored_tasks()` |
| `src/rlm/measure/verdict.py` | read `BenchmarkRules` from the manifest; abstention-aware `load_grid`/`decide`; band/margin/tripwire from rules |
| `src/rlm/measure/stats.py` | `needs_escalation(margin, band)` |
| `src/rlm/measure/bench.py` | `rlm-nosubcalls` in `ARM_ORDER`/`ARM_PROFILE`/`bench_extra`; abstention rows in `run_block`; `bench_manifest_path()` |
| `src/rlm/cli.py` | `rlm_nosubcalls_arm`; manifest path by version; scored-stream default; `BASELINES` from rules in escalation; help text |
| `src/rlm/config.py` | `strategy_templates: dict[str, PromptRef]`; `strategy_templates_nosubcalls`; `render_root(..., no_subcalls=)` |
| `src/rlm/episode.py` | `run_episode(..., no_subcalls=, interactive=)`; refusal in `_on_llm_query` logging a `rejected` step; `_on_env`; index construction |
| `src/rlm/sandbox/child.py`, `sandbox_bootstrap/sandbox_child.py` | `env` facade in `_RESERVED`; wire table |
| `src/rlm/sandbox/manager.py` | `on_env`, `_serve` branch |
| `src/rlm/errors.py`, `src/rlm/trace/schema.sql`, `src/rlm/trace/replay.py` | `ENV_CALL` action type + migration + transcript branch |
| `bench/closed_book.py` | `--manifest` flag |
| `ARCHITECTURE.md`, `CHANGELOG.md` | §8 amendment, v0.4.1 entry |

---

## Phase A — rules and schema (v1 stays byte-identical)

### Task 1: Manifest schema extension without moving v1's sha

**Files:**
- Modify: `bench/manifest.py:45-98`
- Test: `checks/test_manifest_v2.py` (create), `checks/test_bench_manifest.py` (must stay green)

**Interfaces:**
- Produces: `TaskEntry` gains `stream: str | None = None`, `shape_id: str | None = None`, `interactive: bool | None = None`, `min_windows: int | None = None`, `reference_actions: int | None = None`, `label_source: str | None = None`. `BenchmarkManifest` gains `rules: dict | None = None` and `scored_tasks() -> list[TaskEntry]`. `to_json()` omits every one of those keys when its value is `None`.

- [ ] **Step 1: Write the failing test**

```python
# checks/test_manifest_v2.py
import json
from pathlib import Path

import pytest

from bench.manifest import BenchmarkManifest, TaskEntry

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "bench" / "manifest.json"
V1_PIN = "571918d24bd848b8cb4122b7226882d65b929aa4dff0cb80a578d2eb04603c91"

pytestmark = pytest.mark.skipif(not V1.exists(), reason="v1 manifest not built")


def test_v2_fields_leave_the_v1_sha_byte_identical():
    """Adding optional fields must not move the frozen v1 pin: None-valued v2
    fields are omitted from the serialized form, so v1's bytes are unchanged."""
    m = BenchmarkManifest.load(V1)
    assert m.sha256 == V1_PIN
    assert m.to_json() == V1.read_text(encoding="utf-8")


def test_v2_fields_round_trip_when_set(tmp_path):
    t = TaskEntry(task_id="ls-01-train", category="linear_semantic",
                  task_file="bench/tasks/v2/ls-01-train.json",
                  corpus_path="bench/corpora/v2/ls-01-train.txt",
                  corpus_sha256="0" * 64, corpus_tokens=60_000,
                  corpus_date="2026-09-02", checker="int_exact",
                  question_sha256="1" * 64, stream="train", shape_id="ls-01",
                  min_windows=139, label_source="CogComp/trec@sha256:abc")
    m = BenchmarkManifest(benchmark_version="v2", built_at="2026-09-02",
                          token_counter="leaf:/tokenize",
                          assumed_training_cutoff="2025-12-31", tasks=[t],
                          rules={"margin": 2})
    p = tmp_path / "m.json"
    m.write(p)
    back = BenchmarkManifest.load(p)
    assert back.tasks[0].stream == "train"
    assert back.tasks[0].min_windows == 139
    assert back.rules == {"margin": 2}
    assert json.loads(p.read_text())["tasks"][0]["shape_id"] == "ls-01"
    assert "interactive" not in json.loads(p.read_text())["tasks"][0]


def test_scored_tasks_is_the_scored_stream_or_everything():
    train = TaskEntry(task_id="a", category="c", task_file="", corpus_path="",
                      corpus_sha256="", corpus_tokens=1, corpus_date="",
                      checker="int_exact", question_sha256="", stream="train")
    held = TaskEntry(**{**train.__dict__, "task_id": "b", "stream": "held_out"})
    m = BenchmarkManifest(benchmark_version="v2", built_at="", token_counter="",
                          assumed_training_cutoff="", tasks=[train, held],
                          rules={"scored_stream": "train"})
    assert [t.task_id for t in m.scored_tasks()] == ["a"]
    v1like = BenchmarkManifest(benchmark_version="v1", built_at="", token_counter="",
                               assumed_training_cutoff="", tasks=[train, held])
    assert [t.task_id for t in v1like.scored_tasks()] == ["a", "b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest checks/test_manifest_v2.py -q -p no:cacheprovider`
Expected: FAIL — `TypeError: TaskEntry.__init__() got an unexpected keyword argument 'stream'`.

- [ ] **Step 3: Implement**

In `bench/manifest.py`, extend `TaskEntry` (after `closed_book`):

```python
    # v2 (2026-09-02). All optional and None-stripped on serialization so the
    # frozen v1 manifest hashes to the same bytes it did on 2026-08-15.
    stream: str | None = None            # "train" | "held_out" | "practice"
    shape_id: str | None = None          # the task shape shared across streams
    interactive: bool | None = None      # corpus behind `env`, not in `chunks`
    min_windows: int | None = None       # self-read adversary: minimal necessary window set
    reference_actions: int | None = None # interactive: optimal-path `env` operation count
    label_source: str | None = None      # e.g. "CogComp/trec@sha256:<vendored file>"
```

Add to `BenchmarkManifest` a `rules` field and replace `to_json`:

```python
    rules: dict | None = None

    _OPTIONAL_V2 = ("stream", "shape_id", "interactive", "min_windows",
                    "reference_actions", "label_source")

    def to_json(self) -> str:
        d = asdict(self)
        if d.get("rules") is None:
            d.pop("rules", None)
        for t in d["tasks"]:
            for k in self._OPTIONAL_V2:
                if t.get(k) is None:
                    t.pop(k, None)
        return json.dumps(d, indent=2, ensure_ascii=False) + "\n"

    def scored_tasks(self) -> list["TaskEntry"]:
        """The tasks a verdict is computed over: the `rules.scored_stream` when
        the manifest declares one, otherwise every task (v1)."""
        stream = (self.rules or {}).get("scored_stream")
        if stream is None:
            return list(self.tasks)
        return [t for t in self.tasks if t.stream == stream]
```

(`_OPTIONAL_V2` must be declared with `field(default=..., init=False, repr=False)` or as a plain class attribute annotated `ClassVar[tuple[str, ...]]` so `asdict` ignores it — use `ClassVar`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest checks/test_manifest_v2.py checks/test_bench_manifest.py -q -p no:cacheprovider`
Expected: all pass, including `test_config_pins_the_frozen_version`.

- [ ] **Step 5: Commit**

```bash
git add bench/manifest.py checks/test_manifest_v2.py
git commit -m "bench: manifest grows the v2 fields and a rules block, and v1 still hashes to 571918d2"
```

---

### Task 2: `BenchmarkRules` — the scorer's constants become manifest data with v1 defaults

**Files:**
- Create: `bench/rules.py`
- Test: `checks/test_rules.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class BenchmarkRules:
    rlm_arm: str = "rlm"
    baselines: tuple[str, ...] = ("b1", "b2", "b3")
    margin: int = 3
    escalation_band: tuple[int, ...] = (1, 2, 3)
    tripwire_floor: int = 3
    abstentions: dict[str, tuple[str, ...]] = field(default_factory=dict)  # arm -> categories it does not run
    scored_stream: str | None = None
    n_tasks: int | None = None
    @property
    def arms(self) -> tuple[str, ...]
    def abstains(self, arm: str, category: str) -> bool
    @classmethod
    def for_manifest(cls, manifest) -> "BenchmarkRules"   # manifest.rules or v1 defaults
```
- `bench/rules.py` has **no imports from `src/rlm`** (it is imported by the scorer lazily, like `bench.manifest`).

- [ ] **Step 1: Failing test**

```python
# checks/test_rules.py
from bench.rules import BenchmarkRules


class _M:
    def __init__(self, rules): self.rules = rules


def test_v1_defaults_reproduce_the_pre_registered_constants():
    r = BenchmarkRules.for_manifest(_M(None))
    assert r.rlm_arm == "rlm" and r.baselines == ("b1", "b2", "b3")
    assert r.arms == ("rlm", "b1", "b2", "b3")
    assert r.margin == 3 and r.escalation_band == (1, 2, 3) and r.tripwire_floor == 3
    assert r.abstentions == {} and r.scored_stream is None


def test_v2_rules_are_read_from_the_manifest():
    r = BenchmarkRules.for_manifest(_M({
        "rlm_arm": "rlm", "baselines": ["rlm-nosubcalls", "b2"], "margin": 2,
        "escalation_band": [1, 2], "abstentions": {"b2": ["interactive"]},
        "scored_stream": "train", "n_tasks": 16}))
    assert r.baselines == ("rlm-nosubcalls", "b2") and r.margin == 2
    assert r.escalation_band == (1, 2)
    assert r.abstains("b2", "interactive") and not r.abstains("b2", "linear_semantic")
    assert not r.abstains("rlm", "interactive")
    assert r.scored_stream == "train" and r.n_tasks == 16


def test_unknown_rule_keys_are_refused():
    import pytest
    with pytest.raises(ValueError, match="unknown rule"):
        BenchmarkRules.for_manifest(_M({"margn": 2}))
```

- [ ] **Step 2: Run** — Expected: `ModuleNotFoundError: No module named 'bench.rules'`.

- [ ] **Step 3: Implement `bench/rules.py`**

```python
"""§8's pre-registered scoring rules as DATA, read from the benchmark manifest.

Until 2026-09-02 every one of these lived as a module constant in
`src/rlm/measure/verdict.py` (MARGIN_GATE = 3, BASELINES = ("b1","b2","b3"), the
{+1,+2,+3} band, the >=3 zero-floor). v2 needs different values and a per-arm
abstention, and v1's frozen record needs the old ones forever. So the values
move here, the manifest may carry a `rules` block, and a manifest without one
(v1) gets exactly the constants it was scored under.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_KNOWN = {"rlm_arm", "baselines", "margin", "escalation_band", "tripwire_floor",
          "abstentions", "scored_stream", "n_tasks"}


@dataclass(frozen=True)
class BenchmarkRules:
    rlm_arm: str = "rlm"
    baselines: tuple[str, ...] = ("b1", "b2", "b3")
    margin: int = 3
    escalation_band: tuple[int, ...] = (1, 2, 3)
    tripwire_floor: int = 3
    abstentions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scored_stream: str | None = None
    n_tasks: int | None = None

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.rlm_arm, *self.baselines)

    def abstains(self, arm: str, category: str) -> bool:
        return category in self.abstentions.get(arm, ())

    @classmethod
    def for_manifest(cls, manifest) -> "BenchmarkRules":
        raw = getattr(manifest, "rules", None) or {}
        unknown = set(raw) - _KNOWN
        if unknown:
            raise ValueError(f"unknown rule key(s) {sorted(unknown)}; known: {sorted(_KNOWN)}")
        kw = dict(raw)
        if "baselines" in kw:
            kw["baselines"] = tuple(kw["baselines"])
        if "escalation_band" in kw:
            kw["escalation_band"] = tuple(int(x) for x in kw["escalation_band"])
        if "abstentions" in kw:
            kw["abstentions"] = {a: tuple(c) for a, c in kw["abstentions"].items()}
        return cls(**kw)


__all__ = ["BenchmarkRules"]
```

- [ ] **Step 4: Run** `uv run pytest checks/test_rules.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: the scorer's constants become BenchmarkRules, defaulting to what v1 was scored under"`

---

### Task 3: The verdict reads its rules from the manifest (v1 unchanged)

**Files:**
- Modify: `src/rlm/measure/verdict.py:61-70, 399-405, 447-466, 482, 488-489, 505-535, 544, 1004, 1120-1137`
- Modify: `src/rlm/measure/stats.py:44-48`
- Test: `checks/test_verdict.py` (existing must pass), add three tests

**Interfaces:**
- `decide(grid, manifest)` now calls `rules = _rules_for(manifest)` (lazy `from bench.rules import BenchmarkRules`) and uses `rules.rlm_arm`, `rules.baselines`, `rules.margin`, `rules.escalation_band`, `rules.tripwire_floor` everywhere the module constants were used. `MARGIN_GATE`, `BASELINES`, `ARMS`, `RLM_ARM` **stay** as v1-valued module constants (other modules import them) but `decide` no longer reads them.
- `stats.needs_escalation(margin: int, band: tuple[int, ...] = (1, 2, 3)) -> bool`.
- Report text prints `+{rules.margin}` instead of `+{MARGIN_GATE}`.

- [ ] **Step 1: Failing tests** (append to `checks/test_verdict.py`; reuse that file's fixtures `_grid`/`_manifest` helpers — read lines 150-215 first to match their signatures)

```python
def test_a_v2_manifest_scores_under_its_own_margin_and_baselines(six_task_manifest_factory):
    """Same grid, two manifests: v1 rules need +3 against b1/b2/b3; a manifest
    whose rules say margin 2 against rlm-nosubcalls and b2 passes at +2."""
    m = six_task_manifest_factory(rules={"baselines": ["rlm-nosubcalls", "b2"],
                                         "margin": 2, "escalation_band": [1, 2]})
    grid = grid_with_passes(rlm=6, **{"rlm-nosubcalls": 4, "b2": 3})  # +2 and +3
    v = decide(grid, m)
    assert all(p.beats for p in v.pairs.values())
    assert v.gate is True
    assert "+2" in render_report(v, m) and "+3-task threshold" not in render_report(v, m)


def test_v1_manifest_still_needs_plus_three(six_task_manifest_factory):
    m = six_task_manifest_factory(rules=None)
    grid = grid_with_passes(rlm=6, b1=4, b2=3, b3=3)   # +2, +3, +3
    v = decide(grid, m)
    assert v.pairs["b1"].beats is False and v.pairs["b2"].beats is True


def test_needs_escalation_takes_the_band_from_rules():
    from rlm.measure.stats import needs_escalation
    assert needs_escalation(3) and not needs_escalation(3, band=(1, 2))
    assert needs_escalation(2, band=(1, 2))
```

If `six_task_manifest_factory`/`grid_with_passes` do not exist, build them from the file's existing `_manifest(...)`/`_grid(...)` helpers (they construct a `BenchmarkManifest` and a `Grid` from pass tables) — read `checks/test_verdict.py:150-215` and follow the same construction; the factory must accept `rules=`.

- [ ] **Step 2: Run** — FAIL: `decide` ignores `rules` (v2 test asserts `beats` True at +2, gets False).

- [ ] **Step 3: Implement**

In `verdict.py`, add near the constants:

```python
def _rules_for(manifest):
    """§8's rules for THIS manifest. Lazy import: `bench` is the artifact, not
    the package, and `checks/test_import_rules.py` forbids it at module scope."""
    from bench.rules import BenchmarkRules
    return BenchmarkRules.for_manifest(manifest)
```

In `decide(grid, manifest)`: `rules = _rules_for(manifest)` at the top; replace every `RLM_ARM` → `rules.rlm_arm`, `BASELINES` → `rules.baselines`, `MARGIN_GATE` → `rules.margin`, the tripwire `>= 3` → `>= rules.tripwire_floor`, `needs_escalation(margin)` → `needs_escalation(margin, band=rules.escalation_band)`. Store `rules` on the `Verdict` (add a field `rules: Any = None`) so `render_report` prints `+{v.rules.margin}` and the band text from `v.rules.escalation_band`. Update the `partial_grid` finding text to use `rules.margin` and `rules.n_tasks or len(manifest.tasks)`.

In `stats.py`:

```python
def needs_escalation(margin: int, band: tuple[int, ...] = (1, 2, 3)) -> bool:
    """Escalate iff the net margin lands in the pre-registered band (v1: {+1,+2,+3};
    read from the manifest's rules since v2)."""
    return margin in band
```

- [ ] **Step 4: Run** `uv run pytest checks/test_verdict.py -q -p no:cacheprovider` — all pass (the existing +3 tests still pass because v1 defaults are +3).
- [ ] **Step 5: Commit** — `git commit -m "verdict: the margin, the baselines and the band come from the manifest's rules, and v1 keeps +3"`

---

### Task 4: Abstention — an arm may skip a category, and the grid must not call it a hole

**Files:**
- Modify: `src/rlm/measure/verdict.py:216-310` (`load_grid`), `:447-500` (pair math)
- Modify: `src/rlm/measure/bench.py:601-699` (`run_block`, `_run_cell`), `:98-111` (ledger)
- Test: `checks/test_verdict.py`, `checks/test_bench.py`

**Interfaces:**
- `load_grid(db_path, run_id, *, seeds, escalation_seeds, arms=None, abstentions=None)` — `abstentions: dict[str, tuple[str, ...]]` maps arm → categories; cells whose `(arm, task.category)` is abstained are not "missing" and are absent from `grid.cells`. The caller (`cli._bench`) passes `rules.abstentions` with `categories` from the manifest.
- `decide`: for a baseline with abstentions, its pair is computed over the tasks it ran; `PairResult` gains `n_tasks: int` (the pair's denominator) and the report prints "over N tasks" when `n_tasks != len(scored)`.
- `bench.run_block`: for an abstained `(arm, task)` write a ledger row `{... "episode_id": None, "outcome": "abstained", "reason": "abstained:<category>", ...}` and open no episode. `"abstained"` is a ledger value only — not an `Outcome`. `BenchLedger.completed` treats it as decided.

- [ ] **Step 1: Failing tests**

```python
# checks/test_verdict.py (append)
def test_an_abstained_arm_is_not_a_missing_cell(store_with_grid):
    """B2 does not run the interactive category. Its absence there is an
    abstention declared in the rules, not a hole in the grid."""
    db, run_id, manifest = store_with_grid(
        categories={"t1": "linear_semantic", "t2": "interactive"},
        cells={("t1", "rlm"): [True]*3, ("t2", "rlm"): [True]*3,
               ("t1", "b2"): [False]*3},          # no ("t2","b2") cells at all
        rules={"baselines": ["b2"], "margin": 2, "escalation_band": [1, 2],
               "abstentions": {"b2": ["interactive"]}})
    grid = load_grid(db, run_id, seeds=[1, 2, 3], escalation_seeds=[4, 5],
                     abstentions={"b2": ("interactive",)},
                     categories={"t1": "linear_semantic", "t2": "interactive"})
    v = decide(grid, manifest)
    assert v.pairs["b2"].n_tasks == 1          # denominator is the tasks B2 ran
    assert v.pairs["b2"].margin == 1           # rlm 1/1 vs b2 0/1 on the shared task


def test_a_genuinely_missing_cell_is_still_refused(store_with_grid):
    db, run_id, manifest = store_with_grid(
        categories={"t1": "linear_semantic"},
        cells={("t1", "rlm"): [True]*3}, rules={"baselines": ["b2"]})
    with pytest.raises(VerdictError, match="missing"):
        load_grid(db, run_id, seeds=[1, 2, 3], escalation_seeds=[4, 5],
                  categories={"t1": "linear_semantic"})
```

```python
# checks/test_bench.py (append)
async def test_an_abstained_cell_writes_a_ledger_row_and_opens_no_episode(bench_ctx_factory):
    ctx = bench_ctx_factory(arms=("rlm", "b2"),
                            rules={"abstentions": {"b2": ["interactive"]}})
    block = ctx.block_for(task_id="int-01", category="interactive", seed=1)
    records = await run_block(ctx, block, arms=("rlm", "b2"))
    b2 = [r for r in records if r["arm"] == "b2"][0]
    assert b2["outcome"] == "abstained" and b2["episode_id"] is None
    assert b2["reason"] == "abstained:interactive"
    assert ctx.arm_runners["b2"].calls == 0
```

Build `store_with_grid`/`bench_ctx_factory` on the existing fixtures in those files (`checks/test_verdict.py:150-215` writes a DuckDB store from pass tables; `checks/test_bench.py:80-200` has `runners(...)` and `_entry(...)`). The `ctx` must expose `rules` (see Step 3).

- [ ] **Step 2: Run** — FAIL (`VerdictError: missing cells` / `run_block` calls the runner).

- [ ] **Step 3: Implement**

`verdict.load_grid`: add kwargs `abstentions: dict[str, tuple[str, ...]] | None = None, categories: dict[str, str] | None = None`; when computing `missing`, skip `(task, arm)` where `abstentions and categories and categories[task] in abstentions.get(arm, ())`. Record `grid.abstained: set[tuple[str, str]]`.

`verdict.decide`: for each baseline, `shared = [t for t in tasks if (t, baseline) not in grid.abstained]`; compute passes/margin/discordant over `shared`; `PairResult(..., n_tasks=len(shared))`. Add `n_tasks: int` to `PairResult` (default `0` for callers that construct it directly). In `render_report`, when `pair.n_tasks != len(tasks)` append `f" (over {pair.n_tasks} tasks; abstains from {', '.join(cats)})"`.

`bench.py`: give `BenchContext` a `rules` attribute (default `BenchmarkRules()`; `cli._bench` sets it from the manifest). In `run_block`, before `_run_cell`:

```python
        if ctx.rules.abstains(arm, entry.category):
            record = _abstention(ctx, block, arm)
            records.append(record)
            continue
```

```python
def _abstention(ctx, block, arm: str) -> dict[str, Any]:
    """A cell the rules say this arm does not run. No episode, one ledger row,
    so a resume knows it is decided and the verdict knows it is not a hole."""
    record = {"run_id": ctx.run_id, "block": block.idx, "task_id": block.task_entry.task_id,
              "seed": block.seed, "arm": arm, "episode_id": None,
              "outcome": "abstained", "reason": f"abstained:{block.task_entry.category}",
              "wall_s": 0.0, "relaunch_s": 0.0, "superseded_by": None, "ts": _utc_ts()}
    ctx.ledger.append(record)
    return record
```

`BenchLedger.completed`: count `outcome == "abstained"` as decided (it has no episode; `_by_cell` today drops rows without an episode — add the abstained case explicitly).

- [ ] **Step 4: Run** `uv run pytest checks/test_verdict.py checks/test_bench.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench+verdict: an arm may abstain from a category by rule, and the grid stops calling that a hole"`

---

### Task 5: The `rlm-nosubcalls` arm exists as a name

**Files:**
- Modify: `src/rlm/measure/bench.py:75, 82-85, 225-226`
- Modify: `src/rlm/cli.py:2186-2188` (help text), `:1560-1561` (runner map — the runner itself lands in Task 9; register a placeholder that raises `ConfigError("rlm-nosubcalls lands in Task 9")` so `_resolve_arms` accepts the name)
- Test: `checks/test_delegation_arm.py:29-34`, `checks/test_bench.py:354-362, 442, 613-633, 1539-1540`, `checks/test_cli.py:928`

**Interfaces:**
- `ARM_ORDER = ("rlm", "rlm-restricted", "rlm-nosubcalls", "b2", "b1", "b3")`; `ARM_PROFILE["rlm-nosubcalls"] = RESIDENT_PROFILE`; `bench_extra` stamps `arm` for `("rlm", "rlm-restricted", "rlm-nosubcalls")`.

- [ ] **Step 1: Update the pinned-order tests first (they are the spec of the order)** — in `checks/test_delegation_arm.py:33`, `checks/test_bench.py:356-361`, `:1539`, and `checks/test_cli.py:928` change the literal tuples/lists to include `"rlm-nosubcalls"` after `"rlm-restricted"`. Add to `checks/test_bench.py`:

```python
def test_the_nosubcalls_arm_is_resident_and_stamps_its_own_arm_key():
    assert ARM_PROFILE["rlm-nosubcalls"] == RESIDENT_PROFILE
    assert bench_extra("r", 0, 1, "rlm-nosubcalls")["arm"] == "rlm-nosubcalls"
```

- [ ] **Step 2: Run** — FAIL on the tuple literal.
- [ ] **Step 3: Implement** the three edits in `bench.py`; update the help string; add the placeholder runner in `bench_arm_runners`' return dict.
- [ ] **Step 4: Run** `uv run pytest checks/test_bench.py checks/test_delegation_arm.py checks/test_cli.py -q -p no:cacheprovider` — PASS (the `test_a_full_block_relaunches_the_leaf_at_most_twice` count comment at `:393` may need "Four resident handshakes").
- [ ] **Step 5: Commit** — `git commit -m "bench: rlm-nosubcalls is a registered resident arm; its runner lands with the refusal"`

---

### Task 6: One manifest per benchmark version, and the scored stream is the default task set

**Files:**
- Modify: `src/rlm/cli.py:1283-1287, 1326-1347, 1834-1839, 1710-1746`
- Modify: `src/rlm/measure/bench.py` (export `bench_manifest_path`)
- Test: `checks/test_cli.py`

**Interfaces:**
- `bench_manifest_path(version: str | None) -> Path`: `None`/`"v1"` → `bench/manifest.json`; else `bench/manifest.{version}.json`.
- `load_benchmark_manifest(cfg)` takes the config (was argless) and uses `cfg.benchmark.version`.
- `_bench`: `task_ids` default = `[t.task_id for t in manifest.scored_tasks()]`; `--tasks` may name any task in the manifest (held-out too — the S6 gate will need it) but the *default* is the scored stream. `run_escalation` iterates `rules.baselines`.

- [ ] **Step 1: Failing test**

```python
# checks/test_cli.py (append)
def test_the_manifest_path_follows_the_benchmark_version():
    from rlm.measure.bench import bench_manifest_path, REPO_ROOT
    assert bench_manifest_path(None) == REPO_ROOT / "bench" / "manifest.json"
    assert bench_manifest_path("v1") == REPO_ROOT / "bench" / "manifest.json"
    assert bench_manifest_path("v2") == REPO_ROOT / "bench" / "manifest.v2.json"


def test_the_default_task_set_is_the_scored_stream(v2_manifest_file, v2_config_file):
    """A v2 grid without --tasks runs the train stream only; held-out is never
    spent by accident."""
    args = _bench_argv(config=v2_config_file, arms="rlm")
    ids = default_task_ids(args)     # the helper `_bench` uses
    assert all(i.endswith("-train") for i in ids) and len(ids) == 16
```

(`v2_manifest_file`/`v2_config_file` are small fixtures writing a two-stream manifest and a config pointing at it; the real ones arrive in Tasks 18/21. Extract the default-task computation in `_bench` into `default_task_ids(manifest, args)` so it is testable.)

- [ ] **Step 2: Run** — FAIL (`bench_manifest_path` undefined).
- [ ] **Step 3: Implement** — in `bench.py`:

```python
def bench_manifest_path(version: str | None) -> Path:
    """§8's freeze, per version. v1 keeps its historical filename."""
    if version in (None, "v1"):
        return REPO_ROOT / "bench" / "manifest.json"
    return REPO_ROOT / "bench" / f"manifest.{version}.json"
```

In `cli.py`: `BENCH_MANIFEST_PATH` becomes a call to `bench_manifest_path(cfg.benchmark.version)` inside `load_benchmark_manifest(cfg)`; update the two callers (`cmd_bench`, `_bench`). Replace `task_ids or [t.task_id for t in manifest.tasks]` with `manifest.scored_tasks()`. In `run_escalation`, replace `BASELINES` with `_rules_for(manifest).baselines` (import the helper from `verdict`). Store `rules` on the ctx (`ctx.rules = BenchmarkRules.for_manifest(manifest)`) and pass `abstentions`/`categories` into `load_grid`.

- [ ] **Step 4: Run** `uv run pytest checks/test_cli.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "cli: the manifest is chosen by benchmark.version, and a grid defaults to the scored stream"`

**Phase A gate:** `uv run pytest -q` — 890 + new tests pass; `bench/manifest.json` unchanged (`git diff --stat` shows no change to it).

---

## Phase B — prompts and the `rlm-nosubcalls` arm

### Task 7: Strategy templates become an open mapping, with a per-arm `nosubcalls` set

**Files:**
- Modify: `src/rlm/config.py:291-296, 325-344, 681-707, 722-763, 1001-1033`
- Modify: `src/rlm/cli.py` (the `episode_config` replay rebuild that enumerates the five slots — grep `strategy_templates.needle`)
- Test: `checks/test_config.py`, `checks/test_prompts.py`, `checks/test_delegation_arm.py:172-212`

**Interfaces:**
- `PromptsCfg.strategy_templates: dict[str, PromptRef]` (validator: must contain `"default"`); `PromptsCfg.strategy_templates_nosubcalls: dict[str, PromptRef] | None = None`.
- `PromptRegistry.render_root(category, *, restricted=False, no_subcalls=False)`: `no_subcalls` selects `root_nosubcalls` body **and** the block from `strategy_templates_nosubcalls[category]`; refuses loudly (`ConfigError`) when either is missing — same rationale as `restricted`.
- `PromptsCfg.root_nosubcalls: PromptRef | None = None`.
- `PromptRegistry.from_files(..., root_nosubcalls_path=None, strategy_nosubcalls_paths=None, strategy_nosubcalls_sha256=None)`; hashes keyed `strategy_nosubcalls.<category>.file/.body`, `root_nosubcalls.file/.body`.
- The four-enumeration rule (`config.py:681-689`) collapses to iteration over the dict; delete the `StrategyTemplates` class.

- [ ] **Step 1: Failing tests**

```python
# checks/test_config.py (append)
def test_strategy_templates_accept_any_category_key_but_require_default(valid_cfg_dict):
    d = deepcopy(valid_cfg_dict)
    d["scaffold"]["prompts"]["strategy_templates"]["linear_semantic"] = {
        "path": "prompts/strat-needle.v1.md"}
    cfg = Config.model_validate(d)
    assert "linear_semantic" in cfg.scaffold.prompts.strategy_templates
    del d["scaffold"]["prompts"]["strategy_templates"]["default"]
    with pytest.raises(ValidationError, match="default"):
        Config.model_validate(d)


def test_render_root_no_subcalls_needs_both_the_body_and_the_block_set(valid_cfg):
    reg = valid_cfg.prompt_registry().load()
    with pytest.raises(ConfigError, match="root_nosubcalls"):
        reg.render_root("needle", no_subcalls=True)
```

```python
# checks/test_prompts.py (append; runs against a cfg dict that pins the nosubcalls files once Task 8 lands — mark xfail(strict=False) until then, remove the mark in Task 8)
def test_render_root_no_subcalls_uses_the_nosubcalls_body_and_block(nosubcalls_cfg):
    reg = nosubcalls_cfg.prompt_registry().load()
    text = reg.render_root("needle", no_subcalls=True)
    assert "llm_query" not in text and "sub-call" not in text.lower()
```

- [ ] **Step 2: Run** — FAIL (`StrategyTemplates` rejects extra keys).
- [ ] **Step 3: Implement** — replace the class with the dict + validator; thread the two new optional fields; generalize `_prompt_refs` (iterate `strategy_templates.items()` → `("strategy." + k, ref)`, likewise the nosubcalls set), `prompt_registry` (`strategy_paths={k: v.path ...}`), `PromptRegistry.load` (load `strategy_nosubcalls.<k>`), `render_root`:

```python
    def render_root(self, category: str, *, restricted: bool = False,
                    no_subcalls: bool = False) -> str:
        self._ensure_loaded()
        if restricted and no_subcalls:
            raise ConfigError("restricted and no_subcalls are different arms; pick one")
        blocks = self._strategy_bodies
        body = self._root_body
        if restricted:
            ...  # unchanged
        if no_subcalls:
            if self._root_nosubcalls_body is None or not self._strategy_nosubcalls_bodies:
                raise ConfigError(
                    "the rlm-nosubcalls arm needs scaffold.prompts.root_nosubcalls AND "
                    "scaffold.prompts.strategy_templates_nosubcalls; falling back to the "
                    "pinned prompts would teach a root an `llm_query` this arm refuses")
            body, blocks = self._root_nosubcalls_body, self._strategy_nosubcalls_bodies
        if category not in blocks:
            raise ConfigError(f"{category!r} is not a declared strategy category; the model "
                              "never chooses its own strategy (spec §5, I1)")
        return f"{body}\n\n{blocks[category]}"
```

Update `cli.episode_config`'s rebuild to iterate the dict. `config.yaml` needs no edit (its five keys are a valid dict).

- [ ] **Step 4: Run** `uv run pytest checks/test_config.py checks/test_prompts.py checks/test_delegation_arm.py checks/test_cli.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "config: strategy templates are a mapping, and the nosubcalls arm gets its own body and block set"`

---

### Task 8: The two root bodies — `root.v4.md` and `root-nosubcalls.v1.md`

**Files:**
- Create: `src/rlm/_data/prompts/root.v4.md`, `src/rlm/_data/prompts/root-nosubcalls.v1.md`
- Test: `checks/test_prompts.py`

**Interfaces:** the exact derivations below. `checks/test_prompts.py:13-22` `FILES` gains both names (header test at `:35` then covers them).

- [ ] **Step 1: Failing tests**

```python
# checks/test_prompts.py (append)
def _body(name):  # header stripped, like the loader
    from rlm.config import _strip_changelog
    return _strip_changelog((PROMPTS / name).read_text(encoding="utf-8"))


def test_v4_is_v3_with_exactly_line_37_changed():
    v3, v4 = _body("root.v3.md").splitlines(), _body("root.v4.md").splitlines()
    assert len(v3) == len(v4)
    diff = [(a, b) for a, b in zip(v3, v4) if a != b]
    assert len(diff) == 1
    old, new = diff[0]
    assert old.startswith("`llm_query` reaches a small, fast, stateless model.")
    assert "the same model as you" in new and "no REPL, no memory between calls" in new


def test_nosubcalls_body_describes_a_repl_with_no_sub_model():
    body = _body("root-nosubcalls.v1.md")
    for banned in ("llm_query", "sub-model", "sub-call", "Sub-call", "delegat", "leaf"):
        assert banned not in body, banned
    assert "final_answer(value)" in body and "`chunks: list[str]`" in body
    assert body.rstrip().endswith(
        "A strategy block for this task's declared category follows. The scaffold "
        "selected it from the task's category; you do not choose it, and where it is "
        "more specific than the tips above, it wins.")


def test_nosubcalls_body_is_v4_minus_only_the_sub_call_lines():
    v4 = [l for l in _body("root.v4.md").splitlines() if l.strip()]
    ns = [l for l in _body("root-nosubcalls.v1.md").splitlines() if l.strip()]
    kept = [l for l in ns if l in v4]
    renumbered = [l for l in ns if l not in v4]
    # everything not in v4 is a renumbered tip (same text after the "N. ")
    assert all(any(l.split(". ", 1)[1] == k.split(". ", 1)[1] for k in v4 if ". " in k)
               for l in renumbered if ". " in l), renumbered
    assert len(kept) + len(renumbered) == len(ns)
```

- [ ] **Step 2: Run** — FAIL (files missing).
- [ ] **Step 3: Author the two files.**

`root.v4.md`: copy `root.v3.md` byte-for-byte, then (a) header line 1 → `<!-- changelog (prompts/root.v4.md)`; (b) append a changelog line after the v3 line:

```
v4 | 2026-09-02 | v4 = root.v3.md with ONE sentence changed, in "# The sub-model": ":37" said `llm_query` "reaches a small, fast, stateless model", which is false under the same-model configuration benchmark v2 runs (spec 2026-08-25-benchmark-v2-design.md §14.3, D-B7: root and leaf are both Qwen3.6-35B-A3B). Every other byte is identical to root.v3.md, the brake (Budgets' last sentence, tips 1-2) included. root.v3.md is NOT modified: it is the S4 re-validation's pinned prompt.
```

(c) replace body line 37 with:

```
`llm_query` reaches a sub-model — in this configuration the same model as you, but with no REPL, no memory between calls, and no knowledge of your task beyond the string you hand it. It sees exactly one thing: your prompt.
```

`root-nosubcalls.v1.md`: from v4's body, remove: line 25 (the `llm_query` bullet), lines 35–49 (the whole `# The sub-model` section, heading included), line 53's whole paragraph replaced by `Tokens and wall-clock are capped per episode by the scaffold. The caps are enforced, not advisory; you cannot raise them and asking for more has no effect. A breach kills the episode with no answer at all.`, line 55 (`The sub-model cannot delegate further…`), tips 1, 2, 3, 4, 5, 6 (lines 59–64), and in line 9 replace `You act by writing code that inspects those objects, and by delegating chunk-level reading to a cheap sub-model. You are an orchestrator, not a reader.` with `You act by writing code that inspects those objects. You are a programmer over the context, not a reader of it.` Renumber the surviving tips (old 7 → 1, old 8 → 2). Keep the closing sentence last. Header:

```
<!-- changelog (prompts/root-nosubcalls.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | The `rlm-nosubcalls` arm's root body (benchmark v2, spec §6 and §14.3): root.v4.md with everything that names or teaches the sub-model removed — the `llm_query` API bullet, the "# The sub-model" section, the sub-call clause of "# Budgets", "The sub-model cannot delegate further", and tips 1-6 — so the prompt describes a REPL holding `context`, `chunks` and `final_answer` and nothing more. Tips 7-8 are renumbered 1-2. The runtime refuses `llm_query` in this arm (episode.py, `no_subcalls=True`); this file makes the prompt agree with the runtime. Pinned only in config.v2.yaml.
NOTE: the registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->
```

- [ ] **Step 4: Run** `uv run pytest checks/test_prompts.py -q -p no:cacheprovider` — PASS. Record both files' `sha256sum` in the commit message; they are pinned in Task 21.
- [ ] **Step 5: Commit** — `git commit -m "prompts: root.v4 tells the truth about the sub-model, and root-nosubcalls.v1 has none"`

---

### Task 9: `run_episode(no_subcalls=True)` — the refusal, its step, and the arm runner

**Files:**
- Modify: `src/rlm/episode.py:436-476` (ctor), `:539-601` (`_on_llm_query`), `:973-976` (prompt), `:1209` (snapshot), `:1301-1354` (`run_episode`)
- Modify: `src/rlm/cli.py:1560-1561` (replace the Task 5 placeholder)
- Test: `checks/test_delegation_arm.py` (append), `checks/test_episode.py`

**Interfaces:**
- `run_episode(..., no_subcalls: bool = False)`. In `_on_llm_query`, **before** `count_tokens`/`admit`:
```python
        if self.no_subcalls:
            idx = self._alloc()
            self._put({"step_idx": idx, "parent_step_idx": self._parent_idx, "depth": 1,
                       "actor": Actor.LEAF, "action_type": ActionType.LLM_CALL,
                       "status": StepStatus.REJECTED, "call_id": str(uuid.uuid4()),
                       "retry_idx": 0, "action_payload": question or "",
                       "error_detail": NO_SUBCALLS_REFUSED})
            raise RlmError(NO_SUBCALLS_REFUSED)
```
with `NO_SUBCALLS_REFUSED = "llm_query is not available in this arm (rlm-nosubcalls): there is no sub-model; solve in code over `context` and `chunks`"`. Nothing is charged to `max_subcalls` (raised before `admit`), the attempt is on the record as a `rejected` step (C4's pre-flight-rejection shape, `dispatcher.py:1505-1515`), and the root sees the message in its traceback.
- Prompt: `render_root(self.task.category, restricted=self.restrict_chunks, no_subcalls=self.no_subcalls)`. Snapshot: `"no_subcalls": self.no_subcalls`.
- `cli.rlm_nosubcalls_arm`: identical to `rlm_arm` (no `_virgin_resident_pool` — it never calls the leaf) plus `no_subcalls=True`.

- [ ] **Step 1: Failing tests**

```python
# checks/test_delegation_arm.py (append)
NOSUB_CELL = ("```repl\n"
              "try:\n"
              "    print(await llm_query('Q?', chunk=chunks[0]))\n"
              "except Exception as e:\n"
              "    print('REFUSED', type(e).__name__, str(e)[:60])\n"
              "print(len(chunks), chunks[0][:20])\n"
              "```")


async def test_no_subcalls_refuses_llm_query_and_leaves_chunks_readable(episode_env, mock_server, nosubcalls_cfg):
    env = episode_env(root_script=[NOSUB_CELL, FINAL], answer="42",
                      leaf_port=mock_server.port, cfg=nosubcalls_cfg,
                      dispatcher=mock_server.dispatcher(parallel=2, slot_pool=2),
                      no_subcalls=True)
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    obs = env.observation(step=0)
    assert "REFUSED RemoteError RlmError: llm_query is not available in this arm" in obs
    assert mock_server.completion_count == 0            # nothing reached the leaf
    calls = [s for s in env.steps() if s["action_type"] == "llm_call"]
    assert [s["status"] for s in calls] == ["rejected"]
    assert "not available in this arm" in calls[0]["error_detail"]
    assert env.snapshot()["no_subcalls"] is True
    assert env.episode_row()["outcome_reason"] is None    # no budget breach, nothing charged


async def test_no_subcalls_renders_the_nosubcalls_prompt(episode_env, mock_server, nosubcalls_cfg):
    env = episode_env(root_script=[FINAL], answer="42", leaf_port=mock_server.port,
                      cfg=nosubcalls_cfg, dispatcher=mock_server.dispatcher(), no_subcalls=True)
    await env.run()
    system = env.root_system_prompt()
    assert "llm_query" not in system and "# Strategy" in system
```

Add a `nosubcalls_cfg` fixture in `checks/conftest.py` that takes `minimal_cfg_dict`, sets `scaffold.prompts.root_nosubcalls = {"path": "prompts/root-nosubcalls.v1.md"}` and `strategy_templates_nosubcalls = {"default": {"path": "prompts/strat-default.v1.md"}, "needle": {...same...}}` (a real nosubcalls needle block arrives in Task 20; for this test any block without `llm_query` works — write a 3-line temp block into `tmp_path` and point at it). `episode_env` must accept `cfg=` and `no_subcalls=` — extend the fixture (it already forwards `restrict_chunks`).

- [ ] **Step 2: Run** — FAIL (`run_episode() got an unexpected keyword argument 'no_subcalls'`).
- [ ] **Step 3: Implement** per the interfaces; then in `cli.py`:

```python
    async def rlm_nosubcalls_arm(task, cfg, *, bench_extra):
        """`rlm` with the sub-model removed: the paper's "RLM, no sub-calls"
        ablation as a named arm (spec 2026-08-25-benchmark-v2-design.md §6).
        Same dispatcher, same profile, readable `chunks`; `llm_query` is refused
        scaffold-side and the prompt never mentions it. No `_virgin_resident_pool`:
        this arm cannot touch the leaf."""
        try:
            return await run_episode(
                task, cfg, dispatcher=rlm_dispatcher, trace=trace,
                lifecycle=lifecycle, snapshot_extra={"bench": bench_extra},
                process_manager=orchestra.episode_process_manager(),
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha,
                benchmark_version=benchmark_version,
                no_subcalls=True)
        finally:
            reset_dispatcher_steps(rlm_dispatcher)
```

- [ ] **Step 4: Run** `uv run pytest checks/test_delegation_arm.py checks/test_episode.py checks/test_cli.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "episode: the rlm-nosubcalls arm refuses llm_query before admission, logs the attempt as rejected, and reads a prompt that never taught it"`

**Phase B gate:** full suite green.

---

## Phase C — the `env` verb (spec §4, §9; highest risk)

### Task 10: `env_call` is a step type — enum, migration, transcript

**Files:**
- Modify: `src/rlm/errors.py:39-44`, `src/rlm/trace/schema.sql:16-18, 90-96`, `src/rlm/trace/replay.py:219-243`
- Test: `checks/test_trace.py`, `src/rlm/_tests/` (replay tests — grep `_render_transcript`)

**Interfaces:** `ActionType.ENV_CALL = "env_call"`; DuckDB `step_action_v2` ENUM `('repl_exec','llm_call','final','env_call')` with an idempotent `ALTER TABLE steps ALTER COLUMN action_type SET DATA TYPE step_action_v2` migration in the R13 block's style; `_render_transcript` prints `ENV: <payload> -> <observation_view>` for `ENV_CALL` (and the `else: FINAL` branch becomes an explicit `elif ActionType.FINAL` + `else: raise`).

- [ ] **Step 1: Failing tests**

```python
# checks/test_trace.py (append)
async def test_an_env_call_step_round_trips_through_the_store(tmp_path):
    from rlm.errors import ActionType, Actor, StepStatus
    trace = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs"); await trace.start()
    ep = trace.open_episode(task_id="int-01", ...)   # mirror an existing open_episode call in this file
    trace.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.ROOT,
                    "action_type": ActionType.ENV_CALL, "status": StepStatus.OK,
                    "action_payload": '{"op":"search","term":"tithe"}',
                    "observation_view": "[{'doc_id': 'd-03', 'window': 4}]"})
    await trace.drain()
    rows = trace.con.execute("select action_type from steps").fetchall()
    assert rows == [("env_call",)]


async def test_opening_a_v1_store_migrates_the_action_enum(v1_store_path):
    """An operator DB created before env_call exists must accept the new value."""
    trace = TraceLogger(v1_store_path, v1_store_path.parent / "blobs"); await trace.start()
    types = trace.con.execute("select enum_range(null::step_action)").fetchone()[0]
    assert "env_call" in types
```

```python
# src/rlm/_tests/test_replay.py (append, near the transcript tests)
def test_transcript_labels_env_calls_and_never_mislabels_them_final():
    text = _render_transcript([_step("repl_exec"), _step("env_call", payload='{"op":"open"}', view="d-01"), _step("final", payload="42")])
    assert "ENV: {\"op\":\"open\"} -> d-01" in text
    assert text.count("FINAL:") == 1
```

- [ ] **Step 2: Run** — FAIL (`AttributeError: ENV_CALL` / DuckDB conversion error).
- [ ] **Step 3: Implement** the three edits. Migration SQL (append after the R13 block):

```sql
-- v2 (2026-09-02): the interactive category's `env` verb is a fourth action.
-- ENUM evolution per the header: new type, retype the column, idempotent.
CREATE TYPE IF NOT EXISTS step_action_v2 AS ENUM ('repl_exec','llm_call','final','env_call');
ALTER TABLE steps ALTER COLUMN action_type SET DATA TYPE step_action_v2;
```

(Verify against the R13 block whether the schema runner tolerates re-running the ALTER; if it errors on an already-migrated column, guard it the way that block does.)

- [ ] **Step 4: Run** `uv run pytest checks/test_trace.py src/rlm/_tests -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "trace: env_call is a step type; the store migrates, the transcript names it"`

---

### Task 11: Child side — the `env` facade over the bridge

**Files:**
- Modify: `src/rlm/sandbox/child.py:89-101` (wire table), `:382-467` (templates + `_build_injected`), `:547-560` (`_RESERVED`); `sandbox_bootstrap/sandbox_child.py` is refreshed by `install_bootstrap` — do not hand-edit
- Test: `checks/test_sandbox_child.py`, `checks/test_sandbox_manager.py:71-98`

**Interfaces:**
- Cell-visible: `await env.search(term: str) -> list[dict]`, `await env.open(doc_id: str) -> dict`, `await env.window(doc_id: str, i: int) -> str`. Each is `await BRIDGE.request("env", {"op": "search"|"open"|"window", ...})`. Signatures closed (no `**kw`); `i` positional-or-keyword.
- `env` is a `types.SimpleNamespace` whose three attributes are functions rebuilt with `_rebuild(template, name, ns)` over the same `ns = {"__builtins__", "__name__", "BRIDGE", "LOOP"}`; `env.search.__globals__` keys are exactly `['BRIDGE', 'LOOP']`. `"env"` is in `_RESERVED`, not in `_SETTABLE_RESERVED`.
- `manager._FrameGate` already refuses any kind before handshake — add `env` to the test list.

- [ ] **Step 1: Failing tests**

```python
# checks/test_sandbox_child.py (append)
async def test_env_sends_closed_payloads_over_the_bridge(session_with_env_recorder):
    s, seen = session_with_env_recorder      # handler records (kind, payload) and answers []
    await s.exec_cell("print(await env.search('tithe'))\n"
                      "print(await env.open('d-01'))\n"
                      "print(await env.window('d-01', 3))\n")
    assert seen == [("env", {"op": "search", "term": "tithe"}),
                    ("env", {"op": "open", "doc_id": "d-01"}),
                    ("env", {"op": "window", "doc_id": "d-01", "i": 3})]


async def test_env_callables_reach_only_bridge_and_loop(session):
    out = await s.exec_cell("print(sorted(k for k in env.search.__globals__ if not k.startswith('__')))")
    assert out.stdout.strip() == "['BRIDGE', 'LOOP']"


async def test_env_is_reserved_plumbing_and_reinjected(session):
    await s.exec_cell("env = 'hijacked'")
    out = await s.exec_cell("print(type(env).__name__)")
    assert out.stdout.strip() == "SimpleNamespace"
    with pytest.raises(SandboxError, match="plumbing"):
        await s.setvar("env", 1)


async def test_env_takes_no_extra_keywords(session):
    out = await s.exec_cell("await env.search('x', limit=5)")
    assert "TypeError" in out.traceback
```

```python
# checks/test_sandbox_manager.py (extend test_frames_before_the_handshake_are_refused)
    # add ("env", {"op": "open", "doc_id": "d"}) to the list of kinds asserted refused
```

- [ ] **Step 2: Run** — FAIL (`NameError: env`).
- [ ] **Step 3: Implement** in `child.py`:

```python
async def _env_search_template(term):
    """Locate. Returns hits — document id, window index, offset — never text.
    The corpus behind `env` lives scaffold-side; this is the only way in."""
    return await BRIDGE.request("env", {"op": "search", "term": term})


async def _env_open_template(doc_id):
    """Structure and length of one document — window count, chars — not content."""
    return await BRIDGE.request("env", {"op": "open", "doc_id": doc_id})


async def _env_window_template(doc_id, i):
    """One bounded slice of one document. Capped scaffold-side like any observation."""
    return await BRIDGE.request("env", {"op": "window", "doc_id": doc_id, "i": i})


def _build_injected() -> tuple:
    """(llm_query, final_answer, env) over a namespace that cannot reach this module."""
    ns = {"__builtins__": builtins, "__name__": "rlm_sandbox",
          "BRIDGE": BRIDGE, "LOOP": LOOP}
    env = types.SimpleNamespace(
        search=_rebuild(_env_search_template, "search", ns),
        open=_rebuild(_env_open_template, "open", ns),
        window=_rebuild(_env_window_template, "window", ns))
    return (_rebuild(_llm_query_template, "llm_query", ns),
            _rebuild(_final_answer_template, "final_answer", ns), env)


llm_query, final_answer, env = _build_injected()
```

Add `"env": env` to `_RESERVED`. Extend the wire-table docstring with `env {"op":"search","term"} | {"op":"open","doc_id"} | {"op":"window","doc_id","i"} -> reply is the result`. Run `install_bootstrap`'s refresh (whatever `checks/conftest.py`'s session fixture does) so the staged copy matches.

- [ ] **Step 4: Run** `uv run pytest checks/test_sandbox_child.py checks/test_sandbox_manager.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "sandbox: env is a fourth reserved name, three closed calls over the bridge, built like llm_query"`

---

### Task 12: Parent side — `InteractiveIndex`, `_on_env`, and the corpus that never crosses

**Files:**
- Create: `src/rlm/context/interactive.py`
- Modify: `src/rlm/sandbox/manager.py:157-197, 309-328`; `src/rlm/episode.py:436-476, 900-976, 1301-1354`
- Test: `checks/test_interactive_index.py` (create), `checks/test_episode.py`, `checks/test_sandbox_manager.py` (the I1 analogue)

**Interfaces:**
- `src/rlm/context/interactive.py`:
```python
DOC_DELIM = "\n\n=== DOCUMENT "     # builder writes "=== DOCUMENT d-07: <title> ===" lines
@dataclass(frozen=True)
class Hit: doc_id: str; window: int; offset: int
class InteractiveIndex:
    @classmethod
    def from_text(cls, text: str, chunk_cfg: ChunkConfig, count_tokens) -> "InteractiveIndex"
    docs: dict[str, str]                       # doc_id -> body
    windows: dict[str, list[str]]              # doc_id -> C2 windows (split(), same geometry)
    def search(self, term: str, *, max_hits: int = 50) -> list[Hit]   # case-insensitive substring; capped, sorted by (doc_id, window)
    def open(self, doc_id: str) -> dict        # {"doc_id", "title", "n_windows", "n_chars"}
    def window(self, doc_id: str, i: int) -> str   # range-checked, never clamped (like resolve_chunk_ref)
    @property
    def n_docs(self) -> int
```
- `manager.SandboxSession.on_env(handler: Callable[[dict], Awaitable[object]])`; `_serve`: `if kind == "env": ... return await self._env_handler(payload)`; missing handler → `SandboxError("no env handler registered for this episode")`.
- `episode.run_episode(..., interactive: bool = False)`. When `interactive`: build `self._index = InteractiveIndex.from_text(context_text, chunk_cfg, counter)` where `chunks` are built today; `setvar("context", "")`, `setvar("chunks", [])`; register `session.on_env(self._on_env)`. When not interactive, `_on_env` raises `RlmError("env is not available for this task")`.
- `_on_env(payload)`: validate `op` and argument types scaffold-side (the `_on_llm_query` pattern at `episode.py:558-565`); dispatch to the index; `text = json.dumps(result)` for search/open, the raw string for window; `view = truncate_view(text, cap)` with `cap = cfg.scaffold.truncation_cap_chars`; write one `ENV_CALL` step (`actor=Actor.ROOT, depth=0, parent_step_idx=self._parent_idx, action_payload=json.dumps(payload), observation_view=view, status=OK`); increment `self._env_actions`; return `view`. Errors (bad doc_id, out-of-range window, unknown op) write a step with `status=ERROR, error_detail=str(exc)` and re-raise `RlmError` so the cell sees the message. Snapshot: `"interactive": True, "env_actions": self._env_actions`.

- [ ] **Step 1: Failing tests**

```python
# checks/test_interactive_index.py
from rlm.context.chunker import ChunkConfig
from rlm.context.interactive import InteractiveIndex, Hit

CFG = ChunkConfig(size_tokens=64, overhead_tokens=0, snap_to_boundary=True,
                  snap_tolerance=0.10, stride_tokens=48)
count = lambda s: (len(s) + 3) // 4

TEXT = ("=== DOCUMENT d-01: Alpha register ===\n" + "alpha " * 300 + "\n\n"
        "=== DOCUMENT d-02: Beta register ===\n" + "beta tithe barn " * 200)


def test_documents_are_split_on_the_header_and_windowed_with_c2():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    assert ix.n_docs == 2 and set(ix.docs) == {"d-01", "d-02"}
    assert ix.open("d-02") == {"doc_id": "d-02", "title": "Beta register",
                               "n_windows": len(ix.windows["d-02"]), "n_chars": len(ix.docs["d-02"])}
    assert ix.open("d-02")["n_windows"] > 3


def test_search_returns_locations_not_text_and_is_capped():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    hits = ix.search("TITHE", max_hits=5)
    assert len(hits) == 5 and all(isinstance(h, Hit) and h.doc_id == "d-02" for h in hits)
    assert ix.search("nowhere") == []


def test_window_is_range_checked_never_clamped():
    ix = InteractiveIndex.from_text(TEXT, CFG, count)
    n = ix.open("d-01")["n_windows"]
    assert ix.window("d-01", n - 1)
    import pytest
    with pytest.raises(IndexError):
        ix.window("d-01", n)
    with pytest.raises(KeyError):
        ix.window("d-99", 0)
```

```python
# checks/test_episode.py (append)
ENV_CELL = ("```repl\n"
            "print(len(context), len(chunks))\n"
            "hits = await env.search('tithe')\n"
            "print(hits[0]['doc_id'], hits[0]['window'])\n"
            "print((await env.window(hits[0]['doc_id'], hits[0]['window']))[:12])\n"
            "```")


async def test_interactive_episode_keeps_the_corpus_scaffold_side(episode_env, mock_server, interactive_task):
    env = episode_env(root_script=[ENV_CELL, FINAL], answer="42", task=interactive_task,
                      leaf_port=mock_server.port, dispatcher=mock_server.dispatcher(),
                      interactive=True)
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    obs = env.observation(step=0)
    assert obs.startswith("0 0\n")                      # context and chunks are empty
    assert "d-02 " in obs and "beta tithe" in obs
    steps = [s for s in env.steps() if s["action_type"] == "env_call"]
    assert [json.loads(s["action_payload"])["op"] for s in steps] == ["search", "window"]
    assert all(s["status"] == "ok" and s["actor"] == "root" for s in steps)
    assert env.snapshot()["env_actions"] == 2


async def test_env_is_refused_outside_an_interactive_task(episode_env, mock_server):
    env = episode_env(root_script=["```repl\nawait env.open('d-01')\n```", FINAL], answer="42",
                      leaf_port=mock_server.port, dispatcher=mock_server.dispatcher())
    await env.run()
    assert "env is not available for this task" in env.observation(step=0)


async def test_an_env_window_result_is_capped_scaffold_side(episode_env, mock_server, interactive_task, small_cap_cfg):
    env = episode_env(root_script=["```repl\nw = await env.window('d-01', 0)\nprint(len(w))\n```", FINAL],
                      answer="42", task=interactive_task, cfg=small_cap_cfg,
                      leaf_port=mock_server.port, dispatcher=mock_server.dispatcher(), interactive=True)
    await env.run()
    assert int(env.observation(step=0).split()[0]) <= small_cap_cfg.scaffold.truncation_cap_chars
```

```python
# checks/test_sandbox_manager.py (append) — THE I1 ANALOGUE
async def test_hijacked_env_cannot_alter_scaffold_side_control(manager, cfg):
    """Same guarantee as test_hijacked_llm_query_cannot_alter_scaffold_side_control:
    a hijacked stub answers locally and the scaffold's action counter and range
    checks never move."""
    served: list[dict] = []
    async def env_handler(payload):
        served.append(payload)
        if payload["op"] == "window" and payload["i"] > 1:
            raise SandboxError("window out of range")
        return "W"
    async with manager.session("ep-env", cfg) as s:
        s.on_env(env_handler)
        hijack = await s.exec_cell(
            "RESERVED = env.search.__globals__['BRIDGE']._handler.__globals__['_RESERVED']\n"
            "REAL = RESERVED['env']\n"
            "import types\n"
            "RESERVED['env'] = types.SimpleNamespace(window=lambda d, i: 'HIJACKED')\n")
        assert hijack.traceback == ""
        out = await s.exec_cell("print(sorted({env.window('d', i) for i in range(50)}))")
        assert out.stdout.strip() == "['HIJACKED']"
        assert served == []                      # 50 calls the model believes it made; none crossed
        await s.exec_cell("RESERVED['env'] = REAL")
        real = await s.exec_cell(
            "for i in range(4):\n"
            "    try:\n        print(await env.window('d', i))\n"
            "    except Exception as e:\n        print('REFUSED', type(e).__name__)\n")
    assert [p["i"] for p in served] == [0, 1, 2, 3]
    assert real.stdout.count("REFUSED") == 2 and real.stdout.count("W") == 2
```

Fixtures: `interactive_task` writes a task JSON with `category: "interactive"` (add `"interactive"` to the test config's `strategy_templates` pointing at any block) whose `context_path` is a two-document corpus like `TEXT` above; `small_cap_cfg` sets `truncation_cap_chars` to the minimum the validator allows (`MIN_MARKER_CAP`, `config.py:636-642`); `episode_env` gains `interactive=` and `task=`.

- [ ] **Step 2: Run** — FAIL (`ModuleNotFoundError: rlm.context.interactive`, `on_env` missing).
- [ ] **Step 3: Implement** the module, the manager branch, the episode changes. `from_text`:

```python
    @classmethod
    def from_text(cls, text: str, chunk_cfg: ChunkConfig, count_tokens) -> "InteractiveIndex":
        docs: dict[str, str] = {}; titles: dict[str, str] = {}
        parts = re.split(r"^=== DOCUMENT (\S+): (.*?) ===\n", text, flags=re.M)
        # parts = [preamble, id1, title1, body1, id2, title2, body2, ...]
        for j in range(1, len(parts), 3):
            docs[parts[j]] = parts[j + 2].strip("\n"); titles[parts[j]] = parts[j + 1]
        if not docs:
            raise ValueError("an interactive corpus needs at least one '=== DOCUMENT id: title ===' header")
        windows = {d: split(body, chunk_cfg, count_tokens) for d, body in docs.items()}
        return cls(docs=docs, titles=titles, windows=windows)
```

`search`: for each doc, `re.finditer(re.escape(term), body, re.I)`; map each match offset to the first window whose span contains it (precompute window start offsets by `body.find(w, cursor)`); cap at `max_hits`.

- [ ] **Step 4: Run** `uv run pytest checks/test_interactive_index.py checks/test_episode.py checks/test_sandbox_manager.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "episode: an interactive task keeps its corpus behind env, served and capped scaffold-side, every call a step"`

**Phase C gate:** full suite green; `git diff --stat sandbox_bootstrap/` shows the refreshed staged copy.

---

## Phase D — the v2 builder

### Task 13: Vendor the label source

**Files:**
- Create: `bench/sources/trec/fetch.py`, `bench/sources/trec/README.md`, `bench/sources/trec/trec_train.jsonl`, `bench/sources/trec/trec_train.sha256`
- Test: `checks/test_bench_v2_corpus.py` (create; first test)

**Interfaces:**
- `bench/sources/trec/trec_train.jsonl`: one `{"text": str, "coarse_label": int, "fine_label": int}` per line, 5,452 lines, the `train` split of `CogComp/trec` (HF parquet → jsonl), LF endings, `ensure_ascii=True`.
- `bench.corpus_v2.load_trec() -> list[Item]` with `Item(text: str, label: str)` where `label ∈ {"ABBR","ENTY","DESC","HUM","LOC","NUM"}`; refuses if the file's sha differs from `trec_train.sha256`.
- `label_source` string recorded in the manifest: `f"CogComp/trec:train@sha256:{sha[:16]}"`.

- [ ] **Step 1: Failing test**

```python
# checks/test_bench_v2_corpus.py
from pathlib import Path
from bench.corpus_v2 import load_trec, TREC_LABELS

REPO = Path(__file__).resolve().parents[1]


def test_the_vendored_label_source_is_pinned_and_shaped():
    items = load_trec()
    assert len(items) == 5452
    assert {i.label for i in items} == set(TREC_LABELS) == {"ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"}
    assert all(3 <= len(i.text.split()) <= 60 for i in items)
    sha = (REPO / "bench/sources/trec/trec_train.sha256").read_text().split()[0]
    import hashlib
    assert hashlib.sha256((REPO / "bench/sources/trec/trec_train.jsonl").read_bytes()).hexdigest() == sha
```

- [ ] **Step 2: Run** — FAIL (module/file missing).
- [ ] **Step 3: Implement.** `fetch.py` (one-shot, run by hand, documented in README):

```python
"""One-shot: fetch CogComp/trec train split from the Hub and vendor it as jsonl.
    uv run --python 3.12 --no-project --with pyarrow --with requests python bench/sources/trec/fetch.py
Writes trec_train.jsonl + trec_train.sha256 next to itself. Re-running must reproduce
the same bytes; if the Hub file changes, the sha moves and the manifest's label_source
records which bytes every v2 answer was computed from.
"""
```

It downloads `https://huggingface.co/datasets/CogComp/trec/resolve/main/default/train/0000.parquet` (verify the exact path with `hf_fs ls hf://datasets/CogComp/trec` before hardcoding; fall back to the `datasets` loading script's CSV URL if parquet is absent), writes jsonl sorted by original row order, `ensure_ascii=True`, LF. README states: source, licence status (`unknown` on the card; CogComp distributes for research use), the fallback (`PolyAI/banking77`, CC-BY-4.0), and the rule that this file is never edited by hand.

`bench/corpus_v2.py` (first pieces):

```python
TREC_LABELS = ("ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM")
_SRC = Path(__file__).resolve().parent / "sources" / "trec"

@dataclass(frozen=True)
class Item:
    text: str
    label: str

def load_trec() -> list[Item]:
    data = (_SRC / "trec_train.jsonl").read_bytes()
    want = (_SRC / "trec_train.sha256").read_text().split()[0]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise RuntimeError(f"vendored TREC moved: {got} != pinned {want}")
    return [Item(text=r["text"], label=TREC_LABELS[r["coarse_label"]])
            for r in (json.loads(l) for l in data.decode("utf-8").splitlines() if l)]

def label_source_id() -> str:
    return "CogComp/trec:train@sha256:" + (_SRC / "trec_train.sha256").read_text().split()[0][:16]
```

- [ ] **Step 4: Run** `uv run pytest checks/test_bench_v2_corpus.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: TREC vendored and pinned as the v2 label source, licence status recorded, banking77 named as fallback"`

---

### Task 14: The linear-semantic corpus — labelled items in the synthetic register

**Files:**
- Modify: `bench/corpus_v2.py`
- Test: `checks/test_bench_v2_corpus.py`

**Interfaces:**
```python
@dataclass
class LinearSemanticCorpus:
    text: str
    items: list[Item]            # the sampled items, in placement order
    labels: list[str]            # items[i].label
    record_ids: list[str]        # "ENT-5xxxxx" per item
    question_kind: str           # "count_label" | "most_common_label" | "count_two_labels"
    target: tuple[str, ...]      # the label(s) the question is about
    answer: str                  # computed from labels
    checker: str                 # "int_exact" | "name_exact"
    seed: int; measured_tokens: int; counter_name: str
    @property sha256

def build_linear_semantic(seed: int, target_tokens: int, count, counter_name: str,
                          *, question_kind: str, items: list[Item]) -> LinearSemanticCorpus
```
Record grammar (label never appears in the text; the item text is the question the register "files"):
```
[ENT-5{seed%10:01d}{idx:05d}] {organisation}
Filed: {coined date}
Query: {item.text}
Notes: {12–26 filler words}
```
Questions (the task `text`):
- `count_label`: `"Each record in this register files one Query. Count the records whose Query asks about a {LABEL_DESCRIPTION}. Reply with the integer only, nothing else."` with `LABEL_DESCRIPTION = {"HUM": "person or group of people", "LOC": "place or location", "NUM": "number, quantity, date or other numeric value", "ENTY": "thing, object or entity that is not a person or place", "DESC": "description, definition, reason or manner", "ABBR": "abbreviation or its expansion"}`.
- `most_common_label`: `"... Which kind of thing do the most Queries ask about: a person or group, a place, a numeric value, an entity, a description, or an abbreviation? Reply with exactly one of: person, place, number, entity, description, abbreviation."` → checker `name_exact`, answer the canonical word.
- `count_two_labels`: count of records whose Query is about either of two labels.
Answers are computed from `labels`. Sampling: `rng.sample(items, k)` until `target_tokens` (incremental count, then drop-until-fits like `corpus.build`); each task samples a *different* subset (different seed).

- [ ] **Step 1: Failing tests**

```python
def test_linear_semantic_answer_is_computed_from_labels_and_the_label_never_appears():
    items = load_trec()
    c = build_linear_semantic(seed=9101, target_tokens=20_000, count=approx_tokens,
                              counter_name="approx-offline", question_kind="count_label", items=items)
    assert c.answer == str(sum(1 for l in c.labels if l == c.target[0]))
    for lab in TREC_LABELS:
        assert lab not in c.text                     # the class name is nowhere in the register
    assert c.text.count("Query:") == len(c.items) == len(c.labels) == len(c.record_ids)
    assert c.measured_tokens <= 20_000


def test_two_seeds_sample_different_items_and_build_is_deterministic():
    items = load_trec()
    a = build_linear_semantic(9101, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    b = build_linear_semantic(9102, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    a2 = build_linear_semantic(9101, 8_000, approx_tokens, "approx-offline", question_kind="count_label", items=items)
    assert a.sha256 == a2.sha256 and a.sha256 != b.sha256
    assert set(i.text for i in a.items) != set(i.text for i in b.items)


def test_most_common_label_answers_a_canonical_word():
    items = load_trec()
    c = build_linear_semantic(9111, 8_000, approx_tokens, "approx-offline", question_kind="most_common_label", items=items)
    from collections import Counter
    top = Counter(c.labels).most_common(1)[0][0]
    assert c.answer == {"HUM": "person", "LOC": "place", "NUM": "number", "ENTY": "entity",
                        "DESC": "description", "ABBR": "abbreviation"}[top]
    assert c.checker == "name_exact"
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** in `bench/corpus_v2.py`, reusing `bench.vocab.organisation` and a `_coined_date(rng)` (`"{day} {coined month} {year}"` with coined month names from `SYL_A`). Guard: if the most-common label is tied, resample the target seed (`rng` continues) until unique — record the retry count.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: the linear-semantic corpus files human-labelled queries in the synthetic register and computes its answers from the labels"`

---

### Task 15: The two build-time adversaries

**Files:**
- Create: `bench/adversary.py`
- Test: `checks/test_adversary.py`

**Interfaces:**
```python
def parser_adversary(corpus: LinearSemanticCorpus) -> dict[str, float]
    # per-item label prediction by every deterministic strategy a parser could run over the
    # PARSED record (fields: org, filed, query, notes): keyword lexicons per label, wh-word
    # rules ("who"->HUM, "where"->LOC, "when"/"how many"->NUM, "what is"->DESC/ENTY, ...),
    # capitalised-token heuristics, and the bag-of-words nearest-label over the *other* items'
    # texts (a leak-free 1-NN needs labels it does not have, so it is excluded).
    # Returns {strategy: accuracy, "__chance__": max class share}. The task is REJECTED if any
    # strategy's accuracy > chance + 0.02  (v1's tolerance, corpus.py:305).
def self_read_adversary(corpus, chunk_cfg: ChunkConfig, count_tokens, *, k: int = 40) -> int
    # the minimal necessary window set: windows containing at least one record whose label is
    # needed to fix the answer. For count_label / count_two_labels EVERY record is necessary
    # (removing any one changes the count or its certainty), so the set is every window that
    # holds a record. Returns its size; the task is REJECTED if size <= k.
```
The wh-word rule set is the load-bearing strategy — TREC is famous for it. The adversary must run BEFORE freeze and its scores land in the manifest (`regex_at_chance` field, reused; keys are strategy names).

- [ ] **Step 1: Failing tests**

```python
# checks/test_adversary.py
from bench.adversary import parser_adversary, self_read_adversary
from bench.corpus_v2 import build_linear_semantic, load_trec
from bench.tokens import approx_tokens
from rlm.context.chunker import ChunkConfig

CFG = ChunkConfig(size_tokens=640, overhead_tokens=1920, snap_to_boundary=True,
                  snap_tolerance=0.10, stride_tokens=480)


def test_parser_adversary_scores_every_strategy_and_reports_chance():
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec())
    scores = parser_adversary(c)
    assert "__chance__" in scores and "wh_word_rules" in scores and "label_lexicon" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_the_wh_word_rule_is_a_real_adversary_on_trec():
    """If this ever scores at chance the adversary is broken, not TREC: 'who' is HUM."""
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec())
    assert parser_adversary(c)["wh_word_rules"] > 0.5


def test_self_read_adversary_counts_the_windows_the_answer_needs():
    c = build_linear_semantic(9101, 60_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec())
    n = self_read_adversary(c, CFG, approx_tokens, k=40)
    assert n > 40                                   # ~139 windows at 60K tokens
    small = build_linear_semantic(9101, 6_000, approx_tokens, "approx-offline",
                                  question_kind="count_label", items=load_trec())
    assert self_read_adversary(small, CFG, approx_tokens, k=40) <= 40   # a 6K corpus IS self-readable
```

**Note on the second test:** it is *expected* that the wh-word rule beats chance on raw TREC. That is the whole point of running the adversary: a count over TREC coarse labels **is parser-solvable** unless the register defeats the rule. Task 16 makes the register do that (see below). Keep this test as the sanity check that the adversary has teeth.

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `bench/adversary.py` with the strategies named above; `self_read_adversary` runs `split(corpus.text, chunk_cfg, count_tokens)`, maps each record id to the windows containing it (`window.find(record_id)`), and returns the size of the union over necessary records.
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: two adversaries at build time -- a parser that reads fields, and a root that reads forty windows"`

---

### Task 16: Make the register defeat the parser — the question is asked about the ANSWER, not the query

**Files:**
- Modify: `bench/corpus_v2.py` (record grammar + question kinds)
- Test: `checks/test_bench_v2_corpus.py`, `checks/test_adversary.py`

**Why:** Task 15's sanity test shows raw TREC questions are wh-word-solvable. The spec's requirement (§1) is a per-item label *no deterministic program can produce from the item's text*. The fix that keeps human labels while removing the surface cue: each record files the TREC question **together with a coined one-line answer** whose *shape* is what the label describes, and the record's Query field is **paraphrased to drop the wh-word** by the builder using a fixed, seeded template set (`"Identify {rest}"`, `"The register asks after {rest}"`, `"Record the {rest}"`), where `{rest}` is the question with its leading wh-phrase removed. The label is then recoverable only by understanding what kind of thing is being asked — which is exactly what the leaf must do. Both adversaries re-run on every built task; **a task whose parser adversary beats chance + 0.02 is not emitted** (`build_v2` refuses, names the strategy, and the author changes the seed or the template).

**Interfaces:** `build_linear_semantic(..., paraphrase: bool = True)`; `Item.text` unchanged; the record's `Query:` line carries the paraphrase; the manifest records `regex_at_chance` (parser scores) and `regex_solvable: False` for every linear-semantic task.

- [ ] **Step 1: Failing test**

```python
def test_the_paraphrased_register_takes_the_wh_word_rule_to_chance():
    c = build_linear_semantic(9101, 20_000, approx_tokens, "approx-offline",
                              question_kind="count_label", items=load_trec(), paraphrase=True)
    scores = parser_adversary(c)
    chance = scores.pop("__chance__")
    beaten = {k: v for k, v in scores.items() if v > chance + 0.02}
    assert not beaten, beaten
    assert not any(q.split()[0].lower() in {"who", "where", "when", "what", "which", "how", "why"}
                   for q in re.findall(r"^Query: (.+)$", c.text, re.M))
```

- [ ] **Step 2: Run** — FAIL (wh_word_rules ≈ 0.7).
- [ ] **Step 3: Implement** the paraphrase step (`_strip_wh(text) -> str` removing a leading `who|whom|whose|where|when|what|which|how (many|much|long|far|old)?|why|name|define|describe` phrase and the following auxiliary if present; then template it). Keep `paraphrase=False` available for the adversary sanity test in Task 15 (switch that test to `paraphrase=False`).
- [ ] **Step 4: Run** `uv run pytest checks/test_adversary.py checks/test_bench_v2_corpus.py -q -p no:cacheprovider` — PASS. If `label_lexicon` still beats chance on some seed, extend the lexicon strategy's report and the builder's rejection path — do not weaken the tolerance.
- [ ] **Step 5: Commit** — `git commit -m "bench: the register paraphrases each filed query so the label is a judgment, not a wh-word"`

---

### Task 17: The interactive corpus and its reference action count

**Files:**
- Modify: `bench/corpus_v2.py`
- Test: `checks/test_bench_v2_corpus.py`

**Interfaces:**
```python
@dataclass
class InteractiveCorpus:
    text: str                      # "=== DOCUMENT d-NN: <coined title> ===" blocks (Task 12's DOC_DELIM grammar)
    doc_ids: list[str]
    items_by_doc: dict[str, list[Item]]
    question_kind: str             # "count_label_across_docs" | "which_doc_has_most" | "pairs_docs_sharing_label_majority"
    target: tuple[str, ...]; answer: str; checker: str
    reference_actions: int         # optimal env ops: open(d) for every doc + window(d, i) for every window holding a necessary record
    seed: int; measured_tokens: int; counter_name: str
def build_interactive(seed, target_tokens, count, counter_name, *, question_kind, items, n_docs: int = 12) -> InteractiveCorpus
```
Every question needs labels from records spread across **all** documents (the adversary asserts the necessary-window set spans ≥ 75% of documents), so navigation is unavoidable and `reference_actions ≥ n_docs + necessary windows`. `pairs_docs_sharing_label_majority` is the quadratic-flagged shape (spec §3): "How many pairs of registers have the same most-common kind of Query? Reply with the integer only."

- [ ] **Step 1: Failing tests**

```python
def test_interactive_corpus_is_many_documents_and_the_answer_needs_most_of_them():
    c = build_interactive(9201, 40_000, approx_tokens, "approx-offline",
                          question_kind="count_label_across_docs", items=load_trec(), n_docs=8)
    assert len(c.doc_ids) == 8 and c.text.count("=== DOCUMENT ") == 8
    ix = InteractiveIndex.from_text(c.text, CFG, approx_tokens)
    assert set(ix.docs) == set(c.doc_ids)
    assert c.answer == str(sum(1 for d in c.doc_ids for i in c.items_by_doc[d] if i.label == c.target[0]))
    assert c.reference_actions >= 8 + 8                # at least one open and one window per doc


def test_pairs_question_is_quadratic_in_documents():
    c = build_interactive(9202, 40_000, approx_tokens, "approx-offline",
                          question_kind="pairs_docs_sharing_label_majority", items=load_trec(), n_docs=6)
    from collections import Counter
    maj = {d: Counter(i.label for i in c.items_by_doc[d]).most_common(1)[0][0] for d in c.doc_ids}
    pairs = sum(1 for a in range(6) for b in range(a + 1, 6) if maj[c.doc_ids[a]] == maj[c.doc_ids[b]])
    assert c.answer == str(pairs)
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**; `reference_actions` = `n_docs` opens + `len(necessary_windows)` (computed with the real chunker per document, same as Task 15's helper) + 1 search per document title (the optimal navigator still has to find the documents; count `n_docs` searches).
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: the interactive corpus spreads the answer across every document and records the optimal path's length"`

---

### Task 18: The code-solvable controls, and `build_v2` emits the whole artifact

**Files:**
- Modify: `bench/corpus_v2.py` (controls); Create: `bench/build_v2.py`
- Modify: `bench/manifest.py` (`validate()` per-stream + v2 clauses)
- Test: `checks/test_manifest_v2.py`, `checks/test_bench_v2_corpus.py`

**Interfaces:**
- Controls (4 shapes): `ctl-01/02` = regex-count over the v2 register (`"Count the records whose Filed line names the month {M}"`, answer computed; `regex_solvable: True`), `ctl-03/04` = needle over the v2 register (`bench.corpus.build_needle`'s pattern re-implemented on `LinearSemanticCorpus` text: one record carries `Custody key of record: <uuid>`; `uuid_exact`). Controls **must pass** `self_read_adversary <= k` or be trivially code-solvable — the builder asserts `regex_solvable` by running the intended regex and matching the answer.
- `bench/build_v2.py`:
```
uv run --python 3.12 --no-project python -m bench.build_v2 --leaf-port 8081 [--stream train|held_out|both] [--practice --seed N --out DIR]
```
Shapes (16, fixed order): `ls-01..06` (linear-semantic: kinds `count_label ×3`, `count_two_labels ×2`, `most_common_label ×1`; `ls-05`,`ls-06` carry the quadratic flag via `count_two_labels`), `int-01..06` (interactive: `count_label_across_docs ×3`, `which_doc_has_most ×1`, `pairs_docs_sharing_label_majority ×2`; `int-05`,`int-06` carry `adversarial: True` — the v1 `INJECTION` record inserted into one document), `ctl-01..04`. Task ids are `{shape}-{stream}` (`ls-01-train`, `ls-01-held`); seeds derive as `base + 1000*stream_index`. Files: `bench/tasks/v2/{id}.json`, `bench/corpora/v2/{id}.txt`, `bench/manifest.v2.json`. Manifest `rules`:
```json
{"rlm_arm": "rlm", "baselines": ["rlm-nosubcalls", "b2"], "margin": 2,
 "escalation_band": [1, 2], "tripwire_floor": 3,
 "abstentions": {"b2": ["interactive"]}, "scored_stream": "train", "n_tasks": 16}
```
Each `TaskEntry`: `stream`, `shape_id`, `interactive` (int-*), `min_windows` (self-read adversary result), `reference_actions` (int-*), `label_source`, `regex_at_chance` (parser scores, ls-* and int-*), `regex_solvable` (`False` for ls/int, `True` for ctl), `windows`/`subcalls` (ls-*: analytic bound like v1, `ceil(tokens/432)` and `2×`). `benchmark_version` = `"v2-draft"` unless `--leaf-port`.
- `--practice`: writes the same 16 shapes with `stream: "practice"` under `--out` (default `runs/practice/<seed>/`), **no manifest write to `bench/`**, prints the task list; `runs/` is gitignored.
- `manifest.validate()`: when `rules` present — `len(scored_tasks()) == rules["n_tasks"]`; per-stream category counts equal `{"linear_semantic": 6, "interactive": 6, "code_solvable": 4}`; every ls/int task has `min_windows > 40` and parser scores all `<= chance + 0.02`; every int task has `reference_actions`; every ctl task `regex_solvable is True`; `held_out` and `train` have identical shape sets; `MIN_ADVERSARIAL` per scored stream. The v1 branch (no `rules`) is unchanged.

- [ ] **Step 1: Failing tests**

```python
# checks/test_manifest_v2.py (append)
def test_build_v2_emits_sixteen_shapes_in_two_frozen_streams(tmp_path, monkeypatch):
    from bench import build_v2
    monkeypatch.setattr(build_v2, "TASKS", tmp_path / "tasks")
    monkeypatch.setattr(build_v2, "CORPORA", tmp_path / "corpora")
    monkeypatch.setattr(build_v2, "MANIFEST", tmp_path / "manifest.v2.json")
    rc = build_v2.main(["--stream", "both", "--small"])      # --small: 6K/20K tokens for tests
    assert rc == 0
    m = BenchmarkManifest.load(tmp_path / "manifest.v2.json")
    assert len(m.tasks) == 32 and len(m.scored_tasks()) == 16
    assert {t.shape_id for t in m.tasks if t.stream == "train"} == {t.shape_id for t in m.tasks if t.stream == "held_out"}
    assert m.rules["margin"] == 2 and m.rules["abstentions"] == {"b2": ["interactive"]}
    m.validate(require_closed_book=False)


def test_build_v2_refuses_a_task_a_parser_can_solve(tmp_path, monkeypatch):
    from bench import build_v2
    monkeypatch.setattr(build_v2, "PARAPHRASE", False)     # disable the defence
    ...same monkeypatches...
    with pytest.raises(SystemExit) as e:
        build_v2.main(["--stream", "train", "--small"])
    assert "parser adversary" in str(e.value)


def test_practice_stream_writes_outside_bench_and_no_manifest(tmp_path):
    from bench import build_v2
    rc = build_v2.main(["--practice", "--seed", "7", "--out", str(tmp_path / "p"), "--small"])
    assert rc == 0 and (tmp_path / "p" / "tasks" / "ls-01-practice.json").exists()
    assert not (tmp_path / "p" / "manifest.v2.json").exists()
```

(`--small` is a test-only size switch: 6,000-token static corpora, 20,000-token interactive. With `--small`, `min_windows` will be ≤ 40 — validate must then be called with `strict_adversaries=False`; add that flag to `validate` and to `build_v2 --small`. The real build never passes it.)

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** `build_v2.py` following `build.py`'s structure (`_write`, `_task_file`, `_entry`, one `build_<category>` per category, `main(argv)` returning int). Fix the dead branch v1 carried (`corpus.py:299` `"granted (inverted)"`) — v2's adversary has no inverted candidates, so nothing to port. The parser-adversary refusal is `SystemExit(f"parser adversary beat chance on {task_id}: {beaten}")`.
- [ ] **Step 4: Run** `uv run pytest checks/test_manifest_v2.py checks/test_bench_v2_corpus.py -q -p no:cacheprovider` — PASS.
- [ ] **Step 5: Commit** — `git commit -m "bench: build_v2 emits sixteen shapes in train and held-out, refuses any task an adversary solves, and writes practice outside the tree"`

---

### Task 19: Closed-book probe for v2

**Files:**
- Modify: `bench/closed_book.py:32-41, 64-99`
- Test: manual (needs servers); unit-test the arg plumbing

- [ ] **Step 1:** Add `--manifest` (default `bench/manifest.json`) and make the probe write back to that path. Add a test that `main(["--manifest", str(p), "--dry-run"])` reads the given manifest and lists its tasks without calling any server (add `--dry-run`).
- [ ] **Step 2–4:** RED → GREEN.
- [ ] **Step 5: Commit** — `git commit -m "bench: the closed-book probe takes the manifest it is probing"`

---

### Task 20: The v2 strategy blocks, per arm

**Files:**
- Create: `src/rlm/_data/prompts/strat-linear-semantic.v1.md`, `strat-interactive.v1.md`, `strat-code-solvable.v1.md`, `strat-linear-semantic-nosubcalls.v1.md`, `strat-interactive-nosubcalls.v1.md`, `strat-code-solvable-nosubcalls.v1.md`
- Test: `checks/test_prompts.py`

**Content rules (write, then test):**
- `linear-semantic.v1`: the aggregation block's shape (map once over all `chunks` with one `asyncio.gather`, ask for one label word per record id, reduce by identity over `context`), plus: *"The kind of thing a Query asks about is a judgment about meaning; no pattern over the words settles it. Code locates the records; the sub-model labels them."* Carries the evidence-span check (the `verifies` snippet) and the word `evidence`.
- `interactive.v1`: teaches `env`: `hits = await env.search(term)`, `meta = await env.open(doc_id)`, `text = await env.window(doc_id, i)`; *"`context` and `chunks` are empty in this category; the corpus lives behind `env`. Every call is an action and is counted. Plan the navigation: open each document once, read only the windows you need, and label with the sub-model."*
- `code-solvable.v1`: *"This task is answerable in code. Count, match or locate with `re` over `context`; a sub-call here is a cost, not a method."*
- `-nosubcalls` twins: same files minus every sentence naming the sub-model/`llm_query`/sub-calls/`gather`; the interactive twin keeps the `env` API and says *"You have `env` and Python; read what you need through `env.window` and reason over it in code."*
- All six carry the changelog header; the three with the evidence-span check contain `evidence`.

- [ ] **Step 1: Failing tests**

```python
V2_BLOCKS = ["strat-linear-semantic.v1.md", "strat-interactive.v1.md", "strat-code-solvable.v1.md"]

@pytest.mark.parametrize("name", V2_BLOCKS)
def test_v2_blocks_have_headers_and_the_rlm_variants_teach_llm_query_or_code(name):
    assert (PROMPTS / name).read_text(encoding="utf-8").startswith("<!-- changelog")
    body = _body(name)
    assert body.lstrip().startswith("# Strategy: ")
    if "code-solvable" not in name:
        assert "llm_query" in body

@pytest.mark.parametrize("name", [n.replace(".v1.md", "-nosubcalls.v1.md") for n in V2_BLOCKS])
def test_nosubcalls_blocks_never_name_the_sub_model(name):
    body = _body(name).lower()
    for banned in ("llm_query", "sub-model", "sub-call", "asyncio.gather", "delegat"):
        assert banned not in body, (name, banned)

def test_the_interactive_blocks_teach_env():
    for name in ("strat-interactive.v1.md", "strat-interactive-nosubcalls.v1.md"):
        body = _body(name)
        assert "env.search(" in body and "env.open(" in body and "env.window(" in body
```

Add the six names to `FILES` (`checks/test_prompts.py:13-22`) and the two evidence-carrying ones to the `:200` list.

- [ ] **Step 2–4:** RED → author → GREEN.
- [ ] **Step 5: Commit** — `git commit -m "prompts: six v2 strategy blocks, one per category per arm, and the nosubcalls set never names a sub-model"`

**Phase D gate:** full suite green; `bench/manifest.json` still unchanged; **no `bench/manifest.v2.json` committed yet** (the real build needs the leaf server — Task 22).

---

## Phase E — configuration, freeze, smoke, amendment

### Task 21: `config.v2.yaml`

**Files:**
- Create: `config.v2.yaml`
- Test: `checks/test_config.py` (append)

**Content:** copy `config.s5-a3b-root.yaml`; change: header comment; `benchmark.version: v2`; `benchmark.manifest_sha256: null` (filled at freeze, Task 22); `scaffold.prompts.root` → `prompts/root.v4.md` + its sha; `scaffold.prompts.root_nosubcalls` → `prompts/root-nosubcalls.v1.md` + sha; add `linear_semantic`, `interactive`, `code_solvable` entries to `strategy_templates` (keep the five v1 keys — `StrategyTemplates` is a dict now, and `default` is required); add `strategy_templates_nosubcalls` with the three `-nosubcalls` blocks plus `default: prompts/strat-default.v1.md` (unused; required only so an unknown category still refuses rather than KeyErrors — actually not required: `render_root` checks membership; include only the three). `trace.db_path: traces/v2/rlm.duckdb`, `trace.blob_root: traces/v2/blobs` so v2 episodes never mix with the v1 store.

- [ ] **Step 1: Failing test**

```python
def test_config_v2_pins_the_v2_prompts_and_names_the_v2_benchmark():
    cfg = load_config(REPO / "config.v2.yaml")
    assert cfg.benchmark.version == "v2"
    assert cfg.scaffold.prompts.root.path.name == "root.v4.md"
    assert cfg.scaffold.prompts.root_nosubcalls.path.name == "root-nosubcalls.v1.md"
    assert set(cfg.scaffold.prompts.strategy_templates) >= {"linear_semantic", "interactive", "code_solvable", "default"}
    assert set(cfg.scaffold.prompts.strategy_templates_nosubcalls) == {"linear_semantic", "interactive", "code_solvable"}
    assert cfg.servers.root.model == cfg.servers.leaf.model == cfg.servers.bench_leaf.model   # D-B7
    for path, pin in cfg.pinned_prompt_hashes().items():
        assert hashlib.sha256(resolve_prompt_path(Path(path)).read_bytes()).hexdigest() == pin
```

- [ ] **Step 2–4:** RED → write → GREEN. Also run `uv run rlm validate --config config.v2.yaml --no-server-probe` → `config: OK`.
- [ ] **Step 5: Commit** — `git commit -m "config: config.v2.yaml -- one model in every arm, the v2 prompts pinned, the v2 store separate"`

---

### Task 22: Build and freeze v2 (needs the leaf server)

**Files:** `bench/manifest.v2.json`, `bench/tasks/v2/`, `bench/corpora/v2/`, `config.v2.yaml` (pin), `src/rlm/_data/config.default.yaml` (no change — default stays v1)

- [ ] **Step 1:** Start the leaf: `uv run rlm validate --config config.v2.yaml` will launch nothing; start the leaf server by hand with the config's `servers.leaf` line (or run `rlm run --launch-leaf` on any task once) — the build needs `/tokenize` on 8081.
- [ ] **Step 2:** `uv run --python 3.12 --no-project python -m bench.build_v2 --leaf-port 8081 --stream both`. Expected: `32 tasks, manifest sha256 …`, `VALIDATES against §8 and §14 (closed-book probe still owed before freeze)`. If it exits with `parser adversary beat chance on …`, change that shape's seed in `SHAPES` and rerun; record the attempt in the commit message.
- [ ] **Step 3:** Start the root too, then `uv run --python 3.12 --no-project python -m bench.closed_book --manifest bench/manifest.v2.json` — 32 tasks × 2 roles × 3 seeds. Expected: every task `passed_without_corpus: 0`. Any contamination → rewrite that shape's seed and rebuild (Step 2), never edit an answer.
- [ ] **Step 4:** `uv run python -c "from bench.manifest import BenchmarkManifest; print(BenchmarkManifest.load('bench/manifest.v2.json').sha256)"` → paste into `config.v2.yaml` `benchmark.manifest_sha256`. Run `uv run pytest checks/test_manifest_v2.py checks/test_config.py -q -p no:cacheprovider`. Add to `checks/test_manifest_v2.py` a `test_the_v2_freeze_passes_every_precondition` mirroring `test_bench_manifest.py:96-110` for the v2 file (skipif it does not exist).
- [ ] **Step 5: Commit** — `git commit -m "bench: benchmark v2 frozen -- 16 shapes x 2 streams, every task past both adversaries and the closed-book probe, sha pinned in config.v2.yaml"`. Export nothing yet (no episodes).

---

### Task 23: Smoke on real servers, and the archive

- [ ] **Step 1:** `mkdir runs/v2-smoke && uv run rlm bench --config config.v2.yaml --smoke --ledger runs/v2-smoke/ledger.jsonl --report runs/v2-smoke/RESULTS.md > runs/v2-smoke/smoke.log 2>&1`. Alongside: `tail -F traces/logs/leaf-server.log >> runs/v2-smoke/leaf-capture.log &` (the leaf log truncates on relaunch).
- [ ] **Step 2: Check, in the ledger and the store:** 3 smoke tasks (one per category) × 3 arms = 9 cells minus 1 abstention (`b2`/interactive → an `abstained` row, no episode) = 8 episodes. `rlm-nosubcalls` episodes: `llm_call` steps are all `rejected` or absent, `completion_count` on the leaf unchanged during them. Interactive episodes: `env_call` steps present, `snapshot.env_actions` recorded, `context`/`chunks` empty (`print(len(context), len(chunks))` in the first cell shows `0 0`). The `rlm` interactive episode either delegates or hits `context_exhausted` — both are data; note which.
- [ ] **Step 3:** Read the printed calibration table and the projected hours. **If the projection exceeds 36 h, stop and ask the owner** (spec §14.7).
- [ ] **Step 4:** `uv run rlm export <smoke run_id> --config config.v2.yaml --dest D:\AI\rlm-halo-archive\<date>-v2-smoke-<run_id8>\`; write `docs/research/<date>-benchmark-v2-smoke.md` with the table, the projection, the archive path and manifest sha, in the style of `2026-09-01-s5-a3b-root-smoke.md`.
- [ ] **Step 5: Commit** — `git commit -m "research: benchmark v2 smoke -- <one-line result>"`

---

### Task 24: ARCHITECTURE §8 amendment and the changelog entry

**Files:** `ARCHITECTURE.md:3-4, 343-397, 499`, `CHANGELOG.md:14-16`

- [ ] **Step 1:** Bump `ARCHITECTURE.md:3` to `rlm-runtime-spec-v0.4.1`; append to the `:4` status line `**v0.4.1 (2026-09-XX): benchmark v2 frozen (§8 amendment) — 16 tasks, arms rlm/rlm-nosubcalls/b2, margin +2, two build-time adversaries, one sandbox verb.**`
- [ ] **Step 2:** Insert, after `:373` (the v1 split paragraph), a new bold paragraph **"BENCHMARK v2 (2026-09-XX), beside v1, not replacing it."** stating: the rules block is data in `bench/manifest.v2.json` (arms, +2 margin at N=16, band {+1,+2}, B2 abstains from `interactive`, scored stream `train`); the two added preconditions (parser adversary, self-read adversary with K=40 derived from `truncation_cap_chars` and the 32K window); the three streams and the archive-by-export rule; the fourth resident arm and the within-block order `rlm, rlm-restricted, rlm-nosubcalls, b2, b1, b3`; the cost projection is the smoke's, not `:351`'s; and that `rlm-nosubcalls` is a **candidate bias channel named before the run** (`:381`): it shares the root server with `rlm` and nothing with the leaf. Point to the spec file and to this plan.
- [ ] **Step 3:** Add to R16's row (`:499`): `**Benchmark v2 built 2026-09-XX (commit …); the mitigation's first item is done.**`
- [ ] **Step 4:** `CHANGELOG.md`: new first entry `- **v0.4.1 — 2026-09-XX. BENCHMARK v2 FROZEN BESIDE v1. §8 amended: the scoring rules become manifest data; two build-time adversaries; the `env` verb; the `rlm-nosubcalls` arm. No invariant or gate change.** Record …` in the house format (`CHANGELOG.md:16-40`).
- [ ] **Step 5: Commit** — `git commit -m "spec: §8 carries benchmark v2 as an amendment, v0.4.1"`. Run the full suite one last time.

---

## Self-review against the spec

**Coverage.** §0/§3 sixteen tasks, 6·6·4 → Task 18. §1 sizing 60K/200K → Tasks 14/17/18. §2 human labels in a synthetic wrapper, computed answers, parser adversary, fallback → Tasks 13/14/15/16. §4 `env` with search/open/window, no bulk text, capped returns, reference action count → Tasks 11/12/17. §5 three streams, freeze, practice regenerable → Tasks 1/18. §6 three arms, `rlm-nosubcalls`, B2 abstains from interactive, margin +2, N=16, inherited inference layer, no new checkers → Tasks 2–6/9/18. §7 cost → superseded by §14.7 → Task 23. §8 preconditions + parser adversary → Tasks 15/19/22. §9 implementation surface → Tasks 7–12/18. §10 V2-R1 detector = `rlm-nosubcalls` reading (report text in Task 3); V2-R3 hijack test → Task 12; V2-R4 reference count → Task 17; V2-R6 smoke → Task 23. §12 §8 amendment → Task 24. §14.1 config → Task 21; §14.2 self-read adversary → Task 15; §14.3 prompts → Tasks 8/20; §14.4 → nothing to build (recorded in Task 24); §14.5 archive → Task 23; §14.6 TREC → Task 13; §14.7 smoke-before-score → Task 23.

**Gaps found and closed in review:** (1) v1's manifest sha would have moved with any new field — Task 1's None-stripping and its byte-identity test. (2) `render_root` refuses unknown categories, so v2 categories needed blocks in *both* sets — Task 20 authors six. (3) B2 abstention had no representation anywhere — Task 4 adds the ledger value and the grid exemption. (4) The spec's "synthesis has none" is false in spirit; v2 does not carry synthesis, so the six new blocks are authored per arm from scratch. (5) `test_prompts.py:179` reads `config.yaml` — v2 pins live only in `config.v2.yaml` (Task 21), so it stays green. (6) Raw TREC is wh-word-solvable — Task 16 exists because Task 15's sanity test says so.

**Type consistency.** `BenchmarkRules.for_manifest` (Tasks 2/3/4/6); `PairResult.n_tasks` (Task 4); `run_episode(no_subcalls=, interactive=)` (Tasks 9/12); `InteractiveIndex.from_text(text, chunk_cfg, count_tokens)` (Tasks 12/17); `build_linear_semantic(seed, target_tokens, count, counter_name, *, question_kind, items, paraphrase=True)` (Tasks 14/15/16/18); `parser_adversary(corpus) -> dict[str, float]`, `self_read_adversary(corpus, chunk_cfg, count_tokens, *, k=40) -> int` (Tasks 15/18); `bench_manifest_path(version)` (Tasks 6/22); ledger outcome `"abstained"` (Tasks 4/23).

**Placeholders.** None: every step names its file, its code and its expected failure. Task 19 is deliberately short because it is a two-flag change; Tasks 22–23 are operational and state their commands and their pass criteria.
