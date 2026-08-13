# Capa-1 Scaffold (C1–C6 → S1 Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic scaffold (C1–C6, prompt registry, config schema, CLI) to the depth the S1 gate requires, then run the S1 gate.

**Architecture:** A composition-root design. Six components are independent modules with injected dependencies; only C4 (`rlm/dispatcher.py`) may talk to a model server, and `rlm/episode.py` is the single place allowed to import everything and wire them together. The root LLM drives a persistent sandboxed Python interpreter (one per episode) whose only channel to the outside world is a scaffold-owned duplex pipe; every control decision (truncation, budgets, routing, termination) executes scaffold-side where model output cannot reach it.

**Tech Stack:** Python 3.12 (uv-managed), pydantic v2 (config schema), duckdb (trace store), httpx (async streaming HTTP to llama-server), pytest + pytest-asyncio + hypothesis (tests). Windows 11 native; llama.cpp b10375 servers already validated by S0.

**Spec:** `ARCHITECTURE.md` (rlm-runtime-spec-v0.2.2). Read §5 (component contracts), §6 (trajectory schema), §9 S1 (the gate), and Appendix A (config sketch) before starting. The spec is binding; where this plan and the spec disagree, the spec wins and the plan is a bug.

## Global Constraints

- **Python 3.12+** (A1). Run everything through `uv run --python 3.12`.
- **I1 — The LLM proposes; the scaffold disposes.** Truncation caps, budgets, routing, and termination live in `config.yaml` and scaffold code. No model output, prompt content, or REPL side effect may alter them at runtime.
- **I2 — Context by reference.** The full context never enters any model's message array. Models see only scaffold-truncated views.
- **I4 — Every episode is a trajectory.** All runs are logged as `(state, action, observation)` steps with a terminal outcome. A run that is not logged did not happen.
- **Dependency rule (lint-enforced):** `C1–C3, C5, C6 must not import C4 or any LLM client.` Concretely: `rlm/sandbox/*`, `rlm/chunker.py`, `rlm/context.py`, `rlm/truncate.py`, `rlm/budget.py`, `rlm/trace.py` must not import `rlm.dispatcher`, `httpx`, or any HTTP client. Only `rlm/dispatcher.py` and `rlm/episode.py`/`rlm/cli.py` may.
- **No inline prompts.** Prompt text lives only in `prompts/*.md`, referenced from config by path, pinned by sha256. An inline prompt string in config or scaffold code is a spec violation.
- **The name `llm_query` is load-bearing** — it matches the RLM paper harness API. Injected sandbox API is exactly: `context` (str), `chunks` (list[str]), `await llm_query(prompt: str, role: str = "leaf") -> str`, `final_answer(value)`.
- **Config is `extra="forbid"`** with cross-field validators; `config_snapshot` is the canonical JSON dump of the validated model (stable field order ⇒ stable hashing).
- **Never run a foreground interactive process.** `llama-cli` in build b10375 is an interactive chat REPL that wedges the terminal. Servers launch detached with redirected logs (see `s0/RESULTS.md`).
- **Measured constants from S0** (do not re-derive): leaf prefill 949.9 t/s @32K; leaf decode 55.1 fresh / 46.3 @32K; root decode 12.35 fresh / 12.0 @28K; aggregate prefill is FLAT ~950 t/s at k=1..8; root retains 69–81% decode under full leaf load.
- **Server launch flags** (both already S0-validated; `--no-mmap` is deprecated in b10375 — use `-lm none`):
  - leaf: `<rocm>\llama-server.exe -m <leaf.gguf> --host 127.0.0.1 --port 8081 -c 327680 -np 8 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 -lm none --no-kv-unified --cont-batching` with env `ROCBLAS_USE_HIPBLASLT=1`
  - root: `<vulkan>\llama-server.exe -m <root.gguf> --host 127.0.0.1 --port 8080 -c 32768 -np 1 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 -lm none --no-context-shift`

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv project, pinned deps, pytest/hypothesis config |
| `config.yaml` | the operative config (Appendix A shape) |
| `prompts/*.md` | root/leaf/strategy prompt files, sha256-pinned from config |
| `rlm/errors.py` | shared exception + enum types; imports nothing from `rlm` |
| `rlm/truncate.py` | **C3** OutputTruncator — pure functions, no I/O |
| `rlm/chunker.py` | **C2** deterministic chunker — pure given an injected token-count callable |
| `rlm/context.py` | **C2** ContextLoader — materializes `context`/`chunks` payloads |
| `rlm/config.py` | config schema, prompt-registry loading + sha256 pinning, `config_snapshot` |
| `rlm/lifecycle.py` | the narrow JSONL lifecycle log (never episode data) |
| `rlm/trace.py` | **C6** TraceLogger — DuckDB single-writer, blobs-before-rows |
| `rlm/schema.sql` | §6 DDL |
| `rlm/sandbox/winjob.py` | ctypes Job Object helpers (limits + kill-on-close) |
| `rlm/sandbox/child.py` | the sandbox child program: persistent REPL, injected API, bridge client |
| `rlm/sandbox/manager.py` | **C1** parent side: spawn, isolate, exec cell, unconditional kill |
| `rlm/bridge.py` | parent side of the C1/C4 channel: framing, request correlation |
| `rlm/dispatcher.py` | **C4** LLMDispatcher (real + mock), the only module that talks to servers |
| `rlm/rootclient.py` | root conversation: render, `root_view_hash`, reasoning/code parsing |
| `rlm/budget.py` | **C5** BudgetEnforcer — admission, breach, deterministic termination |
| `rlm/episode.py` | composition root: wires C1–C6, runs one episode, owns outcomes |
| `rlm/cli.py` | `rlm validate / run / replay` (bench/export are later slices) |
| `tests/test_*.py` | per-component suites; property suites for C3 and C5 |
| `tests/test_import_rules.py` | the dependency-rule lint |
| `s1/` | S1 fixtures, gate runner, results |

---

## Task 1: Project skeleton + dependency-rule lint

The lint ships first so the architecture's load-bearing rule is enforced from the first component onward.

**Files:**
- Create: `pyproject.toml`, `rlm/__init__.py`, `rlm/errors.py`
- Test: `tests/test_import_rules.py`

**Interfaces:**
- Consumes: nothing.
- Produces: package `rlm`; `rlm.errors.Outcome`, `rlm.errors.StepStatus`, `rlm.errors.BudgetBreach`, `rlm.errors.ConfigError`, `rlm.errors.SandboxError`, `rlm.errors.DispatchError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_rules.py
"""The dependency rule (spec §5): C1-C3, C5, C6 must not import C4 or any LLM client."""
import ast
from pathlib import Path

import pytest

RLM = Path(__file__).resolve().parents[1] / "rlm"

# modules that must never reach a model server, directly or transitively
ISOLATED = [
    "truncate.py",
    "chunker.py",
    "context.py",
    "budget.py",
    "trace.py",
    "sandbox/manager.py",
    "sandbox/child.py",
    "sandbox/winjob.py",
]
FORBIDDEN_ROOTS = {"httpx", "requests", "urllib", "aiohttp", "socket", "http"}
FORBIDDEN_RLM = {"rlm.dispatcher", "rlm.rootclient"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("rel", ISOLATED)
def test_isolated_modules_do_not_import_llm_clients(rel):
    path = RLM / rel
    if not path.exists():
        pytest.skip(f"{rel} not implemented yet")
    for name in _imports(path):
        assert name.split(".")[0] not in FORBIDDEN_ROOTS, f"{rel} imports {name}"
        assert name not in FORBIDDEN_RLM, f"{rel} imports {name}"


def test_lint_covers_every_isolated_module_that_exists():
    """Guard against a component being added without lint coverage."""
    listed = {p.replace("/", "\\") for p in ISOLATED} | {p for p in ISOLATED}
    on_disk = {
        str(p.relative_to(RLM)).replace("\\", "/")
        for p in RLM.rglob("*.py")
        if p.name not in {"__init__.py", "dispatcher.py", "rootclient.py",
                          "episode.py", "cli.py", "config.py", "lifecycle.py",
                          "errors.py"}
    }
    uncovered = on_disk - {p.replace("\\", "/") for p in listed}
    assert not uncovered, f"components missing from ISOLATED lint list: {uncovered}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_import_rules.py -v`
Expected: FAIL — `rlm` package does not exist (collection error / ModuleNotFoundError on path resolution is acceptable; the point is red).

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "rlm-halo"
version = "0.1.0"
description = "Local Recursive Language Model runtime (rlm-runtime-spec v0.2.2)"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.9",
    "pyyaml>=6.0",
    "duckdb>=1.1",
    "httpx>=0.27",
]

[project.scripts]
rlm = "rlm.cli:main"

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "hypothesis>=6.112",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
filterwarnings = ["error"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rlm"]
```

```python
# rlm/__init__.py
"""rlm-halo: a local Recursive Language Model runtime.

The scaffold disposes; the LLM proposes (spec §2, invariant I1).
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

```python
# rlm/errors.py
"""Shared types. Imports nothing from rlm — every component may import this."""
from __future__ import annotations

from enum import StrEnum


class Outcome(StrEnum):
    """episodes.outcome (spec §6)."""

    SUCCESS = "success"
    FAIL = "fail"
    BUDGET_KILL = "budget_kill"
    CONTEXT_EXHAUSTED = "context_exhausted"
    ERROR = "error"


class StepStatus(StrEnum):
    """steps.status (spec §6)."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ActionType(StrEnum):
    """steps.action_type (spec §6)."""

    REPL_EXEC = "repl_exec"
    LLM_CALL = "llm_call"
    FINAL = "final"


class Actor(StrEnum):
    """steps.actor (spec §6)."""

    ROOT = "root"
    LEAF = "leaf"


class RlmError(Exception):
    """Base for all scaffold errors."""


class ConfigError(RlmError):
    """Config failed schema or cross-field validation; the run refuses to start."""


class SandboxError(RlmError):
    """The sandbox interpreter died, refused a cell, or could not be spawned."""


class DispatchError(RlmError):
    """A model-server call failed after its retry budget."""


class BudgetBreach(RlmError):
    """A C5 budget was breached. Carries the outcome the episode must record."""

    def __init__(self, outcome: Outcome, reason: str) -> None:
        super().__init__(f"{outcome}: {reason}")
        self.outcome = outcome
        self.reason = reason
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_import_rules.py -v`
Expected: PASS (all parametrized cases skip — no components exist yet — and the coverage test passes).
Also run `uv sync` first if the venv does not exist.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml rlm/__init__.py rlm/errors.py tests/test_import_rules.py
git commit -m "feat: project skeleton + dependency-rule lint (spec §5)"
```

---

## Task 2: C3 OutputTruncator + mandatory property suite

**Files:**
- Create: `rlm/truncate.py`
- Test: `tests/test_truncate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rlm.truncate.CellOutput` (frozen dataclass with fields `stdout: str`, `stderr: str`, `repr_: str`, `traceback: str`), `rlm.truncate.build_view(out: CellOutput) -> str`, `rlm.truncate.truncate_view(view: str, cap: int) -> str`, `rlm.truncate.observation_view(out: CellOutput, cap: int) -> str`.

Spec §5 C3: the view is the ordered, labeled concatenation of stdout, stderr, last-expression repr, and formatted traceback, truncated **as one unit**, with marker `[truncated: showing 2000 of 184,203 chars]`. It must not be overridable from inside the sandbox.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_truncate.py
from hypothesis import given, settings
from hypothesis import strategies as st

from rlm.truncate import CellOutput, observation_view, truncate_view

CAP = 2000


def test_labeled_order_is_stdout_stderr_repr_traceback():
    out = CellOutput(stdout="A", stderr="B", repr_="C", traceback="D")
    view = observation_view(out, CAP)
    assert view.index("A") < view.index("B") < view.index("C") < view.index("D")
    assert "[stdout]" in view and "[stderr]" in view
    assert "[repr]" in view and "[traceback]" in view


def test_empty_sections_are_omitted():
    out = CellOutput(stdout="only", stderr="", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert "[stderr]" not in view
    assert "only" in view


def test_marker_reports_true_total_and_cap():
    out = CellOutput(stdout="x" * 184_203, stderr="", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert len(view) <= CAP
    assert "[truncated: showing" in view
    assert "184," in view  # thousands-separated true total appears


def test_truncation_is_applied_to_the_concatenated_unit_not_per_stream():
    """A huge stdout must consume the budget so later streams are cut, not each
    stream getting its own allowance."""
    out = CellOutput(stdout="x" * 10_000, stderr="NEEDLE", repr_="", traceback="")
    view = observation_view(out, CAP)
    assert "NEEDLE" not in view
    assert len(view) <= CAP


@settings(max_examples=300, deadline=None)
@given(
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.text(max_size=5000),
    st.integers(min_value=50, max_value=4000),
)
def test_property_view_never_exceeds_cap_and_is_deterministic(a, b, c, d, cap):
    out = CellOutput(stdout=a, stderr=b, repr_=c, traceback=d)
    first = observation_view(out, cap)
    second = observation_view(out, cap)
    assert first == second
    assert len(first) <= cap


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=0, max_size=20_000), st.integers(min_value=50, max_value=3000))
def test_property_marker_itself_is_never_truncated(text, cap):
    view = truncate_view(text, cap)
    if len(text) > cap:
        assert view.endswith("chars]")
        assert view.count("[truncated: showing") == 1
    assert len(view) <= cap


def test_pathological_inputs_survive():
    for payload in ["\x00" * 5000, "🧨" * 5000, "x" * 1_000_000, "\r\n" * 5000]:
        out = CellOutput(stdout=payload, stderr="", repr_="", traceback="")
        view = observation_view(out, CAP)
        assert len(view) <= CAP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_truncate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm.truncate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# rlm/truncate.py
"""C3 — OutputTruncator (spec §5).

The hard cap on everything the root sees from the REPL. Applied scaffold-side,
after execution, over the concatenated view as ONE unit. Nothing running inside
the sandbox can raise, lower, or bypass it (I1).
"""
from __future__ import annotations

from dataclasses import dataclass

_MARKER = "[truncated: showing {shown:,} of {total:,} chars]"
# The longest marker we could ever need to append, used to reserve room so the
# marker itself is never truncated (property-tested).
_MARKER_RESERVE = len(_MARKER.format(shown=10**12, total=10**12))


@dataclass(frozen=True, slots=True)
class CellOutput:
    """Raw, untruncated result of one REPL cell. Stored in full by C6."""

    stdout: str = ""
    stderr: str = ""
    repr_: str = ""
    traceback: str = ""


def build_view(out: CellOutput) -> str:
    """Ordered, labeled concatenation. Empty sections are omitted."""
    parts: list[str] = []
    for label, body in (
        ("stdout", out.stdout),
        ("stderr", out.stderr),
        ("repr", out.repr_),
        ("traceback", out.traceback),
    ):
        if body:
            parts.append(f"[{label}]\n{body}")
    return "\n".join(parts)


def truncate_view(view: str, cap: int) -> str:
    """Truncate the assembled view to `cap` chars, marker included in the cap."""
    total = len(view)
    if total <= cap:
        return view
    budget = max(cap - _MARKER_RESERVE, 0)
    head = view[:budget]
    marker = _MARKER.format(shown=len(head), total=total)
    result = head + marker
    if len(result) > cap:  # pathologically small caps: marker wins, head yields
        result = result[-cap:] if len(marker) >= cap else head[: cap - len(marker)] + marker
    return result


def observation_view(out: CellOutput, cap: int) -> str:
    """What the root actually sees. `steps.observation_view` in §6."""
    return truncate_view(build_view(out), cap)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_truncate.py -v`
Expected: PASS — 7 tests including 2 hypothesis property suites.

- [ ] **Step 5: Commit**

```bash
git add rlm/truncate.py tests/test_truncate.py
git commit -m "feat(C3): OutputTruncator with mandatory property suite (spec §5)"
```

---

## Task 3: C2 deterministic chunker

**Files:**
- Create: `rlm/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing (the token counter is injected — the chunker must not import C4, per the dependency rule).
- Produces: `rlm.chunker.ChunkConfig` (frozen dataclass: `size_tokens: int`, `overhead_tokens: int`, `snap_to_boundary: bool`, `snap_tolerance: float`), `rlm.chunker.split(text: str, cfg: ChunkConfig, count_tokens: Callable[[str], int]) -> list[str]`.

Spec §5 C2 cut rule: fixed target size from config; cut point snapped to the nearest paragraph/newline/code-block boundary within a ±10% token tolerance; deterministic tie-break (earliest boundary). Snap ships on/off behind one config flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
import pytest

from rlm.chunker import ChunkConfig, split


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def count(text: str) -> int:
    """Deterministic stand-in for the leaf server's /tokenize (1 token/word)."""
    return len(text.split())


CFG = ChunkConfig(size_tokens=100, overhead_tokens=20, snap_to_boundary=True,
                  snap_tolerance=0.10)


def test_covers_every_token_exactly_once_when_rejoined():
    text = words(1000)
    chunks = split(text, CFG, count)
    assert "".join(chunks) == text


def test_no_chunk_exceeds_target_plus_tolerance():
    text = words(1000)
    for chunk in split(text, CFG, count):
        assert count(chunk) <= CFG.size_tokens * (1 + CFG.snap_tolerance) + 1


def test_is_deterministic():
    text = words(997)
    assert split(text, CFG, count) == split(text, CFG, count)


def test_snaps_to_paragraph_boundary_within_tolerance():
    # a paragraph break sits at token 95 — inside the -5% tolerance of a 100 cut
    text = words(95) + "\n\n" + words(200)
    chunks = split(text, CFG, count)
    assert chunks[0].endswith("\n\n") or chunks[1].startswith("w95")


def test_snap_disabled_cuts_at_exact_target():
    cfg = ChunkConfig(size_tokens=100, overhead_tokens=20, snap_to_boundary=False,
                      snap_tolerance=0.10)
    text = words(95) + "\n\n" + words(200)
    chunks = split(text, cfg, count)
    assert count(chunks[0]) == 100


def test_earliest_boundary_wins_ties():
    # two equidistant boundaries around the target: the earlier one must win
    text = words(95) + "\n\n" + words(9) + "\n\n" + words(200)
    chunks = split(text, CFG, count)
    assert count(chunks[0]) == 95


def test_short_text_yields_one_chunk():
    assert len(split(words(10), CFG, count)) == 1


def test_empty_text_yields_no_chunks():
    assert split("", CFG, count) == []


def test_rejects_nonsense_config():
    with pytest.raises(ValueError):
        split(words(10), ChunkConfig(0, 20, True, 0.1), count)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm.chunker'`.

- [ ] **Step 3: Write minimal implementation**

```python
# rlm/chunker.py
"""C2 — the deterministic chunker (spec §5).

The root chunks ONLY through this utility: free-form chunking in model code
would make chunk_size advisory (a soft I1 violation) and render the §7 #2 sweep
uncontrolled. B2 and B3 (§8) use this verbatim.

The token counter is injected so this module never imports an LLM client
(dependency rule, §5).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Boundary classes, best first. A cut is snapped to the latest boundary of the
# best available class inside the tolerance window; ties break to the earliest
# character offset (deterministic).
_BOUNDARIES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n```\n"),      # fenced code-block edge
    re.compile(r"\n\s*\n"),      # paragraph break
    re.compile(r"\n"),           # line break
)


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    size_tokens: int
    overhead_tokens: int
    snap_to_boundary: bool
    snap_tolerance: float


def _char_for_token_target(text: str, target: int,
                           count_tokens: Callable[[str], int]) -> int:
    """Smallest char offset whose prefix holds >= target tokens (binary search)."""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def split(text: str, cfg: ChunkConfig,
          count_tokens: Callable[[str], int]) -> list[str]:
    """Split `text` into chunks of ~cfg.size_tokens, snapped to boundaries.

    Concatenating the result reproduces `text` byte-for-byte.
    """
    if cfg.size_tokens <= 0:
        raise ValueError("chunk size_tokens must be positive")
    if not 0.0 <= cfg.snap_tolerance < 1.0:
        raise ValueError("snap_tolerance must be in [0, 1)")
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        rest = text[start:]
        if count_tokens(rest) <= cfg.size_tokens:
            chunks.append(rest)
            break

        cut = _char_for_token_target(rest, cfg.size_tokens, count_tokens)
        if cfg.snap_to_boundary:
            cut = _snap(rest, cut, cfg, count_tokens)
        cut = max(cut, 1)  # never make zero progress
        chunks.append(rest[:cut])
        start += cut
    return chunks


def _snap(rest: str, cut: int, cfg: ChunkConfig,
          count_tokens: Callable[[str], int]) -> int:
    """Move `cut` to the best boundary inside the ±tolerance token window."""
    lo_tokens = int(cfg.size_tokens * (1 - cfg.snap_tolerance))
    hi_tokens = int(cfg.size_tokens * (1 + cfg.snap_tolerance))
    lo = _char_for_token_target(rest, lo_tokens, count_tokens)
    hi = min(_char_for_token_target(rest, hi_tokens, count_tokens), len(rest))
    if lo >= hi:
        return cut
    window = rest[lo:hi]
    for pattern in _BOUNDARIES:
        hits = [m.end() + lo for m in pattern.finditer(window)]
        if hits:
            return min(hits)  # earliest boundary of the best class wins ties
    return cut
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_chunker.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/chunker.py tests/test_chunker.py
git commit -m "feat(C2): deterministic boundary-snapping chunker (spec §5)"
```

---

## Probe-Verified Decisions (binding for Tasks 4–17)

Every mechanic below was **executed on this box** by the design-probe workflow and, where probes disagreed, re-tested by the cross-check. Complete verbatim recipes and reference implementations: **`docs/superpowers/plans/2026-08-13-capa1-probe-recipes.md`** (cited below as *Recipes §probe-name*). Do not re-litigate these without new measurements.

| # | Decision | Why (measured) |
|---|---|---|
| D1 | **Spawn with raw `CreateProcessW`, not `subprocess.Popen`.** One `ProcThreadAttributeList(count=2)` carries BOTH `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` (0x00020002) and `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` (0x00020009); flags `CREATE_SUSPENDED\|EXTENDED_STARTUPINFO_PRESENT\|CREATE_UNICODE_ENVIRONMENT\|CREATE_NO_WINDOW`; then `AssignProcessToJobObject`, then `ResumeThread`. | Popen can carry neither attribute and cannot win the race to the Job Object. Composite spawn verified end-to-end (profile 0.012 s, spawn 0.006 s, 9 cells inside the container). |
| D2 | **Duplicate handle values in the handle list are fatal** — `CreateProcessW` fails `ERROR_INVALID_PARAMETER (87)`. Dedupe with `list(dict.fromkeys(...))`. Include std handles yourself (`STARTF_USESTDHANDLES`: NUL in, per-episode log out/err). `CreatePipe` with `bInheritHandle=TRUE`, then `SetHandleInformation(parent_end, HANDLE_FLAG_INHERIT, 0)`, then `CloseHandle` the child's copies right after spawn or EOF never arrives. | Hit the moment stdout and stderr shared a handle. |
| D3 | **A Job memory cap does not kill** — `JOB_OBJECT_LIMIT_PROCESS_MEMORY` makes the *allocation fail*; the child gets `MemoryError` and survives (measured: 256 MB cap, child peaked 232 MB, caught it, kept running). To actually kill: `JobObjectAssociateCompletionPortInformation` + `TerminateJobObject` from the notification pump (0.06 s). | Model code with `try/except MemoryError` would otherwise hang forever. |
| D4 | **Enforce `max_wall_clock` with a scaffold-side timer** (measured precision 2.01 s for a 2.0 s deadline). Job CPU limits are a backstop only — a 2.0 s CPU cap fired at 7–8.5 s wall. | Coarse kernel enforcement cannot carry C5's contract. |
| D5 | **`KILL_ON_JOB_CLOSE` is the recovery guarantee** — verified that hard-killing the scaffold with zero cleanup code reaped child *and* grandchild. | This is what makes §6's crash recovery real. |
| D6 | **AppContainer (CapabilityCount=0) is the hard default**, minted fresh per episode from ctypes (`CreateAppContainerProfile`, no admin, 0.022 s; `DeleteAppContainerProfile` at close). Blocks external egress (`WSAEACCES 10013`), DNS, cross-process loopback (so the sandbox **cannot** reach llama-server — an I1 win), raw `ws2_32` FFI, and reads of the repo directory. | The one risk that could have killed the C1 design — "AppContainer blocks loopback, so asyncio cannot build its self-pipe" — was **tested and is false**: the self-pipe is same-process (127.0.0.1:51370 built fine inside the container). No fallback branch needed. |
| D7 | **Install-time ACL, granted once to `S-1-15-2-1` (ALL APPLICATION PACKAGES), never to a per-episode SID**, and only on (a) the interpreter tree and (b) a dedicated bootstrap dir containing nothing but the sandbox child. `config.yaml`, `prompts/`, and `traces/` live under the repo, which is denied by default (verified `PermissionError(13)`). | Per-SID grants accrue one dead ACE per episode; an orphan was already observed on disk. |
| D8 | **Bootstrap order is load-bearing: build the asyncio loop FIRST, then `sys.addaudithook`.** | The hook denies `socket.*`, and the Proactor self-pipe *is* a socketpair — hook-first breaks asyncio entirely. |
| D9 | **The child must end with `os._exit(0)`** after closing its write fd. | New defect found by cross-check: `asyncio.gather` inside an AppContainer exits `0xC0000008` at interpreter teardown *with all results intact*. Since gather is the fan-out idiom the prompts teach, every healthy multi-leaf episode would look like a crash. Bisected; `os._exit(0)` fixes it (0xC0000008 → 0x0). |
| D10 | **Delete exit-code semantics from C5/C6.** `outcome_reason` comes from the scaffold-side kill reason (C5 timer, completion-port pump, operator abort) plus an explicit `{"kind":"bye"}` frame the child sends before exiting. Any other exit code = "unattributed sandbox death". | Follows from D9. |
| D11 | **Bridge = two anonymous pipes + handle-list whitelist + length-prefixed JSON with request-id correlation**, pumped by one reader and one writer thread per side into asyncio via `loop.call_soon_threadsafe`. `json.dumps(obj, ensure_ascii=True).encode("ascii")` on both sides — **non-negotiable**, `ensure_ascii=False` raises on the lone surrogates C3's property suite generates (round-trip verified in both directions). 1 MiB read buffer (unbuffered `readline()` cost 3.11 s vs 0.03 s for an 8 MB injection — 100×). | AF_UNIX is inert on Windows (`bind(): bad family`); `socket.socketpair()` is AF_INET loopback (would violate the no-network rule); multiprocessing needs `PROCESS_DUP_HANDLE` on the scaffold (a straight I1 hole) or a machine-visible named pipe (probe enumerated and hijacked one from an unrelated process). Anonymous pipes create **zero** OS-namespace objects. |
| D12 | **Capture cell output at the `sys.stdout`/`sys.stderr` object level (StringIO), never at fd level.** | Keeps every payload `str`, which is what makes the JSON bridge and the DuckDB text columns safe. |
| D13 | **C3 order is: build labeled unit → `safe_text()` → `sanitize_control_tokens()` → truncate → append marker.** Both sanitizers run *before* truncation, on the concatenated unit. | Measured: truncate-first yields a 2002-char view against a 2000 cap and a marker denominator of 4,571 where the truth is 4,696 — breaking two properties §5 makes mandatory. `observation_full_ref` stores the **pre-sanitization** blob, so blob and view legitimately differ in length. |
| D14 | **Root request path: `POST /apply-template` → sha256 the returned string → `POST /completion` with that exact string.** Store the rendered string as a blob (`steps.root_request_ref`). Use `/completion` for the leaf too. | Byte-identity to the internal `/v1/chat/completions` rendering proven three ways (token count, `cache_n` longest-common-prefix, bitwise-equal top-20 logprobs) over 8 message shapes. `/v1/chat/completions` **never reports `id_slot`**, so §6's `slot_id` and §4's slot-affinity contract are unfillable on that path. Chat-template sha256 comes from `GET /props`. |
| D15 | **Run the root with `chat_template_kwargs: {"enable_thinking": false}`** (config flag so S5 can A/B it). | With thinking on, a 400-token budget was consumed entirely by CoT that never closed `</think>` and never emitted code. Off: 28 tokens, clean `stop_type: eos`, clean cell. |
| D16 | **Cell extraction is config-owned: `cell_extraction: {languages: [repl, python, py], select: first}`**, and the prompt sentence is generated from that key so file text and extractor can never disagree. Parse rule: split on the **last** `</think>` and keep the tail, then select per `select`. | Resolves a model-visible contradiction between the shipped prompt text and the extractor; also removes an A/B confound in §9 S1. |
| D17 | **Replay verifies prompt-assembly, not decoding.** Mode (i) offline: rehash the stored rendered string, assert `== root_view_hash`. Mode (ii) online: re-POST the re-derived message array to `/apply-template`, assert byte-equality, and assert `props.chat_template` sha256 matches `config_snapshot`. S3's gate is (i)+(ii); mode (i) alone survives a bare trace bundle. | Greedy decoding is **not** reproducible on this box: 3 identical requests at temperature 0 with a fixed seed produced 3 different 400-token outputs. §6/§8 wording must move from "reproduces the trajectory" to "reproduces the prompt-assembly". |
| D18 | **DuckDB: native ENUM/UUID/JSON, PK `(episode_id, step_idx)`, FK `steps → episodes`. Commits run on a `ThreadPoolExecutor(max_workers=1)`, never inline in the coroutine.** | Inline stalled the event loop **2,344 ms straight**; the executor keeps loop lag at median 5.5 ms (below the 14.7 ms idle baseline) at identical throughput. ENUM chosen over VARCHAR+CHECK because DuckDB supports neither ADD nor DROP CONSTRAINT, while `CREATE TYPE v2 + ALTER COLUMN SET DATA TYPE` is a one-liner. |
| D19 | **Every `str` reaching a text column goes through `safe_text()`, and `config_snapshot` is scrubbed BEFORE `json.dumps`.** | Lone surrogates crash the writer with an opaque C++ cast error; scrubbing after `json.dumps` turns them into escapes DuckDB then rejects. In the probe this cascaded silently: the episodes insert failed, every step failed the FK, and a no-op lifecycle hook swallowed all 240 errors. **Give the lifecycle log a real logger from day one.** |
| D20 | **Blobs are plain files (tiny ASCII-header container) in a per-episode dir, rolled into one `blobs.parquet` at export — never parquet-per-blob.** | Measured on 2000 realistic payloads: parquet-per-blob is strictly dominated (550 blobs/s @ 6,304 B vs plain 868 @ 24,115 B and gzip-1 794 @ 5,554 B), adds ~400 B footer per file, and puts a DuckDB query on the single writer's critical path. |
| D21 | **Write the parquet export at EVERY episode close, and run a whole `rlm bench` in ONE process.** In-process monitoring uses `writer_con.cursor()`, never a second `duckdb.connect(read_only=True)`. | The file lock is *total exclusion* on Windows: a second process fails on `connect`, `connect(read_only)`, `ATTACH`, `ATTACH READ_ONLY`, and even `shutil.copyfile`. Without a per-episode export there is literally no way to observe a multi-hour bench. |
| D22 | **Kill sequence has one owner and one order:** `SandboxSession.kill(reason, code)` → `TerminateJobObject` (never `proc.kill()`) → bridge cancels every in-flight handler → each cancelled handler writes its `status=cancelled` step → `await tl.drain()` → `await tl.aclose()` (CHECKPOINT) → exit. The completion-port pump is a daemon thread that only does `loop.call_soon_threadsafe(session._on_job_notification, ...)`. | Three probes proposed three kill primitives with three owners; `proc.kill()` bypasses the Job Object, and draining before cancellation loses the cancelled-step rows. |
| D23 | **Assert `PROCESS_INFORMATION.dwProcessId == the child's self-reported `os.getpid()`** (first bridge frame is `{"kind":"handshake","pid":...}`); refuse the episode on mismatch. Pin the interpreter as an absolute path in config, never `sys.executable`. | A uv venv trampoline was measured reporting pid 31312 for a real process 30776 — §6 recovery kills `episodes.sandbox_pid`, so a launcher pid orphans the real sandbox. |
| D24 | **Re-inject the reserved names (`context`, `chunks`, `llm_query`, `final_answer`) into `USER_NS` before every cell.** | The reference harness restores `RESERVED_TOOL_NAMES` after each execution; without it, model code rebinding `llm_query` intercepts its own sub-call plumbing — an I1 hole with teeth. |
| D25 | **Child hygiene: define `SandboxPolicyError` at module level in `rlm_sandbox`; install `sys.unraisablehook` + `warnings.showwarning` per cell; shadow `asyncio.new_event_loop`/`run`/`set_event_loop` with stubs that raise before any loop object is constructed.** | Verified leak: a denied `asyncio.new_event_loop()` in turn N leaves a half-built loop whose `__del__` fires during turn N+1, injecting absolute interpreter paths and a misattributed error into the *next* observation. Closure-local exception classes also leak `main.<locals>.` into every denied observation. |
| D26 | **Append-only root conversation.** Everything mutable (turn counter, remaining budget, task text) goes in the newest user message; nothing already sent is ever rewritten. | Measured: append-only growth reuses cleanly (`cache_n = prev_len − 4`, 83.2% over 6 turns); a mid-conversation edit collapses reuse to the edit point, which would silently destroy §7 #3c. |
| D27 | **The startup handshake cannot assert KV cache types via `/props`** — launches with `-ctk q8_0` and `-ctk f16` differed only in `media_marker`, a per-process nonce. Cache-type assertion moves to parsing the server's own stderr at `-lv 4`, which makes the launcher part of the scaffold contract (unique per-launch stderr filename, `-lv 4`, `build_info` cross-check before trusting the log). | §4 currently claims `/props` covers cache types; it does not, and the weakening is written down rather than glossed. |

**Gaps deliberately deferred (recorded so they are not rediscovered as surprises):** the leaf server was never probed (S2 blocker: `/apply-template` byte-identity on the leaf, `id_slot` behaviour at `-np 8`, slot routing under `--slot-prompt-similarity`, `cache_n` under continuous batching); AppContainer under memory pressure with both servers resident; the completion-port pump has never run alongside the full five-thread scaffold; the 32 MB `chunks` setvar was measured on a plain pipe, not through an AppContainer bridge. **Pre-registered S2 fix (write it down now so S2 failing is not a redesign):** nothing enforces §4's `[chunk][question]` layout because C4 receives one opaque `prompt` string — the fix is a `chunk=` kwarg on `llm_query`, an API change, not a prompt edit.

---

## Task 4: Config schema + prompt registry loader

**Files:**
- Create: `rlm/config.py`, `config.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `rlm.errors.ConfigError`.
- Produces: `rlm.config.Config` (pydantic model, `extra="forbid"`), `rlm.config.load_config(path: Path) -> Config`, `rlm.config.config_snapshot(cfg: Config, extra: dict) -> dict`, `rlm.config.PromptRegistry` with `.render_root(category: str) -> str`, `.leaf_prefix() -> str`, `.hashes() -> dict[str, str]`.

Per D7 the registry files live in the repo (denied to the sandbox). Per the resolved gap on hash stability: the loader **strips the `<!-- changelog -->` header before rendering** and records **both** the file-bytes sha256 and the rendered-body sha256.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import json
import textwrap

import pytest

from rlm.config import Config, PromptRegistry, config_snapshot, load_config
from rlm.errors import ConfigError


def write_prompt(tmp_path, name, body):
    p = tmp_path / "prompts"
    p.mkdir(exist_ok=True)
    f = p / name
    f.write_text(f"<!-- changelog\nv1 initial\n-->\n{body}", encoding="utf-8")
    return f


def test_extra_keys_are_forbidden(tmp_path):
    with pytest.raises(ConfigError, match="max_subcals"):
        Config.model_validate({"scaffold": {"budgets": {"max_subcals": 32}}})


def test_leaf_ctx_must_equal_parallel_times_slot(minimal_cfg_dict):
    bad = minimal_cfg_dict
    bad["servers"]["leaf"]["ctx"] = 12345
    with pytest.raises(ConfigError, match="ctx"):
        Config.model_validate(bad)


def test_semaphore_equals_leaf_parallel(valid_cfg):
    assert valid_cfg.scaffold.dispatch_concurrency == valid_cfg.servers.leaf.parallel


def test_mtp_forces_root_single_slot(minimal_cfg_dict):
    minimal_cfg_dict["servers"]["root"]["mtp"] = True
    minimal_cfg_dict["servers"]["root"]["parallel"] = 2
    with pytest.raises(ConfigError, match="mtp"):
        Config.model_validate(minimal_cfg_dict)


def test_prompt_sha256_mismatch_refuses_to_load(tmp_path, valid_cfg):
    f = write_prompt(tmp_path, "root.v1.md", "body")
    reg = PromptRegistry(root_path=f, root_sha256="0" * 64, leaf_prefix_path=f,
                         leaf_prefix_sha256="0" * 64, strategy_paths={},
                         strategy_sha256={})
    with pytest.raises(ConfigError, match="sha256"):
        reg.load()


def test_registry_strips_changelog_header_but_hashes_both(tmp_path):
    f = write_prompt(tmp_path, "root.v1.md", "REAL BODY")
    reg = PromptRegistry.from_files(root_path=f, leaf_prefix_path=f,
                                    strategy_paths={"default": f})
    reg.load()
    assert "changelog" not in reg.render_root("default")
    assert "REAL BODY" in reg.render_root("default")
    h = reg.hashes()
    assert h["root.file"] != h["root.body"]  # both recorded, and they differ


def test_strategy_selection_is_by_declared_category_only(tmp_path):
    a = write_prompt(tmp_path, "strat-needle.v1.md", "NEEDLE BLOCK")
    b = write_prompt(tmp_path, "strat-default.v1.md", "DEFAULT BLOCK")
    root = write_prompt(tmp_path, "root.v1.md", "ROOT")
    reg = PromptRegistry.from_files(root_path=root, leaf_prefix_path=root,
                                    strategy_paths={"needle": a, "default": b})
    reg.load()
    assert "NEEDLE BLOCK" in reg.render_root("needle")
    assert "DEFAULT BLOCK" in reg.render_root("default")
    with pytest.raises(ConfigError):
        reg.render_root("category-the-model-invented")


def test_config_snapshot_is_stable_and_json_serialisable(valid_cfg):
    a = json.dumps(config_snapshot(valid_cfg, {}), sort_keys=False)
    b = json.dumps(config_snapshot(valid_cfg, {}), sort_keys=False)
    assert a == b  # stable field order => stable hashing


def test_snapshot_scrubs_lone_surrogates_before_serialising(valid_cfg):
    snap = config_snapshot(valid_cfg, {"note": "bad" + chr(0xDCFF)})
    json.dumps(snap).encode("utf-8")  # must not raise (D19)


def test_cell_extraction_defaults_match_prompt_promise(valid_cfg):
    assert valid_cfg.scaffold.cell_extraction.select == "first"
    assert valid_cfg.scaffold.cell_extraction.languages[0] == "repl"
```

Add a `tests/conftest.py` providing `minimal_cfg_dict` (a dict mirroring `config.yaml`) and `valid_cfg` (`Config.model_validate(minimal_cfg_dict)`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rlm.config'`.

- [ ] **Step 3: Write minimal implementation**

Implement `rlm/config.py` with pydantic v2 models mirroring **spec Appendix A**, plus these additions this plan introduces: `servers.*.backend_dir` (absolute path to the llama.cpp build), `servers.*.log_path`, `scaffold.sandbox` (`interpreter: Path`, `bootstrap_dir: Path`, `network_isolation: Literal["appcontainer","audit_only"] = "appcontainer"`, `appcontainer_per_episode: bool = True`, `deny_ctypes: bool = False`, `memory_limit_mb: int = 4096`), `scaffold.cell_extraction` (`languages: list[str] = ["repl","python","py"]`, `select: Literal["first","last"] = "first"`), `scaffold.root.enable_thinking: bool = False` (D15), and `trace.export_every_episode: bool = True` (D21).

Cross-field validators (raise `ConfigError`):
- `leaf.ctx == leaf.parallel * (chunk.size_tokens + chunk.overhead_tokens)`
- `root.ctx == root.window_tokens`
- `scaffold.dispatch_concurrency == leaf.parallel`
- `budgets.max_predict[role] + chunk.size_tokens <= slot capacity`
- `root.mtp is True ⇒ root.parallel == 1`
- every prompt path exists and its sha256 matches the pinned value (when pinned)

`config_snapshot(cfg, extra)` returns `cfg.model_dump(mode="json")` merged with `extra`, passed through `safe_text` recursively **before** any `json.dumps` (D19). Wrap `pydantic.ValidationError` into `ConfigError` so callers only catch scaffold types.

`PromptRegistry.load()` reads each file as bytes, computes the file sha256, strips a leading `<!-- changelog ... -->` block, computes the body sha256, and caches both. `render_root(category)` returns `root_body + "\n\n" + strategy_body[category]`, raising `ConfigError` for an unknown category (I1: the model never chooses).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_config.py -v`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/config.py config.yaml tests/test_config.py tests/conftest.py
git commit -m "feat: config schema (extra=forbid) + sha256-pinned prompt registry (spec §5)"
```

---

## Task 5: Lifecycle log

**Files:**
- Create: `rlm/lifecycle.py`
- Test: `tests/test_lifecycle.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `rlm.lifecycle.Lifecycle` with `.event(kind: str, **fields) -> None`, `.close() -> None`; constructed as `Lifecycle(path: Path | None, stream=sys.stderr)`.

Spec §5: this log carries **only** what the trace store structurally cannot record — its own write failures, config refusals, handshake refusals, server health transitions, quiesce waits, recovery actions. Never episode data. D19: it must be a real logger from day one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lifecycle.py
import io
import json

from rlm.lifecycle import ALLOWED_KINDS, Lifecycle


def test_writes_jsonl_to_file_and_stream(tmp_path):
    buf = io.StringIO()
    lc = Lifecycle(tmp_path / "lc.jsonl", stream=buf)
    lc.event("server_health", server="leaf", state="up")
    lc.close()
    line = (tmp_path / "lc.jsonl").read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["kind"] == "server_health" and rec["server"] == "leaf"
    assert "ts" in rec
    assert json.loads(buf.getvalue().strip())["kind"] == "server_health"


def test_rejects_kinds_outside_the_narrow_allowlist():
    lc = Lifecycle(None, stream=io.StringIO())
    for kind in ("step", "episode", "observation", "llm_call"):
        assert kind not in ALLOWED_KINDS
    try:
        lc.event("step", foo=1)
    except ValueError as exc:
        assert "not a lifecycle kind" in str(exc)
    else:
        raise AssertionError("episode data must be refused (spec §5, I4)")


def test_surrogates_do_not_crash_the_logger(tmp_path):
    lc = Lifecycle(tmp_path / "lc.jsonl", stream=io.StringIO())
    lc.event("trace_write_failure", detail="bad" + chr(0xDCFF))
    lc.close()
    json.loads((tmp_path / "lc.jsonl").read_text(encoding="utf-8").strip())


def test_never_raises_into_the_caller(tmp_path):
    lc = Lifecycle(tmp_path / "nonexistent-dir" / "lc.jsonl", stream=io.StringIO())
    lc.event("quiesce_wait", seconds=1)  # must degrade, not explode
    lc.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_lifecycle.py -v`
Expected: FAIL — no module `rlm.lifecycle`.

- [ ] **Step 3: Write minimal implementation**

```python
# rlm/lifecycle.py
"""The narrow JSONL lifecycle log (spec §5).

NOT a second source of truth. Episode data belongs in the trace store (I4);
this file carries only the events the trace store structurally cannot record.
The S3 gate runs with this file deleted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

ALLOWED_KINDS = frozenset({
    "trace_write_failure",
    "config_refused",
    "handshake_refused",
    "server_health",
    "quiesce_wait",
    "recovery_action",
    "sandbox_spawn",
    "sandbox_death",
    "operator_abort",
})


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


class Lifecycle:
    def __init__(self, path: Path | None, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._fh = None
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:  # degrade to stream-only, never explode
                print(f"lifecycle: cannot open {path}: {exc}", file=self._stream)

    def event(self, kind: str, **fields: Any) -> None:
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"{kind!r} is not a lifecycle kind; episode data goes to the "
                "trace store (spec §5, I4)"
            )
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
               **_scrub(fields)}
        line = json.dumps(rec, ensure_ascii=True)
        print(line, file=self._stream, flush=True)
        if self._fh is not None:
            try:
                self._fh.write(line + "\n")
            except OSError as exc:
                print(f"lifecycle: write failed: {exc}", file=self._stream)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_lifecycle.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/lifecycle.py tests/test_lifecycle.py
git commit -m "feat: narrow JSONL lifecycle log (spec §5)"
```

---

## Task 6: C6 TraceLogger, schema, blobs, export, recovery

**Files:**
- Create: `rlm/schema.sql`, `rlm/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `rlm.errors.{Outcome,StepStatus,ActionType,Actor}`, `rlm.lifecycle.Lifecycle`.
- Produces: `rlm.trace.safe_text(s: str) -> str`, `rlm.trace.TraceLogger` with `async .start()`, `.open_episode(ep: dict) -> None`, `.put_step(step: dict, blobs: dict[str, bytes]) -> None`, `.close_episode(episode_id, outcome, reason, final_ref)`, `async .drain()`, `async .aclose()`, `.monitor() -> duckdb cursor`, `.export_bundle(dest: Path)`, and `rlm.trace.recover_orphans(db_path, lifecycle) -> list[str]`.

Implementation is **verbatim from Recipes §tracestore → Reference code** (`rlm/trace/logger.py`, the blob container, the DDL, and the recovery scan), with D18–D21 binding: one-thread executor, ENUM/UUID/JSON types, `safe_text` on every text column, `config_snapshot` scrubbed before `json.dumps`, plain-file blobs rolled to `blobs.parquet` at export, export written at every episode close, monitoring via `writer_con.cursor()`.

Two additions this plan makes beyond the probe: `steps.root_request_ref` (D14/D17 — the rendered root request blob, so offline replay can rehash without a server), and the mid-frame-death convention (deferred gap, now decided): a `repl_exec` whose result frame never arrives is written `status=error, error_detail="sandbox_died_mid_cell"`, and the episode closes `outcome=error, outcome_reason=sandbox_death`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace.py
import json
import uuid

import duckdb
import pytest

from rlm.errors import ActionType, Actor, Outcome, StepStatus
from rlm.trace import TraceLogger, recover_orphans, safe_text


def test_safe_text_survives_lone_surrogates():
    assert safe_text("a" + chr(0xDCFF) + "b")  # must not raise
    json.dumps(safe_text("a" + chr(0xD800)))


async def test_step_rows_and_blobs_round_trip(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t1", "task_hash": "h",
                     "config_snapshot": {"note": "x" + chr(0xDCFF)}})
    tl.put_step(
        {"episode_id": ep, "step_idx": 0, "actor": Actor.ROOT,
         "action_type": ActionType.REPL_EXEC, "status": StepStatus.OK,
         "action_payload": "print(1)", "observation_view": "[stdout]\n1"},
        blobs={"observation_full_ref": b"\x00\x01raw bytes"},
    )
    await tl.drain()
    tl.close_episode(ep, Outcome.SUCCESS, None, None)
    await tl.aclose()

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    row = con.execute(
        "SELECT status, observation_full_ref FROM steps WHERE episode_id = ?",
        [ep]).fetchone()
    assert row[0] == "ok"
    assert (tmp_path / "blobs" / ep / row[1].split("/")[-1]).exists()


async def test_blob_is_written_before_the_referencing_row(tmp_path):
    """Ordering is the durability contract: an orphan blob is recoverable,
    a row pointing at a missing file is not."""
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.ROOT,
                 "action_type": ActionType.REPL_EXEC, "status": StepStatus.OK},
                blobs={"observation_full_ref": b"payload"})
    await tl.drain()
    await tl.aclose()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    refs = con.execute("SELECT observation_full_ref FROM steps").fetchall()
    for (ref,) in refs:
        assert (tmp_path / "blobs" / ref).exists(), "dangling blob reference"


async def test_monitor_uses_a_sibling_cursor_not_a_second_connection(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    cur = tl.monitor()
    assert cur.execute("SELECT count(*) FROM steps").fetchone()[0] == 0
    with pytest.raises(duckdb.Error):
        duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True)
    await tl.aclose()


def test_recover_orphans_tombstones_null_outcome_episodes(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute((__import__("pathlib").Path(__file__).parents[1]
                 / "rlm" / "schema.sql").read_text())
    ep = str(uuid.uuid4())
    con.execute("INSERT INTO episodes (episode_id, task_id, task_hash) "
                "VALUES (?, ?, ?)", [ep, "t", "h"])
    con.close()
    assert recover_orphans(db, lifecycle=None) == [ep]
    con = duckdb.connect(str(db))
    assert con.execute("SELECT outcome, outcome_reason FROM episodes"
                       ).fetchone() == ("error", "orphaned_at_recovery")


async def test_export_bundle_is_self_contained(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.OK},
                blobs={"observation_full_ref": b"leaf answer"})
    await tl.drain()
    tl.close_episode(ep, Outcome.SUCCESS, None, None)
    dest = tmp_path / "bundle"
    tl.export_bundle(dest)
    await tl.aclose()
    con = duckdb.connect()  # in-memory: a foreign reader, no lock, no .duckdb
    got = con.execute(
        f"SELECT b.content FROM '{dest / 'steps.parquet'}' s "
        f"JOIN '{dest / 'blobs.parquet'}' b ON b.rel = s.observation_full_ref"
    ).fetchone()
    assert got[0] == b"leaf answer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_trace.py -v`
Expected: FAIL — no module `rlm.trace`.

- [ ] **Step 3: Write minimal implementation**

Copy `rlm/schema.sql` and `rlm/trace.py` from **Recipes §tracestore → Reference code**, then apply the two additions above (`root_request_ref` column; the mid-frame-death convention). The DDL must cover every §6 column including `dry_run`, `scaffold_instance_id`, `sandbox_pid`, `superseded_by`, `avg_power_w`/`energy_j`, `pkg_temp_c_start`/`pkg_temp_c_end`, `config_snapshot`, `scaffold_git_sha`, `benchmark_version`; and on `steps`: `parent_step_idx`, `call_id`, `retry_idx`, `depth`, `error_detail`, `root_view_hash`, `tokens_in`/`tokens_out`, `tokens_cached`, `slot_id`, `t_dispatch`/`t_first_byte`/`t_end`, `latency_queue_ms`, `latency_prefill_ms`/`latency_decode_ms`. **`cache_hit` must not exist** (removed in v0.1).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_trace.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/trace.py rlm/schema.sql tests/test_trace.py
git commit -m "feat(C6): DuckDB trajectory store, blobs-before-rows, export, recovery (spec §6)"
```

---

## Task 7: Job Object + AppContainer + composite spawn

**Files:**
- Create: `rlm/sandbox/__init__.py`, `rlm/sandbox/winjob.py`, `rlm/sandbox/winproc.py`
- Test: `tests/test_winjob.py`

**Interfaces:**
- Consumes: `rlm.errors.SandboxError`.
- Produces: `rlm.sandbox.winjob.Job` (`.assign(handle)`, `.terminate(code)`, `.watch(callback)`, `.close()`), `rlm.sandbox.winproc.AppContainer` (`.create(name) -> sid`, `.delete()`), `rlm.sandbox.winproc.spawn(exe, args, handles, appcontainer_sid, job, env, stdio) -> SpawnResult(pid, hprocess, hthread)`, `rlm.sandbox.winproc.kill_if_ours(pid, started_at) -> bool`.

Code is **verbatim from Recipes §sandbox → Reference code (`winjob.py`)** and the cross-check's composite-spawn reference (`parent.py`), which is the only version that carries both attributes in one list. D1, D2, D3, D5, D6, D7, D23 all live here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_winjob.py
import subprocess
import sys
import time

import pytest

from rlm.sandbox.winjob import Job
from rlm.sandbox.winproc import AppContainer, spawn

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def test_kill_on_job_close_reaps_the_tree(tmp_path):
    job = Job(memory_limit_mb=512, active_process_limit=1)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job.assign_pid(proc.pid)
    job.close()  # closing the sole handle must kill it (D5)
    assert proc.wait(timeout=10) is not None


def test_memory_limit_notifies_rather_than_killing_silently(tmp_path):
    """D3: the allocation fails; the pump is what turns that into a kill."""
    seen = []
    job = Job(memory_limit_mb=128, active_process_limit=1)
    job.watch(lambda msg, ts: seen.append(msg))
    code = "b = bytearray()\nwhile True: b += bytearray(8*1024*1024)"
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stderr=subprocess.DEVNULL)
    job.assign_pid(proc.pid)
    deadline = time.time() + 30
    while time.time() < deadline and not seen:
        time.sleep(0.2)
    job.terminate(0xB0DE)
    proc.wait(timeout=10)
    assert any("MEMORY" in m for m in seen), f"no memory notification: {seen}"


def test_active_process_limit_blocks_helper_processes():
    job = Job(memory_limit_mb=512, active_process_limit=1)
    code = ("import subprocess, sys\n"
            "try:\n"
            "    subprocess.Popen([sys.executable, '-c', 'pass'])\n"
            "    print('SPAWNED')\n"
            "except OSError:\n"
            "    print('BLOCKED')\n")
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)
    job.assign_pid(proc.pid)
    out, _ = proc.communicate(timeout=30)
    job.close()
    assert "SPAWNED" not in out


def test_appcontainer_profile_lifecycle():
    ac = AppContainer()
    sid = ac.create("rlm-test-probe")
    assert sid
    ac.delete()


def test_spawn_rejects_duplicate_handle_values():
    """D2: duplicates make CreateProcessW fail with ERROR_INVALID_PARAMETER."""
    from rlm.sandbox.winproc import dedupe_handles
    assert dedupe_handles([5, 7, 5, 9, 7]) == [5, 7, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_winjob.py -v`
Expected: FAIL — no module `rlm.sandbox.winjob`.

- [ ] **Step 3: Write minimal implementation**

Paste `winjob.py` from Recipes §sandbox (CreateJobObjectW → SetInformationJobObject with `KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | PROCESS_MEMORY | JOB_MEMORY | ACTIVE_PROCESS`, plus `JobObjectAssociateCompletionPortInformation` and the pump thread). Paste `winproc.py` from the cross-check reference: `CreateAppContainerProfile`/`DeleteAppContainerProfile`, `InitializeProcThreadAttributeList(count=2)` carrying `PROC_THREAD_ATTRIBUTE_HANDLE_LIST` **and** `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES`, `CreateProcessW(CREATE_SUSPENDED|EXTENDED_STARTUPINFO_PRESENT|CREATE_UNICODE_ENVIRONMENT|CREATE_NO_WINDOW)`, `AssignProcessToJobObject`, `ResumeThread`. Per D22 the pump callback must not terminate anything itself — it hands the notification to the caller.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_winjob.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/sandbox/__init__.py rlm/sandbox/winjob.py rlm/sandbox/winproc.py tests/test_winjob.py
git commit -m "feat(C1): Job Object limits, completion-port pump, AppContainer spawn"
```

---

## Task 8: The C1/C4 bridge

**Files:**
- Create: `rlm/bridge.py`
- Test: `tests/test_bridge.py`

**Interfaces:**
- Consumes: `rlm.errors.SandboxError`.
- Produces: `rlm.bridge.MAX_FRAME`, `rlm.bridge.encode_frame(obj) -> bytes`, `rlm.bridge.FrameReader`, `rlm.bridge.BridgeParent` with `async .request(kind, payload) -> dict`, `.on_request(handler)`, `.close()`, and `.pending_count`.

Code verbatim from **Recipes §bridge → Reference code (`rlm_bridge.py`)**. D11 and D12 bind: length-prefixed `json.dumps(..., ensure_ascii=True).encode("ascii")`, request-id correlation, one reader + one writer thread per side, `loop.call_soon_threadsafe` delivery, 1 MiB read buffer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bridge.py
import asyncio
import json

from rlm.bridge import MAX_FRAME, FrameReader, encode_frame


def test_frames_are_ascii_only_so_lone_surrogates_survive():
    """D11: ensure_ascii=False would raise UnicodeEncodeError here."""
    payload = {"text": "lone-" + chr(0xDCFF) + "-end"}
    raw = encode_frame(payload)
    raw.decode("ascii")  # must not raise
    body = json.loads(raw.split(b"\n", 1)[1])
    assert body["text"] == payload["text"]


def test_reader_reassembles_split_and_coalesced_frames():
    frames = [encode_frame({"i": i, "pad": "x" * 100}) for i in range(5)]
    blob = b"".join(frames)
    reader = FrameReader()
    got = []
    for i in range(0, len(blob), 7):  # pathological chunking
        got.extend(reader.feed(blob[i:i + 7]))
    assert [g["i"] for g in got] == [0, 1, 2, 3, 4]


def test_oversize_frame_is_refused_not_buffered():
    reader = FrameReader()
    try:
        reader.feed(f"{MAX_FRAME + 1}\n".encode())
    except ValueError as exc:
        assert "frame" in str(exc).lower()
    else:
        raise AssertionError("oversize frame must be refused")


async def test_eight_concurrent_requests_are_matched_out_of_order():
    """The fan-out idiom the prompt registry teaches must not deadlock."""
    parent, child = _in_process_pair()

    async def handler(kind, payload):
        await asyncio.sleep(0.05 if payload["i"] % 2 else 0.01)  # reply out of order
        return {"echo": payload["i"]}

    parent.on_request(handler)
    results = await asyncio.gather(*[child.request("llm_query", {"i": i})
                                     for i in range(8)])
    assert [r["echo"] for r in results] == list(range(8))


async def test_parent_death_fails_pending_requests_instead_of_hanging():
    parent, child = _in_process_pair()
    parent.on_request(lambda kind, payload: asyncio.sleep(30))
    task = asyncio.create_task(child.request("llm_query", {"i": 0}))
    await asyncio.sleep(0.1)
    parent.close()
    with_timeout = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    assert isinstance(with_timeout[0], Exception)
```

`_in_process_pair()` is a test helper in `tests/conftest.py` wiring two `BridgeParent`-style endpoints over `os.pipe()` pairs, so the framing/correlation logic is testable without spawning a sandbox.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_bridge.py -v`
Expected: FAIL — no module `rlm.bridge`.

- [ ] **Step 3: Write minimal implementation**

Paste from Recipes §bridge, keeping the request-id correlation table and the close path that fails every pending future.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_bridge.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/bridge.py tests/test_bridge.py
git commit -m "feat: C1/C4 bridge — anonymous pipes, ASCII-JSON framing, id correlation (spec §5)"
```

---

## Task 9: The sandbox child

**Files:**
- Create: `rlm/sandbox/child.py` (installed into the ACL'd bootstrap dir per D7)
- Test: `tests/test_sandbox_child.py`

**Interfaces:**
- Consumes: the bridge framing protocol (Task 8) over inherited fds.
- Produces: the child protocol — frames `{"kind": "handshake", "pid": int}`, `{"kind":"setvar","name","value"}`, `{"kind":"exec","cell": str}` → `{"kind":"result","stdout","stderr","repr","traceback"}`, `{"kind":"llm_query","prompt","role"}` (child→parent), `{"kind":"final_answer","value"}`, `{"kind":"bye"}`.

D8, D9, D12, D24, D25 all bind here, and every one of them came from a measured failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sandbox_child.py
"""Child-protocol behaviour, exercised through a real spawned sandbox."""
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


async def test_variables_persist_across_cells(session):
    await session.exec_cell("x = 41")
    out = await session.exec_cell("print(x + 1)")
    assert out.stdout.strip() == "42"


async def test_top_level_await_works(session):
    out = await session.exec_cell(
        "r = await llm_query('hello')\nprint(r)")
    assert "MOCK" in out.stdout


async def test_traceback_is_captured_and_interpreter_survives(session):
    out = await session.exec_cell("1/0")
    assert "ZeroDivisionError" in out.traceback
    assert "rlm" not in out.traceback.lower(), "scaffold frames must be scrubbed"
    out2 = await session.exec_cell("print('alive')")
    assert out2.stdout.strip() == "alive"


async def test_last_expression_repr_is_captured(session):
    out = await session.exec_cell("2 + 3")
    assert out.repr_.strip() == "5"


async def test_reserved_names_are_reinjected_every_cell(session):
    """D24: rebinding llm_query must not let model code intercept its own plumbing."""
    await session.exec_cell("llm_query = lambda *a, **k: 'HIJACKED'")
    out = await session.exec_cell("print(type(llm_query).__name__)")
    assert "HIJACKED" not in out.stdout
    assert out.stdout.strip() in {"function", "method", "coroutine"}


async def test_denied_event_loop_does_not_poison_the_next_cell(session):
    """D25: the half-built loop's __del__ used to leak host paths into turn N+1."""
    await session.exec_cell("import asyncio\nasyncio.new_event_loop()")
    out = await session.exec_cell("print('clean')")
    assert "proactor_events" not in out.stderr
    assert "AppData" not in out.stderr
    assert out.stdout.strip() == "clean"


async def test_policy_error_qualname_does_not_leak_scaffold_structure(session):
    out = await session.exec_cell("import asyncio\nasyncio.run(main())")
    assert "<locals>" not in (out.traceback + out.stderr)


async def test_egress_is_blocked_but_the_bridge_still_works(session):
    out = await session.exec_cell(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=5)\n"
        "    print('REACHED')\n"
        "except Exception as e:\n"
        "    print('BLOCKED', type(e).__name__)\n")
    assert "REACHED" not in out.stdout
    r = await session.exec_cell("print(await llm_query('still works'))")
    assert "MOCK" in r.stdout


async def test_gather_fanout_exits_cleanly(session):
    """D9: this exact shape used to exit 0xC0000008 under AppContainer."""
    out = await session.exec_cell(
        "import asyncio\n"
        "rs = await asyncio.gather(*[llm_query(f'q{i}') for i in range(8)])\n"
        "print(len(rs))")
    assert out.stdout.strip() == "8"
    code = await session.close()
    assert code == 0
```

`session` is a `tests/conftest.py` fixture spawning a real sandbox whose bridge parent answers every `llm_query` with `"MOCK:<prompt>"` — the mock dispatcher, exercising C1 without C4.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_sandbox_child.py -v`
Expected: FAIL — the `session` fixture cannot spawn (`rlm/sandbox/child.py` missing).

- [ ] **Step 3: Write minimal implementation**

Paste `sandbox_child.py` from Recipes §sandbox, then apply, in order:
1. **D8** — construct the asyncio loop *before* `sys.addaudithook`.
2. **D12** — StringIO capture at the `sys.stdout`/`sys.stderr` object level.
3. Cell execution: `ast.parse`, pop a trailing `ast.Expr`, compile body `exec` + tail `eval` **both** with `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`, `eval(code, USER_NS)` (not `exec` — `eval` returns the coroutine when `CO_COROUTINE` is set), await if `iscoroutine`. Register cell source in `linecache` for real traceback lines; scrub scaffold frames from `TracebackException.stack` including through `__cause__`/`__context__`.
4. **D24** — re-inject `context`, `chunks`, `llm_query`, `final_answer` into `USER_NS` before every cell.
5. **D25** — module-level `SandboxPolicyError`; per-cell `sys.unraisablehook` and `warnings.showwarning` routed into that cell's stderr buffer, dropping tracebacks that contain only scaffold/stdlib-internal frames; stub `asyncio.new_event_loop`/`run`/`set_event_loop` to raise *before* a loop object is constructed.
6. **D9** — end with `try: os.close(wfd)\nexcept OSError: pass\nos._exit(0)` after sending `{"kind":"bye"}`.
7. The `answer` guard (decided gap): inject an `answer` object whose `__setitem__` raises `SandboxPolicyError("call final_answer(value) to submit")`, converting the reference harness's trained termination reflex into an observable correction instead of a silent no-op (matters for the S5 LoRA row).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_sandbox_child.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/sandbox/child.py tests/test_sandbox_child.py
git commit -m "feat(C1): sandbox child — persistent REPL, top-level await, policy hygiene"
```

---

## Task 10: C1 SandboxManager session

**Files:**
- Create: `rlm/sandbox/manager.py`
- Test: `tests/test_sandbox_manager.py`

**Interfaces:**
- Consumes: `rlm.sandbox.winjob.Job`, `rlm.sandbox.winproc.{AppContainer,spawn}`, `rlm.bridge.BridgeParent`, `rlm.truncate.CellOutput`.
- Produces: `rlm.sandbox.manager.SandboxManager.session(episode_id, cfg) -> SandboxSession` (async context manager); `SandboxSession` with `async .setvar(name, value)`, `async .exec_cell(cell) -> CellOutput`, `.on_llm_query(handler)`, `.on_final_answer(handler)`, `async .kill(reason: str, code: int)`, `async .close() -> int`, `.pid`.

D22 (kill sequence) and D23 (handshake pid assert) bind here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sandbox_manager.py
import sys
import time

import pytest

from rlm.errors import SandboxError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


async def test_handshake_pid_must_match_the_spawned_pid(manager, cfg):
    async with manager.session("ep-1", cfg) as s:
        assert s.pid > 0
        out = await s.exec_cell("import os; print(os.getpid())")
        assert out.stdout.strip() == str(s.pid)  # D23


async def test_kill_terminates_via_the_job_and_reports_the_reason(manager, cfg):
    async with manager.session("ep-2", cfg) as s:
        await s.exec_cell("x = 1")
        await s.kill("wall_clock", 0xC5)
        assert s.kill_reason == "wall_clock"
        with pytest.raises(SandboxError):
            await s.exec_cell("print(x)")


async def test_setvar_injects_a_32mb_payload_through_the_bridge(manager, cfg):
    """Deferred gap: measure this through an AppContainer bridge, not a plain pipe."""
    payload = "x" * (32 * 1024 * 1024)
    async with manager.session("ep-3", cfg) as s:
        t0 = time.perf_counter()
        await s.setvar("context", payload)
        elapsed = time.perf_counter() - t0
        out = await s.exec_cell("print(len(context))")
        assert out.stdout.strip() == str(len(payload))
        assert elapsed < 10.0, f"32 MB setvar took {elapsed:.2f}s"


async def test_sandbox_cannot_read_the_repo(manager, cfg):
    """D7: config.yaml and prompts/ must be denied by default."""
    async with manager.session("ep-4", cfg) as s:
        out = await s.exec_cell(
            "try:\n"
            "    open(r'D:\\PROJECTS\\rlm-halo-framework\\config.yaml').read()\n"
            "    print('READABLE')\n"
            "except OSError as e:\n"
            "    print('DENIED')\n")
        assert out.stdout.strip() == "DENIED"


async def test_episodes_do_not_share_state(manager, cfg):
    async with manager.session("ep-5", cfg) as s:
        await s.exec_cell("marker = 'first-episode'")
    async with manager.session("ep-6", cfg) as s:
        out = await s.exec_cell("print('marker' in dir())")
        assert out.stdout.strip() == "False"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_sandbox_manager.py -v`
Expected: FAIL — no module `rlm.sandbox.manager`.

- [ ] **Step 3: Write minimal implementation**

Wire Tasks 7–9: create the Job, mint the per-episode AppContainer, `CreatePipe` the two bridge pipes (`bInheritHandle=TRUE`, then clear inheritance on the parent ends, dedupe the handle list per D2), spawn suspended with both attributes, assign to the job, resume, read the handshake frame and assert the pid (D23), close the child's handle copies. `kill()` implements D22's exact order. The job pump callback does only `loop.call_soon_threadsafe(self._on_job_notification, msg, ts)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_sandbox_manager.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/sandbox/manager.py tests/test_sandbox_manager.py
git commit -m "feat(C1): SandboxManager session lifecycle and kill ordering (spec §5)"
```

---

## Task 11: C4 LLMDispatcher + server client

**Files:**
- Create: `rlm/dispatcher.py`
- Test: `tests/test_dispatcher.py`

**Interfaces:**
- Consumes: `rlm.config.Config`, `rlm.errors.{DispatchError,StepStatus}`.
- Produces: `rlm.dispatcher.ServerClient` (`async .apply_template(messages, **kw) -> str`, `async .completion(prompt, *, n_predict, stream=True, id_slot=None) -> CompletionResult`, `async .tokenize(text) -> list[int]`, `async .props() -> dict`), `rlm.dispatcher.LLMDispatcher` (`async .query(prompt, role, call_id) -> str`, `.semaphore`), `rlm.dispatcher.MockDispatcher` (same interface, canned by `(role, sha256(prompt))`).

D14 binds (`/apply-template` + `/completion`, never `/v1/chat/completions` — it never reports `id_slot`). `CompletionResult` carries `content, tokens_in, tokens_out, cache_n, slot_id, t_first_byte, prompt_ms, predicted_ms`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dispatcher.py
import hashlib

import pytest

from rlm.dispatcher import MockDispatcher
from rlm.errors import DispatchError, StepStatus


async def test_mock_dispatcher_is_keyed_by_role_and_prompt_hash(tmp_path):
    fixtures = {f"leaf:{hashlib.sha256(b'q').hexdigest()}": "canned"}
    d = MockDispatcher(fixtures)
    assert await d.query("q", role="leaf", call_id="c1") == "canned"


async def test_preflight_rejects_oversize_prompts_without_dispatching(mock_server):
    d = mock_server.dispatcher(slot_capacity_tokens=100)
    with pytest.raises(DispatchError):
        await d.query("x " * 5000, role="leaf", call_id="c1")
    assert mock_server.dispatch_count == 0
    assert d.last_step["status"] == StepStatus.REJECTED


async def test_retries_share_a_call_id_and_increment_retry_idx(mock_server):
    mock_server.fail_times(2)
    d = mock_server.dispatcher()
    await d.query("q", role="leaf", call_id="c1")
    statuses = [s["status"] for s in d.steps]
    assert len(d.steps) == 3
    assert {s["call_id"] for s in d.steps} == {"c1"}
    assert [s["retry_idx"] for s in d.steps] == [0, 1, 2]
    assert statuses[-1] == StepStatus.OK


async def test_semaphore_never_exceeds_leaf_parallel(mock_server):
    d = mock_server.dispatcher(parallel=8)
    import asyncio
    await asyncio.gather(*[d.query(f"q{i}", role="leaf", call_id=f"c{i}")
                           for i in range(32)])
    assert mock_server.max_concurrent <= 8


async def test_server_death_produces_error_status_not_a_restart(mock_server):
    mock_server.kill()
    d = mock_server.dispatcher()
    with pytest.raises(DispatchError):
        await d.query("q", role="leaf", call_id="c1")
    assert d.steps[-1]["status"] == StepStatus.ERROR
    assert mock_server.restart_count == 0  # the scaffold never restarts servers


async def test_cancellation_aborts_the_stream(mock_server):
    import asyncio
    d = mock_server.dispatcher()
    task = asyncio.create_task(d.query("slow", role="leaf", call_id="c1"))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mock_server.last_request_disconnected
```

`mock_server` is a `tests/conftest.py` fixture: a tiny in-process ASGI/`http.server` stub speaking the `/tokenize`, `/completion` (SSE), `/props`, `/apply-template` shapes captured in Recipes §serverapi.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_dispatcher.py -v`
Expected: FAIL — no module `rlm.dispatcher`.

- [ ] **Step 3: Write minimal implementation**

Use `httpx.AsyncClient` with streamed `POST /completion` (`"stream": true`), parsing SSE events; the final event carries `timings` and `id_slot`. Cancellation closes the response context, which disconnects and aborts server-side generation (verified in Recipes §serverapi). Retries: `max_attempts` 3, backoff 1 s/4 s, per-call timeout 240 s, every attempt logged as its own step with shared `call_id` and incrementing `retry_idx`. The semaphore is `asyncio.Semaphore(cfg.servers.leaf.parallel)` and is owned here — never sized from anything the model can influence.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_dispatcher.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/dispatcher.py tests/test_dispatcher.py
git commit -m "feat(C4): dispatcher, streamed completion, retries, semaphore, mock (spec §5)"
```

---

## Task 12: Root client — render, hash, parse

**Files:**
- Create: `rlm/rootclient.py`
- Test: `tests/test_rootclient.py`

**Interfaces:**
- Consumes: `rlm.dispatcher.ServerClient`, `rlm.config.Config`.
- Produces: `rlm.rootclient.RootConversation` with `.append_user(text)`, `async .turn() -> RootTurn(raw, cell, view_hash, rendered, usage)`, `.messages`, and module functions `rlm.rootclient.strip_reasoning(text) -> str`, `rlm.rootclient.extract_cell(text, languages, select) -> str | None`.

D14, D15, D16, D17, D26 bind here.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rootclient.py
import hashlib

from rlm.rootclient import extract_cell, strip_reasoning


def test_strip_reasoning_keeps_the_tail_after_the_last_close_tag():
    text = "<think>\nplan A\n</think>\nmid<think>more</think>\nFINAL"
    assert strip_reasoning(text).strip() == "FINAL"


def test_strip_reasoning_is_a_noop_without_tags():
    assert strip_reasoning("just text").strip() == "just text"


def test_extract_cell_takes_the_first_block_by_default():
    text = "```repl\nA\n```\nprose\n```repl\nB\n```"
    assert extract_cell(text, ["repl", "python", "py"], "first").strip() == "A"


def test_extract_cell_accepts_every_configured_language():
    for lang in ("repl", "python", "py"):
        assert extract_cell(f"```{lang}\nX\n```", ["repl", "python", "py"],
                            "first").strip() == "X"


def test_extract_cell_returns_none_for_zero_blocks_and_unterminated_fences():
    assert extract_cell("no code here", ["repl"], "first") is None
    assert extract_cell("```repl\nunterminated", ["repl"], "first") is None
    assert extract_cell("```ruby\nputs 1\n```", ["repl"], "first") is None


async def test_view_hash_is_the_sha256_of_the_applied_template(fake_root_server):
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("hello")
    turn = await conv.turn()
    assert turn.view_hash == hashlib.sha256(turn.rendered.encode()).hexdigest()
    assert fake_root_server.last_completion_prompt == turn.rendered  # D14


async def test_conversation_growth_is_append_only(fake_root_server):
    """D26: rewriting a sent message collapses prefix-cache reuse."""
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("turn one")
    first = await conv.turn()
    conv.append_user("turn two")
    second = await conv.turn()
    assert second.rendered.startswith(first.rendered.rstrip("\n")[:200])


async def test_thinking_is_disabled_by_default(fake_root_server):
    conv = fake_root_server.conversation(system="SYS")
    conv.append_user("hi")
    await conv.turn()
    kw = fake_root_server.last_template_kwargs
    assert kw["chat_template_kwargs"]["enable_thinking"] is False  # D15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_rootclient.py -v`
Expected: FAIL — no module `rlm.rootclient`.

- [ ] **Step 3: Write minimal implementation**

`turn()` does: `POST /apply-template` with the current message array and `chat_template_kwargs={"enable_thinking": cfg.scaffold.root.enable_thinking}` → take `resp["prompt"]` as `rendered` → `view_hash = sha256(rendered)` → `POST /completion` with that exact string → `strip_reasoning` → `extract_cell` per config. `append_user` only ever appends (D26). Return `rendered` so the episode runner can store it as the `root_request_ref` blob (D17).

Cell-extraction miss (decided gap): return `cell=None`; the episode runner logs a `repl_exec` step with `status=rejected, error_detail="no_cell_extracted"`, feeds the root a scaffold-authored observation restating the required fence format, and counts the turn against the root window and wall clock but **not** against `max_subcalls`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_rootclient.py -v`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/rootclient.py tests/test_rootclient.py
git commit -m "feat: root conversation — apply-template hashing, reasoning strip, cell extraction"
```

---

## Task 13: C5 BudgetEnforcer + stateful property suite

**Files:**
- Create: `rlm/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `rlm.errors.{BudgetBreach,Outcome}`.
- Produces: `rlm.budget.Budgets` (frozen dataclass from config), `rlm.budget.BudgetEnforcer` with `.admit(prompt_tokens, role) -> Reservation`, `.settle(reservation, actual_in, actual_out)`, `.note_root_usage(used, window)`, `.start_clock()`, `.check_wall_clock()`, `.on_breach(callback)`, `.subcalls_used`, `.tokens_used`.

Spec §5 C5 plus the mandatory hypothesis stateful machine.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budget.py
import pytest
from hypothesis import settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from rlm.budget import BudgetEnforcer, Budgets
from rlm.errors import BudgetBreach, Outcome

B = Budgets(max_depth=1, max_subcalls=32, max_wall_clock_s=900,
            max_total_tokens=1_500_000, max_predict={"root": 1024, "leaf": 512})


def test_admission_refuses_when_reservation_would_exceed_cap():
    b = BudgetEnforcer(Budgets(1, 32, 900, 1000, {"leaf": 500}))
    b.admit(400, "leaf")
    with pytest.raises(BudgetBreach):
        b.admit(400, "leaf")  # 400+500 + 400+500 > 1000


def test_retried_call_counts_once_against_max_subcalls():
    b = BudgetEnforcer(B)
    r = b.admit(10, "leaf", call_id="c1")
    b.settle(r, 10, 5)
    r2 = b.admit(10, "leaf", call_id="c1")  # same logical call, retry
    b.settle(r2, 10, 5)
    assert b.subcalls_used == 1
    assert b.tokens_used == 30  # every attempt's tokens count


def test_root_window_breach_is_context_exhausted():
    b = BudgetEnforcer(B)
    with pytest.raises(BudgetBreach) as exc:
        b.note_root_usage(used=29_500, window=32_768)  # >= 90%
    assert exc.value.outcome == Outcome.CONTEXT_EXHAUSTED


def test_no_admissions_after_breach():
    b = BudgetEnforcer(Budgets(1, 1, 900, 1_000_000, {"leaf": 10}))
    b.settle(b.admit(1, "leaf", call_id="a"), 1, 1)
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="b")
    with pytest.raises(BudgetBreach):
        b.admit(1, "leaf", call_id="c")  # still refusing, never warn-and-continue


class BudgetMachine(RuleBasedStateMachine):
    """The mandatory C5 stateful suite (spec §5 Testing)."""

    calls = Bundle("calls")

    def __init__(self):
        super().__init__()
        self.b = BudgetEnforcer(Budgets(1, 8, 900, 10_000, {"leaf": 100}))
        self.inflight = []
        self.breached = False

    @rule(target=calls, tokens=st_int(1, 200))
    def dispatch(self, tokens):
        try:
            r = self.b.admit(tokens, "leaf", call_id=f"c{len(self.inflight)}")
        except BudgetBreach:
            self.breached = True
            return None
        self.inflight.append(r)
        return r

    @rule(r=calls)
    def complete(self, r):
        if r is not None and r in self.inflight:
            self.inflight.remove(r)
            self.b.settle(r, r.prompt_tokens, 10)

    @rule(r=calls)
    def cancel(self, r):
        if r is not None and r in self.inflight:
            self.inflight.remove(r)
            self.b.cancel(r)

    @invariant()
    def reservations_never_exceed_cap(self):
        assert self.b.reserved_total <= self.b.budgets.max_total_tokens

    @invariant()
    def overshoot_is_bounded_by_inflight_times_max_predict(self):
        bound = len(self.inflight) * self.b.budgets.max_predict["leaf"]
        assert self.b.reserved_total - self.b.tokens_used <= bound

    @invariant()
    def no_admissions_after_breach(self):
        if self.breached:
            with pytest.raises(BudgetBreach):
                self.b.admit(1, "leaf", call_id="post-breach")


def st_int(lo, hi):
    from hypothesis import strategies as st
    return st.integers(min_value=lo, max_value=hi)


TestBudgetMachine = BudgetMachine.TestCase
TestBudgetMachine.settings = settings(max_examples=200, stateful_step_count=40,
                                      deadline=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_budget.py -v`
Expected: FAIL — no module `rlm.budget`.

- [ ] **Step 3: Write minimal implementation**

Track `reserved_total` (admitted reservations) and `tokens_used` (settled actuals) separately — that separation is what makes the overshoot invariant provable. `admit` raises `BudgetBreach` when `reserved_total + prompt_tokens + max_predict[role] > max_total_tokens`, when `subcalls_used + 1 > max_subcalls` for a new `call_id`, or when already breached. `note_root_usage` raises `BudgetBreach(CONTEXT_EXHAUSTED)` at ≥90%. `check_wall_clock` raises `BudgetBreach(BUDGET_KILL, "wall_clock")`. No method ever warns and continues (spec §5 *Must not*).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_budget.py -v`
Expected: PASS — 4 unit tests + the stateful machine.

- [ ] **Step 5: Commit**

```bash
git add rlm/budget.py tests/test_budget.py
git commit -m "feat(C5): budget enforcer with mandatory stateful property suite (spec §5)"
```

---

## Task 14: Prompt files

**Files:**
- Create: `prompts/root.v1.md`, `prompts/root.v2.md`, `prompts/leaf-prefix.v1.md`, `prompts/strat-needle.v1.md`, `prompts/strat-aggregation.v1.md`, `prompts/strat-synthesis.v1.md`, `prompts/strat-codeqa.v1.md`, `prompts/strat-default.v1.md`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `rlm.config.PromptRegistry`.
- Produces: the eight registry files and their pinned sha256 values in `config.yaml`.

The complete final text of all eight files is in **Recipes §prompts → Reference code**, delimited by path — write them **verbatim**. They were authored against the actual conditioning of `alexzhang13/rlm` (`RLM_SYSTEM_PROMPT` + `ORCHESTRATOR_ADDENDUM`, read from source, not from blog quotes of the deprecated one), `dspy.RLM`, and the mit-oasys model card, and the probe ran 29 verification checks over them (29 pass / 0 fail).

Alignment decisions already baked in: `llm_query` name and first positional arg kept exactly; ` ```repl ` adopted as the canonical fence (the convention the S5 LoRA is RL-trained on); `final_answer(value)` kept per §6 even though the current harness uses an `answer` dict (the gap is mitigated by Task 9's `answer` guard); no capacity numbers and no config numbers appear in any file (budgets are described qualitatively, keeping the files hash-stable and I1 clean); the last-expression-repr behaviour is stated as the **opposite** of the harness default, because our C3 includes it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
import hashlib
import re
from pathlib import Path

import pytest

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
FILES = ["root.v1.md", "root.v2.md", "leaf-prefix.v1.md", "strat-needle.v1.md",
         "strat-aggregation.v1.md", "strat-synthesis.v1.md",
         "strat-codeqa.v1.md", "strat-default.v1.md"]


@pytest.mark.parametrize("name", FILES)
def test_every_file_exists_and_opens_with_a_changelog_header(name):
    text = (PROMPTS / name).read_text(encoding="utf-8")
    assert text.startswith("<!-- changelog"), "spec §5 requires the header"
    assert re.search(r"^v\d+ ", text.split("-->")[0], re.M)


def test_leaf_prefix_carries_no_volatile_tokens():
    """§4: byte-identical prefix — no timestamps, ids, counters, dates."""
    text = (PROMPTS / "leaf-prefix.v1.md").read_text(encoding="utf-8")
    body = text.split("-->", 1)[1]
    for pattern in (r"\d{4}-\d{2}-\d{2}", r"\bid\b\s*[:=]", r"\{[a-z_]+\}",
                    r"run[_ ]?id", r"episode"):
        assert not re.search(pattern, body, re.I), f"volatile token {pattern!r}"


def test_root_prompts_use_the_injected_api_names_exactly():
    for name in ("root.v1.md", "root.v2.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8")
        assert "llm_query" in text and "final_answer" in text
        assert "chunks" in text and "context" in text
        assert "llm_query_batched" not in text  # not our API
        assert "SUBMIT(" not in text and "FINAL(" not in text


def test_the_two_ab_variants_differ_only_by_the_exemplar_block():
    v1 = (PROMPTS / "root.v1.md").read_text(encoding="utf-8")
    v2 = (PROMPTS / "root.v2.md").read_text(encoding="utf-8")
    assert len(v2) > len(v1), "v2 is tips + worked exemplars"
    tips_v1 = v1.split("-->", 1)[1].strip()
    assert tips_v1 in v2, "v2 must contain v1's tips verbatim (controlled A/B)"


def test_exemplars_use_the_canonical_fence_and_our_api():
    v2 = (PROMPTS / "root.v2.md").read_text(encoding="utf-8")
    assert "```repl" in v2
    assert "await llm_query(" in v2
    assert "final_answer(" in v2
    assert "asyncio.gather" in v2  # the fan-out idiom


def test_prompt_promise_matches_the_configured_extractor():
    """D16: the file text is generated from cell_extraction; they cannot disagree."""
    from rlm.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    v1 = (PROMPTS / "root.v1.md").read_text(encoding="utf-8")
    if cfg.scaffold.cell_extraction.select == "first":
        assert "only the first runs" in v1
    else:
        assert "only the last runs" in v1


def test_extraction_shaped_strategies_carry_the_evidence_span_check():
    for name in ("strat-needle.v1.md", "strat-aggregation.v1.md",
                 "strat-synthesis.v1.md"):
        text = (PROMPTS / name).read_text(encoding="utf-8").lower()
        assert "evidence" in text, f"{name} missing the R12/R5 evidence-span check"


def test_config_pins_match_the_files_on_disk():
    from rlm.config import load_config
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    for path, pinned in cfg.pinned_prompt_hashes().items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert actual == pinned, f"{path} drifted from its config pin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_prompts.py -v`
Expected: FAIL — `prompts/` is empty.

- [ ] **Step 3: Write minimal implementation**

Write the eight files verbatim from Recipes §prompts, then compute each file's sha256 and paste it into `config.yaml`'s `prompts:` block.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_prompts.py -v`
Expected: PASS — 8 tests (one parametrized ×8).

- [ ] **Step 5: Commit**

```bash
git add prompts/ config.yaml tests/test_prompts.py
git commit -m "feat: prompt registry — root A/B variants, leaf prefix, 5 strategy templates"
```

---

## Task 15: Episode runner (composition root)

**Files:**
- Create: `rlm/context.py`, `rlm/episode.py`
- Test: `tests/test_episode.py`

**Interfaces:**
- Consumes: everything.
- Produces: `rlm.context.load_context(spec) -> str`, `rlm.episode.EpisodeResult(episode_id, outcome, reason, final_answer)`, `async rlm.episode.run_episode(task, cfg, *, dispatcher, trace, lifecycle) -> EpisodeResult`.

This is the only module permitted to import both C4 and the isolated components. It owns: the `/props` startup handshake (§4, weakened per D27 — cache types come from the `-lv 4` launch log, not `/props`), C2 materialisation of `context` and `chunks`, the root turn loop, the `final_answer` channel, budget wiring, and outcome determination.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_episode.py
import pytest

from rlm.errors import Outcome


async def test_happy_path_emits_final_and_logs_every_step(episode_env):
    """Mock dispatcher, real sandbox, real C1/C2/C3/C5/C6 (spec §5 dry-run)."""
    env = episode_env(root_script=[
        "```repl\nprint(len(chunks))\n```",
        "```repl\nfinal_answer('42')\n```",
    ])
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    assert res.final_answer == "42"
    kinds = [s["action_type"] for s in env.steps()]
    assert kinds[-1] == "final"
    assert all(s["episode_id"] == res.episode_id for s in env.steps())


async def test_final_answer_is_the_only_terminal_channel(episode_env):
    """I2: prose is never parsed as an answer — it would smuggle context past C3."""
    env = episode_env(root_script=[
        "The answer is 42.",              # prose only, no cell
        "```repl\nfinal_answer('42')\n```",
    ])
    res = await env.run()
    assert res.outcome == Outcome.SUCCESS
    no_cell = [s for s in env.steps() if s.get("error_detail") == "no_cell_extracted"]
    assert len(no_cell) == 1
    assert no_cell[0]["status"] == "rejected"


async def test_root_never_receives_untruncated_output(episode_env):
    env = episode_env(root_script=[
        "```repl\nprint(context)\n```",
        "```repl\nfinal_answer('done')\n```",
    ], context="X" * 500_000, truncation_cap=2000)
    await env.run()
    exec_step = [s for s in env.steps() if s["action_type"] == "repl_exec"][0]
    assert len(exec_step["observation_view"]) <= 2000
    assert "[truncated: showing" in exec_step["observation_view"]
    assert env.blob(exec_step["observation_full_ref"]).__len__() > 400_000  # I2


async def test_wall_clock_breach_is_budget_kill_and_persists_the_trace(episode_env):
    env = episode_env(root_script=["```repl\nwhile True:\n    pass\n```"],
                      max_wall_clock_s=5)
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert res.reason == "wall_clock"
    assert env.episode_row()["ended_at"] is not None
    assert env.steps(), "partial trace must survive the kill"


async def test_runaway_subcalls_terminate_deterministically(episode_env):
    env = episode_env(root_script=[
        "```repl\nimport asyncio\n"
        "await asyncio.gather(*[llm_query(f'q{i}') for i in range(100)])\n```",
    ], max_subcalls=8)
    res = await env.run()
    assert res.outcome == Outcome.BUDGET_KILL
    assert len([s for s in env.steps() if s["action_type"] == "llm_call"]) <= 8


async def test_dry_run_episodes_are_flagged(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('x')\n```"])
    res = await env.run()
    assert env.episode_row()["dry_run"] is True
    assert env.episode_row()["config_snapshot"]["scaffold"]["dispatcher"] == "mock"


async def test_root_view_hash_is_recorded_for_every_root_turn(episode_env):
    env = episode_env(root_script=["```repl\nfinal_answer('x')\n```"])
    await env.run()
    root_steps = [s for s in env.steps() if s["actor"] == "root"]
    assert all(s["root_view_hash"] for s in root_steps)
    assert all(s["root_request_ref"] for s in root_steps)  # D17 offline replay
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_episode.py -v`
Expected: FAIL — no module `rlm.episode`.

- [ ] **Step 3: Write minimal implementation**

Loop: handshake → open episode row (NULL outcome, so recovery can tombstone it) → spawn sandbox → `setvar("context")` and `setvar("chunks")` (chunked via C2 with the dispatcher's `/tokenize` injected as `count_tokens`) → until terminal: `conv.turn()` → store `rendered` blob + `root_view_hash` → `extract_cell` → `exec_cell` → C3 view → append observation as a **new** user message (D26) → repeat. `llm_query` frames from the sandbox are handled by C4 under the semaphore, each logged with `parent_step_idx` set to the `repl_exec` that spawned it. `final_answer` frames end the episode. Any `BudgetBreach` runs D22's kill sequence and records the outcome.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_episode.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/context.py rlm/episode.py tests/test_episode.py
git commit -m "feat: episode runner — composition root, outcomes, final_answer channel"
```

---

## Task 16: CLI — validate, run, replay

**Files:**
- Create: `rlm/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything.
- Produces: `rlm validate`, `rlm run <task-file>`, `rlm replay <episode-id> [--online]`.

`bench` and `export` belong to later slices; `export_bundle` already exists (Task 6) and `rlm run` calls it at episode close per D21. Non-goals stay non-goals: no daemon, no REST API, no web UI, no interactive chat mode.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import json

from rlm.cli import main


def test_validate_refuses_a_bad_config(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("scaffold:\n  budgets:\n    max_subcals: 32\n")
    rc = main(["validate", "--config", str(tmp_path / "config.yaml")])
    assert rc != 0
    assert "max_subcals" in capsys.readouterr().err


def test_validate_asserts_the_sandbox_cannot_read_the_repo(tmp_path, capsys, valid_config_file):
    """D7: turn the filesystem-confinement claim into a checked invariant."""
    rc = main(["validate", "--config", str(valid_config_file), "--no-server-probe"])
    out = capsys.readouterr().out
    assert "sandbox filesystem confinement: OK" in out
    assert rc == 0


def test_run_prints_episode_id_and_outcome(mock_episode_env, capsys):
    rc = main(["run", str(mock_episode_env.task_file), "--config",
               str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "episode_id" in out and "success" in out


def test_replay_offline_verifies_hashes_with_no_server(mock_episode_env, capsys):
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "root_view_hash: OK" in out


def test_replay_fails_loudly_on_a_tampered_blob(mock_episode_env, capsys):
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    mock_episode_env.tamper_root_request_blob(ep)
    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    assert rc != 0
    assert "hash mismatch" in capsys.readouterr().err


def test_replay_works_with_the_lifecycle_log_deleted(mock_episode_env):
    """S3 gate condition, exercised early: no logs, no stdout, trace store only."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    mock_episode_env.delete_lifecycle_log()
    assert main(["replay", mock_episode_env.last_episode_id(), "--config",
                 str(mock_episode_env.config_file)]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_cli.py -v`
Expected: FAIL — no module `rlm.cli`.

- [ ] **Step 3: Write minimal implementation**

`argparse` with three subcommands. `validate`: load config, probe `/props` on both servers (skippable with `--no-server-probe`), assert build/`n_ctx`/`n_parallel` against config, parse the `-lv 4` launch log for cache types (D27), and spawn a throwaway AppContainer child that tries to read `config.yaml` — refusing to start if it succeeds (D7). `run`: one episode, print `episode_id` and outcome, write the parquet bundle. `replay`: mode (i) offline by default — re-derive the message array from the trace, rehash the stored `root_request_ref` blob, assert `== root_view_hash`, render the transcript; `--online` adds mode (ii) (re-POST `/apply-template`, byte-compare, assert the chat-template sha matches `config_snapshot`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 pytest tests/test_cli.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add rlm/cli.py tests/test_cli.py
git commit -m "feat: CLI — validate, run, replay (offline + online hash verification)"
```

---

## Task 17: S1 fixtures and the gate

**Files:**
- Create: `s1/make_fixtures.py`, `s1/tasks/*.json`, `s1/run_s1.py`, `s1/RESULTS.md`
- Test: `tests/test_s1_fixtures.py`

**Interfaces:**
- Consumes: `rlm.cli`, `rlm.episode`.
- Produces: the S1 gate verdict.

Spec §9 S1, operationalized. Fixtures are **non-benchmark** and synthetic (random entity–UUID pairings that cannot exist in any training corpus), generated deterministically from a fixed seed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_s1_fixtures.py
import json
from pathlib import Path

S1 = Path(__file__).resolve().parents[1] / "s1"


def test_needle_fixture_is_at_least_64k_leaf_tokens():
    meta = json.loads((S1 / "tasks" / "needle.json").read_text())
    assert meta["tokenized_len"] >= 64_000  # asserted programmatically, spec §9
    assert len(meta["context"]) >= 250_000 or Path(meta["context_path"]).exists()


def test_control_truncation_rule_is_deterministic_and_drops_the_needle():
    from s1.make_fixtures import control_truncate
    meta = json.loads((S1 / "tasks" / "needle.json").read_text())
    text = Path(meta["context_path"]).read_text(encoding="utf-8")
    a = control_truncate(text, 28_000)
    assert a == control_truncate(text, 28_000)
    assert meta["answer"] not in a  # needle beyond any retained region


def test_paraphrase_needle_defeats_regex():
    meta = json.loads((S1 / "tasks" / "paraphrase.json").read_text())
    text = Path(meta["context_path"]).read_text(encoding="utf-8")
    assert meta["answer"] not in text  # must require a leaf call, not a scan


def test_fixtures_are_reproducible_from_seed():
    from s1.make_fixtures import build
    assert build(seed=1)["needle"]["answer"] == build(seed=1)["needle"]["answer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 pytest tests/test_s1_fixtures.py -v`
Expected: FAIL — `s1/make_fixtures.py` missing.

- [ ] **Step 3: Write minimal implementation**

`make_fixtures.py` builds two tasks from seed 1: **needle** (≥64K leaf tokens of filler with one `entity → UUID` pairing placed in the final third, verified past the control-truncation point) and **paraphrase-needle** (the fact stated only in paraphrase, so a regex for the answer string cannot find it — forcing ≥1 leaf call). `control_truncate(text, n_tokens)` is the stated deterministic rule for arm (a): keep the first N tokens, drop the rest.

`run_s1.py` runs the gate: arm (a) control — root alone on the truncated document, 3 attempts, must score **0/3**; arm (b) RLM — 3 attempts, must score **≥2/3**; then the paraphrase task; then the **R1 prompt A/B** — `root.v1` (tips-only) vs `root.v2` (tips + exemplars), 3 attempts each on the S1 fixtures only, winner pinned into `config.yaml` and hashed into `config_snapshot`. Finally, re-derive the leaf `max_predict` default from the observed answer-length distribution in the traces and record it.

- [ ] **Step 4: Run the gate**

```bash
uv run --python 3.12 s1/run_s1.py --config config.yaml
```
Expected: `S1 GATE: PASS` with the arm scores, the A/B winner, and the re-derived `max_predict`. **A negative result is a finding, not a failure to hide** (§9 S1, R1) — if the root flails in both arms, record it in `s1/RESULTS.md` and stop for a decision rather than tuning prompts ad hoc.

- [ ] **Step 5: Commit**

```bash
git add s1/ tests/test_s1_fixtures.py config.yaml
git commit -m "feat: S1 fixtures, gate runner, and verdict (spec §9 S1)"
```

---

## Self-Review

**1. Spec coverage.** §5's six components → Tasks 2 (C3), 3+15 (C2), 6 (C6), 7–10 (C1), 11 (C4), 13 (C5). Section-level contracts → bridge (8), prompt registry (4+14), dry-run mode (11+15), config schema (4), CLI (16), lifecycle log (5), dependency rule (1), testing incl. both mandatory property suites (2, 13). §6 schema and the state rule → 6 + 12 + 16. §9 S1 → 17. **Known deliberate omissions, all later slices:** `rlm bench`/`rlm export` verbs (S4; `export_bundle` exists), the S2 fixtures (fan-out, re-query, question-batching, leaf-envelope), the S3 adversarial self-tests and hard-kill durability test (Task 6 covers the storage half), `max_depth > 1` support, and the §8 benchmark itself.

**2. Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". Where a task says "paste from Recipes §x", that names a complete, verbatim, on-disk companion document generated from executed probe code — not a promise to figure it out later.

**3. Type consistency.** `CellOutput(stdout, stderr, repr_, traceback)` is constructed in Task 2 and consumed identically in Tasks 9, 10, 15. `observation_view(out, cap)` keeps one signature throughout. `llm_query(prompt, role="leaf")` matches across the prompt files (14), the child stub (9), and the dispatcher (11). `StepStatus.REJECTED` is used for both dispatcher pre-flight rejects (11) and no-cell turns (12) — one enum, two documented reasons distinguished by `error_detail`. `TraceLogger.put_step(step, blobs)` has the same shape in Tasks 6 and 15.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-capa1-scaffold.md`, with verified recipes in `docs/superpowers/plans/2026-08-13-capa1-probe-recipes.md`.
