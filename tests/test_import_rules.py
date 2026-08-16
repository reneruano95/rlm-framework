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
    # R13's foreign-string detector: C4 calls it on every leaf answer, but it
    # must stay importable by the trace/analysis side (which audits recorded
    # answers offline) without dragging in an HTTP client.
    "leakcheck.py",
    # The leaf JSON envelope's parser/validator: C4 calls it on every leaf
    # answer, and the trace/analysis side re-derives envelope verdicts offline
    # from stored answers. Neither needs an HTTP client.
    "envelope.py",
    # §8's benchmark checkers. Isolation is load-bearing rather than tidy: S4
    # scoring and `rlm replay` must be able to re-derive every pass/fail from
    # the trace alone, with no server reachable, so a checker that could reach
    # an LLM would make the benchmark unscoreable offline -- and would open the
    # door to a model-graded checker, which §8 forbids by requiring answers be
    # "programmatically verifiable".
    "checkers.py",
    "stats.py",
    # §8's baseline arms. They make model calls, but through an INJECTED
    # dispatcher: `rlm/episode.py` is the only module permitted to import both
    # C4 and the isolated components, and widening that to a second module is
    # the drift its docstring forbids. Keeping arms on this side is also what
    # lets the scoring/replay path import an arm's helpers (the truncation
    # policy, the outcome mapping) with no server reachable.
    "arms.py",
    # The Windows RAPL power sampler + one-shot pkg-temp reader (S0 item 8).
    # It shells out to `powershell`/`Get-Counter`, never to a model server, so
    # it stays importable by the trace/analysis side without an HTTP client.
    "power.py",
    "sandbox/manager.py",
    "sandbox/child.py",
    "sandbox/winjob.py",
    "sandbox/winproc.py",
    "bridge.py",
    # The process manager the episode runner drives for a §5 C4 rotation. It
    # owns a PROCESS, never a connection: readiness arrives as an injected
    # health probe, so C4 stays the only module that speaks to a server.
    "serverproc.py",
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
