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
    "sandbox/manager.py",
    "sandbox/child.py",
    "sandbox/winjob.py",
    "sandbox/winproc.py",
    "bridge.py",
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
