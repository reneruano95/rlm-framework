"""The dependency rule (spec §5): C1-C3, C5, C6 must not import C4 or any LLM client."""
import ast
from pathlib import Path

import pytest

RLM = Path(__file__).resolve().parents[1] / "src" / "rlm"

# modules that must never reach a model server, directly or transitively
ISOLATED = [
    "context/truncate.py",
    "context/chunker.py",
    "context/loader.py",
    "budget.py",
    "trace/store.py",
    # R13's foreign-string detector: C4 calls it on every leaf answer, but it
    # must stay importable by the trace/analysis side (which audits recorded
    # answers offline) without dragging in an HTTP client.
    "serve/leakcheck.py",
    # The leaf JSON envelope's parser/validator: C4 calls it on every leaf
    # answer, and the trace/analysis side re-derives envelope verdicts offline
    # from stored answers. Neither needs an HTTP client.
    "serve/envelope.py",
    # §8's benchmark checkers. Isolation is load-bearing rather than tidy: S4
    # scoring and `rlm replay` must be able to re-derive every pass/fail from
    # the trace alone, with no server reachable, so a checker that could reach
    # an LLM would make the benchmark unscoreable offline -- and would open the
    # door to a model-graded checker, which §8 forbids by requiring answers be
    # "programmatically verifiable".
    "measure/checkers.py",
    "measure/stats.py",
    # §8's scoring/inference/report layer. Isolation is the same load-bearing
    # property `checkers.py` and `stats.py` have, one level up: the S4 VERDICT
    # must be recomputable from the closed trace store alone, with no server
    # reachable. A scoring module that could reach a model is a scoring module
    # that could be re-run until it agreed, which is the p-hacking §8's
    # pre-registration exists to prevent. It reads duckdb + `rlm.measure.stats` and
    # takes the manifest as an injected object (`bench/` is not in the wheel).
    "measure/verdict.py",
    # §8's baseline arms. They make model calls, but through an INJECTED
    # dispatcher: `src/rlm/episode.py` is the only module permitted to import both
    # C4 and the isolated components, and widening that to a second module is
    # the drift its docstring forbids. Keeping arms on this side is also what
    # lets the scoring/replay path import an arm's helpers (the truncation
    # policy, the outcome mapping) with no server reachable.
    "measure/arms.py",
    # §8's benchmark scheduler. It has the SHAPE of a composition root -- it
    # sequences four arms across two server profiles -- but the exemption list
    # below is spec-frozen at two modules (episode.py, cli.py) and widening it
    # is the drift `src/rlm/episode.py`'s docstring forbids. So the scheduler stays
    # on this side and every model-facing thing it needs (the four arms,
    # `Task.from_file`, quiesce, the /props handshake, the leaf relaunch)
    # arrives as an injected callable on `BenchCtx`. The same rule that keeps
    # `arms.py` importable offline keeps the scheduler replayable: a scoring
    # pass can import the block/ledger machinery with no server reachable.
    "measure/bench.py",
    # The Windows RAPL power sampler + one-shot pkg-temp reader (S0 item 8).
    # It shells out to `powershell`/`Get-Counter`, never to a model server, so
    # it stays importable by the trace/analysis side without an HTTP client.
    "power.py",
    # D27's launch-log reader (extracted from cli.py, 2026-08-22). Pure text
    # parsing of a server's own stderr: a file read, three regexes, dict
    # arithmetic. It answers §4's cache-type assertion WITHOUT contacting the
    # server -- which is the whole point, since /props cannot report cache
    # types -- so the lint is what keeps it that way.
    "serve/launchlog.py",
    # §8's wall-clock projection model + the --smoke calibration table
    # (extracted from cli.py, 2026-08-22). Pure arithmetic over manifest entries
    # and ledger records. It must never learn to ask a server what something
    # cost -- a projection that can measure is no longer a projection, and §8
    # prints the two side by side precisely to keep them apart.
    "measure/projection.py",
    # §8:343's escalation PLAN (extracted from cli.py, 2026-08-22). Choosing and
    # serialising the cells to re-measure. A planner that could ASK a server
    # which cells to re-run would be choosing its own replicates -- the exact
    # thing §8's pre-registration prevents -- so this lint is load-bearing.
    # Executing the plan (run_escalation) stays in cli.py, which is why it is
    # not here.
    "measure/escalation.py",
    # The root turn's pure text shaping, split out of rootclient.py 2026-08-22.
    # It lived next to `RootConversation`, which holds a ServerClient -- which is
    # why `rlm.serve.rootclient` is in FORBIDDEN_RLM and why the REPLAY path could not
    # be lint-covered while `history_message` and `extract_cell` sat behind an
    # HTTP client. Text has no business importing transport; the lint says so now.
    "serve/roottext.py",
    # I4 in code: rebuild an episode from the trace store ALONE (extracted from
    # cli.py, 2026-08-22). This is the entry that most needs the rule. A
    # re-derivation able to ask a server what the prompt was would be checking
    # the server against itself, and S3 -- 12/12 with the lifecycle log deleted --
    # is the gate that claim rests on. `cmd_replay`/`_verify_online` keep the
    # --online path in cli.py, which is why they are not here.
    "trace/replay.py",
    # `rlm export`'s bundle builder (extracted from cli.py, 2026-08-22). Same
    # family as replay.py: the parquet+blob bundle is what a FOREIGN reader uses
    # to check a published number, so it must depend on the trace store and
    # nothing else. An export that needed a running server would be one nobody
    # else could reproduce.
    "trace/export.py",
    "sandbox/manager.py",
    "sandbox/child.py",
    "sandbox/winjob.py",
    "sandbox/winproc.py",
    "bridge.py",
    # The process manager the episode runner drives for a §5 C4 rotation. It
    # owns a PROCESS, never a connection: readiness arrives as an injected
    # health probe, so C4 stays the only module that speaks to a server.
    "serve/serverproc.py",
]
FORBIDDEN_ROOTS = {"httpx", "requests", "urllib", "aiohttp", "socket", "http"}
FORBIDDEN_RLM = {"rlm.serve.dispatcher", "rlm.serve.rootclient"}


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
