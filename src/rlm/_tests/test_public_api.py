"""The public surface is a literal list, and these tests are what make it one.

Two properties matter and neither is enforced by convention:

1. The surface cannot grow by accident. `__all__` is compared against a literal
   written out below, so adding an export means editing this file -- a deliberate
   act with a reviewer, not a side effect of adding a name somewhere else.

2. `import rlm` must stay cheap and client-free. `sandbox/manager.py` stages this
   package's `__init__.py` into the AppContainer and `sandbox/child.py` imports
   `rlm.bridge` through it, so a facade that pulls the world at module scope breaks
   every episode on the box -- in a place the copy test cannot see, because the
   sandbox does not run off Windows.
"""
import subprocess
import sys

import rlm

EXPECTED = [
    "BudgetBreach",
    "CHECKERS",
    "ChunkIndex",
    "Config",
    "ConfigError",
    "DispatchError",
    "Dispatcher",
    "EpisodeResult",
    "LLMDispatcher",
    "Lifecycle",
    "LlamaServerProcess",
    "MockDispatcher",
    "Outcome",
    "PromptRegistry",
    "RlmError",
    "SandboxError",
    "SandboxManager",
    "StepStatus",
    "Task",
    "TraceLogger",
    "TransportError",
    "__version__",
    "check",
    "config_snapshot",
    "default_config_path",
    "load_config",
    "near_miss_suite",
    "resolve_prompt_path",
    "run_episode",
]


def test_the_public_api_is_exactly_this_list():
    """Changing the surface means editing this literal. That is the point."""
    assert sorted(rlm.__all__) == EXPECTED


def test_every_exported_name_resolves():
    """An __all__ entry that raises on access is a lie in the documentation."""
    for name in rlm.__all__:
        getattr(rlm, name)


def test_no_exported_name_is_private():
    assert [n for n in rlm.__all__ if n.startswith("_") and n != "__version__"] == []


def test_dir_matches_all():
    """`dir(rlm)` is what an editor and `help()` show; it must not disagree."""
    assert dir(rlm) == sorted(rlm.__all__)


def test_a_bare_import_pulls_no_http_client_and_almost_no_package():
    """The property the AppContainer depends on, checked in a FRESH interpreter.

    It has to be a subprocess: by the time this test runs, the suite has already
    imported half the package, so measuring sys.modules in-process would pass no
    matter what __init__ did.
    """
    code = (
        "import sys, rlm; "
        "print(len([m for m in sys.modules if m.startswith('rlm')]), 'httpx' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    n_modules, httpx_loaded = int(out[0]), out[1] == "True"
    assert not httpx_loaded, "bare `import rlm` loaded an HTTP client"
    assert n_modules == 1, f"bare `import rlm` loaded {n_modules} rlm modules, want 1"


def test_an_unknown_name_raises_attribute_error_not_import_error():
    """A typo must look like a typo, not like a broken install."""
    try:
        rlm.NoSuchThing
    except AttributeError as exc:
        assert "NoSuchThing" in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_resolution_is_cached_after_first_access():
    """__getattr__ writes the resolved object into globals(), so the second lookup
    does not re-enter it. Cheap to assert, and it is the reason lazy costs nothing."""
    _ = rlm.Task
    assert "Task" in vars(rlm)


def test_the_export_table_and_all_cannot_drift_apart():
    """`__all__` and `_EXPORTS` are two lists of the same thing; nothing kept them
    in step. A name in `__all__` with no `_EXPORTS` entry raises AttributeError on
    access -- caught by the resolve test above, but only after the fact. A name in
    `_EXPORTS` and not `__all__` is an export nobody can discover."""
    assert set(rlm._EXPORTS) == set(rlm.__all__) - {"__version__"}


def test_the_typing_block_names_exactly_the_exports():
    """The TYPE_CHECKING block is a third copy of the same list, written for editors
    and type checkers. It is invisible at runtime, so nothing else can catch it going
    stale: an export missing there is a name your editor cannot resolve, and a name
    there that is no longer exported is a lie a reader will believe."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(rlm.__file__).read_text(encoding="utf-8"))
    named = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "TYPE_CHECKING":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.ImportFrom):
                    named |= {a.asname or a.name for a in stmt.names}
    assert named == set(rlm._EXPORTS), (
        "the TYPE_CHECKING block and _EXPORTS disagree: "
        f"only in typing {sorted(named - set(rlm._EXPORTS))}, "
        f"only in _EXPORTS {sorted(set(rlm._EXPORTS) - named)}"
    )


def test_the_shipped_config_is_reachable_without_a_repo(tmp_path, monkeypatch):
    """The package shipped a validated config since 3f71147 that no production path
    could reach: `load_config` took a required Path and `--config` defaulted to the
    CWD string "config.yaml". A consumer who copied the package got a config file they
    had to know the name of to find.

    Both directions are pinned here because the ORDER is the safety property.
    """
    from rlm.config import default_config_path, resolve_config_path

    # No config.yaml in sight -> the package's own, which is a copied consumer's case.
    monkeypatch.chdir(tmp_path)
    assert resolve_config_path() == default_config_path()

    # A config.yaml in the CWD wins, always. `cli.py:1279-1281` states the reason a
    # scoring input is an artifact and not a flag: "a scoring run an operator can point
    # at a different manifest is a scoring run they can point at a friendlier one." A
    # fallback that could prefer the shipped default over a real config present on the
    # same machine would be that hazard wearing a convenience's clothes.
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    assert resolve_config_path() == __import__("pathlib").Path("config.yaml")

    # An explicit path always wins over both.
    assert resolve_config_path(tmp_path / "other.yaml") == tmp_path / "other.yaml"
