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
