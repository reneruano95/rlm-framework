"""rlm-halo: a local Recursive Language Model runtime.

The scaffold disposes; the LLM proposes (spec §2, invariant I1).

    import rlm

    cfg    = rlm.load_config(rlm.default_config_path())
    task   = rlm.Task(task_id="q1", text="...", context_path=..., checker="exact",
                      answer="42", category="needle")
    result = await rlm.run_episode(task, cfg, dispatcher=..., trace=..., lifecycle=...)

WHY EVERY NAME IS RESOLVED LAZILY, and it is not a style choice
---------------------------------------------------------------
`sandbox/manager.py` stages FOUR files into the AppContainer -- `sandbox/child.py`,
this file, `errors.py` and `bridge.py` -- and `sandbox/child.py:151` then runs
`from rlm.bridge import BridgeEndpoint, encode_frame` inside it. That import executes
THIS module, in a tree where only three modules exist.

So a facade with module-scope imports raises `ModuleNotFoundError` at child startup on
every episode. Staging the closure instead is worse, not better: `episode.py` pulls
`serve/dispatcher.py`, which pulls `httpx` -- an HTTP client inside the one process the
whole architecture exists to keep clients out of, and `sandbox/child.py` is on the
ISOLATED list in `tests/test_import_rules.py` precisely to forbid that.

PEP 562 resolves it. Measured 2026-09-01 in a three-module tree identical to what the
AppContainer sees: `from rlm.bridge import ...` works, bare `import rlm` loads ONE
module with `httpx` absent, and a lazy name raises only when someone touches it --
which the sandbox never does. `rlm.bridge` is a submodule import, so `__getattr__` does
not even fire for it. Against the real package: one module bare, twenty-two after
resolving everything, which is correct for a consumer and never paid by the child.

WHAT IS AND IS NOT HERE
-----------------------
This list is the surface, and `_tests/test_public_api.py` pins it to a literal, so it
cannot grow by accident. Measured before it was written: the two real consumers outside
the package -- `bench/` and `gate/` -- import exactly five names between them (`Task`,
`CHECKERS`, `check`, `near_miss_suite`, `ChunkIndex`). Everything else below is what
`cli.py` needs to assemble one episode, which is what the library is FOR.

`tests/` imports 126 names, 125 of them tests-only. Those are internals under test, not
API, and they stay reachable by their full path. Deep imports are not forbidden; they
are simply not the supported surface, and nothing here pretends otherwise.
"""
from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # the loop
    "Task", "EpisodeResult", "run_episode",
    # configuration, including the config the package ships with
    "Config", "ConfigError", "load_config", "default_config_path", "config_snapshot",
    "PromptRegistry", "resolve_prompt_path",
    # talking to a model: the real one, and the double the shipped suite runs on
    "LLMDispatcher", "MockDispatcher",
    # recording what happened
    "TraceLogger", "Lifecycle",
    # the isolate the model's code runs in
    "SandboxManager",
    # serving a local llama.cpp
    "LlamaServerProcess",
    # reading a result, and the failures worth catching by name
    "Outcome", "StepStatus", "RlmError", "BudgetBreach", "DispatchError",
    "SandboxError", "TransportError",
    # what bench/ and gate/ consume today
    "CHECKERS", "check", "near_miss_suite", "ChunkIndex",
]

_EXPORTS = {
    "Task": ("rlm.episode", "Task"),
    "EpisodeResult": ("rlm.episode", "EpisodeResult"),
    "run_episode": ("rlm.episode", "run_episode"),
    "Config": ("rlm.config", "Config"),
    "ConfigError": ("rlm.errors", "ConfigError"),
    "load_config": ("rlm.config", "load_config"),
    "default_config_path": ("rlm.config", "default_config_path"),
    "config_snapshot": ("rlm.config", "config_snapshot"),
    "PromptRegistry": ("rlm.config", "PromptRegistry"),
    "resolve_prompt_path": ("rlm.config", "resolve_prompt_path"),
    "LLMDispatcher": ("rlm.serve.dispatcher", "LLMDispatcher"),
    "MockDispatcher": ("rlm.serve.dispatcher", "MockDispatcher"),
    "TraceLogger": ("rlm.trace", "TraceLogger"),
    "Lifecycle": ("rlm.trace.lifecycle", "Lifecycle"),
    "SandboxManager": ("rlm.sandbox.manager", "SandboxManager"),
    "LlamaServerProcess": ("rlm.serve.serverproc", "LlamaServerProcess"),
    "Outcome": ("rlm.errors", "Outcome"),
    "StepStatus": ("rlm.errors", "StepStatus"),
    "RlmError": ("rlm.errors", "RlmError"),
    "BudgetBreach": ("rlm.errors", "BudgetBreach"),
    "DispatchError": ("rlm.errors", "DispatchError"),
    "SandboxError": ("rlm.errors", "SandboxError"),
    "TransportError": ("rlm.errors", "TransportError"),
    "CHECKERS": ("rlm.measure.checkers", "CHECKERS"),
    "check": ("rlm.measure.checkers", "check"),
    "near_miss_suite": ("rlm.measure.checkers", "near_miss_suite"),
    "ChunkIndex": ("rlm.serve.leakcheck", "ChunkIndex"),
}

if TYPE_CHECKING:  # so a type checker and an editor see the real symbols
    from rlm.config import (  # noqa: F401
        Config, PromptRegistry, config_snapshot, default_config_path, load_config,
        resolve_prompt_path,
    )
    from rlm.episode import EpisodeResult, Task, run_episode  # noqa: F401
    from rlm.errors import (  # noqa: F401
        BudgetBreach, ConfigError, DispatchError, Outcome, RlmError, SandboxError,
        StepStatus, TransportError,
    )
    from rlm.measure.checkers import CHECKERS, check, near_miss_suite  # noqa: F401
    from rlm.sandbox.manager import SandboxManager  # noqa: F401
    from rlm.serve.dispatcher import LLMDispatcher, MockDispatcher  # noqa: F401
    from rlm.serve.leakcheck import ChunkIndex  # noqa: F401
    from rlm.serve.serverproc import LlamaServerProcess  # noqa: F401
    from rlm.trace import TraceLogger  # noqa: F401
    from rlm.trace.lifecycle import Lifecycle  # noqa: F401


def __getattr__(name: str):
    try:
        module, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'rlm' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module), attr)
    globals()[name] = value  # resolve once; later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
