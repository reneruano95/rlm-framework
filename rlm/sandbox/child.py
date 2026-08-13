"""C1 sandbox child: ONE long-lived interpreter per episode (spec §5).

This file never runs inside the scaffold. The manager stages it into the
ACL'd bootstrap directory (D7) together with the only two scaffold modules it
imports -- `rlm.errors` and `rlm.bridge` -- and spawns it there under an
AppContainer with `-B -I -u`. `-I` implies `-P`, so the script directory is NOT
prepended to sys.path: the explicit `sys.path.insert` below is what makes
`import rlm.bridge` resolve to the staged copy rather than to the repo (which
the AppContainer cannot read at all).

BOOTSTRAP ORDER IS LOAD-BEARING. Each numbered step below came from a measured
failure, not from taste:

  D8  Build the asyncio event loop BEFORE `sys.addaudithook`. On Windows
      ProactorEventLoop builds its self-pipe from `socket.socketpair()`, an
      AF_INET loopback pair; a hook that denies AF_INET before the loop exists
      breaks asyncio entirely. The pre-built loop keeps working afterwards, and
      `asyncio.new_event_loop()` still gets denied, so user code cannot make a
      second one.
  D9  End with `os._exit(0)` after closing the write fd and sending a final
      `bye` frame. `asyncio.gather` inside an AppContainer makes the
      interpreter exit 0xC0000008 at teardown WITH ALL RESULTS INTACT; gather
      is the fan-out idiom the prompt registry teaches, so every healthy
      multi-leaf episode would otherwise look like a crash.
  D12 Capture cell output at the `sys.stdout`/`sys.stderr` OBJECT level
      (StringIO), never at fd level: everything stays `str`, which is what
      keeps the ASCII-JSON bridge and the DuckDB text columns safe. fd-level
      capture would make non-UTF-8 bytes possible and change the framing
      contract.
  D24 Re-inject the reserved names before EVERY cell. The reference harness
      restores its reserved names after each execution; without this, model
      code that rebinds `llm_query` intercepts its own sub-call plumbing --
      a direct I1 violation.
  D25 (a) `SandboxPolicyError` is defined at module level and re-homed onto a
      module name chosen for the model's benefit, because its qualname is
      user-visible in every denied observation (a closure-local class shows up
      as `main.<locals>.SandboxPolicyError`). (b) `sys.unraisablehook` and
      `warnings.showwarning` are redirected per cell into THAT cell's stderr
      buffer, dropping anything whose traceback is purely scaffold- or
      stdlib-internal. (c) `asyncio.new_event_loop`/`run`/`set_event_loop` are
      shadowed with stubs that raise BEFORE any loop object is constructed --
      not optional: a denied `asyncio.new_event_loop()` in turn N otherwise
      leaves a half-built ProactorEventLoop whose `__del__` fires during turn
      N+1 and injects absolute interpreter paths plus a misattributed
      AttributeError into the NEXT cell's observation.

Protocol. Everything rides the Task 8 bridge envelope, so there is exactly one
framing implementation on the wire (`rlm.bridge`); the `kind` of a bridge
request is this module's message kind.

    child -> parent   handshake {"pid": int}      (the FIRST frame; D23)
                      llm_query {"prompt","role",...} -> reply is the answer
                      final_answer {"value": ...}
                      bye {}                       (last frame before _exit)
    parent -> child   setvar {"name","value"}
                      exec {"cell": str}
                          -> reply {"stdout","stderr","repr","traceback",
                                    "duration_ms"}
                      shutdown null
"""
from __future__ import annotations

import ast
import builtins
import io
import linecache
import os
import sys
import time
import traceback
import warnings

_SELF = os.path.normcase(os.path.abspath(__file__))
_HERE = os.path.dirname(_SELF)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --------------------------------------------------------------------------- #
# 1. steal fds 0/1 as the private protocol channel, then blind real stdio
#
# Model code that writes to fd 1 (a native extension, a stray `os.write(1, ...)`)
# can no longer corrupt the protocol: fd 1 is NUL from here on and the bridge
# owns two duplicates nobody else knows about.
# --------------------------------------------------------------------------- #
_PROTO_R = os.dup(0)
_PROTO_W = os.dup(1)
_NUL = os.open(os.devnull, os.O_RDWR)
os.dup2(_NUL, 0)
os.dup2(_NUL, 1)
for _fd in (_PROTO_R, _PROTO_W):
    try:
        os.set_inheritable(_fd, False)
    except OSError:
        pass

# --------------------------------------------------------------------------- #
# 2. the persistent event loop -- MUST precede the audit hook (D8)
# --------------------------------------------------------------------------- #
import asyncio  # noqa: E402  (import order is the point of this module)

LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)

# --------------------------------------------------------------------------- #
# 3. the bridge (threads start here, before the hook denies anything)
# --------------------------------------------------------------------------- #
from rlm.bridge import BridgeEndpoint, encode_frame  # noqa: E402

BRIDGE = BridgeEndpoint(_PROTO_R, _PROTO_W, loop=LOOP, tag="sandbox")


# --------------------------------------------------------------------------- #
# 4. policy error (D25a) -- module level, and re-homed so the model never sees
#    scaffold structure in the one exception it is most likely to hit
# --------------------------------------------------------------------------- #
class SandboxPolicyError(RuntimeError):
    """An operation the episode's isolation policy refuses.

    Legible on purpose: the root is expected to re-plan from the observation
    rather than burn turns retrying a denied call.
    """


SandboxPolicyError.__module__ = "rlm_sandbox"


# --------------------------------------------------------------------------- #
# 5. per-cell capture at the OBJECT level (D12)
# --------------------------------------------------------------------------- #
class _CaptureStream(io.TextIOBase):
    """A str-only sink. Never an fd: see D12 in the module docstring."""

    def __init__(self) -> None:
        self._buf = io.StringIO()

    def write(self, s: str) -> int:
        return self._buf.write(s)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def take(self) -> str:
        value = self._buf.getvalue()
        self._buf = io.StringIO()
        return value


_OUT = _CaptureStream()
_ERR = _CaptureStream()

# Absolute paths that must never reach the model's observation. `_HERE` covers
# child.py AND the staged rlm.bridge copy; the interpreter prefixes cover the
# stdlib. Frames are judged separately for the two uses -- see
# `_is_scaffold_frame` (tracebacks) vs `_is_internal_frame` (unraisable/warning).
_STDLIB = os.path.normcase(os.path.dirname(os.__file__))
_INTERNAL_ROOTS = tuple(
    p + os.sep for p in {
        _HERE,
        _STDLIB,
        os.path.normcase(os.path.abspath(sys.base_prefix)),
        os.path.normcase(os.path.abspath(sys.prefix)),
    }
)


def _real_path(filename: str | None) -> str | None:
    """None for pseudo-filenames like `<cell:3>` or `<string>`: those are the
    model's own code and must never be scrubbed."""
    if not filename or filename.startswith("<"):
        return None
    try:
        return os.path.normcase(os.path.abspath(filename))
    except (OSError, ValueError):
        return None


def _is_scaffold_frame(filename: str | None) -> bool:
    path = _real_path(filename)
    return path is not None and path.startswith(_HERE + os.sep)


def _is_internal_frame(filename: str | None) -> bool:
    path = _real_path(filename)
    return path is not None and path.startswith(_INTERNAL_ROOTS)


def _traceback_is_internal_only(tb) -> bool:
    """True when every frame belongs to the scaffold or the stdlib -- i.e. the
    event has no connection to anything the model wrote, so surfacing it would
    misattribute a scaffold detail to the model's cell (Conflict 12)."""
    while tb is not None:
        if not _is_internal_frame(tb.tb_frame.f_code.co_filename):
            return False
        tb = tb.tb_next
    return True  # all frames internal, or no frames at all: nothing attributable


# --------------------------------------------------------------------------- #
# 6. unraisable / warning routing (D25b)
# --------------------------------------------------------------------------- #
def _drop_unraisable(unraisable) -> None:
    """Between cells there is no buffer to attribute anything to, and fd 2 is
    the episode log, not the model's observation."""


def _drop_showwarning(message, category, filename, lineno, file=None, line=None) -> None:
    pass


def _cell_unraisablehook(unraisable) -> None:
    tb = unraisable.exc_traceback
    if _traceback_is_internal_only(tb):
        return
    header = unraisable.err_msg or "Exception ignored in"
    text = "".join(traceback.format_exception(
        unraisable.exc_type, unraisable.exc_value, tb))
    _ERR.write(f"{header} {unraisable.object!r}\n{text}")


def _cell_showwarning(message, category, filename, lineno, file=None, line=None) -> None:
    if _is_internal_frame(filename):
        return
    _ERR.write(warnings.formatwarning(message, category, filename, lineno, line))


# --------------------------------------------------------------------------- #
# 7. the audit hook -- defence in depth, NOT the boundary
#
# Measured: a hook covering only socket.* events is DEFEATED by
# ctypes.WinDLL('ws2_32') (live TCP to 1.1.1.1:80). Under `network_isolation:
# appcontainer` the kernel returns WSAEACCES on that path anyway, which is why
# `deny_ctypes` may default to false there; under `audit_only` the ctypes deny
# set is the only thing between model code and 127.0.0.1:8081, and denying
# ctypes.dlopen makes a fresh `import ctypes` fail outright -- the clean policy.
# --------------------------------------------------------------------------- #
_FAMILY_DENY = frozenset({2, 23})  # AF_INET, AF_INET6

_DENY_EVENTS = frozenset({
    "socket.bind", "socket.connect", "socket.connect_ex", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto",
    "socket.sendmsg", "socket.sethostname",
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "pty.spawn",
    "winreg.CreateKey", "winreg.DeleteKey", "winreg.SetValue",
})

_DENY_CTYPES_EVENTS = frozenset({
    "ctypes.dlopen", "ctypes.dlsym", "ctypes.dlsym/handle",
    "ctypes.call_function", "ctypes.set_exception", "ctypes.cdata",
    "ctypes.cdata/buffer",
})

_NETWORK_MSG = (
    "network disabled for this episode: AF_INET/AF_INET6 sockets are not "
    "permitted (C1 no-egress default). Use llm_query(...) for model calls."
)


def _install_audit_hook(deny_ctypes: bool) -> None:
    deny = _DENY_EVENTS | (_DENY_CTYPES_EVENTS if deny_ctypes else frozenset())

    def _hook(event, args):
        if event == "socket.__new__":
            if args[1] in _FAMILY_DENY:
                raise SandboxPolicyError(_NETWORK_MSG)
        elif event in deny:
            raise SandboxPolicyError(f"denied by sandbox policy: {event}")

    sys.addaudithook(_hook)


# --------------------------------------------------------------------------- #
# 8. asyncio stubs (D25c) -- raise BEFORE a loop object is constructed
# --------------------------------------------------------------------------- #
_LOOP_MSG = (
    "asyncio.{name}() is not available in this episode: the scaffold already "
    "runs one event loop for you. Use `await` directly at cell top level."
)


def _make_loop_stub(name: str):
    def _stub(*args, **kwargs):
        raise SandboxPolicyError(_LOOP_MSG.format(name=name))

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__module__ = "rlm_sandbox"
    return _stub


def _install_asyncio_stubs() -> None:
    # Only the `asyncio` package attributes are shadowed; asyncio's own
    # internals call `asyncio.events.set_event_loop`, so the running loop is
    # untouched. `asyncio.get_event_loop()` still returns the running loop.
    asyncio.new_event_loop = _make_loop_stub("new_event_loop")
    asyncio.run = _make_loop_stub("run")
    asyncio.set_event_loop = _make_loop_stub("set_event_loop")


# --------------------------------------------------------------------------- #
# 9. the names the scaffold injects (I1: this is the entire crossing surface)
# --------------------------------------------------------------------------- #
_FINAL_TASKS: list = []


async def llm_query(prompt, role="leaf", **kw):
    """The only way out of the sandbox. C4's semaphore, /tokenize pre-flight,
    retries, timeouts, budget admission and step logging all live on the OTHER
    side of this pipe, where model code cannot reach them."""
    return await BRIDGE.request("llm_query", {"prompt": prompt, "role": role, **kw})


def final_answer(value):
    """Deliberately synchronous: an `async def` here would make the bare call
    `final_answer(x)` -- the shape every prompt teaches -- a silent no-op plus
    a 'coroutine was never awaited' warning. Delivery is still guaranteed:
    `_run_cell` drains the send before it returns the observation."""
    _FINAL_TASKS.append(LOOP.create_task(
        BRIDGE.request("final_answer", {"value": value})))
    return None


class AnswerGuard(dict):
    """The reference harness trains models to submit by assigning into an
    `answer` dict. Without this guard that reflex is a silent no-op; with it,
    it becomes an observable correction the root can act on."""

    __slots__ = ()
    _MSG = "call final_answer(value) to submit"

    def __setitem__(self, key, value):
        raise SandboxPolicyError(self._MSG)

    def update(self, *args, **kwargs):
        raise SandboxPolicyError(self._MSG)

    def setdefault(self, *args, **kwargs):
        raise SandboxPolicyError(self._MSG)


AnswerGuard.__module__ = "rlm_sandbox"

# D24: the authoritative values. USER_NS is re-primed from this dict before
# every cell, so a rebind inside a cell survives only until that cell ends.
_RESERVED: dict = {
    "context": "",
    "chunks": [],
    "llm_query": llm_query,
    "final_answer": final_answer,
    "answer": AnswerGuard(),
}
# Only these may be replaced by a parent `setvar`; the plumbing names cannot.
_SETTABLE_RESERVED = frozenset({"context", "chunks"})

USER_NS: dict = {"__name__": "__sandbox__", "__builtins__": builtins}
USER_NS.update(_RESERVED)


# --------------------------------------------------------------------------- #
# 10. cell execution
# --------------------------------------------------------------------------- #
_FLAGS = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
_CELL_SEQ = 0


def _compile_cell(src: str, name: str):
    """Body as `exec`, a trailing expression as `eval`, BOTH with
    PyCF_ALLOW_TOP_LEVEL_AWAIT. Registering the source in linecache is what
    makes the traceback show the model its own line rather than a blank."""
    linecache.cache[name] = (len(src), None, src.splitlines(keepends=True), name)
    tree = ast.parse(src, filename=name, mode="exec")
    tail = tree.body.pop() if (tree.body and isinstance(tree.body[-1], ast.Expr)) else None
    exec_code = compile(ast.Module(body=tree.body, type_ignores=[]), name, "exec",
                        flags=_FLAGS, dont_inherit=True)
    eval_code = None
    if tail is not None:
        eval_code = compile(ast.Expression(body=tail.value), name, "eval",
                            flags=_FLAGS, dont_inherit=True)
    return exec_code, eval_code


def _format_exception(exc: BaseException) -> str:
    """Scrub scaffold frames, following `__cause__`/`__context__`. Stdlib
    frames are KEPT: they carry real diagnostic value for model code, and
    dropping them would misattribute the error to the cell's own line."""
    te = traceback.TracebackException.from_exception(exc, lookup_lines=True)
    seen: set[int] = set()

    def scrub(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        node.stack = traceback.StackSummary.from_list(
            [f for f in node.stack if not _is_scaffold_frame(f.filename)])
        scrub(node.__cause__)
        scrub(node.__context__)

    scrub(te)
    return "".join(te.format()).rstrip()


async def _run_cell(src: str) -> dict:
    global _CELL_SEQ
    _CELL_SEQ += 1
    name = f"<cell:{_CELL_SEQ}>"

    USER_NS.update(_RESERVED)  # D24, before every cell without exception
    _OUT.take()
    _ERR.take()

    prev_unraisable, prev_showwarning = sys.unraisablehook, warnings.showwarning
    sys.unraisablehook = _cell_unraisablehook
    warnings.showwarning = _cell_showwarning

    started = time.perf_counter()
    value_repr = ""
    tb_text = ""
    try:
        try:
            exec_code, eval_code = _compile_cell(src, name)
            # eval(), NOT exec(): eval returns the coroutine when the compiler
            # set CO_COROUTINE for a top-level await.
            result = eval(exec_code, USER_NS)
            if asyncio.iscoroutine(result):
                await result
            if eval_code is not None:
                value = eval(eval_code, USER_NS)
                if asyncio.iscoroutine(value):
                    value = await value
                if value is not None:
                    USER_NS["_"] = value
                    value_repr = repr(value)
        except BaseException as exc:  # noqa: BLE001 -- the REPL survives anything
            tb_text = _format_exception(exc)
        if _FINAL_TASKS:
            pending, _FINAL_TASKS[:] = _FINAL_TASKS[:], []
            await asyncio.gather(*pending, return_exceptions=True)
    finally:
        sys.unraisablehook = prev_unraisable
        warnings.showwarning = prev_showwarning

    return {
        "stdout": _OUT.take(),
        "stderr": _ERR.take(),
        "repr": value_repr,
        "traceback": tb_text,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


# --------------------------------------------------------------------------- #
# 11. request handling / lifecycle
# --------------------------------------------------------------------------- #
_SHUTDOWN = asyncio.Event()
_CELL_LOCK = asyncio.Lock()


async def _handle(kind: str, payload):
    if kind == "exec":
        async with _CELL_LOCK:  # cells are strictly serialized; sub-calls are not
            return await _run_cell((payload or {}).get("cell", ""))
    if kind == "setvar":
        name, value = payload["name"], payload["value"]
        if name in _RESERVED and name not in _SETTABLE_RESERVED:
            raise SandboxPolicyError(f"{name!r} is scaffold plumbing and cannot be set")
        if name in _SETTABLE_RESERVED:
            _RESERVED[name] = value
        USER_NS[name] = value
        return {"name": name}
    if kind == "shutdown":
        _SHUTDOWN.set()
        return None
    raise SandboxPolicyError(f"unknown request kind: {kind!r}")


async def _serve() -> None:
    # D23: the FIRST frame, and the manager refuses the episode unless this pid
    # matches PROCESS_INFORMATION.dwProcessId.
    await BRIDGE.request("handshake", {"pid": os.getpid()})
    await _SHUTDOWN.wait()


def _emit_raw(obj) -> None:
    """Last-resort write straight down the protocol fd, used only when the
    bridge's own machinery may already be gone."""
    try:
        data = memoryview(encode_frame(obj))
        while data:
            data = data[os.write(_PROTO_W, data):]
    except BaseException:  # noqa: BLE001
        pass


def _farewell(code: int) -> None:
    """D9. `asyncio.gather` inside an AppContainer makes interpreter
    finalization exit 0xC0000008 with every result already delivered, so the
    child never finalizes: it says goodbye explicitly and hard-exits. D10: this
    exit code is not an outcome channel -- `outcome_reason` comes from the
    scaffold-side kill reason plus this `bye` frame."""
    if code == 0:
        try:
            LOOP.run_until_complete(asyncio.wait_for(BRIDGE.request("bye", None), 5))
        except BaseException:  # noqa: BLE001
            pass
    try:
        os.close(_PROTO_W)
    except OSError:
        pass
    os._exit(code)


def main() -> None:
    sys.stdout = _OUT
    sys.stderr = _ERR
    sys.displayhook = lambda value: None
    sys.unraisablehook = _drop_unraisable
    warnings.showwarning = _drop_showwarning

    _install_asyncio_stubs()
    _install_audit_hook("--deny-ctypes" in sys.argv[1:])
    BRIDGE.on_request(_handle)

    code = 0
    try:
        LOOP.run_until_complete(_serve())
    except BaseException:  # noqa: BLE001
        # fd 2 is the episode's own stderr log, never the model's observation.
        try:
            print(traceback.format_exc(), file=sys.__stderr__, flush=True)
        except BaseException:  # noqa: BLE001
            pass
        _emit_raw({"t": "req", "id": 0, "kind": "fatal",
                   "p": {"traceback": traceback.format_exc()}})
        code = 3
    _farewell(code)


if __name__ == "__main__":
    main()
