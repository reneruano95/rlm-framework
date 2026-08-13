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
      a direct I1 violation. Re-injection raises the bar against the ordinary
      case (a cell that rebinds the NAME) and nothing more: it re-reads
      `_RESERVED`, so anything that can reach `_RESERVED` defeats it, and
      several things can -- see ENFORCEMENT LAYERING below. Two cheap routes are
      shut anyway, because cheap is worth shutting: `sys.modules['__main__']`
      (a decoy is installed) and the injected callables' `__globals__` (rebuilt
      over a two-name namespace).
  D25 (a) `SandboxPolicyError` is defined at module level and re-homed onto a
      module name chosen for the model's benefit, because its qualname is
      user-visible in every denied observation (a closure-local class shows up
      as `main.<locals>.SandboxPolicyError`). (b) `sys.unraisablehook` and
      `warnings.showwarning` are redirected per cell into THAT cell's stderr
      buffer, dropping anything whose traceback is purely scaffold- or
      stdlib-internal. (c) EVERY loop-construction route is shadowed --
      `asyncio.new_event_loop`/`run`/`set_event_loop`, the same names on
      `asyncio.events`, the loop classes, and the policy's `new_event_loop` --
      with stubs that raise BEFORE any loop object exists. Package-level
      shadowing alone is one layer thin: `asyncio.events.new_event_loop()`
      reaches the constructor directly and the denial then fires inside
      `ProactorEventLoop.__init__` (on its AF_INET self-pipe), leaving exactly
      the half-built loop whose `__del__` fires during turn N+1 and injects
      absolute interpreter paths plus a misattributed AttributeError into the
      NEXT cell's observation.

ENFORCEMENT LAYERING (spec v0.2.3, §5 C1 -- read this before "hardening"
anything here). The kernel boundary is the AppContainer token plus the Job
Object's `ACTIVE_PROCESS` limit of 1. EVERYTHING IN THIS FILE -- the audit hook,
the shadowed loop constructors, the decoy `__main__`, the re-injected reserved
names, the moved protocol fds -- RAISES THE BAR AND IS NOT A BOUNDARY. Do not
write a comment, a docstring or a test name here that claims otherwise.

These routes from cell code to this module's namespace are KNOWN REACHABLE and
are not defects to be re-filed. All three were measured on-box; each yields
`_RESERVED` and therefore a hijacked `llm_query`/`final_answer`:

  * frame walk -- `sys._getframe()` up to the outermost frame, or
    `asyncio.current_task().get_coro()`; needs nothing injected;
  * the loop -- `asyncio.get_running_loop()._exception_handler.__globals__`;
    the loop is reachable by construction, since top-level `await` requires it;
  * bridge-object pivots -- `llm_query.__globals__['BRIDGE']._handler
    .__globals__`, and the same from `final_answer`.

Sanitizing a namespace cannot close this class: whatever performs re-injection
must itself reach `_RESERVED`, so every sanitized namespace still contains the
target one hop further along. The first two routes do not touch the injected
namespace at all.

WHY I1 STILL HOLDS. Not because the sandbox is escape-proof -- because the
things a model could otherwise subvert (truncation caps, budgets, routing,
termination, step logging, whether a `final_answer` is ACCEPTED) execute
scaffold-side, IN A DIFFERENT PROCESS. A hijacked stub answers the model
locally; the scaffold simply receives no frames, and every call that does cross
the pipe is admitted or refused on the scaffold's count.
`test_hijacked_llm_query_cannot_alter_scaffold_side_control` is the assertion of
that guarantee, and it is the one that must never be weakened.

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
import types
import warnings

_SELF = os.path.normcase(os.path.abspath(__file__))
_HERE = os.path.dirname(_SELF)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --------------------------------------------------------------------------- #
# 1. move fds 0/1 to the private protocol channel, then blind real stdio
#
# This RAISES THE BAR, it does not close the hole. It stops the accident (a
# native extension or a stray `os.write(1, ...)` corrupting the protocol) and it
# stops the cheap guess: `os.write(4, ...)` used to land on the protocol when
# the fds were merely `dup()`ed to the next free numbers. Deliberate model code
# can still find fd 101 and desync the stream -- which is why the scaffold
# classifies that outcome (`bridge_desync`, see rlm.trace) instead of pretending
# it cannot happen.
# --------------------------------------------------------------------------- #
_PROTO_R = 100
_PROTO_W = 101
os.dup2(0, _PROTO_R, inheritable=False)
os.dup2(1, _PROTO_W, inheritable=False)
_NUL = os.open(os.devnull, os.O_RDWR)
os.dup2(_NUL, 0)
os.dup2(_NUL, 1)

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

# NO ABSOLUTE HOST PATH MAY EVER REACH AN OBSERVATION -- it goes straight into
# the root's context and into DuckDB. Scaffold frames are DROPPED; stdlib frames
# are KEPT (they are genuinely useful to the model: `json/decoder.py` tells it
# what failed) but their filenames are REWRITTEN onto a virtual root, and any
# other real path collapses to `<host>/<basename>`. The mapping is
# order-sensitive: the stdlib lives under base_prefix, so it must match first.
_STDLIB = os.path.normcase(os.path.dirname(os.__file__))
_VIRTUAL_ROOTS: tuple[tuple[str, str], ...] = (
    (_HERE + os.sep, None),          # scaffold: drop the frame entirely
    (_STDLIB + os.sep, "<stdlib>"),
    (os.path.normcase(os.path.abspath(sys.base_prefix)) + os.sep, "<python>"),
    (os.path.normcase(os.path.abspath(sys.prefix)) + os.sep, "<python>"),
)
_INTERNAL_ROOTS = tuple(root for root, _ in _VIRTUAL_ROOTS)


def _real_path(filename: str | None) -> str | None:
    """None for pseudo-filenames like `<cell:3>` or `<string>`: those are the
    model's own code and must never be scrubbed or rewritten."""
    if not filename or filename.startswith("<"):
        return None
    try:
        return os.path.normcase(os.path.abspath(filename))
    except (OSError, ValueError):
        return None


def _is_internal_frame(filename: str | None) -> bool:
    path = _real_path(filename)
    return path is not None and path.startswith(_INTERNAL_ROOTS)


def _virtual_filename(filename: str | None) -> str | None:
    """The name the MODEL sees. `None` means "drop this frame"."""
    path = _real_path(filename)
    if path is None:
        return filename                       # `<cell:3>`: the model's own code
    for root, label in _VIRTUAL_ROOTS:
        if path.startswith(root):
            if label is None:
                return None                   # scaffold frame
            return label + "/" + os.path.relpath(path, root).replace("\\", "/")
    # Unknown real path (the repo, a user file, anything): never leak it whole.
    return "<host>/" + os.path.basename(path)


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
    """Shadow every route that would CONSTRUCT a loop, at both the package and
    the implementation layer -- `asyncio.new_event_loop` alone is one layer thin,
    because `asyncio.events.new_event_loop()` reaches the constructor directly
    and the denial then fires inside `ProactorEventLoop.__init__`, leaving the
    half-built object D25(c) exists to prevent.

    `get_event_loop` is deliberately NOT shadowed at the `events` layer:
    `asyncio.gather` calls `events.get_event_loop()` on its `_ensure_future`
    path, and shadowing it breaks gather -- the fan-out idiom the prompt
    registry teaches (measured: `gather BROKE: RuntimeError denied`). It is also
    not a construction route here: the bootstrap already installed LOOP as the
    thread's event loop, so it returns that and builds nothing.
    """
    stubs = {name: _make_loop_stub(name)
             for name in ("new_event_loop", "set_event_loop", "run")}
    for name, stub in stubs.items():
        setattr(asyncio, name, stub)
    for name in ("new_event_loop", "set_event_loop"):
        setattr(asyncio.events, name, stubs[name])
    # The loop classes themselves, and the policy method that instantiates them.
    for module in (asyncio, getattr(asyncio, "windows_events", None)):
        if module is None:
            continue
        for cls_name in ("ProactorEventLoop", "SelectorEventLoop"):
            if hasattr(module, cls_name):
                setattr(module, cls_name, _make_loop_stub(cls_name))
    asyncio.events.BaseDefaultEventLoopPolicy.new_event_loop = _make_loop_stub(
        "EventLoopPolicy.new_event_loop")


# --------------------------------------------------------------------------- #
# 9. the names the scaffold injects -- the entire surface the model CALLS
#
# (Not the entire surface it can REACH: see ENFORCEMENT LAYERING in the module
# docstring. This narrows one route; it does not close the class.)
#
# The two templates below are NEVER injected as written. `_build_injected()`
# rebuilds them with `types.FunctionType` over a MINIMAL globals dict, because
# `f.__globals__` is readable and writable from any cell: injected as-is,
# `final_answer.__globals__["_RESERVED"]["final_answer"] = ...` captures the
# terminal channel and `..["llm_query"] = ..` intercepts the sub-call plumbing --
# both measured. They therefore reference only `BRIDGE` and `LOOP`, and
# `_FINAL_TASK_NAME` is inlined as a literal so no shared mutable container is
# reachable either.
# --------------------------------------------------------------------------- #
_FINAL_TASK_NAME = "rlm-final-answer"


async def _llm_query_template(question, *, chunk=None, role="leaf"):
    """Ask a sub-model about one excerpt. The only way out of the sandbox.

    Pass the excerpt as `chunk=` and the question positionally. The scaffold
    composes the sub-model's prompt as [system prefix][chunk][question] --
    question LAST -- so a chunk asked twice is prefilled once and reused. That
    layout is enforced on the scaffold's side of this pipe, not left to how
    this call site happens to concatenate its string.

    `chunk` may be omitted, in which case `question` is the whole prompt and
    the scaffold has no way to know where the excerpt ended.

    C4's semaphore, /tokenize pre-flight, retries, timeouts, budget admission
    and step logging all live on the OTHER side of this pipe, where model code
    cannot reach them. The signature is closed (no `**kw`): the payload's
    fields are the scaffold's, not something a cell can extend.
    """
    return await BRIDGE.request(
        "llm_query", {"question": question, "chunk": chunk, "role": role})


def _final_answer_template(value):
    """Submit the episode's answer. Call it directly -- no `await` needed.

    Deliberately synchronous: an `async def` here would make the bare call
    `final_answer(x)` -- the shape every prompt teaches -- a silent no-op plus a
    'coroutine was never awaited' warning. Delivery is still guaranteed: the
    scaffold drains the send before it returns the observation, and reports back
    into this cell's stderr if the scaffold rejected it.
    """
    LOOP.create_task(BRIDGE.request("final_answer", {"value": value}),
                     name="rlm-final-answer")
    return None


def _rebuild(template, name: str, namespace: dict):
    fn = types.FunctionType(template.__code__, namespace, name,
                            template.__defaults__, template.__closure__)
    fn.__qualname__ = name
    fn.__module__ = "rlm_sandbox"
    fn.__doc__ = template.__doc__
    fn.__kwdefaults__ = template.__kwdefaults__
    return fn


def _build_injected() -> tuple:
    """(llm_query, final_answer) over a namespace that cannot reach this module."""
    ns = {"__builtins__": builtins, "__name__": "rlm_sandbox",
          "BRIDGE": BRIDGE, "LOOP": LOOP}
    return (_rebuild(_llm_query_template, "llm_query", ns),
            _rebuild(_final_answer_template, "final_answer", ns))


llm_query, final_answer = _build_injected()


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
    """Drop scaffold frames and VIRTUALIZE every remaining real filename,
    following `__cause__`/`__context__`.

    Stdlib frames are kept because they carry real diagnostic value
    (`json.loads('{bad')` should show the model `json/decoder.py`), but their
    absolute paths must not survive: unrewritten they put the interpreter's
    install directory into `CellOutput.traceback`, which reaches both the root's
    context and DuckDB. Cell frames (`<cell:3>`) pass through untouched, keeping
    their `~~^~~` anchors; rewritten frames carry their source line explicitly,
    since linecache can no longer resolve the virtual name.
    """
    te = traceback.TracebackException.from_exception(exc, lookup_lines=True)
    seen: set[int] = set()

    def rewrite(frame):
        virtual = _virtual_filename(frame.filename)
        if virtual is None:
            return None
        if virtual == frame.filename:
            return frame
        return traceback.FrameSummary(virtual, frame.lineno, frame.name,
                                      line=frame.line)

    def scrub(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        node.stack = traceback.StackSummary.from_list(
            [f for f in map(rewrite, node.stack) if f is not None])
        scrub(node.__cause__)
        scrub(node.__context__)

    scrub(te)
    return "".join(te.format()).rstrip()


_REJECTED_MSG = "final_answer was not accepted by the scaffold: {}\n"


async def _drain_final_answers() -> None:
    """Guarantee delivery before the observation is returned, and SURFACE a
    rejection into this cell's stderr instead of swallowing it -- a submission
    the scaffold refused is exactly the thing the root has to be told about.

    The tasks are found by name rather than through a shared list, because any
    container the injected `final_answer` could reach would also be reachable
    from a cell, and clearing it would swallow the submission again.
    """
    pending = [t for t in asyncio.all_tasks(LOOP)
               if t.get_name() == _FINAL_TASK_NAME]
    if not pending:
        return
    for result in await asyncio.gather(*pending, return_exceptions=True):
        if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError):
            _ERR.write(_REJECTED_MSG.format(f"{type(result).__name__}: {result}"))


def _loop_exception_handler(loop, context: dict) -> None:
    """asyncio's default handler logs through `logging`, whose last-resort
    handler writes to `sys.stderr` -- which is a CELL BUFFER here, so scaffold
    noise would land in whatever observation happened to be open. Route a
    rejected final_answer to the cell that caused it and everything else to the
    episode's own log."""
    future = context.get("future") or context.get("task")
    name = future.get_name() if hasattr(future, "get_name") else None
    if name == _FINAL_TASK_NAME:
        _ERR.write(_REJECTED_MSG.format(repr(context.get("exception"))))
        return
    try:
        print(context.get("message"), context.get("exception"),
              file=sys.__stderr__, flush=True)
    except BaseException:  # noqa: BLE001
        pass


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
        await _drain_final_answers()
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


_REAL_MODULE = None  # strong ref: see _hide_this_module


def _hide_this_module() -> None:
    """`import sys; sys.modules['__main__']` was the SHORTEST path from a cell to
    this module's namespace, and from there `_RESERVED['llm_query'] = ...`
    intercepts the sub-call plumbing and `_RESERVED['final_answer'] = ...`
    captures the terminal channel -- both measured against the first cut of this
    file. Swap in a decoy carrying nothing.

    This shuts the shortest path, not the class: the frame walk, the loop's
    exception handler and the bridge-object pivots all still land in the same
    dict (module docstring, ENFORCEMENT LAYERING). Kept because it is three
    lines and removes the route a model would stumble into by accident, not
    because it makes anything safe.
    """
    global _REAL_MODULE
    _REAL_MODULE = sys.modules.get("__main__")
    decoy = types.ModuleType("__main__")
    decoy.__doc__ = "rlm-halo sandbox (episode-scoped interpreter)"
    sys.modules["__main__"] = decoy


def main() -> None:
    sys.stdout = _OUT
    sys.stderr = _ERR
    sys.displayhook = lambda value: None
    sys.unraisablehook = _drop_unraisable
    warnings.showwarning = _drop_showwarning
    LOOP.set_exception_handler(_loop_exception_handler)

    _install_asyncio_stubs()
    _install_audit_hook("--deny-ctypes" in sys.argv[1:])
    BRIDGE.on_request(_handle)
    _hide_this_module()

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
