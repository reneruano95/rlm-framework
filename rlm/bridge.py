"""The C1/C4 bridge: the single channel between the scaffold and the
sandboxed interpreter (spec §5).

Transport is deliberately NOT this module's concern -- it operates on
already-open file descriptors. On Windows those come from two anonymous
`CreatePipe` pairs whose handles were whitelisted into the sandbox via
`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` (see `rlm.sandbox.winproc.spawn`);
`os.pipe()` works identically for the in-process tests below. Rejected
alternatives, all measured, not assumed: AF_UNIX is inert on Windows
(`bind(): bad family` -- no AF_UNIX branch in CPython's Windows sockaddr
code), `socket.socketpair()` is an AF_INET loopback pair (would violate the
no-network rule the sandbox itself enforces), and `multiprocessing` needs
either PROCESS_DUP_HANDLE on the untrusted sandbox or a machine-visible
named pipe (enumerated and hijacked from an unrelated process in testing).
Anonymous pipes create zero OS-namespace objects.

Wire format
-----------
frame = ASCII decimal byte-length, "\\n", that many bytes of
``json.dumps(obj, ensure_ascii=True).encode("ascii")``. ASCII framing on
BOTH sides is non-negotiable: the truncator's hypothesis suite generates
lone surrogates, and ``ensure_ascii=False`` raises ``UnicodeEncodeError`` on
them -- ``ensure_ascii=True`` round-trips them intact (verified both
directions, inside a live AppContainer sandbox).

    {"t":"req",    "id":<int>, "kind":<str>, "p":<any>}
    {"t":"rep",    "id":<int>, "ok":true,  "r":<any>}
    {"t":"rep",    "id":<int>, "ok":false, "e":{"type":.., "msg":..}}
    {"t":"cancel", "id":<int>}

``id`` correlates replies, so either side may answer out of order and hold
unlimited calls in flight.

Threading
---------
Each `BridgeEndpoint` runs one reader thread and one writer thread over
blocking fds and delivers frames to its asyncio loop with
``loop.call_soon_threadsafe`` -- the loop itself never performs a blocking
read or write. This is deliberate: anonymous Windows pipes are not
FILE_FLAG_OVERLAPPED, so ProactorEventLoop cannot drive them.

Symmetry
--------
`BridgeEndpoint` is the ONE implementation used by both sides: the scaffold
(`BridgeParent`, an alias) calls `.request()` for nothing today but will as
soon as it needs to push into the sandbox, and serves `llm_query`/
`final_answer` via `.on_request()`; the sandbox child (Task 9) uses the
identical class the other way around. Sharing one class is what keeps the
protocol from drifting into two implementations that quietly disagree.
"""
from __future__ import annotations

import asyncio
import errno
import itertools
import json
import os
import queue
import threading
from collections.abc import Awaitable, Callable

from rlm.errors import SandboxError

MAX_FRAME = 64 * 1024 * 1024   # 64 MiB hard ceiling on one frame
READ_BUFFER = 1 << 20          # 1 MiB: unbuffered/byte-at-a-time reads measured 100x slower

_EOF_ERRNOS = {errno.EPIPE, errno.EBADF, errno.EINVAL, errno.ESHUTDOWN}
_EOF_WINERRORS = {109, 232}    # ERROR_BROKEN_PIPE, ERROR_NO_DATA


class BridgeError(SandboxError):
    """Base class for every bridge failure."""


class BridgeClosed(BridgeError):
    """The peer went away (crash, kill, budget-kill) while calls were pending."""


class RemoteError(BridgeError):
    """The peer answered with an error (rejected / timed out / raised)."""

    def __init__(self, etype: str, msg: str) -> None:
        super().__init__(f"{etype}: {msg}")
        self.type = etype
        self.msg = msg


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #

def encode_frame(obj) -> bytes:
    payload = json.dumps(obj, ensure_ascii=True).encode("ascii")
    if len(payload) > MAX_FRAME:
        raise ValueError(f"frame too large: {len(payload)} > {MAX_FRAME}")
    return f"{len(payload)}\n".encode("ascii") + payload


class FrameReader:
    """Stateful, feed-driven decoder: `.feed(chunk)` returns zero or more
    decoded objects, buffering whatever's left of a partial frame. An
    oversize frame is refused the instant its header is seen -- it is never
    buffered while waiting for the (potentially unbounded) body."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list:
        self._buf.extend(chunk)
        out = []
        while True:
            nl = self._buf.find(b"\n")
            if nl == -1:
                break
            header = bytes(self._buf[:nl])
            try:
                size = int(header)
            except ValueError as exc:
                raise ValueError(f"invalid frame header: {header!r}") from exc
            if size > MAX_FRAME:
                raise ValueError(f"frame too large: {size} > {MAX_FRAME}")
            end = nl + 1 + size
            if len(self._buf) < end:
                break  # wait for more bytes
            body = bytes(self._buf[nl + 1:end])
            del self._buf[:end]
            out.append(json.loads(body.decode("ascii")))
        return out


# --------------------------------------------------------------------------- #
# duplex channel: reader thread + writer thread bound to one asyncio loop
# --------------------------------------------------------------------------- #

_STOP = object()


class _Channel:
    def __init__(self, rfd: int, wfd: int, loop: asyncio.AbstractEventLoop,
                 on_frame: Callable[[dict], None],
                 on_close: Callable[[BaseException], None], *, tag: str = "bridge") -> None:
        self._rfd, self._wfd = rfd, wfd
        self._loop = loop
        self._on_frame, self._on_close = on_frame, on_close
        self._outq: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._rt = threading.Thread(target=self._read_loop, name=f"{tag}-r", daemon=True)
        self._wt = threading.Thread(target=self._write_loop, name=f"{tag}-w", daemon=True)

    def start(self) -> None:
        self._rt.start()
        self._wt.start()

    def send(self, obj) -> None:
        """Thread-safe; never blocks the caller (and therefore never the loop)."""
        if not self._closed.is_set():
            self._outq.put(obj)

    def close(self, reason: BaseException | None = None) -> None:
        self._fail(reason or BridgeClosed("closed locally"))

    def _fail(self, exc: BaseException) -> None:
        # fds are closed only by the thread that owns them (see the loops
        # below): closing an fd out from under a thread blocked in os.read
        # on Windows can hand that thread a recycled handle.
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
        self._outq.put(_STOP)
        try:
            self._loop.call_soon_threadsafe(self._on_close, exc)
        except RuntimeError:
            pass  # loop already closed

    def _read_loop(self) -> None:
        reader = FrameReader()
        exc: BaseException | None = None
        try:
            while True:
                try:
                    chunk = os.read(self._rfd, READ_BUFFER)
                except OSError as e:
                    if getattr(e, "winerror", None) in _EOF_WINERRORS or e.errno in _EOF_ERRNOS:
                        break
                    raise
                if not chunk:
                    break
                for frame in reader.feed(chunk):
                    try:
                        self._loop.call_soon_threadsafe(self._on_frame, frame)
                    except RuntimeError:
                        break
        except BaseException as e:  # noqa: BLE001
            exc = e
        finally:
            self._fail(exc if exc is not None else BridgeClosed("peer closed the bridge"))
            try:
                os.close(self._rfd)
            except OSError:
                pass

    def _write_loop(self) -> None:
        try:
            while True:
                obj = self._outq.get()
                if obj is _STOP:
                    return
                try:
                    view = memoryview(encode_frame(obj))
                    while view:
                        n = os.write(self._wfd, view)
                        if n == 0:
                            raise BridgeClosed("write returned 0")
                        view = view[n:]
                except BaseException as e:  # noqa: BLE001
                    self._fail(e)
                    return
        finally:
            try:
                os.close(self._wfd)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# the symmetric endpoint
# --------------------------------------------------------------------------- #

RequestHandler = Callable[[str, object], Awaitable[object]]


class BridgeEndpoint:
    """One end of the bridge. Symmetric: `.request()` calls the peer,
    `.on_request()` serves the peer's calls -- both directions are always
    live on the same object, which is what lets both the scaffold and the
    sandbox child use this one class."""

    def __init__(self, rfd: int, wfd: int, *,
                 loop: asyncio.AbstractEventLoop | None = None,
                 tag: str = "bridge") -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._inflight: dict[int, asyncio.Task] = {}
        self._handler: RequestHandler | None = None
        self._dead: BaseException | None = None
        self._chan = _Channel(rfd, wfd, self._loop, self._on_frame, self._on_close, tag=tag)
        self._chan.start()

    # -- public API ----------------------------------------------------------- #

    def on_request(self, handler: RequestHandler) -> None:
        """`handler(kind, payload)` -> awaitable result. Replaces any prior
        handler. This is where C4's semaphore / retries / timeouts / C6 step
        logging live -- all of it on whichever side registers the handler."""
        self._handler = handler

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def request(self, kind: str, payload) -> dict:
        if self._dead is not None:
            raise self._dead
        rid = next(self._ids)
        fut = self._loop.create_future()
        self._pending[rid] = fut
        self._chan.send({"t": "req", "id": rid, "kind": kind, "p": payload})
        try:
            return await fut
        except asyncio.CancelledError:
            self._pending.pop(rid, None)
            self._chan.send({"t": "cancel", "id": rid})
            raise

    def close(self, reason: BaseException | None = None) -> None:
        self._chan.close(reason or BridgeClosed("closed locally"))

    # -- protocol -------------------------------------------------------------- #

    def _on_frame(self, msg: dict) -> None:
        kind = msg.get("t")
        if kind == "req":
            rid = msg["id"]
            task = self._loop.create_task(self._serve(rid, msg.get("kind"), msg.get("p")))
            self._inflight[rid] = task
        elif kind == "rep":
            fut = self._pending.pop(msg["id"], None)
            if fut is None or fut.done():
                return
            if msg.get("ok"):
                fut.set_result(msg.get("r"))
            else:
                err = msg.get("e") or {}
                fut.set_exception(RemoteError(err.get("type", "Error"), err.get("msg", "")))
        elif kind == "cancel":
            task = self._inflight.get(msg["id"])
            if task is not None:
                task.cancel()

    async def _serve(self, rid: int, kind: str, payload) -> None:
        try:
            if self._handler is None:
                raise SandboxError(f"no handler registered for {kind!r}")
            result = await self._handler(kind, payload)
        except asyncio.CancelledError:
            self._inflight.pop(rid, None)
            raise
        except BaseException as exc:  # noqa: BLE001
            self._reply(rid, False, {"type": type(exc).__name__, "msg": str(exc)})
        else:
            self._reply(rid, True, result)
        self._inflight.pop(rid, None)

    def _reply(self, rid: int, ok: bool, value) -> None:
        self._chan.send({"t": "rep", "id": rid, "ok": ok, ("r" if ok else "e"): value})

    def _on_close(self, exc: BaseException) -> None:
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        self._dead = exc if isinstance(exc, BridgeError) else BridgeClosed(f"bridge closed: {exc!r}")
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(self._dead)
        self._pending.clear()


# The scaffold's own name for the class it holds; the sandbox child (Task 9)
# uses the identical BridgeEndpoint -- see the module docstring.
BridgeParent = BridgeEndpoint
