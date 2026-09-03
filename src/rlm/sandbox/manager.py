"""C1 SandboxManager: one Job, one AppContainer, one interpreter, one bridge
per episode (spec §5).

This module owns the composite spawn nobody had run before this plan: a single
`InitializeProcThreadAttributeList(count=2)` carrying BOTH
PROC_THREAD_ATTRIBUTE_HANDLE_LIST and PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
`CreateProcessW(CREATE_SUSPENDED)`, `AssignProcessToJobObject`, `ResumeThread`
(the mechanics live in `rlm.sandbox.winproc`; the ordering is why
`subprocess.Popen` cannot be used at all).

Two decisions here came from measured defects rather than design taste:

  D22 THE KILL SEQUENCE HAS ONE OWNER AND ONE ORDER.
        SandboxSession.kill(reason, code)
          -> TerminateJobObject(hjob, code)    [never proc.kill(): the job is
             the only tree-wide primitive, and it is the one that carries both
             the attributable reason and the chosen exit code]
          -> the bridge cancels every in-flight handler task
          -> each cancelled handler writes its status=cancelled step  (C4/C6)
          -> await tl.drain() -> await tl.aclose()                    (C6)
          -> exit                                                     (episode runner)
      The last three steps belong to the episode runner, not here; what this
      module guarantees is that they run AFTER the cancellations and that the
      kill decision, the reason string and the trace write share one
      serialization point. That is why the Job's completion-port pump only ever
      does `loop.call_soon_threadsafe(session._on_job_notification, msg, ts)`
      and never terminates anything itself.

  D23 THE HANDSHAKE PID IS ASSERTED. The first bridge frame is
      `{"kind":"handshake","pid": os.getpid()}` and the episode is refused
      unless it equals `PROCESS_INFORMATION.dwProcessId`. With `CreateProcessW`
      against an absolute interpreter path the two always agree; the assert
      stays because it costs one comparison and catches the config error that
      would otherwise reintroduce the uv-trampoline defect (a launcher pid
      recorded as `episodes.sandbox_pid`, so §6 recovery reaps the wrong
      process and `TerminateJobObject` targets a stub).

  D10 EXIT CODES ARE NOT AN OUTCOME CHANNEL. `close()` returns the child's exit
      code because callers want it recorded, not because it diagnoses anything:
      `outcome_reason` comes from `kill_reason` plus the child's explicit `bye`
      frame (`said_bye`). Any other exit code means "unattributed sandbox
      death".

D7/Conflict 8: the AppContainer token denies this repository, so the child is
never executed in place -- `install_bootstrap()` stages it, plus the only two
scaffold modules it imports, into a dedicated directory that holds nothing
else. That directory is the ONLY thing granted to ALL APPLICATION PACKAGES;
config.yaml, prompts/ and traces/ stay unreadable to model code by default,
which `test_sandbox_cannot_read_the_repo` checks rather than assumes.

This module -- the token, the Job Object's `ACTIVE_PROCESS` limit of 1, and the
fact that budgets, routing, truncation and termination all run HERE rather than
in the sandbox -- is where C1's isolation actually lives (spec v0.2.3, §5 C1).
The controls inside `child.py` raise the bar and are not a boundary; see its
ENFORCEMENT LAYERING section before assuming anything about them.
"""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import msvcrt
import os
import re
import subprocess  # icacls only, and only from the explicit install step
import uuid
import _winapi
from collections.abc import Awaitable, Callable
from ctypes import wintypes
from pathlib import Path

from rlm import trace
from rlm.bridge import BridgeError, BridgeParent
from rlm.errors import SandboxError
from rlm.sandbox import winproc
from rlm.sandbox.winjob import Job
from rlm.context.truncate import CellOutput

STILL_ACTIVE = 259

# Notifications that mean the sandbox breached a kernel-enforced limit. The
# rest (NEW_PROCESS/EXIT_PROCESS/ACTIVE_PROCESS_ZERO) are lifecycle noise, and
# ABNORMAL_EXIT_PROCESS is what our OWN TerminateJobObject produces -- treating
# it as a violation would make every kill re-enter itself.
_VIOLATIONS = frozenset({
    "PROCESS_MEMORY_LIMIT", "JOB_MEMORY_LIMIT",
    "END_OF_JOB_TIME", "END_OF_PROCESS_TIME",
})

ALL_APPLICATION_PACKAGES = "S-1-15-2-1"
_AC_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_RLM_PKG = Path(__file__).resolve().parent.parent

# What lands in the bootstrap dir, and nothing else: the child, and the two
# stdlib-only scaffold modules it imports. Copies, not a path grant -- one
# bridge implementation on the wire is what keeps the protocol from drifting
# into two that quietly disagree, and `install_bootstrap` re-copies whenever
# the bytes differ.
_STAGED_FILES: tuple[tuple[Path, str], ...] = (
    (_RLM_PKG / "sandbox" / "child.py", "sandbox_child.py"),
    (_RLM_PKG / "__init__.py", "rlm/__init__.py"),
    (_RLM_PKG / "errors.py", "rlm/errors.py"),
    (_RLM_PKG / "bridge.py", "rlm/bridge.py"),
)


# --------------------------------------------------------------------------- #
# install-time helpers (explicit; the runtime never grants an ACL by itself)
# --------------------------------------------------------------------------- #

def bootstrap_acl_command(bootstrap_dir: str | os.PathLike) -> list[str]:
    """The one-time grant an operator must run. Grant to S-1-15-2-1 (ALL
    APPLICATION PACKAGES), NEVER to a per-episode AppContainer SID:
    `DeleteAppContainerProfile` does not remove ACEs naming that SID, so a
    per-SID grant accrues one dead ACE per episode (100 per bench run)."""
    return ["icacls", str(bootstrap_dir), "/grant",
            f"*{ALL_APPLICATION_PACKAGES}:(OI)(CI)(RX)", "/T", "/C", "/Q"]


def install_bootstrap(sandbox_cfg, *, grant_acl: bool = False) -> Path:
    """Stage the child into `sandbox_cfg.bootstrap_dir`.

    Copying is idempotent and safe to run on every session (a few KB, and only
    when the bytes actually differ). `grant_acl` is opt-in and belongs to an
    install/validate step, never to an episode: the runtime must not hand
    filesystem access to AppContainers as a side effect of running a task.
    """
    dest = Path(sandbox_cfg.bootstrap_dir)
    (dest / "rlm").mkdir(parents=True, exist_ok=True)
    for src, rel in _STAGED_FILES:
        dst = dest / rel
        data = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            dst.write_bytes(data)
    if grant_acl:
        subprocess.run(bootstrap_acl_command(dest), capture_output=True,
                       text=True, check=False)
    return dest


def _close_handles(*handles: int) -> None:
    for handle in handles:
        with contextlib.suppress(OSError):
            _winapi.CloseHandle(handle)


def _sandbox_cfg(cfg):
    """Accept either the whole `Config` or just its `scaffold.sandbox` block."""
    scaffold = getattr(cfg, "scaffold", None)
    return getattr(scaffold, "sandbox", None) if scaffold is not None else cfg


# --------------------------------------------------------------------------- #
# session
# --------------------------------------------------------------------------- #

LlmQueryHandler = Callable[[dict], Awaitable[object]]
FinalAnswerHandler = Callable[[object], Awaitable[None]]
EnvHandler = Callable[[dict], Awaitable[object]]


class SandboxSession:
    """One episode's sandbox. Constructed only by `SandboxManager.session`."""

    def __init__(self, episode_id: str, *, bridge, job: Job,
                 appcontainer: winproc.AppContainer | None, hprocess: int,
                 pid: int, loop: asyncio.AbstractEventLoop,
                 shutdown_grace_s: float) -> None:
        self.episode_id = episode_id
        self.pid = pid
        self.kill_reason: str | None = None
        self.said_bye = False
        self.final_answers: list = []
        self.job_notifications: list[tuple[str, float]] = []
        self.fatal: str | None = None

        self._bridge = bridge
        self._job = job
        self._ac = appcontainer
        self._hprocess = hprocess
        self._loop = loop
        self._grace = shutdown_grace_s
        self._exit_code: int | None = None
        self._dead: SandboxError | None = None
        self._llm_query_handler: LlmQueryHandler | None = None
        self._final_answer_handler: FinalAnswerHandler | None = None
        self._env_handler: EnvHandler | None = None

    # -- handler registration --------------------------------------------- #

    def on_llm_query(self, handler: LlmQueryHandler) -> None:
        """`handler(payload) -> awaitable result`, where payload carries at
        least `prompt` and `role`. C4's semaphore, /tokenize pre-flight,
        retries, timeouts, budget admission and C6 step logging all live in
        this handler -- i.e. entirely scaffold-side (I1)."""
        self._llm_query_handler = handler

    def on_final_answer(self, handler: FinalAnswerHandler) -> None:
        self._final_answer_handler = handler

    def on_env(self, handler: EnvHandler) -> None:
        """`handler(payload) -> awaitable result` for the `env` verb's three
        calls (`op` in `payload` is `search`/`open`/`window`). Task 12 owns the
        real handler (`InteractiveIndex` + `_on_env`); this registration slot
        and the `_serve` dispatch below are the minimal plumbing this task
        needs so the child-side tests can exercise the frame end to end."""
        self._env_handler = handler

    # -- the two things the scaffold pushes in ----------------------------- #

    async def setvar(self, name: str, value) -> None:
        self._check_alive()
        async with self._bridge_failure_guard():
            await self._bridge.request("setvar", {"name": name, "value": value})

    async def exec_cell(self, cell: str) -> CellOutput:
        self._check_alive()
        async with self._bridge_failure_guard():
            r = await self._bridge.request("exec", {"cell": cell})
        return CellOutput(
            stdout=r.get("stdout") or "",
            stderr=r.get("stderr") or "",
            repr_=r.get("repr") or "",
            traceback=r.get("traceback") or "",
        )

    @contextlib.asynccontextmanager
    async def _bridge_failure_guard(self):
        """A result frame that never arrives has exactly two causes, and §6's
        `status` ENUM cannot tell them apart on its own. Classify here, where
        both facts are available: the sandbox is either dead
        (`sandbox_died_mid_cell`) or alive with a corrupted stream
        (`bridge_desync` -- reachable from ordinary model code, since a cell can
        write to the protocol fd). Either way it routes through `kill()`, so
        D22's single-owner property holds and `kill_reason` stays the one
        attributable string."""
        try:
            yield
        except BridgeError as exc:
            reason = (trace.BRIDGE_DESYNC_REASON if self._is_alive()
                      else trace.SANDBOX_DEATH_REASON)
            await self.kill(reason, 0xB0DE)
            raise SandboxError(
                f"sandbox bridge failed mid-request ({reason}): {exc}") from exc

    # -- D22 ---------------------------------------------------------------- #

    async def kill(self, reason: str, code: int) -> None:
        """Step 1 and 2 of D22's sequence, in that order and nowhere else.

        The C5 wall-clock timer, a job-limit breach and an operator Ctrl-C all
        enter here, so `kill_reason` is the single attributable string that
        `episodes.outcome_reason` is built from -- not the exit code (D10).
        """
        if self.kill_reason is not None:
            return
        self.kill_reason = reason
        self._dead = SandboxError(f"sandbox killed: {reason}")
        self._job.terminate(code)          # never proc.kill()
        self._bridge.close(self._dead)     # cancels every in-flight handler task
        # let the bridge's on-close callback run before anyone observes us:
        # cancelled handlers write their status=cancelled steps from there.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def close(self) -> int:
        """Graceful when possible, unconditional when not. Returns the child's
        exit code (0 on the clean path, thanks to the child's `os._exit(0)` --
        see D9)."""
        if self._exit_code is not None:
            return self._exit_code
        if self.kill_reason is None:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(
                    self._bridge.request("shutdown", None), self._grace)
        code = await self._wait_exit(self._grace)
        if code is None:
            self._job.terminate(0xB0DE)
            code = await self._wait_exit(5.0)
        self._exit_code = 0xB0DE if code is None else code
        self._dead = self._dead or SandboxError("sandbox session is closed")
        self._bridge.close(self._dead)
        self._job.close()
        if self._ac is not None:
            self._ac.delete()
        if self._hprocess:
            winproc.kernel32.CloseHandle(self._hprocess)
            self._hprocess = 0
        return self._exit_code

    # -- internals ---------------------------------------------------------- #

    def _check_alive(self) -> None:
        # Deterministic: the bridge's own `_dead` is set from a thread via
        # call_soon_threadsafe, so immediately after kill() it may not be set
        # yet and `request()` would wait forever on a corpse.
        if self._dead is not None:
            raise self._dead

    def _is_alive(self) -> bool:
        if not self._hprocess:
            return False
        code = wintypes.DWORD()
        winproc.kernel32.GetExitCodeProcess(self._hprocess, ctypes.byref(code))
        return code.value == STILL_ACTIVE

    async def _wait_exit(self, timeout_s: float) -> int | None:
        """Poll on the loop -- no executor thread, no blocking call."""
        deadline = self._loop.time() + timeout_s
        code = wintypes.DWORD()
        while True:
            winproc.kernel32.GetExitCodeProcess(self._hprocess, ctypes.byref(code))
            if code.value != STILL_ACTIVE:
                return code.value
            if self._loop.time() >= deadline:
                return None
            await asyncio.sleep(0.005)

    async def _serve(self, kind: str, payload):
        """The child's whole vocabulary. Anything else is a protocol error."""
        if kind == "llm_query":
            if self._llm_query_handler is None:
                raise SandboxError(
                    "no llm_query handler registered for this episode")
            return await self._llm_query_handler(payload)
        if kind == "env":
            if self._env_handler is None:
                raise SandboxError(
                    "no env handler registered for this episode")
            return await self._env_handler(payload)
        if kind == "final_answer":
            value = (payload or {}).get("value")
            self.final_answers.append(value)
            if self._final_answer_handler is not None:
                await self._final_answer_handler(value)
            return None
        if kind == "bye":
            self.said_bye = True
            return None
        if kind == "fatal":
            self.fatal = (payload or {}).get("traceback")
            return None
        raise SandboxError(f"unknown child frame kind: {kind!r}")

    def _job_notification_pump(self, msg: str, ts: float) -> None:
        """Runs on the Job's daemon pump thread. D22: it does exactly one
        thing -- hand the notification to the loop. It never terminates
        anything, so the kill decision, the reason string and the trace write
        keep a single serialization point."""
        try:
            self._loop.call_soon_threadsafe(self._on_job_notification, msg, ts)
        except RuntimeError:
            pass  # loop already closed; the job handle reaps the tree anyway

    def _on_job_notification(self, msg: str, ts: float) -> None:
        self.job_notifications.append((msg, ts))
        if msg in _VIOLATIONS and self.kill_reason is None:
            # A memory cap does NOT kill by itself: JOB_OBJECT_LIMIT_*_MEMORY
            # only makes the allocation fail, so model code with a bare
            # `except MemoryError` would run forever. This is the hard kill.
            self._loop.create_task(self.kill(msg.lower(), 0xB0DE))


class _FrameGate:
    """D23 is about ORDER as much as identity.

    Until the handshake has been seen, the episode has not been admitted: no
    `llm_query` may be dispatched and no `final_answer` recorded in its name.
    A second handshake is refused too -- one episode, one identity.
    """

    def __init__(self, handshake: asyncio.Future, serve) -> None:
        self._handshake = handshake
        self._serve = serve

    async def __call__(self, kind: str, payload):
        if kind == "handshake":
            if self._handshake.done():
                raise SandboxError("duplicate handshake frame; refused")
            self._handshake.set_result((payload or {}).get("pid"))
            return None
        if not self._handshake.done():
            raise SandboxError(
                f"{kind!r} frame arrived before the handshake; refused")
        return await self._serve(kind, payload)


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #

class SandboxManager:
    """Spawns and disposes of sandboxes. Holds no per-episode state itself, so
    one manager can serve a whole `rlm bench` run in a single process."""

    def __init__(self, *, stderr_dir: str | os.PathLike | None = None,
                 handshake_timeout_s: float = 60.0,
                 shutdown_grace_s: float = 15.0) -> None:
        self.stderr_dir = Path(stderr_dir) if stderr_dir is not None else None
        self.handshake_timeout_s = handshake_timeout_s
        self.shutdown_grace_s = shutdown_grace_s
        self._shared_ac: winproc.AppContainer | None = None
        self._shared_sid: int | None = None

    def close(self) -> None:
        """Only needed when `appcontainer_per_episode` is false -- otherwise
        every session already deleted its own profile."""
        if self._shared_ac is not None:
            self._shared_ac.delete()
            self._shared_ac = None
            self._shared_sid = None

    @contextlib.asynccontextmanager
    async def session(self, episode_id: str, cfg):
        session = await self._start(episode_id, cfg)
        try:
            yield session
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    # -- spawn --------------------------------------------------------------- #

    async def _start(self, episode_id: str, cfg) -> SandboxSession:
        sbx = _sandbox_cfg(cfg)
        loop = asyncio.get_running_loop()
        bootstrap = install_bootstrap(sbx)
        script = bootstrap / "sandbox_child.py"

        # Two anonymous pipes. `_winapi.CreatePipe` creates both ends
        # NON-inheritable, so only the child's ends are opened up -- the same
        # end state as CreatePipe(bInheritHandle=TRUE) followed by
        # SetHandleInformation(parent_end, HANDLE_FLAG_INHERIT, 0), with one
        # fewer ctypes call to get wrong.
        c_in_r, p_in_w = _winapi.CreatePipe(None, 0)    # parent -> child
        p_out_r, c_out_w = _winapi.CreatePipe(None, 0)  # child -> parent

        # Everything from here to a successful spawn is unwound on ANY failure:
        # a leaked pipe handle is a handle the child never sees EOF on, and a
        # leaked Job handle is a process tree nothing will reap.
        errf = job = ac = None
        sid = None
        try:
            os.set_handle_inheritable(c_in_r, True)
            os.set_handle_inheritable(c_out_w, True)

            errf = self._open_child_stderr(episode_id)
            herr = msvcrt.get_osfhandle(errf.fileno())
            os.set_handle_inheritable(herr, True)

            job = Job(memory_limit_mb=sbx.memory_limit_mb,
                      active_process_limit=sbx.active_process_limit)
            ac, sid = self._mint_appcontainer(episode_id, sbx)

            args = ["-B", "-I", "-u", str(script)]
            # Under `appcontainer` the kernel already returns WSAEACCES on the
            # raw ws2_32 FFI path, so denying ctypes (which also breaks a fresh
            # `import ctypes` outright) is optional there. Under `audit_only`
            # the ctypes deny set is the only thing between model code and the
            # leaf server on 127.0.0.1, so it is forced on regardless of config.
            if sbx.deny_ctypes or sid is None:
                args.append("--deny-ctypes")

            res = winproc.spawn(str(sbx.interpreter), args, [], sid, job, None,
                                winproc.Stdio(c_in_r, c_out_w, herr),
                                cwd=str(bootstrap))
        except BaseException as exc:
            # Every handle, including the parent ends: nothing survives a spawn
            # that never happened.
            _close_handles(c_in_r, c_out_w, p_in_w, p_out_r)
            if errf is not None:
                errf.close()
            if job is not None:
                job.close()
            if ac is not None:
                ac.delete()
            if isinstance(exc, OSError):
                raise SandboxError(
                    f"could not spawn the sandbox interpreter ({exc}). If the "
                    f"AppContainer cannot reach the bootstrap directory, run "
                    f"once: "
                    f"{subprocess.list2cmdline(bootstrap_acl_command(bootstrap))}"
                ) from exc
            raise

        # Our copies of the child's ends must go now, or we never see EOF.
        _close_handles(c_in_r, c_out_w)
        errf.close()
        winproc.kernel32.CloseHandle(res.hthread)

        rfd = msvcrt.open_osfhandle(p_out_r, os.O_RDONLY | os.O_BINARY)
        wfd = msvcrt.open_osfhandle(p_in_w, os.O_BINARY)
        bridge = BridgeParent(rfd, wfd, loop=loop, tag=f"scaffold:{episode_id}")

        session = SandboxSession(
            episode_id, bridge=bridge, job=job, appcontainer=ac,
            hprocess=res.hprocess, pid=res.pid, loop=loop,
            shutdown_grace_s=self.shutdown_grace_s)

        handshake: asyncio.Future = loop.create_future()

        bridge.on_request(_FrameGate(handshake, session._serve))
        job.watch(session._job_notification_pump)

        try:
            child_pid = await asyncio.wait_for(handshake, self.handshake_timeout_s)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            # kill() FIRST: close() would spend the full shutdown grace waiting
            # for a graceful `bye` from a child that has already proved it is
            # not talking to us.
            await session.kill("handshake_timeout", 0xC5)
            await session.close()
            raise SandboxError(
                f"sandbox {episode_id} never sent its handshake frame "
                f"(pid {res.pid}). Most likely the AppContainer cannot read "
                f"the bootstrap directory; run once: "
                f"{subprocess.list2cmdline(bootstrap_acl_command(bootstrap))}"
            ) from exc

        if child_pid != res.pid:  # D23
            await session.kill("handshake_pid_mismatch", 0xC5)
            await session.close()
            raise SandboxError(
                f"handshake pid {child_pid} != PROCESS_INFORMATION.dwProcessId "
                f"{res.pid}: scaffold.sandbox.interpreter is not the real base "
                f"interpreter (a launcher/trampoline re-exec'd it). Episode "
                f"refused -- recording the wrong sandbox_pid would make §6 "
                f"recovery reap the wrong process."
            )
        return session

    # -- helpers -------------------------------------------------------------- #

    def _open_child_stderr(self, episode_id: str):
        """fd 2 is the episode's own log, never the model's observation: the
        child routes cell stderr through the bridge as `str` (D12), so anything
        arriving here is a bootstrap failure worth keeping."""
        if self.stderr_dir is None:
            return open(os.devnull, "wb")
        self.stderr_dir.mkdir(parents=True, exist_ok=True)
        safe = _AC_NAME_UNSAFE.sub("-", episode_id)
        return open(self.stderr_dir / f"{safe}.stderr.log", "wb")

    def _mint_appcontainer(self, episode_id: str, sbx):
        """CapabilityCount=0, minted fresh per episode by default: that is what
        discharges "no state shared between episodes" at the OS level (private
        redirected %TEMP% included) and it costs ~0.02 s."""
        if getattr(sbx, "network_isolation", "appcontainer") != "appcontainer":
            return None, None
        if getattr(sbx, "appcontainer_per_episode", True):
            ac = winproc.AppContainer()
            name = (f"rlmhalo.{_AC_NAME_UNSAFE.sub('-', episode_id)[:24]}."
                    f"{uuid.uuid4().hex[:8]}")
            return ac, ac.create(name)
        # Shared profile: the manager owns it, so no session deletes it out
        # from under the next episode. Sharing forfeits the per-episode %TEMP%
        # isolation, which is why it is not the default.
        if self._shared_ac is None:
            self._shared_ac = winproc.AppContainer()
            self._shared_sid = self._shared_ac.create(
                f"rlmhalo.shared.{uuid.uuid4().hex[:8]}")
        return None, self._shared_sid
