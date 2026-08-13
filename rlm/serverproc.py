"""Who owns a model-server PROCESS (spec §4, §5 C4 v0.2.6).

R13's mitigation gives every window a never-reused slot, so a pool of
`--parallel` slots is spent after `--parallel` windows and a 200K corpus needs
261 of them. §5 C4 therefore permits a ROTATION -- stop the healthy leaf, start
a fresh one, re-run §4's `/props` handshake, resume -- on pool exhaustion only.
Something has to own the process for that, and this module is the seam.

**Why the seam is here and not in C4.** C4 is the one module allowed to talk to
a model server, and it is the module whose testability the whole R13 mitigation
rests on: every slot-discipline test runs against a loopback fake with no
process anywhere. Giving C4 a launcher would put `CreateProcess`, a health
poll, and a copy of the S0-validated launch flags inside the component that
must stay mockable, and would give the dispatcher a code path that restarts a
server -- the exact thing §5 C4 forbids it, so that a FAILED server can never
be relaunched under the trace's nose. Instead the episode runner (the
composition root) drives a rotation through the small interface below, and the
CLI (the process root) supplies an implementation. Launch flags then live in
exactly one place -- `config.yaml` -- which is also the only place
`config_snapshot` can record them from (R11).

**What this module may not do:** talk HTTP. Readiness is an injected
`health_probe` coroutine, supplied by whoever already owns a client, so this
module imports nothing but the standard library and the config schema
(`tests/test_import_rules.py` lints it).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from rlm.config import ServerConfig
from rlm.errors import ServerRotationError

#: Wait this long for a fresh server to answer /health before giving up. A cold
#: load of the 20.6 GB leaf GGUF measured 10.4 s and a warm one 4.6-5.1 s
#: (`s2/R13-mitigations.md` §3), so this is ~10x the measured worst case: long
#: enough that a slow load is not mistaken for a failure, short enough that a
#: server which will never come up does not burn the whole wall clock.
DEFAULT_START_TIMEOUT_S = 120.0
DEFAULT_STOP_TIMEOUT_S = 30.0
DEFAULT_POLL_S = 0.5


@runtime_checkable
class ProcessManager(Protocol):
    """What the episode runner needs from whoever owns a server process.

    Deliberately one method. The runner's rotation is: quiesce C4, call this,
    re-run the §4 handshake, resume -- and every part of that except this call
    is scaffold logic the runner can be tested on with no process in sight.
    """

    async def restart(self) -> None:
        """Replace the running server with a fresh process of the same
        configuration, returning only once it answers /health.

        Raises `ServerRotationError` if that cannot be done -- including when
        the caller does not own the process. Never returns having done
        something OTHER than a full replacement: a rotation that quietly did
        not happen would leave the scaffold believing every slot is virgin
        while every one of them has held a document (R13).
        """
        ...


def _exe_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def launch_argv(server_cfg: ServerConfig) -> list[str]:
    """The exact command line `config.yaml` describes for one server.

    Every value comes from config: the model path, port, `-c`/`-np`, KV cache
    types, flash-attn, `-ub`/`-b`, and whatever `extra_flags` carries verbatim.
    Nothing is defaulted here -- a flag invented in code is a flag
    `config_snapshot` cannot record, and §8's whole comparison rests on the
    snapshot describing what actually ran (R11).

    `-lv 4` is the one addition, and it is a launch CONTRACT rather than a
    preference: D27 measured that `/props` cannot report KV cache types, so
    `rlm validate` recovers them by parsing this log level's output. A server
    launched below it is UNVERIFIED, which validate treats as a refusal.
    """
    argv = [str(Path(server_cfg.backend_dir) / _exe_name()),
            "-m", str(server_cfg.model),
            "--host", "127.0.0.1",
            "--port", str(server_cfg.port),
            "-c", str(server_cfg.ctx),
            "-np", str(server_cfg.parallel),
            "-ctk", server_cfg.cache_type,
            "-ctv", server_cfg.cache_type,
            "-fa", server_cfg.flash_attn,
            "-ub", str(server_cfg.ub),
            "-b", str(server_cfg.b),
            "-lv", "4"]
    for flag in server_cfg.extra_flags:
        argv.extend(flag.split())
    return argv


class LlamaServerProcess:
    """One llama-server process the scaffold owns, and can therefore replace.

    OWNERSHIP IS THE WHOLE POINT. `restart()` refuses unless this object
    spawned the process it is replacing. The alternative -- "stop" a server we
    never started (i.e. do nothing), spawn a second one on a taken port, watch
    it die on bind, poll `/health` and get a cheerful 200 from the ORIGINAL
    process -- ends with the scaffold resuming against a fresh slot pool on a
    server whose every slot has held a document. That is R13 reintroduced by
    R13's own mitigation, and it would be invisible in the trace.
    """

    def __init__(self, server_cfg: ServerConfig, *,
                 health_probe: Callable[[], Awaitable[bool]],
                 argv: list[str] | None = None,
                 env: dict[str, str] | None = None,
                 start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
                 stop_timeout_s: float = DEFAULT_STOP_TIMEOUT_S,
                 poll_s: float = DEFAULT_POLL_S) -> None:
        self.server_cfg = server_cfg
        self.argv = argv if argv is not None else launch_argv(server_cfg)
        self.env = env
        self._health = health_probe
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._poll_s = poll_s
        self._proc: subprocess.Popen | None = None
        self._log: Any = None

    # -- state ---------------------------------------------------------- #

    @property
    def owned(self) -> bool:
        """Did THIS object spawn the process that is running now?"""
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self.owned else None

    # -- lifecycle ------------------------------------------------------ #

    async def start(self) -> None:
        """Spawn the server and wait until it answers /health.

        stderr goes to `server_cfg.log_path` because D27's cache-type
        assertion parses it; a start that never becomes healthy is stopped
        before raising, so a failed rotation cannot leave an orphan holding
        the port (and 20 GB of weights) against the next attempt.
        """
        if self.owned:
            raise ServerRotationError(
                f"a {self.server_cfg.port} server is already owned by this "
                "manager; stop it before starting another")
        log_path = Path(self.server_cfg.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("wb")
        env = None
        if self.env is not None:
            env = {**os.environ, **self.env}
        try:
            self._proc = subprocess.Popen(  # noqa: S603 -- argv from config
                self.argv, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            self._close_log()
            raise ServerRotationError(
                f"could not launch {self.argv[0]!r}: {exc}") from exc
        try:
            await self._await_health()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Terminate the owned process and reap it. Idempotent."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            deadline = time.monotonic() + self._stop_timeout_s
            while proc.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            if proc.poll() is None:
                proc.kill()
        if proc is not None:
            proc.wait()
        self._close_log()

    async def restart(self) -> None:
        """§5 C4's rotation: stop this process, start a fresh one, return when
        it is healthy. Refuses outright if the scaffold does not own it."""
        if not self.owned:
            raise ServerRotationError(
                f"the scaffold does not own the server on port "
                f"{self.server_cfg.port} (it was launched outside `rlm run`), "
                "so it cannot be rotated: starting a second one would bind-fail "
                "while the first keeps answering /health, and the scaffold would "
                "resume with a fresh slot pool against slots that have all held "
                "documents (R13). Launch it with `rlm run --launch-leaf`.")
        await self.stop()
        await self.start()

    # -- internals ------------------------------------------------------ #

    async def _await_health(self) -> None:
        deadline = time.monotonic() + self._start_timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ServerRotationError(
                    f"the server exited with code {self._proc.returncode} "
                    f"before answering /health; see {self.server_cfg.log_path}")
            if await self._health():
                return
            await asyncio.sleep(self._poll_s)
        raise ServerRotationError(
            f"the server on port {self.server_cfg.port} did not report health "
            f"within {self._start_timeout_s}s; see {self.server_cfg.log_path}")

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None
