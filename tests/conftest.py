"""Shared fixtures. `minimal_cfg_dict` / `valid_cfg` are read from the repo's
real config.yaml so the test suite and the shipped config never drift apart.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from rlm.config import Config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def minimal_cfg_dict() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(raw)


@pytest.fixture
def valid_cfg(minimal_cfg_dict: dict) -> Config:
    return Config.model_validate(minimal_cfg_dict)


def _in_process_pair():
    """Wire two BridgeEndpoint instances together over two os.pipe() pairs,
    so the bridge's framing/correlation logic is testable without spawning
    a real sandbox process. Must be called from inside a running event loop
    (every test using it is an async def, per pytest-asyncio's auto mode).
    """
    from rlm.bridge import BridgeEndpoint

    loop = asyncio.get_running_loop()
    p2c_r, p2c_w = os.pipe()  # parent writes, child reads
    c2p_r, c2p_w = os.pipe()  # child writes, parent reads
    parent = BridgeEndpoint(c2p_r, p2c_w, loop=loop, tag="parent")
    child = BridgeEndpoint(p2c_r, c2p_w, loop=loop, tag="child")
    return parent, child


# --------------------------------------------------------------------------- #
# C1 sandbox fixtures
#
# TASK 9 ONLY (controller ruling): `_spawn_test_sandbox` below is a TEST-ONLY
# stand-in for the SandboxManager, which does not exist until Task 10. Task 10
# replaces this helper's BODY with the real `rlm.sandbox.manager.SandboxManager`
# and leaves the `session` fixture NAME unchanged, so Task 9's tests keep
# passing untouched.
# --------------------------------------------------------------------------- #

# D7 / Cross-check Conflict 8: the AppContainer is denied the repo directory,
# so the child cannot be executed in place -- it is staged into a dedicated
# bootstrap directory that holds nothing but the child and the two stdlib-only
# modules it imports, and only that directory gets the ALL APPLICATION PACKAGES
# grant. config.yaml, prompts/ and traces/ stay unreadable.
BOOTSTRAP_DIR = REPO_ROOT / "sandbox_bootstrap"
ALL_APPLICATION_PACKAGES = "S-1-15-2-1"


def bootstrap_acl_command(path: Path) -> list[str]:
    return ["icacls", str(path), "/grant",
            f"*{ALL_APPLICATION_PACKAGES}:(OI)(CI)(RX)", "/T", "/C", "/Q"]


def stage_bootstrap(dest: Path = BOOTSTRAP_DIR, *, grant_acl: bool = False) -> Path:
    """Copy the child and its (stdlib-only) imports into the bootstrap dir."""
    (dest / "rlm").mkdir(parents=True, exist_ok=True)
    pairs = [
        (REPO_ROOT / "rlm" / "sandbox" / "child.py", dest / "sandbox_child.py"),
        (REPO_ROOT / "rlm" / "__init__.py", dest / "rlm" / "__init__.py"),
        (REPO_ROOT / "rlm" / "errors.py", dest / "rlm" / "errors.py"),
        (REPO_ROOT / "rlm" / "bridge.py", dest / "rlm" / "bridge.py"),
    ]
    for src, dst in pairs:
        data = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            dst.write_bytes(data)
    if grant_acl:
        subprocess.run(bootstrap_acl_command(dest), capture_output=True,
                       text=True, check=False)
    return dest


@pytest.fixture(scope="session")
def bootstrap_dir() -> Path:
    """One-time, EXPLICIT install step (never done silently by the runtime):
    stage the child and grant ALL APPLICATION PACKAGES read+execute on that one
    directory. Granting to S-1-15-2-1 -- never to a per-episode AppContainer SID
    -- is what keeps a fresh per-episode container cheap without accruing one
    dead ACE per episode (Cross-check Conflict 8).
    """
    if sys.platform != "win32":
        pytest.skip("Windows only")
    return stage_bootstrap(grant_acl=True)


class _ChildSession:
    """Minimal driver for one spawned sandbox: enough to exercise the child
    protocol, and no more. Task 10's SandboxSession supersedes it."""

    def __init__(self, bridge, job, appcontainer, hprocess, pid):
        self._bridge = bridge
        self._job = job
        self._ac = appcontainer
        self._hprocess = hprocess
        self.pid = pid
        self.finals: list = []
        self._exit_code: int | None = None

    async def setvar(self, name: str, value) -> None:
        await self._bridge.request("setvar", {"name": name, "value": value})

    async def exec_cell(self, cell: str):
        from rlm.truncate import CellOutput

        r = await self._bridge.request("exec", {"cell": cell})
        return CellOutput(stdout=r.get("stdout") or "", stderr=r.get("stderr") or "",
                          repr_=r.get("repr") or "", traceback=r.get("traceback") or "")

    async def close(self) -> int:
        from rlm.sandbox import winproc

        if self._exit_code is not None:
            return self._exit_code
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._bridge.request("shutdown", None), 10)
        code = await _await_exit(self._hprocess, 10.0)
        if code is None:
            self._job.terminate(0xB0DE)
            code = await _await_exit(self._hprocess, 5.0) or 0xB0DE
        self._exit_code = code
        self._bridge.close()
        self._job.close()
        self._ac.delete()
        winproc.kernel32.CloseHandle(self._hprocess)
        return code


async def _await_exit(hprocess: int, timeout_s: float) -> int | None:
    """Poll GetExitCodeProcess on the loop -- no executor thread, no blocking."""
    import ctypes
    from ctypes import wintypes

    from rlm.sandbox import winproc

    deadline = asyncio.get_running_loop().time() + timeout_s
    code = wintypes.DWORD()
    while True:
        winproc.kernel32.GetExitCodeProcess(hprocess, ctypes.byref(code))
        if code.value != 259:  # STILL_ACTIVE
            return code.value
        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def _spawn_test_sandbox(cfg, bootstrap: Path, episode_id: str):
    import msvcrt
    import _winapi

    from rlm.bridge import BridgeParent
    from rlm.errors import SandboxError
    from rlm.sandbox import winproc
    from rlm.sandbox.winjob import Job

    sbx = cfg.scaffold.sandbox
    loop = asyncio.get_running_loop()

    # two anonymous pipes; only the child's ends are inheritable
    c_in_r, p_in_w = _winapi.CreatePipe(None, 0)    # parent -> child
    p_out_r, c_out_w = _winapi.CreatePipe(None, 0)  # child -> parent
    os.set_handle_inheritable(c_in_r, True)
    os.set_handle_inheritable(c_out_w, True)
    errf = open(os.devnull, "wb")
    herr = msvcrt.get_osfhandle(errf.fileno())
    os.set_handle_inheritable(herr, True)

    job = Job(memory_limit_mb=sbx.memory_limit_mb,
              active_process_limit=sbx.active_process_limit)
    ac = winproc.AppContainer()
    sid = ac.create(f"rlmhalo.{episode_id}.{uuid.uuid4().hex[:8]}")
    try:
        args = ["-B", "-I", "-u", str(bootstrap / "sandbox_child.py")]
        if sbx.deny_ctypes:
            args.append("--deny-ctypes")
        res = winproc.spawn(str(sbx.interpreter), args, [], sid, job, None,
                            winproc.Stdio(c_in_r, c_out_w, herr),
                            cwd=str(bootstrap))
    finally:
        _winapi.CloseHandle(c_in_r)
        _winapi.CloseHandle(c_out_w)
        errf.close()
    winproc.kernel32.CloseHandle(res.hthread)

    rfd = msvcrt.open_osfhandle(p_out_r, os.O_RDONLY | os.O_BINARY)
    wfd = msvcrt.open_osfhandle(p_in_w, os.O_BINARY)
    bridge = BridgeParent(rfd, wfd, loop=loop, tag=f"scaffold:{episode_id}")
    handshake: asyncio.Future = loop.create_future()

    async def _serve(kind, payload):
        if kind == "handshake":
            if not handshake.done():
                handshake.set_result(payload["pid"])
            return None
        if kind == "llm_query":
            return f"MOCK:{payload['prompt']}"
        if kind == "final_answer":
            session.finals.append(payload["value"])
            return None
        if kind == "bye":
            return None
        raise SandboxError(f"unknown child frame kind: {kind!r}")

    bridge.on_request(_serve)
    session = _ChildSession(bridge, job, ac, res.hprocess, res.pid)
    try:
        child_pid = await asyncio.wait_for(handshake, 60)
        if child_pid != res.pid:  # D23
            raise SandboxError(
                f"handshake pid {child_pid} != spawned pid {res.pid}")
        yield session
    finally:
        with contextlib.suppress(Exception):
            await session.close()


@pytest.fixture
async def session(valid_cfg, bootstrap_dir):
    async with _spawn_test_sandbox(valid_cfg, bootstrap_dir, "t9") as s:
        yield s
