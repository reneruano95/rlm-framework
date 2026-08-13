"""Shared fixtures. `minimal_cfg_dict` / `valid_cfg` are read from the repo's
real config.yaml so the test suite and the shipped config never drift apart.
"""
from __future__ import annotations

import asyncio
import copy
import os
import sys
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


@pytest.fixture
def cfg(valid_cfg: Config) -> Config:
    return valid_cfg


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
# `session` spawns a REAL sandbox (Job + AppContainer + bridge) through the
# SandboxManager, with a mock dispatcher answering every llm_query -- C1
# exercised without C4. These tests are slow by construction and Windows-only.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def bootstrap_dir() -> Path:
    """The one-time, EXPLICIT install step: stage the child into its dedicated
    directory and grant ALL APPLICATION PACKAGES read+execute on that directory
    ONLY. The runtime itself never grants an ACL -- `install_bootstrap` defaults
    to `grant_acl=False` -- so the sandbox's inability to read config.yaml,
    prompts/ and traces/ stays a property of the repo, not of a lucky default.
    """
    if sys.platform != "win32":
        pytest.skip("Windows only")
    from rlm.sandbox.manager import install_bootstrap

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    sandbox_cfg = Config.model_validate(raw).scaffold.sandbox
    return install_bootstrap(sandbox_cfg, grant_acl=True)


@pytest.fixture
def manager(bootstrap_dir: Path):
    from rlm.sandbox.manager import SandboxManager

    m = SandboxManager(stderr_dir=os.environ.get("RLM_SANDBOX_STDERR_DIR"))
    yield m
    m.close()


async def mock_llm_query(payload: dict) -> str:
    """The mock dispatcher: answers every sub-call without touching a server."""
    return f"MOCK:{payload['prompt']}"


@pytest.fixture
async def session(manager, cfg: Config):
    async with manager.session("child-tests", cfg) as s:
        s.on_llm_query(mock_llm_query)
        yield s
