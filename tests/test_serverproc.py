"""Who owns a model-server process (spec §5 C4 v0.2.6, §4).

The scaffold does not launch servers today -- `rlm validate` probes them --
and R13's mitigation needs one to be replaced mid-episode. These tests pin the
seam: launch flags come from config (never from code), the manager refuses to
"rotate" a process it does not own (which would start a second server on a
taken port, leave the first one answering /health, and let the scaffold
believe it holds a virgin pool while every slot is used), and a start that
never becomes healthy fails loudly without leaving an orphan.
"""
import sys

import pytest

from rlm.config import Config
from rlm.errors import ServerRotationError
from rlm.serverproc import LlamaServerProcess, launch_argv

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

SLEEPER = "import time; time.sleep(30)"


async def always_healthy() -> bool:
    return True


async def never_healthy() -> bool:
    return False


def test_launch_flags_come_from_config_not_from_code(valid_cfg: Config):
    argv = launch_argv(valid_cfg.servers.leaf)
    joined = " ".join(argv)
    leaf = valid_cfg.servers.leaf
    assert argv[0].endswith("llama-server.exe")
    assert str(leaf.backend_dir) in argv[0]
    assert f"-m {leaf.model}" in joined
    assert f"--port {leaf.port}" in joined
    assert f"-c {leaf.ctx}" in joined
    assert f"-np {leaf.parallel}" in joined
    assert f"-ctk {leaf.cache_type}" in joined and f"-ctv {leaf.cache_type}" in joined
    assert f"-fa {leaf.flash_attn}" in joined
    assert f"-ub {leaf.ub}" in joined and f"-b {leaf.b}" in joined
    # D27: the cache-type assertion parses the `-lv 4` launch log, so the log
    # level is part of the launch contract, not an operator preference.
    assert "-lv 4" in joined
    # extra_flags arrive as written in config, split into argv words.
    for flag in leaf.extra_flags:
        for word in flag.split():
            assert word in argv


def test_the_launch_command_carries_no_value_config_does_not_have(valid_cfg: Config):
    """A flag invented here would be a flag `config_snapshot` cannot record,
    so a measured run could not say what it ran with (R11)."""
    leaf = valid_cfg.servers.leaf
    known = {"--host", "127.0.0.1", "-lv", "4", "-m", "--port", "-c", "-np",
             "-ctk", "-ctv", "-fa", "-ub", "-b", str(leaf.model), str(leaf.port),
             str(leaf.ctx), str(leaf.parallel), leaf.cache_type, leaf.flash_attn,
             str(leaf.ub), str(leaf.b)}
    known |= {w for flag in leaf.extra_flags for w in flag.split()}
    assert set(launch_argv(leaf)[1:]) <= known


async def test_rotating_a_process_the_scaffold_does_not_own_is_refused(valid_cfg: Config):
    """The failure this prevents is the worst one available: spawn a second
    server on a taken port, watch it die on bind, poll /health, get a 200 from
    the ORIGINAL process -- and resume with a fresh pool against a server whose
    slots have all held documents."""
    proc = LlamaServerProcess(valid_cfg.servers.leaf, health_probe=always_healthy)
    assert not proc.owned
    with pytest.raises(ServerRotationError, match="does not own"):
        await proc.restart()


async def test_start_stop_and_restart_replace_the_child(valid_cfg: Config):
    proc = LlamaServerProcess(valid_cfg.servers.leaf, health_probe=always_healthy,
                              argv=[sys.executable, "-c", SLEEPER])
    await proc.start()
    try:
        first = proc.pid
        assert first and proc.owned
        await proc.restart()
        assert proc.pid and proc.pid != first
    finally:
        await proc.stop()
    assert proc.pid is None and not proc.owned


async def test_a_start_that_never_becomes_healthy_fails_and_leaves_no_orphan(
        valid_cfg: Config):
    proc = LlamaServerProcess(valid_cfg.servers.leaf, health_probe=never_healthy,
                              argv=[sys.executable, "-c", SLEEPER],
                              start_timeout_s=0.5, poll_s=0.1)
    with pytest.raises(ServerRotationError, match="health"):
        await proc.start()
    assert proc.pid is None


async def test_the_launch_log_is_where_config_says(valid_cfg: Config, tmp_path):
    """`rlm.cli.parse_launch_log` reads cache types out of this file; a server
    launched without it is UNVERIFIED, which `rlm validate` treats as a
    refusal."""
    raw = valid_cfg.model_copy(deep=True)
    raw.servers.leaf.log_path = tmp_path / "leaf-server.log"
    up = tmp_path / "up"

    async def healthy() -> bool:
        return up.exists()

    proc = LlamaServerProcess(
        raw.servers.leaf, health_probe=healthy, poll_s=0.05,
        argv=[sys.executable, "-c",
              "import sys, pathlib, time; "
              "print('hello', file=sys.stderr, flush=True); "
              f"pathlib.Path({str(up)!r}).write_text('up'); time.sleep(30)"])
    await proc.start()          # returns only once the probe says healthy
    await proc.stop()
    assert "hello" in (tmp_path / "leaf-server.log").read_text(encoding="utf-8")


async def test_the_config_env_block_reaches_the_child_merged_over_os_environ(
        valid_cfg: Config, tmp_path):
    """`servers.<role>.env` is a launch value like any other flag: it lives in
    config so `config_snapshot` can record it (R11). MERGED over `os.environ`,
    not substituted for it -- a child launched with a two-entry environment has
    no PATH and cannot load a single backend DLL."""
    cfg = valid_cfg.model_copy(deep=True)
    cfg.servers.leaf.log_path = tmp_path / "leaf-server.log"
    cfg.servers.leaf.env = {"RLM_TEST_ENV": "hipblaslt-on"}
    up = tmp_path / "up"

    async def healthy() -> bool:
        return up.exists()

    proc = LlamaServerProcess(
        cfg.servers.leaf, health_probe=healthy, poll_s=0.05,
        argv=[sys.executable, "-c",
              "import os, sys, pathlib, time; "
              "print(os.environ.get('RLM_TEST_ENV'), 'PATH' in os.environ, "
              "file=sys.stderr, flush=True); "
              f"pathlib.Path({str(up)!r}).write_text('up'); time.sleep(30)"])
    await proc.start()
    await proc.stop()
    assert "hipblaslt-on True" in (tmp_path / "leaf-server.log").read_text(
        encoding="utf-8")


async def test_an_explicit_env_argument_still_overrides_the_config_block(
        valid_cfg: Config):
    """The `env=` parameter stays an override (s2's harnesses build their own),
    and a config with no env block leaves the child inheriting os.environ."""
    cfg = valid_cfg.model_copy(deep=True)
    cfg.servers.leaf.env = {"RLM_TEST_ENV": "from-config"}
    proc = LlamaServerProcess(cfg.servers.leaf, health_probe=always_healthy,
                              env={"RLM_TEST_ENV": "from-caller"})
    assert proc.env == {"RLM_TEST_ENV": "from-caller"}
    cfg.servers.leaf.env = {}
    assert LlamaServerProcess(cfg.servers.leaf,
                              health_probe=always_healthy).env == {}
