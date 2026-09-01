# checks/test_cli.py
import asyncio
import io
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
import yaml

from rlm import cli
from rlm.cli import main
from rlm.config import load_config
from rlm.errors import ActionType, Actor, Outcome, StepStatus
from rlm.trace.lifecycle import Lifecycle
from rlm.trace import TraceLogger, utc_now

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def test_validate_refuses_a_bad_config(tmp_path, capsys):
    (tmp_path / "config.yaml").write_text("scaffold:\n  budgets:\n    max_subcals: 32\n")
    rc = main(["validate", "--config", str(tmp_path / "config.yaml")])
    assert rc != 0
    assert "max_subcals" in capsys.readouterr().err


def test_validate_asserts_the_sandbox_cannot_read_the_repo(tmp_path, capsys, valid_config_file):
    """D7: turn the filesystem-confinement claim into a checked invariant."""
    rc = main(["validate", "--config", str(valid_config_file), "--no-server-probe"])
    out = capsys.readouterr().out
    assert "sandbox filesystem confinement: OK" in out
    assert rc == 0


def test_run_prints_episode_id_and_outcome(mock_episode_env, capsys):
    rc = main(["run", str(mock_episode_env.task_file), "--config",
               str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "episode_id" in out and "success" in out


def test_run_exports_a_bundle_scoped_to_the_episode(mock_episode_env):
    """D21: the bundle is written at episode close. Scoped to THIS episode —
    an unscoped export at every close is O(total episodes) per episode, which
    makes a multi-hour bench quadratic. It must still be self-contained: a
    foreign reader joins steps to blobs with no .duckdb in sight."""
    import duckdb

    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    bundle = mock_episode_env.tmp_path / "bundle" / ep
    assert bundle.is_dir()

    con = duckdb.connect()          # in-memory: no lock, no live handle
    try:
        episodes = con.execute(
            f"SELECT episode_id, outcome FROM '{bundle / 'episodes.parquet'}'"
        ).fetchall()
        assert [(str(e), o) for e, o in episodes] == [(ep, "success")]
        # Every blob referenced by a step resolves inside this bundle alone.
        dangling = con.execute(
            f"SELECT count(*) FROM '{bundle / 'steps.parquet'}' s "
            f"LEFT JOIN '{bundle / 'blobs.parquet'}' b "
            f"  ON b.rel = s.observation_full_ref "
            f"WHERE s.observation_full_ref IS NOT NULL AND b.rel IS NULL"
        ).fetchone()[0]
        assert dangling == 0
    finally:
        con.close()


def test_replay_offline_verifies_hashes_with_no_server(mock_episode_env, capsys):
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "root_view_hash: OK" in out


def test_replay_fails_loudly_on_a_tampered_blob(mock_episode_env, capsys):
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    mock_episode_env.tamper_root_request_blob(ep)
    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    assert rc != 0
    assert "hash mismatch" in capsys.readouterr().err


def test_replay_works_with_the_lifecycle_log_deleted(mock_episode_env):
    """S3 gate condition, exercised early: no logs, no stdout, trace store only."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    mock_episode_env.delete_lifecycle_log()
    assert main(["replay", mock_episode_env.last_episode_id(), "--config",
                 str(mock_episode_env.config_file)]) == 0


# --------------------------------------------------------------------------- #
# Added beyond the brief: the message-array canary (the half of the state rule
# a blob rehash cannot catch), and the non-goals staying non-goals.
# --------------------------------------------------------------------------- #


def test_replay_rederives_the_message_array_from_the_trace_alone(mock_episode_env, capsys):
    """The state rule's real canary: the blob rehash proves the blob is intact,
    this proves TODAY's prompt assembly still reproduces the stored array."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    rc = main(["replay", mock_episode_env.last_episode_id(), "--config",
               str(mock_episode_env.config_file)])
    assert rc == 0
    assert "message array: OK" in capsys.readouterr().out


def test_replay_online_byte_compares_against_apply_template(mock_episode_env, capsys):
    """Mode (ii). Note what this does NOT do: it never re-generates anything.
    Greedy decoding is not reproducible on this box (measured: 3 identical
    requests, temperature 0, fixed seed, 3 different outputs), so replay
    verifies prompt assembly and nothing else."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    rc = main(["replay", mock_episode_env.last_episode_id(), "--online",
               "--config", str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "apply-template byte-equality: OK" in out
    assert "chat_template sha256: OK" in out


def test_replay_uses_the_stored_snapshot_not_the_live_config(mock_episode_env, capsys):
    """Editing config.yaml after a run must NOT be reported as prompt-assembly
    drift. `max_subcalls` feeds the user message D26 composes, so re-deriving
    against the live file would raise a false alarm on the one instrument whose
    job is spotting real drift — and an alarm that cries wolf is worse than no
    alarm."""
    import yaml

    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()

    raw = yaml.safe_load(mock_episode_env.config_file.read_text(encoding="utf-8"))
    assert raw["scaffold"]["budgets"]["max_subcalls"] != 3
    raw["scaffold"]["budgets"]["max_subcalls"] = 3
    raw["scaffold"]["truncation_cap_chars"] = 1500
    mock_episode_env.config_file.write_text(yaml.safe_dump(raw, sort_keys=False),
                                             encoding="utf-8")

    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    out = capsys.readouterr().out
    assert rc == 0, "a live-config edit must not read as prompt-assembly drift"
    assert "message array: OK" in out
    assert "prompt hashes: OK" in out


def test_replay_reports_a_changed_prompt_as_prompt_drift(mock_episode_env, capsys, tmp_path):
    """A genuine prompt change is real drift — but it must be NAMED as a prompt
    change, not disguised as assembly drift or as a config error."""
    import yaml

    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()

    raw = yaml.safe_load(mock_episode_env.config_file.read_text(encoding="utf-8"))
    edited = tmp_path / "root.edited.md"
    original = Path(raw["scaffold"]["prompts"]["root"]["path"])
    edited.write_text(original.read_text(encoding="utf-8") + "\nAN EXTRA LINE.\n",
                       encoding="utf-8")
    # Repoint the STORED snapshot's prompt path at the edited file by editing
    # the episode row, which is what a prompt file changing under a pinned
    # path amounts to for replay.
    import json

    import duckdb
    con = duckdb.connect(str(mock_episode_env.db_path))
    try:
        snap = json.loads(con.execute(
            "SELECT config_snapshot FROM episodes WHERE episode_id = ?",
            [ep]).fetchone()[0])
        snap["scaffold"]["prompts"]["root"]["path"] = str(edited)
        snap["scaffold"]["prompts"]["root"]["sha256"] = None
        con.execute("UPDATE episodes SET config_snapshot = ? WHERE episode_id = ?",
                    [json.dumps(snap), ep])
    finally:
        con.close()

    rc = main(["replay", ep, "--config", str(mock_episode_env.config_file)])
    err = capsys.readouterr().err
    assert rc != 0
    assert "prompt drift" in err
    assert "prompt-assembly drift" not in err.split("prompt drift")[0]


def test_validate_refuses_when_the_cache_type_is_unverified(valid_config_file, capsys,
                                                             monkeypatch):
    """A validation gate that goes green without asserting is worse than no
    gate: it converts 'never checked' into 'fine'. UNVERIFIED must be a
    refusal whenever a probe was actually requested."""
    from rlm import cli

    async def fake_probe(cfg, lifecycle, out):
        return {"root": {"build_info": "10375 (ba360efe1)"},
                "leaf": {"build_info": "10375 (ba360efe1)"}}

    monkeypatch.setattr(cli, "_probe_servers", fake_probe)
    # No launch logs exist, so the cache-type assertion cannot be made.
    rc = main(["validate", "--config", str(valid_config_file)])
    captured = capsys.readouterr()
    assert rc != 0
    assert "UNVERIFIED" in captured.err
    assert "Refusing to start" in captured.err
    # …and the same state is tolerated only when the operator opted out.
    assert main(["validate", "--config", str(valid_config_file),
                 "--no-server-probe"]) == 0


def _insert_orphan(db_path, *, pid, started_at, episode_id=None):
    """A NULL-outcome episode row: what a crashed run leaves behind."""
    import uuid as _uuid

    import duckdb

    from rlm.trace import _schema_sql

    episode_id = episode_id or str(_uuid.uuid4())
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_schema_sql())
        con.execute(
            "INSERT INTO episodes (episode_id, task_id, task_hash, started_at, "
            "sandbox_pid) VALUES (?, ?, ?, ?, ?)",
            [episode_id, "t", "h", started_at, pid])
    finally:
        con.close()
    return episode_id


def test_recovery_reaps_then_quiesces_then_tombstones(mock_episode_env, monkeypatch):
    """§6 crash recovery. `recover_orphans` is DB-only by design; this is the
    half the CLI owns, and the ORDER is the contract: a surviving sandbox must
    be dead and both servers drained BEFORE the row is tombstoned, or recovery
    tombstones an episode that is still generating."""
    import datetime as dt

    from rlm import cli

    started = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    ep = _insert_orphan(mock_episode_env.db_path, pid=4242, started_at=started)

    order = []
    monkeypatch.setattr(cli.winproc, "kill_if_ours",
                        lambda pid, at, **kw: order.append(("reap", pid)) or True)

    async def fake_quiesce(cfg, lifecycle):
        order.append(("quiesce", None))

    monkeypatch.setattr(cli, "_quiesce_all", fake_quiesce)
    real_tombstone = cli.recover_orphans
    monkeypatch.setattr(cli, "recover_orphans",
                        lambda db, lc=None: order.append(("tombstone", None))
                        or real_tombstone(db, lc))

    cfg = load_config(mock_episode_env.config_file)
    lifecycle = Lifecycle(None, stream=io.StringIO())
    assert cli.recover(cfg, lifecycle) == [ep]
    assert [kind for kind, _ in order] == ["reap", "quiesce", "tombstone"]
    assert order[0][1] == 4242

    import duckdb
    con = duckdb.connect(str(mock_episode_env.db_path), read_only=True)
    try:
        assert con.execute("SELECT outcome, outcome_reason FROM episodes WHERE "
                           "episode_id = ?", [ep]).fetchone() == (
            "error", "orphaned_at_recovery")
    finally:
        con.close()


def test_recovery_refuses_to_kill_a_reused_pid(mock_episode_env):
    """The creation-time guard is the whole reason `kill_if_ours` exists: pid
    reuse is real, and a scaffold that damages an unrelated process during its
    own cleanup is far worse than one orphaned sandbox. Here the episode claims
    to have started a year ago, so THIS still-running process (our own pid,
    created moments ago) must not be touched."""
    import datetime as dt
    import os

    from rlm import cli

    long_ago = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=365)
    ep = _insert_orphan(mock_episode_env.db_path, pid=os.getpid(), started_at=long_ago)

    cfg = load_config(mock_episode_env.config_file)
    lifecycle = Lifecycle(None, stream=io.StringIO())
    assert cli.recover(cfg, lifecycle) == [ep]
    # Still alive: the guard refused it. (If it had not, this process would be
    # gone and there would be no assertion to run.)
    assert os.getpid() > 0
    # And it is reported as NOT reaped, so the operator learns the sandbox
    # outlived its episode rather than being told it was cleaned up.
    assert cli.winproc.kill_if_ours(os.getpid(), long_ago) is False


def test_launch_log_parse_reads_the_real_lv4_lines(tmp_path):
    """D27: /props cannot report cache types, so these two lines are the whole
    assertion. The sample is the verbatim shape a b10375 root server emits."""
    from rlm.cli import log_is_current, parse_launch_log

    log = tmp_path / "root-server.log"
    log.write_text(
        "build: 10375 (ba360efe1) with clang for x86_64-pc-windows-msvc\n"
        "llama_context: flash_attn = enabled\n"
        "llama_kv_cache: size =  1088.00 MiB (  32768 cells,  16 layers, "
        "1/1 seqs), K (q8_0):  544.00 MiB, V (q8_0):  544.00 MiB\n",
        encoding="utf-8")
    parsed = parse_launch_log(log)
    assert parsed["type_k"] == "q8_0" and parsed["type_v"] == "q8_0"
    assert parsed["flash_attn"] == "enabled"
    assert parsed["kv_cells"] == 32768 and parsed["kv_layers"] == 16
    # A stale log from a previous launch must not satisfy the assertion.
    assert log_is_current(parsed, {"build_info": "10375 (ba360efe1)"}) is True
    assert log_is_current(parsed, {"build_info": "10999 (deadbeef)"}) is False
    assert log_is_current(parsed, None) is False
    assert parse_launch_log(tmp_path / "absent.log") == {}


def test_launch_log_parse_reads_the_build_line_b10375_ACTUALLY_prints(tmp_path):
    """S1, measured on-box: b10375 prints no separator after `build`, and
    /props reports `b10375-ba360efe1` rather than `10375 (ba360efe1)`. The
    sample above was written from the older upstream shape; these three lines
    are copied verbatim out of a live `-lv 4` leaf log, so a future rewrite
    cannot regress the parse to the shape that fails closed on a good server.
    """
    from rlm.cli import log_is_current, parse_launch_log

    log = tmp_path / "leaf-server.log"
    log.write_text(
        "0.00.361.126 I cmn  common_param: common_params_print_info: build "
        "10375 (ba360efe1) with Clang 20.1.8 for Windows x86_64\n"
        "0.06.080.326 I llama_context: flash_attn            = enabled\n"
        "0.06.170.335 I llama_kv_cache: size = 3400.00 MiB ( 40960 cells,  10 "
        "layers,  8/8 seqs), K (q8_0): 1700.00 MiB, V (q8_0): 1700.00 MiB\n",
        encoding="utf-8")
    parsed = parse_launch_log(log)
    assert parsed["build_number"] == "10375" and parsed["build_commit"] == "ba360efe1"
    assert parsed["type_k"] == "q8_0" and parsed["type_v"] == "q8_0"
    assert parsed["flash_attn"] == "enabled"
    assert parsed["kv_cells"] == 40960 and parsed["kv_seqs"] == 8
    assert log_is_current(parsed, {"build_info": "b10375-ba360efe1"}) is True


def test_launch_log_parse_binds_to_the_target_not_the_drafter(tmp_path):
    """A speculative launch (`-md`, milestones/s2/DFLASH2.md) builds TWO contexts, so the
    log carries two kv_cache lines and two flash_attn lines -- target first,
    drafter second. This parse used to take the LAST of each, so attaching a
    drafter silently moved §4's assertion off the target and onto the draft
    cache, which defaults to f16 and would then fail the q8_0 check for the
    wrong reason. Sample lines are copied verbatim from a live DFlash2 root."""
    from rlm.cli import parse_launch_log

    log = tmp_path / "root-server.log"
    log.write_text(
        "0.00.36 I cmn  common_param: common_params_print_info: build "
        "10375 (ba360efe1) with Clang 20.1.8 for Windows x86_64\n"
        "0.10.63 I llama_context: flash_attn            = enabled\n"
        "0.10.79 I llama_kv_cache: size = 1088.00 MiB ( 32768 cells,  16 "
        "layers,  1/1 seqs), K (q8_0):  544.00 MiB, V (q8_0):  544.00 MiB\n"
        # the drafter's own context, logged after the target's
        "0.13.19 I llama_kv_cache: size =    0.00 MiB ( 32768 cells,   0 "
        "layers,  1/1 seqs), K (f16):    0.00 MiB, V (f16):    0.00 MiB\n"
        "0.13.19 I llama_context: flash_attn            = enabled\n"
        "0.13.19 I llama_kv_cache: size =   50.00 MiB (  2560 cells,   5 "
        "layers,  1/1 seqs), K (f16):   25.00 MiB, V (f16):   25.00 MiB\n",
        encoding="utf-8")
    parsed = parse_launch_log(log)
    # the target, which is what §4 asserts on
    assert parsed["type_k"] == "q8_0" and parsed["type_v"] == "q8_0"
    assert parsed["kv_cells"] == 32768 and parsed["kv_layers"] == 16
    assert parsed["flash_attn"] == "enabled"
    # the drafter, recorded rather than silently shadowing the target. The
    # zero-layer line is a context llama.cpp never allocated and must not be
    # mistaken for the draft cache.
    assert parsed["draft_type_k"] == "f16" and parsed["draft_kv_layers"] == 5
    assert parsed["draft_kv_mib"] == 50.0


def test_the_cli_has_exactly_five_verbs():
    """Non-goals are written into the spec: no daemon, no REST API, no web UI,
    no interactive chat mode.

    THREE became FIVE, and that is the whole renegotiation. `src/rlm/cli.py:7` named
    `bench` and `export` as "later slices (S4)" from the start, so S4 landing
    them is the argued-for change this test was always going to record — a
    SIXTH verb appearing here still means a slice shipped without arguing
    for it.
    """
    import argparse

    from rlm.cli import build_parser

    sub = [a for a in build_parser()._actions
           if isinstance(a, argparse._SubParsersAction)]
    assert len(sub) == 1
    assert set(sub[0].choices) == {"validate", "run", "replay", "bench", "export"}


# --------------------------------------------------------------------------- #
# Who owns the leaf process (spec §5 C4, v0.2.6).
#
# A rotation replaces a process, and the scaffold can only replace one it
# started. The manager is built HERE, in the process root, so the launch flags
# stay in config.yaml (where `config_snapshot` reads them from) and C4 keeps
# having no code path that restarts a server.
# --------------------------------------------------------------------------- #


def test_a_dry_run_has_no_leaf_process_to_rotate(valid_config_file):
    from rlm.cli import leaf_process_manager

    cfg = load_config(valid_config_file)
    mock = cfg.model_copy(deep=True)
    mock.scaffold.dispatcher = "mock"
    assert leaf_process_manager(mock, launch=True) is None


def test_without_launch_leaf_the_run_owns_nothing_and_says_so_once(valid_config_file):
    """Default: the servers were launched outside `rlm run`. Returning None
    beats returning a manager that refuses every rotation -- the refusal is a
    property of the run, not of each exhausted pool."""
    from rlm.cli import leaf_process_manager

    cfg = load_config(valid_config_file)
    assert cfg.scaffold.dispatcher == "real"
    assert leaf_process_manager(cfg, launch=False) is None


def test_launch_leaf_builds_the_manager_from_config_flags(valid_config_file):
    from rlm.cli import leaf_process_manager
    from rlm.serve.serverproc import launch_argv

    cfg = load_config(valid_config_file)
    manager = leaf_process_manager(cfg, launch=True)
    assert manager is not None
    assert manager.argv == launch_argv(cfg.servers.leaf)
    assert not manager.owned          # nothing started yet; `run` starts it


def test_launch_leaf_carries_the_config_env_to_the_child(valid_config_file):
    """`servers.leaf.env` is merged over os.environ at launch. The CLI passed
    `env=None` here, which silently dropped ROCBLAS_USE_HIPBLASLT -- the
    variable every S2 leaf measurement was taken with (`milestones/s2/run_occupancy.py`
    :455) -- so a `--launch-leaf` S4 block would have been compared against
    numbers from a differently configured BLAS.

    v0.3.22 moved the leaf to Vulkan and deleted that variable: it is a rocBLAS
    setting that does nothing on Vulkan, and `config_snapshot` records this
    block, so a snapshot naming a library the server never loads is a false
    record of what ran. The variable is gone; THE DEFECT IT GUARDED IS NOT, so
    this now asserts the mechanism rather than the value -- an empty
    declaration must arrive empty rather than picking up a default, and a
    non-empty one must arrive intact."""
    from rlm.cli import leaf_process_manager

    cfg = load_config(valid_config_file)
    manager = leaf_process_manager(cfg, launch=True)
    assert manager is not None
    assert manager.env == dict(cfg.servers.leaf.env)
    assert "ROCBLAS_USE_HIPBLASLT" not in manager.env, (
        "a rocBLAS setting on a Vulkan backend would be a false snapshot")

    spiked = cfg.model_copy(deep=True)
    object.__setattr__(spiked.servers.leaf, "env", {"SOME_RUNTIME_KNOB": "7"})
    spiked_manager = leaf_process_manager(spiked, launch=True)
    assert spiked_manager is not None
    assert spiked_manager.env == {"SOME_RUNTIME_KNOB": "7"}, (
        "this is the assertion that would have caught the original env=None "
        "bug, and it survives the variable that first exposed it")


def test_replay_survives_an_s4_era_snapshot_with_baseline_prompts(mock_episode_env):
    """An episode recorded with baselines loaded must replay clean: the
    episode_config rebuild includes baselines.* or PromptDrift fires on every
    S4 episode (the leaf_envelope bug class, cli.py:659-663)."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    assert main(["replay", mock_episode_env.last_episode_id(), "--config",
                 str(mock_episode_env.config_file)]) == 0


def test_replay_of_a_pre_s4_snapshot_without_the_new_keys_still_works(mock_episode_env):
    """The other half of the same trap, in the other direction.

    Every episode recorded BEFORE S4 has a snapshot with no `baselines` block,
    no `servers.bench_leaf`, no `servers.*.env` and no benchmark pins. A rebuild
    that reached for `prompts["baselines"]` unconditionally, or a schema whose
    new fields had no defaults, would turn all of them into a ConfigError --
    which is a spec violation, not a cosmetic one: the trace store is the
    experimental record and replay is how it is audited.
    """
    import json

    import duckdb

    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()

    con = duckdb.connect(str(mock_episode_env.db_path))
    try:
        snap = json.loads(con.execute(
            "SELECT config_snapshot FROM episodes WHERE episode_id = ?",
            [ep]).fetchone()[0])
        # Age the snapshot back to its pre-S4 shape: the keys S4 introduced
        # simply were not there, including the hashes the registry recorded.
        snap["scaffold"]["prompts"].pop("baselines")
        snap["servers"].pop("bench_leaf")
        for role in ("root", "leaf"):
            snap["servers"][role].pop("env")
        for key in ("manifest_sha256", "escalation_seeds"):
            snap["benchmark"].pop(key)
        snap["prompt_hashes"] = {k: v for k, v in snap["prompt_hashes"].items()
                                 if not k.startswith("baselines.")}
        con.execute("UPDATE episodes SET config_snapshot = ? WHERE episode_id = ?",
                    [json.dumps(snap), ep])
    finally:
        con.close()

    assert main(["replay", ep, "--config",
                 str(mock_episode_env.config_file)]) == 0


def test_the_recorded_prompt_hashes_include_the_baselines(mock_episode_env):
    """The half the replay test cannot see on its own: if the baselines never
    reached `prompt_hashes`, the rebuild would agree with the snapshot by both
    being empty and the trap would look shut while standing open."""
    import json

    import duckdb

    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    ep = mock_episode_env.last_episode_id()
    con = duckdb.connect(str(mock_episode_env.db_path), read_only=True)
    try:
        snap = json.loads(con.execute(
            "SELECT config_snapshot FROM episodes WHERE episode_id = ?",
            [ep]).fetchone()[0])
    finally:
        con.close()
    hashes = snap["prompt_hashes"]
    for name in ("b1_single_shot", "b2_leaf_summary", "b2_root_final",
                 "b3_single_shot"):
        assert f"baselines.{name}.file" in hashes
        assert f"baselines.{name}.body" in hashes


def test_run_takes_launch_leaf_and_defaults_to_off():
    """Off by default: taking ownership silently would kill an operator's own
    leaf server at the end of an episode."""
    from rlm.cli import build_parser

    args = build_parser().parse_args(["run", "t.json"])
    assert args.launch_leaf is False
    assert build_parser().parse_args(["run", "t.json", "--launch-leaf"]).launch_leaf


# =========================================================================== #
# S4 (Task 12): `rlm bench` and `rlm export`
#
# NO SERVERS AND NO ARMS. `cli.ServerOrchestra` and the four arm entry points
# are replaced by recorders, so what is under test is the WIRING -- which hook
# reaches which runner, which config each arm is handed, what the escalation
# phase drives, which exit code the gate produces -- rather than a model. The
# frozen manifest, the real task files, the real scheduler (`rlm.measure.bench`) and
# the real scorer (`rlm.measure.verdict`) all run for real; only the two ends that
# need a GPU are doubles.
# =========================================================================== #


class _FakeOrchestra:
    """`ServerOrchestra`'s surface with no processes and no HTTP.

    `handshake_profile` REFUSES a profile that is not the one running, which
    the real one does too and for the same reason: §4 asserts the live
    `/props` against the config for `profile`, so asking for `resident` while
    the bench-profile server is up is a `total_slots` mismatch, not a no-op.
    Modelling that is what lets a test see a scheduler running on a stale
    profile assumption at all -- a handshake double that accepted anything
    blinds every assertion downstream of it.
    """

    def __init__(self, cfg, *, launch=True, lifecycle=None, out=None, err=None,
                 **_kw):
        self.cfg = cfg
        self.launch = launch
        self.events = []
        self.current_profile = None
        self.last_relaunch_s = 0.0
        #: A stand-in for the leaf process object, so `pool_rotating_swap` can
        #: tell a real relaunch from a no-op swap the way it does in
        #: production (identity, not the reported seconds).
        self.leaf_proc = None

    async def start_resident(self):
        self.events.append(("start_resident", None))
        self.current_profile = "resident"
        self.leaf_proc = object()

    async def quiesce(self, profile):
        self.events.append(("quiesce", profile))
        return {"root": True}

    async def handshake_profile(self, profile):
        self.events.append(("handshake", profile))
        if profile != self.current_profile:
            from rlm.errors import ConfigError

            raise ConfigError(
                f"leaf /props mismatch: handshaking the {profile!r} profile "
                f"against the {self.current_profile!r} server that is running")
        return {"root": {}}

    async def swap_to(self, profile):
        self.events.append(("swap", profile))
        if profile == self.current_profile:
            return 0.0                       # already there: `_bring_up_leaf`
        self.current_profile = profile
        self.leaf_proc = object()            # a new process, a virgin pool
        self.last_relaunch_s = 11.5
        return 11.5

    class _Manager:
        """`LeafProcessManager`'s surface, recording the restart.

        Was a sentinel string until the delegating arms began restarting the
        leaf before every episode (`cli._virgin_resident_pool`): R13's pool is
        per leaf PROCESS, and a delegating arm spends all 128 slots inside one
        episode, so the next episode on that generation -- B2, a SCORED
        baseline -- opened on a drained pool and starved. Measured in
        `rlm bench --smoke`, 2026-08-20."""

        def __init__(self, orchestra, label):
            self._orchestra = orchestra
            self.label = label

        async def restart(self):
            self._orchestra.events.append(("leaf_restart", self.label))
            self._orchestra.leaf_proc = object()   # new process, virgin pool

    def episode_process_manager(self):
        return _FakeOrchestra._Manager(self, "episode")

    def b2_process_manager(self):
        return _FakeOrchestra._Manager(self, "b2")

    async def stop_all(self):
        self.events.append(("stop_all", None))

    def kinds(self, kind):
        return [p for k, p in self.events if k == kind]


class _StubSampler:
    """`PowerSampler`'s surface with no PowerShell child.

    `power_sampling.enabled` is TRUE in the shipped config (S0 measured the
    collector's overhead at noise and §8 makes energy a scorecard column), so a
    bench test that did not stub this would spawn a real 1 Hz Get-Counter child
    per test. It reports `alive() is False`, which is the honest "no readings"
    state `rlm.measure.bench._stamp_metrics` records as NULL rather than as a
    fabricated zero."""

    def __init__(self, *_a, **_kw) -> None:
        self.started = self.stopped = False
        self.stderr_path = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def alive(self) -> bool:
        return False

    def reading(self):
        return None


@pytest.fixture
def fake_orchestra(monkeypatch):
    """The no-hardware bench fixture: no server processes, no power sampler."""
    made = []

    def factory(cfg, **kw):
        orchestra = _FakeOrchestra(cfg, **kw)
        made.append(orchestra)
        return orchestra

    monkeypatch.setattr(cli, "ServerOrchestra", factory)
    monkeypatch.setattr(cli, "PowerSampler", _StubSampler)
    return made


class _ArmRecorder:
    """Stands in for `run_episode` / `run_b1` / `run_b2` / `run_b3`.

    Records everything the CLI handed each arm and, when `write=True`, opens
    and closes a REAL episode row through the injected `TraceLogger` -- so the
    scoring half of `rlm bench` runs against a store the arms actually wrote,
    with the `config_snapshot.bench` identity the grid is keyed by.
    """

    def __init__(self, outcomes=None, *, write=False):
        self.calls = []
        self._outcomes = dict(outcomes or {})
        self._write = write

    def outcome_for(self, arm, task_id):
        return self._outcomes.get((arm, task_id), Outcome.SUCCESS)

    def entry(self, fallback_arm):
        async def run(task, cfg, **kw):
            extra = kw.get("bench_extra")
            if extra is None:
                extra = dict((kw.get("snapshot_extra") or {}).get("bench") or {})
            # The CLI stamps the arm into `bench_extra`, and TWO arms now come
            # through `run_episode` -- `rlm` and `rlm-restricted`. Labelling by
            # the patched function alone would record both as "rlm" and make
            # the delegation arm invisible to every assertion here.
            arm = extra.get("arm") or fallback_arm
            self.calls.append({
                "arm": arm, "task_id": task.task_id, "seed": extra.get("seed"),
                "block": extra.get("block"), "run_id": extra.get("run_id"),
                "root_seed": cfg.scaffold.sampling.root.seed,
                "leaf_seed": cfg.scaffold.sampling.leaf.seed,
                "leaf_parallel": cfg.servers.leaf.parallel,
                "leaf_ctx": cfg.servers.leaf.ctx,
                "kwargs": kw,
            })
            outcome = self.outcome_for(arm, task.task_id)
            episode_id = str(uuid.uuid4())
            trace = kw.get("trace")
            if self._write and trace is not None:
                trace.open_episode({
                    "episode_id": episode_id, "task_id": task.task_id,
                    "task_hash": "h", "started_at": utc_now(),
                    "config_snapshot": {
                        "scaffold": {"chunk": {"size_tokens":
                                               cfg.scaffold.chunk.size_tokens}},
                        "bench": {**extra, "arm": arm}}})
                trace.close_episode(episode_id, outcome, None)
                await trace.drain()
            return SimpleNamespace(episode_id=episode_id, outcome=outcome,
                                   reason=None, answer=None, final_answer=None)

        return run

    def patch(self, monkeypatch):
        monkeypatch.setattr(cli, "run_episode", self.entry("rlm"))
        monkeypatch.setattr(cli, "run_b1", self.entry("b1"))
        monkeypatch.setattr(cli, "run_b2", self.entry("b2"))
        monkeypatch.setattr(cli, "run_b3", self.entry("b3"))
        return self

    def order(self):
        return [c["arm"] for c in self.calls]


SMOKE_TASKS = "needle-02,agg-02"
FOUR_TASKS = ("needle-02", "agg-02", "synth-01", "codeqa-01")


#: §8's four scored arms. Pinned explicitly here since `rlm-restricted` joined
#: ARM_ORDER (the delegation arm): every test below asserts GATE and ESCALATION
#: arithmetic, which is defined over RLM vs the three baselines, and letting a
#: fifth arm into the grid would change those numbers without changing anything
#: the tests are about. The new arm's registration, ordering and relaunch cost
#: are covered in `test_bench.py` and `test_delegation_arm.py` instead.
GATE_ARMS = "rlm,b1,b2,b3"


def _bench_argv(config_file, tmp_path, *extra):
    """Every path a bench run writes to, redirected into tmp: the ledger and the
    report. Both now default under `runs/` rather than into `milestones/s4/`, so
    a stray write is no longer an edit to S4's evidence -- but redirecting stays
    correct regardless: a test that writes into the repo at all is a test that
    leaves state behind."""
    argv = ["bench", "--config", str(config_file),
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--report", str(tmp_path / "RESULTS.md"), *extra]
    if "--arm" not in argv:
        argv += ["--arm", GATE_ARMS]
    return argv


# --------------------------------------------------------------------------- #
# refusals: the things that must never start a 39-hour run
# --------------------------------------------------------------------------- #


def test_bench_refuses_a_mock_dispatcher(valid_config_file, tmp_path, capsys):
    """§8 scores model behaviour. A `dispatcher: mock` grid would be 360
    fixture replays wearing the benchmark's name, and `load_grid` would refuse
    them at scoring time (`NOT dry_run`) -- after 39 hours instead of before."""
    raw = yaml.safe_load(valid_config_file.read_text(encoding="utf-8"))
    raw["scaffold"]["dispatcher"] = "mock"
    mock_cfg = tmp_path / "mock.yaml"
    mock_cfg.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    rc = main(_bench_argv(mock_cfg, tmp_path, "--smoke"))
    assert rc == cli.EXIT_REFUSED
    assert "dispatcher" in capsys.readouterr().err


def test_bench_refuses_when_the_frozen_manifest_moved(valid_config_file, tmp_path,
                                                      monkeypatch, capsys):
    """The pin in `benchmark.manifest_sha256` becomes a precondition here for
    the first time: a manifest that moved would score a different task set than
    the report names."""
    moved = json.loads(cli.BENCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    moved["tasks"][0]["corpus_sha256"] = "f" * 64
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(moved, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    monkeypatch.setattr(cli, "BENCH_MANIFEST_PATH", tampered)

    rc = main(_bench_argv(valid_config_file, tmp_path, "--smoke"))
    assert rc == cli.EXIT_REFUSED
    assert "manifest_sha256 mismatch" in capsys.readouterr().err


def test_bench_refuses_an_unpinned_config(valid_config_file, tmp_path, capsys):
    raw = yaml.safe_load(valid_config_file.read_text(encoding="utf-8"))
    raw["benchmark"].pop("manifest_sha256")
    unpinned = tmp_path / "unpinned.yaml"
    unpinned.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    rc = main(_bench_argv(unpinned, tmp_path, "--smoke"))
    assert rc == cli.EXIT_REFUSED
    assert "manifest_sha256" in capsys.readouterr().err


def test_bench_refuses_an_unknown_task_id(valid_config_file, tmp_path, capsys,
                                          fake_orchestra, monkeypatch):
    _ArmRecorder().patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                          "--tasks", "needle-02,not-a-task"))
    assert rc == cli.EXIT_REFUSED
    assert "not-a-task" in capsys.readouterr().err


def test_bench_refuses_an_unbound_hook(valid_config_file, tmp_path):
    """`BenchCtx`'s hook defaults are deliberate NO-OPS (`src/rlm/measure/bench.py`: the
    module is dry-run with no servers at all). That makes a forgotten binding
    invisible: the run would not crash, it would complete 39 hours of episodes
    with no quiesce, no §4 re-assertion and no leaf relaunch -- every B1/B3
    cell measured against the RLM topology. So the composition root asserts
    the wiring instead of assuming it."""
    from rlm.measure.bench import ARM_ORDER, BenchCtx, BenchLedger
    from rlm.errors import ConfigError

    cfg = load_config(valid_config_file)
    raw = yaml.safe_load(valid_config_file.read_text(encoding="utf-8"))
    bare = BenchCtx(raw_cfg=raw, cfg=cfg, run_id="r",
                    manifest=cli.load_benchmark_manifest(),
                    ledger=BenchLedger(tmp_path / "ledger.jsonl"))
    with pytest.raises(ConfigError) as excinfo:
        cli.assert_bench_wiring(bare, ARM_ORDER)
    message = str(excinfo.value)
    for unbound in ("quiesce_fn", "handshake_fn", "swap_servers_fn",
                    "load_task_fn", "trace", "arm_runners['rlm']"):
        assert unbound in message

    async def _hook(_profile):
        return None

    async def _arm(_task, _cfg, *, bench_extra):
        return None

    wired = BenchCtx(raw_cfg=raw, cfg=cfg, run_id="r",
                     manifest=bare.manifest, ledger=bare.ledger,
                     trace=object(), quiesce_fn=_hook, handshake_fn=_hook,
                     swap_servers_fn=_hook, load_task_fn=lambda p: None,
                     arm_runners={a: _arm for a in ARM_ORDER})
    cli.assert_bench_wiring(wired, ARM_ORDER)       # no raise


# --------------------------------------------------------------------------- #
# --smoke
# --------------------------------------------------------------------------- #


def test_the_smoke_default_task_set_is_one_non_adversarial_task_per_category():
    """§8 flags two tasks as adversarial-context. A calibration run must not be
    timed against them: they are the tasks most likely to behave unlike their
    category, and the whole point of the smoke run is a per-category
    seconds-per-episode number to project 39 hours from."""
    manifest = cli.load_benchmark_manifest()
    ids = cli.smoke_task_ids(manifest)
    by_id = {t.task_id: t for t in manifest.tasks}
    assert len(ids) == len({by_id[i].category for i in ids}) == 4
    assert set(ids) == {"needle-02", "agg-02", "synth-01", "codeqa-01"}
    assert not any(by_id[i].adversarial for i in ids)


def test_smoke_runs_one_seed_of_every_arm_and_never_writes_a_report(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    arms = _ArmRecorder().patch(monkeypatch)
    report = tmp_path / "RESULTS.md"

    rc = main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                          "--tasks", SMOKE_TASKS))
    out = capsys.readouterr().out

    assert rc == cli.EXIT_OK
    # 2 tasks x 1 seed x 4 arms, in §8's within-block order.
    assert arms.order() == ["rlm", "b2", "b1", "b3"] * 2
    assert {c["seed"] for c in arms.calls} == {1}
    assert not report.exists(), "a smoke run must never write the S4 report"
    assert "smoke" in out.lower() and "calibration" in out.lower()
    # ...and the calibration is stated against the pre-registered projection
    # constants, not against nothing.
    assert "450" in out and "2.78" in out and "60" in out


def test_a_smoke_run_id_is_its_own_and_never_scores(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """'Never counts toward scoring' is enforced by identity, not by a flag a
    later reader has to remember: the grid is scoped to a run_id, and this one
    is minted fresh and thrown away."""
    arms = _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "needle-02"))
    first = {c["run_id"] for c in arms.calls}
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "needle-02"))
    every = {c["run_id"] for c in arms.calls}
    assert len(first) == 1 and len(every) == 2
    assert "smoke run_id" in capsys.readouterr().out


def test_smoke_projects_the_full_grid_in_hours(valid_config_file, tmp_path,
                                               capsys, fake_orchestra, monkeypatch):
    _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "agg-02"))
    out = capsys.readouterr().out
    assert " h" in out and "full grid" in out.lower()
    # The projection is over the WHOLE frozen grid, not the smoke subset.
    assert "30 tasks" in out


# --------------------------------------------------------------------------- #
# the wiring itself: which config and which manager each arm is handed
# --------------------------------------------------------------------------- #


def test_each_arm_is_handed_its_own_profile_and_process_manager(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """§8's two topologies. RLM and B2 run on the resident leaf; B1 and B3 on
    the `bench_leaf` relaunch profile, and they must be BUILT against it --
    `bench_slot_capacity` and B1's overflow policy read `servers.leaf`, so an
    arm handed the resident config would truncate against the 128-slot
    topology while running on the 2-slot one.

    The process managers are two DIFFERENT objects on purpose (the ledgered
    ruling): `run_episode` re-handshakes itself after a rotation, `run_b2`
    cannot.
    """
    arms = _ArmRecorder().patch(monkeypatch)
    # ALL FIVE here, overriding `_bench_argv`'s gate-arms default: this test is
    # about topology and process-manager wiring, which the delegation arm has
    # too, not about the gate arithmetic that default exists to protect.
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "needle-02",
                     "--arm", "rlm,rlm-restricted,b2,b1,b3"))

    by_arm = {c["arm"]: c for c in arms.calls}
    cfg = load_config(valid_config_file)
    resident, bench_leaf = cfg.servers.leaf, cfg.servers.bench_leaf
    assert resident.parallel != bench_leaf.parallel       # or this proves nothing
    for arm in ("rlm", "rlm-restricted", "b2"):
        assert by_arm[arm]["leaf_parallel"] == resident.parallel
        assert by_arm[arm]["leaf_ctx"] == resident.ctx
    for arm in ("b1", "b3"):
        assert by_arm[arm]["leaf_parallel"] == bench_leaf.parallel
        assert by_arm[arm]["leaf_ctx"] == bench_leaf.ctx

    # `.label`, not identity: the managers became real objects when the
    # delegating arms started restarting the leaf per episode.
    assert by_arm["rlm"]["kwargs"]["process_manager"].label == "episode"
    assert by_arm["rlm-restricted"]["kwargs"]["process_manager"].label == "episode"
    assert by_arm["b2"]["kwargs"]["process_manager"].label == "b2"
    assert by_arm["b2"]["kwargs"]["root_client"] is not None
    # §8's v0.2.6 slot pin: B1 on slot 0, B3 on slot 1 of the bench profile.
    # The CLI passes no `slot_id` at all -- the pin is pre-registered as each
    # arm's own default, and an override here would be a second place it could
    # drift from. Asserted as "absent, and the defaults are still 0 and 1",
    # because `kwargs.get("slot_id", 0) == 0` would pass either way.
    import inspect

    from rlm.measure.arms import run_b1, run_b3
    assert "slot_id" not in by_arm["b1"]["kwargs"]
    assert "slot_id" not in by_arm["b3"]["kwargs"]
    assert inspect.signature(run_b1).parameters["slot_id"].default == 0
    assert inspect.signature(run_b3).parameters["slot_id"].default == 1
    # The RLM arm's identity travels in `snapshot_extra`; a baseline's in
    # `bench_extra` (`ArmEpisode.snapshot` adds its own `arm` key).
    assert "bench" in by_arm["rlm"]["kwargs"]["snapshot_extra"]
    assert by_arm["b1"]["kwargs"]["bench_extra"]["run_id"]
    # Both seeds move together, on every arm, including the bench-profile ones.
    assert all(c["root_seed"] == c["leaf_seed"] == c["seed"] for c in arms.calls)


def test_the_orchestra_hooks_are_the_ones_the_scheduler_drives(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "needle-02"))
    orchestra = fake_orchestra[0]
    assert orchestra.kinds("swap") == ["bench"]
    assert orchestra.kinds("quiesce") == ["resident", "resident", "bench", "bench"]
    assert orchestra.kinds("handshake") == orchestra.kinds("quiesce")
    assert ("start_resident", None) in orchestra.events
    assert ("stop_all", None) in orchestra.events


def test_the_ledger_carries_the_relaunch_the_orchestra_reported(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """The whole path for the one column the store cannot hold:
    `ServerOrchestra.swap_to` returns its `last_relaunch_s`, `rlm.measure.bench._prepare`
    catches it, and the cell that PAID for the swap ledgers it. §8 excludes
    relaunch time from per-task wall-clock, so without this the ~10 s a swap
    costs would be spent 179 times over a full grid and recorded nowhere."""
    _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks", "needle-02"))
    rows = [json.loads(line) for line
            in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["b1"]["relaunch_s"] == fake_orchestra[0].last_relaunch_s == 11.5
    # …and only that cell: rlm/b2 were already resident, b3 rode b1's relaunch.
    assert [by_arm[a]["relaunch_s"] for a in ("rlm", "b2", "b3")] == [0.0, 0.0, 0.0]
    # It is beside wall_s, never inside it.
    assert by_arm["b1"]["wall_s"] < 11.5


def test_no_launch_servers_is_carried_into_the_orchestra(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--smoke", "--tasks",
                     "needle-02", "--no-launch-servers"))
    assert fake_orchestra[0].launch is False


# --------------------------------------------------------------------------- #
# the graded run: grid -> verdict -> escalation -> report -> exit code
# --------------------------------------------------------------------------- #


def _graded(valid_config_file, tmp_path, monkeypatch, outcomes, *extra):
    arms = _ArmRecorder(outcomes, write=True).patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path,
                          "--tasks", ",".join(FOUR_TASKS), *extra))
    return rc, arms


def _baselines_fail_everything():
    return {(arm, task): Outcome.FAIL
            for arm in ("b1", "b2", "b3") for task in FOUR_TASKS}


def test_a_gate_pass_exits_zero_and_writes_the_report(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """RLM 4/4, every baseline 0/4 -> margin +4 against all three, which clears
    §8's +3 threshold. The report is written with the narrative marker
    preserved, and the exit code is the gate."""
    from rlm.measure.verdict import NARRATIVE_MARKER

    rc, _arms = _graded(valid_config_file, tmp_path, monkeypatch,
                        _baselines_fail_everything())
    out = capsys.readouterr().out
    report = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")

    assert rc == cli.EXIT_OK
    assert "S4 GATE: PASS" in out and "S4 GATE: PASS" in report
    assert NARRATIVE_MARKER in report
    assert (tmp_path / "RESULTS.pareto.svg").exists()


def test_a_gate_failure_exits_one(valid_config_file, tmp_path, capsys,
                                  fake_orchestra, monkeypatch):
    """A tie fails the gate (§8), and the exit code has to agree with the
    heading -- a report that says FAIL beside an exit 0 is a green CI run over
    a failed benchmark."""
    rc, _arms = _graded(valid_config_file, tmp_path, monkeypatch, outcomes={})
    out = capsys.readouterr().out
    assert rc == cli.EXIT_FAILED
    assert "S4 GATE: FAIL" in out


def test_escalation_runs_both_arms_of_every_flagged_pair_then_recomputes_once(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """§8:343 end to end. RLM takes 3 of 4, every baseline 0 -> margin +3,
    inside the {+1,+2,+3} band, so seeds {4,5} are owed on that pair's
    DISCORDANT tasks -- and on BOTH arms of the pair, because a re-decided task
    is re-decided for the COMPARISON, not for RLM alone."""
    outcomes = _baselines_fail_everything()
    outcomes[("rlm", "codeqa-01")] = Outcome.FAIL
    rc, arms = _graded(valid_config_file, tmp_path, monkeypatch, outcomes)
    out = capsys.readouterr().out
    report = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")

    escalated = [c for c in arms.calls if c["seed"] in (4, 5)]
    discordant = {"needle-02", "agg-02", "synth-01"}
    assert {c["task_id"] for c in escalated} == discordant
    assert {c["arm"] for c in escalated} == {"rlm", "b1", "b2", "b3"}
    for arm in ("rlm", "b1", "b2", "b3"):
        assert {(c["task_id"], c["seed"]) for c in escalated if c["arm"] == arm} == {
            (t, s) for t in discordant for s in (4, 5)}
    # ...and RLM's cells are run ONCE across the three pairs that all flagged
    # them, not once per pair.
    rlm_cells = [(c["task_id"], c["seed"]) for c in escalated if c["arm"] == "rlm"]
    assert len(rlm_cells) == len(set(rlm_cells)) == 6

    assert rc == cli.EXIT_OK
    assert "Post-escalation gate: PASS" in report
    assert "escalation" in out.lower()


def test_the_escalation_phase_starts_from_the_profile_the_grid_left_behind(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """Each phase rebuilds its `BenchCtx`, and `BenchCtx.current_profile`
    defaults to `resident` — correct for a fresh run, wrong for every later
    phase. §8's within-block order ends on **b3**, so the main grid leaves the
    leaf on the BENCH profile. A rebuilt context that assumed `resident` sees
    no profile change at the escalation phase's first (rlm) cell, skips the
    swap, and hands §4 the resident config to assert against the bench-profile
    server that is actually running: a `total_slots` mismatch, a `ConfigError`,
    and the entire escalation phase gone (phase 1.5's healing with it).

    `_FakeOrchestra.handshake_profile` refuses a profile that is not the one
    running, exactly as the real §4 assertion would, so this test can see the
    difference at all.
    """
    outcomes = _baselines_fail_everything()
    outcomes[("rlm", "codeqa-01")] = Outcome.FAIL
    rc, arms = _graded(valid_config_file, tmp_path, monkeypatch, outcomes)
    orchestra = fake_orchestra[0]

    assert rc == cli.EXIT_OK, "the escalation phase refused on a stale profile"
    # 4 tasks x 3 seeds = 12 blocks. Block 1 swaps once (resident -> bench for
    # b1/b3); every later block swaps twice (back for rlm/b2, out again for
    # b1/b3) -> 1 + 2*11 = 23 swaps in the grid. The 24th is the escalation
    # phase's first cell (rlm) asking for the resident leaf.
    swaps = orchestra.kinds("swap")
    assert len(swaps) > 23, "the escalation phase issued no swap at all"
    assert swaps[23] == "resident"
    # …and every handshake in the whole run agreed with the live profile,
    # which the fake enforces by raising rather than by being asked politely.
    assert [c for c in arms.calls if c["seed"] in (4, 5)]


def test_a_margin_outside_the_band_escalates_nothing(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    _rc, arms = _graded(valid_config_file, tmp_path, monkeypatch,
                        _baselines_fail_everything())
    assert [c for c in arms.calls if c["seed"] in (4, 5)] == []


def test_the_exit_code_is_the_post_escalation_gate():
    """§8 makes the recomputation the DECISION and the pre-escalation figures a
    reporting obligation. An exit code taken from the pre-escalation verdict
    would report the number §8 says may not be chosen between."""
    pre = SimpleNamespace(gate_pass=True)
    post = SimpleNamespace(gate_pass=False)
    assert cli.bench_exit_code(pre, None) == cli.EXIT_OK
    assert cli.bench_exit_code(pre, post) == cli.EXIT_FAILED
    assert cli.bench_exit_code(post, pre) == cli.EXIT_OK


def test_bench_resumes_into_the_cells_it_has_not_decided(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """The 39-hour rule: an interrupted run resumes by run_id and re-runs only
    what it never decided."""
    arms = _ArmRecorder(_baselines_fail_everything(), write=True).patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--tasks", "needle-02",
                     "--seeds", "1"))
    run_id = arms.calls[0]["run_id"]
    first = len(arms.calls)
    assert first == 4

    main(_bench_argv(valid_config_file, tmp_path, "--tasks", "needle-02",
                     "--seeds", "1", "--resume", run_id))
    assert len(arms.calls) == first, "a resumed run re-ran a decided cell"


# --------------------------------------------------------------------------- #
# `rlm export`
# --------------------------------------------------------------------------- #


def _synthetic_run(cfg, run_id, n=2):
    """A closed store with `n` bench episodes, each carrying a step blob."""
    async def build():
        tl = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root)
        await tl.start()
        ids = []
        for i in range(n):
            episode_id = str(uuid.uuid4())
            ids.append(episode_id)
            tl.open_episode({
                "episode_id": episode_id, "task_id": f"t{i}", "task_hash": "h",
                "started_at": utc_now(),
                "config_snapshot": {"bench": {"run_id": run_id, "arm": "rlm",
                                              "seed": 1, "block": i}}})
            tl.put_step({"episode_id": episode_id, "actor": Actor.ROOT,
                         "action_type": ActionType.REPL_EXEC,
                         "status": StepStatus.OK},
                        {"root_request_ref": f"request-{i}".encode()})
            tl.close_episode(episode_id, Outcome.SUCCESS, None)
        await tl.drain()
        await tl.aclose()
        return ids

    return asyncio.run(build())


def test_export_round_trips_a_whole_run(valid_config_file, tmp_path):
    """The bundle is what a foreign reader gets: a second process cannot open
    the .duckdb file at all on Windows, so the parquet pair plus the blob
    directories are the only way anyone else ever sees a run."""
    cfg = load_config(valid_config_file)
    run_id = str(uuid.uuid4())
    episode_ids = _synthetic_run(cfg, run_id)
    dest = tmp_path / "bundle"

    rc = main(["export", run_id, "--config", str(valid_config_file),
               "--dest", str(dest)])
    assert rc == cli.EXIT_OK

    manifest = json.loads((dest / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_id
    assert sorted(manifest["episode_ids"]) == sorted(episode_ids)
    assert len(manifest["config_snapshot_sha256"]) == 64

    con = duckdb.connect()          # in-memory: no lock, no live handle
    try:
        rows = con.execute(
            f"SELECT CAST(episode_id AS VARCHAR) FROM '{dest / 'episodes.parquet'}'"
        ).fetchall()
        assert sorted(r[0] for r in rows) == sorted(episode_ids)
        steps = con.execute(
            f"SELECT root_request_ref FROM '{dest / 'steps.parquet'}'").fetchall()
        assert len(steps) == 2
        # every blob a step points at resolves inside the bundle alone
        for (rel,) in steps:
            assert (dest / "blobs" / rel).is_file()
    finally:
        con.close()


def test_export_resolves_a_bare_episode_id_too(valid_config_file, tmp_path):
    cfg = load_config(valid_config_file)
    episode_ids = _synthetic_run(cfg, str(uuid.uuid4()))
    dest = tmp_path / "one"
    rc = main(["export", episode_ids[1], "--config", str(valid_config_file),
               "--dest", str(dest)])
    assert rc == cli.EXIT_OK
    manifest = json.loads((dest / "bundle-manifest.json").read_text(encoding="utf-8"))
    assert manifest["episode_ids"] == [episode_ids[1]]
    assert manifest["resolved_by"] == "episode_id"


def test_export_of_an_unknown_id_exits_two(valid_config_file, tmp_path, capsys):
    cfg = load_config(valid_config_file)
    _synthetic_run(cfg, str(uuid.uuid4()))
    rc = main(["export", str(uuid.uuid4()), "--config", str(valid_config_file),
               "--dest", str(tmp_path / "nothing")])
    assert rc == cli.EXIT_REFUSED
    assert "no episode" in capsys.readouterr().err.lower()


def test_export_refuses_an_id_that_is_not_a_uuid(valid_config_file, tmp_path, capsys):
    """`trace.py`'s SQL-injection caveat: the filter is INTERPOLATED into COPY
    (DuckDB takes no bound parameter there). Validating the id as a UUID first
    is what makes that interpolation safe, so a non-UUID is refused rather
    than quoted."""
    cfg = load_config(valid_config_file)
    _synthetic_run(cfg, str(uuid.uuid4()))
    rc = main(["export", "' OR 1=1 --", "--config", str(valid_config_file),
               "--dest", str(tmp_path / "nothing")])
    assert rc == cli.EXIT_REFUSED
    assert "uuid" in capsys.readouterr().err.lower()


def test_export_refuses_a_live_store(valid_config_file, tmp_path, capsys):
    """`TraceLogger.start` cannot open a database another connection holds
    under a different configuration -- which is exactly the state a live bench
    run leaves the file in. That refusal is the CORRECT behaviour: a bundle
    exported from a half-written store is a bundle of a different run."""
    cfg = load_config(valid_config_file)
    run_id = str(uuid.uuid4())
    _synthetic_run(cfg, run_id)

    holder = duckdb.connect(str(cfg.trace.db_path), read_only=True)
    try:
        rc = main(["export", run_id, "--config", str(valid_config_file),
                   "--dest", str(tmp_path / "live")])
    finally:
        holder.close()
    assert rc == cli.EXIT_REFUSED
    assert "closed" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# the argument surface, and one cosmetic
# --------------------------------------------------------------------------- #


def test_bench_defaults_are_the_pre_registered_ones():
    from rlm.cli import build_parser

    args = build_parser().parse_args(["bench"])
    assert args.arm is None and args.seeds is None and args.tasks is None
    assert args.smoke is False and args.no_launch_servers is False
    assert args.resume is None
    assert Path(args.report) == cli.DEFAULT_REPORT_PATH
    assert Path(args.ledger) == cli.LEDGER_PATH


def test_export_defaults_to_a_dest_beside_the_trace_store():
    from rlm.cli import build_parser

    args = build_parser().parse_args(["export", "abc"])
    assert args.dest is None and args.id == "abc"


def test_the_transcript_labels_an_llm_call_by_its_actor(valid_config_file):
    """A B2 episode's root reduce call is an `llm_call` with `actor='root'`.
    Labelling every llm_call 'leaf' made the transcript say a call happened on
    a server it never touched."""
    cfg = load_config(valid_config_file)
    steps = [
        {"step_idx": 0, "actor": "root", "action_type": ActionType.LLM_CALL,
         "status": "ok", "parent_step_idx": None, "retry_idx": 0,
         "tokens_in": 10, "tokens_out": 2, "action_payload": "",
         "observation_view": ""},
        {"step_idx": 1, "actor": "leaf", "action_type": ActionType.LLM_CALL,
         "status": "ok", "parent_step_idx": 0, "retry_idx": 0,
         "tokens_in": 3, "tokens_out": 4, "action_payload": "",
         "observation_view": ""},
    ]
    out = io.StringIO()
    cli._render_transcript(cfg, steps, out)
    rendered = out.getvalue()
    assert "root llm_call" in rendered
    assert rendered.count("leaf llm_call") == 1


# =========================================================================== #
# Review fixes (S4 Task 12, round 1)
# =========================================================================== #


def test_the_arm_runners_hand_every_arm_the_blocks_own_seed(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """The bench half of the per-call-seed fix.

    §8's three replicates are three seeds of the WHOLE system, and
    `rlm.measure.bench.seeded_config` re-seeds the CONFIG per attempt. What the arm
    runners must therefore hand each arm is a config carrying THAT block's
    seed -- on both the root and the leaf, and on the bench-profile arms too,
    which derive their own `Config` and could silently keep the shipped one.
    (`src/rlm/episode.py` and `src/rlm/measure/arms.py` then put that seed on the wire per
    call; `checks/test_dispatcher.py` pins the wire end.)
    """
    arms = _ArmRecorder().patch(monkeypatch)
    main(_bench_argv(valid_config_file, tmp_path, "--tasks", "needle-02",
                     "--seeds", "1,2,3"))
    shipped = load_config(valid_config_file).scaffold.sampling.leaf.seed
    seen = {(c["arm"], c["seed"], c["root_seed"], c["leaf_seed"])
            for c in arms.calls}
    assert seen == {(arm, seed, seed, seed)
                    for arm in ("rlm", "b1", "b2", "b3") for seed in (1, 2, 3)}
    # …and at least one of those is NOT the config's shipped value, or this
    # test would pass against a scaffold that never re-seeded anything.
    assert {c["leaf_seed"] for c in arms.calls} - {shipped}


def test_the_sampler_is_stopped_even_when_teardown_raises(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """A PowerShell child polling an energy counter is the only resource in
    this teardown that outlives the process. An unclosed httpx client dies with
    us; a detached sampler runs until the machine reboots. So it is stopped
    FIRST and every later step is individually suppressed -- an orchestra that
    will not stop, or a client whose transport is already gone, must not leak
    it."""
    _ArmRecorder().patch(monkeypatch)

    sampler = _StubSampler()
    monkeypatch.setattr(cli, "PowerSampler", lambda *a, **kw: sampler)

    class _ExplodingDispatcher:
        @classmethod
        def from_config(cls, cfg, **kw):
            return cls()

        async def aclose(self):
            raise RuntimeError("transport already closed")

        def rotate_pool(self):
            """A no-op, but it has to EXIST: the delegating arms rotate before
            every episode, and this double stands in for the real dispatcher on
            the path that reaches them."""

        # The delegating arms now rotate before every episode, via the full
        # §5 C4 sequence: `rotating()` (quiesce -> replace -> resume) around
        # restart + `rotate_pool`, then an assertion that the pool really is
        # virgin. A dispatcher double has to model all three or it no longer
        # stands in for the real one on the path that reaches them.
        def rotating(self):
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield

            return _cm()

        @property
        def restart_required(self):
            return False

        async def count_tokens(self, text, *, role="leaf"):
            return 1

    monkeypatch.setattr(cli, "LLMDispatcher", _ExplodingDispatcher)
    monkeypatch.setattr(cli, "bench_dispatcher", lambda *a, **kw: _ExplodingDispatcher())

    raw = yaml.safe_load(valid_config_file.read_text(encoding="utf-8"))
    raw["power_sampling"] = {"enabled": True}
    powered = tmp_path / "powered.yaml"
    powered.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    rc = main(_bench_argv(powered, tmp_path, "--smoke", "--tasks", "needle-02"))
    assert sampler.started and sampler.stopped, "the sampler outlived the run"
    assert rc == cli.EXIT_OK


def test_a_scaffold_failure_mid_grid_is_a_refusal_not_a_gate_failure(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """The exit-code taxonomy: 0 = gate PASS, 1 = gate FAIL, 2 = no verdict.

    1 is a RESULT -- a grid that ran, was scored, and lost. A server that never
    came up produced no verdict at all, and reporting it as 1 would put "the
    scaffold lost to its baselines" into CI, and into the S4 record, for a run
    nothing scored."""
    from rlm.errors import ServerRotationError

    _ArmRecorder().patch(monkeypatch)

    async def explode(_self):
        raise ServerRotationError("leaf did not come up on port 8081")

    monkeypatch.setattr(_FakeOrchestra, "start_resident", explode)
    rc = main(_bench_argv(valid_config_file, tmp_path, "--tasks", "needle-02"))
    err = capsys.readouterr().err
    assert rc == cli.EXIT_REFUSED
    assert rc != cli.EXIT_FAILED
    assert "no verdict" in err and "--resume" in err


def test_an_operator_abort_is_a_refusal_not_a_gate_failure(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """Ctrl-C at hour 30 of 39 must not be filed as a benchmark loss."""
    _ArmRecorder().patch(monkeypatch)

    async def interrupt(_self):
        raise KeyboardInterrupt

    monkeypatch.setattr(_FakeOrchestra, "start_resident", interrupt)
    rc = main(_bench_argv(valid_config_file, tmp_path, "--tasks", "needle-02"))
    err = capsys.readouterr().err
    assert rc == cli.EXIT_REFUSED
    assert "no verdict" in err and "--resume" in err


def test_only_the_banded_pairs_arms_are_escalated(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """§8 escalates a PAIR, not a grid. One baseline in the {+1,+2,+3} band
    must draw seeds {4,5} for itself and RLM and for nobody else -- an
    escalation that swept every arm would spend episodes §8 never registered
    and re-decide tasks against baselines whose margin was never in doubt.

    RLM 4/4; B1 1/4 (margin +3, banded); B2 and B3 0/4 (margin +4, outside).
    """
    outcomes = {(arm, task): Outcome.FAIL
                for arm in ("b1", "b2", "b3") for task in FOUR_TASKS}
    outcomes[("b1", "codeqa-01")] = Outcome.SUCCESS
    _rc, arms = _graded(valid_config_file, tmp_path, monkeypatch, outcomes)

    escalated = [c for c in arms.calls if c["seed"] in (4, 5)]
    assert {c["arm"] for c in escalated} == {"rlm", "b1"}
    assert not [c for c in escalated if c["arm"] in ("b2", "b3")]
    # B1's discordant tasks are the three RLM took and B1 did not.
    discordant = {"needle-02", "agg-02", "synth-01"}
    assert {c["task_id"] for c in escalated} == discordant
    assert len(escalated) == len(discordant) * 2 * 2      # tasks x seeds x arms


def test_smoke_and_resume_are_refused_together(valid_config_file, tmp_path,
                                                capsys, fake_orchestra, monkeypatch):
    """A smoke pass is unscored ONLY because its run_id is a throwaway.
    `--resume` would write its calibration episodes into an existing run's
    grid, where `load_grid` reads them as ordinary cells."""
    _ArmRecorder().patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                          "--tasks", "needle-02", "--resume", str(uuid.uuid4())))
    err = capsys.readouterr().err
    assert rc == cli.EXIT_REFUSED
    assert "mutually exclusive" in err


# =========================================================================== #
# Fix wave 2 (whole-branch review)
# =========================================================================== #


class _PooledDispatcher:
    """A dispatcher double carrying the REAL `SlotPool`, at the bench
    profile's real size of 2.

    Everything R13's mitigation actually does is in that object: a slot index
    is handed out at most once for the lifetime of one leaf PROCESS, and
    exhaustion RAISES rather than wrapping. So this double uses the shipped
    class rather than modelling it -- a hand-rolled counter would let the test
    pass against semantics the production pool does not have.
    """

    def __init__(self, size: int = 2) -> None:
        from rlm.serve.dispatcher import SlotPool

        self._SlotPool = SlotPool
        self.slots = SlotPool(size)
        self.rotations = 0
        self.calls: list[str] = []
        self.steps: list[dict] = []
        self._recorded: dict[str, int] = {}

    def rotate_pool(self) -> None:
        self.slots = self._SlotPool(self.slots.size)
        self.rotations += 1

    async def query(self, prompt, *, role, call_id, chunk=None, **_kw):
        # B1/B3 call with `chunk=None`, so every call is its own window and no
        # window is ever repeated -- which is exactly why a 2-slot pool is
        # spent after two calls and why the pool MUST be rotated per block.
        from rlm.serve.dispatcher import window_key

        window = window_key(chunk, call_id)
        slot = self.slots.acquire(window)           # raises SlotPoolExhausted
        self.slots.mark_answered(window)
        self.calls.append(f"{prompt}@{slot}")
        self.steps.append({"call_id": call_id, "rendered": "x" * 1024})
        return "ANSWER"


async def test_a_relaunched_bench_leaf_gets_a_virgin_slot_pool(tmp_path,
                                                                minimal_cfg_dict):
    """C1: the bench-profile pool is never rotated across blocks.

    R13's mitigation gives every window a slot no other window ever gets, for
    the lifetime of one leaf PROCESS. `ServerOrchestra` restarts that process at
    every profile change, but the POOL lives in the dispatcher, which a bench
    run builds once and keeps for all 90 blocks. The bench leaf has TWO slots
    and B1/B3 spend one each per block, so from block 2 onward every B1 and B3
    cell raised `SlotPoolExhausted` against a server that had just been
    restarted with two virgin slots — §8's two single-shot arms failing 58 of
    60 blocks structurally, which is a manufactured result, not a measurement.

    Driven through the REAL scheduler (`rlm.measure.bench.run_block`) with the REAL
    swap wrapper, over two blocks, because one block cannot see the defect.
    """
    import copy as copymod

    from rlm.measure.bench import ARM_ORDER, BenchCtx, BenchLedger, build_blocks
    from rlm.config import Config
    from rlm.episode import Task

    raw = copymod.deepcopy(minimal_cfg_dict)
    raw["trace"]["db_path"] = str(tmp_path / "rlm.duckdb")
    raw["trace"]["blob_root"] = str(tmp_path / "blobs")
    manifest = cli.load_benchmark_manifest()
    resident, bench_leaf = _PooledDispatcher(128), _PooledDispatcher(2)

    class _Orchestra:
        def __init__(self):
            self.swaps: list[str] = []

        async def swap_to(self, profile):
            self.swaps.append(profile)
            return 1.0

    orchestra = _Orchestra()
    swap = cli.pool_rotating_swap(orchestra, resident_dispatcher=resident,
                                   bench_dispatcher=bench_leaf)

    async def _noop(_profile):
        return None

    def _runner(arm, dispatcher):
        async def run(task, cfg, *, bench_extra):
            answer = await dispatcher.query(f"{arm}:{task.task_id}", role="leaf",
                                             call_id=str(uuid.uuid4()))
            return SimpleNamespace(episode_id=str(uuid.uuid4()),
                                   outcome=Outcome.SUCCESS, reason=None,
                                   answer=answer)
        return run

    ctx = BenchCtx(
        raw_cfg=raw, cfg=Config.model_validate(copymod.deepcopy(raw)),
        run_id="pool-run", manifest=manifest,
        ledger=BenchLedger(tmp_path / "ledger.jsonl"),
        trace=SimpleNamespace(mark_superseded=lambda *a: None,
                              update_episode_metrics=lambda *a, **k: None),
        arm_runners={"rlm": _runner("rlm", resident),
                     "rlm-restricted": _runner("rlm-restricted", resident),
                     "b2": _runner("b2", resident),
                     "b1": _runner("b1", bench_leaf),
                     "b3": _runner("b3", bench_leaf)},
        load_task_fn=lambda p: Task(task_id=Path(p).stem, text="q", context=""),
        quiesce_fn=_noop, handshake_fn=_noop, swap_servers_fn=swap,
        repo_root=Path(tmp_path))

    from rlm.measure.bench import run_block

    records = []
    for block in build_blocks(manifest, [1])[:2]:
        records += await run_block(block, list(ARM_ORDER), ctx)

    assert len(records) == 10
    assert all(str(r["outcome"]) == "success" for r in records), \
        [(r["arm"], r["outcome"], r["reason"]) for r in records]
    # Block 2's B1/B3 are the cells the defect killed.
    assert [(r["arm"], str(r["outcome"])) for r in records[5:]] == [
        ("rlm", "success"), ("rlm-restricted", "success"), ("b2", "success"),
        ("b1", "success"), ("b3", "success")]
    # …and each rotation followed a real swap, in both directions.
    assert orchestra.swaps == ["bench", "resident", "bench"]
    assert bench_leaf.rotations == 2 and resident.rotations == 1


class _SwappingOrchestra:
    """`swap_to`'s real shape: a no-op when the requested profile is already
    live (`_bring_up_leaf`'s `already_there`), a NEW process object otherwise."""

    def __init__(self, profile=None):
        self.current_profile = profile
        self.leaf_proc = object() if profile else None

    async def swap_to(self, profile):
        if profile == self.current_profile:
            return 0.0
        self.current_profile = profile
        self.leaf_proc = object()
        return 9.0


async def test_the_pool_is_rotated_only_for_the_profile_that_swapped(tmp_path):
    """A swap to the bench profile must not hand the RESIDENT dispatcher a
    fresh pool: the resident leaf process is untouched by that swap, and
    telling C4 its used slots are virgin again is R13 itself."""
    resident, bench_leaf = _PooledDispatcher(4), _PooledDispatcher(2)
    swap = cli.pool_rotating_swap(_SwappingOrchestra(cli.RESIDENT_PROFILE),
                                   resident_dispatcher=resident,
                                   bench_dispatcher=bench_leaf)
    await swap(cli.BENCH_PROFILE)
    assert (bench_leaf.rotations, resident.rotations) == (1, 0)
    await swap(cli.RESIDENT_PROFILE)
    assert (bench_leaf.rotations, resident.rotations) == (1, 1)


async def test_a_swap_that_restarted_nothing_does_not_adopt_a_virgin_pool(tmp_path):
    """`swap_to` NO-OPS when the requested profile is already live. Rotating
    then would tell C4 every slot is untouched while the same process still
    holds every document it has served — the hazard the never-reuse pool
    exists to prevent, reintroduced by the fix for the opposite one."""
    resident, bench_leaf = _PooledDispatcher(4), _PooledDispatcher(2)
    orchestra = _SwappingOrchestra(cli.BENCH_PROFILE)
    swap = cli.pool_rotating_swap(orchestra, resident_dispatcher=resident,
                                   bench_dispatcher=bench_leaf)

    assert await swap(cli.BENCH_PROFILE) == 0.0        # already there
    assert (bench_leaf.rotations, resident.rotations) == (0, 0)
    # …and a genuine swap still rotates, so this is a gate and not a mute.
    assert await swap(cli.RESIDENT_PROFILE) == 9.0
    assert (bench_leaf.rotations, resident.rotations) == (0, 1)


async def test_a_relaunch_that_measured_zero_seconds_still_rotates(tmp_path):
    """The signal is the PROCESS OBJECT, not the clock. `last_relaunch_s` is
    `round(elapsed, 3)`, so a fast (or faked) relaunch legitimately reports
    0.0 — and a rotation skipped on that arithmetic is a used pool the
    scaffold believes is virgin."""
    bench_leaf = _PooledDispatcher(2)

    class _InstantOrchestra(_SwappingOrchestra):
        async def swap_to(self, profile):
            self.current_profile = profile
            self.leaf_proc = object()      # it really did restart
            return 0.0                     # …in under a millisecond

    swap = cli.pool_rotating_swap(_InstantOrchestra(cli.RESIDENT_PROFILE),
                                   resident_dispatcher=None,
                                   bench_dispatcher=bench_leaf)
    await swap(cli.BENCH_PROFILE)
    assert bench_leaf.rotations == 1


async def test_the_swap_hook_still_reports_what_the_relaunch_cost(tmp_path):
    """The wrapper must stay transparent: `rlm.measure.bench._prepare` ledgers the
    hook's return value as `relaunch_s`."""
    class _Orchestra:
        async def swap_to(self, profile):
            return 12.5

    swap = cli.pool_rotating_swap(_Orchestra(), resident_dispatcher=None,
                                   bench_dispatcher=None)
    assert await swap(cli.BENCH_PROFILE) == 12.5


def test_the_step_log_does_not_grow_across_episodes(valid_config_file, tmp_path,
                                                     fake_orchestra, monkeypatch):
    """I4: `LLMDispatcher.steps` is append-only and every row holds the FULL
    rendered prompt. B1 renders a ~256K-token document, so one B1 episode
    retains ~0.5 MB that nothing reads again — an arm reads `steps` only for
    the call_ids of the episode it is closing. Over 360 episodes on a
    64 GiB-carved box that is not a leak to shrug at, and the composition root
    owns the dispatcher's lifetime."""
    seen: list[int] = []
    dispatchers: dict[str, object] = {}

    class _Recording:
        @classmethod
        def from_config(cls, cfg, **kw):
            return cls()

        def __init__(self):
            self.steps: list[dict] = []
            self._recorded: dict[str, int] = {}
            self.rotations = 0

        def rotate_pool(self):
            """The delegating arms rotate before every episode now
            (`cli._virgin_resident_pool`), so a dispatcher double that cannot
            be rotated no longer models one."""
            self.rotations += 1

        # The delegating arms now rotate before every episode, via the full
        # §5 C4 sequence: `rotating()` (quiesce -> replace -> resume) around
        # restart + `rotate_pool`, then an assertion that the pool really is
        # virgin. A dispatcher double has to model all three or it no longer
        # stands in for the real one on the path that reaches them.
        def rotating(self):
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm():
                yield

            return _cm()

        @property
        def restart_required(self):
            return False

        async def aclose(self):
            pass

    def _capture(arm):
        async def run(task, cfg, **kw):
            d = kw.get("dispatcher")
            dispatchers[arm] = d
            d.steps.append({"rendered": "x" * 4096})
            seen.append(len(d.steps))
            return SimpleNamespace(episode_id=str(uuid.uuid4()),
                                   outcome=Outcome.SUCCESS, reason=None,
                                   answer=None, final_answer=None)
        return run

    monkeypatch.setattr(cli, "LLMDispatcher", _Recording)
    monkeypatch.setattr(cli, "bench_dispatcher", lambda *a, **kw: _Recording())
    for name, arm in (("run_episode", "rlm"), ("run_b1", "b1"),
                      ("run_b2", "b2"), ("run_b3", "b3")):
        monkeypatch.setattr(cli, name, _capture(arm))

    main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                     "--tasks", "needle-02,agg-02"))
    # Every episode saw a step log holding ONLY its own row.
    assert seen == [1] * 8
    assert all(d.steps == [] for d in dispatchers.values())


def test_a_smoke_run_whose_cells_errored_exits_non_zero(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """M2: a calibration pass whose cells errored has measured nothing, and
    exiting 0 on it is how a broken topology gets a green light on the way into
    a 39-hour run. A smoke run has no gate, so 1 keeps EXIT_FAILED's literal
    meaning — the command did not complete cleanly."""
    _ArmRecorder({("b1", "needle-02"): Outcome.ERROR}).patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                          "--tasks", "needle-02"))
    captured = capsys.readouterr()
    assert rc == cli.EXIT_FAILED
    assert "smoke cell(s) errored" in captured.err
    assert "b1/needle-02" in captured.err


def test_a_clean_smoke_run_still_exits_zero(valid_config_file, tmp_path,
                                             fake_orchestra, monkeypatch):
    _ArmRecorder().patch(monkeypatch)
    assert main(_bench_argv(valid_config_file, tmp_path, "--smoke",
                            "--tasks", "needle-02")) == cli.EXIT_OK


# --------------------------------------------------------------------------- #
# I2: the escalation phase survives a crash
# --------------------------------------------------------------------------- #


def test_the_escalation_plan_is_written_before_any_escalation_episode(
        valid_config_file, tmp_path, fake_orchestra, monkeypatch):
    """The pre-escalation figures stop being recoverable the instant the first
    escalation episode lands: the store then holds a 5-seed grid and no query
    over it reproduces what the 3-seed grid decided. §8 requires BOTH reported,
    so they are written down while they are still true."""
    outcomes = _baselines_fail_everything()
    outcomes[("rlm", "codeqa-01")] = Outcome.FAIL
    _rc, arms = _graded(valid_config_file, tmp_path, monkeypatch, outcomes)

    run_id = arms.calls[0]["run_id"]
    plan_path = cli.escalation_plan_path(tmp_path / "ledger.jsonl", run_id)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["run_id"] == run_id
    assert plan["escalation_seeds"] == [4, 5]
    discordant = {"needle-02", "agg-02", "synth-01"}
    for baseline in ("b1", "b2", "b3"):
        assert set(plan["escalation_plan"][baseline]) == discordant
        pair = plan["pairs"][baseline]
        assert pair["margin"] == 3 and pair["rlm_passes"] == 3
        assert pair["baseline_passes"] == 0
        assert pair["p"] is not None and pair["ci"] is not None


def test_a_crash_mid_escalation_resumes_into_the_plan(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """The whole point of the artifact. A run that dies mid-escalation leaves
    HALF-escalated cells (some of {4,5}, not all), which `load_grid` refuses
    outright and correctly — §8 re-decides at >=3/5 and a 4-long cell would be
    scored at a denominator it never pre-registered. So a resume finishes the
    plan FIRST and only then asks for a grid."""
    outcomes = _baselines_fail_everything()
    outcomes[("rlm", "codeqa-01")] = Outcome.FAIL

    # -- the crash: die on the first seed-5 escalation episode --------------
    arms = _ArmRecorder(outcomes, write=True)
    real_entry = arms.entry

    def crashing_entry(arm):
        inner = real_entry(arm)

        async def run(task, cfg, **kw):
            extra = kw.get("bench_extra") or (kw.get("snapshot_extra") or {}).get("bench")
            if (extra or {}).get("seed") == 5:
                raise KeyboardInterrupt          # the operator, or the power
            return await inner(task, cfg, **kw)

        return run

    arms.entry = crashing_entry
    arms.patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path,
                          "--tasks", ",".join(FOUR_TASKS)))
    run_id = arms.calls[0]["run_id"]
    assert rc == cli.EXIT_REFUSED, "an abort mid-escalation is not a gate result"
    assert cli.escalation_plan_path(tmp_path / "ledger.jsonl", run_id).exists()
    # The store is now half-escalated: seed 4 landed, seed 5 did not.
    assert {c["seed"] for c in arms.calls if c["seed"] in (4, 5)} == {4}

    # -- the resume: heal, then score --------------------------------------
    healed = _ArmRecorder(outcomes, write=True).patch(monkeypatch)
    rc = main(_bench_argv(valid_config_file, tmp_path,
                          "--tasks", ",".join(FOUR_TASKS), "--resume", run_id))
    out = capsys.readouterr().out
    report = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")

    assert rc == cli.EXIT_OK
    assert "resuming an escalation" in out
    # Only the OWED cells were re-run. The crash landed after agg-02/seed 4,
    # so that cell is decided and must not be drawn a second time, while
    # agg-02/seed 5 (the half of a half-escalated cell that `load_grid`
    # refuses) is exactly what the resume owed.
    healed_cells = {(c["task_id"], c["seed"], c["arm"]) for c in healed.calls}
    assert ("agg-02", 4, "rlm") not in healed_cells
    assert ("agg-02", 5, "rlm") in healed_cells
    assert all(c["seed"] in (4, 5) for c in healed.calls), \
        "a resume re-ran the base grid"
    # …and the report still states BOTH figures, the pre one from the plan.
    assert "Post-escalation gate: PASS" in report
    assert "| RLM vs B1 | +3 |" in report


def test_the_plan_file_carries_the_pre_escalation_figures_into_the_report(
        valid_config_file, tmp_path, capsys, fake_orchestra, monkeypatch):
    """A resumed escalation scores a grid that already carries {4,5}, so the
    'pre' column cannot come from the store. It comes from the plan, or it
    would silently become a second copy of the post column."""
    outcomes = _baselines_fail_everything()
    outcomes[("rlm", "codeqa-01")] = Outcome.FAIL
    _graded(valid_config_file, tmp_path, monkeypatch, outcomes)
    capsys.readouterr()

    run_id = json.loads(
        next((tmp_path).glob("escalation-*.json")).read_text(encoding="utf-8"))["run_id"]
    plan_path = cli.escalation_plan_path(tmp_path / "ledger.jsonl", run_id)
    restored = cli.load_escalation_plan(plan_path)
    assert restored is not None
    verdict, seeds = restored
    assert seeds == [4, 5]
    assert verdict.run_id == run_id
    assert verdict.escalated is False               # it is the PRE verdict
    assert verdict.pairs["b1"].margin == 3
    assert set(verdict.escalation_plan["b2"]) == {"needle-02", "agg-02", "synth-01"}
    assert cli.load_escalation_plan(tmp_path / "absent.json") is None


def test_an_unreadable_plan_refuses_rather_than_losing_the_pre_figures(tmp_path):
    from rlm.errors import ConfigError

    bad = tmp_path / "escalation-x.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="pre-escalation figures"):
        cli.load_escalation_plan(bad)


def test_replay_reconstructs_under_the_snapshots_history_mode(mock_episode_env, capsys, tmp_path):
    """The reconstruction rule is read from the EPISODE's config_snapshot:
    an episode recorded under prefix_plus_raw replays under prefix_plus_raw
    even when the live config says raw, and vice versa. Two turns, so the
    second turn's stored array actually contains an assistant message built
    by history_message() -- a one-turn episode would never compare one."""
    import yaml
    two_turn = ["```repl\nprint(1)\n```", "```repl\nfinal_answer('42')\n```"]

    def run_with(config_file: Path) -> str:
        # FakeRootServer serves script[turns] and never resets `turns` on its
        # own; a second `rlm run` against a spent script gets HTTP 500.
        mock_episode_env.server.script = list(two_turn)
        mock_episode_env.server.turns = 0
        main(["run", str(mock_episode_env.task_file), "--config", str(config_file)])
        return mock_episode_env.last_episode_id()

    raw_cfg = yaml.safe_load(Path(mock_episode_env.config_file).read_text(encoding="utf-8"))
    raw_cfg["scaffold"]["root"]["history_mode"] = "prefix_plus_raw"
    old_cfg = tmp_path / "config-old-history.yaml"
    old_cfg.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    old_episode = run_with(old_cfg)
    new_episode = run_with(Path(mock_episode_env.config_file))
    for episode_id in (old_episode, new_episode):
        # replay always takes the LIVE config file on the command line; the
        # rule must come from the snapshot regardless of which file that is
        rc = main(["replay", episode_id, "--config", str(mock_episode_env.config_file)])
        assert rc == 0
        assert "message array: OK" in capsys.readouterr().out
