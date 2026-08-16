# tests/test_cli.py
import io
import sys
from pathlib import Path

import pytest

from rlm.cli import main
from rlm.config import load_config
from rlm.lifecycle import Lifecycle

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


def test_the_cli_has_exactly_three_verbs():
    """Non-goals are written into the spec: no daemon, no REST API, no web UI,
    no interactive chat mode. bench/export are later slices, so a fourth verb
    appearing here means a slice shipped without arguing for it."""
    import argparse

    from rlm.cli import build_parser

    sub = [a for a in build_parser()._actions
           if isinstance(a, argparse._SubParsersAction)]
    assert len(sub) == 1
    assert set(sub[0].choices) == {"validate", "run", "replay"}


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
    from rlm.serverproc import launch_argv

    cfg = load_config(valid_config_file)
    manager = leaf_process_manager(cfg, launch=True)
    assert manager is not None
    assert manager.argv == launch_argv(cfg.servers.leaf)
    assert not manager.owned          # nothing started yet; `run` starts it


def test_launch_leaf_carries_the_config_env_to_the_child(valid_config_file):
    """`servers.leaf.env` is merged over os.environ at launch. The CLI passed
    `env=None` here, which silently dropped ROCBLAS_USE_HIPBLASLT -- the
    variable every S2 leaf measurement was taken with (`s2/run_occupancy.py`
    :455) -- so a `--launch-leaf` S4 block would have been compared against
    numbers from a differently configured BLAS."""
    from rlm.cli import leaf_process_manager

    cfg = load_config(valid_config_file)
    manager = leaf_process_manager(cfg, launch=True)
    assert manager is not None
    assert manager.env == dict(cfg.servers.leaf.env)
    assert manager.env["ROCBLAS_USE_HIPBLASLT"] == "1"


def test_replay_survives_an_s4_era_snapshot_with_baseline_prompts(mock_episode_env):
    """An episode recorded with baselines loaded must replay clean: the
    episode_config rebuild includes baselines.* or PromptDrift fires on every
    S4 episode (the leaf_envelope bug class, cli.py:659-663)."""
    main(["run", str(mock_episode_env.task_file), "--config",
          str(mock_episode_env.config_file)])
    assert main(["replay", mock_episode_env.last_episode_id(), "--config",
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
