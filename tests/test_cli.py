# tests/test_cli.py
import sys

import pytest

from rlm.cli import main

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
