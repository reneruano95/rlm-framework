import datetime as dt
import json
import uuid

import duckdb
import pytest

from rlm.errors import ActionType, Actor, Outcome, StepStatus
from rlm.trace import TraceLogger, recover_orphans, safe_text


def _episode_row(episode_id: str, **over) -> dict:
    """Minimal `open_episode` dict, matching the inline shape used throughout
    this file (episode_id/task_id/task_hash/config_snapshot)."""
    row = {"episode_id": episode_id, "task_id": "t", "task_hash": "h",
           "config_snapshot": {}}
    row.update(over)
    return row


def test_safe_text_survives_lone_surrogates():
    assert safe_text("a" + chr(0xDCFF) + "b")  # must not raise
    json.dumps(safe_text("a" + chr(0xD800)))


async def test_step_rows_and_blobs_round_trip(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t1", "task_hash": "h",
                     "config_snapshot": {"note": "x" + chr(0xDCFF)}})
    tl.put_step(
        {"episode_id": ep, "step_idx": 0, "actor": Actor.ROOT,
         "action_type": ActionType.REPL_EXEC, "status": StepStatus.OK,
         "action_payload": "print(1)", "observation_view": "[stdout]\n1"},
        blobs={"observation_full_ref": b"\x00\x01raw bytes"},
    )
    await tl.drain()
    tl.close_episode(ep, Outcome.SUCCESS, None, None)
    await tl.aclose()

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    row = con.execute(
        "SELECT status, observation_full_ref FROM steps WHERE episode_id = ?",
        [ep]).fetchone()
    assert row[0] == "ok"
    assert (tmp_path / "blobs" / ep / row[1].split("/")[-1]).exists()


async def test_blob_is_written_before_the_referencing_row(tmp_path):
    """Ordering is the durability contract: an orphan blob is recoverable,
    a row pointing at a missing file is not."""
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.ROOT,
                 "action_type": ActionType.REPL_EXEC, "status": StepStatus.OK},
                blobs={"observation_full_ref": b"payload"})
    await tl.drain()
    await tl.aclose()
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    refs = con.execute("SELECT observation_full_ref FROM steps").fetchall()
    for (ref,) in refs:
        assert (tmp_path / "blobs" / ref).exists(), "dangling blob reference"


async def test_monitor_uses_a_sibling_cursor_not_a_second_connection(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    cur = tl.monitor()
    assert cur.execute("SELECT count(*) FROM steps").fetchone()[0] == 0
    with pytest.raises(duckdb.Error):
        duckdb.connect(str(tmp_path / "t.duckdb"), read_only=True)
    await tl.aclose()


def test_recover_orphans_tombstones_null_outcome_episodes(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute((__import__("pathlib").Path(__file__).parents[1]
                 / "rlm" / "schema.sql").read_text())
    ep = str(uuid.uuid4())
    con.execute(
        "INSERT INTO episodes (episode_id, task_id, task_hash, started_at) "
        "VALUES (?, ?, ?, ?)",
        [ep, "t", "h", dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)])
    con.close()
    assert recover_orphans(db, lifecycle=None) == [ep]
    con = duckdb.connect(str(db))
    assert con.execute("SELECT outcome, outcome_reason FROM episodes"
                       ).fetchone() == ("error", "orphaned_at_recovery")


async def test_export_bundle_is_self_contained(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.OK},
                blobs={"observation_full_ref": b"leaf answer"})
    await tl.drain()
    tl.close_episode(ep, Outcome.SUCCESS, None, None)
    await tl.drain()  # drain-before-export contract: close_episode() alone
                       # is not enough -- its row may still be unprocessed
                       # in the writer queue when export_bundle() runs.
    dest = tmp_path / "bundle"
    tl.export_bundle(dest)
    await tl.aclose()
    con = duckdb.connect()  # in-memory: a foreign reader, no lock, no .duckdb
    got = con.execute(
        f"SELECT b.content FROM '{dest / 'steps.parquet'}' s "
        f"JOIN '{dest / 'blobs.parquet'}' b ON b.rel = s.observation_full_ref"
    ).fetchone()
    assert got[0] == b"leaf answer"
    outcome_row = con.execute(
        f"SELECT outcome, ended_at FROM '{dest / 'episodes.parquet'}' "
        f"WHERE episode_id = ?", [ep]).fetchone()
    assert outcome_row[0] == "success"
    assert outcome_row[1] is not None


async def test_leak_columns_round_trip_and_default_to_not_checked(tmp_path):
    """R13's detector records its verdict on the step (§5 C4). The column is
    NULLABLE and defaults to NULL on purpose: NULL means "not checked", and a
    step that was never checked must not read as one that was checked and came
    back clean -- 138 clean calls bound the leak rate at 2.2%, they do not
    zero it."""
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.OK,
                 "leak_detected": True,
                 "leak_detail": "1 foreign identifier(s): ENT-19022@chunk[3]"})
    tl.put_step({"episode_id": ep, "step_idx": 1, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.OK,
                 "leak_detected": False})
    tl.put_step({"episode_id": ep, "step_idx": 2, "actor": Actor.ROOT,
                 "action_type": ActionType.REPL_EXEC, "status": StepStatus.OK})
    await tl.drain()
    await tl.aclose()

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    rows = con.execute("SELECT leak_detected, leak_detail FROM steps "
                       "ORDER BY step_idx").fetchall()
    assert rows[0] == (True, "1 foreign identifier(s): ENT-19022@chunk[3]")
    assert rows[1] == (False, None)
    assert rows[2] == (None, None)


def test_the_leak_columns_are_added_to_a_pre_existing_steps_table(tmp_path):
    """Every trace store written before R13 already exists on disk. The schema
    is applied with CREATE TABLE IF NOT EXISTS, which is a no-op against an
    older table -- so the new columns must arrive by ALTER, or the first
    INSERT after the upgrade fails against the operator's real database."""
    import pathlib

    schema = (pathlib.Path(__file__).parents[1] / "rlm" / "schema.sql").read_text()
    pre_r13 = (schema.split("-- Migration for stores")[0]
               .replace("    leak_detected BOOLEAN,\n", "")
               .replace("    leak_detail TEXT,\n", ""))
    assert "leak_" not in pre_r13
    db = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db))
    con.execute(pre_r13)
    assert "leak_detected" not in [r[0] for r in con.execute("DESCRIBE steps").fetchall()]
    con.execute(schema)  # re-applied, the way TraceLogger does at start()
    cols = [r[0] for r in con.execute("DESCRIBE steps").fetchall()]
    assert "leak_detected" in cols and "leak_detail" in cols
    con.close()


async def test_the_rotation_stamp_rides_on_the_step_that_triggered_it(tmp_path):
    """§5 C4 (v0.2.6): a slot-pool rotation is logged as a lifecycle event AND
    stamped on the step that triggered it. The lifecycle log is deleted for the
    S3 gate, so the trace has to carry the fact by itself -- NULL on every step
    that triggered nothing, the rotation's 1-based index on the one that did."""
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    ep = str(uuid.uuid4())
    tl.open_episode({"episode_id": ep, "task_id": "t", "task_hash": "h",
                     "config_snapshot": {}})
    tl.put_step({"episode_id": ep, "step_idx": 0, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.ERROR,
                 "server_rotation": 1})
    tl.put_step({"episode_id": ep, "step_idx": 1, "actor": Actor.LEAF,
                 "action_type": ActionType.LLM_CALL, "status": StepStatus.OK})
    await tl.drain()
    await tl.aclose()

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    assert con.execute("SELECT server_rotation FROM steps ORDER BY step_idx"
                       ).fetchall() == [(1,), (None,)]


def test_the_rotation_column_is_added_to_a_pre_existing_steps_table(tmp_path):
    import pathlib

    schema = (pathlib.Path(__file__).parents[1] / "rlm" / "schema.sql").read_text()
    pre = (schema.split("-- Migration for stores")[0]
           .replace("    server_rotation INTEGER,\n", ""))
    assert "server_rotation" not in pre
    con = duckdb.connect(str(tmp_path / "old.duckdb"))
    con.execute(pre)
    assert "server_rotation" not in [r[0] for r in con.execute("DESCRIBE steps").fetchall()]
    con.execute(schema)
    assert "server_rotation" in [r[0] for r in con.execute("DESCRIBE steps").fetchall()]
    con.close()


async def test_update_episode_metrics_sets_only_supplied_columns(tmp_path):
    # episode_id must be a real UUID string (schema.sql:21 -- episodes.episode_id
    # is typed UUID); a non-UUID literal fails the INSERT with a DuckDB
    # ConversionException that the writer loop swallows silently (no
    # `lifecycle` wired up here), which would otherwise masquerade as this
    # test failing for the wrong reason.
    ep = "33333333-3333-3333-3333-333333333333"
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    try:
        tl.open_episode(_episode_row(ep))
        tl.update_episode_metrics(ep, pkg_temp_c_start=61.5)
        tl.update_episode_metrics(ep, avg_power_w=117.2,
                                   energy_j=42_000.0, pkg_temp_c_end=88.0)
        await tl.drain()
        row = tl.monitor().execute(
            "SELECT pkg_temp_c_start, pkg_temp_c_end, avg_power_w, energy_j "
            "FROM episodes WHERE episode_id = ?", [ep]).fetchone()
        # pytest.approx: the columns are REAL (float32, schema.sql:34-37), so
        # a round trip through storage does not preserve every float64 bit
        # pattern -- 117.2 comes back 117.19999694824219.
        assert row == pytest.approx((61.5, 88.0, 117.2, 42_000.0))
    finally:
        await tl.aclose()


async def test_mark_superseded_links_the_rerun(tmp_path):
    tl = TraceLogger(tmp_path / "t.duckdb", tmp_path / "blobs")
    await tl.start()
    try:
        tl.open_episode(_episode_row("11111111-1111-1111-1111-111111111111"))
        tl.open_episode(_episode_row("22222222-2222-2222-2222-222222222222"))
        tl.mark_superseded("11111111-1111-1111-1111-111111111111",
                            "22222222-2222-2222-2222-222222222222")
        await tl.drain()
        got = tl.monitor().execute(
            "SELECT superseded_by FROM episodes WHERE episode_id = ?",
            ["11111111-1111-1111-1111-111111111111"]).fetchone()[0]
        assert str(got) == "22222222-2222-2222-2222-222222222222"
    finally:
        await tl.aclose()
