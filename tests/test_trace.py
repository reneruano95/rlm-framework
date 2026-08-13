import datetime as dt
import json
import uuid

import duckdb
import pytest

from rlm.errors import ActionType, Actor, Outcome, StepStatus
from rlm.trace import TraceLogger, recover_orphans, safe_text


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
