"""C6 TraceLogger -- single-writer asyncio queue consumer over DuckDB.

Contracts (ARCHITECTURE.md v0.2.2 §5 C6 / §6, I4: "a run that is not logged
did not happen"):
  * one writer task, fed by an asyncio.Queue
  * exactly ONE transaction commit per step
  * blobs fsync'd BEFORE the row that references them (the durability
    contract: a crash orphans a blob -- recoverable ground truth on disk --
    never leaves a row pointing at a file that is not there)
  * blobs are plain files in a per-episode directory, referenced by
    episode-relative path; rolled into ONE blobs.parquet at export
    (never parquet-per-blob: measured strictly dominated, see recipes doc)
  * step_idx assigned by the writer in commit order unless the caller
    already supplies one
  * monitoring reads are in-process only (writer_con.cursor(), never a
    second duckdb.connect(read_only=True) -- Windows excludes ALL other
    processes from a file a writer holds open, so cursor() on the writer's
    own connection is the only way to see anything while a run is live)

This module is schema-adjacent only: it imports duckdb + stdlib, nothing
that reaches a model server (checks/test_import_rules.py enforces this).
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import duckdb

_SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")


def _schema_sql() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def utc_now() -> dt.datetime:
    """Naive UTC. DuckDB TIMESTAMP is tz-less; store one clock everywhere.
    Never bind a tz-AWARE datetime: DuckDB silently converts it to
    session-local wall clock instead of UTC."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


# --- text safety: DuckDB VARCHAR/JSON reject a lone surrogate with an
# opaque pybind11 cast error. Everything reaching a text column goes
# through this first (aligned with rlm.config.safe_text's algorithm; kept
# local and None-tolerant so trace.py has no dependency on rlm.config). ---
def safe_text(s: str | None) -> str | None:
    if s is None:
        return None
    return s.encode("utf-8", "backslashreplace").decode("utf-8")


def _scrub(o: Any) -> Any:
    if isinstance(o, str):
        return safe_text(o)
    if isinstance(o, dict):
        return {_scrub(k): _scrub(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_scrub(v) for v in o]
    return o


def safe_json(obj: Any) -> str:
    """Canonical JSON for config_snapshot (stable field order => stable
    hashing). Scrub BEFORE dumping: dumping first and scrubbing after turns
    a lone surrogate into the JSON escape \\udcff, which DuckDB then
    rejects -- this order bug is what cascaded 240 swallowed errors in the
    probe (episodes insert failed, every step then failed the FK)."""
    return json.dumps(_scrub(obj), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str)


# --- blob container: ASCII JSON header line + raw byte payload, byte-exact.
# Available for callers (e.g. the C1/C4 bridge) that want to pack several
# named streams (stdout/stderr/repr/traceback) into ONE blob before handing
# it to put_step. TraceLogger itself does not invoke this -- put_step's
# `blobs` values are written to disk exactly as given, so a caller that
# passes raw bytes for one ref gets byte-exact round trip with no wrapping. ---
BLOB_MAGIC = b"RLMBLOB1"


def pack_blob(streams: dict[str, bytes]) -> bytes:
    header = json.dumps({"v": 1, "streams": [[k, len(v)] for k, v in streams.items()]},
                         ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return b"".join([BLOB_MAGIC, b"\n", header, b"\n", *streams.values()])


def unpack_blob(buf: bytes) -> dict[str, bytes]:
    if not buf.startswith(BLOB_MAGIC + b"\n"):
        raise ValueError("not an RLM blob")
    nl = buf.index(b"\n", len(BLOB_MAGIC) + 1)
    header = json.loads(buf[len(BLOB_MAGIC) + 1:nl].decode("ascii"))
    off, out = nl + 1, {}
    for name, n in header["streams"]:
        out[name] = buf[off:off + n]
        off += n
    return out


# --- mid-frame-death convention (this plan's addition beyond spec/probe):
# a repl_exec step whose result frame never arrives from the sandbox is
# written with these exact values, so C1/C4 and every future reader share
# one vocabulary. The detection itself lives in C1/C4 (the bridge reader);
# trace.py only owns the constants so the strings cannot drift. ---
SANDBOX_DIED_MID_CELL = "sandbox_died_mid_cell"    # steps.error_detail
SANDBOX_DEATH_REASON = "sandbox_death"             # episodes.outcome_reason

# The other way a result frame never arrives: the sandbox is still ALIVE but the
# protocol stream is unparseable. Model code can reach the protocol fd
# (`os.write(101, b'garbage')`) and desync it deliberately, so this is ordinary
# model behaviour to be classified, not a crash. Both statuses are `error`; only
# the reason distinguishes "the sandbox died" from "the sandbox corrupted the
# channel", and C5/§6 need that distinction to attribute the episode.
BRIDGE_DESYNC = "bridge_desync"                    # steps.error_detail
BRIDGE_DESYNC_REASON = "bridge_desync"             # episodes.outcome_reason


EPISODE_OPEN_COLS = (
    "episode_id", "task_id", "task_hash", "tokenized_task_len", "started_at",
    "dry_run", "scaffold_instance_id", "sandbox_pid", "config_snapshot",
    "scaffold_git_sha", "benchmark_version", "pkg_temp_c_start",
)
INSERT_EPISODE = (f"INSERT INTO episodes ({','.join(EPISODE_OPEN_COLS)}) "
                   f"VALUES ({','.join('?' * len(EPISODE_OPEN_COLS))})")
_EPISODE_OPEN_DEFAULTS: dict[str, Any] = {
    "tokenized_task_len": None,
    "dry_run": False,
    "scaffold_instance_id": "",
    "sandbox_pid": None,
    "scaffold_git_sha": "",
    "benchmark_version": None,
    "pkg_temp_c_start": None,
}

STEP_COLS = (
    "episode_id", "step_idx", "parent_step_idx", "call_id", "retry_idx", "depth",
    "actor", "action_type", "status", "error_detail", "action_payload",
    "root_view_hash", "root_request_ref", "observation_view", "observation_full_ref",
    "tokens_in", "tokens_out", "tokens_cached", "slot_id",
    "t_dispatch", "t_first_byte", "t_end",
    "latency_queue_ms", "latency_prefill_ms", "latency_decode_ms",
    # R13's foreign-string detector, recorded per leaf answer (§5 C4). NULL
    # means NOT CHECKED -- see schema.sql for why that is not the same as
    # False, and src/rlm/serve/leakcheck.py for the 2.2% bound that forbids the phrase
    # "leak-free" anywhere near a column of Falses.
    "leak_detected", "leak_detail",
    # The slot-pool rotation this step triggered (§5 C4, v0.2.6). NULL on every
    # step that triggered none, which is all but two per 200K corpus.
    "server_rotation",
)
INSERT_STEP = (f"INSERT INTO steps ({','.join(STEP_COLS)}) "
               f"VALUES ({','.join('?' * len(STEP_COLS))})")
_STEP_DEFAULTS: dict[str, Any] = {
    "parent_step_idx": None, "call_id": None, "retry_idx": 0, "depth": 0,
    "error_detail": None, "action_payload": None, "root_view_hash": None,
    "root_request_ref": None, "observation_view": None, "observation_full_ref": None,
    "tokens_in": None, "tokens_out": None, "tokens_cached": None, "slot_id": None,
    "t_dispatch": None, "t_first_byte": None, "t_end": None,
    "latency_queue_ms": None, "latency_prefill_ms": None, "latency_decode_ms": None,
    "leak_detected": None, "leak_detail": None, "server_rotation": None,
}
_STEP_TEXT_FIELDS = ("error_detail", "action_payload", "observation_view",
                     "leak_detail")


def blob_rel(episode_id: Any, step_idx: int, col: str) -> str:
    """The episode-relative path `put_step` will write `blobs[col]` to.

    Public and pure so a caller that supplies its own `step_idx` can name a
    blob BEFORE the writer commits it -- which is what lets the episode
    runner point a terminal `final` step at the `root_request_ref` blob its
    parent turn already wrote, instead of storing a second copy of a 32K-token
    request. The format lives here, once: a caller re-deriving it from a
    literal would drift silently the day this changes."""
    return f"{episode_id}/step-{step_idx:06d}.{col}.blob"


@dataclass(slots=True)
class _OpenMsg:
    row: dict[str, Any]


@dataclass(slots=True)
class _StepMsg:
    row: dict[str, Any]
    blobs: dict[str, bytes] = field(default_factory=dict)


@dataclass(slots=True)
class _CloseMsg:
    episode_id: Any
    outcome: Any
    outcome_reason: Any
    final_answer_ref: Any
    ended_at: dt.datetime
    #: Task 12: an interactive episode's `env_actions` is only known once the
    #: episode is over, unlike every other `config_snapshot` field (chosen
    #: upfront, at `open_episode`). `None` (every OTHER caller) leaves the
    #: column exactly as `open_episode` wrote it -- the same "supplied
    #: columns only" convention `update_episode_metrics`/`_commit_metrics`
    #: already uses for the two timestamps that straddle an episode's life.
    config_snapshot: Any = None


@dataclass(slots=True)
class _MetricsMsg:
    episode_id: Any
    pkg_temp_c_start: float | None
    pkg_temp_c_end: float | None
    avg_power_w: float | None
    energy_j: float | None


@dataclass(slots=True)
class _SupersedeMsg:
    old_episode_id: Any
    new_episode_id: Any


@dataclass(slots=True)
class _Shutdown:
    pass


class TraceLogger:
    def __init__(self, db_path: str | os.PathLike, blob_root: str | os.PathLike,
                 *, queue_max: int = 0, lifecycle: Any = None) -> None:
        self.db_path = pathlib.Path(db_path)
        self.blob_root = pathlib.Path(blob_root)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self._con: duckdb.DuckDBPyConnection | None = None
        self._task: asyncio.Task | None = None
        # exactly one thread => the single-writer invariant survives the
        # executor. MEASURED: running the commit inline in the coroutine
        # stalled the event loop for 2,344 ms straight; one thread keeps
        # loop lag at ~5.5 ms median (below the Windows timer floor) with
        # identical throughput.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="c6-writer")
        self._next_idx: dict[str, int] = {}
        self._lifecycle = lifecycle

    async def start(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self._con = await self._run(self._connect)
        self._task = asyncio.create_task(self._writer_loop(), name="c6-trace-writer")

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(self.db_path))
        con.execute(_schema_sql())
        return con

    async def _run(self, fn):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn)

    async def aclose(self) -> None:
        if self._task is None:
            return
        await self.queue.put(_Shutdown())
        await self._task
        self._task = None
        con = self._con
        if con is not None:
            await self._run(lambda: (con.execute("CHECKPOINT"), con.close()))
            self._con = None
        self._pool.shutdown(wait=True)

    async def drain(self) -> None:
        """Block until everything queued so far has been committed."""
        await self.queue.join()

    # -- enqueue methods: synchronous, never block (unbounded queue). --

    def open_episode(self, ep: dict[str, Any]) -> None:
        self.queue.put_nowait(_OpenMsg(dict(ep)))

    def put_step(self, step: dict[str, Any], blobs: dict[str, bytes] | None = None) -> None:
        self.queue.put_nowait(_StepMsg(dict(step), dict(blobs or {})))

    def close_episode(self, episode_id: Any, outcome: Any,
                       outcome_reason: Any = None, final_answer_ref: Any = None,
                       config_snapshot: Any = None) -> None:
        self.queue.put_nowait(
            _CloseMsg(episode_id, outcome, outcome_reason, final_answer_ref,
                      utc_now(), config_snapshot))

    def update_episode_metrics(self, episode_id: Any, *, pkg_temp_c_start: float | None = None,
                                pkg_temp_c_end: float | None = None,
                                avg_power_w: float | None = None,
                                energy_j: float | None = None) -> None:
        """Cost-scorecard columns (schema.sql:34-37). Only the supplied
        (non-None) columns are written -- callers stamp pkg_temp_c_start at
        launch and the rest at teardown, in separate calls, and neither call
        should clobber columns the other one owns."""
        self.queue.put_nowait(
            _MetricsMsg(episode_id, pkg_temp_c_start, pkg_temp_c_end,
                        avg_power_w, energy_j))

    def mark_superseded(self, old_episode_id: Any, new_episode_id: Any) -> None:
        """§8's rerun rule: `old_episode_id` was superseded by `new_episode_id`
        (schema.sql:33, episodes.superseded_by)."""
        self.queue.put_nowait(_SupersedeMsg(old_episode_id, new_episode_id))

    # -- monitoring: in-process ONLY (DuckDB holds an exclusive file lock on
    #    Windows -- connect/connect-read-only/ATTACH/copyfile all fail from a
    #    second process). con.cursor() is a sibling on the same
    #    DatabaseInstance; duckdb.connect(path, read_only=True) INSIDE the
    #    writer process fails ("different configuration than existing
    #    connections"), so cursor() is the only in-process option too. --
    def monitor(self) -> duckdb.DuckDBPyConnection:
        assert self._con is not None, "TraceLogger not started"
        return self._con.cursor()

    def read_blob(self, rel: str) -> bytes:
        return (self.blob_root / rel).read_bytes()

    def export_bundle(self, dest: str | os.PathLike, run_filter_sql: str = "TRUE",
                       *, blob_scope: str | None = None) -> pathlib.Path:
        """The append-only bundle external readers get: a second process
        cannot open the live .duckdb file at all, so this -- episodes.parquet
        + steps.parquet + blobs.parquet -- is the only way anyone else ever
        sees a run in progress. Written synchronously on the caller's
        thread (infrequent -- once per episode close -- unlike per-step
        commits, which must never run inline).

        DRAIN-BEFORE-EXPORT CONTRACT: this reads whatever is already
        committed in `episodes`/`steps` at the moment it runs. It does NOT
        wait for the writer queue -- a `close_episode()` call sitting
        unprocessed in the queue is invisible here, so calling this right
        after `close_episode()` with no intervening `await self.drain()`
        exports a stale row (outcome/ended_at still NULL) for the episode
        that was just closed. Callers MUST `await tl.drain()` after the
        last `open_episode`/`put_step`/`close_episode` call they care about
        and before calling this, every time -- including at every episode
        close per D21, where a perpetually-one-drain-behind bundle would
        make external monitoring of a multi-hour bench read wrong outcomes.

        `blob_scope` narrows the blob half to ONE episode's directory. Without
        it the blob glob walks every episode on disk no matter how tight
        `run_filter_sql` is -- so a per-episode bundle would still pay O(all
        blobs), which is the expensive dimension and the whole reason to
        scope. Blob `rel` paths are unchanged either way (they stay relative
        to `blob_root`), so a scoped bundle joins to `steps.parquet` exactly
        like a full one.

        SQL INJECTION CAVEAT: `run_filter_sql` is interpolated directly into
        the COPY statements below, not parameterised -- DuckDB's COPY/FROM
        clauses do not accept bound parameters in this position. Fine for a
        CLI-driven local tool operating under R6's single-user threat model;
        would need validation or a safe expression builder before any
        run-id sourced from a task file or external input ever reaches this
        argument."""
        out = pathlib.Path(dest)
        out.mkdir(parents=True, exist_ok=True)
        con = self._con
        assert con is not None, "TraceLogger not started"
        blob_glob = (self.blob_root / (blob_scope or "*") / "*").as_posix()
        root_re = self.blob_root.as_posix().replace("\\", "/")
        cur = con.cursor()
        try:
            cur.execute(
                f"COPY (SELECT * FROM episodes WHERE {run_filter_sql}) "
                f"TO '{(out / 'episodes.parquet').as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)")
            cur.execute(
                f"COPY (SELECT s.* FROM steps s JOIN episodes e USING (episode_id) "
                f"WHERE {run_filter_sql}) "
                f"TO '{(out / 'steps.parquet').as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)")
            # normalise separators FIRST, then strip blob_root: read_blob()
            # returns native (backslash) paths on Windows.
            cur.execute(
                f"COPY (SELECT replace(replace(filename,'\\','/'),'{root_re}/','') AS rel, "
                f"      content FROM read_blob('{blob_glob}')) "
                f"TO '{(out / 'blobs.parquet').as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)")
        finally:
            cur.close()
        return out

    async def _writer_loop(self) -> None:
        while True:
            msg = await self.queue.get()
            try:
                if isinstance(msg, _Shutdown):
                    self.queue.task_done()
                    return
                await self._run(lambda m=msg: self._commit(m))
            except Exception as exc:   # never kill the scaffold (I4 needs the
                                        # writer alive; report, don't raise)
                if self._lifecycle is not None:
                    try:
                        self._lifecycle.event(
                            "trace_write_failure", error=repr(exc),
                            msg_type=type(msg).__name__)
                    except Exception:
                        pass
            finally:
                if not isinstance(msg, _Shutdown):
                    self.queue.task_done()

    # everything below runs on the single writer thread ---------------------

    def _commit(self, msg: Any) -> None:
        con = self._con
        assert con is not None
        if isinstance(msg, _OpenMsg):
            self._commit_open(con, msg)
        elif isinstance(msg, _StepMsg):
            self._commit_step(con, msg)
        elif isinstance(msg, _CloseMsg):
            self._commit_close(con, msg)
        elif isinstance(msg, _MetricsMsg):
            self._commit_metrics(con, msg)
        elif isinstance(msg, _SupersedeMsg):
            self._commit_supersede(con, msg)
        else:
            raise TypeError(f"unknown trace message: {type(msg)!r}")

    def _commit_open(self, con: duckdb.DuckDBPyConnection, msg: _OpenMsg) -> None:
        row = dict(msg.row)
        episode_id = row["episode_id"]
        (self.blob_root / str(episode_id)).mkdir(parents=True, exist_ok=True)
        self._next_idx[str(episode_id)] = 0
        for k, v in _EPISODE_OPEN_DEFAULTS.items():
            row.setdefault(k, v)
        row.setdefault("started_at", utc_now())
        values = [
            episode_id,
            safe_text(row["task_id"]),
            safe_text(row["task_hash"]),
            row["tokenized_task_len"],
            row["started_at"],
            row["dry_run"],
            safe_text(row["scaffold_instance_id"]),
            row["sandbox_pid"],
            safe_json(row.get("config_snapshot", {})),
            safe_text(row["scaffold_git_sha"]),
            safe_text(row["benchmark_version"]),
            row["pkg_temp_c_start"],
        ]
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(INSERT_EPISODE, values)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def _commit_step(self, con: duckdb.DuckDBPyConnection, msg: _StepMsg) -> None:
        row = dict(msg.row)
        episode_id = row["episode_id"]
        key = str(episode_id)
        idx = row.get("step_idx")
        if idx is None:
            idx = self._next_idx.get(key, 0)
        # 1. blobs first, fsync'd: a crash can only ever ORPHAN a blob,
        #    never leave a row pointing at a file that is not there.
        for col, content in msg.blobs.items():
            rel = blob_rel(episode_id, idx, col)
            self._write_blob(rel, content)
            row[col] = rel
        for k, v in _STEP_DEFAULTS.items():
            row.setdefault(k, v)
        row["step_idx"] = idx
        for field_name in _STEP_TEXT_FIELDS:
            row[field_name] = safe_text(row.get(field_name))
        values = [row[c] for c in STEP_COLS]
        # 2. then the row, in its own transaction
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(INSERT_STEP, values)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        self._next_idx[key] = idx + 1

    def _commit_close(self, con: duckdb.DuckDBPyConnection, msg: _CloseMsg) -> None:
        con.execute("BEGIN TRANSACTION")
        try:
            sets = ["ended_at=?", "outcome=?", "outcome_reason=?", "final_answer_ref=?"]
            params: list[Any] = [msg.ended_at, msg.outcome, safe_text(msg.outcome_reason),
                                 msg.final_answer_ref]
            if msg.config_snapshot is not None:
                sets.append("config_snapshot=?")
                params.append(safe_json(msg.config_snapshot))
            con.execute(
                f"UPDATE episodes SET {', '.join(sets)} WHERE episode_id=?",
                [*params, msg.episode_id])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        self._next_idx.pop(str(msg.episode_id), None)

    def _commit_metrics(self, con: duckdb.DuckDBPyConnection, msg: _MetricsMsg) -> None:
        sets, params = [], []
        for col in ("pkg_temp_c_start", "pkg_temp_c_end", "avg_power_w", "energy_j"):
            val = getattr(msg, col)
            if val is not None:
                sets.append(f"{col} = ?")
                params.append(val)
        if not sets:
            return
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"UPDATE episodes SET {', '.join(sets)} WHERE episode_id = ?",
                        [*params, msg.episode_id])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def _commit_supersede(self, con: duckdb.DuckDBPyConnection, msg: _SupersedeMsg) -> None:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("UPDATE episodes SET superseded_by = ? WHERE episode_id = ?",
                        [msg.new_episode_id, msg.old_episode_id])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    def _write_blob(self, rel: str, data: bytes) -> None:
        p = self.blob_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())   # the ordering promise, not an OS-cache accident


# --- crash recovery (§6): tombstone, never resume. Runs at startup, BEFORE
# the TraceLogger for the new run opens the DB -- it is itself the single
# writer at that point. Process-killing (surviving sandbox_pid) and the
# servers-idle quiesce wait are C1/C5 integration concerns layered on by the
# caller (rlm run); this function owns only the DB-level scan + tombstone. ---
def recover_orphans(db_path: str | os.PathLike, lifecycle: Any = None) -> list[str]:
    db_path = pathlib.Path(db_path)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(_schema_sql())
        rows = con.execute(
            "SELECT episode_id FROM episodes WHERE outcome IS NULL "
            "ORDER BY started_at").fetchall()
        ids = [str(r[0]) for r in rows]
        if not ids:
            return []
        if lifecycle is not None:
            lifecycle.event("recovery_action", action="scan", orphans=len(ids))
        for episode_id in ids:
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(
                    "UPDATE episodes SET outcome='error', "
                    "outcome_reason='orphaned_at_recovery', "
                    "ended_at=COALESCE(ended_at, ?) WHERE episode_id=?",
                    [utc_now(), episode_id])
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            if lifecycle is not None:
                lifecycle.event("recovery_action", action="tombstoned",
                                 episode_id=episode_id)
        return ids
    finally:
        con.execute("CHECKPOINT")
        con.close()


def sweep_orphan_blobs(db_path: str | os.PathLike, blob_root: str | os.PathLike) -> list[str]:
    """Blobs whose referencing row never committed (at most one per crashed
    episode). Reported, never deleted -- they are the lost step's ground
    truth. Raises if the reverse is ever true, which blob-before-row
    ordering makes impossible."""
    blob_root = pathlib.Path(blob_root)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        refs = {r[0] for r in con.execute(
            "SELECT observation_full_ref FROM steps WHERE observation_full_ref IS NOT NULL "
            "UNION SELECT root_request_ref FROM steps WHERE root_request_ref IS NOT NULL "
            "UNION SELECT final_answer_ref FROM episodes WHERE final_answer_ref IS NOT NULL"
        ).fetchall()}
    finally:
        con.close()
    on_disk = {p.relative_to(blob_root).as_posix() for p in blob_root.rglob("*.blob")}
    dangling = sorted(refs - on_disk)
    if dangling:
        raise RuntimeError(f"trace corrupt: {len(dangling)} rows reference missing "
                            f"blobs: {dangling[:5]}")
    return sorted(on_disk - refs)
