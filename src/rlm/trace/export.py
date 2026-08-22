"""`rlm export`: the parquet + blob bundle a FOREIGN reader gets.

The bundle is what someone who was not here uses to check a published number.
Its whole value is being self-contained -- so building it must depend on the
trace store and nothing else.

WHY THIS IS ITS OWN MODULE. Same family as `src/rlm/trace/replay.py`, same reason: this
is offline re-derivation, and inside `src/rlm/cli.py` nothing stopped it growing a
dependency on a live server. Here §5's dependency-rule lint
(`tests/test_import_rules.py` ISOLATED) forbids `httpx`, `rlm.serve.dispatcher` and
`rlm.serve.rootclient` on every run. An export that needed a running server would be
an export nobody else could reproduce, which defeats the point of shipping one.

The `async` here is the TraceLogger's lifecycle (`start`/`aclose`), not network
I/O -- worth saying, because `async` reads like a server at a glance.

`cmd_export` stays in `src/rlm/cli.py` with the other verb handlers.

Extracted from `src/rlm/cli.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import duckdb

from rlm.config import Config
from rlm.errors import EXIT_OK, EXIT_REFUSED, ConfigError
from rlm.trace import TraceLogger


def _export_filter(ident: str, *, by_run: bool) -> str:
    """The SQL predicate, with `ident` interpolated -- which is safe ONLY
    because every caller validated it as a UUID first.

    `trace.export_bundle`'s own docstring flags this: DuckDB does not accept a
    bound parameter in a COPY's FROM clause, so the filter is a string. A
    canonical `str(uuid.UUID(...))` cannot carry a quote, a comment or a
    semicolon, which is what makes the interpolation a non-issue rather than a
    caveat to remember.
    """
    if by_run:
        return f"json_extract_string(config_snapshot, '$.bench.run_id') = '{ident}'"
    return f"episode_id = '{ident}'"


async def _export(cfg: Config, ident: str, dest: Path, out, err) -> int:
    trace = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root)
    try:
        await trace.start()
    except (duckdb.Error, OSError) as exc:
        print(f"refused: cannot open the trace store at {cfg.trace.db_path} "
              f"({exc}). `rlm export` reads a CLOSED store: on Windows DuckDB "
              f"excludes every other connection from a file its writer holds "
              f"open, so this means the run is still live. Let it finish and "
              f"export then — a bundle taken from a half-written store is a "
              f"bundle of a different run.", file=err)
        return EXIT_REFUSED
    try:
        con = trace.monitor()
        sql = ("SELECT CAST(episode_id AS VARCHAR), CAST(config_snapshot AS VARCHAR) "
               "FROM episodes WHERE {} ORDER BY started_at, episode_id")
        rows = con.execute(
            sql.format("json_extract_string(config_snapshot, '$.bench.run_id') = ?"),
            [ident]).fetchall()
        by_run = bool(rows)
        if not rows:
            # An episode_id, then. Tried SECOND on purpose: a run_id is what an
            # operator has after `rlm bench`, and the two id spaces are both
            # UUIDs, so the order decides which one wins a (vanishingly
            # unlikely) collision. The run is the more useful answer.
            rows = con.execute(sql.format("CAST(episode_id AS VARCHAR) = ?"),
                                [ident]).fetchall()
        if not rows:
            print(f"nothing to export: no episode and no bench run in "
                  f"{cfg.trace.db_path} matches {ident}", file=err)
            return EXIT_REFUSED

        episode_ids = [str(r[0]) for r in rows]
        dest.mkdir(parents=True, exist_ok=True)
        where = _export_filter(ident, by_run=by_run)
        cur = con.cursor()
        try:
            # The two row tables, filtered, ONCE. `export_bundle` is not reused
            # here: its blob half globs exactly one episode directory, which
            # cannot express "this run's 360 episodes", and globbing `*`
            # instead would drag in every other run on disk.
            cur.execute(
                f"COPY (SELECT * FROM episodes WHERE {where}) "
                f"TO '{(dest / 'episodes.parquet').as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)")
            cur.execute(
                f"COPY (SELECT s.* FROM steps s JOIN episodes e USING (episode_id) "
                f"WHERE {where}) "
                f"TO '{(dest / 'steps.parquet').as_posix()}' "
                f"(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)")
        finally:
            cur.close()

        # The blobs, as DIRECTORIES rather than a parquet BLOB column: the
        # `*_ref` values in steps.parquet are paths relative to `blob_root`, so
        # copying each episode's directory under `<dest>/blobs/` keeps every
        # reference resolvable with no join and no decoder. Per episode, so the
        # cost is this run's blobs and not the whole store's.
        blob_root = Path(cfg.trace.blob_root)
        copied = 0
        for episode_id in episode_ids:
            source = blob_root / episode_id
            if not source.is_dir():
                continue
            target = dest / "blobs" / episode_id
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            copied += 1

        digest = hashlib.sha256()
        for _episode_id, snapshot in sorted((str(r[0]), r[1] or "") for r in rows):
            digest.update(snapshot.encode("utf-8", "replace"))
        (dest / "bundle-manifest.json").write_text(json.dumps({
            "run_id": ident if by_run else None,
            "resolved_by": "run_id" if by_run else "episode_id",
            "episode_ids": episode_ids,
            "n_episodes": len(episode_ids),
            "blob_dirs": copied,
            "config_snapshot_sha256": digest.hexdigest(),
            "source_db": str(cfg.trace.db_path),
            "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"),
            "layout": ("episodes.parquet + steps.parquet (both filtered to this "
                       "bundle) and blobs/<episode_id>/<file>, where every "
                       "steps.*_ref value is a path relative to blobs/"),
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"exported {len(episode_ids)} episode(s) "
              f"({'run ' + ident if by_run else 'episode ' + ident}) to {dest}",
              file=out)
        return EXIT_OK
    finally:
        await trace.aclose()
