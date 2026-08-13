"""The operator surface (spec §5). THREE verbs, and they are the whole thing:

    rlm validate                     config schema + /props probe + D7 + D27
    rlm run <task-file>              one episode; prints episode_id + outcome
    rlm replay <episode-id> [--online]   §6 replay check + transcript render

`bench` and `export` are later slices (S4). **Non-goals stay non-goals: no
daemon, no REST API, no web UI, no interactive chat mode.** If a future change
adds a fourth verb, it belongs to a slice that argued for it.

WHAT REPLAY VERIFIES, AND WHAT IT CANNOT. Replay verifies PROMPT ASSEMBLY, not
decoding. Greedy decoding is not reproducible on this box: three identical
requests at temperature 0 with a fixed seed produced three different outputs
(measured). §8 says the same thing structurally -- continuous batching breaks
bitwise reproducibility at fixed seed. So replay never re-generates anything
and never compares model output; it re-derives the REQUEST and checks it three
ways:

  (i)  offline, the default and the S3 gate condition -- rehash the stored
       `root_request_ref` blob and assert it equals `root_view_hash`, then
       re-derive the message array from the trace ALONE and compare it with
       the array that was actually sent. The first check proves the record is
       intact; the second is the standing canary for prompt-assembly drift,
       and it is the one that needs no server and no lifecycle log.
  (ii) `--online`, additionally -- re-POST the re-derived messages to
       /apply-template and assert byte-equality with the stored render, and
       assert the live `props.chat_template` sha256 still matches the one in
       `config_snapshot`.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import duckdb

from rlm.config import Config, load_config
from rlm.dispatcher import LLMDispatcher, MockDispatcher, ServerClient
from rlm.episode import (
    Task,
    assert_props,
    compose_user_message,
    no_cell_observation,
    run_episode,
)
from rlm.errors import ActionType, ConfigError, RlmError, StepStatus
from rlm.lifecycle import Lifecycle
from rlm.rootclient import assistant_prefix, extract_cell
from rlm.sandbox import winproc
from rlm.sandbox.manager import SandboxManager, install_bootstrap
from rlm.trace import TraceLogger, recover_orphans, unpack_blob

EXIT_OK = 0
EXIT_REFUSED = 2       # config/handshake/invariant refusal
EXIT_MISMATCH = 3      # replay found a discrepancy
EXIT_FAILED = 1        # the command itself failed to complete


# --------------------------------------------------------------------------- #
# D27: cache types come from the -lv 4 launch log, never from /props
# --------------------------------------------------------------------------- #

# Verbatim from the probe recipes (§serverapi), against a live b10375 server.
# These two lines exist ONLY at `-lv 4`; the default `-lv 3` omits both.
_KV_LINE = re.compile(
    r"llama_kv_cache: size\s*=\s*([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers,"
    r"\s*(\d+)/(\d+) seqs\), K \((\w+)\):\s*[\d.]+ MiB, V \((\w+)\)")
_FA_LINE = re.compile(r"llama_context: flash_attn\s*=\s*(\w+)")
_BUILD_LINE = re.compile(r"build\s*[:=]\s*(\d+)\s*\(([0-9a-f]+)\)")


def parse_launch_log(path: str | os.PathLike) -> dict[str, Any]:
    """Recover what a llama-server was ACTUALLY launched with, from its own
    `-lv 4` stderr log.

    D27, measured: `/props` CANNOT report KV cache types or flash-attn state.
    Byte-diffing `/props` between a `-ctk q8_0 -ctv q8_0` launch and a
    `-ctk f16 -ctv f16` launch with otherwise identical flags left exactly one
    differing key -- `media_marker`, a per-process random nonce. §4's "assert
    ... cache types" is therefore unimplementable against that endpoint, and
    the assertion moves here. That makes the LAUNCHER part of the scaffold
    contract: `-lv 4`, and stderr redirected to a per-launch file.

    Returns `{}` when the log is missing or carries neither line -- which
    `validate` reports as UNVERIFIED, never as a pass.
    """
    found: dict[str, Any] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        m = _KV_LINE.search(line)
        if m:
            found.update(kv_mib=float(m.group(1)), kv_cells=int(m.group(2)),
                          kv_layers=int(m.group(3)), kv_seqs=int(m.group(5)),
                          type_k=m.group(6).lower(), type_v=m.group(7).lower())
        m = _FA_LINE.search(line)
        if m:
            found["flash_attn"] = m.group(1).lower()
        m = _BUILD_LINE.search(line)
        if m:
            found["build_number"] = m.group(1)
            found["build_commit"] = m.group(2)
    return found


def log_is_current(parsed: dict[str, Any], props: dict | None) -> bool:
    """Is this log from the server that is answering right now?

    A stale log from a previous launch would silently satisfy the cache-type
    assertion -- the exact failure the assertion exists to catch (R11: a server
    that crashed and relaunched with different flags mid-benchmark). The log is
    only trusted when its build line matches the live `/props` build_info. With
    no live probe to compare against there is nothing to cross-check, and the
    caller must say "unverified" rather than "OK".
    """
    if not props:
        return False
    build_info = str(props.get("build_info") or "")
    commit = parsed.get("build_commit")
    number = parsed.get("build_number")
    if not build_info or not (commit or number):
        return False
    return bool((commit and commit in build_info) or (number and number in build_info))


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #


def _lifecycle_path(cfg: Config, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path(cfg.trace.db_path).parent / "lifecycle.jsonl"


def _scaffold_git_sha() -> str:
    """§6 records which scaffold wrote the episode. A dirty tree is marked,
    not hidden: an unrecorded local edit is exactly the drift the column
    exists to catch."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False, timeout=10).stdout.strip()
        if not sha:
            return ""
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=False, timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return ""


def _build_dispatcher(cfg: Config, task_path: Path | None):
    """Real C4, or the dry-run mock fed from the task file's own fixtures."""
    if cfg.scaffold.dispatcher == "real":
        return LLMDispatcher.from_config(cfg), True
    fixtures: dict[str, str] = {}
    if task_path is not None:
        with contextlib.suppress(OSError, json.JSONDecodeError):
            raw = json.loads(task_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                fixtures = dict(raw.get("fixtures") or {})
    return MockDispatcher(fixtures, parallel=cfg.scaffold.dispatch_concurrency), False


# --------------------------------------------------------------------------- #
# §6 crash recovery -- the half `recover_orphans` deliberately does not own
# --------------------------------------------------------------------------- #


async def _slots_idle(cfg: Config, role: str, lifecycle: Lifecycle,
                       timeout_s: float = 60.0) -> bool:
    """Wait for one server to report every slot idle, draining generation an
    orphaned episode left running. Returns False when that could not be
    established -- an unreachable server (nothing to drain) or a build with
    `/slots` disabled. Either way recovery proceeds: the alternative is
    refusing to start forever because a server will not answer a question
    about work we already know we abandoned."""
    url = f"http://127.0.0.1:{getattr(cfg.servers, role).port}"
    client = ServerClient(url, timeout=10.0)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
        while loop.time() < deadline:
            try:
                slots = await client.slots()
            except Exception as exc:  # noqa: BLE001
                lifecycle.event("quiesce_wait", role=role, state="unverified",
                                 error=repr(exc))
                return False
            if all(not s.get("is_processing") for s in slots):
                lifecycle.event("quiesce_wait", role=role, state="idle",
                                 slots=len(slots))
                return True
            await asyncio.sleep(0.5)
        lifecycle.event("quiesce_wait", role=role, state="timeout")
        return False
    finally:
        await client.aclose()


def _orphan_rows(db_path: Path) -> list[tuple[str, int | None, dt.datetime]]:
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path))
    try:
        return [(str(r[0]), r[1], r[2]) for r in con.execute(
            "SELECT episode_id, sandbox_pid, started_at FROM episodes "
            "WHERE outcome IS NULL ORDER BY started_at").fetchall()]
    finally:
        con.close()


def recover(cfg: Config, lifecycle: Lifecycle) -> list[str]:
    """§6 crash recovery, in full: scan, reap, quiesce, THEN tombstone.

    `rlm.trace.recover_orphans` is DB-only by design -- process killing and
    the servers-idle wait are C1/C5 integration concerns and would drag both
    into a module that must import neither. This is where they live.

    Resume is rejected as unsound and always will be: the sandbox interpreter
    heap is not stored, and §8's own caveat (continuous batching breaks
    bitwise reproducibility at fixed seed) means a resumed episode could not
    satisfy the state rule even if it were.

    Runs BEFORE the TraceLogger for this run opens the database -- it is
    itself the single writer at that point, which on Windows is not a nicety:
    DuckDB excludes every other process from a file a writer holds open.
    """
    db_path = Path(cfg.trace.db_path)
    orphans = _orphan_rows(db_path)
    if not orphans:
        return []
    for episode_id, pid, started_at in orphans:
        if not pid:
            continue
        # kill_if_ours guards on process creation time vs episodes.started_at:
        # pid reuse is real (post-reboot, or just a busy hour), and killing a
        # reused pid would be a scaffold that damages an unrelated process
        # during its own cleanup.
        reaped = winproc.kill_if_ours(pid, started_at)
        lifecycle.event("recovery_action", action="reap", episode_id=episode_id,
                         pid=pid, reaped=reaped)
    asyncio.run(_quiesce_all(cfg, lifecycle))
    return recover_orphans(db_path, lifecycle)


async def _quiesce_all(cfg: Config, lifecycle: Lifecycle) -> None:
    await _slots_idle(cfg, "root", lifecycle)
    if cfg.scaffold.dispatcher == "real":
        await _slots_idle(cfg, "leaf", lifecycle)


# --------------------------------------------------------------------------- #
# verb: validate
# --------------------------------------------------------------------------- #


async def _confinement_probe(cfg: Config, config_path: Path) -> tuple[bool, str]:
    """D7 as a CHECKED INVARIANT, not a claim.

    Spawn a throwaway AppContainer child and have it try to read the very
    config file this command was pointed at. If it succeeds, the filesystem
    confinement the whole isolation argument rests on is not in force and
    `validate` refuses to start -- because every other guarantee (budgets,
    truncation caps, routing) is only meaningful if model code cannot read
    the config that sets them.

    `install_bootstrap(grant_acl=True)` belongs HERE and only here: the
    runtime must never hand an AppContainer filesystem access as a side
    effect of running a task, but an install/validate step is exactly when
    the one-time grant is supposed to happen.
    """
    if not config_path.is_file():
        # Guard against a probe that proves nothing: a missing file also
        # raises OSError in the child, and would read as a denial.
        return False, (f"sandbox confinement probe is meaningless: {config_path} "
                       f"does not exist, so the child's OSError would prove "
                       f"nothing about confinement.")
    install_bootstrap(cfg.scaffold.sandbox, grant_acl=True)
    manager = SandboxManager()
    probe_id = f"validate-{uuid.uuid4().hex[:8]}"
    try:
        async with manager.session(probe_id, cfg) as session:
            out = await session.exec_cell(
                "try:\n"
                f"    open({str(config_path)!r}, 'rb').read()\n"
                "    print('READABLE')\n"
                "except OSError as e:\n"
                "    print('DENIED', type(e).__name__)\n")
            verdict = out.stdout.strip()
    finally:
        manager.close()
    if verdict.startswith("DENIED") and "FileNotFoundError" not in verdict:
        return True, f"sandbox filesystem confinement: OK ({verdict.lower()})"
    if verdict.startswith("DENIED"):
        return False, (f"sandbox confinement probe is inconclusive: the child "
                       f"reported {verdict!r} for a file that exists on this "
                       f"host, so the denial cannot be attributed to the "
                       f"AppContainer token.")
    return False, (f"sandbox filesystem confinement: FAILED — the AppContainer "
                   f"read {config_path} (probe said {verdict!r}). Refusing to "
                   f"start: model code that can read config.yaml can read every "
                   f"budget and cap it is supposed to be bound by.")


async def _probe_servers(cfg: Config, lifecycle: Lifecycle, out) -> dict[str, dict]:
    probed: dict[str, dict] = {}
    for role in ("root", "leaf"):
        server_cfg = getattr(cfg.servers, role)
        client = ServerClient(f"http://127.0.0.1:{server_cfg.port}", timeout=15.0)
        try:
            props = await client.props()
        finally:
            await client.aclose()
        assert_props(props, server_cfg, role)
        probed[role] = props
        lifecycle.event("server_health", role=role, state="ok",
                         build_info=str(props.get("build_info", "")))
        print(f"{role} server /props: OK (build {props.get('build_info')!r}, "
              f"model {props.get('model_path')!r}, slots {props.get('total_slots')}, "
              f"n_ctx {(props.get('default_generation_settings') or {}).get('n_ctx')})",
              file=out)
    return probed


def _check_cache_types(cfg: Config, probed: dict[str, dict], out, err) -> bool:
    """D27's half of the §4 handshake: cache types + flash-attn, from the
    launch log, cross-checked against the live build so a stale log cannot
    satisfy the assertion. Reports UNVERIFIED rather than passing when the log
    is absent, unparseable, or cannot be tied to a running server."""
    for role in ("root", "leaf"):
        server_cfg = getattr(cfg.servers, role)
        parsed = parse_launch_log(server_cfg.log_path)
        if not parsed:
            print(f"{role} KV cache type: UNVERIFIED — no parseable `-lv 4` "
                  f"launch log at {server_cfg.log_path}. D27: /props cannot "
                  f"report cache types, so nothing here has been checked.",
                  file=out)
            continue
        if not log_is_current(parsed, probed.get(role)):
            print(f"{role} KV cache type: UNVERIFIED — the launch log at "
                  f"{server_cfg.log_path} could not be tied to a live server "
                  f"build, and a stale log from a previous launch would satisfy "
                  f"this check while the running server does not.", file=out)
            continue
        want = server_cfg.cache_type.lower()
        bad = {k: parsed.get(k) for k in ("type_k", "type_v")
               if parsed.get(k) != want}
        if parsed.get("flash_attn") != ("enabled" if server_cfg.flash_attn == "on"
                                         else "disabled"):
            bad["flash_attn"] = parsed.get("flash_attn")
        if bad:
            print(f"{role} KV cache/flash-attn MISMATCH — the launch log says "
                  f"{bad}, config says cache_type={server_cfg.cache_type} "
                  f"flash_attn={server_cfg.flash_attn}. Quantized V-cache "
                  f"hard-requires flash attention; refusing to start.", file=err)
            return False
        print(f"{role} KV cache type: OK (K/V {parsed['type_k']}, flash_attn "
              f"{parsed['flash_attn']}, from the launch log)", file=out)
    return True


def cmd_validate(args) -> int:
    out, err = sys.stdout, sys.stderr
    try:
        cfg = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config refused: {exc}", file=err)
        return EXIT_REFUSED
    print(f"config: OK ({args.config})", file=out)

    lifecycle = Lifecycle(_lifecycle_path(cfg, args.lifecycle_log))
    try:
        try:
            registry = cfg.prompt_registry().load()
        except ConfigError as exc:
            lifecycle.event("config_refused", error=str(exc))
            print(f"prompt registry refused: {exc}", file=err)
            return EXIT_REFUSED
        print(f"prompt registry: OK ({len(registry.hashes()) // 2} prompts pinned)",
              file=out)

        probed: dict[str, dict] = {}
        if args.no_server_probe:
            print("server probe: SKIPPED (--no-server-probe)", file=out)
        else:
            try:
                probed = asyncio.run(_probe_servers(cfg, lifecycle, out))
            except (ConfigError, OSError, RlmError) as exc:
                lifecycle.event("config_refused", error=str(exc))
                print(f"server probe refused: {exc}", file=err)
                return EXIT_REFUSED

        # Runs AFTER the probe: the launch log is only trustworthy when it can
        # be tied to the build that is actually answering (D27).
        if not _check_cache_types(cfg, probed, out, err):
            lifecycle.event("config_refused", error="kv cache type mismatch")
            return EXIT_REFUSED

        try:
            ok, message = asyncio.run(
                _confinement_probe(cfg, Path(args.config).resolve()))
        except RlmError as exc:
            print(f"sandbox confinement probe could not run: {exc}", file=err)
            return EXIT_REFUSED
        if not ok:
            lifecycle.event("config_refused", error="sandbox confinement failed")
            print(message, file=err)
            return EXIT_REFUSED
        print(message, file=out)
        return EXIT_OK
    finally:
        lifecycle.close()


# --------------------------------------------------------------------------- #
# verb: run
# --------------------------------------------------------------------------- #


async def _run_one(cfg: Config, task: Task, lifecycle: Lifecycle, task_path: Path):
    dispatcher, owns_http = _build_dispatcher(cfg, task_path)
    trace = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root, lifecycle=lifecycle)
    await trace.start()
    try:
        result = await run_episode(
            task, cfg, dispatcher=dispatcher, trace=trace, lifecycle=lifecycle,
            scaffold_instance_id=f"{os.getpid()}",
            scaffold_git_sha=_scaffold_git_sha(),
            benchmark_version=cfg.benchmark.version)
        if cfg.trace.export_every_episode:
            # D21. `export_bundle` reads what is COMMITTED, so the drain is
            # not optional: run_episode drains before returning, and nothing
            # is enqueued between there and here.
            await trace.drain()
            trace.export_bundle(Path(cfg.trace.db_path).parent / "bundle")
        return result
    finally:
        await trace.aclose()
        if owns_http:
            await dispatcher.aclose()


def cmd_run(args) -> int:
    out, err = sys.stdout, sys.stderr
    try:
        cfg = load_config(Path(args.config))
        task_path = Path(args.task_file)
        task = Task.from_file(task_path)
    except ConfigError as exc:
        print(f"refused: {exc}", file=err)
        return EXIT_REFUSED

    lifecycle = Lifecycle(_lifecycle_path(cfg, args.lifecycle_log))
    try:
        tombstoned = recover(cfg, lifecycle)
        if tombstoned:
            print(f"recovery: tombstoned {len(tombstoned)} orphaned episode(s)",
                  file=out)
        try:
            result = asyncio.run(_run_one(cfg, task, lifecycle, task_path))
        except ConfigError as exc:
            print(f"refused: {exc}", file=err)
            return EXIT_REFUSED
        except KeyboardInterrupt:
            # C5's path already ran inside run_episode (kill -> cancelled steps
            # -> drain), with outcome_reason=operator_abort recorded.
            print("aborted by operator", file=err)
            return EXIT_FAILED
        print(f"episode_id: {result.episode_id}", file=out)
        print(f"outcome: {result.outcome}"
              + (f" ({result.reason})" if result.reason else ""), file=out)
        if result.final_answer is not None:
            print(f"final_answer: {result.final_answer!r}", file=out)
        return EXIT_OK
    finally:
        lifecycle.close()


# --------------------------------------------------------------------------- #
# verb: replay
# --------------------------------------------------------------------------- #


def _read_episode(cfg: Config, episode_id: str) -> tuple[dict, list[dict]]:
    db_path = Path(cfg.trace.db_path)
    if not db_path.exists():
        raise ConfigError(f"no trace store at {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cur = con.execute("SELECT * FROM episodes WHERE episode_id = ?", [episode_id])
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            raise ConfigError(f"no episode {episode_id} in {db_path}")
        episode = dict(zip(cols, row))
        if isinstance(episode.get("config_snapshot"), str):
            episode["config_snapshot"] = json.loads(episode["config_snapshot"])
        cur = con.execute(
            "SELECT * FROM steps WHERE episode_id = ? ORDER BY step_idx", [episode_id])
        cols = [d[0] for d in cur.description]
        steps = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()
    for step in steps:
        for key in ("call_id", "episode_id"):
            if isinstance(step.get(key), uuid.UUID):
                step[key] = str(step[key])
    return episode, steps


def _rederive_messages(cfg: Config, episode: dict, steps: list[dict],
                        blob_root: Path) -> list[list[dict]]:
    """Rebuild, from the trace ALONE, the message array sent at every root turn.

    Nothing here reads the lifecycle log, the task file, or a server -- that is
    the S3 gate condition, and it is why `compose_user_message` is a pure
    function of trace-recoverable values and why the task's instruction text is
    carried in `config_snapshot`.
    """
    snapshot = episode.get("config_snapshot") or {}
    task_meta = snapshot.get("task") or {}
    registry = cfg.prompt_registry().load()
    system = registry.render_root(task_meta.get("category", "default"))
    max_subcalls = cfg.scaffold.budgets.max_subcalls

    turns = [s for s in steps
             if s["action_type"] == ActionType.REPL_EXEC and s["root_request_ref"]]
    arrays: list[list[dict]] = []
    messages: list[dict] = [{"role": "system", "content": system}]
    for n, step in enumerate(turns, start=1):
        # Sub-calls already spent when this turn's message was composed: the
        # distinct call_ids hanging off every EARLIER turn.
        earlier = {t["step_idx"] for t in turns[:n - 1]}
        spent = {s["call_id"] for s in steps
                 if s["action_type"] == ActionType.LLM_CALL
                 and s["parent_step_idx"] in earlier and s["call_id"]}
        remaining = max(0, max_subcalls - len(spent))
        if n == 1:
            content = compose_user_message(
                turn=1, subcalls_remaining=remaining,
                task_text=task_meta.get("text", ""))
        else:
            prev = turns[n - 2]
            observation = prev["observation_view"]
            if prev["status"] == StepStatus.REJECTED:
                observation = no_cell_observation(cfg)
            content = compose_user_message(
                turn=n, subcalls_remaining=remaining,
                observation=observation if observation is not None else "")
        messages.append({"role": "user", "content": content})
        arrays.append([dict(m) for m in messages])
        rendered = _rendered(blob_root, step["root_request_ref"])
        messages.append({
            "role": "assistant",
            # D26: the template's own tail after the last assistant marker,
            # plus the model's raw reply. Both come from the trace.
            "content": assistant_prefix(rendered) + (step["action_payload"] or ""),
        })
    return arrays


def _blob(blob_root: Path, rel: str) -> dict[str, bytes]:
    return unpack_blob((blob_root / rel).read_bytes())


def _rendered(blob_root: Path, rel: str) -> str:
    return _blob(blob_root, rel)["rendered"].decode("utf-8", "replace")


def _render_transcript(cfg: Config, steps: list[dict], out) -> None:
    langs = cfg.scaffold.cell_extraction.languages
    select = cfg.scaffold.cell_extraction.select
    print("\n--- transcript ---", file=out)
    for step in steps:
        kind = step["action_type"]
        if kind == ActionType.REPL_EXEC:
            cell = extract_cell(step["action_payload"] or "", langs, select)
            head = "cell" if cell else "no cell"
            print(f"\n[{step['step_idx']}] root {kind} ({step['status']}, {head})",
                  file=out)
            if cell:
                print("  " + "\n  ".join(cell.strip().splitlines()[:20]), file=out)
            view = (step["observation_view"] or "").strip()
            if view:
                print("  -> " + "\n     ".join(view.splitlines()[:12]), file=out)
        elif kind == ActionType.LLM_CALL:
            print(f"[{step['step_idx']}] leaf llm_call ({step['status']}, "
                  f"parent {step['parent_step_idx']}, retry {step['retry_idx']}, "
                  f"tokens {step['tokens_in']}/{step['tokens_out']})", file=out)
        else:
            print(f"\n[{step['step_idx']}] FINAL: {step['action_payload']!r}", file=out)


async def _verify_online(cfg: Config, episode: dict, arrays: list[list[dict]],
                          rendered: list[str], out, err) -> bool:
    client = ServerClient(f"http://127.0.0.1:{cfg.servers.root.port}", timeout=60.0)
    try:
        props = await client.props()
        want = (episode.get("config_snapshot") or {}).get("chat_template_sha256")
        got = hashlib.sha256(
            (props.get("chat_template") or "").encode("utf-8")).hexdigest()
        if want and got != want:
            print(f"chat template drift: config_snapshot recorded {want}, the "
                  f"live server serves {got}", file=err)
            return False
        print(f"chat_template sha256: OK ({got[:16]}…)", file=out)
        for n, (messages, stored) in enumerate(zip(arrays, rendered), start=1):
            live = await client.apply_template(
                messages,
                chat_template_kwargs={
                    "enable_thinking": cfg.scaffold.root.enable_thinking})
            if live != stored:
                print(f"apply-template drift at turn {n}: the live server renders "
                      f"{len(live)} bytes, the trace stored {len(stored)}", file=err)
                return False
        print(f"apply-template byte-equality: OK ({len(arrays)} turns)", file=out)
        return True
    finally:
        await client.aclose()


def cmd_replay(args) -> int:
    out, err = sys.stdout, sys.stderr
    try:
        cfg = load_config(Path(args.config))
        episode, steps = _read_episode(cfg, args.episode_id)
    except ConfigError as exc:
        print(f"refused: {exc}", file=err)
        return EXIT_REFUSED

    blob_root = Path(cfg.trace.blob_root)
    print(f"episode {args.episode_id}: outcome={episode['outcome']}"
          + (f" ({episode['outcome_reason']})" if episode["outcome_reason"] else "")
          + (" [DRY RUN]" if episode["dry_run"] else ""), file=out)

    # (i) The state-rule instrument. Dedup by ref: a terminal `final` step
    # points at its parent turn's blob rather than storing a second copy.
    checked: dict[str, str] = {}
    for step in steps:
        ref, want = step["root_request_ref"], step["root_view_hash"]
        if not ref or not want:
            continue
        if checked.get(ref) == want:
            continue
        try:
            rendered = _blob(blob_root, ref)["rendered"]
        except (OSError, KeyError, ValueError) as exc:
            print(f"root_view_hash: hash mismatch — step {step['step_idx']} "
                  f"references {ref}, which could not be read as a request blob "
                  f"({exc})", file=err)
            return EXIT_MISMATCH
        got = hashlib.sha256(rendered).hexdigest()
        if got != want:
            print(f"root_view_hash: hash mismatch at step {step['step_idx']} — "
                  f"steps.root_view_hash says {want}, the stored blob {ref} "
                  f"hashes to {got}", file=err)
            return EXIT_MISMATCH
        checked[ref] = want
    if not checked:
        print("root_view_hash: hash mismatch — this episode stored no root "
              "request at all, so the state rule cannot be checked", file=err)
        return EXIT_MISMATCH
    print(f"root_view_hash: OK ({len(checked)} root turns rehashed offline)", file=out)

    # (i, second half) The prompt-assembly canary.
    turns = [s for s in steps
             if s["action_type"] == ActionType.REPL_EXEC and s["root_request_ref"]]
    try:
        derived = _rederive_messages(cfg, episode, steps, blob_root)
    except (ConfigError, OSError, KeyError, ValueError) as exc:
        print(f"message array: could not be re-derived from the trace: {exc}",
              file=err)
        return EXIT_MISMATCH
    stored_arrays = [
        json.loads(_blob(blob_root, s["root_request_ref"])["messages"].decode("ascii"))
        for s in turns
    ]
    for n, (want_msgs, got_msgs) in enumerate(zip(stored_arrays, derived), start=1):
        if want_msgs != got_msgs:
            print(f"message array: drift at turn {n} — today's prompt assembly no "
                  f"longer reproduces what this episode sent. First difference: "
                  f"{_first_difference(want_msgs, got_msgs)}", file=err)
            return EXIT_MISMATCH
    print(f"message array: OK ({len(derived)} turns re-derived from the trace alone)",
          file=out)

    if args.online:
        rendered = [_rendered(blob_root, s["root_request_ref"]) for s in turns]
        if not asyncio.run(_verify_online(cfg, episode, derived, rendered, out, err)):
            return EXIT_MISMATCH

    _render_transcript(cfg, steps, out)
    return EXIT_OK


def _first_difference(want: list[dict], got: list[dict]) -> str:
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return (f"message {i} ({a.get('role')}): stored {a.get('content', '')[:120]!r} "
                    f"vs re-derived {b.get('content', '')[:120]!r}")
    return f"stored {len(want)} messages, re-derived {len(got)}"


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="Local Recursive Language Model runtime. Three verbs: "
                    "validate, run, replay.")
    sub = parser.add_subparsers(dest="verb", required=True)

    def common(p):
        p.add_argument("--config", default="config.yaml", help="path to config.yaml")
        p.add_argument("--lifecycle-log", default=None,
                       help="JSONL lifecycle log (default: next to the trace store)")
        return p

    v = common(sub.add_parser("validate", help="check config, servers and isolation"))
    v.add_argument("--no-server-probe", action="store_true",
                   help="skip the /props handshake (config and isolation only)")
    v.set_defaults(func=cmd_validate)

    r = common(sub.add_parser("run", help="run one episode"))
    r.add_argument("task_file", help="task JSON file")
    r.set_defaults(func=cmd_run)

    p = common(sub.add_parser("replay", help="verify and render a stored episode"))
    p.add_argument("episode_id")
    p.add_argument("--online", action="store_true",
                   help="additionally re-POST the re-derived messages to "
                        "/apply-template and compare byte for byte")
    p.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("aborted by operator", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
