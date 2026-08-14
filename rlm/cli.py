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

from rlm.config import Config, PromptRegistry, load_config
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
from rlm.serverproc import LlamaServerProcess
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
# The separator after `build` is OPTIONAL because b10375 does not print one:
# measured on-box at S1, a live server emits
#   `common_param: common_params_print_info: build 10375 (ba360efe1) with Clang…`
# while this regex originally required `build:` or `build =`. A required
# separator made every correctly-launched server parse as "no build line",
# which `log_is_current` turns into UNVERIFIED and `validate` turns into a
# refusal -- the gate failing closed on a server that was in fact exactly what
# config said it was. Both shapes are accepted now; the test pins both.
_BUILD_LINE = re.compile(r"build\s*[:=]?\s*(\d+)\s*\(([0-9a-f]+)\)")


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

    Both halves of the build line must match when both were parsed. Accepting
    either alone is too weak to be worth having: build NUMBERS increment, so a
    stale log from the previous build of the same commit (or a rebuild at the
    same number from a different commit) would pass on the half that happens to
    agree. The check exists to catch exactly that kind of near-miss.
    """
    if not props:
        return False
    build_info = str(props.get("build_info") or "")
    commit = parsed.get("build_commit")
    number = parsed.get("build_number")
    if not build_info or not (commit or number):
        return False
    return all(part in build_info for part in (commit, number) if part)


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


def leaf_process_manager(cfg: Config, *, launch: bool):
    """The `ProcessManager` an episode rotates the leaf through (§5 C4).

    Built HERE, in the process root, and not in C4: the launch flags live in
    `config.yaml` (which is also the only place `config_snapshot` can record
    them from), and the dispatcher must keep having no code path that restarts
    a server. `run_episode` takes it injected, so every rotation test runs
    against a fake and no test ever starts a llama-server.

    Returns None for a dry run (no leaf server exists to rotate) and for a real
    run that did not ask to own the leaf -- in which case the manager would
    refuse every rotation anyway, and refusing at construction says so once
    instead of once per exhausted pool. `--launch-leaf` is what makes the
    R13 mitigation usable past window `--parallel`: the scaffold can only
    replace a process it started.
    """
    if cfg.scaffold.dispatcher != "real" or not launch:
        return None
    leaf = cfg.servers.leaf
    url = f"http://127.0.0.1:{leaf.port}"

    async def health() -> bool:
        # A client per poll, deliberately: the poll spans the death of one
        # process and the birth of another, and a pooled keep-alive connection
        # to the old one is exactly the handle that reports a stale answer.
        client = ServerClient(url, timeout=5.0)
        try:
            return await client.health()
        finally:
            await client.aclose()

    return LlamaServerProcess(leaf, health_probe=health)


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


def _check_cache_types(cfg: Config, probed: dict[str, dict], out, err,
                        *, probe_ran: bool) -> bool:
    """D27's half of the §4 handshake: cache types + flash-attn, from the
    launch log, cross-checked against the live build so a stale log cannot
    satisfy the assertion.

    UNVERIFIED IS A FAILURE, not a note. A validation gate that goes green
    without asserting anything is worse than no gate: it converts "we never
    checked the KV cache types" into "the KV cache types are fine" on the one
    surface an operator runs specifically to be told otherwise. The single
    exception is `--no-server-probe`, where the caller has explicitly asked for
    config-and-isolation only and there is no live build to tie a log to;
    `probe_ran` carries that distinction rather than letting an empty `probed`
    dict stand in for it.
    """
    ok = True
    for role in ("root", "leaf"):
        server_cfg = getattr(cfg.servers, role)
        parsed = parse_launch_log(server_cfg.log_path)
        unverified = None
        if not parsed:
            unverified = (f"no parseable `-lv 4` launch log at "
                          f"{server_cfg.log_path}. D27: /props cannot report "
                          f"cache types, so nothing here has been checked")
        elif not log_is_current(parsed, probed.get(role)):
            unverified = (f"the launch log at {server_cfg.log_path} could not be "
                          f"tied to a live server build, and a stale log from a "
                          f"previous launch would satisfy this check while the "
                          f"running server does not")
        if unverified is not None:
            if not probe_ran:
                print(f"{role} KV cache type: UNVERIFIED — {unverified} "
                      f"(not a failure: --no-server-probe was requested)", file=out)
                continue
            print(f"{role} KV cache type: UNVERIFIED — {unverified}. Refusing to "
                  f"start: this gate cannot pass without asserting.", file=err)
            ok = False
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
            ok = False
            continue
        print(f"{role} KV cache type: OK (K/V {parsed['type_k']}, flash_attn "
              f"{parsed['flash_attn']}, from the launch log)", file=out)
    return ok


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
        if not _check_cache_types(cfg, probed, out, err,
                                   probe_ran=not args.no_server_probe):
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


async def _run_one(cfg: Config, task: Task, lifecycle: Lifecycle, task_path: Path,
                    *, launch_leaf: bool = False):
    dispatcher, owns_http = _build_dispatcher(cfg, task_path)
    trace = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root, lifecycle=lifecycle)
    await trace.start()
    leaf_proc = leaf_process_manager(cfg, launch=launch_leaf)
    if leaf_proc is not None:
        # Ownership is what makes a rotation possible at all (§5 C4): the
        # scaffold can only replace a process it started, and `restart()`
        # refuses otherwise rather than starting a second server on a taken
        # port while the first keeps answering.
        lifecycle.event("server_health", role="leaf", state="launching",
                         port=cfg.servers.leaf.port)
        await leaf_proc.start()
    try:
        result = await run_episode(
            task, cfg, dispatcher=dispatcher, trace=trace, lifecycle=lifecycle,
            scaffold_instance_id=f"{os.getpid()}",
            scaffold_git_sha=_scaffold_git_sha(),
            benchmark_version=cfg.benchmark.version,
            process_manager=leaf_proc)
        if cfg.trace.export_every_episode:
            # D21. `export_bundle` reads what is COMMITTED, so the drain is
            # not optional: run_episode drains before returning, and nothing
            # is enqueued between there and here.
            #
            # Scoped to THIS episode, not the whole store: an unscoped export
            # at every close is O(total episodes) per episode, which turns a
            # multi-hour bench into quadratic work for a bundle nobody asked
            # to be re-written. The episode_id is a UUID this process just
            # minted, so interpolating it into `run_filter_sql` (which DuckDB
            # will not take as a bound parameter in a COPY) cannot carry
            # anything a task file influenced.
            await trace.drain()
            bundles = Path(cfg.trace.db_path).parent / "bundle"
            trace.export_bundle(bundles / result.episode_id,
                                 f"episode_id = '{uuid.UUID(result.episode_id)}'",
                                 blob_scope=str(uuid.UUID(result.episode_id)))
        return result
    finally:
        await trace.aclose()
        if owns_http:
            await dispatcher.aclose()
        if leaf_proc is not None:
            # A leaf this process launched dies with the episode: leaving 20 GB
            # of weights resident on a port the next run expects to be free is
            # how two servers end up disagreeing about what is running (R11).
            await leaf_proc.stop()
            lifecycle.event("server_health", role="leaf", state="stopped",
                             port=cfg.servers.leaf.port)


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
            result = asyncio.run(_run_one(cfg, task, lifecycle, task_path,
                                           launch_leaf=args.launch_leaf))
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


class PromptDrift(RlmError):
    """A prompt file changed since the episode ran.

    Kept distinct from assembly drift on purpose: they have different causes
    and different responses, and collapsing them makes the assembly canary
    useless (see `episode_config`).
    """


def episode_config(snapshot: dict) -> tuple[Config, Any]:
    """Rebuild the config THIS EPISODE ACTUALLY RAN UNDER, from its own snapshot.

    Replay must never re-derive against the LIVE config file. Bumping
    `max_subcalls` or editing a prompt would otherwise change the re-derived
    message array and be reported as prompt-ASSEMBLY drift -- a false alarm on
    the one instrument whose entire job is detecting real drift, and the fastest
    way to teach an operator to ignore it. `config_snapshot` is the canonical
    dump of the validated model precisely so this is possible.

    Prompt changes are surfaced SEPARATELY, as `PromptDrift`. The registry here
    is built UNPINNED over the episode's own prompt paths, so a changed file is
    reported rather than thrown: the pinned path (`Config`'s own validator)
    raises a sha256 mismatch that reads like a config error, which buries the
    finding instead of naming it.

    The live config still decides WHERE to read from (`trace.db_path`,
    `trace.blob_root`) and which server to talk to; the snapshot decides what
    everything MEANT.
    """
    fields = {k: v for k, v in snapshot.items() if k in Config.model_fields}
    if "scaffold" not in fields:
        raise ConfigError("config_snapshot carries no scaffold block; this "
                          "episode predates snapshot-based replay")
    prompts = (fields["scaffold"].get("prompts") or {})
    # The envelope block is optional and only present from the S2 A/B onward.
    # It must be rebuilt when the snapshot has it, or `registry.hashes()` comes
    # back missing the `leaf_envelope.*` entries the episode recorded and every
    # replay of an envelope episode reads as prompt DRIFT.
    envelope_ref = prompts.get("leaf_envelope")
    try:
        registry = PromptRegistry.from_files(
            root_path=Path(prompts["root"]["path"]),
            leaf_prefix_path=Path(prompts["leaf_prefix"]["path"]),
            leaf_envelope_path=(Path(envelope_ref["path"]) if envelope_ref else None),
            strategy_paths={cat: Path(ref["path"])
                            for cat, ref in prompts["strategy_templates"].items()},
        ).load()
    except KeyError as exc:
        raise ConfigError(f"config_snapshot has no prompt path for {exc}") from exc

    recorded = snapshot.get("prompt_hashes") or {}
    if recorded and registry.hashes() != recorded:
        now = registry.hashes()
        differing = sorted(k for k in set(recorded) | set(now)
                           if recorded.get(k) != now.get(k))
        raise PromptDrift(
            f"{len(differing)} prompt hash(es) changed since this episode ran: "
            f"{differing}. The message array cannot be re-derived against prompt "
            f"text the episode never saw — this is a prompt change, not "
            f"prompt-assembly drift.")
    return Config.model_validate(fields), registry


def _rederive_messages(cfg: Config, registry, episode: dict, steps: list[dict],
                        blob_root: Path) -> list[list[dict]]:
    """Rebuild, from the trace ALONE, the message array sent at every root turn.

    `cfg`/`registry` are the EPISODE's (from `episode_config`), never the live
    config's. Nothing here reads the lifecycle log, the task file, or a server
    -- that is the S3 gate condition, and it is why `compose_user_message` is a
    pure function of trace-recoverable values and why the task's instruction
    text is carried in `config_snapshot`.
    """
    snapshot = episode.get("config_snapshot") or {}
    task_meta = snapshot.get("task") or {}
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


async def _verify_online(cfg: Config, ep_cfg: Config, episode: dict,
                          arrays: list[list[dict]], rendered: list[str],
                          out, err) -> bool:
    """Mode (ii). `cfg` is the LIVE config (which server to ask); `ep_cfg` is
    the episode's (how its render was parameterised)."""
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
                    "enable_thinking": ep_cfg.scaffold.root.enable_thinking})
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
    # LEAF steps are rehashed here too -- they carry the same pair since the S2
    # leaf-transport fix, and the §4 prefix contract is exactly what a drifted
    # leaf request would break. Only the message-array re-derivation below is
    # root-only (it filters on `repl_exec`).
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
    print(f"root_view_hash: OK ({len(checked)} stored requests rehashed offline)",
          file=out)

    # (i, second half) The prompt-assembly canary, re-derived against the
    # config THIS EPISODE ran under -- never the live file (see episode_config).
    try:
        ep_cfg, ep_registry = episode_config(episode.get("config_snapshot") or {})
    except PromptDrift as exc:
        print(f"prompt drift: {exc}", file=err)
        return EXIT_MISMATCH
    except ConfigError as exc:
        print(f"config_snapshot could not be rebuilt: {exc}", file=err)
        return EXIT_MISMATCH
    print("prompt hashes: OK (every prompt matches what this episode ran)",
          file=out)

    turns = [s for s in steps
             if s["action_type"] == ActionType.REPL_EXEC and s["root_request_ref"]]
    try:
        derived = _rederive_messages(ep_cfg, ep_registry, episode, steps, blob_root)
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
        # Live config says WHERE the server is; the episode's says how its
        # render was parameterised (enable_thinking shaped the stored bytes).
        if not asyncio.run(_verify_online(cfg, ep_cfg, episode, derived, rendered,
                                           out, err)):
            return EXIT_MISMATCH

    _render_transcript(ep_cfg, steps, out)
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
    # §5 C4: R13's mitigation spends one slot per window, so an episode that
    # covers more windows than `--parallel` needs the leaf ROTATED -- and the
    # scaffold can only replace a process it started. Off by default: the
    # servers are normally launched and owned outside `rlm run`, and taking
    # ownership silently would kill an operator's server at episode end.
    r.add_argument("--launch-leaf", action="store_true",
                   help="launch (and own) the leaf server from config, so its "
                        "slot pool can be rotated when it is exhausted")
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
