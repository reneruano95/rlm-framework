"""The operator surface (spec §5). FIVE verbs, and they are the whole thing:

    rlm validate                     config schema + /props probe + D7 + D27
    rlm run <task-file>              one episode; prints episode_id + outcome
    rlm replay <episode-id> [--online]   §6 replay check + transcript render
    rlm bench [--smoke]              §8's benchmark run: grid, verdict, report
    rlm export <run-id|episode-id>   the parquet+blob bundle a foreign reader gets

THREE BECAME FIVE, AND THAT IS THE WHOLE RENEGOTIATION. This docstring said
"`bench` and `export` are later slices (S4)" from the day it was written, and
S4 is that slice: the two verbs it named are the two that landed. **Non-goals
stay non-goals: no daemon, no REST API, no web UI, no interactive chat mode.**
A SIXTH verb still belongs to a slice that argued for it.

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
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import duckdb
import yaml

from rlm.arms import run_b1, run_b2, run_b3
from rlm.bench import (
    ARM_ORDER,
    BENCH_PROFILE,
    LEDGER_PATH,
    RESIDENT_PROFILE,
    BenchCtx,
    BenchLedger,
    Block,
    assert_manifest_pinned,
    build_blocks,
    run_bench,
    seeded_config,
)
# The two hook defaults `assert_bench_wiring` has to be able to RECOGNISE. They
# are private to `rlm/bench.py` because nothing else should install them; this
# module is the composition root that must detect them still installed.
from rlm.bench import _no_hook as _BENCH_NO_HOOK
from rlm.bench import _no_task_loader as _BENCH_NO_TASK_LOADER
from rlm.config import Config, PromptRegistry, load_config
from rlm.dispatcher import LLMDispatcher, MockDispatcher, ServerClient
from rlm.episode import (
    Task,
    assert_props,
    compose_user_message,
    handshake,
    no_cell_observation,
    run_episode,
)
from rlm.errors import (
    ActionType,
    ConfigError,
    Outcome,
    RlmError,
    ServerRotationError,
    StepStatus,
)
from rlm.lifecycle import Lifecycle
from rlm.power import PowerSampler, read_pkg_temp_c
from rlm.rootclient import assistant_prefix, extract_cell
from rlm.serverproc import LlamaServerProcess
from rlm.sandbox import winproc
from rlm.sandbox.manager import SandboxManager, install_bootstrap
from rlm.trace import TraceLogger, recover_orphans, unpack_blob
from rlm.verdict import (
    BASELINES,
    RLM_ARM,
    PairResult,
    Verdict,
    VerdictError,
    cost_scorecard,
    decide,
    leak_report,
    load_grid,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

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

    FIRST OCCURRENCE WINS, and that is load-bearing since the DFlash2 swap
    (2026-08-19, `s2/DFLASH2.md`). A speculative launch with `-md` builds TWO
    contexts, so the log carries two `llama_kv_cache:` lines and two
    `flash_attn =` lines: the target's first, then the drafter's. This loop used
    to `update()` on every match, i.e. LAST wins, so the moment a drafter was
    attached §4's assertion silently stopped describing the target and started
    describing the draft cache -- which defaults to f16 and 5 layers and would
    have failed the q8_0 check for entirely the wrong reason. Measured on the
    shipped root: target `K (q8_0) 544.00 MiB / 32768 cells / 16 layers`, draft
    `K (f16) 25.00 MiB / 2560 cells / 5 layers`.

    The draft context is not discarded, it is recorded under `draft_*` -- a
    silently shadowed value is exactly what this function exists to prevent, and
    the drafter's own cache types are worth having in the snapshot.
    """
    found: dict[str, Any] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return found
    n_kv = n_fa = 0
    for line in text.splitlines():
        m = _KV_LINE.search(line)
        if m:
            fields = dict(kv_mib=float(m.group(1)), kv_cells=int(m.group(2)),
                          kv_layers=int(m.group(3)), kv_seqs=int(m.group(5)),
                          type_k=m.group(6).lower(), type_v=m.group(7).lower())
            # A zero-layer line is llama.cpp reporting a context it did not
            # actually allocate (the drafter prints one before its real cache);
            # it names no cache type and must not consume the target slot.
            if fields["kv_layers"] > 0:
                if n_kv == 0:
                    found.update(fields)
                elif n_kv == 1:
                    found.update({f"draft_{k}": v for k, v in fields.items()})
                n_kv += 1
        m = _FA_LINE.search(line)
        if m:
            if n_fa == 0:
                found["flash_attn"] = m.group(1).lower()
            elif n_fa == 1:
                found["draft_flash_attn"] = m.group(1).lower()
            n_fa += 1
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

    The child's environment comes from `servers.leaf.env` (merged over
    `os.environ` by `LlamaServerProcess`), NOT from this process's environment
    alone. No `env=` is passed here precisely so that the launch environment is
    a config value `config_snapshot` records -- passing `env=None` is what
    silently dropped ROCBLAS_USE_HIPBLASLT, which every leaf number in `s2/`
    was measured with (`s2/run_occupancy.py:455`).
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
                        *, probe_ran: bool,
                        roles: tuple[str, ...] = ("root", "leaf")) -> bool:
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

    `roles` defaults to `cmd_validate`'s pair, but `ServerOrchestra.to_bench_leaf`
    (S4 Task 10) calls this with `roles=("bench_leaf",)` -- the D27 gap
    `rlm validate` never closes on its own, since it only ever probes
    `root`/`leaf`. Every role named here must be an attribute of
    `cfg.servers` (`getattr(cfg.servers, role)`).
    """
    ok = True
    for role in roles:
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
    # Same rule, same bug class, for §8's baseline prompts (S4): a slot the
    # registry loads but this rebuild skips comes back missing from
    # `registry.hashes()`, and EVERY episode recorded since it landed replays as
    # prompt DRIFT. `Config.prompt_registry` is the enumeration this one has to
    # agree with; the `or {}` keeps a pre-S4 snapshot (no baselines block at
    # all) rebuilding exactly as it did before.
    baseline_refs = prompts.get("baselines") or {}
    # Same rule again for the delegation arm's root prompt (2026-08-20). Absent
    # from every snapshot recorded before it, present in every `rlm-restricted`
    # one, and a rebuild that skipped it would replay those as prompt DRIFT.
    restricted_ref = prompts.get("root_restricted")
    try:
        registry = PromptRegistry.from_files(
            root_path=Path(prompts["root"]["path"]),
            root_restricted_path=(Path(restricted_ref["path"])
                                  if restricted_ref else None),
            leaf_prefix_path=Path(prompts["leaf_prefix"]["path"]),
            leaf_envelope_path=(Path(envelope_ref["path"]) if envelope_ref else None),
            strategy_paths={cat: Path(ref["path"])
                            for cat, ref in prompts["strategy_templates"].items()},
            baseline_paths={name: Path(ref["path"])
                            for name, ref in baseline_refs.items()},
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
            # The ACTOR, not a literal "leaf": B2's reduce step is an
            # `llm_call` with `actor='root'` (`rlm/arms.py`), and a transcript
            # that labels it "leaf" says the call happened on a server it never
            # touched — the one fact a reader most needs this line for.
            actor = step.get("actor") or "leaf"
            print(f"[{step['step_idx']}] {actor} llm_call ({step['status']}, "
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


# =========================================================================== #
# ServerOrchestra (S4 Task 10): who owns BOTH server profiles for a bench run
# =========================================================================== #
#
# WHY THIS CLASS LIVES HERE, IN THE PROCESS ROOT, AND NOT IN A NEW MODULE.
# `rlm/bench.py`'s dependency-rule exemption is spec-frozen at exactly two
# modules -- `rlm/episode.py` and `rlm/cli.py` (`tests/test_import_rules.py`'s
# `ISOLATED` list and its comment on `bench.py`). A standalone `rlm/benchserve.py`
# was the first design considered, and it does not survive that lint. Every
# module on disk that is not one of the two exempt composition roots (or the
# small always-exempt set -- `dispatcher.py`, `rootclient.py`, `config.py`,
# `lifecycle.py`, `errors.py`, `__init__.py`) MUST appear in `ISOLATED`
# (`test_lint_covers_every_isolated_module_that_exists`), and every module IN
# `ISOLATED` is forbidden from importing `rlm.dispatcher`/`rlm.rootclient`
# directly (`FORBIDDEN_RLM`). `ServerOrchestra` needs `ServerClient` for the §4
# handshake (`rlm.episode.handshake` takes one as its first argument) -- so a
# `benchserve.py` housing it would have to import `rlm.dispatcher`, which is
# exactly the one import `ISOLATED` membership forbids. There is no import
# shape that gets HTTP into a new isolated module; the class joins
# `leaf_process_manager` here instead of widening the exemption list, exactly
# as `rlm/serverproc.py`'s own docstring anticipates ("the CLI ... supplies an
# implementation").
#
# WHAT IT OWNS. `root_proc` is started once by `start_resident()` and only
# ever RE-PROBED after that -- the root never changes across a bench run.
# `leaf_proc` moves between the RESIDENT profile (`servers.leaf`, RLM/B2) and
# the BENCH profile (`servers.bench_leaf`, B1/B3); the two SHARE port 8081 by
# config (its own comment: "this is a RELAUNCH of the one leaf process, not a
# second server"), so a swap always stops whatever is live before starting the
# other. `_bring_up_leaf` is the one place that happens -- used by
# `start_resident`, `to_bench_leaf` and `to_resident_leaf` alike, so
# `swap_to`'s two callers can never race each other into a bound port.
#
# NO ADOPTION (ledgered ruling, superseding an earlier design). Whatever is
# found live but UNOWNED on the shared leaf port -- on a resume, or in
# principle at any other bring-up -- is always stopped (force-killed via
# `_ensure_leaf_stopped`, after a shape check that refuses to touch anything
# that doesn't look like a real `/props` body) and replaced with a FRESH,
# OWNED process, even when it already matches the profile being requested.
# `leaf_proc` is therefore never `None` while a leaf lives: rotation
# (`LeafProcessManager`/`HandshakingProcessManager`) and teardown
# (`stop_all`) are unconditional, and every reclaim is logged the same way
# regardless of whether it happened during a resume or not -- there is no
# separate "this was actually a resume" event to get wrong. The accepted
# cost is one extra ~10s relaunch on a resume that would otherwise have
# found a perfectly good survivor.
#
# THE HOOK SURFACE. `rlm/bench.py`'s `BenchCtx` takes three server-facing
# callables; Task 12 binds them straight off one instance:
#
#     BenchCtx(quiesce_fn=orchestra.quiesce,
#              handshake_fn=orchestra.handshake_profile,
#              swap_servers_fn=orchestra.swap_to, ...)
#
# and the mid-episode `ProcessManager` rotation is TWO DIFFERENT managers,
# not one (ledgered ruling): `orchestra.episode_process_manager()` for
# `run_episode` ('rlm' arm, which already re-handshakes itself after
# `restart()` -- see `rlm.episode.Episode._rotate_leaf`) and
# `orchestra.b2_process_manager()` for `run_b2` (which cannot, so its
# manager wraps the re-handshake in). Both target the resident leaf --
# `rlm.bench.ARM_PROFILE` runs both arms there exclusively.
#
# EVERYTHING THAT TOUCHES A PROCESS, THE NETWORK, OR THE OS IS INJECTED --
# `rlm/arms.py`'s own discipline, restated here: `process_factory` defaults to
# `LlamaServerProcess`, `client_factory` to `ServerClient`, `handshake_fn` to
# `rlm.episode.handshake`, `cache_check_fn` to `_check_cache_types`,
# `slots_idle_fn` to `_slots_idle`, `force_kill_fn` to a best-effort Windows
# port-reclaim. Unit tests substitute a `FakeProcess`/fake-client pair for the
# first two and let the REAL `handshake`/`assert_props` logic run against
# synthesized `/props` bodies, so the tests prove the WIRING -- which config
# reaches which check -- not merely that a mock was called.


#: The §4 handshake's own read timeout, reused for the swap-time probe.
DEFAULT_ORCHESTRA_HANDSHAKE_TIMEOUT_S = 15.0
#: A liveness probe (resume reconciliation, health polling) is not the
#: handshake itself -- it exists to answer "is anything there at all", so it
#: fails fast rather than waiting out the full handshake budget.
DEFAULT_ORCHESTRA_LIVENESS_TIMEOUT_S = 5.0


class _NullLifecycle:
    """A `Lifecycle`-shaped no-op. `_slots_idle` calls `.event(...)`
    unconditionally, and `ServerOrchestra` is meant to be constructible (and
    testable) with no lifecycle log at all."""

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


_NULL_LIFECYCLE = _NullLifecycle()


def _pid_on_port(port: int) -> int | None:
    """Best-effort: which PID (if any) is LISTENING on `port`, via `netstat`.
    Windows-only, like the rest of this module's process ownership. Returns
    `None` on any failure to parse or run the command -- the caller's own
    `start()` still fails loudly if the port turns out to still be bound."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True, check=False, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f"127.0.0.1:{port} "
    for line in out.splitlines():
        if "LISTENING" in line and needle in line:
            parts = line.split()
            if parts and parts[-1].isdigit():
                return int(parts[-1])
    return None


async def _default_force_kill(port: int) -> None:
    """Reclaim a port this orchestra does not own (§5 C4 resume safety,
    ledgered ruling): a survivor -- of a crash, or simply never spawned by
    this object -- that `LlamaServerProcess.restart()`'s ownership check
    cannot touch, because there is no OBJECT to call `.restart()` on.
    Best-effort and silent on failure (no `netstat`/`taskkill`, permission
    denied, already exited); the caller's own `start()` still fails loudly
    (a bind error surfaces as `ServerRotationError`) if the port stays
    occupied."""
    pid = _pid_on_port(port)
    if pid is None:
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True,
                       check=False, timeout=10)
    await asyncio.sleep(0.5)   # let the OS release the socket before a retry


def _looks_like_llama_props(body: Any) -> bool:
    """A loose shape check: does `body` look like a llama-server `/props`
    response AT ALL, as opposed to some unrelated process that happens to
    answer HTTP on this port? Checked before `_ensure_leaf_stopped` ever
    force-kills a survivor -- `model_path`/`total_slots`/
    `default_generation_settings` are fields a real `/props` body always
    carries, even when their VALUES mismatch what config expects (that
    mismatch is not this function's job; `assert_props` covers it). Not
    matching this shape is grounds to refuse the kill outright, not to
    proceed cautiously -- an unidentified process on a configured port is an
    operator problem, not one `ServerOrchestra` should try to solve by
    guessing."""
    return (isinstance(body, dict)
            and isinstance(body.get("model_path"), str)
            and isinstance(body.get("total_slots"), int)
            and isinstance(body.get("default_generation_settings"), dict))


def bench_leaf_raw(raw_cfg: dict) -> dict:
    """The RAW config dict with `servers.leaf` REPLACED by
    `servers.bench_leaf`'s fields.

    Raw rather than validated because it has a second caller: §8's B1/B3 arms
    need this swap AND `rlm.bench.seeded_config`'s per-seed patch applied to
    the same dict, and composing two raw patches then validating ONCE is the
    only order in which every cross-field rule sees the config the arm will
    actually run under. Validating here first and re-patching afterwards would
    check a topology no episode uses.
    """
    raw = copy.deepcopy(raw_cfg)
    bench = (raw.get("servers") or {}).get("bench_leaf")
    if bench is None:
        raise ConfigError(
            "servers.bench_leaf is not configured; §8's B1/B3 dispatcher "
            "cannot be built without it (ARCHITECTURE.md §8)")
    raw["servers"]["leaf"] = copy.deepcopy(bench)
    return raw


def bench_leaf_config(raw_cfg: dict) -> Config:
    """The `Config` §8's B1/B3 dispatcher is built against: `servers.leaf`
    REPLACED by `servers.bench_leaf`'s fields, so `LLMDispatcher.from_config`
    (and `rlm.arms.bench_slot_capacity`) read the TRUE 2-slot/524288-ctx
    topology instead of the resident 128-slot one.

    Patches the RAW dict and re-validates -- `rlm.bench.seeded_config`'s
    pattern, verbatim: never mutate a built `Config`, so every cross-field
    rule in `rlm.config` (slot capacity vs chunk budget, dispatch_concurrency
    vs parallel) runs again against the swapped-in values rather than being
    assumed to still hold.
    """
    return Config.model_validate(bench_leaf_raw(raw_cfg))


def bench_dispatcher(raw_cfg: dict, *,
                      on_step: Callable[[dict[str, Any]], None] | None = None
                      ) -> LLMDispatcher:
    """§8's B1/B3 leaf dispatcher: the same construction as the RLM/B2
    dispatcher (`LLMDispatcher.from_config`), against the swapped-in
    `bench_leaf` topology (`bench_leaf_config`)."""
    return LLMDispatcher.from_config(bench_leaf_config(raw_cfg), on_step=on_step)


class LeafProcessManager:
    """`ProcessManager` (one method, `.restart()`) over WHATEVER the
    orchestra currently owns as the resident leaf -- looked up FRESH on
    every call, not captured once, so a manager built at startup survives
    any later leaf swap: the RLM/B2 arms always run on the resident
    profile, but `leaf_proc`'s IDENTITY changes across swaps (a new
    `LlamaServerProcess` object each time).

    PLAIN: restart only, no re-handshake. This is `run_episode`'s manager
    (`ServerOrchestra.episode_process_manager`) -- `rlm/episode.py`'s own
    `Episode._rotate_leaf` already calls `_rehandshake_leaf()` immediately
    after `process_manager.restart()` returns, so wrapping the handshake
    HERE too would double the §4 probe on every RLM rotation for no
    benefit. `HandshakingProcessManager` below is the subclass that adds it,
    for the ONE caller that actually needs it (`run_b2`).
    """

    def __init__(self, orchestra: "ServerOrchestra") -> None:
        self._orchestra = orchestra

    async def restart(self) -> None:
        proc = self._orchestra.leaf_proc
        if proc is None:
            raise ServerRotationError(
                "no leaf process is owned by this orchestra; there is "
                "nothing to rotate (was start_resident() called?)")
        await proc.restart()


class HandshakingProcessManager(LeafProcessManager):
    """`LeafProcessManager` plus a re-handshake against `server_cfg` after
    every restart.

    Required because `arms.py` cannot itself speak HTTP (the dependency
    rule -- see `rlm.arms.ArmEpisode._rotate_leaf`'s docstring), and `run_b2`
    has no built-in re-handshake of its own -- unlike `run_episode`, which is
    why THAT caller gets the plain base class instead (ledgered ruling: do
    not double-handshake the 'rlm' arm). Ledgered ruling: restore §5's full
    rotation contract ("stop -> start -> re-handshake -> resume") for `run_b2`,
    the one caller that was missing it. `rlm.arms.ArmEpisode._rotate_leaf`'s
    own docstring names this exact composition as the way to close that gap.
    """

    def __init__(self, orchestra: "ServerOrchestra", role: str, server_cfg: Any) -> None:
        super().__init__(orchestra)
        self._role = role
        self._server_cfg = server_cfg

    async def restart(self) -> None:
        await super().restart()
        await self._orchestra._probe(self._role, self._server_cfg)


class ServerOrchestra:
    """Owns the root process, and swaps the leaf process between the
    RESIDENT (RLM/B2) and BENCH (B1/B3) profiles for one bench run (§8's
    within-block order: RLM -> B2 -> [swap] -> B1 -> B3 -> [swap, lazily, at
    the next block's RLM arm]).
    """

    def __init__(self, cfg: Config, *, launch: bool = True,
                 lifecycle: Any = None, out: Any = None, err: Any = None,
                 process_factory: Callable[..., Any] = LlamaServerProcess,
                 client_factory: Callable[..., Any] = ServerClient,
                 handshake_fn: Callable[..., Awaitable[dict]] = handshake,
                 cache_check_fn: Callable[..., bool] = _check_cache_types,
                 slots_idle_fn: Callable[..., Awaitable[bool]] = _slots_idle,
                 force_kill_fn: Callable[[int], Awaitable[None]] | None = None,
                 handshake_timeout_s: float = DEFAULT_ORCHESTRA_HANDSHAKE_TIMEOUT_S
                 ) -> None:
        self.cfg = cfg
        #: False is `--no-launch-servers`: assert-only. Handshakes still run
        #: (against whatever an operator already has up); swaps refuse.
        self.launch = launch
        self.lifecycle = lifecycle
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self._process_factory = process_factory
        self._client_factory = client_factory
        self._handshake_fn = handshake_fn
        self._cache_check_fn = cache_check_fn
        self._slots_idle_fn = slots_idle_fn
        self._force_kill_fn = force_kill_fn if force_kill_fn is not None else _default_force_kill
        self._handshake_timeout_s = handshake_timeout_s
        self.root_proc: Any = None
        self.leaf_proc: Any = None
        #: Which leaf profile `leaf_proc` currently holds. `None` until
        #: `start_resident` runs.
        self.current_profile: str | None = None
        #: Wall time the MOST RECENT swap spent stopped-to-healthy. §8
        #: excludes this from per-task wall-clock BY CONSTRUCTION: every
        #: caller awaits it from `rlm.bench._prepare`, entirely before that
        #: function's caller (`_run_cell`) reads its own clock for `t0`.
        self.last_relaunch_s: float = 0.0

    # -- low-level probes --------------------------------------------------- #

    def _health_probe(self, port: int) -> Callable[[], Awaitable[bool]]:
        async def health() -> bool:
            client = self._client_factory(
                f"http://127.0.0.1:{port}", timeout=DEFAULT_ORCHESTRA_LIVENESS_TIMEOUT_S)
            try:
                return await client.health()
            finally:
                await client.aclose()
        return health

    async def _probe(self, role: str, server_cfg: Any) -> dict:
        """The §4 handshake against ONE server, `role`'s OWN `server_cfg` --
        never a different role's, which is exactly the total_slots 2!=128
        mismatch a bench_leaf process handshaked against `servers.leaf` would
        raise."""
        client = self._client_factory(f"http://127.0.0.1:{server_cfg.port}",
                                       timeout=self._handshake_timeout_s)
        try:
            return await self._handshake_fn(client, server_cfg, role, self.lifecycle)
        finally:
            await client.aclose()

    async def _probe_liveness(self, port: int) -> dict | None:
        """Raw `/props`, or `None` for "nothing answered" -- resume
        reconciliation's read, deliberately weaker than `_probe`: an
        unreachable port is not a handshake failure here, it is the ordinary
        "nothing to reconcile against" case."""
        client = self._client_factory(
            f"http://127.0.0.1:{port}", timeout=DEFAULT_ORCHESTRA_LIVENESS_TIMEOUT_S)
        try:
            return await client.props()
        except Exception:  # noqa: BLE001 -- unreachable IS "nothing live"
            return None
        finally:
            await client.aclose()

    def _leaf_cfg(self, profile: str) -> Any:
        if profile == RESIDENT_PROFILE:
            return self.cfg.servers.leaf
        if profile == BENCH_PROFILE:
            bench = self.cfg.servers.bench_leaf
            if bench is None:
                raise ConfigError(
                    "servers.bench_leaf is not configured; §8's B1/B3 "
                    "relaunch profile cannot run without it (ARCHITECTURE.md §8)")
            return bench
        raise ConfigError(f"unknown server profile {profile!r}; expected "
                          f"{RESIDENT_PROFILE!r} or {BENCH_PROFILE!r}")

    def _verify_bench_cache_types(self, props: dict) -> None:
        """D27 gap (ledgered ruling): `rlm validate` never checks
        `bench_leaf`'s cache types -- it only ever probes `root`/`leaf`
        (`_probe_servers`). Reuses the exact verification `cmd_validate` runs
        for the other two servers (`parse_launch_log` + `_check_cache_types`),
        scoped to just this role, so a bench run that relaunched into a
        mis-flagged `bench_leaf` refuses rather than measuring B1/B3 against
        an unverified KV cache configuration."""
        ok = self._cache_check_fn(self.cfg, {"bench_leaf": props}, self.out, self.err,
                                   probe_ran=True, roles=("bench_leaf",))
        if not ok:
            raise ConfigError(
                f"bench_leaf KV cache type verification FAILED after relaunch "
                f"(see {self.cfg.servers.bench_leaf.log_path}); refusing to "
                "run B1/B3 against an unverified cache configuration (D27)")

    # -- resume reconciliation ------------------------------------------- #

    async def _ensure_leaf_stopped(self) -> None:
        """Whatever is on the shared leaf port is GONE by the time this
        returns -- OWNED (via `.stop()`) or an unowned survivor
        (force-killed). No adoption (ledgered ruling, superseding an
        earlier design that left a matching survivor unowned): a leaf this
        orchestra did not spawn is never trusted to be rotatable later
        (`LeafProcessManager`/`HandshakingProcessManager` need an OWNED
        `leaf_proc` to call `.restart()` on -- an adopted one leaves
        mid-episode rotation permanently broken for that leaf's entire
        tenure), so even a survivor that already matches the TARGET profile
        is replaced. The ~10s extra relaunch on a resume is accepted:
        correctness (rotation always works, every event means what it says)
        over one saved relaunch.

        Called for EVERY leaf bring-up, not only on resume: in ordinary
        operation `leaf_proc` is already owned once `start_resident()` has
        run once, so the unowned-survivor branch below is reachable only
        before the very first bring-up in a process's lifetime -- an
        ordinary swap between two OWNED processes never probes or logs
        anything here at all.
        """
        if self.leaf_proc is not None:
            await self.leaf_proc.stop()
            self.leaf_proc = None
            return
        port = self.cfg.servers.leaf.port
        live = await self._probe_liveness(port)
        if live is None:
            return          # nothing there; nothing to reclaim
        if not _looks_like_llama_props(live):
            raise ConfigError(
                f"port {port} is answering but its /props response does not "
                f"look like a llama-server one ({live!r}); refusing to "
                "force-kill a process that might not be a leaf server at "
                "all -- stop whatever is bound to this port manually and "
                "retry")
        if self.lifecycle is not None:
            self.lifecycle.event("server_health", role="leaf",
                                 state="reclaiming_unowned_port", port=port)
        await self._force_kill_fn(port)

    # -- leaf lifecycle -------------------------------------------------- #

    async def _bring_up_leaf(self, profile: str) -> float:
        already_there = (self.current_profile == profile
                         and (self.leaf_proc is not None or not self.launch))
        if already_there:
            return 0.0
        if not self.launch:
            raise ConfigError(
                "ServerOrchestra is running --no-launch-servers (assert-only): "
                f"swapping the leaf to the {profile!r} profile requires the "
                "scaffold to OWN the server process, which an operator-managed "
                "run does not grant. The full B1/B2/B3/RLM grid is not "
                "supported against operator-managed servers (§5); drop "
                "--no-launch-servers to run it.")
        target_cfg = self._leaf_cfg(profile)
        t0 = time.monotonic()
        await self._ensure_leaf_stopped()
        self.leaf_proc = self._process_factory(
            target_cfg, health_probe=self._health_probe(target_cfg.port))
        await self.leaf_proc.start()
        self.current_profile = profile
        self.last_relaunch_s = round(time.monotonic() - t0, 3)
        role = "leaf" if profile == RESIDENT_PROFILE else "bench_leaf"
        props = await self._probe(role, target_cfg)
        if profile == BENCH_PROFILE:
            self._verify_bench_cache_types(props)
        return self.last_relaunch_s

    async def start_resident(self) -> None:
        """Bring up the RESIDENT topology (root + the RLM/B2 leaf) and probe
        both. In `--no-launch-servers` mode this is handshake-only -- no
        process is spawned, matching an operator-managed server."""
        if not self.launch:
            await self._probe("root", self.cfg.servers.root)
            await self._probe("leaf", self.cfg.servers.leaf)
            self.current_profile = RESIDENT_PROFILE
            return
        if self.root_proc is None:
            self.root_proc = self._process_factory(
                self.cfg.servers.root,
                health_probe=self._health_probe(self.cfg.servers.root.port))
            await self.root_proc.start()
        await self._probe("root", self.cfg.servers.root)
        await self._bring_up_leaf(RESIDENT_PROFILE)

    async def to_bench_leaf(self) -> float:
        """Stop the RLM leaf, start `bench_leaf`, handshake vs `bench_leaf`'s
        OWN config, verify its cache types (D27 gap). Returns the relaunch
        wall time (0.0 if nothing needed to move)."""
        return await self._bring_up_leaf(BENCH_PROFILE)

    async def to_resident_leaf(self) -> float:
        """The reverse of `to_bench_leaf`."""
        return await self._bring_up_leaf(RESIDENT_PROFILE)

    async def stop_all(self) -> None:
        if self.leaf_proc is not None:
            await self.leaf_proc.stop()
            self.leaf_proc = None
        if self.root_proc is not None:
            await self.root_proc.stop()
            self.root_proc = None
        self.current_profile = None

    # -- the BenchCtx hook surface (rlm.bench.BenchCtx) ------------------- #

    async def quiesce(self, profile: str) -> dict[str, bool]:
        """`BenchCtx.quiesce_fn` -- the §5 C5 quiesce point, reusing
        `_slots_idle` for both servers this profile serves. Root's slot is
        awaited too: a root call mid-flight when the leaf swaps is still a
        call the swap must not step on."""
        role = "leaf" if profile == RESIDENT_PROFILE else "bench_leaf"
        lc = self.lifecycle if self.lifecycle is not None else _NULL_LIFECYCLE
        root_idle = await self._slots_idle_fn(self.cfg, "root", lc)
        leaf_idle = await self._slots_idle_fn(self.cfg, role, lc)
        return {"root": root_idle, role: leaf_idle}

    async def handshake_profile(self, profile: str) -> dict[str, dict]:
        """`BenchCtx.handshake_fn` -- §4's per-episode `/props` re-assertion,
        against root and whichever leaf `profile` names."""
        role = "leaf" if profile == RESIDENT_PROFILE else "bench_leaf"
        server_cfg = self._leaf_cfg(profile)
        root_props = await self._probe("root", self.cfg.servers.root)
        leaf_props = await self._probe(role, server_cfg)
        return {"root": root_props, role: leaf_props}

    async def swap_to(self, profile: str) -> float:
        """`BenchCtx.swap_servers_fn` -- the leaf relaunch, dispatched by
        profile name."""
        if profile == RESIDENT_PROFILE:
            return await self.to_resident_leaf()
        if profile == BENCH_PROFILE:
            return await self.to_bench_leaf()
        raise ConfigError(f"unknown server profile {profile!r}; expected "
                          f"{RESIDENT_PROFILE!r} or {BENCH_PROFILE!r}")

    # -- the ProcessManager run_b2/run_episode rotate mid-episode --------- #
    #
    # BOTH arms run exclusively on the resident topology (`rlm.bench.ARM_PROFILE`),
    # so both managers target the OWNED resident leaf -- but they are NOT
    # interchangeable (ledgered ruling): `run_episode` already re-handshakes
    # itself after `process_manager.restart()` (`Episode._rotate_leaf` ->
    # `_rehandshake_leaf`), so it gets the PLAIN manager; `run_b2` has no such
    # built-in re-handshake (`arms.py` cannot itself speak HTTP), so it gets
    # the one that adds it.

    def episode_process_manager(self) -> LeafProcessManager:
        """For `run_episode` ('rlm' arm). PLAIN restart -- `run_episode`'s
        own rotation already re-runs the §4 handshake right after this
        returns; wrapping it here too would double the probe on every RLM
        rotation for no benefit."""
        return LeafProcessManager(self)

    def b2_process_manager(self) -> HandshakingProcessManager:
        """For `run_b2`. Wraps the OWNED resident leaf so `.restart()` also
        re-runs the §4 handshake before returning -- the ONLY caller that
        needs it, since `run_b2` cannot re-handshake itself."""
        return HandshakingProcessManager(self, "leaf", self.cfg.servers.leaf)


# =========================================================================== #
# verb: bench (S4 Task 12) -- §8's benchmark run, end to end
# =========================================================================== #
#
# WHAT THIS SECTION IS. Three modules do the work and none of them may do it
# alone: `rlm/bench.py` schedules the grid but may not reach a model server,
# `rlm/arms.py` runs the baselines but may not construct one, `rlm/verdict.py`
# scores the record but may not open a live store. This is where they meet.
# Every seam those modules declare as INJECTED is bound here, once.
#
# AND THE BINDING IS ASSERTED, NOT ASSUMED (`assert_bench_wiring`). `BenchCtx`'s
# hook defaults are deliberate no-ops -- that module is exercised with no
# servers at all -- so a forgotten `quiesce_fn` would not crash a 39-hour run,
# it would complete one with no quiesce point, no §4 re-assertion and no leaf
# relaunch: every B1/B3 cell measured against the RLM topology, and nothing in
# the record to say so. The startup assertion is the only place that can catch
# it, because by construction nothing downstream can.
#
# PHASES ARE SEPARABLE, and each ends with the store CLOSED:
#
#   1. the grid          -> TraceLogger open, `run_bench`, TraceLogger closed
#   2. the verdict       -> `load_grid` + `decide` on the closed file
#   3. escalation (§8:343, only when a margin lands in {+1,+2,+3})
#                        -> a SECOND TraceLogger, then closed again
#   4. the recomputation -> `decide` once more; `render_report(escalated=)`
#
# The close between 1 and 2 is not tidiness: on Windows DuckDB excludes every
# other connection from a file its writer holds open, so a verdict simply
# cannot be computed while the writer lives. A new `TraceLogger` per phase (not
# `start()` twice) because `aclose()` shuts its writer thread pool down for good.

#: §8's freeze, as an artifact path rather than a CLI flag: a scoring run an
#: operator can point at a different manifest is a scoring run they can point at
#: a friendlier one. Moving it buys no bypass either way -- whatever it names is
#: checked against `benchmark.manifest_sha256` (`assert_manifest_pinned`).
BENCH_MANIFEST_PATH = REPO_ROOT / "bench" / "manifest.json"

#: Where the generated half of the S4 report lands. `write_report` preserves
#: everything below `verdict.NARRATIVE_MARKER`, so re-running a bench run
#: regenerates the tables and never destroys the findings.
DEFAULT_REPORT_PATH = REPO_ROOT / "s4" / "RESULTS.md"

# ---- the projection constants a --smoke run calibrates against -------------- #
#
# Copied from `s2/aggregation_options.py:19-38` rather than imported: `s2/` is
# an analysis directory, not part of the shipped wheel (`pyproject.toml`
# packages = ["rlm"]), and `rlm/` importing it would break `rlm` on install for
# the sake of four floats. They are PROJECTIONS from measured inputs, never
# measurements -- which is exactly why the smoke run prints them beside the
# numbers it just measured instead of trusting them.
PROJ_S_PER_WINDOW = 2.78          # §8, serial, both sub-calls per window
PROJ_CHEAP_ARM_S = 60.0           # a single-shot arm: one big call + scoring
PROJ_NON_AGG_EXPENSIVE_S = 450.0  # needle/synthesis/codeQA on a chunked arm
PROJ_ROOT_OVERHEAD_FRAC = 0.30    # share of an agg episode NOT in leaf windows
PROJ_S4_BUDGET_H = 60             # §8's pre-registered wall budget for S4
#: §8: "two chunked-and-exposed (RLM, B2) versus two single-shot-and-spared".
CHUNKED_ARMS = ("rlm", "b2")


def _raw_config(path: Path) -> dict:
    """`load_config`'s input, kept.

    `rlm.bench.seeded_config` patches the RAW dict and re-validates for every
    seed (never mutating a built `Config`), so a bench run needs both halves
    and `load_config` returns only one.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config {path} as YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    return raw


def load_benchmark_manifest():
    """The frozen manifest -- imported LAZILY, and that is load-bearing.

    `bench/` is the benchmark artifact, not part of the shipped wheel, and
    nothing under `rlm/` may import it at module scope: `import rlm.cli` would
    then fail outright on an installed wheel, for the four verbs that have
    nothing to do with §8. So the import happens inside the one verb that needs
    it, and its absence is a refusal with a sentence rather than a traceback.
    """
    try:
        from bench.manifest import BenchmarkManifest
    except ImportError as exc:                  # pragma: no cover - wheel only
        raise ConfigError(
            f"the benchmark package `bench/` is not importable ({exc}). It is "
            f"the benchmark ARTIFACT and is deliberately not shipped in the "
            f"wheel; `rlm bench` runs from a checkout of the repository") from exc
    path = Path(BENCH_MANIFEST_PATH)
    try:
        return BenchmarkManifest.load(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ConfigError(
            f"cannot read the benchmark manifest at {path}: {exc}") from exc


def smoke_task_ids(manifest) -> list[str]:
    """The default `--smoke` task set: the first NON-ADVERSARIAL task of each
    category, in manifest order.

    Non-adversarial is the whole point. §8 flags a small number of tasks as
    adversarial-context, and those are the tasks most likely to behave unlike
    their category -- while a smoke run exists to produce ONE seconds-per-
    episode number per (arm, category) to project 39 hours from. Calibrating
    on the outlier is how a projection ends up wrong in the direction nobody
    checks. Derived, never hardcoded: the manifest is the freeze.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for entry in manifest.tasks:
        if entry.adversarial or entry.category in seen:
            continue
        seen.add(entry.category)
        ids.append(entry.task_id)
    return ids


def blocks_for(manifest, task_ids, seeds: list[int], *, offset: int = 0) -> list[Block]:
    """§8's blocked schedule, restricted to `task_ids`.

    `build_blocks` is the one place the task-major/seed-minor order lives, so
    this filters ITS output rather than re-deriving the order: a subset run and
    a full run then visit the same cells in the same relative order, and each
    block keeps the index it has in the canonical grid.

    `offset` keeps a later phase's block numbers disjoint from the main grid's
    -- the escalation phase runs cells the base grid never had, and two rows
    sharing a block number would make the ledger ambiguous about which pass
    wrote them.
    """
    wanted = set(task_ids)
    return [Block(task_entry=b.task_entry, seed=b.seed, idx=b.idx + offset)
            for b in build_blocks(manifest, seeds)
            if b.task_entry.task_id in wanted]


def bench_arm_runners(raw_cfg: dict, *, trace, lifecycle, orchestra, registry,
                      rlm_dispatcher, leaf_dispatcher, root_client,
                      scaffold_instance_id: str, scaffold_git_sha: str,
                      benchmark_version: str | None) -> dict[str, Any]:
    """§8's arms, each closed over everything `rlm/bench.py` may not
    import. This IS `BenchCtx.arm_runners`.

    THE FOUR ARE NOT SYMMETRIC, and every asymmetry below is a ruling:

      * `rlm` is `run_episode`, whose bench identity travels in
        `snapshot_extra={"bench": ...}` -- it has no arm concept of its own, so
        `bench_extra` carries `arm="rlm"` for it and for nobody else
        (`ArmEpisode.snapshot` writes its own).
      * `b2` gets the RESIDENT dispatcher (it is a chunked arm on the RLM
        topology) plus a root client for its reduce step, plus the
        HANDSHAKING process manager -- it cannot re-handshake itself after a
        rotation the way `run_episode` can.
      * `b1`/`b3` get the BENCH-profile dispatcher AND a bench-profile
        `Config`. Handing them the resident config would be a silent
        measurement error rather than a crash: `bench_slot_capacity` and B1's
        head+tail overflow policy read `servers.leaf`, so the arm would size
        its prompt against a 128-slot/4096-per-slot topology while running on
        the 2-slot/262144-per-slot one. Their slot pins (0 and 1) are §8's
        v0.2.6 correction and are `run_b1`/`run_b3`'s own defaults.

    The bench-profile config is derived per SEED, from the same
    `seeded_config` the scheduler uses, so B1/B3 vary with §8's seeds exactly
    as the other two arms do -- and it is cached because that derivation
    re-validates the whole config, which is not free 360 times.
    """
    bench_cfgs: dict[int, Config] = {}

    def _bench_cfg(seed: int) -> Config:
        """`seeded_config` composed with `bench_leaf_raw`, validated once.

        Built LAZILY: a run that asks only for `--arm rlm,b2` must not be
        refused for want of a `servers.bench_leaf` block it never uses, and a
        run that does ask for B1/B3 without one gets `run_b1`'s own per-cell
        refusal (a `config_refused` ledger row), which is what
        `rlm.bench._run_cell` is written to contain.
        """
        cfg = bench_cfgs.get(seed)
        if cfg is None:
            cfg = bench_cfgs[seed] = seeded_config(bench_leaf_raw(raw_cfg), seed)
        return cfg

    # Each runner drops its dispatcher's step log on the way out (see
    # `reset_dispatcher_steps`): the episode is over, nothing reads those rows
    # again, and they hold every rendered prompt the episode sent.
    async def rlm_arm(task, cfg, *, bench_extra):
        try:
            return await run_episode(
                task, cfg, dispatcher=rlm_dispatcher, trace=trace,
                lifecycle=lifecycle, snapshot_extra={"bench": bench_extra},
                process_manager=orchestra.episode_process_manager(),
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha,
                benchmark_version=benchmark_version)
        finally:
            reset_dispatcher_steps(rlm_dispatcher)

    async def _virgin_resident_pool(manager) -> None:
        """Start a DELEGATING episode on a leaf whose every slot is virgin.

        THE DEFECT THIS CLOSES (measured, `rlm bench --smoke` 2026-08-20).
        R13's pool is per leaf PROCESS, not per episode, and `rotate_pool` was
        only ever called on a profile SWAP. That was sufficient while the only
        heavy delegator was B2, which is separated from the next block's B2 by
        the B1/B3 swap -- and while the `rlm` arm made ZERO leaf calls, which
        S4 measured it doing in all 90 episodes.

        `rlm-restricted` breaks both assumptions: it delegates for every chunk
        read, spends all 128 slots inside one episode, and sits on the RESIDENT
        profile next to B2 with no swap between them. The smoke showed the
        consequence exactly -- rlm-restricted/agg-02 drained the pool, then
        needle-02 opened on the drained generation and died with 0 answered
        windows, and B2/agg-02 starved behind it with `slot_pool_exhausted`.
        A SCORED BASELINE was being failed by the arm that ran before it.

        A restart, not merely a `rotate_pool()`: adopting a virgin pool for a
        process that never restarted is R13 reintroduced by its own mitigation
        (`pool_rotating_swap` says the same thing from the other side).

        Only the delegating arms pay the ~10 s: `rlm` does not call the leaf.
        """
        await manager.restart()
        rlm_dispatcher.rotate_pool()

    async def b2_arm(task, cfg, *, bench_extra):
        manager = orchestra.b2_process_manager()
        await _virgin_resident_pool(manager)
        try:
            return await run_b2(
                task, cfg, dispatcher=rlm_dispatcher, root_client=root_client,
                trace=trace, registry=registry, bench_extra=bench_extra,
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha,
                process_manager=manager)
        finally:
            reset_dispatcher_steps(rlm_dispatcher)

    async def b1_arm(task, _cfg, *, bench_extra):
        try:
            return await run_b1(
                task, _bench_cfg(bench_extra["seed"]), dispatcher=leaf_dispatcher,
                trace=trace, registry=registry, bench_extra=bench_extra,
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha)
        finally:
            reset_dispatcher_steps(leaf_dispatcher)

    async def b3_arm(task, _cfg, *, bench_extra):
        try:
            return await run_b3(
                task, _bench_cfg(bench_extra["seed"]), dispatcher=leaf_dispatcher,
                trace=trace, registry=registry, bench_extra=bench_extra,
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha)
        finally:
            reset_dispatcher_steps(leaf_dispatcher)

    async def rlm_restricted_arm(task, cfg, *, bench_extra):
        """`rlm`, with `llm_query` as the only route to chunk content.

        Identical to `rlm_arm` in every other respect -- same dispatcher, same
        topology, same profile -- and differs in the one flag whose effect the
        arm exists to measure. It DOES pay a leaf restart first (see
        `_virgin_resident_pool`), which `rlm_arm` does not need because that arm
        never calls the leaf.
        See `docs/superpowers/plans/2026-08-20-delegation-arm.md`.
        """
        manager = orchestra.episode_process_manager()
        await _virgin_resident_pool(manager)
        try:
            return await run_episode(
                task, cfg, dispatcher=rlm_dispatcher, trace=trace,
                lifecycle=lifecycle, snapshot_extra={"bench": bench_extra},
                process_manager=manager,
                scaffold_instance_id=scaffold_instance_id,
                scaffold_git_sha=scaffold_git_sha,
                benchmark_version=benchmark_version,
                restrict_chunks=True)
        finally:
            reset_dispatcher_steps(rlm_dispatcher)

    return {"rlm": rlm_arm, "rlm-restricted": rlm_restricted_arm,
            "b2": b2_arm, "b1": b1_arm, "b3": b3_arm}


def pool_rotating_swap(orchestra, *, resident_dispatcher, bench_dispatcher):
    """`BenchCtx.swap_servers_fn`, plus the half `ServerOrchestra` cannot do:
    telling C4 that its never-reuse slot pool is virgin again.

    THE DEFECT THIS CLOSES. R13's mitigation gives every window a slot that is
    never handed out twice for the lifetime of one leaf PROCESS, and
    `SlotPool` says so ("a restarted server means a new pool"). The orchestra
    restarts the process at every profile change, but the pool lives in the
    DISPATCHER, which a bench run builds once and keeps for all 90 blocks. The
    bench-profile leaf has TWO slots and B1/B3 spend one each per block (both
    call with `chunk=None`, so `window_key` is per-call and no window ever
    repeats), so from block 2 onward every B1 and B3 cell raised
    `SlotPoolExhausted` against a server that had just been restarted with two
    virgin slots. §8's two single-shot arms would have failed 58 of 60 blocks
    structurally -- a manufactured result, in the exact class §8's
    contamination paragraph names.

    Rotating AFTER the swap returns is what makes it safe: `rotate_pool`
    refuses with calls in flight, and `rlm.bench._prepare` runs the swap
    strictly between cells, with nothing dispatched.

    Both directions rotate, and the resident one matters as much: the RLM/B2
    leaf is restarted just as often, and an un-rotated resident pool would run
    RLM's later blocks against slots the scaffold believes are virgin and the
    server has already used -- R13 itself, reintroduced by its own mitigation.

    ONLY WHEN A PROCESS ACTUALLY RESTARTED, which is the same rule read from
    the other side: `swap_to` no-ops when the requested profile is already
    live, and adopting a virgin pool for a server that never restarted is
    precisely the hazard `SlotPool` names -- the scaffold would believe every
    slot untouched while the process still holds every document it has served.

    The signal is the leaf PROCESS OBJECT's identity, not the reported
    relaunch seconds. `_bring_up_leaf` builds a new `LlamaServerProcess` for
    every bring-up, so a changed object is a restarted server by construction,
    whereas the timing value is `round(elapsed, 3)` and a fast (or faked)
    relaunch can legitimately report 0.0. The seconds are kept as the fallback
    for an orchestra that exposes no process at all.
    """
    async def swap(profile: str):
        before = getattr(orchestra, "leaf_proc", None)
        relaunch_s = await orchestra.swap_to(profile)
        after = getattr(orchestra, "leaf_proc", None)
        restarted = (after is not None and after is not before) or bool(relaunch_s)
        if not restarted:
            return relaunch_s
        dispatcher = (bench_dispatcher if profile == BENCH_PROFILE
                      else resident_dispatcher)
        if dispatcher is not None and hasattr(dispatcher, "rotate_pool"):
            dispatcher.rotate_pool()
        return relaunch_s

    return swap


def reset_dispatcher_steps(*dispatchers) -> None:
    """Drop the per-call step records C4 accumulated for the episode that just
    ended.

    `LLMDispatcher.steps` is an append-only list of every attempt, each row
    holding the FULL rendered prompt, and the dispatcher is a bench run's
    long-lived object while an episode is not. B1's single call renders a
    ~256K-token document, so one B1 episode retains roughly half a megabyte
    that nothing will read again -- an arm reads `steps` only for the call_ids
    of the episode it is closing (`ArmEpisode.log_call`,
    `_EpisodeRun._log_attempts`), never across episodes. Over 360 episodes on a
    64 GiB-carved box that is not a leak to shrug at.

    Cleared by the COMPOSITION ROOT, which owns the dispatcher's lifetime,
    rather than by an arm: an arm clearing a shared object's state would be
    reaching outside the episode it was handed.
    """
    for dispatcher in dispatchers:
        if dispatcher is None:
            continue
        steps = getattr(dispatcher, "steps", None)
        if isinstance(steps, list):
            steps.clear()
        # `_retry_base` continues a call's retry_idx across a mid-call
        # rotation; the counter is meaningless once the call is over.
        recorded = getattr(dispatcher, "_recorded", None)
        if isinstance(recorded, dict):
            recorded.clear()


def assert_bench_wiring(ctx: BenchCtx, arms) -> None:
    """Refuse a bench run whose `BenchCtx` still carries a default.

    `rlm/bench.py` chose no-op hook defaults on purpose (that module is
    dry-run with no servers), and named THIS file as where a missing one is a
    startup bug. This is that check. It is not defensive programming: an
    unbound `swap_servers_fn` produces a complete, plausible, fully-recorded
    39-hour grid in which B1 and B3 ran on the RLM leaf -- a result, not an
    error, and one nothing downstream can detect.

    `temp_fn` and `sampler` are deliberately NOT required. Both are optional by
    design (`power_sampling.enabled` is a config switch, and `read_pkg_temp_c`
    returns None on hosts without the ACPI class), and both are recorded as
    honest NULLs when absent rather than silently changing what ran.
    """
    missing: list[str] = []
    for arm in arms:
        if not ctx.arm_runners.get(arm):
            missing.append(f"arm_runners[{arm!r}]")
    for name, default in (("quiesce_fn", _BENCH_NO_HOOK),
                          ("handshake_fn", _BENCH_NO_HOOK),
                          ("swap_servers_fn", _BENCH_NO_HOOK),
                          ("load_task_fn", _BENCH_NO_TASK_LOADER)):
        value = getattr(ctx, name, None)
        if value is None or value is default:
            missing.append(name)
    if ctx.trace is None:
        missing.append("trace")
    if missing:
        raise ConfigError(
            f"refusing to start a benchmark run with unbound wiring: "
            f"{', '.join(missing)}. `BenchCtx`'s defaults are no-ops, so a run "
            f"started like this would not fail -- it would produce a complete "
            f"grid measured without the §4 handshake, the §5 quiesce point or "
            f"the B1/B3 leaf relaunch, and no column would record that")


def bench_exit_code(verdict, escalated=None) -> int:
    """0 on gate PASS, 1 on FAIL -- from the POST-escalation verdict when there
    is one.

    §8 makes the recomputation the decision and the pre-escalation figures a
    reporting obligation, "not because either may be chosen". An exit code
    taken from the pre-escalation verdict would choose.
    """
    final = escalated if escalated is not None else verdict
    return EXIT_OK if final.gate_pass else EXIT_FAILED


# --------------------------------------------------------------------------- #
# escalation execution (§8:343)
# --------------------------------------------------------------------------- #


def escalation_plan_path(ledger_path, run_id: str) -> Path:
    """Beside the ledger, and named by run: a plan is per-run state, exactly
    like the ledger it sits next to."""
    return Path(ledger_path).parent / f"escalation-{run_id}.json"


def save_escalation_plan(path, verdict, *, seeds) -> Path:
    """Write the pre-escalation verdict's REPORTING FACTS and the plan, BEFORE
    a single escalation episode runs.

    WHY THIS FILE EXISTS. §8 permits exactly one recomputation, and the report
    must state both the pre- and post-escalation figures. Both of those become
    unrecoverable the moment the first escalation episode lands: the store then
    holds a 5-seed grid, and no query over it can reproduce what the 3-seed
    grid decided (`load_grid` builds a cell from every seed present). A run that
    crashed mid-escalation could therefore be resumed to completion and STILL
    not be reportable -- the pre-escalation column would be gone, and the only
    honest thing left to print would be "this grid was escalated against a
    baseline we can no longer name".

    So the figures are written down while they are still true. `pairs` carries
    the whole `PairResult` per baseline (margin, p, CI, the discordant list),
    which is everything `render_report` reads off the pre-escalation verdict,
    and the discordant lists double as the work list a resume finishes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": verdict.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "escalation_seeds": list(seeds),
        "n_tasks": verdict.n_tasks,
        "n_manifest_tasks": verdict.n_manifest_tasks,
        "task_ids": list(verdict.task_ids),
        "arms": list(verdict.arms),
        "passes": {arm: list(tasks) for arm, tasks in verdict.passes.items()},
        "success_rate": dict(verdict.success_rate),
        "gate_pass": verdict.gate_pass,
        "clean_pass": verdict.clean_pass,
        "chunk_sizes": list(verdict.chunk_sizes),
        "escalation_plan": {b: list(t) for b, t in verdict.escalation_plan.items()},
        "pairs": {
            b: {"baseline": p.baseline, "present": p.present,
                "rlm_passes": p.rlm_passes, "baseline_passes": p.baseline_passes,
                "margin": p.margin, "wins": p.wins, "losses": p.losses,
                "discordant": list(p.discordant), "p": p.p,
                "ci": list(p.ci) if p.ci is not None else None,
                "mean_delta": p.mean_delta, "escalates": p.escalates,
                "beats": p.beats}
            for b, p in verdict.pairs.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8",
                    newline="\n")
    return path


def load_escalation_plan(path):
    """The pre-escalation verdict, rebuilt from `save_escalation_plan`'s file,
    or `None` when no escalation was ever planned for this run.

    Rebuilt as a real `Verdict` so `render_report`/`_final` take it unchanged:
    they read `run_id`, `pairs` and `escalation_plan` off the pre-escalation
    verdict and everything else off the post-escalation one. `scores`,
    `categories` and `findings` are left empty on purpose -- the report never
    reads them from this side, and inventing them would be inventing figures.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"the escalation plan at {path} could not be read ({exc}). It "
            f"records the pre-escalation figures §8 requires reported beside "
            f"the post-escalation ones, and they cannot be recomputed from a "
            f"store that already carries seeds {{4, 5}}; move the file aside "
            f"only if you accept losing them") from exc
    pairs = {
        b: PairResult(
            baseline=d["baseline"], present=d["present"],
            rlm_passes=d["rlm_passes"], baseline_passes=d["baseline_passes"],
            margin=d["margin"], wins=d["wins"], losses=d["losses"],
            discordant=tuple(d["discordant"]), p=d["p"],
            ci=tuple(d["ci"]) if d["ci"] is not None else None,
            mean_delta=d["mean_delta"], escalates=d["escalates"],
            beats=d["beats"])
        for b, d in (raw.get("pairs") or {}).items()}
    verdict = Verdict(
        run_id=raw["run_id"], n_tasks=raw["n_tasks"],
        n_manifest_tasks=raw["n_manifest_tasks"],
        task_ids=tuple(raw["task_ids"]), arms=tuple(raw["arms"]),
        passes={a: tuple(t) for a, t in (raw.get("passes") or {}).items()},
        success_rate=dict(raw.get("success_rate") or {}), scores={},
        pairs=pairs, categories=(), findings=(),
        escalation_plan={b: tuple(t)
                         for b, t in (raw.get("escalation_plan") or {}).items()},
        gate_pass=raw["gate_pass"], clean_pass=raw["clean_pass"],
        chunk_sizes=tuple(raw.get("chunk_sizes") or ()), escalated=False)
    return verdict, [int(s) for s in raw.get("escalation_seeds") or ()]


async def run_escalation(ctx: BenchCtx, verdict, *, seeds: list[int],
                          offset: int = 0, out=None) -> list[dict]:
    """Run §8's escalation draws: seeds {4, 5} on each banded pair's discordant
    tasks, for BOTH arms of that pair.

    BOTH ARMS is the part worth stating. The escalation re-decides a task at
    >=3/5, and a task is decided for a COMPARISON: re-drawing RLM alone would
    change one side of the margin and leave the other at thirds, which is not a
    de-noised comparison, it is a different one.

    ONE `run_bench` CALL PER PAIR, not one over the union. Pairs share
    discordant tasks (three baselines flagging the same RLM win is the common
    case), and `run_bench` reads its resume state once per call -- so the
    second pair's call sees the first pair's rows already decided and skips
    RLM's cells instead of drawing them twice. Two draws of one cell would be
    a duplicate row `load_grid` refuses outright, after the episodes were
    spent.

    Nothing here verifies completeness, deliberately: `verdict.load_grid`
    refuses a half-escalated cell (some of {4,5} but not all) on its own, so a
    partial run is caught at scoring by the module that owns the rule rather
    than by a second implementation of it here.
    """
    records: list[dict] = []
    for baseline in BASELINES:
        tasks = verdict.escalation_plan.get(baseline) or ()
        if not tasks:
            continue
        blocks = blocks_for(ctx.manifest, tasks, list(seeds), offset=offset)
        offset += len(blocks)
        if out is not None:
            print(f"escalation: RLM vs {baseline.upper()} — seeds "
                  f"{list(seeds)} on {len(tasks)} discordant task(s) "
                  f"({', '.join(tasks)}), both arms", file=out)
        records += await run_bench(ctx, arms=(RLM_ARM, baseline), seeds=list(seeds),
                                    blocks=blocks)
    return records


# --------------------------------------------------------------------------- #
# --smoke: the calibration table
# --------------------------------------------------------------------------- #


def projected_episode_s(entry, arm: str, *, wall_cap: float) -> float:
    """What `s2/aggregation_options.py` predicts one (task, arm) episode costs.

    Aggregation on a chunked arm is the only size-dependent case: its windows
    are stated in the manifest (§8 requires it, so "the affordability claim is
    checkable rather than assumed"), and the root's share is added back the
    way `s2.episode_seconds` does before the per-episode wall cap applies.
    """
    if arm not in CHUNKED_ARMS:
        return PROJ_CHEAP_ARM_S
    if entry.category == "aggregation" and entry.windows:
        leaf_s = entry.windows * PROJ_S_PER_WINDOW
        return min(leaf_s / (1.0 - PROJ_ROOT_OVERHEAD_FRAC), float(wall_cap))
    return PROJ_NON_AGG_EXPENSIVE_S


def projected_grid_hours(manifest, *, seeds, arms, wall_cap: float,
                          measured: dict | None = None) -> float:
    """The full frozen grid in hours, per (arm, CATEGORY) seconds.

    `measured` (keyed `(arm, category)`) overrides the projection wherever the
    smoke run actually timed something; everything it did not reach falls back
    to the pre-registered constant. Per category rather than per task because
    that is the granularity a 4-episode smoke run can support: one measured
    needle episode says something about the other seven needle tasks and
    nothing at all about aggregation.
    """
    total = 0.0
    for entry in manifest.tasks:
        for arm in arms:
            per = (measured or {}).get((arm, entry.category))
            if per is None:
                per = projected_episode_s(entry, arm, wall_cap=wall_cap)
            total += per * len(seeds)
    return total / 3600.0


def _median(values) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def print_calibration(records, manifest, cfg, *, arms, run_id, out) -> None:
    """§8's affordability claim, measured against its own projection.

    Prints one row per measured cell and then the projected full-grid hours
    twice: once from the pre-registered constants (what the plan was costed
    with) and once with the measurements substituted in. Both, because a
    single number would hide which of the two moved.
    """
    by_id = {t.task_id: t for t in manifest.tasks}
    wall_cap = float(cfg.scaffold.budgets.max_wall_clock_s)
    print(f"\n--- smoke calibration (run_id {run_id}; NOT scored, no report "
          f"written) ---", file=out)
    print(f"projection constants (s2/aggregation_options.py): "
          f"{PROJ_NON_AGG_EXPENSIVE_S:.0f} s chunked-arm non-aggregation, "
          f"{PROJ_CHEAP_ARM_S:.0f} s single-shot, {PROJ_S_PER_WINDOW} s/window "
          f"aggregation (+{PROJ_ROOT_OVERHEAD_FRAC:.0%} root overhead)",
          file=out)
    print(f"{'arm':<5} {'task':<12} {'category':<12} {'measured s':>11} "
          f"{'projected s':>12} {'ratio':>7}  outcome", file=out)
    for record in records:
        entry = by_id.get(record["task_id"])
        if entry is None:
            continue
        want = projected_episode_s(entry, record["arm"], wall_cap=wall_cap)
        got = record.get("wall_s")
        # A cell that refused never opened an episode, so it has no measured
        # seconds. Printing 0.0 for it would read as "instant" -- a
        # measurement -- on the one table whose job is to be believed.
        measured_s = "n/a" if got is None else f"{got:.1f}"
        ratio = f"{got / want:.2f}x" if got and want else "n/a"
        print(f"{record['arm']:<5} {record['task_id']:<12} {entry.category:<12} "
              f"{measured_s:>11} {want:>12.1f} "
              f"{ratio:>7}  {record['outcome']}"
              + (f" ({record['reason']})" if record.get("reason") else ""),
              file=out)

    measured: dict[tuple[str, str], float] = {}
    for arm in arms:
        for category in {t.category for t in manifest.tasks}:
            walls = [r["wall_s"] for r in records
                     if r["arm"] == arm and by_id.get(r["task_id"]) is not None
                     and by_id[r["task_id"]].category == category]
            median = _median(walls)
            if median is not None:
                measured[(arm, category)] = median

    full_seeds = list(cfg.benchmark.seeds)
    from_constants = projected_grid_hours(manifest, seeds=full_seeds,
                                           arms=ARM_ORDER, wall_cap=wall_cap)
    from_measured = projected_grid_hours(manifest, seeds=full_seeds,
                                          arms=ARM_ORDER, wall_cap=wall_cap,
                                          measured=measured)
    agg_ep = max((projected_episode_s(t, "rlm", wall_cap=wall_cap)
                  for t in manifest.tasks if t.category == "aggregation"),
                 default=0.0)
    escalation_h = 32 * agg_ep * 0.5 / 3600.0     # §8: "typically 8-32 extra episodes"
    total = from_measured + escalation_h
    print(f"\nfull grid = {len(manifest.tasks)} tasks x {len(full_seeds)} seeds "
          f"x {len(ARM_ORDER)} arms, plus §8's escalation allowance "
          f"({escalation_h:.1f} h, up to 32 extra episodes)", file=out)
    print(f"  from the pre-registered constants:  {from_constants:>6.1f} h grid "
          f"+ {escalation_h:.1f} h = {from_constants + escalation_h:>6.1f} h",
          file=out)
    print(f"  with these measurements substituted:{from_measured:>6.1f} h grid "
          f"+ {escalation_h:.1f} h = {total:>6.1f} h", file=out)
    print(f"  the measured figure is the one to judge: {total:.1f} h against "
          f"§8's {PROJ_S4_BUDGET_H} h budget — "
          f"{'WITHIN' if total <= PROJ_S4_BUDGET_H else 'OVER'}", file=out)
    if total > PROJ_S4_BUDGET_H:
        print(f"  ** the projection breaches the pre-registered "
              f"{PROJ_S4_BUDGET_H} h budget: that is a decision for a human, "
              f"not a number to proceed past **", file=out)


# --------------------------------------------------------------------------- #
# the verdict block printed to stdout
# --------------------------------------------------------------------------- #


def print_verdict_block(verdict, escalated, *, run_id: str, report_path, out) -> None:
    """The operator's four lines. The REPORT is the artifact; this is the part
    that has to be true at a glance from a terminal, so it states the gate, the
    margins with their inference (§8 forbids a bare margin anywhere), and every
    named finding."""
    final = escalated if escalated is not None else verdict
    # `render_report` says this in the report; the terminal must not be the one
    # surface where a gate decided on a grid that still owes seeds {4,5} reads
    # as final. `cmd_bench` always runs a plan it produced, so this only fires
    # for a caller that did not -- which is exactly when it matters.
    provisional = ("" if escalated is not None or not final.escalation_plan
                   else " · PROVISIONAL: escalation owed")
    print(f"\n## S4 GATE: {'PASS' if final.gate_pass else 'FAIL'}", file=out)
    print(f"run_id {run_id} · {final.n_tasks}/{final.n_manifest_tasks} tasks "
          f"scored · "
          f"{'post-escalation' if final.escalated else 'pre-escalation'} grid · "
          f"{'clean pass' if final.clean_pass else ('NOT a clean pass' if final.gate_pass else 'gate failed')}"
          f"{provisional}", file=out)
    for baseline in BASELINES:
        pair = final.pairs.get(baseline)
        if pair is None:
            continue
        margin = "n/a" if pair.margin is None else f"{pair.margin:+d}"
        p = "p=n/a" if pair.p is None else f"p={pair.p:.4f}"
        ci = ("CI=n/a" if pair.ci is None
              else f"CI=[{pair.ci[0]:+.3f}, {pair.ci[1]:+.3f}]")
        print(f"  margin {margin} vs {baseline.upper()} — {p}, {ci}"
              + ("" if pair.present else "  (arm absent from this grid)"), file=out)
    for finding in final.findings:
        print(f"  [{finding.kind}] {finding.text}", file=out)
    print(f"report: {report_path}", file=out)


# --------------------------------------------------------------------------- #
# cmd_bench
# --------------------------------------------------------------------------- #


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


async def _open_trace(cfg: Config, lifecycle) -> TraceLogger:
    trace = TraceLogger(cfg.trace.db_path, cfg.trace.blob_root, lifecycle=lifecycle)
    await trace.start()
    return trace


async def _bench(args, cfg: Config, raw_cfg: dict, manifest, lifecycle, *,
                  out, err) -> int:
    arms = tuple(_csv(args.arm) or ARM_ORDER)
    unknown_arms = [a for a in arms if a not in ARM_ORDER]
    if unknown_arms:
        raise ConfigError(f"unknown arm(s) {unknown_arms}; §8's arms are "
                          f"{list(ARM_ORDER)}")
    if args.smoke and args.resume:
        # The two flags mean opposite things about the run_id, and the smoke
        # pass is the one that must lose: `--resume` writes into an EXISTING
        # run's identity, and a smoke pass writes one seed of unscored,
        # calibration-only episodes. Together they would inject those episodes
        # into a scored run's grid, where `load_grid` sees them as ordinary
        # cells. "Never counts toward scoring" is enforced by the throwaway
        # run_id and by nothing else, so the combination is refused rather
        # than silently resolved either way.
        raise ConfigError(
            "--smoke and --resume are mutually exclusive: a smoke pass runs "
            "under a THROWAWAY run_id precisely so it can never be scored, "
            "and --resume would write its calibration episodes into an "
            "existing run's grid, where nothing downstream can tell them apart")
    seeds = [int(s) for s in (_csv(args.seeds) or cfg.benchmark.seeds)]
    task_ids = _csv(args.tasks)
    if args.smoke:
        # A throwaway identity, one seed, every arm: the calibration question
        # is "what does one episode of each (arm, category) cost", and a second
        # seed answers it no better while costing another full pass.
        seeds = seeds[:1]
        task_ids = task_ids or smoke_task_ids(manifest)
    task_ids = task_ids or [t.task_id for t in manifest.tasks]
    unknown_tasks = sorted(set(task_ids) - {t.task_id for t in manifest.tasks})
    if unknown_tasks:
        raise ConfigError(
            f"task(s) {unknown_tasks} are not in benchmark manifest "
            f"{manifest.benchmark_version!r}: they are outside the freeze, so "
            f"§8 pre-registers nothing about them")

    run_id = args.resume or str(uuid.uuid4())
    blocks = blocks_for(manifest, task_ids, seeds)
    ledger = BenchLedger(args.ledger)
    report_path = Path(args.report)

    # Everything with a lifetime longer than one episode, and every one of them
    # torn down in the single `finally` below -- including on the refusal paths
    # BELOW this line (an unbuildable `bench_leaf` dispatcher, an unloadable
    # prompt registry), which is why the names are bound before the `try`.
    sampler = orchestra = rlm_dispatcher = leaf_dispatcher = root_client = None
    registry = None
    git_sha = _scaffold_git_sha()
    instance_id = str(os.getpid())

    def _ctx(trace: TraceLogger) -> BenchCtx:
        """A `BenchCtx` bound to THIS phase's TraceLogger.

        Rebuilt per phase rather than mutated because the arm runners close
        over the trace: after `aclose()` a `TraceLogger` is finished (its
        writer pool is shut down), so phase 2 has a different object and every
        runner has to be pointed at it.

        AND THE PROFILE IS READ OFF THE ORCHESTRA, NOT DEFAULTED. `BenchCtx`
        defaults `current_profile` to `resident` -- correct for a fresh run,
        wrong for every later phase: §8's within-block order ends on b3, so the
        main grid leaves the leaf on the BENCH profile. A rebuilt context that
        assumed `resident` would see no profile change at the escalation
        phase's first (rlm) cell, skip the swap, and then handshake the
        RESIDENT config against the bench-profile server that is actually
        running -- a `total_slots` mismatch, a `ConfigError` out of §4, and the
        entire escalation phase aborted (phase 1.5's healing with it). The
        orchestra owns which process is live, so it is asked, every time.
        """
        return BenchCtx(
            raw_cfg=raw_cfg, cfg=cfg, run_id=run_id, manifest=manifest,
            ledger=ledger, trace=trace, lifecycle=lifecycle,
            store=trace.monitor(), sampler=sampler,
            arm_runners=bench_arm_runners(
                raw_cfg, trace=trace, lifecycle=lifecycle, orchestra=orchestra,
                registry=registry, rlm_dispatcher=rlm_dispatcher,
                leaf_dispatcher=leaf_dispatcher, root_client=root_client,
                scaffold_instance_id=instance_id, scaffold_git_sha=git_sha,
                benchmark_version=cfg.benchmark.version),
            load_task_fn=Task.from_file,
            quiesce_fn=orchestra.quiesce,
            handshake_fn=orchestra.handshake_profile,
            # NOT `orchestra.swap_to` bare: a relaunched process has virgin
            # slots and C4 has to be told, or B1/B3 exhaust a 2-slot pool in
            # block 1 and error for the rest of the run.
            swap_servers_fn=pool_rotating_swap(
                orchestra, resident_dispatcher=rlm_dispatcher,
                bench_dispatcher=leaf_dispatcher),
            temp_fn=read_pkg_temp_c,
            current_profile=orchestra.current_profile or RESIDENT_PROFILE,
            repo_root=REPO_ROOT)

    print(f"{'smoke run_id' if args.smoke else 'run_id'}: {run_id}"
          + ("  (resumed)" if args.resume else ""), file=out)
    print(f"grid: {len(task_ids)} task(s) x {len(seeds)} seed(s) x "
          f"{len(arms)} arm(s) = {len(blocks) * len(arms)} cell(s)", file=out)

    try:
        if cfg.power_sampling.enabled:
            # A child process, stopped in the same `finally` as everything
            # else: one left running past a 39-hour grid is a PowerShell
            # polling an energy counter forever. Its stderr goes to a file
            # beside the lifecycle log, because dying silently at launch is
            # its documented failure mode and `alive()` reports only THAT it
            # died -- on a 39-hour run whose energy column then goes NULL,
            # "why" is worth one file.
            sampler = PowerSampler(
                stderr_path=Path(cfg.trace.db_path).parent / "power-sampler.err")
            sampler.start()
            print(f"power sampler: started (stderr -> {sampler.stderr_path})",
                  file=out)
        orchestra = ServerOrchestra(cfg, launch=not args.no_launch_servers,
                                     lifecycle=lifecycle, out=out, err=err)
        rlm_dispatcher = LLMDispatcher.from_config(cfg)
        # Built only when an arm that needs it was asked for: a `--arm rlm,b2`
        # run must not be refused for want of a `servers.bench_leaf` block it
        # will never reach.
        leaf_dispatcher = (bench_dispatcher(raw_cfg)
                           if {"b1", "b3"} & set(arms) else None)
        root_client = ServerClient(
            f"http://127.0.0.1:{cfg.servers.root.port}",
            timeout=cfg.scaffold.retries.per_call_timeout_s)
        registry = cfg.prompt_registry().load()

        await orchestra.start_resident()

        # -- phase 1: the grid ------------------------------------------- #
        trace = await _open_trace(cfg, lifecycle)
        try:
            ctx = _ctx(trace)
            assert_bench_wiring(ctx, arms)
            records = await run_bench(ctx, arms=arms, seeds=seeds, blocks=blocks)
            await trace.drain()
        finally:
            await trace.aclose()

        if args.smoke:
            print_calibration(records, manifest, cfg, arms=arms,
                               run_id=run_id, out=out)
            # A smoke run has no gate, so EXIT_FAILED keeps its literal
            # meaning here -- the command did not complete cleanly. A
            # calibration pass whose cells errored has measured nothing, and
            # exiting 0 on it is how a broken topology gets a green light on
            # the way into a 39-hour run.
            errored = [r for r in records if str(r["outcome"]) == str(Outcome.ERROR)]
            if errored:
                print(f"\n{len(errored)} of {len(records)} smoke cell(s) errored "
                      f"— this calibration measured nothing for them: "
                      + ", ".join(f"{r['arm']}/{r['task_id']}"
                                  f"({r['reason'] or 'error'})" for r in errored[:8]),
                      file=err)
                return EXIT_FAILED
            return EXIT_OK

        esc_seeds = list(cfg.benchmark.escalation_seeds)
        plan_path = escalation_plan_path(args.ledger, run_id)

        # -- phase 1.5: heal a crashed escalation, BEFORE any load_grid --- #
        #
        # A run that died mid-escalation leaves half-escalated cells (some of
        # {4,5}, not all), which `load_grid` refuses outright and correctly --
        # §8 re-decides at >=3/5 and a 4-long cell would be scored at a
        # denominator it never pre-registered. So a resume finishes the plan
        # FIRST and only then asks for a grid. `run_bench` skips cells already
        # decided, so this runs exactly what is still owed.
        saved = load_escalation_plan(plan_path)
        if saved is not None:
            pre_verdict, saved_seeds = saved
            print(f"resuming an escalation planned at {plan_path}: finishing "
                  f"seeds {saved_seeds} on "
                  f"{sum(len(t) for t in pre_verdict.escalation_plan.values())} "
                  f"pair-task(s) before scoring", file=out)
            trace = await _open_trace(cfg, lifecycle)
            try:
                await run_escalation(_ctx(trace), pre_verdict,
                                      seeds=saved_seeds or esc_seeds,
                                      offset=len(blocks), out=out)
                await trace.drain()
            finally:
                await trace.aclose()

        # -- phase 2: the verdict, on the CLOSED store ------------------- #
        verdict = decide(load_grid(cfg.trace.db_path, run_id, seeds=seeds,
                                    escalation_seeds=esc_seeds), manifest)

        # -- phase 3: escalation, only where §8 owes it ------------------ #
        escalated = None
        if saved is not None:
            # Already escalated (this run, or the crashed one this resumed):
            # the grid just scored IS the post-escalation one, and the
            # pre-escalation figures come from the plan written before the
            # first escalation episode -- the only place they still exist.
            verdict, escalated = saved[0], verdict
        elif verdict.escalation_plan:
            save_escalation_plan(plan_path, verdict, seeds=esc_seeds)
            trace = await _open_trace(cfg, lifecycle)
            try:
                await run_escalation(_ctx(trace), verdict, seeds=esc_seeds,
                                      offset=len(blocks), out=out)
                await trace.drain()
            finally:
                await trace.aclose()
            # -- phase 4: ONE recomputation (§8: "no other recomputation is
            # permitted"). `render_report` refuses a pair that is not this
            # run's, or one computed on a grid carrying no escalation seeds.
            escalated = decide(load_grid(cfg.trace.db_path, run_id, seeds=seeds,
                                          escalation_seeds=esc_seeds), manifest)

        scorecard = cost_scorecard(cfg.trace.db_path, run_id)
        leaks = leak_report(cfg.trace.db_path, run_id)
        write_report(report_path, verdict, scorecard, leaks, escalated=escalated)
        print_verdict_block(verdict, escalated, run_id=run_id,
                             report_path=report_path, out=out)
        return bench_exit_code(verdict, escalated)
    finally:
        # THE SAMPLER GOES FIRST, and every other teardown step is individually
        # suppressed. It is the only resource here that is a detached OS
        # process: an httpx client that is never closed dies with this process,
        # a PowerShell child polling an energy counter does not. Ordering it
        # after an unprotected `aclose()` meant one teardown exception -- a
        # server that would not stop, a client whose transport was already
        # gone -- leaked it for the machine's uptime. Suppressed for the same
        # reason `stop_all` always was: teardown must not replace the run's own
        # outcome (or its exception) with a cleanup error.
        if sampler is not None:
            with contextlib.suppress(Exception):
                sampler.stop()
        if orchestra is not None:
            with contextlib.suppress(Exception):
                await orchestra.stop_all()
        for dispatcher in (rlm_dispatcher, leaf_dispatcher, root_client):
            if dispatcher is not None:
                with contextlib.suppress(Exception):
                    await dispatcher.aclose()


def cmd_bench(args) -> int:
    out, err = sys.stdout, sys.stderr
    try:
        config_path = Path(args.config)
        cfg = load_config(config_path)
        raw_cfg = _raw_config(config_path)
        if cfg.scaffold.dispatcher != "real":
            raise ConfigError(
                f"scaffold.dispatcher is {cfg.scaffold.dispatcher!r}; §8 scores "
                f"MODEL behaviour, and a mock-dispatcher grid would be fixture "
                f"replays wearing the benchmark's name. `load_grid` refuses "
                f"dry-run episodes at scoring time, so this refusal only moves "
                f"the same answer 39 hours earlier")
        manifest = load_benchmark_manifest()
        # Checked here for a good error surface, and AGAIN inside `run_bench`
        # where it cannot be bypassed by wiring.
        assert_manifest_pinned(manifest, cfg)
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
            return asyncio.run(_bench(args, cfg, raw_cfg, manifest, lifecycle,
                                       out=out, err=err))
        # THE EXIT-CODE TAXONOMY, and why nothing here may return 1:
        #
        #   0 = §8's gate PASSED     1 = §8's gate FAILED     2 = no verdict
        #
        # 1 is a RESULT -- a grid that ran, was scored, and lost. Every branch
        # below is a run that never produced a verdict at all: a server that
        # would not come up (`ServerRotationError`), a leaf that drifted mid-grid
        # (`ConfigError` out of the §4 handshake), an unscoreable grid
        # (`VerdictError`), an operator's Ctrl-C. Reporting any of those as 1
        # would put "the scaffold lost to its baselines" into CI, and into the
        # S4 record, for a run that was never scored -- the one confusion §8's
        # pre-registration cannot survive. `RlmError` is the root of the
        # scaffold's own exception tree (`ConfigError` included), so this is
        # every attributable scaffold failure; a genuine BUG (TypeError,
        # KeyError) still propagates uncaught and gets a traceback, because a
        # crash the scaffold cannot name must not be quietly filed as "refused".
        except (RlmError, VerdictError) as exc:
            print(f"refused: {exc}. This run produced no verdict; nothing was "
                  f"scored. Fix the cause and resume the grid with "
                  f"`rlm bench --resume <run_id>` (the run_id is printed above "
                  f"and every decided cell is skipped).", file=err)
            return EXIT_REFUSED
        except KeyboardInterrupt:
            print("aborted by operator — this run produced no verdict; resume "
                  "the grid with `rlm bench --resume <run_id>` (the run_id is "
                  "printed above and every decided cell is skipped).", file=err)
            return EXIT_REFUSED
    finally:
        lifecycle.close()


# =========================================================================== #
# verb: export
# =========================================================================== #


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


def cmd_export(args) -> int:
    out, err = sys.stdout, sys.stderr
    try:
        cfg = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"refused: {exc}", file=err)
        return EXIT_REFUSED
    try:
        ident = str(uuid.UUID(str(args.id)))
    except (ValueError, AttributeError, TypeError):
        print(f"refused: {args.id!r} is not a UUID. `rlm export` takes a bench "
              f"run_id or an episode_id, both of which are UUIDs, and the id is "
              f"INTERPOLATED into the export's SQL (DuckDB takes no bound "
              f"parameter in a COPY's FROM clause) — so anything that is not a "
              f"UUID is refused rather than quoted.", file=err)
        return EXIT_REFUSED
    db_path = Path(cfg.trace.db_path)
    if not db_path.exists():
        print(f"refused: no trace store at {db_path}; there is nothing to "
              f"export", file=err)
        return EXIT_REFUSED
    dest = Path(args.dest) if args.dest else db_path.parent / "export" / ident
    return asyncio.run(_export(cfg, ident, dest, out, err))


# --------------------------------------------------------------------------- #
# argv
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="Local Recursive Language Model runtime. Five verbs: "
                    "validate, run, replay, bench, export.")
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

    b = common(sub.add_parser("bench", help="run §8's benchmark grid and score it"))
    b.add_argument("--arm", default=None,
                   help="comma-separated subset of rlm,rlm-restricted,b1,b2,b3 (default: all "
                        "four, always in §8's pre-registered within-block order)")
    b.add_argument("--seeds", default=None,
                   help="comma-separated base seeds (default: benchmark.seeds)")
    b.add_argument("--tasks", default=None,
                   help="comma-separated task_ids from the frozen manifest "
                        "(default: the whole benchmark; with --smoke, the first "
                        "non-adversarial task of each category)")
    b.add_argument("--resume", default=None, metavar="RUN_ID",
                   help="continue an interrupted run: cells already decided are "
                        "skipped, cells still owed §8's one rerun are re-run")
    b.add_argument("--smoke", action="store_true",
                   help="calibration pass: one seed, every arm, a THROWAWAY "
                        "run_id. Prints measured vs projected seconds and the "
                        "projected full-grid hours; never writes the report and "
                        "never counts toward scoring")
    b.add_argument("--report", default=str(DEFAULT_REPORT_PATH),
                   help="where the S4 report is written (the hand-written half "
                        "below the narrative marker is preserved)")
    b.add_argument("--ledger", default=str(LEDGER_PATH),
                   help="the crash-resilient JSONL mirror a --resume reads "
                        "(default: §8's pre-registered path)")
    b.add_argument("--no-launch-servers", action="store_true",
                   help="assert against operator-managed servers instead of "
                        "owning them. The full four-arm grid needs the B1/B3 "
                        "leaf relaunch, which requires ownership, so this only "
                        "supports a resident-profile subset (§5)")
    b.set_defaults(func=cmd_bench)

    e = sub.add_parser("export", help="export a run or episode as a bundle")
    e.add_argument("--config", default="config.yaml", help="path to config.yaml")
    e.add_argument("id", help="a bench run_id, or an episode_id (both UUIDs)")
    e.add_argument("--dest", default=None,
                   help="bundle directory (default: <trace dir>/export/<id>)")
    e.set_defaults(func=cmd_export)
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
