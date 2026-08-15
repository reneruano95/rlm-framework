"""S2 — OCCUPANCY: where the per-call wall-clock goes as the slot pool fills.

THE BLOCKER THIS EXISTS TO CLEAR (`s2/DISTANCE.md` §4b, §5). Identical work —
same layout, same cells, same server — took a median 1.15 s at slots 28-35 and
5.02 s at slots 100-103, while the server's OWN reported prefill and decode
stayed flat (572 -> 561 ms, 537 -> 553 ms). A 4.4x effect with no explanation
confounds every per-arm wall-clock S2 has measured and is the binding
constraint on the window-geometry decision (417 windows at 640/480 against 261
at the shipped 1024/768).

THE INSTRUMENT, AND WHY IT IS THE RIGHT ONE. §6 already defines
`steps.latency_queue_ms` as `(t_first_byte - t_dispatch) - timings.prompt_ms`
precisely because llama-server reports no per-request queue wait. This runner
decomposes every call into buckets that sum to the wall:

    wall = send + queue + prefill + decode + tail

  * `send_ms`   = t_headers - t_dispatch. The client's write and the server's
                  HTTP accept. Grows only if the CLIENT is the bottleneck.
  * `queue_ms`  = (t_first_byte - t_headers) - prompt_ms - one token of decode.
                  Server-side time between accepting the request and the first
                  generated token that is NOT prompt processing: task queue,
                  slot selection, server-side tokenisation, cache save/restore,
                  memory find_slot.
  * `prefill_ms`= timings.prompt_ms, the server's own number.
  * `decode_ms` = timings.predicted_ms, the server's own number.
  * `tail_ms`   = (t_end - t_first_byte) - decode_ms * (n-1)/n. Streaming and
                  teardown after the last token.

Two extra probes ride in front of every call and separate "the server's HTTP
thread is slow" from "the server's inference loop is slow":

  * `GET /health` is answered by the HTTP thread WITHOUT touching the task
    queue. If it grows, the whole process is stalled.
  * `GET /slots` is answered by the inference loop through the task queue. If
    /slots grows while /health does not, the inference loop is busy between
    requests — which is a server-side cost no `timings` field can show.

The server process's private bytes are sampled through `psapi` on every call
(hypothesis 3: allocator pressure as 62.8 MiB/slot of recurrent state fills).

SEPARABILITY IS THE DESIGN. One factor moves per condition and the workload is
byte-identical across all of them: the same 128 synthetic chunks, the same
question, the same pinned leaf prefix, the same sampling. The factors:

  * `--order {asc,desc,shuffle}` separates SLOT INDEX from CALL ORDINAL. Under
    `desc` the two are anti-correlated and under `shuffle` they are
    uncorrelated, so one run decides whether the cost tracks which slot a call
    lands on or how many calls the process has already served.
  * `--np` with `--calls` held at 8 separates POOL SIZE from SLOTS-IN-USE
    (hypothesis 1: an LCP scan linear in slot count).
  * `--extra` carries the one server flag under test verbatim: `-sps 0` /
    `-sps 1.0` (hypothesis 1), `--cache-ram 0` and `--no-cache-idle-slots`
    (the host prompt cache, which llama-server b10375 enables by default at
    8192 MiB and which saves idle slots to host RAM ON EVERY NEW TASK).
  * `--concurrency` with `--http-limits` varies `httpx.Limits` at fixed
    occupancy (hypothesis 2: calls queueing in the CLIENT, never reaching the
    server).

R13 (§10, §5 C4). Every call gets a slot no other call in the run will ever
get, the server's returned `id_slot` is ASSERTED against the requested one, and
every answer goes through R13's foreign-identifier detector against the whole
synthetic corpus. Hits are reported per condition. The `--order`/`--np`
conditions never reuse a slot; nothing here is exempt.

WHAT THIS RUNNER DOES NOT DO. No retries (a failed call is a recorded fact), no
scoring (this measures time, not quality), and no prompt edits: the system
prefix is `Config.prompt_registry()`'s sha256-pinned text and the user segment
is composed by `rlm.dispatcher.compose_leaf_user`, so the calls timed here are
the calls production sends.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `uv run s2/run_occupancy.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.config import load_config  # noqa: E402
from rlm.dispatcher import (  # noqa: E402
    chat_control_markers,
    compose_leaf_user,
    neutralise_control_tokens,
)
from rlm.leakcheck import ChunkIndex  # noqa: E402

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "occupancy.jsonl"
REPORT_MD = S2_DIR / "OCCUPANCY.md"

#: The launch line §4/the task pins, minus the flags a condition varies. Kept
#: here verbatim rather than rebuilt from `launch_argv()` because that function
#: adds `-lv 4` (D27's cache-type log contract), and verbose logging is itself
#: per-token work on the path this experiment is timing.
BASE_FLAGS: tuple[str, ...] = (
    "--host", "127.0.0.1",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "-fa", "on", "-ub", "512", "-b", "2048",
    "-lm", "none", "--no-kv-unified", "--cont-batching",
)


def base_flags(*, cont_batching: bool = True, ub: int = 512,
               batch: int = 2048) -> list[str]:
    """`BASE_FLAGS` with the three knobs R14's hypotheses 1 and 3 move.

    ADDED 2026-08-14 for `s2/R14.md`. The defaults reproduce `BASE_FLAGS`
    token-for-token, so every condition recorded before this existed is
    re-runnable from the same runner with no argument at all -- that
    byte-identity is the whole point of adding the knobs here rather than
    forking the runner.
    """
    flags = list(BASE_FLAGS)
    flags[flags.index("-ub") + 1] = str(ub)
    flags[flags.index("-b") + 1] = str(batch)
    if not cont_batching:
        # BOTH, deliberately. Continuous batching DEFAULTS to on in this build,
        # so dropping `--cont-batching` alone would not switch it off; and
        # leaving `--cont-batching` in front of `--no-cont-batching` would rest
        # the condition on last-wins argument parsing. Removing the positive
        # and adding the negation makes the recorded argv say what it did.
        flags.remove("--cont-batching")
        flags.append("--no-cont-batching")
    return flags

QUESTION = "What is the record identifier stated in this document?"

#: Neutral filler with no entity bindings other than the one identifier each
#: chunk carries, so R13's detector has an unambiguous foreign string to find
#: and the prompt has no second identifier to confuse it.
FILLER = (
    "The committee reviewed the quarterly maintenance schedule and confirmed "
    "that the routine inspection intervals remain unchanged from the previous "
    "period.",
    "Operational guidance for the facility continues to require that all "
    "scheduled checks be logged at the point of completion rather than "
    "retrospectively.",
    "A summary of the amended handling procedure was circulated to the "
    "relevant teams, who acknowledged receipt without further comment.",
    "The revised timetable allocates additional capacity to the afternoon "
    "window, which had previously been identified as a constraint.",
    "Storage conditions were verified against the published tolerance ranges "
    "and no deviation was recorded during the reporting period.",
    "Training materials were updated to reflect the change in terminology, "
    "with no alteration to the underlying process.",
    "The audit trail requirement applies to every transfer regardless of "
    "duration, and partial entries are not accepted.",
    "Environmental monitoring continued at the agreed frequency and the "
    "results fell within the expected band throughout.",
)


# --------------------------------------------------------------------------- #
# Process memory, without psutil (hypothesis 3).
# --------------------------------------------------------------------------- #


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory_mib(pid: int | None) -> tuple[float | None, float | None]:
    """(working set, private bytes) in MiB, or (None, None) off Windows or on
    any failure -- this is a diagnostic and must never fail a run."""
    if pid is None or os.name != "nt":
        return (None, None)
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | VM_READ, False, pid)
        if not handle:
            return (None, None)
        try:
            counters = _PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb)
            if not ok:
                return (None, None)
            return (counters.WorkingSetSize / 1048576.0,
                    counters.PrivateUsage / 1048576.0)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001 -- diagnostic only
        return (None, None)


# --------------------------------------------------------------------------- #
# The corpus: 128 distinct, same-length documents.
# --------------------------------------------------------------------------- #


def chunk_identifier(idx: int) -> str:
    """A UUID-shaped identifier derived from the index, so the corpus is
    reproducible byte-for-byte across every condition."""
    h = hashlib.sha256(f"s2-occupancy-doc-{idx}".encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def chunk_text(idx: int, sentences: int) -> str:
    """One document: `sentences` neutral sentences with the record identifier
    stated once, at a fixed position near the top."""
    rng = random.Random(9000 + idx)
    body = [rng.choice(FILLER) for _ in range(sentences)]
    body.insert(1, f"Record identifier: {chunk_identifier(idx)}.")
    return "\n".join(body)


# --------------------------------------------------------------------------- #
# The timed transport.
# --------------------------------------------------------------------------- #


@dataclass
class TimedLeaf:
    """One llama-server, addressed one pinned slot at a time, with every
    boundary the wall-clock decomposition needs timed."""

    base_url: str
    system_prefix: str
    max_predict: int
    temperature: float
    top_p: float
    seed: int
    enable_thinking: bool = False
    timeout: float = 240.0
    limits: httpx.Limits | None = None
    markers: tuple[str, ...] = field(default_factory=tuple)
    prefix_tokens: int | None = None
    #: R14 hypothesis 5 (2026-08-14). False = the shipped pattern: break out of
    #: the SSE loop on the final event and let the context manager close the
    #: connection, which aborts the response mid-body. True = read the stream to
    #: its natural end so the SERVER closes it. `rlm/dispatcher.py` uses the
    #: break-on-stop pattern too, so this switch tests production, not just this
    #: runner. `tail_ms` grows under drain by construction -- it now contains the
    #: server's own teardown -- so wall-clock is not comparable across the switch.
    drain_stream: bool = False
    _client: httpx.AsyncClient | None = None

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits or httpx.Limits(
                    max_connections=100, max_keepalive_connections=20))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, path: str) -> tuple[Any, float]:
        t0 = time.perf_counter()
        resp = await self.client().get(f"{self.base_url}{path}")
        ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        return resp.json(), ms

    async def health_ok(self) -> bool:
        try:
            resp = await self.client().get(f"{self.base_url}/health", timeout=5.0)
        except Exception:  # noqa: BLE001 -- 503 while loading, refused while down
            return False
        return resp.status_code == 200

    async def tokenize(self, text: str, *, add_special: bool = False) -> int:
        resp = await self.client().post(
            f"{self.base_url}/tokenize",
            json={"content": text, "add_special": add_special})
        resp.raise_for_status()
        toks = resp.json().get("tokens", [])
        if text and not toks:
            raise RuntimeError("/tokenize returned 0 tokens for non-empty input")
        return len(toks)

    async def render(self, messages: list[dict[str, str]]) -> str:
        resp = await self.client().post(
            f"{self.base_url}/apply-template",
            json={"messages": messages, "add_generation_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": self.enable_thinking}})
        resp.raise_for_status()
        return resp.json()["prompt"]

    async def prepare(self) -> None:
        """Marker set and the rendered §4 head, once. Same derivation as
        `s2.leafcall.PinnedLeafCaller.prepare` -- a drifted prompt here would
        be a different experiment."""
        try:
            props, _ = await self.get("/props")
            template = props.get("chat_template") or ""
        except Exception:  # noqa: BLE001 -- /props widens the guarantee
            template = ""
        self.markers = chat_control_markers(template)
        probe = "RLM-S2-OCC-PREFIX-PROBE"
        rendered = await self.render(
            [{"role": "system", "content": self.system_prefix},
             {"role": "user", "content": probe}])
        cut = rendered.rfind(probe)
        if cut > 0:
            self.prefix_tokens = await self.tokenize(rendered[:cut], add_special=True)

    def compose(self, *, question: str, chunk: str) -> list[dict[str, str]]:
        """§4's shipped layout, through production's own composer."""
        return [
            {"role": "system", "content": self.system_prefix},
            {"role": "user", "content": neutralise_control_tokens(
                compose_leaf_user(question, chunk), self.markers)},
        ]

    async def ask(self, rendered: str, *, id_slot: int) -> dict[str, Any]:
        """One streamed /completion, fully instrumented. No retry."""
        body = {
            "prompt": rendered,
            "n_predict": self.max_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "cache_prompt": True,
            "stream": True,
            "return_tokens": False,
            "id_slot": id_slot,
        }
        parts: list[str] = []
        final: dict[str, Any] | None = None
        t_dispatch = time.perf_counter()
        t_headers = None
        t_first_byte = None
        async with self.client().stream(
                "POST", f"{self.base_url}/completion", json=body,
                timeout=self.timeout) as resp:
            t_headers = time.perf_counter()
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    continue
                if t_first_byte is None:
                    t_first_byte = time.perf_counter()
                ev = json.loads(payload)
                if ev.get("stop"):
                    final = ev
                    if not self.drain_stream:
                        break
                    continue
                if ev.get("content"):
                    parts.append(ev["content"])
        t_end = time.perf_counter()
        if final is None:
            raise RuntimeError("/completion stream ended without a final event")
        tm = final.get("timings", {}) or {}
        return {
            "content": "".join(parts),
            "id_slot_returned": final.get("id_slot", -1),
            "stop_type": final.get("stop_type", ""),
            "truncated": bool(final.get("truncated")),
            "prompt_n": tm.get("prompt_n", 0),
            "cache_n": tm.get("cache_n", 0),
            "predicted_n": tm.get("predicted_n", 0),
            "prompt_ms": tm.get("prompt_ms", 0.0),
            "predicted_ms": tm.get("predicted_ms", 0.0),
            "t_dispatch": t_dispatch,
            "t_headers": t_headers,
            "t_first_byte": t_first_byte,
            "t_end": t_end,
        }


def buckets(rec: dict[str, Any]) -> dict[str, float]:
    """The decomposition. Every number in ms, and they sum to `wall_ms`.

    `queue_ms` deliberately subtracts ONE token of decode from the
    dispatch-to-first-byte gap: the first SSE data event carries the first
    generated token, so that gap contains prefill AND one decode step, and
    charging that step to the queue would make the queue look like it grows
    whenever decode does.
    """
    t0 = rec["t_dispatch"]
    wall_ms = (rec["t_end"] - t0) * 1000.0
    send_ms = (rec["t_headers"] - t0) * 1000.0
    ttfb_ms = ((rec["t_first_byte"] - t0) * 1000.0
               if rec["t_first_byte"] is not None else wall_ms)
    n = max(int(rec["predicted_n"] or 0), 1)
    per_tok = float(rec["predicted_ms"] or 0.0) / n
    queue_ms = ttfb_ms - send_ms - float(rec["prompt_ms"] or 0.0) - per_tok
    stream_ms = wall_ms - ttfb_ms
    tail_ms = stream_ms - (float(rec["predicted_ms"] or 0.0) - per_tok)
    return {
        "wall_ms": wall_ms,
        "send_ms": send_ms,
        "queue_ms": queue_ms,
        "prefill_ms": float(rec["prompt_ms"] or 0.0),
        "decode_ms": float(rec["predicted_ms"] or 0.0),
        "tail_ms": tail_ms,
        "ttfb_ms": ttfb_ms,
        # The §6 definition, verbatim, so this run's numbers sit next to any
        # `steps.latency_queue_ms` a real episode records.
        "latency_queue_ms_spec": ttfb_ms - float(rec["prompt_ms"] or 0.0),
        "residual_ms": wall_ms - float(rec["prompt_ms"] or 0.0)
                       - float(rec["predicted_ms"] or 0.0),
    }


# --------------------------------------------------------------------------- #
# Server lifecycle.
# --------------------------------------------------------------------------- #


def launch_argv(model: str, exe: str, port: int, ctx: int, np: int,
                extra: list[str], *, cont_batching: bool = True,
                ub: int = 512, batch: int = 2048) -> list[str]:
    argv = [exe, "-m", model, "--port", str(port), "-c", str(ctx), "-np", str(np)]
    argv.extend(base_flags(cont_batching=cont_batching, ub=ub, batch=batch))
    argv.extend(extra)
    return argv


async def launch(argv: list[str], log_path: Path,
                 probe: TimedLeaf, timeout_s: float = 900.0) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    env = {**os.environ, "ROCBLAS_USE_HIPBLASLT": "1"}
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"server exited {proc.returncode}; see {log_path}")
        if await probe.health_ok():
            return proc
        await asyncio.sleep(1.0)
    proc.terminate()
    log.close()
    raise RuntimeError(f"server never became healthy; see {log_path}")


def shutdown(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    deadline = time.monotonic() + 60.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if proc.poll() is None:
        proc.kill()
    proc.wait()


# --------------------------------------------------------------------------- #
# The run.
# --------------------------------------------------------------------------- #


def slot_order(calls: int, np: int, order: str, seed: int) -> list[int]:
    """Which slot each call gets. Never a repeat -- R13 stands even here."""
    if calls > np:
        raise SystemExit(f"--calls {calls} exceeds --np {np}: a run may not "
                         "reuse a slot (R13)")
    if order == "asc":
        return list(range(calls))
    if order == "desc":
        return list(range(np - 1, np - 1 - calls, -1))
    if order == "shuffle":
        pool = list(range(np))
        random.Random(seed).shuffle(pool)
        return pool[:calls]
    raise SystemExit(f"unknown --order {order!r}")


async def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(Path(args.config))
    leaf_cfg = cfg.servers.leaf
    exe = str(Path(leaf_cfg.backend_dir) / "llama-server.exe")
    prefix = cfg.prompt_registry().load().leaf_prefix()

    leaf = TimedLeaf(
        base_url=f"http://127.0.0.1:{args.port}",
        system_prefix=prefix,
        max_predict=args.n_predict,
        # `--temperature`/`--top-p` default to None, i.e. to config, so an
        # unqualified run is the shipped sampling. R14 hypothesis 4 (is the
        # degeneracy a sampling pathology?) is the only caller that overrides.
        temperature=(cfg.scaffold.sampling.leaf.temperature
                     if args.temperature is None else args.temperature),
        top_p=(cfg.scaffold.sampling.leaf.top_p
               if args.top_p is None else args.top_p),
        seed=cfg.scaffold.sampling.leaf.seed,
        enable_thinking=cfg.scaffold.leaf.enable_thinking,
        limits=httpx.Limits(max_connections=args.max_connections,
                            max_keepalive_connections=args.max_keepalive),
        drain_stream=args.drain_stream,
    )

    proc: subprocess.Popen | None = None
    argv = launch_argv(str(leaf_cfg.model), exe, args.port, args.ctx, args.np,
                       args.extra.split() if args.extra else [],
                       cont_batching=args.cont_batching, ub=args.ub,
                       batch=args.batch)
    try:
        if not args.no_launch:
            print(f"[{args.condition}] launching: {' '.join(argv)}", flush=True)
            proc = await launch(argv, Path(args.log), leaf)
            print(f"[{args.condition}] healthy, pid {proc.pid}", flush=True)
        else:
            if not await leaf.health_ok():
                raise SystemExit("no healthy server and --no-launch given")
        pid = proc.pid if proc is not None else None

        await leaf.prepare()
        print(f"[{args.condition}] prefix_tokens={leaf.prefix_tokens}", flush=True)

        # --- the corpus, measured once ------------------------------------ #
        sentences = args.sentences
        probe_tokens = await leaf.tokenize(chunk_text(0, sentences))
        # One binary search on document 0; every document has the same shape so
        # the count carries, and each one's measured length is recorded anyway.
        lo, hi = 1, 4 * sentences
        while lo < hi:
            mid = (lo + hi + 1) // 2
            n = await leaf.tokenize(chunk_text(0, mid))
            if n <= args.chunk_tokens:
                lo = mid
            else:
                hi = mid - 1
        sentences = lo
        probe_tokens = await leaf.tokenize(chunk_text(0, sentences))
        print(f"[{args.condition}] {sentences} sentences -> {probe_tokens} "
              f"chunk tokens (target {args.chunk_tokens})", flush=True)

        chunks = {f"occ-doc-{i}": chunk_text(i, sentences) for i in range(args.calls)}
        index = ChunkIndex.from_chunks(chunks)

        # Render every prompt BEFORE the run so /apply-template is not on the
        # timed path (its cost is probed separately, per call, by /health).
        rendered: dict[int, str] = {}
        for i in range(args.calls):
            rendered[i] = await leaf.render(
                leaf.compose(question=QUESTION, chunk=chunks[f"occ-doc-{i}"]))
        rendered_tokens = await leaf.tokenize(rendered[0], add_special=True)
        print(f"[{args.condition}] rendered prompt = {rendered_tokens} tokens; "
              f"slot capacity {args.ctx // args.np}", flush=True)
        if rendered_tokens > args.ctx // args.np:
            raise SystemExit("rendered prompt exceeds slot capacity")

        slots = slot_order(args.calls, args.np, args.order, args.seed)
        run_id = f"{args.condition}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        meta = {
            "run_id": run_id, "condition": args.condition, "argv": argv,
            "np": args.np, "ctx": args.ctx, "order": args.order,
            "calls": args.calls, "concurrency": args.concurrency,
            "max_connections": args.max_connections,
            "max_keepalive": args.max_keepalive,
            "chunk_tokens": probe_tokens, "rendered_tokens": rendered_tokens,
            "prefix_tokens": leaf.prefix_tokens, "extra": args.extra or "",
            "n_predict": args.n_predict,
            "cont_batching": args.cont_batching, "ub": args.ub,
            "batch": args.batch, "temperature": leaf.temperature,
            "top_p": leaf.top_p, "drain_stream": args.drain_stream,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []

        async def one_call(ordinal: int, slot: int) -> dict[str, Any]:
            doc = f"occ-doc-{ordinal}"
            sent = compose_leaf_user(QUESTION, chunks[doc])
            health_ms = slots_ms = None
            if args.probes:
                try:
                    _, health_ms = await leaf.get("/health")
                except Exception:  # noqa: BLE001
                    health_ms = None
                try:
                    _, slots_ms = await leaf.get("/slots")
                except Exception:  # noqa: BLE001
                    slots_ms = None
            ws, priv = process_memory_mib(pid)
            err = None
            try:
                res = await leaf.ask(rendered[ordinal], id_slot=slot)
            except Exception as exc:  # noqa: BLE001 -- a failure is a fact
                rec = {**meta, "ordinal": ordinal, "id_slot": slot,
                       "status": "error", "error": f"{type(exc).__name__}: {exc}",
                       "health_rtt_ms": health_ms, "slots_rtt_ms": slots_ms,
                       "ws_mib": ws, "private_mib": priv}
                return rec
            b = buckets(res)
            verdict = index.foreign(res["content"], sent=sent)
            rec = {
                **meta,
                "ordinal": ordinal,
                "id_slot": slot,
                "id_slot_returned": res["id_slot_returned"],
                "slot_mismatch": res["id_slot_returned"] != slot,
                "status": "ok",
                "error": err,
                "doc": doc,
                "answer": res["content"],
                "answer_correct": chunk_identifier(ordinal) in res["content"],
                "prompt_n": res["prompt_n"], "cache_n": res["cache_n"],
                "predicted_n": res["predicted_n"],
                "stop_type": res["stop_type"], "truncated": res["truncated"],
                "health_rtt_ms": health_ms, "slots_rtt_ms": slots_ms,
                "ws_mib": ws, "private_mib": priv,
                "leak_detected": verdict.detected, "leak_detail": verdict.detail,
                **{k: round(v, 3) for k, v in b.items()},
            }
            return rec

        t_run = time.perf_counter()
        if args.concurrency <= 1:
            for ordinal, slot in enumerate(slots):
                rec = {"occupancy_before": ordinal, **await one_call(ordinal, slot)}
                records.append(rec)
                if rec["status"] == "ok":
                    print(f"  [{args.condition}] #{ordinal:>3} slot {slot:>3} "
                          f"wall {rec['wall_ms']/1000:6.2f}s  send {rec['send_ms']:7.1f} "
                          f"queue {rec['queue_ms']:8.1f}  prefill {rec['prefill_ms']:7.1f} "
                          f"decode {rec['decode_ms']:7.1f}  tail {rec['tail_ms']:7.1f} "
                          f"| /health {(-1 if rec['health_rtt_ms'] is None else rec['health_rtt_ms']):6.1f} "
                          f"/slots {(-1 if rec['slots_rtt_ms'] is None else rec['slots_rtt_ms']):8.1f}",
                          flush=True)
                else:
                    print(f"  [{args.condition}] #{ordinal:>3} slot {slot:>3} "
                          f"ERROR {rec['error']}", flush=True)
        else:
            sem = asyncio.Semaphore(args.concurrency)

            async def guarded(ordinal: int, slot: int) -> dict[str, Any]:
                async with sem:
                    return {"occupancy_before": ordinal,
                            **await one_call(ordinal, slot)}

            records = list(await asyncio.gather(
                *(guarded(o, s) for o, s in enumerate(slots))))
        # THE ONE THING THE FIX MUST NOT BREAK (§7 #3 (d), §4). Under R13 the
        # intra-window re-query -- the second question about a window, on that
        # window's own slot -- is the only cache lever left intact, and it is
        # worth 20-40x. If `--cache-ram 0` also removed it, the recommendation
        # here would be trading one cost for a worse one. So the same document
        # is re-asked on its own slot (same-document reuse, measured clean) and
        # its prefill is recorded next to the cold one.
        if args.requery:
            for ordinal, slot in list(enumerate(slots))[:args.requery]:
                res = await leaf.ask(rendered[ordinal], id_slot=slot)
                b = buckets(res)
                records.append({
                    **meta, "condition": args.condition + "-requery",
                    "occupancy_before": len(slots), "ordinal": ordinal,
                    "id_slot": slot, "id_slot_returned": res["id_slot_returned"],
                    "slot_mismatch": res["id_slot_returned"] != slot,
                    "status": "ok", "error": None, "doc": f"occ-doc-{ordinal}",
                    "answer": res["content"], "answer_correct":
                        chunk_identifier(ordinal) in res["content"],
                    "prompt_n": res["prompt_n"], "cache_n": res["cache_n"],
                    "predicted_n": res["predicted_n"],
                    "stop_type": res["stop_type"], "truncated": res["truncated"],
                    "health_rtt_ms": None, "slots_rtt_ms": None,
                    "ws_mib": None, "private_mib": None,
                    "leak_detected": None, "leak_detail": None,
                    **{k: round(v, 3) for k, v in b.items()},
                })
                print(f"  [{args.condition}-requery] #{ordinal:>3} slot {slot:>3} "
                      f"wall {b['wall_ms']/1000:6.2f}s  prefill {b['prefill_ms']:7.1f} "
                      f"cache_n {res['cache_n']}", flush=True)

        run_s = time.perf_counter() - t_run

        with out.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        ok = [r for r in records if r["status"] == "ok"]
        summary = {
            "run_id": run_id, "condition": args.condition, "n": len(records),
            "ok": len(ok), "run_s": round(run_s, 1),
            "leaks": sum(1 for r in ok if r.get("leak_detected")),
            "slot_mismatches": sum(1 for r in ok if r.get("slot_mismatch")),
            "correct": sum(1 for r in ok if r.get("answer_correct")),
        }
        print(f"[{args.condition}] {json.dumps(summary)}", flush=True)
        return summary
    finally:
        await leaf.aclose()
        if proc is not None:
            shutdown(proc)
            print(f"[{args.condition}] server shut down", flush=True)


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def read_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def band_table(records: list[dict[str, Any]], *, key: str = "occupancy_before",
               width: int = 16) -> list[dict[str, Any]]:
    """Per-band medians of every bucket, banded on occupancy (or slot)."""
    ok = [r for r in records if r["status"] == "ok"]
    bands: dict[int, list[dict[str, Any]]] = {}
    for r in ok:
        bands.setdefault(int(r[key]) // width, []).append(r)
    rows = []
    for b in sorted(bands):
        rs = bands[b]
        rows.append({
            "band": f"{b*width}-{b*width+width-1}",
            "n": len(rs),
            "wall_s": round(med([r["wall_ms"] for r in rs]) / 1000, 3),
            "send_ms": round(med([r["send_ms"] for r in rs]), 1),
            "queue_ms": round(med([r["queue_ms"] for r in rs]), 1),
            "prefill_ms": round(med([r["prefill_ms"] for r in rs]), 1),
            "decode_ms": round(med([r["decode_ms"] for r in rs]), 1),
            "tail_ms": round(med([r["tail_ms"] for r in rs]), 1),
            "residual_ms": round(med([r["residual_ms"] for r in rs]), 1),
            "health_ms": round(med([r["health_rtt_ms"] for r in rs
                                    if r.get("health_rtt_ms") is not None]), 1),
            "slots_ms": round(med([r["slots_rtt_ms"] for r in rs
                                   if r.get("slots_rtt_ms") is not None]), 1),
            "private_mib": round(med([r["private_mib"] for r in rs
                                      if r.get("private_mib") is not None]), 0),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--np", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=327680)
    ap.add_argument("--calls", type=int, default=128)
    ap.add_argument("--order", default="asc", choices=("asc", "desc", "shuffle"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--extra", default="", help="server flags under test")
    ap.add_argument("--chunk-tokens", type=int, default=1024)
    ap.add_argument("--sentences", type=int, default=64)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--requery", type=int, default=0,
                    help="after the fill, re-ask the first N documents on "
                         "their own slots (§7 #3 (d) warm-reuse check)")
    # --- R14 knobs (2026-08-14). Every default reproduces the pre-R14 runner. #
    ap.add_argument("--no-cont-batching", dest="cont_batching",
                    action="store_false", default=True,
                    help="drop --cont-batching and pass --no-cont-batching "
                         "(R14 hypothesis 1: batched decode)")
    ap.add_argument("--ub", type=int, default=512,
                    help="server -ub (R14 hypothesis 3)")
    ap.add_argument("--batch", type=int, default=2048,
                    help="server -b (R14 hypothesis 3)")
    ap.add_argument("--drain-stream", action="store_true", default=False,
                    help="read the SSE stream to its natural end instead of "
                         "breaking on the final event (R14 hypothesis 5)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override leaf sampling temperature (R14 hypothesis 4)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="override leaf sampling top_p (R14 hypothesis 4)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-connections", type=int, default=100)
    ap.add_argument("--max-keepalive", type=int, default=20)
    ap.add_argument("--probes", action="store_true", default=True)
    ap.add_argument("--no-probes", dest="probes", action="store_false")
    ap.add_argument("--no-launch", action="store_true")
    ap.add_argument("--log", default="traces/logs/occupancy-leaf.log")
    ap.add_argument("--out", default=str(RUNS_PATH))
    ap.add_argument("--report", action="store_true",
                    help="print per-condition tables from the JSONL and exit")
    args = ap.parse_args()

    if args.report:
        recs = read_runs(Path(args.out))
        conds: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            conds.setdefault(r["run_id"], []).append(r)
        for run_id, rs in conds.items():
            print(f"\n### {run_id}  (n={len(rs)}, extra={rs[0].get('extra')!r}, "
                  f"np={rs[0].get('np')}, order={rs[0].get('order')})")
            for row in band_table(rs):
                print(json.dumps(row))
        return

    asyncio.run(run_condition(args))


if __name__ == "__main__":
    main()
