"""Shared fixtures. `minimal_cfg_dict` / `valid_cfg` are read from the repo's
real config.yaml so the test suite and the shipped config never drift apart.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import http.server
import io
import json
import os
import re
import socket
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from rlm.config import Config, resolve_prompt_path

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def minimal_cfg_dict() -> dict:
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(raw)


@pytest.fixture
def valid_cfg(minimal_cfg_dict: dict) -> Config:
    return Config.model_validate(minimal_cfg_dict)


@pytest.fixture
def cfg(valid_cfg: Config) -> Config:
    return valid_cfg


@pytest.fixture
def nosubcalls_cfg(minimal_cfg_dict: dict) -> Config:
    """A config with the `rlm-nosubcalls` arm's own root + strategy blocks
    declared -- pointed at file names Task 8 has not authored yet.

    Validates cleanly (unpinned refs are never checked for existence at
    `Config.model_validate` time), but `PromptRegistry.load()` will raise
    `ConfigError` until Task 8 lands those files -- which is exactly the
    `xfail(strict=False)` `test_render_root_no_subcalls_uses_the_nosubcalls_
    body_and_block` is marked for. Task 8 removes that mark; this fixture then
    needs no change, since the paths below are the ones it is expected to add.
    """
    d = copy.deepcopy(minimal_cfg_dict)
    prompts = d["scaffold"]["prompts"]
    prompts["root_nosubcalls"] = {"path": "prompts/root-nosubcalls.v1.md", "sha256": None}
    prompts["strategy_templates_nosubcalls"] = {
        cat: {"path": f"prompts/strat-{cat.replace('_', '-')}-nosubcalls.v1.md", "sha256": None}
        for cat in prompts["strategy_templates"]
    }
    return Config.model_validate(d)


@pytest.fixture
def leaf_prefix() -> str:
    """The pinned registry leaf prefix (`leaf_prefix_text()` as a fixture)."""
    return leaf_prefix_text()


def _in_process_pair():
    """Wire two BridgeEndpoint instances together over two os.pipe() pairs,
    so the bridge's framing/correlation logic is testable without spawning
    a real sandbox process. Must be called from inside a running event loop
    (every test using it is an async def, per pytest-asyncio's auto mode).
    """
    from rlm.bridge import BridgeEndpoint

    loop = asyncio.get_running_loop()
    p2c_r, p2c_w = os.pipe()  # parent writes, child reads
    c2p_r, c2p_w = os.pipe()  # child writes, parent reads
    parent = BridgeEndpoint(c2p_r, p2c_w, loop=loop, tag="parent")
    child = BridgeEndpoint(p2c_r, c2p_w, loop=loop, tag="child")
    return parent, child


# --------------------------------------------------------------------------- #
# C1 sandbox fixtures
#
# `session` spawns a REAL sandbox (Job + AppContainer + bridge) through the
# SandboxManager, with a mock dispatcher answering every llm_query -- C1
# exercised without C4. These tests are slow by construction and Windows-only.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def bootstrap_dir() -> Path:
    """The one-time, EXPLICIT install step: stage the child into its dedicated
    directory and grant ALL APPLICATION PACKAGES read+execute on that directory
    ONLY. The runtime itself never grants an ACL -- `install_bootstrap` defaults
    to `grant_acl=False` -- so the sandbox's inability to read config.yaml,
    prompts/ and traces/ stays a property of the repo, not of a lucky default.
    """
    if sys.platform != "win32":
        pytest.skip("Windows only")
    from rlm.sandbox.manager import install_bootstrap

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    sandbox_cfg = Config.model_validate(raw).scaffold.sandbox
    return install_bootstrap(sandbox_cfg, grant_acl=True)


@pytest.fixture
def manager(bootstrap_dir: Path):
    from rlm.sandbox.manager import SandboxManager

    m = SandboxManager(stderr_dir=os.environ.get("RLM_SANDBOX_STDERR_DIR"))
    yield m
    m.close()


async def mock_llm_query(payload: dict) -> str:
    """The mock dispatcher: answers every sub-call without touching a server.

    Composes the two payload fields the way C4 does, so what the child sent is
    visible in the answer."""
    from rlm.serve.dispatcher import compose_leaf_user

    return f"MOCK:{compose_leaf_user(payload['question'], payload.get('chunk'))}"


@pytest.fixture
async def session(manager, cfg: Config):
    async with manager.session("child-tests", cfg) as s:
        s.on_llm_query(mock_llm_query)
        yield s


# --------------------------------------------------------------------------- #
# C4 dispatcher fixtures: `mock_server` is a REAL loopback HTTP server
# (stdlib http.server, threaded) speaking the /tokenize + /completion (SSE)
# + /props shapes captured in Recipes §serverapi. It must be a real server,
# not monkeypatched httpx, so that streaming and cancellation (closing the
# response context, which genuinely aborts server-side generation per D14)
# are actually exercised end to end.
# --------------------------------------------------------------------------- #


def _read_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    raw = handler.rfile.read(length) if length else b""
    return json.loads(raw.decode("utf-8")) if raw else {}


def _send_json(handler, code: int, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _write_sse_event(handler, ev: dict) -> None:
    handler.wfile.write(f"data: {json.dumps(ev)}\n\n".encode("utf-8"))
    handler.wfile.flush()


# The ChatML control tokens a real Qwen3.6 tokenizer resolves to single
# SPECIAL token ids (measured on-box: `/tokenize` maps `<|im_start|>` to id 6).
# The mock models them for one reason: a whitespace tokenizer cannot tell a
# forged `<|im_start|>` from the words around it, so a mock without this could
# never certify that scaffold-side sanitisation actually kept a forged marker
# off the wire -- it would agree with any implementation at all.
CONTROL_TOKEN_IDS: dict[str, int] = {
    "<|im_start|>": 6, "<|im_end|>": 7, "<think>": 8, "</think>": 9,
    # A template-specific marker, so tests can prove the marker set is derived
    # from /props's chat_template rather than hardcoded in the scaffold.
    "<|custom_marker|>": 10,
}
_CONTROL_SPLIT_RE = re.compile(
    "(" + "|".join(re.escape(m) for m in CONTROL_TOKEN_IDS) + ")")
BOS_TOKEN_ID = 1


def _naive_tokenize_pieces(text: str, *, add_special: bool = False) -> list[dict]:
    """Whitespace tokenizer with SPECIAL-token parsing, `with_pieces`-shaped.

    Two real behaviours are modelled, because C4 depends on both:
      * control-token literals resolve to their own single piece with a
        special id (llama.cpp parses them in the prompt string, which is what
        makes an unsanitised forged marker a real control token on the wire);
      * `add_special` prepends BOS -- the measured +1 between a pre-flight
        `/tokenize` (default `add_special=false`) and `/completion`'s
        `prompt_n`.
    """
    out: list[dict] = []
    if add_special:
        out.append({"id": BOS_TOKEN_ID, "piece": "<s>"})
    if not text:
        return out
    for part in _CONTROL_SPLIT_RE.split(text):
        if not part:
            continue
        if part in CONTROL_TOKEN_IDS:
            out.append({"id": CONTROL_TOKEN_IDS[part], "piece": part})
        else:
            out.extend({"id": 1000 + i, "piece": word}
                       for i, word in enumerate(part.split()))
    return out


def _naive_tokenize(text: str, *, add_special: bool = False) -> list[int]:
    return [t["id"] for t in _naive_tokenize_pieces(text, add_special=add_special)]


def _tokenize_response(body: dict) -> dict:
    text = body.get("content", "") or ""
    pieces = _naive_tokenize_pieces(text, add_special=bool(body.get("add_special")))
    if body.get("with_pieces"):
        return {"tokens": pieces}
    return {"tokens": [t["id"] for t in pieces]}


def _render_chatml(messages: list[dict], enable_thinking: bool) -> str:
    """Qwen3.8-27B's ChatML shape (checks/fixtures/repetition/qwen38_chat_template.jinja),
    the branches the scaffold exercises. Shared by both fake servers: since the
    S2 leaf-template fix, the LEAF renders through /apply-template too (D14).

    The load-bearing detail: every PAST assistant message is rendered with the
    template's own `<think>\\n{reasoning_content}\\n</think>\\n\\n` in front of
    `content` (preserve_thinking undefined -> the preserving branch), and
    `content` is emitted verbatim -- the template never parses think tags out
    of it. The generation prompt is `<think>\\n\\n</think>\\n\\n` with thinking
    off and `<think>\\n` with it on."""
    parts = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if m["role"] == "assistant":
            reasoning = (m.get("reasoning_content") or "").strip()
            parts.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n"
                         f"{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


def _user_segment(prompt: str) -> str:
    """The last user message body inside a ChatML-rendered prompt (the whole
    string when it is not rendered at all).

    The mock server's behaviour switches (`"slow"`) and its echo are keyed on
    what the CALLER asked, which since the leaf-template fix is no longer the
    string that arrives at /completion -- that is now the rendered prompt."""
    marker = "<|im_start|>user\n"
    if marker not in prompt:
        return prompt
    return prompt.rsplit(marker, 1)[1].split("<|im_end|>", 1)[0]


_LEAF_PREFIX: str | None = None


def leaf_prefix_text() -> str:
    """The registry's leaf system prefix, read exactly the way the runtime
    reads it (changelog header stripped by PromptRegistry). Paths come from
    the shipped config.yaml so a prompt rename cannot leave this behind."""
    global _LEAF_PREFIX
    if _LEAF_PREFIX is None:
        from rlm.config import PromptRegistry

        raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
        prompts = raw["scaffold"]["prompts"]
        _LEAF_PREFIX = PromptRegistry.from_files(
            root_path=resolve_prompt_path(Path(prompts["root"]["path"])),
            leaf_prefix_path=resolve_prompt_path(Path(prompts["leaf_prefix"]["path"])),
            strategy_paths={},
        ).leaf_prefix()
    return _LEAF_PREFIX


class MockLlamaServer:
    """Stub of the leaf-shaped llama-server endpoints C4 talks to."""

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.max_concurrent = 0
        self.restart_count = 0
        self.last_request_disconnected = False
        self.last_completion_body: dict | None = None
        # R13: how many slots this fake server has, and what it does with the
        # `id_slot` it is asked for. `answer` overrides the echo so a test can
        # put a foreign identifier in a leaf answer; `slot_override` forces the
        # reported slot so a test can stage a mismatch on an IN-range request.
        self.total_slots = 8
        self.answer: str | None = None
        #: Successive answers, one per /completion, for tests whose subject is
        #: what happens BETWEEN attempts -- the envelope's retry-on-parse-failure
        #: is the only way to see a call recover from a bad draw. The last entry
        #: repeats once the script runs out, so a test can end on a good answer
        #: without counting attempts.
        self.answer_script: list[str] | None = None
        self.slot_override: int | None = None
        self.completion_count = 0
        # Request-level capture, for the D14 leaf-template contract: the
        # ORDER of the endpoints hit, every /apply-template body, and every
        # string /apply-template handed back (which is what /completion must
        # then receive, byte for byte).
        self.request_paths: list[str] = []
        self.template_bodies: list[dict] = []
        self.rendered_prompts: list[str] = []
        self.completion_bodies: list[dict] = []
        self.tokenize_bodies: list[dict] = []
        # /props is what C4 derives its control-marker set from. It is counted
        # so a test can prove that derivation costs ONE round trip per
        # dispatcher, not one per call.
        self.props_count = 0
        self.healthy = True
        #: `take_down()`/`bring_up()`: is the process there at all? (Distinct
        #: from `healthy`, which is a process that IS there and answers 503
        #: while it loads its model.)
        self.down = False
        self.chat_template = "mock-template"
        self._concurrent = 0
        self._fail_remaining = 0
        self._lock = threading.Lock()
        self._dispatchers: list[Any] = []
        # Every live connection, so `take_down()` can break them. Closing the
        # LISTENER is not enough to model a killed process: this handler speaks
        # HTTP/1.1, httpx keeps its connections pooled, and a keep-alive
        # connection's handler thread keeps answering long after the listener
        # is gone -- so a "removed" server would go on serving the very
        # requests a rotation test needs to see fail.
        self._conns: set[Any] = set()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: A002 -- stdlib signature
                pass

            def setup(self):
                super().setup()
                with outer._lock:
                    outer._conns.add(self.connection)

            def finish(self):
                with outer._lock:
                    outer._conns.discard(self.connection)
                super().finish()

            def handle_one_request(self):
                # `take_down()` -> the process is gone: drop the connection
                # with no response, which is what the client sees when the
                # server it was talking to is killed.
                if outer.down:
                    self.close_connection = True
                    with contextlib.suppress(OSError):
                        self.connection.shutdown(socket.SHUT_RDWR)
                    return
                super().handle_one_request()

            def do_GET(self):
                if self.path == "/health":
                    # llama-server answers 200 {"status":"ok"} once loaded and
                    # 503 while loading; `healthy` lets a test stage the wait a
                    # rotation's readiness poll actually sits through.
                    _send_json(self, 200 if outer.healthy else 503,
                               {"status": "ok" if outer.healthy else "loading model"})
                elif self.path == "/props":
                    with outer._lock:
                        outer.props_count += 1
                    _send_json(self, 200, {
                        "model_path": "mock-leaf.gguf",
                        "total_slots": outer.total_slots,
                        "default_generation_settings": {"n_ctx": 4096},
                        "build_info": "mock",
                        "chat_template": outer.chat_template,
                        "media_marker": "mock-nonce",
                        "is_sleeping": False,
                    })
                else:
                    _send_json(self, 404, {"error": "not found"})

            def do_POST(self):
                outer.request_paths.append(self.path)
                if self.path == "/tokenize":
                    body = _read_json_body(self)
                    outer.tokenize_bodies.append(body)
                    _send_json(self, 200, _tokenize_response(body))
                elif self.path == "/apply-template":
                    body = _read_json_body(self)
                    outer.template_bodies.append(body)
                    enable_thinking = bool(
                        (body.get("chat_template_kwargs") or {}).get("enable_thinking", True))
                    rendered = _render_chatml(body.get("messages", []), enable_thinking)
                    outer.rendered_prompts.append(rendered)
                    _send_json(self, 200, {"prompt": rendered})
                elif self.path == "/completion":
                    self._handle_completion()
                else:
                    _send_json(self, 404, {"error": "not found"})

            def _handle_completion(self):
                body = _read_json_body(self)
                prompt = body.get("prompt", "")
                outer.last_completion_body = body
                outer.completion_bodies.append(body)
                with outer._lock:
                    outer.dispatch_count += 1
                    outer._concurrent += 1
                    outer.max_concurrent = max(outer.max_concurrent, outer._concurrent)
                    fail = outer._fail_remaining > 0
                    if fail:
                        outer._fail_remaining -= 1
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.close_connection = True
                    self.end_headers()
                    if fail:
                        # Simulate a broken attempt: one delta, no final
                        # event, then the connection just closes.
                        try:
                            _write_sse_event(self, {"content": "partial", "stop": False,
                                                     "id_slot": -1})
                        except OSError:
                            pass
                        return
                    served = outer.served_slot(body.get("id_slot"))
                    if _user_segment(prompt) == "slow":
                        self._stream_slow(served)
                    else:
                        self._stream_normal(_user_segment(prompt), served)
                finally:
                    with outer._lock:
                        outer._concurrent -= 1

            def _stream_normal(self, prompt, served):
                content = outer.next_answer(prompt)
                _write_sse_event(self, {"content": content, "stop": False,
                                         "id_slot": -1, "tokens_predicted": 1,
                                         "tokens_evaluated": 4})
                _write_sse_event(self, {
                    "content": "", "stop": True, "id_slot": served, "stop_type": "eos",
                    "tokens_predicted": max(1, len(content.split())),
                    "tokens_evaluated": 4, "truncated": False, "tokens_cached": 4,
                    "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 5.0,
                                "predicted_n": max(1, len(content.split())),
                                "predicted_ms": 10.0},
                })

            def _stream_slow(self, served=0):
                # Long enough that a cancel fired ~0.1s in reliably lands
                # mid-stream. Detection is event-driven, not polled: a
                # dedicated thread blocks on recv() on this same connection,
                # which the client never sends more bytes on, so recv()
                # only ever unblocks when the client closes it (D14:
                # cancellation genuinely aborts server-side generation).
                # Gating detection on the writer loop's own next write
                # attempt would race the assertion in
                # test_cancellation_aborts_the_stream, which checks
                # immediately after `await task` with no extra wait.
                sock = self.connection

                def _watch_for_close():
                    try:
                        sock.recv(1)
                    except OSError:
                        pass
                    outer.last_request_disconnected = True

                watcher = threading.Thread(target=_watch_for_close, daemon=True)
                watcher.start()
                try:
                    for i in range(300):
                        _write_sse_event(self, {"content": "x", "stop": False,
                                                 "id_slot": -1, "tokens_predicted": i + 1})
                        time.sleep(0.02)
                    _write_sse_event(self, {
                        "content": "", "stop": True, "id_slot": served, "stop_type": "limit",
                        "tokens_predicted": 300, "tokens_evaluated": 4, "tokens_cached": 4,
                        "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 1.0,
                                    "predicted_n": 300, "predicted_ms": 600.0},
                    })
                except OSError:
                    pass
                finally:
                    watcher.join(timeout=2.0)

        self._handler = Handler
        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # -- genuine removal, for rotation tests ------------------------------- #
    #
    # `FakeLeafProcess` (checks/test_episode.py) replaces NOTHING: the mock
    # server keeps answering across a "restart", so every rotation test written
    # against it passes whether or not the scaffold actually quiesced the
    # traffic it was about to pull the process out from under. These two make
    # the removal real -- stop listening (new connections are refused, exactly
    # like a killed process), then rebind the SAME port, which is what a
    # relaunched llama-server does.

    def take_down(self) -> None:
        """The process is gone: every live connection is broken and every
        request that arrives is answered by a reset.

        Modelled as a RESET rather than as a closed listener, deliberately, and
        the difference is measured. Closing the listener is what a killed
        process really does, but on Windows loopback httpx spends ~2 s
        retrying the connect before it gives up (measured), so a sub-second
        outage is silently papered over and a test built on it asserts nothing.
        A reset is the same failure class for the client -- the request dies
        with no response, which is exactly what a pre-flight round trip that
        has no retry cannot survive -- and it happens deterministically, on the
        first attempt, at the moment the process disappears.

        Breaking the live connections is the other half: this handler speaks
        HTTP/1.1 and httpx keeps its connections pooled, so a client that
        already holds one would otherwise keep being served by a handler
        thread belonging to a process the test believes is gone.
        """
        self.down = True
        with self._lock:
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                conn.close()

    def bring_up(self) -> None:
        """A fresh process answering on the same port, as a relaunch does."""
        self.down = False

    def served_slot(self, requested: int | None) -> int:
        """Which slot this server answers on -- llama.cpp's MEASURED
        behaviour, which is the whole reason C4 must assert the returned id.

        An in-range `id_slot` is honoured, including under contention (four
        concurrent requests pinned to slot 5 were all queued onto slot 5). An
        OUT-OF-RANGE one is silently reassigned and still answers HTTP 200 --
        measured on a 128-slot server: asked for 200, got 72
        (`milestones/s2/R13-mitigations.md` §4.5). No error, no warning, and the caller
        believes it holds a virgin slot while sharing a used one."""
        if self.slot_override is not None:
            return self.slot_override
        if requested is None:
            return 0
        if 0 <= requested < self.total_slots:
            return requested
        return requested % self.total_slots

    def next_answer(self, prompt: str) -> str:
        """`answer_script` wins, then `answer`, then the echo every existing
        test is written against."""
        with self._lock:
            i = self.completion_count
            self.completion_count += 1
        if self.answer_script:
            return self.answer_script[min(i, len(self.answer_script) - 1)]
        if self.answer is not None:
            return self.answer
        return f"echo:{prompt}"

    def requested_slots(self) -> list[int | None]:
        return [b.get("id_slot") for b in self.completion_bodies]

    def fail_times(self, n: int) -> None:
        self._fail_remaining = n

    def kill(self) -> None:
        """Simulate server death: stop listening, so new connections fail
        with a connection-refused error, same as a crashed process."""
        self.shutdown()

    def dispatcher(self, *, slot_capacity_tokens: int = 32768, parallel: int = 4,
                    max_predict: int = 64, max_attempts: int = 3,
                    temperature: float = 0.3, top_p: float = 0.9, seed: int = 1,
                    backoff_s: list[float] | None = None,
                    system_prefix: str | None = None,
                    enable_thinking: bool = False,
                    slot_pool: int | None = None,
                    envelope: bool = False):
        from rlm.config import Retries
        from rlm.serve.dispatcher import DispatchTarget, LLMDispatcher, ServerClient, SlotPool

        client = ServerClient(self.base_url, timeout=5.0)
        # The REAL registry prefix by default (§4/§5: the leaf prefix is
        # never an inline string), so every dispatcher test runs against the
        # same bytes the runtime ships.
        target = DispatchTarget(client=client, max_predict=max_predict,
                                 slot_capacity_tokens=slot_capacity_tokens,
                                 temperature=temperature, top_p=top_p, seed=seed,
                                 system_prefix=(leaf_prefix_text() if system_prefix is None
                                                else system_prefix),
                                 enable_thinking=enable_thinking,
                                 envelope=envelope)
        # Fast backoff by default -- retry TIMING isn't what most of these
        # tests assert on, and the real 1s/4s backoff (scaffold.retries in
        # config.yaml) would make them take ~5s for no additional coverage.
        # A caller that IS testing backoff timing (fix round 1: the
        # semaphore-around-backoff concurrency test) passes backoff_s
        # explicitly.
        retries = Retries(max_attempts=max_attempts, backoff_s=backoff_s or [0.01, 0.02],
                           per_call_timeout_s=5.0)
        # Only "leaf" -- mirrors the real from_config() shape (fix 4): root
        # traffic never reaches LLMDispatcher, so no test should be able to
        # exercise role="root" here either.
        # R13 slot discipline. Production ties the pool to `--parallel`
        # (`from_config`), which is also the number of WINDOWS one leaf process
        # can serve before it must be restarted. `slot_pool` decouples the two
        # HERE ONLY, for tests whose subject is the semaphore or the retry
        # loop rather than the slot rule, and which would otherwise be capped
        # at `parallel` distinct windows.
        pool = SlotPool(parallel if slot_pool is None else slot_pool)
        dispatcher = LLMDispatcher(targets={"leaf": target},
                                    parallel=parallel, retries=retries, slots=pool)
        self._dispatchers.append(dispatcher)
        return dispatcher

    async def aclose_dispatchers(self) -> None:
        for d in self._dispatchers:
            await d.aclose()
        self._dispatchers.clear()

    def shutdown(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except OSError:
            pass


@pytest.fixture
async def mock_server():
    """Async fixture (not a plain `yield` fixture) so teardown -- closing
    every httpx.AsyncClient `.dispatcher()` handed out -- runs inside the
    SAME event loop the test used. A synchronous fixture's teardown runs
    after that loop is gone, which leaves ProactorEventLoop transports to be
    garbage-collected with no loop to close them on, firing
    ResourceWarning/PytestUnraisableExceptionWarning at the NEXT test's
    setup (observed while writing this suite)."""
    server = MockLlamaServer()
    yield server
    await server.aclose_dispatchers()
    server.shutdown()


# --------------------------------------------------------------------------- #
# C4/rootclient fixture: `fake_root_server` renders a ChatML-shaped prompt
# for /apply-template (Qwen3.6's own format, recipes §serverapi) and always
# answers /completion with one fenced ```repl block, so rootclient's
# render -> hash -> dispatch -> parse cycle is exercised end to end.
# --------------------------------------------------------------------------- #


class FakeRootServer:
    """Stub of the root-shaped llama-server endpoints rootclient talks to.

    `script`, when given, is the exact sequence of raw completions the fake
    root emits, one per turn -- that is how `episode_env` drives a whole
    episode deterministically. Running past the end of the script answers
    HTTP 500 rather than looping: an episode that outlives its script is a
    test bug, and it must surface as a loud `error` outcome instead of
    spinning until the wall clock.
    """

    def __init__(self, base_cfg_dict: dict, script: list[str] | None = None) -> None:
        self._base_cfg_dict = base_cfg_dict
        self.script = list(script) if script is not None else None
        self.turns = 0
        self.last_completion_prompt: str | None = None
        self.last_completion_body: dict | None = None
        self.last_template_kwargs: dict | None = None
        self._clients: list[Any] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: A002
                pass

            def do_GET(self):
                if self.path == "/props":
                    _send_json(self, 200, {
                        "model_path": "mock-root.gguf",
                        "total_slots": 1,
                        "default_generation_settings": {"n_ctx": 32768},
                        "build_info": "mock",
                        "chat_template": "mock-root-template",
                        "media_marker": "mock-nonce",
                        "is_sleeping": False,
                    })
                else:
                    _send_json(self, 404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/apply-template":
                    body = _read_json_body(self)
                    outer.last_template_kwargs = body
                    enable_thinking = bool(
                        (body.get("chat_template_kwargs") or {}).get("enable_thinking", True))
                    rendered = _render_chatml(body.get("messages", []), enable_thinking)
                    _send_json(self, 200, {"prompt": rendered})
                elif self.path == "/tokenize":
                    _send_json(self, 200, _tokenize_response(_read_json_body(self)))
                elif self.path == "/completion":
                    self._handle_completion()
                else:
                    _send_json(self, 404, {"error": "not found"})

            def _handle_completion(self):
                body = _read_json_body(self)
                outer.last_completion_prompt = body.get("prompt", "")
                outer.last_completion_body = body
                if outer.script is None:
                    content = "```repl\nfinal_answer(1)\n```"
                elif outer.turns < len(outer.script):
                    content = outer.script[outer.turns]
                else:
                    _send_json(self, 500, {"error": "root script exhausted"})
                    return
                outer.turns += 1
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                _write_sse_event(self, {"content": content, "stop": False,
                                         "id_slot": -1, "tokens_predicted": 1,
                                         "tokens_evaluated": 4})
                _write_sse_event(self, {
                    "content": "", "stop": True, "id_slot": 0, "stop_type": "eos",
                    "tokens_predicted": max(1, len(content.split())),
                    "tokens_evaluated": 4, "truncated": False, "tokens_cached": 4,
                    "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 3.0,
                                "predicted_n": max(1, len(content.split())),
                                "predicted_ms": 6.0},
                })

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def conversation(self, *, system: str | None = None, enable_thinking: bool | None = None,
                     seed_schedule: str | None = None, history_mode: str | None = None):
        from rlm.serve.dispatcher import ServerClient
        from rlm.serve.rootclient import RootConversation

        raw = copy.deepcopy(self._base_cfg_dict)
        raw["servers"]["root"]["port"] = self.port
        if enable_thinking is not None:
            raw["scaffold"]["root"]["enable_thinking"] = enable_thinking
        if seed_schedule is not None:
            raw["scaffold"]["root"]["seed_schedule"] = seed_schedule
        if history_mode is not None:
            raw["scaffold"]["root"]["history_mode"] = history_mode
        cfg = Config.model_validate(raw)
        client = ServerClient(self.base_url, timeout=5.0)
        self._clients.append(client)
        return RootConversation(client, cfg, system=system)

    async def aclose_clients(self) -> None:
        for c in self._clients:
            await c.aclose()
        self._clients.clear()

    def shutdown(self) -> None:
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except OSError:
            pass


@pytest.fixture
async def fake_root_server(minimal_cfg_dict: dict):
    """Async fixture -- see `mock_server`'s docstring for why teardown must
    run in the test's own event loop."""
    server = FakeRootServer(minimal_cfg_dict)
    yield server
    await server.aclose_clients()
    server.shutdown()


# --------------------------------------------------------------------------- #
# Task 15/16: whole-episode fixtures.
#
# These run the REAL composition root: real sandbox (Job + AppContainer +
# bridge), real C2/C3/C5/C6, a real loopback HTTP root server (`FakeRootServer`
# driven by a script), and a canned dispatcher standing in for C4's leaf
# traffic -- i.e. exactly spec §5's dry-run mode, which is why the config these
# fixtures build carries `scaffold.dispatcher: mock` and every episode they
# produce is flagged `dry_run=true`.
# --------------------------------------------------------------------------- #


class CannedDispatcher:
    """`MockDispatcher`'s interface with a default answer.

    `MockDispatcher` refuses a prompt it has no fixture for, which is the
    right behaviour for a real dry run (an unkeyed prompt means the fixture
    file is stale) but useless for tests that fan out over generated prompts.
    This subclass mints a fixture on first sight; anything explicitly seeded
    still wins.
    """

    def __init__(self, fixtures: dict[str, str] | None = None, *, parallel: int = 8) -> None:
        from rlm.serve.dispatcher import MockDispatcher

        self._inner = MockDispatcher(dict(fixtures or {}), parallel=parallel)
        #: The `seed=`/`n_predict=` every caller passed, in order. §8's seed
        #: discipline is only real if the seed reaches the dispatcher PER
        #: CALL, and B2's summary budget only binds if `n_predict` does.
        self.seeds: list[int | None] = []
        self.n_predicts: list[int | None] = []

    def __getattr__(self, name):          # semaphore, steps, last_step, ...
        return getattr(self._inner, name)

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        return await self._inner.count_tokens(text, role=role)

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, seed: int | None = None,
                     n_predict: int | None = None) -> str:
        # `seed`/`n_predict` recorded, never consulted: this double replays
        # fixtures, so the ASSERTION a test makes about them is that the caller
        # passed the right ones -- not that a draw changed or a decode stopped.
        # Both are declared rather than swallowed by `**kwargs` for the reason
        # `_accepts_slot_id` gives: a double that absorbs an unknown keyword
        # lets a caller believe it applied something it never sent.
        self.seeds.append(seed)
        self.n_predicts.append(n_predict)
        import hashlib

        from rlm.serve.dispatcher import compose_leaf_user

        composed = compose_leaf_user(prompt, chunk)
        key = f"{role}:{hashlib.sha256(composed.encode('utf-8')).hexdigest()}"
        self._inner._fixtures.setdefault(key, f"LEAF:{composed}")
        return await self._inner.query(prompt, role=role, call_id=call_id,
                                        chunk=chunk)


def _point_at_mock_leaf(raw: dict, port: int) -> None:
    """Re-point the leaf at a `MockLlamaServer`, matching what its /props
    reports (model, total_slots, per-slot n_ctx) so the §4 handshake -- the one
    a slot-pool rotation must re-run before resuming -- is genuinely exercised
    rather than skipped."""
    leaf = raw["servers"]["leaf"]
    leaf["port"] = port
    leaf["model"] = "mock-leaf.gguf"          # == MockLlamaServer /props
    leaf["parallel"] = 8                      # == /props total_slots
    leaf["ctx"] = 8 * 4096                    # per-slot 4096 == /props n_ctx
    raw["scaffold"]["dispatch_concurrency"] = 8


def _episode_cfg_dict(base: dict, *, tmp_path: Path, root_port: int, **over) -> dict:
    """The shipped config, re-pointed at a fake root server and a tmp trace
    store. Prompt paths are absolutized so the registry's sha256 pins resolve
    regardless of the working directory; the model path is set to whatever
    `FakeRootServer./props` reports, so the §4 startup handshake is genuinely
    exercised rather than bypassed."""
    raw = copy.deepcopy(base)
    raw["servers"]["root"]["port"] = root_port
    raw["servers"]["root"]["model"] = "mock-root.gguf"   # == FakeRootServer /props
    raw["scaffold"]["dispatcher"] = "mock"
    prompts = raw["scaffold"]["prompts"]
    prompts["root"]["path"] = str(resolve_prompt_path(Path(prompts["root"]["path"])))
    prompts["leaf_prefix"]["path"] = str(resolve_prompt_path(Path(prompts["leaf_prefix"]["path"])))
    if prompts.get("leaf_envelope"):
        prompts["leaf_envelope"]["path"] = str(resolve_prompt_path(Path(prompts["leaf_envelope"]["path"])))
    for ref in prompts["strategy_templates"].values():
        ref["path"] = str(REPO_ROOT / ref["path"])
    for ref in (prompts.get("baselines") or {}).values():
        ref["path"] = str(REPO_ROOT / ref["path"])
    raw["trace"]["db_path"] = str(tmp_path / "rlm.duckdb")
    raw["trace"]["blob_root"] = str(tmp_path / "blobs")
    if over.get("truncation_cap") is not None:
        raw["scaffold"]["truncation_cap_chars"] = over["truncation_cap"]
    if over.get("max_wall_clock_s") is not None:
        raw["scaffold"]["budgets"]["max_wall_clock_s"] = over["max_wall_clock_s"]
    if over.get("max_subcalls") is not None:
        raw["scaffold"]["budgets"]["max_subcalls"] = over["max_subcalls"]
    if over.get("max_total_tokens") is not None:
        raw["scaffold"]["budgets"]["max_total_tokens"] = over["max_total_tokens"]
    if over.get("leaf_port") is not None:
        _point_at_mock_leaf(raw, over["leaf_port"])
    if over.get("leaf_seed") is not None:
        # §8's replicate identity, patched on the RAW dict exactly as
        # `rlm.measure.bench.seeded_config` patches it per attempt.
        raw["scaffold"]["sampling"]["leaf"]["seed"] = over["leaf_seed"]
    if over.get("history_mode") is not None:
        raw["scaffold"]["root"]["history_mode"] = over["history_mode"]
    return raw


def _rows(con, sql: str, params: list | None = None) -> list[dict]:
    """DuckDB hands UUID columns back as `uuid.UUID`; the scaffold speaks in
    strings (EpisodeResult.episode_id, blob paths), so normalise once here
    rather than at every assertion."""
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [
        {c: (str(v) if isinstance(v, uuid.UUID) else v) for c, v in zip(cols, r)}
        for r in cur.fetchall()
    ]


def _decode_episode_row(row: dict) -> dict:
    row = dict(row)
    snap = row.get("config_snapshot")
    if isinstance(snap, str):
        row["config_snapshot"] = json.loads(snap)
    return row


class _EpisodeEnv:
    def __init__(self, *, cfg, task, dispatcher, server, run_kwargs: dict) -> None:
        self.cfg = cfg
        self.task = task
        self.dispatcher = dispatcher
        self.server = server
        self.run_kwargs = run_kwargs
        self._episode: dict | None = None
        self._steps: list[dict] = []
        self._lifecycle_stream = io.StringIO()

    def lifecycle_events(self) -> list[dict]:
        """The JSONL lifecycle log this episode wrote (spec §5): the narrow
        channel for what the trace store structurally cannot record."""
        return [json.loads(line) for line in
                self._lifecycle_stream.getvalue().splitlines() if line.strip()]

    async def run(self):
        from rlm.episode import run_episode
        from rlm.trace.lifecycle import Lifecycle
        from rlm.trace import TraceLogger

        tl = TraceLogger(self.cfg.trace.db_path, self.cfg.trace.blob_root)
        await tl.start()
        lifecycle = Lifecycle(None, stream=self._lifecycle_stream)
        try:
            result = await run_episode(self.task, self.cfg, dispatcher=self.dispatcher,
                                       trace=tl, lifecycle=lifecycle, **self.run_kwargs)
        finally:
            await tl.drain()
            await tl.aclose()
        self._load()
        return result

    def _load(self) -> None:
        con = duckdb.connect(str(self.cfg.trace.db_path), read_only=True)
        try:
            eps = _rows(con, "SELECT * FROM episodes ORDER BY started_at")
            self._episode = _decode_episode_row(eps[-1]) if eps else None
            self._steps = _rows(con, "SELECT * FROM steps ORDER BY step_idx")
        finally:
            con.close()

    def episode_row(self) -> dict:
        assert self._episode is not None, "call await env.run() first"
        return self._episode

    def steps(self) -> list[dict]:
        return self._steps

    def blob(self, rel: str) -> bytes:
        return (Path(self.cfg.trace.blob_root) / rel).read_bytes()

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def episode_env(minimal_cfg_dict: dict, tmp_path: Path, bootstrap_dir: Path):
    """Factory: `episode_env(root_script=[...]) -> env`, then `await env.run()`."""
    from rlm.config import Config
    from rlm.episode import Task

    built: list[_EpisodeEnv] = []

    def factory(*, root_script=None, context="", task_text="What is the answer?",
                category="default", answer=None, truncation_cap=None,
                max_wall_clock_s=None, max_subcalls=None, max_total_tokens=None,
                max_turns=None, leaf_fixtures=None, dispatcher=None,
                leaf_port=None, process_manager=None, leaf_seed=None,
                history_mode=None):
        server = FakeRootServer(minimal_cfg_dict, script=root_script)
        raw = _episode_cfg_dict(minimal_cfg_dict, tmp_path=tmp_path,
                                 root_port=server.port, truncation_cap=truncation_cap,
                                 max_wall_clock_s=max_wall_clock_s,
                                 max_subcalls=max_subcalls,
                                 max_total_tokens=max_total_tokens,
                                 leaf_port=leaf_port, leaf_seed=leaf_seed,
                                 history_mode=history_mode)
        cfg = Config.model_validate(raw)
        task = Task(task_id="fixture-task", text=task_text, context=context,
                    category=category, answer=answer)
        env = _EpisodeEnv(
            cfg=cfg, task=task,
            dispatcher=dispatcher or CannedDispatcher(
                leaf_fixtures, parallel=cfg.scaffold.dispatch_concurrency),
            server=server,
            run_kwargs={"max_turns": max_turns,
                        "process_manager": process_manager},
        )
        built.append(env)
        return env

    yield factory
    for env in built:
        env.close()


# --------------------------------------------------------------------------- #
# Task 16: CLI fixtures. `main()` is synchronous, so these are too -- the fake
# root server is a stdlib threading HTTP server, and every asyncio object the
# CLI builds lives and dies inside its own `asyncio.run`.
# --------------------------------------------------------------------------- #


def _write_cfg(path: Path, raw: dict) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def valid_config_file(minimal_cfg_dict: dict, tmp_path: Path) -> Path:
    """The shipped config with only its trace paths redirected: `rlm validate`
    must be exercised against the REAL sandbox interpreter and bootstrap dir,
    since the D7 confinement probe is the point of it."""
    raw = copy.deepcopy(minimal_cfg_dict)
    prompts = raw["scaffold"]["prompts"]
    prompts["root"]["path"] = str(resolve_prompt_path(Path(prompts["root"]["path"])))
    prompts["leaf_prefix"]["path"] = str(resolve_prompt_path(Path(prompts["leaf_prefix"]["path"])))
    if prompts.get("leaf_envelope"):
        prompts["leaf_envelope"]["path"] = str(resolve_prompt_path(Path(prompts["leaf_envelope"]["path"])))
    for ref in prompts["strategy_templates"].values():
        ref["path"] = str(REPO_ROOT / ref["path"])
    for ref in (prompts.get("baselines") or {}).values():
        ref["path"] = str(REPO_ROOT / ref["path"])
    raw["trace"]["db_path"] = str(tmp_path / "rlm.duckdb")
    raw["trace"]["blob_root"] = str(tmp_path / "blobs")
    # …and its launch logs. The shipped config points these at
    # `traces/logs/*-server.log` RELATIVE TO THE CWD, so a suite run from the
    # repo root while real servers happen to be up reads real, current logs --
    # and `test_validate_refuses_when_the_cache_type_is_unverified`, whose whole
    # premise is that no launch log exists, silently inverts. Found the first
    # time the S1 gate ran the suite with both servers running (S1, 2026-08-13).
    for role in ("root", "leaf"):
        raw["servers"][role]["log_path"] = str(tmp_path / f"{role}-server.log")
    return _write_cfg(tmp_path / "config.yaml", raw)


class _MockEpisodeEnv:
    def __init__(self, *, config_file: Path, task_file: Path, server, tmp_path: Path) -> None:
        self.config_file = config_file
        self.task_file = task_file
        self.server = server
        self.tmp_path = tmp_path
        self.db_path = tmp_path / "rlm.duckdb"
        self.blob_root = tmp_path / "blobs"
        self.lifecycle_log = tmp_path / "lifecycle.jsonl"

    def last_episode_id(self) -> str:
        con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            return str(con.execute(
                "SELECT episode_id FROM episodes ORDER BY started_at DESC LIMIT 1"
            ).fetchone()[0])
        finally:
            con.close()

    def tamper_root_request_blob(self, episode_id: str) -> Path:
        """Flip one byte INSIDE the stored request. Appending would not do:
        the blob container slices its streams by declared length, so trailing
        junk is never hashed and the tamper would go unnoticed -- which is
        itself worth knowing about the container."""
        blobs = sorted((self.blob_root / episode_id).glob("*.root_request_ref.blob"))
        assert blobs, "no root_request_ref blob to tamper with"
        target = blobs[0]
        data = bytearray(target.read_bytes())
        data[-1] = ord("Z") if data[-1] != ord("Z") else ord("Y")
        target.write_bytes(bytes(data))
        return target

    def delete_lifecycle_log(self) -> None:
        self.lifecycle_log.unlink(missing_ok=True)

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def mock_episode_env(minimal_cfg_dict: dict, tmp_path: Path, bootstrap_dir: Path):
    server = FakeRootServer(minimal_cfg_dict,
                             script=["```repl\nfinal_answer('42')\n```"])
    raw = _episode_cfg_dict(minimal_cfg_dict, tmp_path=tmp_path, root_port=server.port)
    config_file = _write_cfg(tmp_path / "config.yaml", raw)
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({
        "task_id": "cli-fixture", "text": "What is the answer?",
        "category": "default", "context": "a tiny context", "answer": "42",
    }), encoding="utf-8")
    env = _MockEpisodeEnv(config_file=config_file, task_file=task_file,
                           server=server, tmp_path=tmp_path)
    yield env
    env.close()
