"""Shared fixtures. `minimal_cfg_dict` / `valid_cfg` are read from the repo's
real config.yaml so the test suite and the shipped config never drift apart.
"""
from __future__ import annotations

import asyncio
import copy
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from rlm.config import Config

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
    """The mock dispatcher: answers every sub-call without touching a server."""
    return f"MOCK:{payload['prompt']}"


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


def _naive_tokenize(text: str) -> list[int]:
    """Whitespace tokenizer -- good enough to make pre-flight admission
    (token count vs slot capacity) exercisable without a real model."""
    return list(range(len(text.split()))) if text else []


class MockLlamaServer:
    """Stub of the leaf-shaped llama-server endpoints C4 talks to."""

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.max_concurrent = 0
        self.restart_count = 0
        self.last_request_disconnected = False
        self._concurrent = 0
        self._fail_remaining = 0
        self._lock = threading.Lock()
        self._dispatchers: list[Any] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: A002 -- stdlib signature
                pass

            def do_GET(self):
                if self.path == "/props":
                    _send_json(self, 200, {
                        "model_path": "mock-leaf.gguf",
                        "total_slots": 8,
                        "default_generation_settings": {"n_ctx": 4096},
                        "build_info": "mock",
                        "chat_template": "mock-template",
                        "media_marker": "mock-nonce",
                        "is_sleeping": False,
                    })
                else:
                    _send_json(self, 404, {"error": "not found"})

            def do_POST(self):
                if self.path == "/tokenize":
                    body = _read_json_body(self)
                    text = body.get("content", "") or ""
                    _send_json(self, 200, {"tokens": _naive_tokenize(text)})
                elif self.path == "/completion":
                    self._handle_completion()
                else:
                    _send_json(self, 404, {"error": "not found"})

            def _handle_completion(self):
                body = _read_json_body(self)
                prompt = body.get("prompt", "")
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
                    if prompt == "slow":
                        self._stream_slow()
                    else:
                        self._stream_normal(prompt)
                finally:
                    with outer._lock:
                        outer._concurrent -= 1

            def _stream_normal(self, prompt):
                content = f"echo:{prompt}"
                _write_sse_event(self, {"content": content, "stop": False,
                                         "id_slot": -1, "tokens_predicted": 1,
                                         "tokens_evaluated": 4})
                _write_sse_event(self, {
                    "content": "", "stop": True, "id_slot": 0, "stop_type": "eos",
                    "tokens_predicted": max(1, len(content.split())),
                    "tokens_evaluated": 4, "truncated": False, "tokens_cached": 4,
                    "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 5.0,
                                "predicted_n": max(1, len(content.split())),
                                "predicted_ms": 10.0},
                })

            def _stream_slow(self):
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
                        "content": "", "stop": True, "id_slot": 0, "stop_type": "limit",
                        "tokens_predicted": 300, "tokens_evaluated": 4, "tokens_cached": 4,
                        "timings": {"cache_n": 0, "prompt_n": 4, "prompt_ms": 1.0,
                                    "predicted_n": 300, "predicted_ms": 600.0},
                    })
                except OSError:
                    pass
                finally:
                    watcher.join(timeout=2.0)

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def fail_times(self, n: int) -> None:
        self._fail_remaining = n

    def kill(self) -> None:
        """Simulate server death: stop listening, so new connections fail
        with a connection-refused error, same as a crashed process."""
        self.shutdown()

    def dispatcher(self, *, slot_capacity_tokens: int = 32768, parallel: int = 4,
                    max_predict: int = 64, max_attempts: int = 3):
        from rlm.config import Retries
        from rlm.dispatcher import DispatchTarget, LLMDispatcher, ServerClient

        client = ServerClient(self.base_url, timeout=5.0)
        target = DispatchTarget(client=client, max_predict=max_predict,
                                 slot_capacity_tokens=slot_capacity_tokens)
        # Fast backoff -- retry TIMING isn't what these tests assert on, and
        # the real 1s/4s backoff (scaffold.retries in config.yaml) would
        # make the retry test take ~5s for no additional coverage.
        retries = Retries(max_attempts=max_attempts, backoff_s=[0.01, 0.02],
                           per_call_timeout_s=5.0)
        dispatcher = LLMDispatcher(targets={"leaf": target, "root": target},
                                    parallel=parallel, retries=retries)
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


def _render_chatml(messages: list[dict], enable_thinking: bool) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


class FakeRootServer:
    """Stub of the root-shaped llama-server endpoints rootclient talks to."""

    def __init__(self, base_cfg_dict: dict) -> None:
        self._base_cfg_dict = base_cfg_dict
        self.last_completion_prompt: str | None = None
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
                    body = _read_json_body(self)
                    text = body.get("content", "") or ""
                    _send_json(self, 200, {"tokens": _naive_tokenize(text)})
                elif self.path == "/completion":
                    self._handle_completion()
                else:
                    _send_json(self, 404, {"error": "not found"})

            def _handle_completion(self):
                body = _read_json_body(self)
                outer.last_completion_prompt = body.get("prompt", "")
                content = "```repl\nfinal_answer(1)\n```"
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

    def conversation(self, *, system: str | None = None, enable_thinking: bool | None = None):
        from rlm.dispatcher import ServerClient
        from rlm.rootclient import RootConversation

        raw = copy.deepcopy(self._base_cfg_dict)
        raw["servers"]["root"]["port"] = self.port
        if enable_thinking is not None:
            raw["scaffold"]["root"]["enable_thinking"] = enable_thinking
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
