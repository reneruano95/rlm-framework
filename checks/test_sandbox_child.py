"""Child-protocol behaviour, exercised through a real spawned sandbox."""
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

REPO_ROOT = Path(__file__).resolve().parents[1]


async def test_variables_persist_across_cells(session):
    await session.exec_cell("x = 41")
    out = await session.exec_cell("print(x + 1)")
    assert out.stdout.strip() == "42"


async def test_top_level_await_works(session):
    out = await session.exec_cell(
        "r = await llm_query('hello')\nprint(r)")
    assert "MOCK" in out.stdout


async def test_traceback_is_captured_and_interpreter_survives(session):
    out = await session.exec_cell("1/0")
    assert "ZeroDivisionError" in out.traceback
    assert "rlm" not in out.traceback.lower(), "scaffold frames must be scrubbed"
    out2 = await session.exec_cell("print('alive')")
    assert out2.stdout.strip() == "alive"


async def test_last_expression_repr_is_captured(session):
    out = await session.exec_cell("2 + 3")
    assert out.repr_.strip() == "5"


async def test_reserved_names_are_reinjected_every_cell(session):
    """D24: a cell that rebinds the NAME `llm_query` must not carry that rebind
    into the next cell. That is the ordinary case and all re-injection covers --
    reaching `_RESERVED` itself defeats it, by design (spec v0.2.3 §5 C1)."""
    await session.exec_cell("llm_query = lambda *a, **k: 'HIJACKED'")
    out = await session.exec_cell("print(type(llm_query).__name__)")
    assert "HIJACKED" not in out.stdout
    assert out.stdout.strip() in {"function", "method", "coroutine"}


async def test_denied_event_loop_does_not_poison_the_next_cell(session):
    """D25: the half-built loop's __del__ used to leak host paths into turn N+1."""
    await session.exec_cell("import asyncio\nasyncio.new_event_loop()")
    out = await session.exec_cell("print('clean')")
    assert "proactor_events" not in out.stderr
    assert "AppData" not in out.stderr
    assert out.stdout.strip() == "clean"


async def test_policy_error_qualname_does_not_leak_scaffold_structure(session):
    out = await session.exec_cell("import asyncio\nasyncio.run(main())")
    assert "<locals>" not in (out.traceback + out.stderr)


async def test_egress_is_blocked_but_the_bridge_still_works(session):
    out = await session.exec_cell(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=5)\n"
        "    print('REACHED')\n"
        "except Exception as e:\n"
        "    print('BLOCKED', type(e).__name__)\n")
    assert "REACHED" not in out.stdout
    r = await session.exec_cell("print(await llm_query('still works'))")
    assert "MOCK" in r.stdout


async def test_main_module_route_to_the_reserved_names_is_closed(session):
    """Documents ONE closed route, not a sealed namespace (spec v0.2.3 §5 C1).

    `sys.modules['__main__']._RESERVED['llm_query'] = ...` used to make the NEXT
    cell's llm_query return 'HIJACKED'; re-injection did not help, because it
    re-reads the very dict the cell had just rewritten. Other routes to the same
    dict remain reachable BY DESIGN -- see child.py's ENFORCEMENT LAYERING and
    `test_hijacked_llm_query_cannot_alter_scaffold_side_control`, which asserts
    the guarantee that actually holds.
    """
    probe = await session.exec_cell(
        "import sys\n"
        "m = sys.modules['__main__']\n"
        "print([n for n in ('_RESERVED', 'USER_NS', 'BRIDGE', '_PROTO_W')\n"
        "       if hasattr(m, n)])\n")
    assert probe.stdout.strip() == "[]", "the child module is reachable as __main__"

    await session.exec_cell(
        "import sys\n"
        "try:\n"
        "    sys.modules['__main__']._RESERVED['llm_query'] = lambda *a, **k: 'HIJACKED'\n"
        "except Exception as e:\n"
        "    print('refused', type(e).__name__)\n")
    out = await session.exec_cell("print(await llm_query('x'))")
    assert "HIJACKED" not in out.stdout
    assert out.stdout.strip() == "MOCK:x"


async def test_injected_globals_route_to_the_reserved_names_is_closed(session):
    """The second closed route. `final_answer.__globals__['_RESERVED']
    ['final_answer'] = ...` used to swallow the submission with the scaffold
    recording nothing. The injected namespace is now two names wide -- which
    narrows this route and closes nothing else; `BRIDGE` is deliberately still
    in it, and pivots off it are documented as reachable in child.py."""
    probe = await session.exec_cell(
        "g = final_answer.__globals__\n"
        "print(sorted(k for k in g if k not in ('__builtins__', '__name__')))\n")
    assert probe.stdout.strip() == "['BRIDGE', 'LOOP']", probe.stdout

    await session.exec_cell(
        "try:\n"
        "    final_answer.__globals__['_RESERVED']['final_answer'] = lambda v: None\n"
        "except Exception as e:\n"
        "    print('refused', type(e).__name__)\n"
        "try:\n"
        "    llm_query.__globals__['_RESERVED']['llm_query'] = lambda *a, **k: 'X'\n"
        "except Exception as e:\n"
        "    print('refused', type(e).__name__)\n")
    await session.exec_cell("final_answer('real answer')")
    out = await session.exec_cell("print(await llm_query('y'))")
    assert out.stdout.strip() == "MOCK:y"
    assert session.final_answers == ["real answer"]


async def test_event_loop_construction_is_denied_before_any_loop_exists(session):
    """D25(c): shadowing the `asyncio` package alone is one layer thin --
    `asyncio.events.new_event_loop()` reaches the constructor directly, and the
    denial then fires INSIDE ProactorEventLoop.__init__, which is exactly the
    half-built object whose __del__ contaminates the next cell.

    Scope: the shadows, not the interpreter. `importlib.reload(asyncio.events)`
    restores the real constructors and is not defended against (spec v0.2.3).
    """
    out = await session.exec_cell(
        "import asyncio, asyncio.events as ev\n"
        "cands = [('asyncio.new_event_loop', asyncio.new_event_loop),\n"
        "         ('events.new_event_loop', ev.new_event_loop),\n"
        "         ('events.set_event_loop', ev.set_event_loop),\n"
        "         ('ProactorEventLoop', asyncio.windows_events.ProactorEventLoop),\n"
        "         ('policy.new_event_loop',\n"
        "          asyncio.get_event_loop_policy().new_event_loop)]\n"
        "for name, fn in cands:\n"
        "    try:\n"
        "        fn()\n"
        "        print('CONSTRUCTED', name)\n"
        "    except Exception as e:\n"
        "        early = 'is not available in this episode' in str(e)\n"
        "        print('denied', name, 'BEFORE-CONSTRUCTION' if early else 'TOO-LATE')\n")
    # 'TOO-LATE' means the denial came from the audit hook firing INSIDE
    # ProactorEventLoop.__init__ (on its AF_INET self-pipe) -- i.e. a loop object
    # was already half-built, which is precisely the D25(c) defect.
    assert "CONSTRUCTED" not in out.stdout
    assert "TOO-LATE" not in out.stdout, out.stdout
    assert out.stdout.count("BEFORE-CONSTRUCTION") == 5, out.stdout
    clean = await session.exec_cell("print('clean')")
    assert clean.stderr == ""
    assert clean.stdout.strip() == "clean"


async def test_traceback_never_carries_a_host_path(session, cfg):
    """Keeping stdlib frames (they are useful to the model) must not mean
    keeping absolute host paths: those reach the root's context and DuckDB."""
    out = await session.exec_cell("import json\njson.loads('{bad')")
    assert "JSONDecodeError" in out.traceback
    assert "<stdlib>/json/decoder.py" in out.traceback, out.traceback

    blob = (out.traceback + out.stderr).lower()
    forbidden = [
        str(os.path.dirname(cfg.scaffold.sandbox.interpreter)).lower(),
        str(REPO_ROOT).lower(),
        str(cfg.scaffold.sandbox.bootstrap_dir).lower(),
        "appdata", ":\\", ":/",
    ]
    for needle in forbidden:
        assert needle not in blob, f"{needle!r} leaked into the observation"


async def test_protocol_fd_desync_is_survivable_and_not_trivially_reachable(session):
    """The fds are moved out of the low range so an ordinary `os.write(4, ...)`
    can no longer land on the protocol -- the accident, and the cheap guess, are
    both gone. A deliberate write to fd 101 still desyncs; that is classified by
    the manager (see test_sandbox_manager.py), not prevented here."""
    out = await session.exec_cell(
        "import os\n"
        "hit = []\n"
        "for fd in range(3, 32):\n"
        "    try:\n"
        "        os.write(fd, b'garbage\\n')\n"
        "        hit.append(fd)\n"
        "    except OSError:\n"
        "        pass\n"
        "print('wrote to', hit)\n")
    assert "wrote to" in out.stdout
    alive = await session.exec_cell("print('still here')")
    assert alive.stdout.strip() == "still here"


async def test_llm_query_sends_chunk_and_question_as_separate_fields(session):
    """§4's layout is enforced by the scaffold, not hoped for: the model hands
    over two fields and C4 composes `[prefix][chunk][question]` itself. The
    single-string form stays valid (`chunk=None`)."""
    seen: list[dict] = []

    async def handler(payload):
        seen.append(payload)
        return "ANSWERED"

    session.on_llm_query(handler)
    out = await session.exec_cell(
        "print(await llm_query('Q?', chunk='CHUNK'))")
    assert out.stdout.strip() == "ANSWERED"
    await session.exec_cell("await llm_query('just a question')")

    assert seen == [
        {"question": "Q?", "chunk": "CHUNK", "role": "leaf"},
        {"question": "just a question", "chunk": None, "role": "leaf"},
    ]


async def test_llm_query_takes_chunk_by_keyword_only(session):
    """`llm_query(question, *, chunk=None, role="leaf")`. Keyword-only keeps the
    paper harness's positional call site (`llm_query(prompt)`) working while
    making a two-positional call a loud TypeError rather than a silently
    mis-assigned role."""
    out = await session.exec_cell("await llm_query('Q?', 'CHUNK')")
    assert "TypeError" in out.traceback


async def test_gather_fanout_exits_cleanly(session):
    """D9: this exact shape used to exit 0xC0000008 under AppContainer."""
    out = await session.exec_cell(
        "import asyncio\n"
        "rs = await asyncio.gather(*[llm_query(f'q{i}') for i in range(8)])\n"
        "print(len(rs))")
    assert out.stdout.strip() == "8"
    code = await session.close()
    assert code == 0
