"""Child-protocol behaviour, exercised through a real spawned sandbox."""
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


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
    """D24: rebinding llm_query must not let model code intercept its own plumbing."""
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


async def test_gather_fanout_exits_cleanly(session):
    """D9: this exact shape used to exit 0xC0000008 under AppContainer."""
    out = await session.exec_cell(
        "import asyncio\n"
        "rs = await asyncio.gather(*[llm_query(f'q{i}') for i in range(8)])\n"
        "print(len(rs))")
    assert out.stdout.strip() == "8"
    code = await session.close()
    assert code == 0
