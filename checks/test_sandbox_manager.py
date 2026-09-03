from pathlib import Path
import asyncio
import sys
import time

import pytest

from rlm.errors import SandboxError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


async def test_handshake_pid_must_match_the_spawned_pid(manager, cfg):
    async with manager.session("ep-1", cfg) as s:
        assert s.pid > 0
        out = await s.exec_cell("import os; print(os.getpid())")
        assert out.stdout.strip() == str(s.pid)  # D23


async def test_kill_terminates_via_the_job_and_reports_the_reason(manager, cfg):
    async with manager.session("ep-2", cfg) as s:
        await s.exec_cell("x = 1")
        await s.kill("wall_clock", 0xC5)
        assert s.kill_reason == "wall_clock"
        with pytest.raises(SandboxError):
            await s.exec_cell("print(x)")


async def test_setvar_injects_a_32mb_payload_through_the_bridge(manager, cfg):
    """Deferred gap: measure this through an AppContainer bridge, not a plain pipe."""
    payload = "x" * (32 * 1024 * 1024)
    async with manager.session("ep-3", cfg) as s:
        t0 = time.perf_counter()
        await s.setvar("context", payload)
        elapsed = time.perf_counter() - t0
        out = await s.exec_cell("print(len(context))")
        assert out.stdout.strip() == str(len(payload))
        assert elapsed < 10.0, f"32 MB setvar took {elapsed:.2f}s"


async def test_sandbox_cannot_read_the_repo(manager, cfg):
    """D7: config.yaml and prompts/ must be denied by default."""
    # Derived, not hardcoded: this must name THIS checkout's config.yaml, and a
    # stale absolute path would make the test pass for the wrong reason (a
    # missing file is also an OSError, so the confinement would go unchecked).
    target = Path(__file__).resolve().parents[1] / "config.yaml"
    assert target.exists(), "the confinement target must exist to be meaningful"
    async with manager.session("ep-4", cfg) as s:
        out = await s.exec_cell(
            "try:\n"
            f"    open(r'{target}').read()\n"
            "    print('READABLE')\n"
            "except OSError as e:\n"
            "    print('DENIED')\n")
        assert out.stdout.strip() == "DENIED"


async def test_bridge_desync_is_classified_not_an_unattributed_death(manager, cfg):
    """A cell CAN reach the protocol fd and corrupt the stream. §6's status
    ENUM cannot express that on its own, so C1 attributes it: the sandbox is
    still alive, therefore `bridge_desync`, not `sandbox_death`."""
    from rlm import trace

    async with manager.session("ep-desync", cfg) as s:
        with pytest.raises(SandboxError) as excinfo:
            await s.exec_cell("import os\nos.write(101, b'garbage\\n')")
        assert trace.BRIDGE_DESYNC_REASON in str(excinfo.value)
        assert s.kill_reason == trace.BRIDGE_DESYNC_REASON == "bridge_desync"


async def test_frames_before_the_handshake_are_refused():
    """D23 is about ORDER as much as identity: nothing may be dispatched in an
    episode's name before that episode has been admitted."""
    import asyncio

    from rlm.sandbox.manager import _FrameGate

    served = []

    async def serve(kind, payload):
        served.append(kind)
        return "served"

    fut = asyncio.get_running_loop().create_future()
    gate = _FrameGate(fut, serve)

    with pytest.raises(SandboxError, match="before the handshake"):
        await gate("llm_query", {"prompt": "sneak"})
    with pytest.raises(SandboxError, match="before the handshake"):
        await gate("final_answer", {"value": "sneak"})
    with pytest.raises(SandboxError, match="before the handshake"):
        await gate("env", {"op": "open", "doc_id": "d"})
    assert served == []

    assert await gate("handshake", {"pid": 1234}) is None
    assert fut.result() == 1234
    assert await gate("llm_query", {"prompt": "ok"}) == "served"

    with pytest.raises(SandboxError, match="duplicate handshake"):
        await gate("handshake", {"pid": 4321})


async def test_hijacked_llm_query_cannot_alter_scaffold_side_control(manager, cfg):
    """THE I1 TEST (spec v0.2.3 §5 C1). This is the guarantee that holds.

    The in-process namespace is not a boundary and I1 does not depend on it
    being one: cell code can hijack `llm_query` outright, and this test does
    exactly that through one of the reachable routes child.py documents. What
    the sandbox cannot touch is the concurrency cap, the budget accounting, the
    routing or the admission decision, because those run IN A DIFFERENT PROCESS.

    A hijacked stub answers the model locally and the scaffold receives no
    frames at all -- it cannot be tricked into spending budget, holding a
    semaphore slot, or recording a sub-call that never happened. Every call that
    does cross the pipe is admitted or refused on the scaffold's count.

    Unlike the two route-specific regression tests, this one stays true no
    matter which pivot the sandbox uses. If a future change makes it fail, the
    scaffold has a real I1 bug.
    """
    admitted: list[tuple[str, str]] = []
    refused: list[str] = []
    max_subcalls = 2
    semaphore = asyncio.Semaphore(1)

    async def dispatcher(payload):
        async with semaphore:                       # C4's concurrency cap
            if len(admitted) >= max_subcalls:       # C5's budget
                refused.append(payload["question"])
                raise SandboxError("max_subcalls exhausted")
            admitted.append((payload["question"], payload["role"]))
            return f"LEAF:{payload['question']}"

    async with manager.session("ep-i1", cfg) as s:
        s.on_llm_query(dispatcher)

        hijack = await s.exec_cell(
            "RESERVED = llm_query.__globals__['BRIDGE']._handler.__globals__['_RESERVED']\n"
            "REAL = RESERVED['llm_query']\n"
            "RESERVED['llm_query'] = lambda *a, **k: 'HIJACKED'\n")
        assert hijack.traceback == "", "the hijack itself must succeed"

        out = await s.exec_cell(
            "print(sorted({llm_query(f'q{i}') for i in range(50)}))")
        assert out.stdout.strip() == "['HIJACKED']", out.stdout

        # 50 sub-calls the model believes it made, and the scaffold did not move:
        # no frames, no budget spent, no slot held, no step to log.
        assert admitted == []
        assert refused == []
        assert not semaphore.locked()
        assert s.final_answers == []

        # And the budget it "escaped" was never its to spend. Restore the real
        # stub: the same parent-side gate admits exactly max_subcalls and
        # refuses the rest, on the scaffold's count.
        await s.exec_cell("RESERVED['llm_query'] = REAL")
        real = await s.exec_cell(
            "for i in range(4):\n"
            "    try:\n"
            "        print(await llm_query(f'real{i}'))\n"
            "    except Exception as e:\n"
            "        print('REFUSED', type(e).__name__)\n")

    assert [p for p, _ in admitted] == ["real0", "real1"]
    assert {role for _, role in admitted} == {"leaf"}   # routing stays the scaffold's
    assert refused == ["real2", "real3"]
    assert real.stdout.count("REFUSED") == 2
    assert "LEAF:real0" in real.stdout
    assert not semaphore.locked()


async def test_env_call_without_a_registered_handler_is_refused(manager, cfg):
    """`manager.py`'s `_serve` 'env' branch had zero coverage: `_EpisodeRun`
    always registers `on_env` before a cell runs, so no other test ever
    exercises the unregistered path. This one deliberately never calls
    `s.on_env` at all."""
    async with manager.session("ep-env-unregistered", cfg) as s:
        out = await s.exec_cell(
            "try:\n"
            "    print(await env.open('d-01'))\n"
            "except Exception as e:\n"
            "    print(type(e).__name__, str(e))\n")
    assert "no env handler registered for this episode" in out.stdout


async def test_hijacked_env_dict_entry_cannot_alter_scaffold_side_control(manager, cfg):
    """THE DICT-ITEM-SWAP VECTOR. Same guarantee as
    test_hijacked_llm_query_cannot_alter_scaffold_side_control: a hijacked
    stub answers locally and the scaffold's action counter and range checks
    never move. Reaches `_RESERVED` through the bridge-object reflection
    pivot and replaces the WHOLE dict entry (`RESERVED['env'] = ...`) -- a
    real route (`sandbox/child.py`'s module docstring lists it), but not the
    sharper one Task 11's review flagged: see
    test_hijacked_env_attribute_cannot_alter_scaffold_side_control, below,
    for the direct attribute-write vector.
    """
    served: list[dict] = []
    async def env_handler(payload):
        served.append(payload)
        if payload["op"] == "window" and payload["i"] > 1:
            raise SandboxError("window out of range")
        return "W"
    async with manager.session("ep-env", cfg) as s:
        s.on_env(env_handler)
        hijack = await s.exec_cell(
            "RESERVED = env.search.__globals__['BRIDGE']._handler.__globals__['_RESERVED']\n"
            "REAL = RESERVED['env']\n"
            "import types\n"
            "RESERVED['env'] = types.SimpleNamespace(window=lambda d, i: 'HIJACKED')\n")
        assert hijack.traceback == ""
        out = await s.exec_cell("print(sorted({env.window('d', i) for i in range(50)}))")
        assert out.stdout.strip() == "['HIJACKED']"
        assert served == []                      # 50 calls the model believes it made; none crossed
        await s.exec_cell("RESERVED['env'] = REAL")
        real = await s.exec_cell(
            "for i in range(4):\n"
            "    try:\n        print(await env.window('d', i))\n"
            "    except Exception as e:\n        print('REFUSED', type(e).__name__)\n")
    assert [p["i"] for p in served] == [0, 1, 2, 3]
    assert real.stdout.count("REFUSED") == 2 and real.stdout.count("W") == 2


async def test_hijacked_env_attribute_cannot_alter_scaffold_side_control(manager, cfg):
    """THE SHARPER VECTOR (Task 11's review; distinct code path from the
    dict-item swap above). `env` is a `types.SimpleNamespace` held BY
    REFERENCE in `_RESERVED`, and `USER_NS.update(_RESERVED)` rebinds that
    SAME object into every cell -- so an ordinary attribute write reaches it
    directly, with NO reflection pivot at all: `env.search = ...` is one
    line of ordinary syntax, because `env` is already a live name in the
    cell's namespace. `llm_query` has no equivalent surface (replacing its
    behaviour needs `__code__`/`__globals__` tricks, not attribute
    assignment), which is exactly why this is a DIFFERENT and easier attack
    surface than the dict-item swap, and why the brief's plain name-rebinding
    test (`env = 'hijacked'`) is insufficient on its own.

    There is no `REAL` object to restore here, unlike the dict-item-swap
    test: the write mutates the LIVE `SimpleNamespace` `_RESERVED['env']`
    itself, not a dict entry pointing at it -- so `env.search` stays
    hijacked for the rest of THIS session's life; nothing reachable from
    inside the sandbox can undo it (a fresh episode gets a fresh object,
    this one does not get a fresh `.search`). What the guarantee still has
    to hold: the hijack cannot reach anything scaffold-side, and the other
    two verbs on that SAME live object -- never touched by the write --
    still enforce the scaffold's own counting and range check.
    """
    served: list[dict] = []
    env_actions = 0

    async def env_handler(payload):
        nonlocal env_actions
        served.append(payload)
        if payload["op"] == "window" and payload["i"] > 1:
            raise SandboxError("window out of range")
        env_actions += 1
        return "REAL"

    async with manager.session("ep-env-attr", cfg) as s:
        s.on_env(env_handler)

        hijack = await s.exec_cell("env.search = lambda term: 'HIJACKED'\n")
        assert hijack.traceback == "", "the attribute write itself must succeed"

        out = await s.exec_cell(
            "print(sorted({env.search(f'q{i}') for i in range(50)}))")
        assert out.stdout.strip() == "['HIJACKED']"
        # 50 calls the model believes it made through `env.search`: none
        # crossed the bridge, and the scaffold's own action counter (the
        # `env_actions` surrogate here) never moved.
        assert served == []
        assert env_actions == 0

        # Proven again: there is no dict entry to restore, so the hijack is
        # permanent for this session -- `env.search` stays hijacked with no
        # action taken by the cell.
        again = await s.exec_cell("print(env.search('still-hijacked'))")
        assert again.stdout.strip() == "HIJACKED"
        assert served == []
        assert env_actions == 0

        # `.open`/`.window` were never assigned to -- only `.search` was
        # overwritten -- so the counter and the range check on THOSE two
        # verbs are still the scaffold's, on the very same live object whose
        # `.search` stays permanently hijacked.
        real = await s.exec_cell(
            "for i in range(4):\n"
            "    try:\n        print(await env.window('d', i))\n"
            "    except Exception as e:\n        print('REFUSED', type(e).__name__)\n")
    assert [p["i"] for p in served if p["op"] == "window"] == [0, 1, 2, 3]
    assert real.stdout.count("REFUSED") == 2 and real.stdout.count("REAL") == 2
    assert env_actions == 2   # only the two ACCEPTED window calls counted


async def test_episodes_do_not_share_state(manager, cfg):
    async with manager.session("ep-5", cfg) as s:
        await s.exec_cell("marker = 'first-episode'")
    async with manager.session("ep-6", cfg) as s:
        out = await s.exec_cell("print('marker' in dir())")
        assert out.stdout.strip() == "False"
