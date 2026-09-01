import asyncio
import json

import pytest

from rlm._tests._helpers import _in_process_pair
from rlm.bridge import MAX_FRAME, FrameReader, encode_frame

# The two async tests below carry an EXPLICIT marker rather than relying on
# `asyncio_mode = "auto"`, which the repo sets and a consumer running
# `pytest --pyargs rlm` does not. A `pytest_collection_modifyitems` hook was tried
# first and does not work: pytest-asyncio decides during collection, so the marker
# arrives too late and the tests fail with "async def functions are not natively
# supported". A module-level `pytestmark` works but also marks the three sync tests
# in this file, which pytest-asyncio warns about. Measured 2026-09-01.


def test_frames_are_ascii_only_so_lone_surrogates_survive():
    """D11: ensure_ascii=False would raise UnicodeEncodeError here."""
    payload = {"text": "lone-" + chr(0xDCFF) + "-end"}
    raw = encode_frame(payload)
    raw.decode("ascii")  # must not raise
    body = json.loads(raw.split(b"\n", 1)[1])
    assert body["text"] == payload["text"]


def test_reader_reassembles_split_and_coalesced_frames():
    frames = [encode_frame({"i": i, "pad": "x" * 100}) for i in range(5)]
    blob = b"".join(frames)
    reader = FrameReader()
    got = []
    for i in range(0, len(blob), 7):  # pathological chunking
        got.extend(reader.feed(blob[i:i + 7]))
    assert [g["i"] for g in got] == [0, 1, 2, 3, 4]


def test_oversize_frame_is_refused_not_buffered():
    reader = FrameReader()
    try:
        reader.feed(f"{MAX_FRAME + 1}\n".encode())
    except ValueError as exc:
        assert "frame" in str(exc).lower()
    else:
        raise AssertionError("oversize frame must be refused")


@pytest.mark.asyncio
async def test_eight_concurrent_requests_are_matched_out_of_order():
    """The fan-out idiom the prompt registry teaches must not deadlock."""
    parent, child = _in_process_pair()

    async def handler(kind, payload):
        await asyncio.sleep(0.05 if payload["i"] % 2 else 0.01)  # reply out of order
        return {"echo": payload["i"]}

    parent.on_request(handler)
    results = await asyncio.gather(*[child.request("llm_query", {"i": i})
                                     for i in range(8)])
    assert [r["echo"] for r in results] == list(range(8))


@pytest.mark.asyncio
async def test_parent_death_fails_pending_requests_instead_of_hanging():
    parent, child = _in_process_pair()
    parent.on_request(lambda kind, payload: asyncio.sleep(30))
    task = asyncio.create_task(child.request("llm_query", {"i": 0}))
    await asyncio.sleep(0.1)
    parent.close()
    with_timeout = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    assert isinstance(with_timeout[0], Exception)
