"""Helpers shared by the shipped contract suite.

These live in a module rather than in `conftest.py` because a consumer's pytest
resolves `conftest` by rootdir, not by package, so `from conftest import ...`
breaks the moment the suite is collected from anywhere but this repo. An explicit
`from rlm._tests._helpers import ...` works from any working directory.
"""
import asyncio
import os


def _in_process_pair():
    """Wire two BridgeEndpoint instances together over two os.pipe() pairs,
    so the bridge's framing/correlation logic is testable without spawning
    a real sandbox process. Must be called from inside a running event loop.
    """
    from rlm.bridge import BridgeEndpoint

    loop = asyncio.get_running_loop()
    p2c_r, p2c_w = os.pipe()  # parent writes, child reads
    c2p_r, c2p_w = os.pipe()  # child writes, parent reads
    parent = BridgeEndpoint(c2p_r, p2c_w, loop=loop, tag="parent")
    child = BridgeEndpoint(p2c_r, c2p_w, loop=loop, tag="child")
    return parent, child
