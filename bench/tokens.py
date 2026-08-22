"""Token counting for the benchmark builder: one offline proxy, one real count.

COPIED FROM `milestones/s1/make_fixtures.py`, NOT IMPORTED, and that is the
house style rather than a shortcut. `src/rlm/projection.py:28` documents the
same move for the same reason: `milestones/` is an evidence archive, not a
library, and live code reaching into it makes an archive un-archivable. The
repo already carries three copies of `approx_tokens` on purpose -- s1's,
`rlm/dispatcher.py`'s, and `upstream/make_fixtures.py`'s, that last one made
expressly so the reproducer could stand alone.

`approx_tokens` is one executable line. `leaf_counter` is stdlib `urllib`. The
duplication is smaller than the coupling it removes.

WHAT DOES *NOT* GET COPIED: the syllable pools. Comparing the benchmark's name
space against a snapshot of a fixture's would assert a copy against itself and
see nothing. That check imports the real modules, at build time, in
`bench/build.py` -- see `assert_name_space_disjoint` there.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable


def approx_tokens(text: str) -> int:
    """The offline token proxy: ~4 chars per token.

    Stated, not measured -- deliberately the same approximation
    `rlm.dispatcher.MockDispatcher.count_tokens` documents, so the one place in
    this repo that guesses at token counts guesses the same way every time. It
    is monotonic in prefix length and deterministic, which is all a binary
    search over prefixes requires. It is NOT what a control document is cut
    with: that uses the leaf server's own tokenizer via `leaf_counter`.
    """
    return (len(text) + 3) // 4


def leaf_counter(port: int = 8081, timeout: float = 300.0) -> Callable[[str], int]:
    """`/tokenize` on the leaf server -- the only token count that means
    anything for chunk sizing (spec §5 C2: chunk size is measured in
    target-leaf tokens).

    A zero count for non-empty input is a fault, never a legitimate answer:
    `/tokenize` fails silently (probe recipes, §serverapi), so a corpus built
    against a dead endpoint would otherwise be sized as if every document were
    empty.
    """
    url = f"http://127.0.0.1:{port}/tokenize"

    def count(text: str) -> int:
        if not text:
            return 0
        req = urllib.request.Request(
            url, data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            tokens = json.loads(resp.read().decode("utf-8")).get("tokens", [])
        if not tokens:
            raise RuntimeError("/tokenize returned 0 tokens for non-empty input")
        return len(tokens)

    return count
