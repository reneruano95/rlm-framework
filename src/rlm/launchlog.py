"""D27: what a llama-server was ACTUALLY launched with, read from its own log.

`/props` cannot report KV cache types or flash-attn state -- measured, see
`parse_launch_log`'s docstring -- so §4's "assert cache types" is answered from
the server's own `-lv 4` stderr instead. That makes the LAUNCHER part of the
scaffold contract.

WHY THIS IS ITS OWN MODULE. It is pure text parsing: a file read, three regexes
and dict arithmetic. It contacts no server and imports no client, so it belongs
UNDER §5's dependency-rule lint (`tests/test_import_rules.py` ISOLATED) rather
than inside `rlm/cli.py`, where nothing would check that it stays that way.
`rlm/cli.py` re-exports both public names, so `from rlm.cli import
parse_launch_log` keeps working.

Extracted from `rlm/cli.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


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
    (2026-08-19, `milestones/s2/DFLASH2.md`). A speculative launch with `-md` builds TWO
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
