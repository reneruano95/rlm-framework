#!/usr/bin/env python3
"""usage.py - the measurement parser for the prime-agent local spike (plan section 5).

Usage
-----
    python usage.py --runs-root <dir> --out <csv> [--session-dir <dir> ...] [-v]
    python usage.py --runs-root <dir> --dump-shape [--session-dir <dir> ...]

Runs on the HOST (Windows), against the tree copied out of WSL with

    wsl -d Ubuntu --cd /home/spike -- tar cf - runs work | tar xf - -C D:\\spike\\out

so `--runs-root D:\\spike\\out` sees both `work/` (the per-run working dirs that
hold metrics.pre/metrics.post/wall.txt/exit.txt/answer.txt/score.json) and
`runs/` (the session JSONLs and `session-artifacts/`).  It also works when
pointed straight at `/home/spike` inside WSL.

What counts as a run directory
------------------------------
Any directory that contains BOTH `metrics.pre` and `metrics.post`, OR contains
`wall.txt`.  Discovery does not descend into a directory once it qualifies, nor
into `sessions/`, `session-artifacts/`, `harness.pre/`, `harness.post/`,
`kernel-venv/`, `node_modules/`, `.git/`.

task / run / arm come from, in order: `meta.json` ({"task":..,"run":..,"arm":..}),
`task.txt` / `run.txt` / `arm.txt`, then the path itself
(`.../work/A/agg-03/run1` -> task=agg-03, run=1; `.../agg-03-run2` also works).
`arm` is P2 when arm.txt/meta.json say so or when any path component normalises
to one of p2 / aprime / a-prime / astrategy / strat / strategy; otherwise P1.

Matching a run to its session JSONL
-----------------------------------
prime-agent writes every print-mode session of a phase into one shared
`--session-dir`, so the run <-> session link has to be reconstructed.  Candidate
root sessions are every `*.jsonl` (plus non-blacklisted `*.json`) under the
`--session-dir` values, else under any directory named `sessions/` beneath
--runs-root, else any `*.jsonl` beneath --runs-root that is not itself a child
session.  Each candidate is scored against each run:

    1000  the run dir names the session explicitly
          (session.txt / session-id.txt / session_id.txt, or meta.json.session_id)
     100  a `cwd`-ish string inside the session ends with the run dir's last
          3 (or 2) path components - e.g. session cwd `/home/spike/work/A/agg-03/run1`
          matches host dir `D:\\spike\\out\\work\\A\\agg-03\\run1`
      60  the session's first user text starts with the run's prompt.txt text
      25  the session file's mtime falls inside [metrics.pre, metrics.post] (+/-120 s)
     0-10 mtime proximity to that window (tie-break)

Assignment is greedy and one-to-one (highest score first); only an explicit-id
match may be shared by two runs.  A run with no candidate above 0 still gets a
CSV row - every session-derived column is left empty and `-v` says why.

Child (subagent) sessions
-------------------------
Once the root session `<id>.jsonl` is known, children are every `*.jsonl` under
`<any indexed session-artifacts dir>/<id>/sub-*/` (recursively, so a depth-2
subagent under `sub-*/session-artifacts/<childid>/sub-*/` is found too).  `<id>`
is the file stem and, if the session declares one, its `sessionId`/`id` field.
`subagents` is the number of distinct child JSONLs.

JSONL shapes handled (the format is not pinned by the plan, so the walker is
shape-agnostic; `--dump-shape` prints the key paths actually present)
-------------------------------------------------------------------
Records are read one JSON value per line; a whole-file JSON array, or a JSON
object with a `messages` / `entries` / `events` / `records` list, is also
accepted.  For each record the deepest "message-like" dict is taken by
unwrapping `message` / `msg` / `data` / `entry` / `payload` / `record` / `item`
/ `body` / `turn` (up to 6 levels); the outer record stays in scope for field
lookups, so usage on the wrapper is still found.  A dict is message-like when it
has a string `role`, or a `type` naming a message kind, or a `content` together
with `usage`/`stopReason`.  All key lookups are case- and separator-insensitive
(`stopReason` == `stop_reason` == `STOP-REASON`).

  * flat        {"role":"assistant","content":"..","usage":{"input":1,"output":2,
                 "cacheRead":3},"stopReason":"end_turn"}
  * wrapped     {"type":"assistant","message":{"role":"assistant",...}}
  * evented     {"event":"message","data":{"role":"assistant",...}}
  * openai      {"role":"assistant","tool_calls":[{"id":"..","type":"function",
                 "function":{"name":"ipython","arguments":"{\\"code\\":\\"...\\"}"}}],
                 "usage":{"prompt_tokens":..,"completion_tokens":..,
                          "prompt_tokens_details":{"cached_tokens":..}}}
  * anthropic   content blocks [{"type":"tool_use","name":"ipython",
                 "input":{"code":"..."}}] with usage.input_tokens /
                 output_tokens / cache_read_input_tokens
  * ai-sdk      {"type":"tool-call","toolName":"ipython","args":{"code":"..."}}
  * standalone  a record that is not message-like but is itself a tool call
                (only then, to avoid double counting)

Token field aliases: input | inputTokens | input_tokens | prompt_tokens |
promptTokens | in; output | outputTokens | output_tokens | completion_tokens |
completionTokens | out; cacheRead | cache_read | cacheReadTokens |
cache_read_input_tokens | cachedTokens | cached_tokens (also picked up from
`prompt_tokens_details.cached_tokens`).

Column definitions (exactly the plan's list, plus cache_read_harness and
cached_tokens_metrics)
----------------------------------------------------------------------
task, run, arm            see above
pass                      score.json -> pass|fail|"" (bool, "pass"/"fail"/"ok",
                          0/1, or {"pass":..}/{"passed":..}/{"result":..}/
                          {"verdict":..}/{"score":..})
wall_s                    wall.txt (`/usr/bin/time -f '%e'`); a trailing
                          "Command exited with non-zero status" line is ignored,
                          and m:ss / h:mm:ss are converted
turns                     assistant messages, root + children
tool_calls                tool calls named "ipython", root + children
tokens_in_harness         sum of usage.input over assistant messages, root + children
tokens_out_harness        sum of usage.output    "
cache_read_harness        sum of usage.cacheRead "
prompt_tokens_metrics     metrics.post - metrics.pre for llamacpp:prompt_tokens_total
predicted_tokens_metrics  ... for llamacpp:tokens_predicted_total
cached_tokens_metrics     ... for llamacpp:prompt_tokens_cached_total
max_identical_streak      longest run of consecutive ipython calls with a
                          byte-identical `code` argument; computed per session
                          file and maxed over root + children (0 = no ipython
                          call, 1 = no repeat).  A-loop / C4.
refine_calls              ipython calls whose code contains "refine.run" or
                          "rlm.harness."  (A-refine)
continuations             non-assistant messages containing "No human input is
                          available in autonomous mode"; if that count is 0 and
                          the session carries a `continuationsUsed` field, the
                          largest such value is used instead
errors                    assistant messages with stopReason == "error"
stop_reason               the last assistant stopReason in the ROOT session
                          (falls back to the last one seen anywhere)
subagents                 number of child session JSONLs found
exit_code                 exit.txt

Prometheus parsing: `<name>{labels} <value>` lines, comments skipped, values
with the same name summed, `Nan`/`+Inf` dropped.  A missing counter (either
file) leaves the cell empty rather than reporting a false 0.

Exit status: 0 on success, 2 on bad arguments, 1 if no run directory was found.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

CSV_COLUMNS = [
    "task",
    "run",
    "arm",
    "pass",
    "wall_s",
    "turns",
    "tool_calls",
    "tokens_in_harness",
    "tokens_out_harness",
    "cache_read_harness",
    "prompt_tokens_metrics",
    "predicted_tokens_metrics",
    "cached_tokens_metrics",
    "max_identical_streak",
    "refine_calls",
    "continuations",
    "errors",
    "stop_reason",
    "subagents",
    "exit_code",
]

CONTINUATION_PHRASE = "no human input is available in autonomous mode"
REFINE_MARKERS = ("refine.run", "rlm.harness.")
IPYTHON_NAMES = {"ipython", "python", "ipythontool", "runipython", "execpython"}

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    "kernel-venv",
    ".venv",
    "venv",
    "sessions",
    "session-artifacts",
}
SKIP_DIR_PREFIXES = ("harness.",)

# files that live in a session dir but are not sessions
NON_SESSION_JSON = {
    "settings.json",
    "models.json",
    "harness_state.json",
    "score.json",
    "results.json",
    "meta.json",
    "manifest.json",
    "package.json",
    "index.json",
    "state.json",
}

WRAPPER_KEYS = ("message", "msg", "data", "entry", "payload", "record", "item", "body", "turn")

MESSAGE_TYPE_HINTS = (
    "assistant",
    "user",
    "human",
    "system",
    "tool",
    "message",
    "response",
    "agent",
    "model",
    "prompt",
    "completion",
)

TOOL_CALL_TYPES = {"tooluse", "toolcall", "functioncall", "toolinvocation", "toolrequest"}
TOOL_CALL_PARENT_KEYS = {
    "toolcalls",
    "toolcall",
    "tooluse",
    "tooluses",
    "toolinvocations",
    "functioncalls",
    "calls",
    "tools",
}

NAME_KEYS = ("name", "toolname", "tool", "functionname", "function")
ARG_KEYS = ("input", "args", "arguments", "parameters", "params", "toolinput", "argsjson", "argument")
CODE_KEYS = ("code", "source", "script", "cell", "snippet", "python", "command")

USAGE_KEYS = ("usage", "tokenusage", "tokens", "usagestats")
IN_KEYS = ("input", "inputtokens", "prompttokens", "in", "inputtokencount")
OUT_KEYS = ("output", "outputtokens", "completiontokens", "out")
CACHE_KEYS = (
    "cacheread",
    "cachereadtokens",
    "cachereadinputtokens",
    "cachedtokens",
    "cachereadtoken",
    "cached",
)
STOP_KEYS = ("stopreason", "finishreason", "stop", "endreason")

METRIC_ALIASES = {
    "prompt_tokens_metrics": (
        "llamacpp:prompt_tokens_total",
        "llamacpp_prompt_tokens_total",
        "llamacpp:n_prompt_tokens_processed_total",
    ),
    "predicted_tokens_metrics": (
        "llamacpp:tokens_predicted_total",
        "llamacpp_tokens_predicted_total",
        "llamacpp:n_tokens_predicted_total",
    ),
    "cached_tokens_metrics": (
        "llamacpp:prompt_tokens_cached_total",
        "llamacpp_prompt_tokens_cached_total",
        "llamacpp:n_prompt_tokens_cached_total",
        "llamacpp:cache_tokens_total",
    ),
}

MAX_WALK_DEPTH = 14


# --------------------------------------------------------------------------
# small generic helpers
# --------------------------------------------------------------------------


def nk(key: Any) -> str:
    """Normalise a key: lowercase, drop everything that is not a-z0-9."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def dget(d: Any, *aliases: str) -> Any:
    """Case/separator-insensitive dict lookup; aliases must already be normalised."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if nk(k) in aliases:
            return v
    return None


def as_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return int(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if m:
            try:
                return int(float(m.group(0)))
            except ValueError:
                return None
    return None


def num_str(v: float | int | None) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def read_text(path: Path, limit: int = 1_000_000) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------
# JSON record iteration
# --------------------------------------------------------------------------


def iter_records(path: Path) -> Iterator[Any]:
    """Yield JSON values from a session file.

    Handles: one JSON value per line (JSONL, the expected shape), a whole-file
    JSON array, and a whole-file object carrying a list of messages/entries/
    events/records.  Unparseable lines are skipped (counted by the caller via
    iter_records_counted).
    """
    for _lineno, obj, _bad in iter_records_counted(path):
        if not _bad:
            yield obj


def iter_records_counted(path: Path) -> Iterator[tuple[int, Any, bool]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
        except json.JSONDecodeError:
            arr = None
        if isinstance(arr, list):
            for i, obj in enumerate(arr):
                yield i + 1, obj, False
            return
    if stripped.startswith("{") and "\n" not in stripped.strip():
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict):
            for key in ("messages", "entries", "events", "records", "history", "turns"):
                lst = dget(doc, nk(key))
                if isinstance(lst, list):
                    for i, obj in enumerate(lst):
                        yield i + 1, obj, False
                    return
            yield 1, doc, False
            return
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield lineno, json.loads(line), False
        except json.JSONDecodeError:
            yield lineno, None, True


# --------------------------------------------------------------------------
# message classification
# --------------------------------------------------------------------------


def type_of(d: dict) -> str:
    for key in ("type", "kind", "event", "eventtype", "role"):
        v = dget(d, key)
        if isinstance(v, str):
            return nk(v)
    return ""


def is_message_like(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    role = dget(d, "role")
    if isinstance(role, str) and role:
        return True
    t = type_of(d)
    if t and any(h in t for h in MESSAGE_TYPE_HINTS):
        return True
    if dget(d, "content") is not None and (
        dget(d, *USAGE_KEYS) is not None or dget(d, *STOP_KEYS) is not None
    ):
        return True
    return False


def unwrap(rec: Any) -> tuple[dict | None, list[dict]]:
    """Return (deepest message-like dict, all dicts in the unwrap chain).

    The chain is kept so field lookups can fall back to the wrapper (some
    formats put `usage` on the envelope and `content` on the inner message).
    """
    chain: list[dict] = []
    cur = rec
    best: dict | None = None
    for _ in range(6):
        if not isinstance(cur, dict):
            break
        chain.append(cur)
        if is_message_like(cur):
            best = cur
        nxt = None
        for wk in WRAPPER_KEYS:
            v = dget(cur, wk)
            if isinstance(v, dict):
                nxt = v
                break
        if nxt is None:
            break
        cur = nxt
    return best, chain


def role_of(msg: dict, chain: Iterable[dict]) -> str:
    for node in [msg, *chain]:
        r = dget(node, "role")
        if isinstance(r, str) and r:
            return nk(r)
    for node in [msg, *chain]:
        t = type_of(node)
        if not t:
            continue
        if t.startswith(("assistant", "agent", "model", "aimessage", "completion", "response")):
            return "assistant"
        if t.startswith(("user", "human", "prompt", "input")):
            return "user"
        if t.startswith("system"):
            return "system"
        if t.startswith("tool"):
            return "tool"
    return ""


# --------------------------------------------------------------------------
# usage extraction
# --------------------------------------------------------------------------


def find_usage(nodes: Iterable[dict]) -> dict | None:
    nodes = list(nodes)
    for node in nodes:
        u = dget(node, *USAGE_KEYS)
        if isinstance(u, dict):
            return u
    # bounded recursive search
    for node in nodes:
        found = _search_usage(node, 0)
        if found is not None:
            return found
    return None


def _search_usage(obj: Any, depth: int) -> dict | None:
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if nk(k) in USAGE_KEYS and isinstance(v, dict):
                return v
        for v in obj.values():
            r = _search_usage(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:20]:
            r = _search_usage(v, depth + 1)
            if r is not None:
                return r
    return None


def usage_triple(nodes: list[dict]) -> tuple[int, int, int]:
    """(input, output, cacheRead) for one assistant message."""
    u = find_usage(nodes)
    src: list[dict] = []
    if isinstance(u, dict):
        src.append(u)
    src.extend(nodes)

    def pick(keys: tuple[str, ...]) -> int:
        for s in src:
            v = as_int(dget(s, *keys))
            if v is not None:
                return v
        return 0

    tin = pick(IN_KEYS)
    tout = pick(OUT_KEYS)
    tcache = pick(CACHE_KEYS)
    if tcache == 0 and isinstance(u, dict):
        details = dget(u, "prompttokensdetails", "inputtokensdetails", "tokendetails")
        if isinstance(details, dict):
            tcache = as_int(dget(details, *CACHE_KEYS)) or 0
    return tin, tout, tcache


def stop_reason_of(nodes: list[dict]) -> str:
    for node in nodes:
        v = dget(node, *STOP_KEYS)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            inner = dget(v, "type", "reason")
            if isinstance(inner, str) and inner:
                return inner
    return ""


# --------------------------------------------------------------------------
# tool call extraction
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    name: str
    code: str

    @property
    def is_ipython(self) -> bool:
        return nk(self.name) in IPYTHON_NAMES


def _args_to_code(name: str, args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        text = args.strip()
        if text.startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text
            return _args_to_code(name, parsed)
        return text
    if isinstance(args, dict):
        v = dget(args, *CODE_KEYS)
        if isinstance(v, str):
            return v
        if v is not None:
            return json.dumps(v, sort_keys=True, ensure_ascii=False)
        # no recognised code key: canonical dump is still a stable fingerprint
        try:
            return json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(args)
    return str(args)


def _call_from_dict(d: dict) -> ToolCall | None:
    fn = dget(d, "function")
    if isinstance(fn, dict):
        name = dget(fn, "name") or dget(d, "name") or ""
        args = dget(fn, *ARG_KEYS)
        return ToolCall(str(name), _args_to_code(str(name), args))
    name = None
    for key in NAME_KEYS:
        v = dget(d, key)
        if isinstance(v, str) and v:
            name = v
            break
    if name is None:
        return None
    args = dget(d, *ARG_KEYS)
    if args is None:
        args = dget(d, *CODE_KEYS)
        if args is not None:
            args = {"code": args}
    return ToolCall(str(name), _args_to_code(str(name), args))


def collect_tool_calls(obj: Any, parent_key: str = "", depth: int = 0) -> list[ToolCall]:
    """Collect tool CALLS (never tool results) in document order."""
    out: list[ToolCall] = []
    if depth > MAX_WALK_DEPTH:
        return out
    if isinstance(obj, dict):
        t = type_of(obj)
        if t and ("result" in t or "response" in t or "output" in t or "return" in t):
            return out  # a tool result: do not descend, it echoes the code
        looks_like_call = False
        if t in TOOL_CALL_TYPES or t.replace("-", "") in TOOL_CALL_TYPES:
            looks_like_call = True
        elif t == "function" and isinstance(dget(obj, "function"), dict):
            looks_like_call = True
        elif parent_key in TOOL_CALL_PARENT_KEYS:
            looks_like_call = True
        elif t == "" and isinstance(dget(obj, "function"), dict) and dget(obj, "id") is not None:
            looks_like_call = True
        if looks_like_call:
            call = _call_from_dict(obj)
            if call is not None and call.name:
                out.append(call)
                return out
        for k, v in obj.items():
            out.extend(collect_tool_calls(v, nk(k), depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_tool_calls(v, parent_key, depth + 1))
    return out


CONTINUATION_FIELD_KEYS = ("continuationsused", "continuations", "continuationcount", "numcontinuations")


def deep_find_int(obj: Any, keys: tuple[str, ...], depth: int = 0) -> int | None:
    """Largest integer found under any of `keys` (normalised), bounded depth."""
    best: int | None = None
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if nk(k) in keys:
                iv = as_int(v)
                if iv is not None and (best is None or iv > best):
                    best = iv
            sub = deep_find_int(v, keys, depth + 1)
            if sub is not None and (best is None or sub > best):
                best = sub
    elif isinstance(obj, list):
        for v in obj[:50]:
            sub = deep_find_int(v, keys, depth + 1)
            if sub is not None and (best is None or sub > best):
                best = sub
    return best


def collect_text(obj: Any, depth: int = 0, acc: list[str] | None = None) -> str:
    if acc is None:
        acc = []
    if depth > MAX_WALK_DEPTH or len(acc) > 400:
        return " ".join(acc)
    if isinstance(obj, str):
        acc.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_text(v, depth + 1, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_text(v, depth + 1, acc)
    return " ".join(acc)


# --------------------------------------------------------------------------
# session parsing
# --------------------------------------------------------------------------


@dataclass
class SessionStats:
    path: Path
    session_id: str = ""
    declared_id: str = ""
    assistant_msgs: int = 0
    ipython_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    refine_calls: int = 0
    continuations: int = 0
    continuations_field: int = 0
    errors: int = 0
    last_stop_reason: str = ""
    max_identical_streak: int = 0
    parse_errors: int = 0
    records: int = 0
    cwd_hints: list[str] = field(default_factory=list)
    first_user_text: str = ""
    mtime: float = 0.0


PATHISH = re.compile(r"(?:/[\w.@+ -]+){2,}")


def parse_session(path: Path) -> SessionStats:
    st = SessionStats(path=path, session_id=path.stem, mtime=mtime(path) or 0.0)
    codes: list[str] = []
    hints: set[str] = set()
    for _lineno, rec, bad in iter_records_counted(path):
        if bad:
            st.parse_errors += 1
            continue
        st.records += 1
        if isinstance(rec, dict):
            if not st.declared_id:
                for key in ("sessionid", "id", "session"):
                    v = dget(rec, key)
                    if isinstance(v, str) and len(v) >= 6 and "/" not in v:
                        st.declared_id = v
                        break
            cu = deep_find_int(rec, CONTINUATION_FIELD_KEYS)
            if cu is not None and cu > st.continuations_field:
                st.continuations_field = cu

        msg, chain = unwrap(rec)
        nodes = chain if msg is None else [msg, *[c for c in chain if c is not msg]]
        role = role_of(msg, chain) if msg is not None else ""

        # collect path hints for run<->session matching (cheap, bounded)
        if len(hints) < 60:
            for s in re.findall(r'"((?:[A-Za-z]:)?[^"\\]{4,300})"', json.dumps(rec, ensure_ascii=False))[:40]:
                if "/work/" in s or "/runs/" in s or PATHISH.fullmatch(s):
                    hints.add(s.replace("\\", "/"))
                    if len(hints) >= 60:
                        break

        if msg is not None and role == "assistant":
            st.assistant_msgs += 1
            tin, tout, tcache = usage_triple(nodes)
            st.tokens_in += tin
            st.tokens_out += tout
            st.cache_read += tcache
            sr = stop_reason_of(nodes)
            if sr:
                st.last_stop_reason = sr
                if nk(sr) == "error":
                    st.errors += 1
            for call in collect_tool_calls(msg):
                if call.is_ipython:
                    st.ipython_calls += 1
                    codes.append(call.code)
                    if any(m in call.code for m in REFINE_MARKERS):
                        st.refine_calls += 1
        elif msg is not None:
            if role in ("user", "system", "host", "tool", ""):
                text = collect_text(msg).lower()
                if CONTINUATION_PHRASE in " ".join(text.split()):
                    st.continuations += 1
                if role == "user" and not st.first_user_text:
                    st.first_user_text = " ".join(collect_text(msg).split())[:400]
        else:
            # not a message: it may still be a standalone tool-call event
            for call in collect_tool_calls(rec):
                if call.is_ipython:
                    st.ipython_calls += 1
                    codes.append(call.code)
                    if any(m in call.code for m in REFINE_MARKERS):
                        st.refine_calls += 1
            text = " ".join(collect_text(rec).lower().split())
            if CONTINUATION_PHRASE in text:
                st.continuations += 1

    st.cwd_hints = sorted(hints)
    st.max_identical_streak = longest_identical_streak(codes)
    return st


def longest_identical_streak(codes: list[str]) -> int:
    best = 0
    run = 0
    prev: str | None = None
    for c in codes:
        if prev is not None and c == prev:
            run += 1
        else:
            run = 1
        prev = c
        best = max(best, run)
    return best


# --------------------------------------------------------------------------
# Prometheus metrics
# --------------------------------------------------------------------------

PROM_LINE = re.compile(r"^\s*([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[-+]?Inf)\s*$")


def parse_prom(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    text = read_text(path, limit=4_000_000)
    if text is None:
        return out
    for line in text.splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        m = PROM_LINE.match(line)
        if not m:
            continue
        name, _labels, val = m.groups()
        try:
            fval = float(val)
        except ValueError:
            continue
        if math.isnan(fval) or math.isinf(fval):
            continue
        out[name] = out.get(name, 0.0) + fval
    return out


def metric_delta(pre: dict[str, float], post: dict[str, float], aliases: tuple[str, ...]) -> float | None:
    for name in aliases:
        if name in pre and name in post:
            return post[name] - pre[name]
    for name in aliases:  # counter appeared only after the run
        if name in post and not pre:
            return post[name]
    return None


# --------------------------------------------------------------------------
# run directories
# --------------------------------------------------------------------------


@dataclass
class RunDir:
    path: Path
    task: str = ""
    run: str = ""
    arm: str = "P1"
    wall_s: str = ""
    exit_code: str = ""
    passed: str = ""
    t_pre: float | None = None
    t_post: float | None = None
    explicit_session: str = ""
    prompt_head: str = ""
    metrics: dict[str, str] = field(default_factory=dict)


RUN_SUFFIX = re.compile(r"^(?:run|r)[-_]?(\d+)$", re.I)
RUN_EMBEDDED = re.compile(r"^(?P<task>.+?)[-_](?:run|r)[-_]?(?P<run>\d+)$", re.I)
P2_TOKENS = {"p2", "aprime", "astrategy", "strat", "strategy", "arm2", "a2"}
P1_TOKENS = {"p1", "arm1", "base"}


def dir_is_run(p: Path) -> bool:
    try:
        if (p / "metrics.pre").is_file() and (p / "metrics.post").is_file():
            return True
        return (p / "wall.txt").is_file()
    except OSError:
        return False


def find_run_dirs(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIR_NAMES and not d.startswith(SKIP_DIR_PREFIXES) and not d.startswith("sub-")
        ]
        if here != root and dir_is_run(here):
            found.append(here)
            dirnames[:] = []
    if not found and dir_is_run(root):
        found.append(root)
    return sorted(found)


def parse_wall(text: str | None) -> str:
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("command exited"):
            continue
        m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", line)
        if m:
            return num_str(int(m.group(1)) * 60 + float(m.group(2)))
        m = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", line)
        if m:
            return num_str(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)))
        m = re.search(r"-?\d+(?:\.\d+)?", line)
        if m:
            return num_str(float(m.group(0)))
    return ""


def parse_score(path: Path) -> str:
    text = read_text(path, limit=200_000)
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        low = text.lower()
        if "pass" in low:
            return "pass"
        if "fail" in low:
            return "fail"
        return ""
    return score_from_obj(doc)


def score_from_obj(doc: Any) -> str:
    if isinstance(doc, bool):
        return "pass" if doc else "fail"
    if isinstance(doc, (int, float)):
        return "pass" if doc else "fail"
    if isinstance(doc, str):
        low = doc.strip().lower()
        if low in ("pass", "passed", "true", "ok", "1", "correct"):
            return "pass"
        if low in ("fail", "failed", "false", "0", "incorrect"):
            return "fail"
        return ""
    if isinstance(doc, dict):
        for key in ("pass", "passed", "ispass", "result", "verdict", "outcome", "score", "correct"):
            v = dget(doc, key)
            if v is None:
                continue
            r = score_from_obj(v)
            if r:
                return r
    return ""


def load_run(path: Path, root: Path) -> RunDir:
    rd = RunDir(path=path)
    meta: dict = {}
    mp = path / "meta.json"
    if mp.is_file():
        try:
            loaded = json.loads(mp.read_text(encoding="utf-8", errors="replace"))
            if isinstance(loaded, dict):
                meta = loaded
        except (OSError, json.JSONDecodeError):
            meta = {}

    def sidecar(name: str) -> str:
        t = read_text(path / name, limit=4096)
        return t.strip() if t else ""

    task = str(dget(meta, "task", "taskid") or "").strip() or sidecar("task.txt")
    run = str(dget(meta, "run", "runid", "runindex") or "").strip() or sidecar("run.txt")
    arm = str(dget(meta, "arm", "prompt", "promptarm") or "").strip() or sidecar("arm.txt")

    name = path.name
    m = RUN_SUFFIX.match(name)
    if not run:
        if m:
            run = m.group(1)
        else:
            m2 = RUN_EMBEDDED.match(name)
            if m2:
                run = m2.group("run")
    if not task:
        if m:
            task = path.parent.name
        else:
            m2 = RUN_EMBEDDED.match(name)
            task = m2.group("task") if m2 else name

    if not arm:
        parts = {nk(p) for p in path.relative_to(root).parts} if _is_relative(path, root) else {nk(p) for p in path.parts}
        if parts & P2_TOKENS:
            arm = "P2"
        elif parts & P1_TOKENS:
            arm = "P1"
        else:
            arm = "P1"
    arm = arm.strip().upper()
    if arm in ("A'", "APRIME", "A-PRIME", "2"):
        arm = "P2"
    elif arm in ("A", "1"):
        arm = "P1"

    rd.task, rd.run, rd.arm = task, run, arm
    rd.wall_s = parse_wall(read_text(path / "wall.txt", limit=65536))
    ec = read_text(path / "exit.txt", limit=4096)
    if ec:
        v = as_int(ec.strip())
        rd.exit_code = "" if v is None else str(v)
    sp = path / "score.json"
    if sp.is_file():
        rd.passed = parse_score(sp)

    rd.t_pre = mtime(path / "metrics.pre")
    rd.t_post = mtime(path / "metrics.post")
    if rd.t_post is None:
        rd.t_post = mtime(path / "wall.txt")
    if rd.t_pre is None:
        rd.t_pre = rd.t_post

    for key in ("session.txt", "session-id.txt", "session_id.txt", "sessionid.txt"):
        v = sidecar(key)
        if v:
            rd.explicit_session = v.splitlines()[0].strip()
            break
    if not rd.explicit_session:
        v = dget(meta, "sessionid", "session")
        if isinstance(v, str):
            rd.explicit_session = v.strip()

    ptxt = read_text(path / "prompt.txt", limit=65536)
    if ptxt:
        rd.prompt_head = " ".join(ptxt.split())[:200]

    pre = parse_prom(path / "metrics.pre")
    post = parse_prom(path / "metrics.post")
    for col, aliases in METRIC_ALIASES.items():
        rd.metrics[col] = num_str(metric_delta(pre, post, aliases))
    return rd


def _is_relative(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# session discovery + matching
# --------------------------------------------------------------------------


def is_child_path(p: Path) -> bool:
    return any(part.startswith("sub-") for part in p.parts)


def find_session_files(root: Path, session_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve()).lower()
        if key in seen:
            return
        if p.name.lower() in NON_SESSION_JSON:
            return
        if is_child_path(p):
            return
        seen.add(key)
        out.append(p)

    for sd in session_dirs:
        if not sd.is_dir():
            continue
        for p in sorted(sd.rglob("*.jsonl")):
            add(p)
        for p in sorted(sd.rglob("*.json")):
            add(p)

    if not out:
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__"}]
            if here.name == "sessions":
                for fn in sorted(filenames):
                    if fn.lower().endswith((".jsonl", ".json")):
                        add(here / fn)

    if not out:
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            dirnames[:] = [
                d for d in dirnames if d not in {".git", "node_modules", "__pycache__"} and not d.startswith("sub-")
            ]
            if "session-artifacts" in here.parts:
                continue
            for fn in sorted(filenames):
                if fn.lower().endswith(".jsonl"):
                    add(here / fn)
    return out


def find_artifact_dirs(root: Path, session_dirs: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        if p.is_dir():
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                dirs.append(p)

    for sd in session_dirs:
        add(sd.parent / "session-artifacts")
    for dirpath, dirnames, _filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__"}]
        if here.name == "session-artifacts":
            add(here)
            dirnames[:] = []
    return dirs


def find_child_sessions(ids: Iterable[str], artifact_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for sid in {i for i in ids if i}:
        for ad in artifact_dirs:
            base = ad / sid
            if not base.is_dir():
                continue
            for sub in sorted(base.glob("sub-*")):
                if not sub.is_dir():
                    continue
                for p in sorted(sub.rglob("*.jsonl")):
                    key = str(p.resolve()).lower()
                    if key not in seen and p.name.lower() not in NON_SESSION_JSON:
                        seen.add(key)
                        out.append(p)
    return out


def path_suffixes(p: Path) -> list[str]:
    parts = [x for x in p.parts if not re.fullmatch(r"[A-Za-z]:\\?", x)]
    parts = [x.replace("\\", "") for x in parts]
    sufs = []
    for n in (4, 3, 2):
        if len(parts) >= n:
            sufs.append("/".join(parts[-n:]).lower())
    return sufs


def match_score(rd: RunDir, st: SessionStats) -> float:
    score = 0.0
    if rd.explicit_session and rd.explicit_session in (st.session_id, st.declared_id):
        return 1000.0
    sufs = path_suffixes(rd.path)
    for hint in st.cwd_hints:
        h = hint.rstrip("/").lower()
        for i, suf in enumerate(sufs):
            if h.endswith(suf):
                score += 100.0 - i * 5
                break
        if score:
            break
    if rd.prompt_head and st.first_user_text:
        a = rd.prompt_head[:120].lower()
        b = st.first_user_text[:200].lower()
        if a and (a in b or b.startswith(a[:60])):
            score += 60.0
    if rd.t_pre and rd.t_post:
        lo, hi = min(rd.t_pre, rd.t_post) - 120, max(rd.t_pre, rd.t_post) + 120
        if lo <= st.mtime <= hi:
            score += 25.0
        else:
            gap = min(abs(st.mtime - lo), abs(st.mtime - hi))
            score += max(0.0, 10.0 - gap / 60.0)
    return score


def assign_sessions(runs: list[RunDir], sessions: list[SessionStats], verbose: bool) -> dict[Path, SessionStats]:
    pairs: list[tuple[float, int, int]] = []
    for i, rd in enumerate(runs):
        for j, st in enumerate(sessions):
            s = match_score(rd, st)
            if s > 0:
                pairs.append((s, i, j))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_runs: set[int] = set()
    used_sessions: set[int] = set()
    result: dict[Path, SessionStats] = {}
    for s, i, j in pairs:
        if i in used_runs:
            continue
        if j in used_sessions and s < 1000.0:
            continue
        used_runs.add(i)
        if s < 1000.0:
            used_sessions.add(j)
        result[runs[i].path] = sessions[j]
        if verbose:
            print(
                f"[match] {runs[i].path} <- {sessions[j].path.name} (score {s:.0f})",
                file=sys.stderr,
            )
    if verbose:
        for i, rd in enumerate(runs):
            if i not in used_runs:
                print(f"[match] {rd.path} <- NO SESSION (no candidate scored > 0)", file=sys.stderr)
    return result


# --------------------------------------------------------------------------
# shape dump
# --------------------------------------------------------------------------


def shape_paths(obj: Any, prefix: str, out: dict[str, dict[str, Any]], depth: int = 0) -> None:
    if depth > MAX_WALK_DEPTH:
        return
    if isinstance(obj, dict):
        if not obj:
            _bump(out, prefix or ".", "empty-object", "{}")
        for k, v in obj.items():
            shape_paths(v, f"{prefix}.{k}", out, depth + 1)
    elif isinstance(obj, list):
        if not obj:
            _bump(out, f"{prefix}[]", "empty-array", "[]")
        for v in obj[:50]:
            shape_paths(v, f"{prefix}[]", out, depth + 1)
    else:
        sample = obj if isinstance(obj, str) else json.dumps(obj)
        _bump(out, prefix or ".", type(obj).__name__, str(sample))


def _bump(out: dict[str, dict[str, Any]], path: str, typename: str, sample: str) -> None:
    rec = out.setdefault(path, {"count": 0, "types": set(), "sample": ""})
    rec["count"] += 1
    rec["types"].add(typename)
    if not rec["sample"]:
        rec["sample"] = " ".join(sample.split())[:70]


def dump_shape(files: list[Path]) -> None:
    if not files:
        print("no session files found", file=sys.stderr)
        return
    total: dict[str, dict[str, Any]] = {}
    for f in files:
        per: dict[str, dict[str, Any]] = {}
        n = 0
        bad = 0
        for _lineno, rec, is_bad in iter_records_counted(f):
            if is_bad:
                bad += 1
                continue
            n += 1
            shape_paths(rec, "", per, 0)
            shape_paths(rec, "", total, 0)
        print(f"\n=== {f}  ({n} records, {bad} unparseable) ===")
        for path in sorted(per):
            rec = per[path]
            types = ",".join(sorted(rec["types"]))
            print(f"  {rec['count']:>7}  {path:<52} {types:<22} {rec['sample']}")
    print(f"\n=== ALL FILES ({len(files)}) ===")
    for path in sorted(total):
        rec = total[path]
        types = ",".join(sorted(rec["types"]))
        print(f"  {rec['count']:>7}  {path:<52} {types:<22} {rec['sample']}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_row(rd: RunDir, root_st: SessionStats | None, children: list[SessionStats]) -> dict[str, str]:
    row = {c: "" for c in CSV_COLUMNS}
    row["task"] = rd.task
    row["run"] = rd.run
    row["arm"] = rd.arm
    row["pass"] = rd.passed
    row["wall_s"] = rd.wall_s
    row["exit_code"] = rd.exit_code
    for col in METRIC_ALIASES:
        row[col] = rd.metrics.get(col, "")

    all_st = ([root_st] if root_st else []) + children
    if not all_st:
        row["subagents"] = "0" if root_st is not None else ""
        return row

    row["turns"] = str(sum(s.assistant_msgs for s in all_st))
    row["tool_calls"] = str(sum(s.ipython_calls for s in all_st))
    row["tokens_in_harness"] = str(sum(s.tokens_in for s in all_st))
    row["tokens_out_harness"] = str(sum(s.tokens_out for s in all_st))
    row["cache_read_harness"] = str(sum(s.cache_read for s in all_st))
    row["max_identical_streak"] = str(max(s.max_identical_streak for s in all_st))
    row["refine_calls"] = str(sum(s.refine_calls for s in all_st))
    cont = sum(s.continuations for s in all_st)
    if cont == 0:
        cont = max((s.continuations_field for s in all_st), default=0)
    row["continuations"] = str(cont)
    row["errors"] = str(sum(s.errors for s in all_st))
    if root_st is not None and root_st.last_stop_reason:
        row["stop_reason"] = root_st.last_stop_reason
    else:
        for s in reversed(all_st):
            if s.last_stop_reason:
                row["stop_reason"] = s.last_stop_reason
                break
    row["subagents"] = str(len(children))
    return row


def sort_key(row: dict[str, str]) -> tuple:
    run = row.get("run", "")
    try:
        rnum = (0, int(run))
    except (TypeError, ValueError):
        rnum = (1, 0)
    return (row.get("task", ""), row.get("arm", ""), rnum, run)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Parse prime-agent spike run dirs into one CSV row per run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs-root", required=True, help="root of the copied-out tree (holds work/ and runs/)")
    ap.add_argument("--out", help="output CSV path (required unless --dump-shape)")
    ap.add_argument(
        "--session-dir",
        action="append",
        default=[],
        help="explicit session dir (repeatable); default: sessions/ dirs under --runs-root",
    )
    ap.add_argument("--dump-shape", action="store_true", help="print distinct JSON key paths and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="explain run<->session matching on stderr")
    args = ap.parse_args(argv)

    root = Path(args.runs_root).expanduser()
    if not root.is_dir():
        print(f"error: --runs-root is not a directory: {root}", file=sys.stderr)
        return 2
    session_dirs = [Path(s).expanduser() for s in args.session_dir]
    for sd in session_dirs:
        if not sd.is_dir():
            print(f"warning: --session-dir does not exist: {sd}", file=sys.stderr)

    session_files = find_session_files(root, session_dirs)
    if args.dump_shape:
        artifact_dirs = find_artifact_dirs(root, session_dirs)
        kids: list[Path] = []
        for f in session_files:
            kids.extend(find_child_sessions([f.stem], artifact_dirs))
        dump_shape(session_files + [k for k in kids if k not in session_files])
        return 0

    if not args.out:
        print("error: --out is required (or use --dump-shape)", file=sys.stderr)
        return 2

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print(f"error: no run directory under {root} (need metrics.pre+metrics.post or wall.txt)", file=sys.stderr)
        return 1
    runs = [load_run(p, root) for p in run_dirs]
    if args.verbose:
        print(f"[scan] {len(runs)} run dirs, {len(session_files)} candidate root sessions", file=sys.stderr)

    sessions = [parse_session(p) for p in session_files]
    for st in sessions:
        if st.parse_errors and args.verbose:
            print(f"[warn] {st.path.name}: {st.parse_errors} unparseable line(s)", file=sys.stderr)

    assignment = assign_sessions(runs, sessions, args.verbose)
    artifact_dirs = find_artifact_dirs(root, session_dirs)
    if args.verbose:
        print(f"[scan] {len(artifact_dirs)} session-artifacts dir(s)", file=sys.stderr)

    rows: list[dict[str, str]] = []
    for rd in runs:
        root_st = assignment.get(rd.path)
        children: list[SessionStats] = []
        if root_st is not None:
            ids = {root_st.session_id, root_st.declared_id}
            for cp in find_child_sessions(ids, artifact_dirs):
                children.append(parse_session(cp))
            if args.verbose and children:
                print(f"[match] {rd.path.name}: {len(children)} child session(s)", file=sys.stderr)
        rows.append(build_row(rd, root_st, children))

    rows.sort(key=sort_key)
    out_path = Path(args.out).expanduser()
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
