#!/usr/bin/env python3
"""Score one prime-agent spike run directory (plan section 5, `score.py`).

    python score.py --run-dir <dir> --task <task_id> [--repo <repo-root>]
                    [--session-dir <dir>] [--session-jsonl <file>] [--arm P1]
    python score.py --dump-shape --session-jsonl <file>        # inspect a JSONL
    python score.py --dump-shape --session-dir <dir>           # inspect all of them

What it does
------------
1. The answer comes from ``<run-dir>/answer.txt`` when that file exists and is
   non-empty (plan section 4: the model is told to write exactly one
   ``FINAL: <answer>`` line there).  Otherwise it comes from the prime-agent
   session JSONL: the **first** assistant message carrying a line matching
   ``^FINAL:`` (plan section 5, v2 -- the autonomous continuation nudge can
   produce a second, later answer).
2. Either way the session is also scanned, when one can be located, so that
   ``final_changed`` can be reported: True when some *later* assistant message
   carries a ``FINAL:`` line whose value differs from the first one.
3. ``<repo>/bench/tasks/<task_id>.json`` supplies ``checker`` and ``answer``;
   the verdict is ``rlm.measure.checkers.check(checker, got, want)``, imported
   by putting ``<repo>/src`` on ``sys.path`` (no repo venv needed -- that module
   imports only ``re`` and ``rlm.errors``).
4. ``<run-dir>/score.json`` is written with
   ``{task, got, want, checker, pass, final_changed, source}`` plus diagnostic
   extras (see EXTRA KEYS below), and a one-line summary is printed.

Exit codes: 0 = the run was scored (pass **or** fail); 2 = it could not be
scored (bad repo, unknown task, unreadable run dir).  A failed run is not an
error, so batch loops need no ``|| true``.

JSONL shapes this handles
-------------------------
The exact nesting of prime-agent v0.8.1's session JSONL is not pinned down, so
the walkers are structural rather than positional.  Concretely:

* **File**: one JSON object per line.  Line 1 is normally a session header
  ``{"type":"session", ...}``.  Unparseable lines are counted (``bad_lines``)
  and skipped, never fatal.
* **Entry envelopes**: an entry is ``{"type": ..., ...}`` where ``type`` is
  ``"message"``, ``"compaction"``, ``"custom"``, ``"child_usage_attributed"``,
  and so on.  The message itself may sit at the top level of the entry
  (``{"type":"message","role":"assistant","content":...}``) or under any key --
  ``message``, ``entry``, ``data``, ``payload``, ``item``, ``value``,
  ``record`` -- at any depth up to 12.  The walker does not care which: it
  recurses and treats **any dict with a string ``role`` and one of
  ``content`` / ``text`` / ``parts``** as a message, and stops descending there.
* **Entry types skipped by default** when hunting for ``FINAL:``:
  ``compaction``, ``custom``, ``child_usage_attributed``, ``session``,
  ``summary`` -- a compaction summary that quotes an earlier ``FINAL:`` line
  must not be mistaken for the model saying it.  ``--include-all-types`` turns
  the filter off.  Entries with no recognisable type are always scanned.
* **Message content**: a plain string; or a list whose items are strings or
  blocks.  Text is taken from blocks of type ``text`` / ``output_text`` /
  ``response_text`` / ``input_text``, and from any block that has a ``text``
  field and no ``type`` at all.  Blocks of type ``thinking`` / ``reasoning`` /
  ``redacted_thinking`` and the fields ``reasoning`` / ``reasoning_content`` /
  ``thinking`` are **excluded** by default (a plan inside the model's thinking
  is not its reply); ``--include-thinking`` includes them.  A top-level
  ``text``/``parts`` field is read the same way.
* **Tool calls** (recorded as a diagnostic, not used for scoring): Anthropic
  style blocks ``{"type":"tool_use","name":...,"input":...}``; OpenAI style
  ``message.tool_calls[] = {"function":{"name":...,"arguments":"<json str>"}}``;
  bare ``{"type":"tool_call"|"function_call","name":...,"arguments":...}``
  blocks; and a ``tool_calls`` list wherever it is nested inside the message.

If a real run turns out to nest something differently, run ``--dump-shape`` on
its JSONL: it prints every distinct key path in the file with value types, a
count and a truncated sample, which is what the adaptation should be based on.

EXTRA KEYS written into score.json besides the seven required ones:
``run_dir``, ``arm``, ``run``, ``answer_file``, ``session_jsonl``,
``session_match`` (how the session was chosen), ``session_weak_match``,
``final_values`` (every FINAL value seen, in order),
``final_count``, ``assistant_messages``,
``tool_calls``, ``bad_lines``, ``repo``, ``scored_at``, ``notes``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# repo / checker import


def default_repo() -> Path:
    """The repo root: the first ancestor of this file that has bench/tasks and
    src/rlm.  Falls back to the absolute path the plan names."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "bench" / "tasks").is_dir() and (parent / "src" / "rlm").is_dir():
            return parent
    return Path("D:/PROJECTS/rlm-halo-framework")


def die(msg: str):
    print("score.py: error: " + msg, file=sys.stderr)
    raise SystemExit(2)


def load_checker(repo: Path):
    """Import rlm.measure.checkers.check from <repo>/src."""
    src = repo / "src"
    if not (src / "rlm" / "measure" / "checkers.py").is_file():
        die("no rlm/measure/checkers.py under %s -- pass --repo" % src)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from rlm.measure.checkers import check
    except Exception as exc:
        die("could not import rlm.measure.checkers from %s: %r" % (src, exc))
    return check


# --------------------------------------------------------------------------- #
# FINAL: extraction

# Leading markdown/quote framing is forgiven; the value is whatever follows the
# colon on that line.  Case-sensitive by default (the plan writes ^FINAL:); a
# case-insensitive second pass runs only if the strict pass finds nothing.
_FINAL_RE = re.compile(r"^\s*(?:[*_`>#\-]\s*)*FINAL\s*:\s*(.*)$")
_FINAL_RE_CI = re.compile(_FINAL_RE.pattern, re.I)
_TRIM = " \t\r\n*_`"


def final_values(text: str, ci: bool = False) -> list:
    """Every FINAL: value on its own line, in order, framing trimmed."""
    rx = _FINAL_RE_CI if ci else _FINAL_RE
    out = []
    for line in (text or "").splitlines():
        m = rx.match(line)
        if m:
            out.append(m.group(1).strip().strip(_TRIM).strip())
    return out


# --------------------------------------------------------------------------- #
# resilient JSONL walkers

MESSAGE_HINTS = ("content", "text", "parts")
SKIP_ENTRY_TYPES = {"compaction", "custom", "child_usage_attributed",
                    "session", "summary"}
TEXT_BLOCK_TYPES = {"text", "output_text", "response_text", "input_text"}
THINKING_BLOCK_TYPES = {"thinking", "reasoning", "redacted_thinking",
                        "reasoning_content"}
THINKING_FIELDS = ("thinking", "reasoning", "reasoning_content")
ASSISTANT_ROLES = ("assistant", "model")
USER_ROLES = ("user", "human")


def read_jsonl(path: Path):
    """Return (entries, bad_line_count).  Never raises on malformed JSON."""
    entries = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            entries.append(obj if isinstance(obj, dict) else {"_value": obj})
    return entries, bad


def _is_message(node) -> bool:
    return (isinstance(node, dict)
            and isinstance(node.get("role"), str)
            and any(k in node for k in MESSAGE_HINTS))


def find_messages(node, depth: int = 0, out=None) -> list:
    """Every message-shaped dict reachable from `node`, in document order.

    A message is any dict with a string `role` and one of content/text/parts.
    Recursion stops at a message (its content blocks are not messages)."""
    if out is None:
        out = []
    if depth > 12:
        return out
    if _is_message(node):
        out.append(node)
        return out
    if isinstance(node, dict):
        for value in node.values():
            find_messages(value, depth + 1, out)
    elif isinstance(node, list):
        for value in node:
            find_messages(value, depth + 1, out)
    return out


def message_text(msg: dict, include_thinking: bool = False) -> str:
    """All visible text of one message, blocks joined by newlines."""
    chunks = []

    def add_content(content):
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type")
                    if btype in THINKING_BLOCK_TYPES:
                        if include_thinking:
                            for field in ("text",) + THINKING_FIELDS:
                                if isinstance(block.get(field), str):
                                    chunks.append(block[field])
                                    break
                        continue
                    if btype in TEXT_BLOCK_TYPES or btype is None:
                        if isinstance(block.get("text"), str):
                            chunks.append(block["text"])
                        elif isinstance(block.get("content"), str):
                            chunks.append(block["content"])
        elif isinstance(content, dict):
            add_content(content.get("content", content.get("text")))

    add_content(msg.get("content"))
    for field in ("text", "parts"):
        if field in msg:
            add_content(msg[field])
    if include_thinking:
        for field in THINKING_FIELDS:
            if isinstance(msg.get(field), str):
                chunks.append(msg[field])
    return "\n".join(c for c in chunks if c)


def message_tool_calls(msg: dict) -> list:
    """(name, arguments-as-text) for every tool call in one message."""
    calls = []

    def arg_text(value) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            return repr(value)

    def from_openai(items):
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else item
            name = fn.get("name") or item.get("name")
            if isinstance(name, str):
                calls.append((name, arg_text(fn.get("arguments", fn.get("input")))))

    def scan(node, depth: int = 0):
        if depth > 12:
            return
        if isinstance(node, dict):
            if (node.get("type") in ("tool_use", "tool_call", "function_call")
                    and isinstance(node.get("name"), str)):
                calls.append((node["name"],
                              arg_text(node.get("input", node.get("arguments")))))
            if "tool_calls" in node:
                from_openai(node["tool_calls"])
            for key, value in node.items():
                if key != "tool_calls":
                    scan(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                scan(value, depth + 1)

    scan(msg)
    seen = set()
    unique = []
    for call in calls:            # the same call seen through two paths, once
        if call not in seen:
            seen.add(call)
            unique.append(call)
    return unique


def assistant_messages(entries: list, include_all_types: bool) -> list:
    """Assistant messages in document order, honouring the entry-type filter."""
    out = []
    for entry in entries:
        etype = entry.get("type")
        if (not include_all_types and isinstance(etype, str)
                and etype.lower() in SKIP_ENTRY_TYPES):
            continue
        for msg in find_messages(entry):
            if str(msg.get("role", "")).lower() in ASSISTANT_ROLES:
                out.append(msg)
    return out


def user_texts(entries: list) -> list:
    out = []
    for entry in entries:
        for msg in find_messages(entry):
            if str(msg.get("role", "")).lower() in USER_ROLES:
                out.append(message_text(msg, include_thinking=False))
    return out


# --------------------------------------------------------------------------- #
# locating the session JSONL for a run


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def candidate_jsonls(run_dir, session_dir) -> list:
    """JSONL candidates, most specific source first, de-duplicated."""
    found = []

    def add(paths):
        for p in sorted(paths):
            if p.is_file() and p not in found:
                found.append(p)

    if session_dir:
        add(Path(session_dir).rglob("*.jsonl"))
    if run_dir:
        run_dir = Path(run_dir)
        add(run_dir.glob("*.jsonl"))
        for sub in ("sessions", "session", "runs"):
            if (run_dir / sub).is_dir():
                add((run_dir / sub).rglob("*.jsonl"))
        # The plan's layout: --session-dir /home/spike/runs/A/sessions is shared
        # by every run of a phase, while the run dir is
        # /home/spike/work/A/<task>/run<i>.  Walk up a few levels looking for a
        # sibling sessions/ (or session-artifacts/) directory.
        for up in list(run_dir.parents)[:4]:
            for sub in ("sessions", "session-artifacts"):
                if (up / sub).is_dir():
                    add((up / sub).rglob("*.jsonl"))
    return found


def _run_window(run_dir):
    """(start, end) mtimes bracketing the run, from the files section 4 writes."""
    if not run_dir:
        return None
    starts = [f for f in ("metrics.pre", "prompt.txt") if (run_dir / f).is_file()]
    ends = [f for f in ("metrics.post", "answer.txt", "wall.txt", "exit.txt",
                        "stdout.txt") if (run_dir / f).is_file()]
    if not starts or not ends:
        return None
    t0 = min((run_dir / f).stat().st_mtime for f in starts)
    t1 = max((run_dir / f).stat().st_mtime for f in ends)
    return (t0 - 120.0, t1 + 120.0)


def _cwd_probe(run_dir) -> str:
    """`agg-03/run1` -- the tail of the run dir, which survives the tar copy out
    of WSL (the session records /home/spike/work/..., the host sees D:\\spike\\out)."""
    if not run_dir:
        return ""
    parent = run_dir.parent.name
    return ("%s/%s" % (parent, run_dir.name)) if parent else run_dir.name


def _header_mentions(path: Path, probe: str) -> bool:
    """Does the session header (line 1, where `cwd` lives) name this run dir?"""
    if not probe:
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
    except Exception:
        return False
    return probe in head.replace("\\\\", "/").replace("\\", "/")


def pick_session(run_dir, session_dir, explicit):
    """Choose the session JSONL belonging to this run.

    Signals, strongest first: (1) --session-jsonl; (2) the session header names
    the run dir (prime-agent records the `--cwd` it was launched with, and the
    plan gives every run its own dir); (3) the run's [metrics.pre ..
    metrics.post] mtime window contains the session; (4) a user message matches
    the run's prompt.txt -- necessary but NOT sufficient on its own, because the
    three runs of one task are launched with byte-identical prompts.

    Returns (path, reason).  A reason that is not `explicit`, `cwd`, or one of
    the unique `...-window` forms is a WEAK match: main() then refuses to let
    that session's FINAL: history speak for the run (see `weak_match`)."""
    if explicit:
        explicit = Path(explicit)
        if not explicit.is_file():
            die("--session-jsonl %s does not exist" % explicit)
        return explicit, "explicit"
    cands = candidate_jsonls(run_dir, session_dir)
    if not cands:
        return None, "none-found"

    probe = _cwd_probe(run_dir)
    cwd_hits = [p for p in cands if _header_mentions(p, probe)]
    if len(cwd_hits) == 1:
        return cwd_hits[0], "cwd"
    if cwd_hits:
        cands = cwd_hits

    window = _run_window(run_dir)
    inside = ([p for p in cands if window[0] <= p.stat().st_mtime <= window[1]]
              if window else [])
    tag = "cwd+window" if cwd_hits else "window"
    if len(inside) == 1:
        return inside[0], tag
    pool = inside or cands

    prompt = ""
    if run_dir and (run_dir / "prompt.txt").is_file():
        prompt = _norm((run_dir / "prompt.txt").read_text(
            encoding="utf-8", errors="replace"))
    if prompt:
        head = prompt[:160]
        matched = []
        for path in pool:
            entries, _ = read_jsonl(path)
            for text in user_texts(entries):
                text = _norm(text)
                if text and (head in text or text[:160] in prompt):
                    matched.append(path)
                    break
        if len(matched) == 1:
            return matched[0], ("prompt+" + tag) if inside else "prompt"
        if matched:
            return (max(matched, key=lambda p: p.stat().st_mtime),
                    "prompt-ambiguous-newest-of-%d" % len(matched))
        if pool is not cands:
            pool = cands

    if len(pool) == 1:
        return pool[0], "only-candidate"
    return (max(pool, key=lambda p: p.stat().st_mtime),
            "newest-of-%d" % len(pool))


CONFIDENT_MATCHES = {"explicit", "cwd", "cwd+window", "window", "prompt",
                     "prompt+cwd+window", "prompt+window", "only-candidate"}


# --------------------------------------------------------------------------- #
# --dump-shape


def key_paths(node, prefix: str = "", acc=None, depth: int = 0) -> dict:
    """path -> {"n": count, "types": {type names}, "sample": first scalar seen}"""
    if acc is None:
        acc = {}
    if depth > 14:
        return acc
    if prefix:
        slot = acc.setdefault(prefix, {"n": 0, "types": set(), "sample": None})
        slot["n"] += 1
        slot["types"].add(type(node).__name__)
        if slot["sample"] is None and not isinstance(node, (dict, list)):
            slot["sample"] = node
    if isinstance(node, dict):
        for key, value in node.items():
            key_paths(value, ("%s.%s" % (prefix, key)) if prefix else str(key),
                      acc, depth + 1)
    elif isinstance(node, list):
        for value in node:
            key_paths(value, prefix + "[]", acc, depth + 1)
    return acc


def dump_shape(paths: list) -> None:
    for path in paths:
        entries, bad = read_jsonl(path)
        acc = {}
        types = {}
        for entry in entries:
            key = str(entry.get("type"))
            types[key] = types.get(key, 0) + 1
            key_paths(entry, "", acc)
        print("\n=== %s" % path)
        print("    %d entries, %d unparseable lines" % (len(entries), bad))
        print("    entry types: " + ", ".join(
            "%s=%d" % (k, v)
            for k, v in sorted(types.items(), key=lambda kv: -kv[1])))
        print("    %7s  %-18s path = sample" % ("count", "types"))
        for key in sorted(acc):
            slot = acc[key]
            sample = slot["sample"]
            if isinstance(sample, str):
                sample = sample.replace("\n", "\\n")
                if len(sample) > 60:
                    sample = sample[:60] + "..."
                sample = repr(sample)
            print("    %7d  %-18s %s = %s"
                  % (slot["n"], ",".join(sorted(slot["types"])), key, sample))


# --------------------------------------------------------------------------- #


def infer_arm(run_dir, given):
    """P2 (the A' strategy arm) is inferred from the path when not given."""
    if given:
        return given
    if run_dir:
        low = str(run_dir).replace("\\", "/").lower()
        for token in ("/p2", "p2/", "aprime", "a-prime", "a_prime", "strat"):
            if token in low:
                return "P2"
    return "P1"


def infer_run(run_dir):
    if not run_dir:
        return ""
    m = re.search(r"(\d+)\s*$", run_dir.name)
    return m.group(1) if m else run_dir.name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score one prime-agent spike run directory.")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--task")
    ap.add_argument("--repo", type=Path, default=default_repo())
    ap.add_argument("--session-dir", type=Path,
                    help="the --session-dir the run was launched with")
    ap.add_argument("--session-jsonl", type=Path,
                    help="the exact session JSONL for this run")
    ap.add_argument("--arm", help="P1 (base prompt) or P2 (the A' strategy arm)")
    ap.add_argument("--include-thinking", action="store_true",
                    help="scan thinking/reasoning blocks for FINAL: too")
    ap.add_argument("--include-all-types", action="store_true",
                    help="do not skip compaction/custom/child entries")
    ap.add_argument("--dump-shape", action="store_true",
                    help="print the distinct key paths of the session JSONL(s)")
    ap.add_argument("--out", type=Path,
                    help="score.json path (default <run-dir>/score.json)")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve() if args.run_dir else None

    if args.dump_shape:
        if args.session_jsonl:
            paths = [args.session_jsonl]
        elif args.session_dir:
            paths = candidate_jsonls(None, args.session_dir)
        else:
            paths = candidate_jsonls(run_dir, None)
        if not paths:
            die("--dump-shape found no .jsonl (pass --session-jsonl/--session-dir)")
        dump_shape([Path(p) for p in paths])
        return 0

    if not run_dir or not args.task:
        die("--run-dir and --task are required (or use --dump-shape)")
    if not run_dir.is_dir():
        die("--run-dir %s is not a directory" % run_dir)

    repo = args.repo.resolve()
    task_path = repo / "bench" / "tasks" / ("%s.json" % args.task)
    if not task_path.is_file():
        die("no task JSON at %s" % task_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    want = str(task.get("answer", ""))
    checker = str(task.get("checker", ""))
    if not checker:
        die("%s has no 'checker' field" % task_path)
    check = load_checker(repo)

    notes = []

    # ---- session scan (best effort; also the source of final_changed) -------
    session_path, session_match = pick_session(run_dir, args.session_dir,
                                               args.session_jsonl)
    session_finals = []
    n_assistant = 0
    n_tool_calls = 0
    bad_lines = 0
    ci_used = False
    if session_path:
        entries, bad_lines = read_jsonl(session_path)
        msgs = assistant_messages(entries, args.include_all_types)
        n_assistant = len(msgs)
        for msg in msgs:
            n_tool_calls += len(message_tool_calls(msg))
            session_finals.extend(
                final_values(message_text(msg, args.include_thinking)))
        if not session_finals:
            for msg in msgs:
                session_finals.extend(final_values(
                    message_text(msg, args.include_thinking), ci=True))
            if session_finals:
                ci_used = True
                notes.append("FINAL: matched case-insensitively")
        if bad_lines:
            notes.append("%d unparseable JSONL line(s)" % bad_lines)
    else:
        notes.append("no session JSONL located")

    weak_match = bool(session_path) and session_match not in CONFIDENT_MATCHES
    if weak_match:
        notes.append("session match is weak (%s): %s" % (session_match,
                                                         session_path.name))

    # ---- the answer --------------------------------------------------------
    answer_file = run_dir / "answer.txt"
    got = ""
    source = "none"
    if answer_file.is_file():
        raw = answer_file.read_text(encoding="utf-8", errors="replace")
        if raw.strip():
            vals = final_values(raw) or final_values(raw, ci=True)
            if vals:
                got, source = vals[0], "answer.txt"
                if len(set(vals)) > 1:
                    notes.append("answer.txt holds %d distinct FINAL values"
                                 % len(set(vals)))
            else:
                got, source = raw.strip().strip(_TRIM).strip(), "answer.txt:raw"
                notes.append("answer.txt has no FINAL: line; used its whole text")
    if got and weak_match:
        # An answer.txt is in hand; a session we are not sure belongs to this
        # run must not be allowed to report final_changed against it.
        session_finals = []
        notes.append("weakly matched session ignored for final_changed")
    if not got and session_finals:
        got = session_finals[0]
        source = "session:" + session_path.name + (":ci" if ci_used else "")
        if weak_match:
            notes.append("ANSWER TAKEN FROM A WEAKLY MATCHED SESSION -- check "
                         "it by hand or pass --session-jsonl")
    if source == "none":
        notes.append("no answer found in answer.txt or the session")

    final_changed = any(v != session_finals[0] for v in session_finals[1:])
    if source.startswith("answer.txt") and session_finals \
            and session_finals[0] != got:
        final_changed = True
        notes.append("answer.txt and the session's first FINAL: disagree")

    passed = bool(check(checker, got, want)) if got else False

    record = {
        "task": args.task,
        "got": got,
        "want": want,
        "checker": checker,
        "pass": passed,
        "final_changed": final_changed,
        "source": source,
        # diagnostics
        "run_dir": str(run_dir),
        "arm": infer_arm(run_dir, args.arm),
        "run": infer_run(run_dir),
        "answer_file": str(answer_file) if answer_file.is_file() else None,
        "session_jsonl": str(session_path) if session_path else None,
        "session_match": session_match,
        "session_weak_match": weak_match,
        "final_values": session_finals,
        "final_count": len(session_finals),
        "assistant_messages": n_assistant,
        "tool_calls": n_tool_calls,
        "bad_lines": bad_lines,
        "repo": str(repo),
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "notes": notes,
    }
    out_path = args.out.resolve() if args.out else run_dir / "score.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print("%s %s arm=%s run=%s checker=%s got=%r want=%r source=%s "
          "final_changed=%s%s"
          % ("PASS" if passed else "FAIL", args.task, record["arm"],
             record["run"] or "-", checker, got, want, source, final_changed,
             (" [" + "; ".join(notes) + "]") if notes else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
