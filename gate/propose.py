"""The proposer: candidate artifacts mined from TRAIN-split traces (spec §3, step 5).

    python gate/propose.py --runs-root <train-run-dir> --out gate/artifacts/round-N.json

WHY THE SCAFFOLD PROPOSES, and not the model from its own conversation. Measured in
the spike: given a free hand inside its own session, the local root wrote 8 of 8
artifacts as `memory`, and their content was the answers to the tasks it had just been
shown. Given the same model a TRAJECTORY to read instead -- tool calls, their results,
and the outcome -- and a schema that forbids naming anything from the corpus, the job
changes from "what did I just learn" to "what would have prevented this failure".

WHAT IT READS. Only train-split episodes. Never a live conversation, never a held-out
episode, never an episode's answer key. It runs offline, between episodes, which is
`2026-08-22-long-horizon-agent-design.md` §3.4's first demand verbatim: "propose
offline from traces, apply BETWEEN episodes as versioned sha-pinned config".

WHAT IT EMITS. `prompt` or `skill` candidates only (spec D-S2), each a HarnessEntry
with the gate's fields fixed, written to one file that `gate/screens.py` then screens
and `gate/run_decision.sh` then evaluates. The proposer NEVER writes to prime-agent's
store: it produces a candidate, and only an ACCEPT puts one in front of a model.

THE FAILURE THIS IS BUILT AGAINST, from decision pc-02's agg-07 trace: the model
followed a hand-written rule exactly -- two methods, buckets reconciled, anomalies
scanned -- and still answered wrongly, because its two "independent" methods varied
the record splitter and shared one broken predicate extractor, and because it narrated
a residual bucket holding 43% of records and answered anyway. A proposal that names a
generic virtue ("be careful", "double-check") would not have prevented it. The prompt
below therefore demands the specific mechanism and the specific observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.request
from typing import Any

REPO_DEFAULT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
MAX_CONTENT = 900          # spec §3.3
TRACE_CHARS = 12_000       # per-episode slice handed to the proposer

SYSTEM = """You read the execution trace of an agent that solved (or failed) one task \
and you propose ONE reusable operating rule that would make the next attempt at a task \
of this KIND go better.

You are not answering the task. You are not summarising what happened. You are writing \
a rule for a future agent that will never see this trace.

HARD CONSTRAINTS, and a proposal that breaks any of them is discarded unread:
- Name no number, count, file path, identifier, organisation or entity that appears in \
the corpus or in the answer. A rule that mentions a specific value is a memorised \
answer wearing a rule's clothes.
- Name no configuration setting, budget, timeout, cap, route or termination rule. You \
cannot change those and asking for more of them is not a rule.
- Do not write a generic virtue. "Be careful", "double-check your work" and "think step \
by step" are worthless: the agent in this trace was already trying. A rule earns its \
place by naming the SPECIFIC MECHANISM that went wrong and the SPECIFIC OBSERVABLE that \
would have revealed it.

Reply with ONE JSON object and nothing else:
{"kind": "prompt" | "skill", "title": "<short imperative>", "content": "<the rule>"}

`kind` is "skill" only when the rule is a concrete procedure a future agent could \
follow mechanically; otherwise "prompt". Keep `content` under 900 characters."""


def _walk_episode(session: pathlib.Path) -> dict[str, Any]:
    """Trajectory summary: assistant text, ipython cells, and their results."""
    steps: list[str] = []
    cwd = ""
    for line in session.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "session":
            cwd = o.get("cwd", "")
        m = o.get("message") or {}
        role = m.get("role")
        if role == "assistant":
            for c in m.get("content") or []:
                if c.get("type") == "toolCall":
                    code = (c.get("arguments") or {}).get("code", "")
                    steps.append(f"CELL:\n{code}")
                elif c.get("type") == "text":
                    steps.append(f"SAID: {c.get('text','')}")
        elif role == "toolResult":
            d = m.get("details") or {}
            out = (d.get("result") or d.get("stdout") or d.get("stderr") or "")
            steps.append(f"RESULT[{d.get('status')}]: {out[:400]}")
    return {"cwd": cwd, "trace": "\n\n".join(steps)}


def _ask(url: str, model: str, trace: str, task_kind: str, outcome: str) -> str:
    user = (f"The task was of kind `{task_kind}`. The episode outcome was: {outcome}.\n\n"
            f"Execution trace:\n\n{trace[-TRACE_CHARS:]}")
    body = {"model": model, "temperature": 0, "max_tokens": 900, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        o = json.loads(r.read().decode())
    return (o["choices"][0]["message"].get("content") or "").strip()


def _parse(reply: str) -> dict | None:
    """First JSON object in the reply. The model is asked for exactly one; a model
    that wraps it in prose or a fence still yields a usable candidate, and one that
    yields nothing is recorded as a proposer miss rather than retried into silence."""
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "content" not in d:
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, help="a TRAIN run directory")
    ap.add_argument("--out", required=True, help="candidate set to write")
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-episodes", type=int, default=8)
    ap.add_argument("--failed-only", action="store_true",
                    help="propose only from episodes that got the answer wrong")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    split_path = pathlib.Path(args.split or repo / "bench" / "splits" / "s6lite-v0.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = {r["task_id"] for r in split["train"]}
    held_ids = {r["task_id"] for r in split["held_out"]}

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    from gate.decide import collect  # noqa: E402

    root = pathlib.Path(args.runs_root)
    cells = collect(root, repo)
    if not cells:
        print(f"no episodes under {root}", file=sys.stderr)
        return 2

    # HARD REFUSAL: the proposer may not see a held-out episode, ever.
    leaked = sorted({c.task for c in cells} & held_ids)
    if leaked:
        print(f"REFUSING: {root} contains held-out episodes {leaked}. The proposer reads "
              "TRAIN traces only -- an artifact derived from a held-out task voids the run "
              "(ARCHITECTURE.md:472).", file=sys.stderr)
        return 2
    off_split = sorted({c.task for c in cells} - train_ids)
    if off_split:
        print(f"REFUSING: {root} contains tasks that are not in the train split: {off_split}",
              file=sys.stderr)
        return 2

    chosen = [c for c in cells if (not args.failed_only or not c.passed)]
    chosen.sort(key=lambda c: (c.passed, -(c.tokens or 0)))   # failures first, then costliest
    chosen = chosen[: args.max_episodes]
    print(f"proposing from {len(chosen)} of {len(cells)} train episodes "
          f"({sum(1 for c in chosen if not c.passed)} failed)")

    entries: dict[str, dict] = {}
    misses: list[str] = []
    for cell in chosen:
        ep = root / cell.arm / cell.task / f"rep{cell.rep}"
        sessions = sorted((root / cell.arm / "sessions").glob("*.jsonl"),
                          key=lambda p: p.stat().st_mtime)
        target = None
        for s in sessions:
            try:
                head = json.loads(s.open(encoding="utf-8").readline())
            except Exception:
                continue
            if str(head.get("cwd", "")).endswith(f"{cell.task}/rep{cell.rep}"):
                target = s
        if target is None:
            misses.append(f"{cell.task}/rep{cell.rep}: no session")
            continue
        info = _walk_episode(target)
        kind = json.loads((repo / "bench" / "tasks" / f"{cell.task}.json")
                          .read_text(encoding="utf-8"))["category"]
        outcome = "answered correctly" if cell.passed else f"answered WRONGLY ({cell.answer!r})"
        try:
            reply = _ask(args.url, args.model, info["trace"], kind, outcome)
        except Exception as err:
            misses.append(f"{cell.task}/rep{cell.rep}: {err}")
            continue
        cand = _parse(reply)
        if not cand:
            misses.append(f"{cell.task}/rep{cell.rep}: unparseable reply")
            continue

        content = str(cand.get("content", "")).strip()[:MAX_CONTENT]
        ckind = cand.get("kind") if cand.get("kind") in ("prompt", "skill") else "prompt"
        cid = f"rlmh-{ckind}-{hashlib.sha256(content.encode()).hexdigest()[:8]}"
        if cid in entries:
            continue
        entries[cid] = {
            "id": cid, "kind": ckind,
            "title": str(cand.get("title", "")).strip()[:120] or "proposed rule",
            "content": content,
            "path": f"00-gate/{ckind}/{len(entries):02d}",
            "scope": "global", "reference": {}, "arguments": {},
            "metadata": {"rlmh": {"version": 1, "origin": "proposer",
                                  "proposed_from": [f"{cell.task}/rep{cell.rep}"],
                                  "episode_outcome": outcome}},
            "source": "rlmh-gate",
            "created_at": "2026-08-28T00:00:00Z", "updated_at": "2026-08-28T00:00:00Z",
            "version": 1,
        }
        print(f"  {cell.task}/rep{cell.rep} ({'pass' if cell.passed else 'FAIL'}) -> "
              f"{ckind} {cid}: {entries[cid]['title']}")

    state = {"schema": 1,
             "_rlmh_note": (f"Proposed by the local root from {len(chosen)} TRAIN episodes under "
                            f"{root}. Candidates only: nothing here has been screened or gated."),
             "entries": {"prompt": {k: v for k, v in entries.items() if v["kind"] == "prompt"},
                         "memory": {}, "skill": {k: v for k, v in entries.items() if v["kind"] == "skill"},
                         "subagent": {}},
             "refinements": []}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(state, indent=2) + "\n"
    out.write_text(body, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(body.encode()).hexdigest()
    (out.parent / (out.stem + ".sha256")).write_text(sha + "\n", encoding="utf-8", newline="\n")

    print(f"\nwrote {out}  ({len(entries)} candidates)  sha256 {sha}")
    if misses:
        print(f"proposer misses ({len(misses)}), recorded rather than retried:")
        for m in misses:
            print(f"  {m}")
    print("\nNext: python gate/screens.py " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
