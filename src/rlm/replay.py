"""Replay: rebuild an episode's state from the trace store ALONE.

This is I4 in code. `_rederive_messages` reconstructs, from recorded
`(action, observation_view)` pairs and nothing else, the exact message array
the root was sent at every turn; `rlm replay` fails when the rebuilt array
differs from the one recorded. That property is what S3 passed on -- 12/12,
lifecycle log deleted, zero `rlm/` changes -- and it is the reason a published
number can be checked by someone who was not there.

WHY THIS IS ITS OWN MODULE, and why it matters more here than anywhere else.
The whole value of replay is that it CANNOT consult a live system: a
re-derivation able to ask a server what the prompt was would be checking the
server against itself. Inside `rlm/cli.py` nothing maintained that -- the
composition root imports HTTP clients and dispatchers, so only convention stood
between this code and a shortcut. Here §5's dependency-rule lint
(`tests/test_import_rules.py` ISOLATED) enforces it: no `httpx`, no
`rlm.dispatcher`, no `rlm.rootclient`, checked on every run.

Reaching this required splitting `rlm/roottext.py` out of `rlm/rootclient.py`
first: `history_message` and `extract_cell` are needed here and were sitting
behind a `ServerClient`.

What stays in `rlm/cli.py`: `_verify_online` and `cmd_replay`. The `--online`
mode contacts a server by design, and mixing it in here is exactly what the
lint above exists to prevent.

Extracted from `rlm/cli.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import duckdb

from rlm.config import Config, PromptRegistry
from rlm.errors import ActionType, Actor, ConfigError, RlmError, StepStatus
from rlm.episode import compose_user_message, no_cell_observation
from rlm.roottext import extract_cell, history_message
from rlm.trace import unpack_blob


def _read_episode(cfg: Config, episode_id: str) -> tuple[dict, list[dict]]:
    db_path = Path(cfg.trace.db_path)
    if not db_path.exists():
        raise ConfigError(f"no trace store at {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cur = con.execute("SELECT * FROM episodes WHERE episode_id = ?", [episode_id])
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            raise ConfigError(f"no episode {episode_id} in {db_path}")
        episode = dict(zip(cols, row))
        if isinstance(episode.get("config_snapshot"), str):
            episode["config_snapshot"] = json.loads(episode["config_snapshot"])
        cur = con.execute(
            "SELECT * FROM steps WHERE episode_id = ? ORDER BY step_idx", [episode_id])
        cols = [d[0] for d in cur.description]
        steps = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()
    for step in steps:
        for key in ("call_id", "episode_id"):
            if isinstance(step.get(key), uuid.UUID):
                step[key] = str(step[key])
    return episode, steps


class PromptDrift(RlmError):
    """A prompt file changed since the episode ran.

    Kept distinct from assembly drift on purpose: they have different causes
    and different responses, and collapsing them makes the assembly canary
    useless (see `episode_config`).
    """


def episode_config(snapshot: dict) -> tuple[Config, Any]:
    """Rebuild the config THIS EPISODE ACTUALLY RAN UNDER, from its own snapshot.

    Replay must never re-derive against the LIVE config file. Bumping
    `max_subcalls` or editing a prompt would otherwise change the re-derived
    message array and be reported as prompt-ASSEMBLY drift -- a false alarm on
    the one instrument whose entire job is detecting real drift, and the fastest
    way to teach an operator to ignore it. `config_snapshot` is the canonical
    dump of the validated model precisely so this is possible.

    Prompt changes are surfaced SEPARATELY, as `PromptDrift`. The registry here
    is built UNPINNED over the episode's own prompt paths, so a changed file is
    reported rather than thrown: the pinned path (`Config`'s own validator)
    raises a sha256 mismatch that reads like a config error, which buries the
    finding instead of naming it.

    The live config still decides WHERE to read from (`trace.db_path`,
    `trace.blob_root`) and which server to talk to; the snapshot decides what
    everything MEANT.
    """
    fields = {k: v for k, v in snapshot.items() if k in Config.model_fields}
    if "scaffold" not in fields:
        raise ConfigError("config_snapshot carries no scaffold block; this "
                          "episode predates snapshot-based replay")
    prompts = (fields["scaffold"].get("prompts") or {})
    # The envelope block is optional and only present from the S2 A/B onward.
    # It must be rebuilt when the snapshot has it, or `registry.hashes()` comes
    # back missing the `leaf_envelope.*` entries the episode recorded and every
    # replay of an envelope episode reads as prompt DRIFT.
    envelope_ref = prompts.get("leaf_envelope")
    # Same rule, same bug class, for §8's baseline prompts (S4): a slot the
    # registry loads but this rebuild skips comes back missing from
    # `registry.hashes()`, and EVERY episode recorded since it landed replays as
    # prompt DRIFT. `Config.prompt_registry` is the enumeration this one has to
    # agree with; the `or {}` keeps a pre-S4 snapshot (no baselines block at
    # all) rebuilding exactly as it did before.
    baseline_refs = prompts.get("baselines") or {}
    # Same rule again for the delegation arm's root prompt (2026-08-20). Absent
    # from every snapshot recorded before it, present in every `rlm-restricted`
    # one, and a rebuild that skipped it would replay those as prompt DRIFT.
    restricted_ref = prompts.get("root_restricted")
    try:
        registry = PromptRegistry.from_files(
            root_path=Path(prompts["root"]["path"]),
            root_restricted_path=(Path(restricted_ref["path"])
                                  if restricted_ref else None),
            leaf_prefix_path=Path(prompts["leaf_prefix"]["path"]),
            leaf_envelope_path=(Path(envelope_ref["path"]) if envelope_ref else None),
            strategy_paths={cat: Path(ref["path"])
                            for cat, ref in prompts["strategy_templates"].items()},
            baseline_paths={name: Path(ref["path"])
                            for name, ref in baseline_refs.items()},
        ).load()
    except KeyError as exc:
        raise ConfigError(f"config_snapshot has no prompt path for {exc}") from exc

    recorded = snapshot.get("prompt_hashes") or {}
    if recorded and registry.hashes() != recorded:
        now = registry.hashes()
        differing = sorted(k for k in set(recorded) | set(now)
                           if recorded.get(k) != now.get(k))
        raise PromptDrift(
            f"{len(differing)} prompt hash(es) changed since this episode ran: "
            f"{differing}. The message array cannot be re-derived against prompt "
            f"text the episode never saw — this is a prompt change, not "
            f"prompt-assembly drift.")
    return Config.model_validate(fields), registry


def _rederive_messages(cfg: Config, registry, episode: dict, steps: list[dict],
                        blob_root: Path) -> list[list[dict]]:
    """Rebuild, from the trace ALONE, the message array sent at every root turn.

    `cfg`/`registry` are the EPISODE's (from `episode_config`), never the live
    config's. Nothing here reads the lifecycle log, the task file, or a server
    -- that is the S3 gate condition, and it is why `compose_user_message` is a
    pure function of trace-recoverable values and why the task's instruction
    text is carried in `config_snapshot`.
    """
    snapshot = episode.get("config_snapshot") or {}
    task_meta = snapshot.get("task") or {}
    system = registry.render_root(task_meta.get("category", "default"))
    max_subcalls = cfg.scaffold.budgets.max_subcalls

    turns = [s for s in steps
             if s["action_type"] == ActionType.REPL_EXEC and s["root_request_ref"]]
    arrays: list[list[dict]] = []
    messages: list[dict] = [{"role": "system", "content": system}]
    for n, step in enumerate(turns, start=1):
        # Sub-calls already spent when this turn's message was composed: the
        # distinct call_ids hanging off every EARLIER turn.
        earlier = {t["step_idx"] for t in turns[:n - 1]}
        spent = {s["call_id"] for s in steps
                 if s["action_type"] == ActionType.LLM_CALL
                 and s["parent_step_idx"] in earlier and s["call_id"]}
        remaining = max(0, max_subcalls - len(spent))
        if n == 1:
            content = compose_user_message(
                turn=1, subcalls_remaining=remaining,
                task_text=task_meta.get("text", ""))
        else:
            prev = turns[n - 2]
            observation = prev["observation_view"]
            if prev["status"] == StepStatus.REJECTED:
                observation = no_cell_observation(cfg)
            content = compose_user_message(
                turn=n, subcalls_remaining=remaining,
                observation=observation if observation is not None else "")
        messages.append({"role": "user", "content": content})
        arrays.append([dict(m) for m in messages])
        rendered = _rendered(blob_root, step["root_request_ref"])
        messages.append(history_message(
            rendered, step["action_payload"] or "", cfg.scaffold.root.history_mode))
    return arrays


def _blob(blob_root: Path, rel: str) -> dict[str, bytes]:
    return unpack_blob((blob_root / rel).read_bytes())


def _rendered(blob_root: Path, rel: str) -> str:
    return _blob(blob_root, rel)["rendered"].decode("utf-8", "replace")


def _render_transcript(cfg: Config, steps: list[dict], out) -> None:
    langs = cfg.scaffold.cell_extraction.languages
    select = cfg.scaffold.cell_extraction.select
    print("\n--- transcript ---", file=out)
    for step in steps:
        kind = step["action_type"]
        if kind == ActionType.REPL_EXEC:
            cell = extract_cell(step["action_payload"] or "", langs, select)
            head = "cell" if cell else "no cell"
            print(f"\n[{step['step_idx']}] root {kind} ({step['status']}, {head})",
                  file=out)
            if cell:
                print("  " + "\n  ".join(cell.strip().splitlines()[:20]), file=out)
            view = (step["observation_view"] or "").strip()
            if view:
                print("  -> " + "\n     ".join(view.splitlines()[:12]), file=out)
        elif kind == ActionType.LLM_CALL:
            # The ACTOR, not a literal "leaf": B2's reduce step is an
            # `llm_call` with `actor='root'` (`rlm/arms.py`), and a transcript
            # that labels it "leaf" says the call happened on a server it never
            # touched — the one fact a reader most needs this line for.
            actor = step.get("actor") or "leaf"
            print(f"[{step['step_idx']}] {actor} llm_call ({step['status']}, "
                  f"parent {step['parent_step_idx']}, retry {step['retry_idx']}, "
                  f"tokens {step['tokens_in']}/{step['tokens_out']})", file=out)
        else:
            print(f"\n[{step['step_idx']}] FINAL: {step['action_payload']!r}", file=out)


def _first_difference(want: list[dict], got: list[dict]) -> str:
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return (f"message {i} ({a.get('role')}): stored {a.get('content', '')[:120]!r} "
                    f"vs re-derived {b.get('content', '')[:120]!r}")
    return f"stored {len(want)} messages, re-derived {len(got)}"
