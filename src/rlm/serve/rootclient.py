"""Task 12: Root client -- render, hash, parse (spec §5/§6, D14-D17, D26).

D14: the root's conversation is driven through POST /apply-template (never
locally re-implemented Jinja -- "a canary that cannot fail is not a
canary", recipes §serverapi) then POST /completion with the exact string
`/apply-template` returned. `root_view_hash` is the sha256 of that string,
computed here so the caller (the episode runner) can log it and store the
string itself as the `root_request_ref` blob.

D15: the root always runs with `chat_template_kwargs.enable_thinking` from
`scaffold.root.enable_thinking` (default false, config.yaml). With it off,
Qwen3.6's template renders a pre-closed, empty think block
(`<think>\n\n</think>\n\n`) as part of the PROMPT, so `strip_reasoning` must
work whether or not that block is present or the model reopens one anyway.

D16 parse rule: split on the LAST `</think>` and keep the tail (the prompt
itself may open the block, so an unconditional prefix search would be
wrong), THEN select a fenced code block per `scaffold.cell_extraction`
(`languages`, `select`) -- never hardcoded, so the extractor and the shipped
prompt text (generated from the same config key, Conflict 5) can never
disagree.

Sampling: every /completion call carries `cfg.scaffold.sampling.root`
(temperature, top_p, seed) verbatim -- real, non-defaulted config, not
`rlm.serve.dispatcher.ServerClient.completion`'s (deliberately absent) defaults.

D26 append-only: `append_user` only ever appends; nothing already sent is
ever rewritten. `turn()` builds the assistant history message with
`history_message(rendered, raw, mode)`: under `prefix_plus_raw` (the
pre-amendment rule, kept so old snapshots replay exactly) that is the
template's own text after the last assistant marker plus the model's raw
completion, verbatim; under `raw` (shipped default) it is the model's
completion alone, reasoning split into `reasoning_content` via
`split_reasoning`, and the template re-renders the think block itself --
either way the template trims every message's content, so the next render
is a byte-for-byte extension of the previous one (measured: cache_n =
prev_len-4 over 6 turns, recipes §serverapi).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from rlm.config import Config
from rlm.serve.dispatcher import CompletionResult, ServerClient

# The pure text shaping -- assistant_prefix, split_reasoning,
# history_message, turn_seed, strip_reasoning, extract_cell -- lives in
# `rlm/roottext.py`. It is string and regex work with no transport, so it
# sits under §5's dependency-rule lint; this module cannot, because
# `RootConversation` holds a `ServerClient` and so `rlm.serve.rootclient` is in the
# rule's FORBIDDEN_RLM set. Splitting them is what lets the REPLAY path --
# which needs `history_message` and `extract_cell` to re-derive an episode
# from the trace store alone -- be lint-covered too.
#
# Re-exported here: every existing `from rlm.serve.rootclient import ...` site
# keeps working, including tests and `rlm/cli.py`.
from rlm.serve.roottext import (  # noqa: F401
    assistant_prefix, extract_cell, history_message, split_reasoning,
    strip_reasoning, turn_seed,
)


@dataclass(slots=True)
class RootTurn:
    """One root turn: the model's raw reply, the extracted REPL cell (or
    None on an extraction miss), the request hash + rendered string (D14),
    and the completion's usage/timing info for the caller to log as a
    step (rlm.serve.dispatcher.CompletionResult already carries exactly the
    fields steps.tokens_in/out/cached/slot_id/latency_* need)."""

    raw: str
    cell: str | None
    view_hash: str
    rendered: str
    usage: CompletionResult
    prefix_extended: bool | None = None


class RootConversation:
    """The root's append-only chat history plus the render/hash/parse cycle
    for one turn (D14-D17, D26). Talks to a `ServerClient` directly -- no
    retries/semaphore here, unlike leaf sub-calls: the root is one serial
    conversation, driven by the episode runner, which owns retry policy for
    the root turn itself if it wants one."""

    def __init__(self, client: ServerClient, cfg: Config, *, system: str | None = None) -> None:
        self._client = client
        self._enable_thinking = cfg.scaffold.root.enable_thinking
        self._max_predict = cfg.scaffold.budgets.max_predict.root
        self._languages = cfg.scaffold.cell_extraction.languages
        self._select = cfg.scaffold.cell_extraction.select
        # Real, non-defaulted sampling config (§6: config_snapshot must
        # record what actually ran; the benchmark's per-seed discipline is
        # meaningless if this never reaches the server).
        root_sampling = cfg.scaffold.sampling.root
        self._temperature = root_sampling.temperature
        self._top_p = root_sampling.top_p
        self._seed = root_sampling.seed
        self._seed_schedule = cfg.scaffold.root.seed_schedule
        self._history_mode = cfg.scaffold.root.history_mode
        self._prev_rendered: str | None = None
        self._prev_raw: str | None = None
        self._turns = 0
        self.messages: list[dict[str, Any]] = []
        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def append_user(self, text: str) -> None:
        """D26: append-only. Never rewrites an existing message -- everything
        mutable (turn counter, remaining budget, task text) belongs in the
        newest user message the caller composes before calling this."""
        self.messages.append({"role": "user", "content": text})

    async def turn(self) -> RootTurn:
        rendered = await self._client.apply_template(
            self.messages,
            chat_template_kwargs={"enable_thinking": self._enable_thinking},
        )
        view_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

        self._turns += 1
        seed = turn_seed(self._seed, self._turns, self._seed_schedule)
        result = await self._client.completion(
            rendered, n_predict=self._max_predict, temperature=self._temperature,
            top_p=self._top_p, seed=seed, stream=True)
        raw = result.content

        stripped = strip_reasoning(raw)
        cell = extract_cell(stripped, self._languages, self._select)

        # D26: the history message is `history_message(rendered, raw, mode)` --
        # under `raw`, the template supplies the think block and the next
        # render is the previous render + raw, byte for byte (v0.3.16); under
        # `prefix_plus_raw` the pre-amendment rule is reproduced for old
        # episodes.
        extended = None
        if self._prev_rendered is not None:
            # The template renders every message's content through `|trim`
            # (Qwen3.8 template line 103), so the previous turn's completion
            # reappears stripped -- that is the byte-for-byte contract.
            # (The tests assert the stronger `+ "<|im_end|>\n"` form; this
            # check is the prefix only, so it can never report a false
            # divergence.)
            extended = rendered.startswith(self._prev_rendered + self._prev_raw.strip())
        self.messages.append(history_message(rendered, raw, self._history_mode))
        self._prev_rendered, self._prev_raw = rendered, raw

        return RootTurn(raw=raw, cell=cell, view_hash=view_hash,
                         rendered=rendered, usage=result, prefix_extended=extended)
