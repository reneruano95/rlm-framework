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
`rlm.dispatcher.ServerClient.completion`'s (deliberately absent) defaults.

D26 append-only: `append_user` only ever appends; nothing already sent is
ever rewritten. `turn()` reconstructs the assistant history message as
exactly what the token stream contained -- the template's own text between
the last `<|im_start|>assistant\n` marker and the model's raw completion --
so a later render is a byte-for-byte prefix extension of an earlier one
(measured: cache_n = prev_len-4 over 6 turns, recipes §serverapi). Getting
this wrong (e.g. re-deriving the think-block text independently) would
silently break prefix-cache reuse without ever showing up as a test
failure on message *content* alone.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from rlm.config import Config
from rlm.dispatcher import CompletionResult, ServerClient

# The exact marker Qwen3.6's ChatML template (and every ChatML-family
# template) opens the generation turn with. Used only to recover, verbatim,
# whatever text the template placed after it (e.g. a pre-closed think
# block) so conversation history reconstructs the real token stream -- see
# the D26 note above.
_ASSISTANT_MARKER = "<|im_start|>assistant\n"

_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Belt-and-braces reasoning strip (D16). With enable_thinking=false the
    model emits no think tags at all -- the prompt already closed the
    block -- but this must be safe under either setting: strip a leading
    `<think>...</think>` if present, then split on the LAST `</think>` and
    keep the tail (the prompt may have opened a block the model never
    closed, or closed one the model reopens). Never regex `<think>` out of
    the middle."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.lstrip()


def extract_cell(text: str, languages: list[str], select: str) -> str | None:
    """Pick a fenced code block whose language tag is one of `languages`.
    `select` is "first" or "last" (scaffold.cell_extraction.select) --
    driven entirely by config so the extractor and the prompt files' wording
    (generated from the same key) can never disagree (Conflict 5).

    Returns None for zero blocks, an unterminated fence, or a fence in a
    language not listed -- callers treat that as a normal, correctable
    observation, never as an episode error (decided gap, task-12 brief)."""
    lang_alt = "|".join(re.escape(lang) for lang in languages)
    pattern = re.compile(
        rf"```[ \t]*(?:{lang_alt})[ \t]*\r?\n(.*?)(?:\r?\n)?```",
        re.DOTALL,
    )
    blocks = pattern.findall(text)
    if not blocks:
        return None
    return blocks[0] if select == "first" else blocks[-1]


@dataclass(slots=True)
class RootTurn:
    """One root turn: the model's raw reply, the extracted REPL cell (or
    None on an extraction miss), the request hash + rendered string (D14),
    and the completion's usage/timing info for the caller to log as a
    step (rlm.dispatcher.CompletionResult already carries exactly the
    fields steps.tokens_in/out/cached/slot_id/latency_* need)."""

    raw: str
    cell: str | None
    view_hash: str
    rendered: str
    usage: CompletionResult


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

        result = await self._client.completion(
            rendered, n_predict=self._max_predict, temperature=self._temperature,
            top_p=self._top_p, seed=self._seed, stream=True)
        raw = result.content

        stripped = strip_reasoning(raw)
        cell = extract_cell(stripped, self._languages, self._select)

        # D26: reconstruct the assistant history message as exactly what the
        # token stream contained -- the template's own tail after the LAST
        # assistant marker (e.g. a pre-closed think block), followed by the
        # model's raw completion -- so the NEXT turn's render is a
        # byte-for-byte prefix extension of this one.
        idx = rendered.rfind(_ASSISTANT_MARKER)
        assistant_prefix = rendered[idx + len(_ASSISTANT_MARKER):] if idx != -1 else ""
        self.messages.append({"role": "assistant", "content": assistant_prefix + raw})

        return RootTurn(raw=raw, cell=cell, view_hash=view_hash,
                         rendered=rendered, usage=result)
