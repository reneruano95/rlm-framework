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


def assistant_prefix(rendered: str) -> str:
    """The template's OWN text after the last assistant marker in `rendered`
    (e.g. Qwen3.6's pre-closed `<think>\\n\\n</think>\\n\\n` block when
    `enable_thinking` is false).

    D26: the assistant history message is exactly this prefix plus the model's
    raw completion, so the next render is a byte-for-byte prefix extension of
    this one. Public because `rlm replay` has to reproduce the same message
    array offline from the stored render -- re-deriving the think-block text
    independently there would break the state-rule check for a reason that has
    nothing to do with prompt-assembly drift."""
    idx = rendered.rfind(_ASSISTANT_MARKER)
    return rendered[idx + len(_ASSISTANT_MARKER):] if idx != -1 else ""


_LEADING_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.DOTALL)


def split_reasoning(raw: str) -> tuple[str, str]:
    """(reasoning, content) for a raw completion. With thinking off the model
    emits no tags and reasoning is ''. With thinking on the reply opens with
    `<think>…</think>`; the template re-renders that as its own block from
    `reasoning_content`, so the content must not carry it twice."""
    m = _LEADING_THINK_RE.match(raw)
    if not m:
        return "", raw
    reasoning = m.group(1).strip()
    content = raw[m.end():]
    return reasoning, content


def history_message(rendered: str, raw: str, mode: str) -> dict[str, str]:
    """The assistant message appended to the root's history for the turn whose
    request rendered as `rendered` and whose completion was `raw`, under
    `scaffold.root.history_mode`. THE one definition: `RootConversation.turn`
    and `rlm replay` both call it, which is what makes the offline
    re-derivation exact (§6 state rule)."""
    if mode == "prefix_plus_raw":
        return {"role": "assistant", "content": assistant_prefix(rendered) + raw}
    if mode != "raw":
        raise ValueError(f"unknown history_mode {mode!r}")
    reasoning, content = split_reasoning(raw)
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def turn_seed(base: int, turn: int, schedule: str) -> int:
    """The sampling seed for root turn `turn` (1-based) under
    `scaffold.root.seed_schedule`. `fixed` reproduces the pre-v0.3.16
    behaviour; `per_turn` gives every turn its own seed while staying a pure
    function of (episode seed, turn index), so the snapshot still determines
    the run. The stride is 1,000: turns beyond it would collide with the next
    base seed's schedule, and a 32K root window cannot hold 1,000 turns --
    deliberately not asserted, because an exception here would surface as a
    mislabelled `server_unreachable` in the turn loop."""
    if schedule == "fixed":
        return base
    return base * 1000 + turn


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
            extended = rendered.startswith(self._prev_rendered + self._prev_raw.strip())
        self.messages.append(history_message(rendered, raw, self._history_mode))
        self._prev_rendered, self._prev_raw = rendered, raw

        return RootTurn(raw=raw, cell=cell, view_hash=view_hash,
                         rendered=rendered, usage=result, prefix_extended=extended)
