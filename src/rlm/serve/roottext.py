"""Pure text shaping for a root turn: render in, message or cell out.

Six functions, all of them string and regex work: recover what the chat
template wrote, split reasoning from answer, build the history message a past
turn becomes, derive a per-turn seed, strip reasoning, extract the REPL cell.

WHY THIS IS ITS OWN MODULE, and what it unblocks. These functions used to live
in `src/rlm/serve/rootclient.py` next to `RootConversation`, which holds a `ServerClient`
-- so `rlm.serve.rootclient` is in the dependency rule's FORBIDDEN_RLM set and no
lint-covered module may import it. That is a real cost, not a label: the replay
path needs `history_message` and `extract_cell` to re-derive an episode from
the trace store, and could not be isolated while they sat behind an HTTP
client. Splitting the text from the transport puts these under §5's
dependency-rule lint (`tests/test_import_rules.py` ISOLATED) and lets replay
follow.

`src/rlm/serve/rootclient.py` re-exports all six, so every existing import site keeps
working.

Extracted from `src/rlm/serve/rootclient.py` on 2026-08-22, unchanged.
"""
from __future__ import annotations

import re


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


def split_reasoning(raw: str, *, open_block: bool = False) -> tuple[str, str]:
    """(reasoning, content) for a raw completion.

    With thinking OFF the prompt closes the think block before the model
    speaks, the model emits no tags, and reasoning is ''. With thinking ON
    the prompt ends in an OPEN `<think>\\n`, so the completion carries the
    reasoning, then `</think>`, then the answer -- and never an opening
    tag; `open_block=True` says so and the split happens at the first
    `</think>`. A leading `<think>...</think>` in the completion itself is
    also honoured (belt and braces; the template would double it otherwise).
    The template re-renders reasoning inside its own think block and trims
    both parts, so with the model's `\\n</think>\\n\\n` the re-render is a
    byte-for-byte extension either way.

    Splits at the FIRST `</think>`, unlike `strip_reasoning` (D16), which
    keeps the tail after the LAST: that one finds the cell the model meant,
    this one restores the history the template will re-render."""
    if open_block:
        head, sep, tail = raw.partition("</think>")
        if sep:
            return head.strip(), tail.lstrip("\n")
        return "", raw
    m = _LEADING_THINK_RE.match(raw)
    if not m:
        return "", raw
    return m.group(1).strip(), raw[m.end():]


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
    reasoning, content = split_reasoning(
        raw, open_block=assistant_prefix(rendered).rstrip().endswith("<think>"))
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
    the middle.

    Counterpart: `split_reasoning` splits at the FIRST close tag for history
    reconstruction."""
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
