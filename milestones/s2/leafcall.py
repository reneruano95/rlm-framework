"""One pinned, un-retried leaf call — the transport both S2 experiments share.

WHY THIS EXISTS RATHER THAN `LLMDispatcher`. C4 is the production path and it
is right to be what it is, but two of its deliberate properties make it the
wrong instrument for §7 #2 and §7 #3:

  * **It never sets `id_slot`, on purpose** ("a failed attempt's slot is not
    pinned, so there is no affinity to preserve"). Both experiments here are
    *about* slot residency: the sweep exploits a resident chunk to ask three
    questions for the price of one prefill (§7 #3 (d): 1.71 s warm vs 35.45 s
    cold), and the ub experiment's whole design is two chunks alternating on
    ONE slot. Without the pin, LCP routing decides where a call lands and the
    measurement is of the router, not of the cache.
  * **Its sampling seed is fixed per `DispatchTarget`**, set once at
    construction. The sweep needs three trials at the SAME temperature with
    DIFFERENT seeds, so the cell reports a rate under production sampling
    rather than one deterministic draw re-run three times.

Everything that determines what the model SEES is production's, not this
module's: the system prefix is the sha256-pinned registry text via
`Config.prompt_registry()`, the user segment is composed by
`rlm.dispatcher.compose_leaf_user` (chunk first, question LAST — §4), the
model's own control-token literals are neutralised by
`rlm.dispatcher.neutralise_control_tokens` against the marker set derived from
the server's own `/props` chat template, and the render goes through
`/apply-template` on the target server. If any of that drifted from C4, the
sweep would be measuring a prompt that production never sends.

What is deliberately NOT here: retries (a failed call is a recorded fact, and
a silent retry would hide a server-side truncation behind a second draw),
step logging (§6's `steps` table belongs to episodes; these calls are not
episodes), and the semaphore (every call here is serial by construction —
concurrency is exactly what §7 #3 (e) measured as a 3x pricing error).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from rlm.config import Config
from rlm.dispatcher import (
    ServerClient,
    chat_control_markers,
    compose_leaf_user,
    neutralise_control_tokens,
)

#: A string that cannot occur in the leaf prefix, used once by `prepare()` to
#: find where the user message starts in the rendered prompt so the §4 head can
#: be tokenized on its own. `rfind` on a generic character (C4 uses the user
#: message itself, which it already has) would be ambiguous here, because
#: `prepare()` renders a probe rather than a real request.
PROBE_MARKER = "RLM-S2-PREFIX-PROBE"

#: The three instruction LAYOUTS (`milestones/s2/run_distance.py`, §4 "INSTRUCTION DECAY").
#: Composed here, scaffold-side, for the same reason C4 composes the user
#: segment itself: a layout the model could alter is not a layout.
#:
#:   A — `[system prefix][chunk][question]`. TODAY'S SHIPPED LAYOUT (§4), and
#:       the one every leaf measurement so far was taken under. Default, so
#:       every existing caller keeps sending exactly the bytes it sent before.
#:   B — `[system prefix][chunk][same prefix text again][question]`. The
#:       leading prefix stays byte-identical and the chunk stays at a constant
#:       offset, so §4's cache contract survives; the cost is the repeated
#:       tokens.
#:   C — `[chunk][system prefix][question]`, no leading prefix at all. Maximum
#:       treatment, and it destroys the byte-identical head — it prices the
#:       extreme rather than proposing it.
LAYOUT_A = "A"
LAYOUT_B = "B"
LAYOUT_C = "C"
LAYOUTS = (LAYOUT_A, LAYOUT_B, LAYOUT_C)


@dataclass(slots=True)
class LeafAnswer:
    """One /completion call, shaped for a JSONL record.

    `raw` is the model's output VERBATIM — never stripped, never truncated.
    Every number here comes from the server's own final SSE event
    (`timings`), not from this process's arithmetic, with the single
    exception of `wall_s`/`overhead_s`.
    """

    raw: str
    tokens_in: int
    tokens_out: int
    tokens_cached: int
    slot_id: int
    stop_type: str
    truncated: bool
    prefill_ms: float
    decode_ms: float
    wall_s: float
    overhead_s: float
    rendered_chars: int
    rendered_sha256: str
    prefix_tokens: int | None = None
    seed: int = 0
    id_slot: int | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "raw_output": self.raw,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_cached": self.tokens_cached,
            "slot_id": self.slot_id,
            "stop_type": self.stop_type,
            "truncated": self.truncated,
            "prefill_ms": round(self.prefill_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "wall_s": round(self.wall_s, 3),
            "overhead_s": round(self.overhead_s, 3),
            "rendered_chars": self.rendered_chars,
            "rendered_sha256": self.rendered_sha256,
            "prefix_tokens": self.prefix_tokens,
            "seed": self.seed,
            "id_slot": self.id_slot,
        }


@dataclass
class PinnedLeafCaller:
    """The leaf server, addressed one pinned slot at a time."""

    client: ServerClient
    system_prefix: str
    max_predict: int
    temperature: float
    top_p: float
    enable_thinking: bool = False
    slot_capacity_tokens: int | None = None
    markers: tuple[str, ...] = field(default_factory=tuple)
    prefix_tokens: int | None = None
    #: Does this caller's prefix ask for the JSON envelope? Carried, not acted
    #: on: the caller still sends and records ONE un-retried draw, and the
    #: parse/validate happens in the scorer. A retry here would confound the S2
    #: refusal A/B -- an arm that emits more malformed replies would silently
    #: get more draws at the same question (`milestones/s2/run_refusal_ab.py`).
    envelope: bool = False
    #: Which of the three instruction layouts this caller composes. `"A"` is
    #: the shipped one, so a caller that does not ask for a layout sends the
    #: bytes every earlier S2 measurement was taken with.
    layout: str = LAYOUT_A
    #: Tokens in the PREFIX TEXT ITSELF (no chat-template markup), measured by
    #: `prepare()`. Layouts B and C carry that text inside the user message, so
    #: admission has to price it there rather than in the system head.
    prefix_body_tokens: int | None = None

    @property
    def prefix_sha256(self) -> str:
        """The rendered prefix's own hash, on every record. §4's byte-identical
        head is the contract the whole A/B rests on, and an arm labelled `v2`
        that actually sent `v1` bytes would be undetectable without this."""
        return hashlib.sha256(self.system_prefix.encode("utf-8")).hexdigest()

    @classmethod
    def from_config(cls, cfg: Config, *, timeout: float | None = None,
                    system_prefix: str | None = None,
                    envelope: bool | None = None) -> "PinnedLeafCaller":
        """Production's bytes and production's sampling, from the validated
        config — including the sha256-pinned leaf prefix, which is the ONE
        string §4's byte-identical-head contract is about.

        `system_prefix`/`envelope` override that pair, and ONLY that pair, for
        the refusal A/B: its arms differ in exactly the system head (v1 or v2,
        with or without the envelope block), and every other byte — sampling,
        max_predict, slot capacity, the user-segment composition — has to stay
        the shipped config's or the arms would differ in more than one thing.
        Both overrides still come from sha256-pinned registry files; there is no
        path here for an inline prompt string.
        """
        leaf = cfg.servers.leaf
        sampling = cfg.scaffold.sampling.leaf
        return cls(
            client=ServerClient(
                f"http://127.0.0.1:{leaf.port}",
                timeout=timeout or cfg.scaffold.retries.per_call_timeout_s),
            system_prefix=(cfg.prompt_registry().load().leaf_prefix()
                           if system_prefix is None else system_prefix),
            max_predict=cfg.scaffold.budgets.max_predict.leaf,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            enable_thinking=cfg.scaffold.leaf.enable_thinking,
            slot_capacity_tokens=leaf.ctx // leaf.parallel,
            envelope=(cfg.scaffold.leaf_envelope.enabled if envelope is None
                      else envelope),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def prepare(self) -> int | None:
        """Derive the marker set and measure the rendered §4 head, ONCE.

        Both are needed before the first real call: the markers because an
        unsanitised chunk could otherwise forge a system turn (C4's I1), and
        `prefix_tokens` because it is the number §7 #3 (a) compares
        `tokens_cached` against and because the sweep admits chunks against
        slot capacity BEFORE paying for a 32K prefill.
        """
        try:
            template = (await self.client.props()).get("chat_template") or ""
        except Exception:  # noqa: BLE001 -- /props widens the guarantee, it is
            # not the guarantee: fall back to the hardcoded floor set.
            template = ""
        self.markers = chat_control_markers(template)

        rendered = await self.client.apply_template(
            [{"role": "system", "content": self.system_prefix},
             {"role": "user", "content": PROBE_MARKER}],
            chat_template_kwargs={"enable_thinking": self.enable_thinking})
        cut = rendered.rfind(PROBE_MARKER)
        if cut > 0:
            head = await self.client.tokenize(rendered[:cut], add_special=True)
            self.prefix_tokens = len(head)
        self.prefix_body_tokens = len(
            await self.client.tokenize(self.system_prefix, add_special=False))
        return self.prefix_tokens

    def head_tokens(self) -> int:
        """Tokens this caller's LAYOUT spends on instructions and markup.

        Layout A pays the rendered system head once. B pays it AND the prefix
        text again inside the user message. C pays no system head — the whole
        instruction rides after the chunk — but still pays the user/assistant
        markup, approximated by the measured head minus the prefix body.
        """
        head = self.prefix_tokens or 0
        body = self.prefix_body_tokens or 0
        if self.layout == LAYOUT_B:
            return head + body
        if self.layout == LAYOUT_C:
            return body + max(head - body, 0)
        return head

    def admits(self, chunk_tokens: int, *, question_tokens: int = 128) -> bool:
        """Would `chunk_tokens` plus this layout's head plus a question fit one
        slot?

        Arithmetic on numbers already measured (the manifest's chunk length,
        `prepare()`'s head) rather than a pre-flight /tokenize per call: at
        32K a pre-flight round trip costs as much as it saves, and the sweep
        already knows exactly how long every chunk is.
        """
        if self.slot_capacity_tokens is None:
            return True
        return (self.head_tokens() + chunk_tokens + question_tokens
                <= self.slot_capacity_tokens)

    def compose(self, *, question: str, chunk: str | None) -> list[dict[str, str]]:
        """The message array for this layout, composed SCAFFOLD-SIDE.

        Layout A goes through `rlm.dispatcher.compose_leaf_user` — production's
        own function, not a copy — so the control arm is byte-identical to what
        C4 sends. B and C interpolate the SAME prefix text (never a reworded
        one: the variable under test is POSITION) between the chunk and the
        question, and the chunk is still the first byte of the user message, so
        a re-query of the same chunk still extends the cached prefix.
        """
        if self.layout == LAYOUT_A:
            return [{"role": "system", "content": self.system_prefix},
                    {"role": "user", "content": compose_leaf_user(question, chunk)}]
        if chunk is None:
            body = f"{self.system_prefix}\n\n{question}"
        else:
            body = f"{chunk}\n\n{self.system_prefix}\n\n{question}"
        if self.layout == LAYOUT_B:
            return [{"role": "system", "content": self.system_prefix},
                    {"role": "user", "content": body}]
        if self.layout == LAYOUT_C:
            return [{"role": "user", "content": body}]
        raise ValueError(f"unknown layout {self.layout!r}")

    async def ask(self, *, question: str, chunk: str | None, seed: int,
                  id_slot: int | None) -> LeafAnswer:
        """One call. No retry: a failure raises, and the caller records it."""
        t_start = time.perf_counter()
        messages = [
            msg if msg["role"] != "user"
            else {**msg, "content": neutralise_control_tokens(msg["content"],
                                                              self.markers)}
            for msg in self.compose(question=question, chunk=chunk)
        ]
        rendered = await self.client.apply_template(
            messages,
            chat_template_kwargs={"enable_thinking": self.enable_thinking})
        t_dispatch = time.perf_counter()
        result = await self.client.completion(
            rendered, n_predict=self.max_predict, temperature=self.temperature,
            top_p=self.top_p, seed=seed, stream=True, id_slot=id_slot,
            cache_prompt=True)
        t_end = time.perf_counter()
        return LeafAnswer(
            raw=result.content,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            tokens_cached=result.cache_n,
            slot_id=result.slot_id,
            stop_type=result.stop_type,
            truncated=result.truncated,
            prefill_ms=result.prompt_ms,
            decode_ms=result.predicted_ms,
            wall_s=t_end - t_dispatch,
            overhead_s=t_dispatch - t_start,
            rendered_chars=len(rendered),
            rendered_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            prefix_tokens=self.prefix_tokens,
            seed=seed,
            id_slot=id_slot,
        )
