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

    @classmethod
    def from_config(cls, cfg: Config, *, timeout: float | None = None) -> "PinnedLeafCaller":
        """Production's bytes and production's sampling, from the validated
        config — including the sha256-pinned leaf prefix, which is the ONE
        string §4's byte-identical-head contract is about."""
        leaf = cfg.servers.leaf
        sampling = cfg.scaffold.sampling.leaf
        return cls(
            client=ServerClient(
                f"http://127.0.0.1:{leaf.port}",
                timeout=timeout or cfg.scaffold.retries.per_call_timeout_s),
            system_prefix=cfg.prompt_registry().load().leaf_prefix(),
            max_predict=cfg.scaffold.budgets.max_predict.leaf,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            enable_thinking=cfg.scaffold.leaf.enable_thinking,
            slot_capacity_tokens=leaf.ctx // leaf.parallel,
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
        return self.prefix_tokens

    def admits(self, chunk_tokens: int, *, question_tokens: int = 128) -> bool:
        """Would `chunk_tokens` plus the head plus a question fit one slot?

        Arithmetic on numbers already measured (the manifest's chunk length,
        `prepare()`'s head) rather than a pre-flight /tokenize per call: at
        32K a pre-flight round trip costs as much as it saves, and the sweep
        already knows exactly how long every chunk is.
        """
        if self.slot_capacity_tokens is None:
            return True
        head = self.prefix_tokens or 0
        return head + chunk_tokens + question_tokens <= self.slot_capacity_tokens

    async def ask(self, *, question: str, chunk: str | None, seed: int,
                  id_slot: int | None) -> LeafAnswer:
        """One call. No retry: a failure raises, and the caller records it."""
        t_start = time.perf_counter()
        user_message = neutralise_control_tokens(
            compose_leaf_user(question, chunk), self.markers)
        rendered = await self.client.apply_template(
            [{"role": "system", "content": self.system_prefix},
             {"role": "user", "content": user_message}],
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
