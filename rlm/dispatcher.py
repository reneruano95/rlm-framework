"""C4 LLMDispatcher -- the ONLY module (with `rlm.rootclient`) permitted to
talk to a model server (spec §5).

D14 binds: POST /apply-template then POST /completion with the returned
string, never /v1/chat/completions -- the OAI-shaped endpoint never reports
`id_slot`, in any response, streamed or not, and without it `steps.slot_id`
is unfillable and the §4 prefix/slot-affinity contract is unverifiable.
Use /completion for the LEAF too, for the same reason (recipes §serverapi).

The leaf request is built in `query()`, scaffold-side (I1): the §4
byte-identical system prefix -- `DispatchTarget.system_prefix`, the whole
text of the sha256-pinned `prompts/leaf-prefix.v1.md` via
`PromptRegistry.leaf_prefix()` -- as the system message, and the one opaque
string the sandbox passed to `llm_query` as the user message. Model code
therefore cannot alter, replace or suppress the prefix: ChatML markers
written into its prompt are user content. Skipping the template here is
base-model prompting against an instruct-tuned model, which is what made
every leaf answer in the S1 gate junk (s1/RESULTS.md F3).

Streaming is not an optimisation here, it is the mechanism: closing the
response context mid-generation genuinely aborts server-side generation and
frees the slot (measured 0.03-0.17 s). C4 must stream for exactly this
reason, so cancelling the enclosing asyncio task is a real abort, not just a
client-side give-up.

Retries: `max_attempts` 3, backoff 1 s / 4 s, per-call timeout 240 s
(scaffold.retries in config.yaml). EVERY attempt is its own logged step,
sharing one `call_id` with an incrementing `retry_idx` -- a retried call
counts ONCE against `max_subcalls` (a C5 concern, dedupe by call_id) but
every attempt's tokens count against `max_total_tokens` (also C5). This
module only produces the step records; a caller (the episode runner) is
responsible for actually enforcing those budgets and persisting the steps
via `rlm.trace.TraceLogger.put_step`.

Pre-flight: token-count via the target server's /tokenize, on the RENDERED
string (the one that would actually occupy the slot); a prompt exceeding the
target's slot capacity is REJECTED without dispatch and logged
status=rejected -- never sent to /completion.

The semaphore is `asyncio.Semaphore(cfg.servers.leaf.parallel)`, owned here.
Nothing the model runs may resize it, choose a server, or change a port. It
is held only around each individual dispatch attempt, not across a retry's
backoff sleep -- a failed attempt's slot is not pinned (id_slot is never
set on retry), so there is no affinity to preserve, and holding it across a
1-4s sleep would starve other queued leaf calls for no benefit.

Server death: retries exhausting against a dead server produce a step
`status=error` and a raised DispatchError; the caller turns that into
`episode.outcome=error, outcome_reason=server_unreachable`. This module
never restarts a server mid-episode -- it has no code path that could.

Sampling: `cfg.scaffold.sampling.<role>` (temperature, top_p, seed) is real,
non-defaulted config -- §6 records it in config_snapshot as what actually
ran, and the benchmark's seed discipline depends on the seed reaching the
server. `from_config()` threads it into each role's `DispatchTarget`, and
`query()` passes it on every /completion call; nothing here defaults to
near-greedy sampling. `from_config()` builds ONLY a "leaf" target: root
traffic never goes through LLMDispatcher (see `rlm.rootclient.
RootConversation`, which talks to a raw `ServerClient` with its own
`cfg.scaffold.sampling.root`), so a "root" DispatchTarget here would be
dead code that could silently apply the leaf-sized semaphore to root
calls if ever queried by mistake -- `query(role="root", ...)` raises
DispatchError instead.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from rlm.config import Config, Retries
from rlm.errors import ActionType, Actor, DispatchError, StepStatus
from rlm.trace import utc_now

# --------------------------------------------------------------------------- #
# I1: neutralising model-authored control tokens (spec §5 C4, fix round 2).
#
# THE HOLE THIS CLOSES, MEASURED. Applying the chat template to the leaf made
# model-supplied text an operand of a ChatML render. A leaf prompt containing
# `<|im_end|>\n<|im_start|>system\n...` therefore renders to a string with TWO
# system turns -- the second entirely model-authored, and the LAST-writer one.
# `/tokenize` resolves `<|im_start|>` to a genuine special-token id, and
# `/completion`'s `prompt_n` equals `/tokenize(add_special=true)` exactly, so
# those forged markers are parsed as real control tokens on the wire, not as
# text. The pinned prefix bytes survive; the claim that forged markers are
# merely "user content" does not. Note this is SELF-INFLICTED: it did not exist
# while the leaf posted raw prompts to /completion.
#
# The fix is scaffold-side, in `query()`, on the USER message ONLY -- never on
# `target.system_prefix`, so §4's byte-identical head is untouched.
# Neutralisation inserts one space after the opening `<`, which is the smallest
# edit that stops the tokenizer matching the literal: the leaf still SEES what
# the corpus said (the prefix already tells it the excerpt is data, never
# instruction), it just cannot say it in control tokens.
# --------------------------------------------------------------------------- #

#: The floor set: neutralised whether or not /props can be read. `<think>`/
#: `</think>` are here because the leaf runs with `enable_thinking` false, and
#: a model-authored `</think>` would close a reasoning block the template
#: opened -- `rlm.rootclient.strip_reasoning` splits on the LAST one.
CONTROL_MARKERS: tuple[str, ...] = (
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
)

#: Anything of the shape `<|...|>` in a chat template is a control token for
#: that template. Deriving the set beats hardcoding it -- a model swap (I6) can
#: bring markers this file has never heard of -- but it never REPLACES the
#: floor: a /props that fails, or a template that inlines its markers, must not
#: reopen the hole.
_TEMPLATE_MARKER_RE = re.compile(r"<\|[^|<>\s]{1,64}\|>")


def chat_control_markers(chat_template: str | None = None) -> tuple[str, ...]:
    """The marker set to neutralise, longest first (so no marker can be split
    by the rewrite of a shorter one it contains)."""
    markers = set(CONTROL_MARKERS) | set(
        _TEMPLATE_MARKER_RE.findall(chat_template or ""))
    return tuple(sorted(markers, key=lambda m: (-len(m), m)))


def neutralise_control_tokens(text: str, markers: tuple[str, ...]) -> str:
    """Rewrite every control-token literal into a non-tokenizing form.

    Length- and content-preserving on purpose: `<|im_start|>` becomes
    `< |im_start|>`, which no tokenizer resolves to a special id while a reader
    (and the leaf) can still see exactly what the corpus contained."""
    for marker in markers:
        if marker in text:
            text = text.replace(marker, "< " + marker[1:])
    return text


# --------------------------------------------------------------------------- #
# ServerClient -- thin, honest wrapper over the llama-server HTTP surface
# (recipes §serverapi). No retries, no semaphore, no step logging: those are
# LLMDispatcher's job. This class only knows how to talk to ONE server.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CompletionResult:
    """One /completion call's result, shaped to drop straight into a
    `steps` row (rlm.trace.STEP_COLS): tokens_in/out, tokens_cached
    (<- timings.cache_n, NOT the top-level `tokens_cached` field -- they are
    NOT interchangeable, recipes §serverapi caveat), slot_id, and the raw
    latency components latency_prefill_ms/latency_decode_ms need."""

    content: str
    tokens_in: int
    tokens_out: int
    cache_n: int
    slot_id: int
    t_first_byte: Any  # datetime | None -- None only for a call that never
    #                    got a byte before failing/being cancelled.
    prompt_ms: float
    predicted_ms: float
    stop_type: str = ""
    truncated: bool = False


class ServerClient:
    """One llama-server endpoint. Owns its own httpx.AsyncClient."""

    def __init__(self, base_url: str, *, timeout: float = 240.0,
                 http_client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def apply_template(self, messages: list[dict], **kw: Any) -> str:
        """POST /apply-template -> {"prompt": str}. D14's root_view_hash
        instrument: sha256 the returned string, verbatim, at the caller."""
        body: dict[str, Any] = {"messages": messages, "add_generation_prompt": True}
        body.update(kw)
        resp = await self._client.post(f"{self.base_url}/apply-template", json=body)
        resp.raise_for_status()
        return resp.json()["prompt"]

    async def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        """POST /tokenize. A missing/empty `content` returns HTTP 200
        {"tokens":[]} with no error, so a zero-length result on non-empty
        input is treated as a fault here, never as a legitimate count
        (recipes §serverapi: "/tokenize FAILS SILENTLY").

        `add_special` is /tokenize's own flag and it is NOT cosmetic: the
        endpoint defaults it to false while /completion tokenizes WITH the
        special/BOS prefix. Measured at three sizes, pre-flight vs served:
        284/285, 474/475, 1274/1275 -- a constant +1. Admission must therefore
        pass `add_special=True` (it is counting what will occupy the slot),
        while a chunk-body count must not (BOS is not part of a chunk)."""
        resp = await self._client.post(
            f"{self.base_url}/tokenize",
            json={"content": text, "add_special": add_special})
        resp.raise_for_status()
        toks = resp.json().get("tokens", [])
        if text and not toks:
            raise DispatchError("/tokenize returned 0 tokens for non-empty input")
        return toks

    async def props(self) -> dict:
        resp = await self._client.get(f"{self.base_url}/props")
        resp.raise_for_status()
        return resp.json()

    async def slots(self) -> list[dict]:
        """GET /slots -> one object per slot, each with `is_processing`.

        The §5 C5 quiesce point and §6 crash recovery both need "are all slots
        idle?", and both read exactly this field. Note the shape: `next_token`
        is a LIST (one entry per sequence), not an object -- a slot-state
        parser must index it rather than `.get()` it (recipes §serverapi)."""
        resp = await self._client.get(f"{self.base_url}/slots")
        resp.raise_for_status()
        return resp.json()

    async def completion(self, prompt: str, *, n_predict: int, temperature: float,
                          top_p: float, seed: int, stream: bool = True,
                          id_slot: int | None = None, cache_prompt: bool = True,
                          stop: list[str] | None = None) -> CompletionResult:
        """POST /completion, streamed SSE. Only the FINAL event carries
        `id_slot` and `timings`; intermediate events report `id_slot: -1`.
        Closing this call's connection (task cancellation) is a genuine
        server-side abort -- see the module docstring.

        `temperature`/`top_p`/`seed` are required, not defaulted: this is
        `cfg.scaffold.sampling.<role>`, real per-role config that must reach
        the server on every call -- defaulting them here would make it easy
        to silently ship near-greedy decoding no matter what config says."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "cache_prompt": cache_prompt,
            "stream": stream,
            "return_tokens": False,
        }
        if id_slot is not None:
            body["id_slot"] = id_slot
        if stop:
            body["stop"] = stop

        parts: list[str] = []
        final_event: dict[str, Any] | None = None
        t_first_byte = None
        async with self._client.stream(
            "POST", f"{self.base_url}/completion", json=body, timeout=self._timeout
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if t_first_byte is None:
                    t_first_byte = utc_now()
                ev = json.loads(line[len("data: "):])
                if ev.get("stop"):
                    final_event = ev
                    break
                if ev.get("content"):
                    parts.append(ev["content"])

        if final_event is None:
            raise DispatchError("/completion stream ended without a final event")

        timings = final_event.get("timings", {}) or {}
        return CompletionResult(
            content="".join(parts),
            tokens_in=timings.get("prompt_n", 0) + timings.get("cache_n", 0),
            tokens_out=timings.get("predicted_n", 0),
            cache_n=timings.get("cache_n", 0),
            slot_id=final_event.get("id_slot", -1),
            t_first_byte=t_first_byte,
            prompt_ms=timings.get("prompt_ms", 0.0),
            predicted_ms=timings.get("predicted_ms", 0.0),
            stop_type=final_event.get("stop_type", ""),
            truncated=bool(final_event.get("truncated")),
        )


# --------------------------------------------------------------------------- #
# LLMDispatcher -- semaphore, pre-flight admission, retries, step records.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class DispatchTarget:
    """Everything C4 needs to know about ONE role's server to dispatch a
    call and admit it: the client, the per-role generation budget, the slot
    capacity pre-flight rejects against (servers.<role>.ctx //
    servers.<role>.parallel), the per-role sampling params
    (cfg.scaffold.sampling.<role>) that must reach every /completion call,
    and the request-construction pair the chat template needs.

    `system_prefix` is §4's byte-identical leaf prefix -- the whole text of
    the sha256-pinned registry file, held here as ONE constant string for
    the target's lifetime so every call renders the same bytes. It is
    required, not defaulted: an empty default is precisely the S1 defect
    (finding F3, leaf calls with no system prompt at all) and would fail
    silently. `enable_thinking` is scaffold.<role>.enable_thinking."""

    client: ServerClient
    max_predict: int
    slot_capacity_tokens: int
    temperature: float
    top_p: float
    seed: int
    system_prefix: str
    enable_thinking: bool = False
    #: Control-token literals neutralised out of the USER message (I1). Filled
    #: lazily from this server's /props `chat_template` on the first call and
    #: cached for the target's lifetime -- deriving it beats hardcoding, and
    #: caching keeps that derivation off the per-call path.
    markers: tuple[str, ...] | None = None


def _new_step(call_id: str, retry_idx: int, role: str) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "retry_idx": retry_idx,
        "actor": Actor.LEAF if role == "leaf" else Actor.ROOT,
        "action_type": ActionType.LLM_CALL,
        "status": None,
        "error_detail": None,
        "tokens_in": None,
        "tokens_out": None,
        "tokens_cached": None,
        "slot_id": None,
        "t_dispatch": None,
        "t_first_byte": None,
        "t_end": None,
        "latency_queue_ms": None,
        "latency_prefill_ms": None,
        "latency_decode_ms": None,
    }


class LLMDispatcher:
    """The ONLY object in the system permitted to talk to a model server on
    the leaf's behalf. Owns the semaphore, the retry loop, and the pre-flight
    admission check; produces (but does not persist) one step dict per
    attempt, appended to `.steps` and handed to `on_step` if given."""

    def __init__(self, *, targets: dict[str, DispatchTarget], parallel: int,
                 retries: Retries, on_step: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._targets = targets
        self.semaphore = asyncio.Semaphore(parallel)
        self._retries = retries
        self._on_step = on_step
        self.steps: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg: Config, *,
                     on_step: Callable[[dict[str, Any]], None] | None = None) -> "LLMDispatcher":
        """Build the real, network-talking dispatcher from a validated
        Config. Host is always 127.0.0.1 -- this is a single-box runtime and
        ServerConfig carries no host field.

        ONLY a "leaf" target is built. Root traffic never goes through
        LLMDispatcher -- `rlm.rootclient.RootConversation` talks to a raw
        `ServerClient` directly with `cfg.scaffold.sampling.root` -- so a
        "root" DispatchTarget here would be dead code that real traffic
        never reaches, and would silently apply the leaf-sized semaphore to
        root calls if anything ever queried `role="root"` by mistake.
        `query(role="root", ...)` raises DispatchError instead."""
        max_predict = cfg.scaffold.budgets.max_predict
        leaf_sampling = cfg.scaffold.sampling.leaf
        # §4/§5: the leaf prefix comes from the sha256-pinned registry file,
        # never an inline string here (an inline prompt is unpinnable, so
        # config_snapshot would stop describing what actually ran). Read once,
        # at construction: re-reading per call could not change the bytes for
        # the better and could only introduce mid-episode prefix drift.
        leaf_prefix = cfg.prompt_registry().load().leaf_prefix()
        targets = {
            "leaf": DispatchTarget(
                client=ServerClient(f"http://127.0.0.1:{cfg.servers.leaf.port}",
                                     timeout=cfg.scaffold.retries.per_call_timeout_s),
                max_predict=max_predict.leaf,
                slot_capacity_tokens=cfg.servers.leaf.ctx // cfg.servers.leaf.parallel,
                temperature=leaf_sampling.temperature,
                top_p=leaf_sampling.top_p,
                seed=leaf_sampling.seed,
                system_prefix=leaf_prefix,
                enable_thinking=cfg.scaffold.leaf.enable_thinking,
            ),
        }
        return cls(targets=targets, parallel=cfg.servers.leaf.parallel,
                    retries=cfg.scaffold.retries, on_step=on_step)

    @property
    def last_step(self) -> dict[str, Any] | None:
        return self.steps[-1] if self.steps else None

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        """Token count via the TARGET server's own tokenizer.

        C2's chunker takes its counter injected precisely so it never imports
        an LLM client, and "chunk size measured in target-leaf tokens" (spec
        §5 C2) means this counter and no other -- a local approximation would
        make `chunk_size` advisory again and de-calibrate the §7 #2 sweep.
        Exposed here rather than reaching into `_targets` from the episode
        runner: the semaphore, the retry policy and the token counter are all
        C4's, and only C4 should know which client serves which role.

        Deliberately `add_special=False`, unlike the pre-flight: BOS is not
        part of a chunk BODY, and adding it here would bias every boundary the
        chunker's binary search finds by one token."""
        target = self._targets.get(role)
        if target is None:
            raise DispatchError(f"unknown dispatch role {role!r}")
        if not text:
            return 0
        return len(await target.client.tokenize(text))

    def _record(self, step: dict[str, Any]) -> None:
        self.steps.append(step)
        if self._on_step is not None:
            self._on_step(step)

    async def aclose(self) -> None:
        for target in self._targets.values():
            await target.client.aclose()

    async def _markers(self, target: DispatchTarget) -> tuple[str, ...]:
        """This target's control-marker set: derived once from the server's own
        `chat_template`, then cached on the target.

        A failed /props degrades to the hardcoded floor set rather than to no
        sanitisation at all, and never fails the call: the derivation widens
        the guarantee, it is not the guarantee."""
        if target.markers is None:
            template = ""
            try:
                template = (await target.client.props()).get("chat_template") or ""
            except Exception:  # noqa: BLE001 -- /props is optional here, see above
                template = ""
            target.markers = chat_control_markers(template)
        return target.markers

    async def query(self, prompt: str, *, role: str, call_id: str) -> str:
        target = self._targets.get(role)
        if target is None:
            raise DispatchError(f"unknown dispatch role {role!r}")

        # D14: render through the server's OWN chat template, never post the
        # caller's string raw -- that is base-model prompting against an
        # instruct-tuned model, and it made every leaf answer in the S1 gate
        # junk (`''` or a bare `<think></think>`; s1/RESULTS.md F3).
        #
        # I1 + §4: the system prefix is prepended HERE, scaffold-side, from a
        # constant the sandbox cannot reach, and the model's own text is
        # NEUTRALISED before it becomes an operand of the render. Applying a
        # chat template made "the model's string lands in the user message" an
        # insufficient defence: unescaped, a forged `<|im_start|>system` in it
        # renders as a second, model-authored, LAST-writer system turn whose
        # markers the tokenizer resolves to real control tokens (see
        # `neutralise_control_tokens`). Sanitisation runs on the USER message
        # only -- `target.system_prefix` is emitted verbatim, so §4's
        # byte-identical head is untouched.
        #
        # Exactly two messages, prefix first: the layout is
        # [system prefix][chunk][question] with the question last (the
        # caller composes the chunk+question user string), so a re-queried
        # chunk extends the slot's resident prefix instead of invalidating it
        # at token 0.
        user_message = neutralise_control_tokens(prompt, await self._markers(target))
        messages = [
            {"role": "system", "content": target.system_prefix},
            {"role": "user", "content": user_message},
        ]
        try:
            rendered = await target.client.apply_template(
                messages,
                chat_template_kwargs={"enable_thinking": target.enable_thinking},
            )
        except Exception as exc:  # noqa: BLE001 -- same class as a failed
            # pre-flight /tokenize below: nothing has been dispatched, so
            # there is no attempt to retry, only a call that cannot be built.
            step = _new_step(call_id, 0, role)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = f"pre-flight /apply-template failed: {exc}"
            self._record(step)
            raise DispatchError(
                f"pre-flight /apply-template failed for role={role!r}: {exc}") from exc

        # Pre-flight: token-count via the target server's /tokenize. A
        # prompt exceeding slot capacity is REJECTED without dispatch,
        # logged status=rejected, and never sent to /completion. It counts
        # the RENDERED string -- that is the string that would occupy the
        # slot, and admitting on the shorter unrendered one would under-count
        # by the whole system prefix plus the template's own markup --
        # and it counts it WITH `add_special`, because /completion does:
        # without that flag the pre-flight is short by exactly one BOS token
        # (measured 284/474/1274 vs served 285/475/1275), so a prompt of
        # exactly `slot_capacity_tokens` was admitted and then occupied cap+1.
        try:
            tokens = await target.client.tokenize(rendered, add_special=True)
        except Exception as exc:  # noqa: BLE001 -- server unreachable at
            # the pre-flight stage is a distinct failure from "prompt too
            # big"; there is nothing to retry a dispatch attempt against.
            step = _new_step(call_id, 0, role)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = f"pre-flight /tokenize failed: {exc}"
            self._record(step)
            raise DispatchError(f"pre-flight /tokenize failed for role={role!r}: {exc}") from exc

        if len(tokens) > target.slot_capacity_tokens:
            step = _new_step(call_id, 0, role)
            step["status"] = StepStatus.REJECTED
            step["error_detail"] = (
                f"prompt ({len(tokens)} tokens) exceeds slot capacity "
                f"({target.slot_capacity_tokens} tokens)"
            )
            self._record(step)
            raise DispatchError(step["error_detail"])

        last_exc: Exception | None = None
        for attempt in range(self._retries.max_attempts):
            step = _new_step(call_id, attempt, role)
            t_dispatch = utc_now()
            step["t_dispatch"] = t_dispatch
            try:
                # Held only around THIS attempt, not the backoff sleep
                # below: a failed attempt pins no id_slot, so there is no
                # affinity to preserve, and holding the semaphore across a
                # 1-4s backoff would starve other queued leaf calls for no
                # benefit (up to 8 idle slots under a 32-call fan-out).
                async with self.semaphore:
                    # The EXACT string /apply-template returned, verbatim on
                    # every attempt -- re-rendering or re-assembling it here
                    # would break byte-identity for a reason no test on
                    # message *content* could ever catch.
                    result = await target.client.completion(
                        rendered, n_predict=target.max_predict,
                        temperature=target.temperature, top_p=target.top_p,
                        seed=target.seed, stream=True)
            except asyncio.CancelledError:
                # A genuine abort: closing the stream already freed the
                # slot server-side. No final event exists, so slot_id/
                # timings/tokens stay NULL on this step (recipes
                # §serverapi integration notes).
                step["status"] = StepStatus.CANCELLED
                step["t_end"] = utc_now()
                self._record(step)
                raise
            except Exception as exc:  # noqa: BLE001 -- network/HTTP/
                # protocol failures are all retryable the same way.
                last_exc = exc
                step["status"] = StepStatus.ERROR
                step["error_detail"] = str(exc)
                step["t_end"] = utc_now()
                self._record(step)
                if attempt < self._retries.max_attempts - 1:
                    backoff = self._retries.backoff_s[
                        min(attempt, len(self._retries.backoff_s) - 1)]
                    await asyncio.sleep(backoff)
                    continue
                raise DispatchError(
                    f"dispatch failed after {self._retries.max_attempts} attempts "
                    f"(role={role!r}, call_id={call_id!r}): {exc}") from exc
            else:
                t_end = utc_now()
                queue_ms = None
                if result.t_first_byte is not None:
                    queue_ms = (
                        (result.t_first_byte - t_dispatch).total_seconds() * 1000.0
                        - result.prompt_ms
                    )
                step.update(
                    status=StepStatus.OK,
                    t_first_byte=result.t_first_byte,
                    t_end=t_end,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    tokens_cached=result.cache_n,
                    slot_id=result.slot_id,
                    latency_queue_ms=queue_ms,
                    latency_prefill_ms=result.prompt_ms,
                    latency_decode_ms=result.predicted_ms,
                )
                self._record(step)
                return result.content
        # unreachable: the loop above always returns or raises.
        raise DispatchError(f"exhausted retries with no result: {last_exc}")


# --------------------------------------------------------------------------- #
# MockDispatcher -- same interface, canned by (role, sha256(prompt)). No
# network. For dry runs and for tests that must not construct an httpx
# client.
# --------------------------------------------------------------------------- #


class MockDispatcher:
    """Same interface as LLMDispatcher (`.query`, `.semaphore`), answering
    from a fixed `{f"{role}:{sha256(prompt).hexdigest()}": response}` table
    instead of a live server."""

    def __init__(self, fixtures: dict[str, str], *, parallel: int = 1024) -> None:
        self._fixtures = dict(fixtures)
        self.semaphore = asyncio.Semaphore(parallel)
        self.steps: list[dict[str, Any]] = []

    @property
    def last_step(self) -> dict[str, Any] | None:
        return self.steps[-1] if self.steps else None

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        """A dry run has no server to ask, so this is a stated approximation,
        not a measurement: ~4 chars per token, monotonic in prefix length
        (which C2's binary search requires) and deterministic (which the
        chunker's determinism contract requires). Dry-run chunk boundaries
        therefore do NOT match a real run's, and no dry-run episode may be
        scored -- which is already true for other reasons (`dry_run=true`)."""
        return (len(text) + 3) // 4

    async def query(self, prompt: str, *, role: str, call_id: str) -> str:
        key = f"{role}:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}"
        step = _new_step(call_id, 0, role)
        async with self.semaphore:
            if key not in self._fixtures:
                step["status"] = StepStatus.ERROR
                step["error_detail"] = f"MockDispatcher: no fixture for {key}"
                self.steps.append(step)
                raise DispatchError(step["error_detail"])
            step["status"] = StepStatus.OK
            self.steps.append(step)
            return self._fixtures[key]
