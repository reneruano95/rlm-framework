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
`PromptRegistry.leaf_prefix()` -- as the system message, and the model's
(chunk, question) pair composed into the user message here, not in the
sandbox. Skipping the template is base-model prompting against an
instruct-tuned model, which is what made every leaf answer in the S1 gate
junk (s1/RESULTS.md F3).

WHAT KEEPS THE PREFIX THE PREFIX, stated exactly. It is NOT "model text lands
in the user message, so its ChatML markers are just user content" -- that
claim was made when the template was introduced and it is FALSE, measured: a
forged `<|im_start|>system` in the user message renders as a second,
model-authored, LAST-writer system turn whose markers the tokenizer resolves
to genuine control-token ids. What holds is (a) the prefix is prepended here,
from a constant the sandbox cannot reach, and (b) every control-token literal
in the model's own text is neutralised before it becomes an operand of the
render (`neutralise_control_tokens`, below). Do not restore the old claim.

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

The semaphore is `asyncio.Semaphore(cfg.scaffold.dispatch_concurrency)`, owned
here -- NOT `servers.leaf.parallel`, which since v0.2.6 sizes the never-reuse
slot POOL (how many windows one leaf process serves before it is rotated, a
memory-derived number: 62.8125 MiB of recurrent state per slot,
`s2/R13-slotcount.md`) rather than how many calls may be in flight.
Nothing the model runs may resize it, choose a server, or change a port. It
is held only around each individual dispatch attempt, not across a retry's
backoff sleep -- this window's slot is its own and can be handed to nothing
else (see SLOT DISCIPLINE below), so releasing the semaphore during a backoff
costs the call nothing, while holding it across a 1-4s sleep would starve
other queued leaf calls for no benefit.

SLOT DISCIPLINE (v0.2.6, R13) -- the correctness contract of this module.
The leaf returns content from documents previously held on the same slot.
Measured under a paired control in ONE process with byte-identical prompts at
the same moment: shared pinned slot 24/54 leaked, virgin slot 0/54 (Fisher
p = 4.4e-9, `s2/R13.md` §1). It survives a cold full re-prefill
(`timings.cache_n == 0`) and survives `action=erase` returning a truthful
`n_erased`, and a verified full-attention control model leaked MORE than the
hybrid -- so it is neither the prompt cache nor recurrent state, and no
configuration flag suppresses it. `cache_prompt: false` (15/18) and
`--parallel 1` (4/18, which makes reuse mandatory) both leak and are NOT
mitigations. The only measured-clean configurations are one process per
document and ONE NEVER-REUSED SLOT PER WINDOW, and the second costs a slot
rather than a process (13.4 s per 200K corpus against 29 min of redundant
model loading, `s2/R13-mitigations.md` §8.2).

So C4 owns slot assignment: `SlotPool`, sized by the server's `--parallel`,
hands each window one slot that no other window will ever get, and REFUSES
when the pool is empty (`SlotPoolExhausted`) so the caller rotates the
server instead of wrapping around into the defect.

ROTATION, AND WHAT C4 DOES *NOT* OWN (v0.2.6). A pool of `--parallel` slots
dies at window `--parallel` while a 200K corpus needs 424 windows, so the
mitigation is inert without a rotation. §5 C4 permits one, narrowly: on pool
exhaustion only, never on an error -- restarting a FAILED server would mask
the fault the trace exists to record, while rotating a HEALTHY one on a
deterministic, scaffold-owned schedule is a resource-lifecycle operation.
This module still has no code path that starts or stops a process: it owns
the three primitives that make someone else's rotation safe --
`quiesce()` (no call may be mid-flight while the process it is talking to is
replaced), `rotate_pool()` (a new process means a new pool; carrying old
assignments onto it would be R13 reintroduced by R13's own mitigation), and
`resume()`. The episode runner drives them and a process manager injected by
the CLI owns the process (`rlm.serverproc`). Both questions about the
same window go to that window's own slot -- same-document reuse is warm and
measured clean (0/72), and it is the one performance lever R13 leaves intact.
Every reply's `id_slot` is ASSERTED against the requested one, because an
out-of-range request is silently reassigned with HTTP 200 (asked 200, got 72),
which would leave the scaffold certain it held a virgin slot while sharing a
used one. And every answer is run through R13's foreign-string detector
(`rlm.leakcheck`), whose verdict lands on the step as `leak_detected` /
`leak_detail`.

None of this makes the leaf clean: 138 virgin-slot calls with zero leaks give
a 95% upper bound of 2.2%, and a 200K episode is ~848 leaf calls, so the
evidence permits roughly 11 contaminated answers per episode. The phrase
"leak-free" is not available to this module or anything reporting on it.

Server death: retries exhausting against a dead server produce a step
`status=error` and a raised DispatchError; the caller turns that into
`episode.outcome=error, outcome_reason=server_unreachable`. This module
never restarts a server mid-episode -- it has no code path that could.

Sampling: `cfg.scaffold.sampling.<role>` (temperature, top_p, seed) is real,
non-defaulted config -- §6 records it in config_snapshot as what actually
ran, and the benchmark's seed discipline depends on the seed reaching the
server. `from_config()` threads it into each role's `DispatchTarget`, and
`query()` passes it on every /completion call; nothing here defaults to
near-greedy sampling.

THE SEED IS ALSO PER CALL (`query(seed=...)`), because a dispatcher outlives a
seed. §8 varies the seed of the WHOLE system across three replicates while one
bench run holds one leaf dispatcher for all of them, so a construction-time-only
seed would decode every replicate at the first seed while `config_snapshot`
recorded three -- three draws of one leaf, reported as three seeds. The
override is what makes the recorded value true; `None` keeps the target's.

`from_config()` builds ONLY a "leaf" target: root
traffic never goes through LLMDispatcher (see `rlm.rootclient.
RootConversation`, which talks to a raw `ServerClient` with its own
`cfg.scaffold.sampling.root`), so a "root" DispatchTarget here would be
dead code that could silently apply the leaf-sized semaphore to root
calls if ever queried by mistake -- `query(role="root", ...)` raises
DispatchError instead.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from rlm.config import Config, Retries
from rlm.envelope import ParseResult as EnvelopeParse
from rlm.envelope import parse as envelope_parse
from rlm.envelope import payload as envelope_payload
from rlm.errors import (
    ActionType,
    Actor,
    DispatchError,
    EnvelopeParseError,
    PreflightFailed,
    PrefixDrift,
    SlotMismatch,
    SlotPoolExhausted,
    StepStatus,
)
from rlm.leakcheck import NOT_CHECKED, ChunkIndex, LeakVerdict
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
# §4's prompt layout, composed scaffold-side (fix round 2).
#
# MEASURED: a chunk-first re-query reported `cache_n=546`; the SAME chunk with
# the question moved to the front reported `cache_n=0`, twice. As built,
# `llm_query` took ONE opaque string which the dispatcher dropped whole into a
# user message -- so §4's `[prefix][chunk][question]` layout, and therefore S2
# gate (b)'s >80% reuse target, rested entirely on the root model's formatting
# discipline. That makes the gate a test of prompt compliance, not of the
# scaffold. The scaffold now composes the user message itself, exactly as it
# already composes the system prefix.
# --------------------------------------------------------------------------- #

#: How many times ONE call may lose its pre-flight to a rotation and re-run it.
#: A rotation frees a whole pool, so a call needs a second re-run only when
#: other calls in the same wave took every slot first; needing 16 means a
#: fan-out wider than 16 x --parallel windows, which is a structural problem
#: the wall clock should not have to discover slowly. Mirrors
#: `rlm.episode.MAX_ROTATIONS_PER_CALL`, which bounds the same thing one layer
#: up (a call that lost its SLOT rather than its pre-flight).
_MAX_PREFLIGHT_ROTATIONS = 16

#: Which form a call used. `steps` needs no column for it (§6): it rides in the
#: leaf request blob's `meta` stream, so an S2 gate can score only the calls
#: that actually supplied `chunk=` rather than crediting the scaffold for a
#: layout the model happened to type correctly.
LAYOUT_CHUNK_QUESTION = "chunk_question"
LAYOUT_QUESTION_ONLY = "question_only"


def compose_leaf_user(question: str, chunk: str | None) -> str:
    """§4's `[chunk][question]` user segment, question LAST.

    One function, so the string C4 sends, the string C5 admits against and the
    string C6 logs as `action_payload` cannot drift apart. `chunk=None` is the
    single-string form: the caller composed it itself and the scaffold has no
    way to know where the chunk ended -- kept working because the paper
    harness's own call site (and every S1 prompt) uses it."""
    if chunk is None:
        return question
    return f"{chunk}\n\n{question}"


# --------------------------------------------------------------------------- #
# R13 slot discipline: one never-reused slot per window (spec §5 C4, §10 R13).
# --------------------------------------------------------------------------- #


def window_key(chunk: str | None, call_id: str) -> str:
    """Which WINDOW a call is about, for slot-allocation purposes.

    Identity is the chunk's bytes: two calls carrying the same chunk are two
    questions about one window and share its slot (warm, measured clean --
    `s2/R13-mitigations.md` §4.3). A call with no `chunk=` is the single-string
    form, where C4 cannot see where the document ends and therefore cannot
    prove two such calls carry the same document; the safe reading is that
    they do not, so each gets a window of its own. That costs slots on the
    legacy call form, which is the correct direction to be wrong in."""
    if chunk is None:
        return f"call:{call_id}"
    return f"chunk:{hashlib.sha256(chunk.encode('utf-8')).hexdigest()}"


class SlotPool:
    """The `--parallel`-sized pool of leaf slots, handed out once each.

    Two invariants, and they are the entire R13 prevention:

      * a slot index is handed out to AT MOST ONE window for this pool's
        lifetime (which is the lifetime of the leaf server process it
        describes -- a restarted server means a new pool);
      * every index handed out is in `[0, size)`, because an out-of-range
        `id_slot` is silently reassigned by the server onto a slot that has
        held other documents.

    Exhaustion RAISES. Wrapping around would hand document B a slot that held
    document A while the scaffold believed otherwise, which is R13 exactly;
    the caller's answer to `SlotPoolExhausted` is to restart the leaf server
    (spec §5 C4) -- C4 itself has no code path that restarts anything.
    """

    __slots__ = ("size", "_by_window", "_next", "_answered", "_failed")

    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError(f"slot pool size must be >= 1, got {size}")
        self.size = size
        self._by_window: dict[str, int] = {}
        self._next = 0
        # WHY each slot was consumed, not merely THAT it was. §5 C4 permits a
        # rotation on pool exhaustion and forbids restarting a FAILED server,
        # and without this the two are indistinguishable: a pool drained by
        # three consecutive dispatch failures raises exactly the same
        # `SlotPoolExhausted` as a pool drained by three answered windows, so
        # the runner relaunched a failing server and logged it as a planned
        # rotation.
        self._answered: set[str] = set()
        self._failed: set[str] = set()

    @property
    def remaining(self) -> int:
        """Virgin slots left, i.e. how many NEW windows this leaf process can
        still serve before it must be restarted."""
        return self.size - self._next

    @property
    def restart_required(self) -> bool:
        return self.remaining <= 0

    @property
    def assigned(self) -> dict[str, int]:
        return dict(self._by_window)

    @property
    def answered(self) -> int:
        """Windows on this pool that produced an answer."""
        return len(self._answered)

    @property
    def failed(self) -> int:
        """Windows on this pool whose call ended in an error."""
        return len(self._failed)

    @property
    def error_drained(self) -> bool:
        """The pool is spent and NOT ONE window on it was ever answered.

        This is the "the server is failing" shape, and it is the case §5 C4's
        original rule is about: exhaustion caused by repeated failures is not
        healthy-pool exhaustion, and rotating on it relaunches a FAILED server
        while the lifecycle log calls it a planned rotation -- masking exactly
        the fault the trace exists to record.

        A MIXED drain (some windows answered, some failed) is deliberately NOT
        error-drained: a server that answered a window on this generation has
        demonstrated it can serve, and refusing to rotate there would turn a
        single transient failure into a dead episode. What is refused is the
        generation that served nothing at all.
        """
        return self.restart_required and not self._answered

    def mark_answered(self, window: str) -> None:
        self._answered.add(window)
        self._failed.discard(window)

    def mark_failed(self, window: str) -> None:
        """Only when the window has never been answered: a second question
        about an already-answered window failing does not make the generation
        an error-drained one."""
        if window not in self._answered:
            self._failed.add(window)

    def acquire(self, window: str) -> int:
        """This window's slot: its existing one, or the next virgin one."""
        slot = self._by_window.get(window)
        if slot is not None:
            return slot
        if self._next >= self.size:
            raise SlotPoolExhausted(
                f"leaf slot pool exhausted: all {self.size} slots have held a "
                f"window, and reusing one is R13's defect. The leaf server "
                f"must be restarted before another window is dispatched "
                f"(--parallel {self.size})")
        slot = self._next
        self._next += 1
        self._by_window[window] = slot
        return slot


# --------------------------------------------------------------------------- #
# ServerClient -- thin, honest wrapper over the llama-server HTTP surface
# (recipes §serverapi). No retries, no semaphore, no step logging: those are
# LLMDispatcher's job. This class only knows how to talk to ONE server.
# --------------------------------------------------------------------------- #


def predicted_reuse(n_resident: int, lcp: int, ub: int) -> int:
    """How many tokens the server WILL reuse -- the measured law behind §7 #3.

    `n_resident` is the token length of the prompt that last occupied the slot,
    `lcp` the longest common token prefix between the incoming prompt and that
    one, `ub` the server's `-ub`. Both inputs are numbers the scaffold already
    holds: C4's pre-flight tokenizes every prompt anyway.

    This replaces every threshold §7 #3 used to state against `timings.cache_n`
    directly. `cache_n` is honest -- over 373 calls it never once exceeded what
    the process had actually seen -- but the spec's truth model was wrong twice:
    reuse is quantised to ONE rollback point per slot at `n_resident - ub - 4`
    (so `cache_n` under-reported the true shared prefix on 111/373 calls, by up
    to 497 tokens), and b10375's HOST prompt cache restores an idle slot's state
    onto a DIFFERENT slot (so it over-reported against §4's per-slot model by up
    to +961). Scored against this function, `cache_n` is EXACT on 239/239 calls
    at `-ub` 512 and 128 -- `s2/CACHE-INSTRUMENT.md` §3.

    TWO PRECONDITIONS, both load-bearing:

    * `--cache-ram 0` (or `--no-cache-idle-slots`) must be in the launch line,
      or cross-slot restore adds reuse this model cannot see and `cache_n`
      stops being a function of the prompts at all. `config.yaml` pins it.
    * `ub + 4` is a property of a BUILD and a FLAG (R11), measured at 512 and
      at 128 (gaps 516 and 132). Re-measure it whenever `-ub` or the llama.cpp
      build changes; `s2/run_cache_instrument.py` is the regression test.

    The `+ 4` is the generation-prompt markup -- the same 4 tokens a
    byte-identical re-send still re-evaluates (962 reused of 966).

    Not wired into the dispatch path: its callers are the S2 gate runner and
    `rlm replay`, which have both the resident prompt and the incoming one.
    """
    if n_resident <= 0 or lcp <= 0:
        return 0
    if lcp >= n_resident:                 # byte-identical re-send
        return n_resident - 4
    if lcp >= n_resident - 4:             # CONTINUATION: the prompt extends the slot
        return lcp
    rollback = n_resident - ub - 4        # DIVERGENCE: the single rollback point
    if lcp >= rollback:
        return max(rollback, 0)
    return 0                              # the shared prefix is NOT available


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

    async def health(self) -> bool:
        """GET /health as a POLL, not an assertion: True only on a 200.

        Returns False rather than raising for every other outcome, because the
        two interesting ones are indistinguishable from faults and both mean
        "not yet" -- 503 while the model loads, and a refused connection in the
        seconds after a slot-pool rotation kills the old process. A rotation's
        readiness wait (`rlm.serverproc`) is built on exactly this, and a
        rotation that treated a refused connection as an error would fail every
        time it worked.
        """
        try:
            resp = await self._client.get(f"{self.base_url}/health")
        except Exception:  # noqa: BLE001 -- see the docstring: not-yet, not fault
            return False
        return resp.status_code == 200

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
    #: The RENDERED system head's token length -- the whole prefix as the slot
    #: actually holds it (template markup and BOS included). Measured once, on
    #: the first render, and kept: recomputing it per call could not change it
    #: (the prefix is one constant string for this target's lifetime).
    #: §7 #3's gate (a1) asserts THIS number is stable (311 on the pinned
    #: prefix, measured across 6 server launches and 373 calls) alongside the
    #: prefix sha256 -- it is no longer a floor on `tokens_cached`. That
    #: comparison is retired: `cache_n >= prefix_len` on a first-sight chunk is
    #: unmeetable at every window and every `-ub` bar a one-token coincidence
    #: (it needs `n_resident <= 315 + ub`), and under R13's never-reuse policy
    #: production has no warm-slot first-sight call at all -- `cache_n` was 0 on
    #: 236/236. The cache-side residue that CAN fail is `predicted_reuse`.
    prefix_tokens: int | None = None
    #: `sha256(rendered_head)` -- the OTHER half of §7 #3's gate (a1), and the
    #: half that actually detects drift: two different prefixes of equal token
    #: length compare equal on length alone. Pinned on the first render and
    #: compared on EVERY call thereafter. Comparing costs no round trip, which
    #: is why it can be always-on where re-tokenizing could not: the head is
    #: byte-identical by construction, so a matching hash implies a matching
    #: token count under this target's (fixed) tokenizer, and re-deriving that
    #: count per call would add one /tokenize per leaf call -- up to
    #: `max_subcalls` of them per episode -- to learn nothing new.
    prefix_sha256: str | None = None
    #: Does this target's prefix ask for the JSON envelope, and must C4
    #: therefore parse and validate one scaffold-side (spec §5, `rlm.envelope`)?
    #: Off by default: the envelope is opt-in, decided by the S2 A/B
    #: (`s2/REFUSAL-AB.md`), and every measurement recorded before it exists was
    #: taken with plain-text answers.
    envelope: bool = False


def _new_step(call_id: str, retry_idx: int, role: str, *,
               layout: str | None = None, rendered: str | None = None,
               prefix_tokens: int | None = None,
               prefix_sha256: str | None = None) -> dict[str, Any]:
    """One attempt's step record.

    `root_view_hash` is a `steps` column (§6's state-rule instrument, defined
    for the ROOT and equally meaningful here); `rendered`, `layout` and
    `prefix_tokens` are not, and need not be -- the caller (the episode runner)
    turns them into the step's `root_request_ref` blob, and `rlm.trace`
    silently drops anything not in STEP_COLS.
    """
    return {
        "call_id": call_id,
        "retry_idx": retry_idx,
        "layout": layout,
        "rendered": rendered,
        "prefix_tokens": prefix_tokens,
        "prefix_sha256": prefix_sha256,
        "root_view_hash": (None if rendered is None else
                           hashlib.sha256(rendered.encode("utf-8")).hexdigest()),
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
        # R13's detector (rlm.leakcheck), filled on every answered leaf call.
        # None means NOT CHECKED -- never "checked and clean".
        "leak_detected": None,
        "leak_detail": None,
        # The model's output VERBATIM on every attempt that produced one. Not a
        # `steps` column (§6 stores it as the observation blob); it is here so
        # an attempt REJECTED for a malformed envelope still carries the text it
        # was rejected for -- otherwise a run reports a count of envelope
        # failures with nothing to audit them against.
        "response_text": None,
    }


class LLMDispatcher:
    """The ONLY object in the system permitted to talk to a model server on
    the leaf's behalf. Owns the semaphore, the retry loop, and the pre-flight
    admission check; produces (but does not persist) one step dict per
    attempt, appended to `.steps` and handed to `on_step` if given."""

    def __init__(self, *, targets: dict[str, DispatchTarget], parallel: int,
                 retries: Retries, on_step: Callable[[dict[str, Any]], None] | None = None,
                 slots: SlotPool | None = None) -> None:
        self._targets = targets
        self.semaphore = asyncio.Semaphore(parallel)
        self._retries = retries
        self._on_step = on_step
        self.steps: list[dict[str, Any]] = []
        # R13: one never-reused slot per window, from a pool the size of the
        # server's --parallel -- which is NOT `parallel` above (that is the
        # dispatch concurrency). `from_config` is the only production
        # construction and passes both explicitly; the fallback here exists so
        # a bare test dispatcher still gets a pool.
        self.slots = slots if slots is not None else SlotPool(parallel)
        self._chunk_index: ChunkIndex | None = None
        # Rotation plumbing (v0.2.6). `_gate` is open except while a rotation
        # is in progress; a call waits on it in the same step in which it takes
        # its slot, so no call can be holding an assignment from a pool that is
        # about to be replaced. `_idle` is what `quiesce()` waits on.
        self._gate = asyncio.Event()
        self._gate.set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._in_flight = 0
        self._pool_generation = 0
        # How many steps each call_id has already recorded, so a re-dispatch of
        # one logical call after a rotation continues its `retry_idx` sequence
        # instead of colliding with the refusal that triggered the rotation.
        self._recorded: dict[str, int] = {}

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
        use_envelope = cfg.scaffold.leaf_envelope.enabled
        leaf_prefix = cfg.prompt_registry().load().render_leaf(envelope=use_envelope)
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
                envelope=use_envelope,
            ),
        }
        # R13/§5 C4: the pool is sized by the server's --parallel, which is
        # therefore also the number of WINDOWS one leaf process can serve
        # before it must be restarted. `slot_policy` is config-declared and
        # has exactly one supported value today, so the choice is explicit
        # and greppable rather than implied by this line.
        if cfg.servers.leaf.slot_policy != "never_reuse":  # pragma: no cover
            raise DispatchError(
                f"servers.leaf.slot_policy={cfg.servers.leaf.slot_policy!r} is "
                "not supported; R13 has exactly one measured-clean policy")
        # TWO different numbers, and they stopped being the same one at v0.2.6.
        # The SEMAPHORE is `scaffold.dispatch_concurrency`: how many leaf calls
        # may be in flight at once, tuned against S0's flat aggregate prefill.
        # The POOL is `servers.leaf.parallel`: how many WINDOWS this process can
        # serve before it must be rotated, sized by the measured 62.8125 MiB of
        # per-slot recurrent state (`s2/R13-slotcount.md`). Passing the pool
        # size as the semaphore would put 128 calls on the wire because the
        # memory bill happened to allow 128 slots.
        return cls(targets=targets, parallel=cfg.scaffold.dispatch_concurrency,
                    retries=cfg.scaffold.retries, on_step=on_step,
                    slots=SlotPool(cfg.servers.leaf.parallel))

    @property
    def last_step(self) -> dict[str, Any] | None:
        return self.steps[-1] if self.steps else None

    @property
    def restart_required(self) -> bool:
        """True once the slot pool has no virgin slot left for a new window.
        The signal exists so exhaustion is handled by rotating the leaf,
        never by reusing a slot (R13)."""
        return self.slots.restart_required

    @property
    def pool_error_drained(self) -> bool:
        """True when the pool is spent and NOT ONE window on it was answered.

        The caller's question is not "may I have more slots?" but "is this
        server healthy enough for §5 C4 to let me relaunch it?". Exhaustion
        reached through failures is not healthy-pool exhaustion, and rotating
        on it restarts a FAILED server -- which §5 C4 forbids, and which is the
        distinction that made rotation permissible at all.
        """
        return self.slots.error_drained

    # -- rotation primitives (spec §5 C4, v0.2.6) ---------------------------- #

    @property
    def in_flight(self) -> int:
        """Calls that hold a slot on the CURRENT leaf process right now."""
        return self._in_flight

    @property
    def pool_generation(self) -> int:
        """How many pools this dispatcher has had; equivalently, how many leaf
        processes it has talked to. Stamped nowhere by C4 -- exposed so the
        runner can tell "someone already rotated while I was queued" from "I am
        the one who has to rotate"."""
        return self._pool_generation

    async def quiesce(self) -> None:
        """Close the gate and wait until no call is talking to the leaf.

        After this returns, every call that was talking to the leaf has
        finished -- pre-flight round trips included -- and every new one is
        parked BEFORE it sends its first byte, which is the precondition for
        replacing the process underneath. Deliberately without a timeout of its
        own: a call can legitimately be mid-retry for
        `max_attempts x per_call_timeout_s`, and the thing that must bound this
        wait is C5's wall clock (whose budget the rotation is spending), not a
        second, quieter deadline here.

        THE WAIT IS EXCEPTION-SAFE, and that is not decoration. Closing the
        gate and then being cancelled -- a C5 budget kill, an operator Ctrl-C,
        a task group unwinding -- would otherwise leave the gate closed with
        nobody left to reopen it, and every later `query()` would park on it
        forever. That is a hang, not a refusal: it produces no step, no
        outcome and no trace row. So any failure to complete the wait reopens
        the gate on the way out; the caller that wanted a rotation takes its
        error, and the traffic it was quiescing carries on against the process
        that is still there.
        """
        self._gate.clear()
        try:
            await self._idle.wait()
        except BaseException:
            self._gate.set()
            raise

    @contextlib.asynccontextmanager
    async def rotating(self) -> "AsyncIterator[None]":
        """quiesce -> (caller replaces the process) -> resume, reopen guaranteed.

        The whole §5 C4 sequence a caller must not get wrong, with the reopen
        in a `finally`: a rotation that raises (a restart that failed, a
        handshake that refused the new process, a cancellation) must still
        leave parked calls able to take the refusal they would have taken
        anyway. `rotate_pool()` stays the caller's to invoke, inside the block,
        because only the caller knows whether the process was actually
        replaced -- adopting a fresh pool for a process that never restarted is
        R13 reintroduced by R13's own mitigation.
        """
        await self.quiesce()
        try:
            yield
        finally:
            self.resume()

    def rotate_pool(self) -> None:
        """Adopt a fresh pool: a new process has every slot virgin again.

        Refuses while a call is in flight. Replacing the pool underneath a live
        call would hand that call's slot index straight back out to a different
        window -- R13's defect, reintroduced by R13's own mitigation, and
        invisible because the scaffold would believe both windows held virgin
        slots.
        """
        if self._in_flight:
            raise DispatchError(
                f"cannot rotate the slot pool with {self._in_flight} call(s) in "
                "flight; quiesce() first (spec §5 C4)")
        self.slots = SlotPool(self.slots.size)
        self._pool_generation += 1

    def resume(self) -> None:
        """Reopen the gate. Safe to call on any path, including a rotation that
        failed: parked calls then take the refusal (or the exhaustion) they
        would have taken anyway, instead of hanging on a gate nobody will
        reopen."""
        self._gate.set()

    def set_corpus(self, chunks: Any) -> None:
        """Hand C4 the corpus C2 chunked, ONCE per episode, so R13's detector
        can run on every answer (spec §5 C4).

        Without it the detector reports NOT CHECKED rather than clean: a leak
        verdict with nothing to compare against is not a verdict."""
        self._chunk_index = ChunkIndex.from_chunks(chunks) if chunks else None

    @property
    def chunk_index(self) -> ChunkIndex | None:
        return self._chunk_index

    def leak_verdict(self, answer: str, sent: str) -> LeakVerdict:
        """R13's foreign-string check for one answer: identifier-shaped tokens
        that are absent from what was sent (chunk AND question) and present in
        another chunk. Zero model calls; see `rlm.leakcheck` for its two
        stated limits and for why a clean verdict is evidence rather than a
        certificate."""
        if self._chunk_index is None:
            return NOT_CHECKED
        return self._chunk_index.foreign(answer, sent=sent)

    def prefix_sha256(self, role: str = "leaf") -> str | None:
        """This role's pinned `sha256(rendered_head)`, or None before the first
        call has rendered anything.

        The other half of gate (a1), and the enforcing half: `prefix_tokens` is
        measured once and could not detect a prefix that changed to a different
        string of the same length, whereas this is compared on every call and
        raises `PrefixDrift`. Exposed for the gate runner and for
        `config_snapshot`, which records it as the run's pinned value."""
        target = self._targets.get(role)
        return None if target is None else target.prefix_sha256

    def prefix_tokens(self, role: str = "leaf") -> int | None:
        """This role's rendered system-head length in tokens, or None before
        the first call has rendered anything.

        Exposed for a gate runner: §7 #3's gate (a1) asserts this number is
        CONSTANT (311 on the pinned prefix) beside the prefix sha256 -- that
        pair is the whole of R3's drift detector, and it fails the moment a
        byte moves. It is no longer compared against `tokens_cached`: that
        inequality is unmeetable on this stack (see `Target.prefix_tokens`),
        and the assertion that can actually fail is `predicted_reuse`. Every
        step also carries this number (`prefix_tokens`), so a post-hoc reader
        gets it next to the `tokens_cached` it is judging."""
        target = self._targets.get(role)
        if target is None:
            raise DispatchError(f"unknown dispatch role {role!r}")
        return target.prefix_tokens

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
        chunker's binary search finds by one token.

        Gated and counted like every other leaf round trip (`_admitted`): this
        is C4's own `/tokenize` call and it sits on the critical path twice --
        C5 admits every sub-call against it, and C2's chunker binary-searches
        every window boundary through it. A rotation that replaced the process
        underneath one of those would kill C2's chunking or C5's admission with
        a bare connection error, outside any retry loop.
        """
        target = self._targets.get(role)
        if target is None:
            raise DispatchError(f"unknown dispatch role {role!r}")
        if not text:
            return 0
        async with self._admitted():
            return len(await target.client.tokenize(text))

    def _record(self, step: dict[str, Any]) -> None:
        self.steps.append(step)
        call_id = step.get("call_id")
        if call_id is not None:
            self._recorded[call_id] = self._recorded.get(call_id, 0) + 1
        if self._on_step is not None:
            self._on_step(step)

    def _retry_base(self, call_id: str) -> int:
        """The `retry_idx` this dispatch's first attempt gets.

        Normally 0. It is non-zero only when this logical call has already
        recorded attempts -- which happens when a pool exhaustion refused it,
        the runner rotated the leaf, and the SAME call (it counts once against
        `max_subcalls`) is dispatched again. Continuing the sequence keeps one
        trace row per attempt: the runner writes rows keyed by
        (call_id, retry_idx), so restarting at 0 would drop the attempt that
        actually answered in favour of the refusal that preceded it.
        """
        return self._recorded.get(call_id, 0)

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

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None,
                     seed: int | None = None,
                     n_predict: int | None = None) -> "str | dict[str, Any]":
        """Dispatch one leaf call. Returns the answer STRING, or -- when this
        target asks for the JSON envelope -- the parsed envelope as a dict
        (`rlm.envelope.payload`). Off by default, so every existing caller and
        every measurement recorded before the S2 A/B keeps the string. `prompt` is the QUESTION; `chunk`, when
        given, is the excerpt it is about, and C4 composes
        `[system prefix][chunk][question]` from the two (§4). `chunk=None`
        keeps the single-string behaviour.

        THE ROTATION GATE SITS AHEAD OF EVERYTHING, pre-flight included, and
        the pre-flight is inside the in-flight accounting -- see `_admitted`.
        A call makes three leaf round trips before it asks for a slot, and
        while the gate sat after them a rotation could take the process away
        mid-pre-flight: `quiesce()` counted only slot-holders, so it returned
        while those requests were on the wire, and a pre-flight has no retry
        loop. Measured on a 6-window concurrent map with a process manager that
        genuinely removes the server: 4 of 6 windows dispatched, two lost with
        no step at all, and the episode still reported SUCCESS.

        A LOST WINDOW IS WORSE THAN A FAILED CALL HERE. The aggregation
        strategy template prescribes a full-coverage MAP over `chunks` -- one
        `asyncio.gather`, no early stopping -- for a benchmark category (§8)
        whose entire purpose is to force coverage and punish sampling. A
        partial map still prints an answer, so the failure surfaces as a wrong
        result in the arm being measured, not as an error anyone can see.

        `seed` OVERRIDES the construction-time `target.seed` FOR THIS CALL, and
        exists because a dispatcher outlives a seed. §8 runs the whole system
        at three seeds, and a bench run builds ONE leaf dispatcher for all of
        them (rebuilding per seed would hand a fresh, virgin-pool view of an
        already-used server to C4 -- an R13 hazard -- and relaunching the leaf
        at every seed boundary costs 90 extra relaunches). Without a per-call
        seed the leaf would decode at `sampling.leaf.seed` as it stood when the
        dispatcher was built, for all three seeds, while `config_snapshot`
        recorded that it varied: three replicates of one leaf, reported as
        three seeds. `None` keeps the construction-time value, which is what
        every non-bench caller wants.

        `n_predict` OVERRIDES `target.max_predict` FOR THIS CALL, on the same
        terms and for a related reason: §8's B2 sizes its per-chunk summary
        budget so that ALL of them fit 80% of the ROOT window
        (`rlm.arms.b2_summary_n_predict`), and a budget the caller can only
        RECORD is not a budget. Unenforced, a 299-chunk aggregation corpus
        decodes 299 x `max_predict` tokens of summary into a reduce prompt
        sized for 0.8 x 32K -- the arm overflows the root window by
        construction, which is a manufactured §8 result rather than a
        measurement. Never used to RAISE a budget: the caller passes a value
        it derived, and `target.max_predict` remains what an omitted call gets.
        """
        for attempt in range(_MAX_PREFLIGHT_ROTATIONS + 1):
            generation = self._pool_generation
            # The gate FIRST, and the flight counter with it: `Event.wait()` on
            # a set event does not suspend, so nothing runs between them and no
            # call can start talking to a process that is about to be replaced.
            async with self._admitted():
                try:
                    return await self._query_once(prompt, role=role,
                                                   call_id=call_id, chunk=chunk,
                                                   seed=seed, n_predict=n_predict)
                except PreflightFailed:
                    # The pre-flight died against a process the scaffold itself
                    # was replacing. That is not this call's fault and must not
                    # be its outcome: it re-runs its pre-flight against the new
                    # process (the refusal is already recorded as its own
                    # attempt, and `_retry_base` keeps the retry_idx sequence
                    # continuous). A pre-flight that failed with NO rotation in
                    # sight is re-raised as the DispatchError it always was --
                    # §5 C4's server-death rule is untouched.
                    if self._rotation_seen(generation):
                        continue
                    raise
        raise DispatchError(
            f"leaf call {call_id!r} lost its pre-flight to "
            f"{_MAX_PREFLIGHT_ROTATIONS} consecutive rotations")

    def _rotation_seen(self, generation: int) -> bool:
        """Is a rotation in progress, or has one completed, since `generation`?

        Both halves matter: the gate being closed means one is being set up
        right now (the call will park on it), and a changed pool generation
        means one completed while this call was mid-pre-flight.
        """
        return (not self._gate.is_set()) or self._pool_generation != generation

    @contextlib.asynccontextmanager
    async def _admitted(self) -> "AsyncIterator[None]":
        """Park on the rotation gate, then count as in flight until done.

        This is what `quiesce()` waits on, and it deliberately covers EVERY
        leaf round trip rather than only the ones holding a slot: a rotation's
        precondition is that nothing is talking to the process being replaced,
        and `/apply-template` and `/tokenize` talk to it exactly as much as
        `/completion` does.
        """
        await self._gate.wait()
        self._in_flight += 1
        self._idle.clear()
        try:
            yield
        finally:
            # Every exit path, cancellation included: a call that no longer
            # talks to the server must not keep a rotation waiting.
            self._in_flight -= 1
            if self._in_flight == 0:
                self._idle.set()

    async def _query_once(self, prompt: str, *, role: str, call_id: str,
                           chunk: str | None = None,
                           seed: int | None = None,
                           n_predict: int | None = None) -> "str | dict[str, Any]":
        """One pre-flight-and-dispatch pass, already gated and counted."""
        target = self._targets.get(role)
        if target is None:
            raise DispatchError(f"unknown dispatch role {role!r}")
        # Both decoding parameters are resolved ONCE, here, so every attempt of
        # this call decodes identically: a retry that silently changed either
        # would make the retry a different call from the one being retried.
        seed = target.seed if seed is None else seed
        n_predict = target.max_predict if n_predict is None else n_predict
        layout = LAYOUT_QUESTION_ONLY if chunk is None else LAYOUT_CHUNK_QUESTION
        # Where this dispatch's retry_idx sequence starts -- 0 unless a
        # rotation is re-dispatching a call that already recorded attempts.
        base = self._retry_base(call_id)

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
        # Exactly two messages, prefix first, and the user segment composed
        # HERE from (chunk, question) so the [system prefix][chunk][question]
        # layout holds by construction rather than by the root model's
        # formatting discipline: a re-queried chunk then extends the slot's
        # resident prefix instead of invalidating it at token 0 (measured
        # cache_n 546 chunk-first vs 0 question-first).
        # `sent` is the user segment BEFORE sanitisation -- chunk and question
        # together, which is exactly what R13's oracle treats as "own" text
        # (a leak is absent from the document AND absent from the question).
        sent = compose_leaf_user(prompt, chunk)
        user_message = neutralise_control_tokens(
            sent, await self._markers(target))
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
            step = _new_step(call_id, base, role, layout=layout)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = f"pre-flight /apply-template failed: {exc}"
            self._record(step)
            raise PreflightFailed(
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
            step = _new_step(call_id, base, role, layout=layout, rendered=rendered)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = f"pre-flight /tokenize failed: {exc}"
            self._record(step)
            raise PreflightFailed(
                f"pre-flight /tokenize failed for role={role!r}: {exc}") from exc

        # The §4 head, measured ONCE per target. Done here rather than from a
        # synthetic `[system, user=""]` probe because this is a real request:
        # the head measured is the head that was actually sent, and it costs no
        # extra render. The server is known reachable at this point (the
        # pre-flight above just succeeded), and a failure is still only
        # diagnostic -- it must not fail a call C4 could otherwise serve.
        if user_message:
            cut = rendered.rfind(user_message)
            if cut > 0:
                head_text = rendered[:cut]
                head_hash = hashlib.sha256(head_text.encode("utf-8")).hexdigest()
                if target.prefix_sha256 is None:
                    target.prefix_sha256 = head_hash
                elif head_hash != target.prefix_sha256:
                    # R3 / §7 #3 (a1). NOT retried and NOT tolerated: §4's whole
                    # prefix contract is that this string is constant for the
                    # target's lifetime, and every cache number -- the
                    # `cache_n == N_resident - ub - 4` identity, gate (b)'s
                    # re-query ratio, the 311-token length -- is denominated in
                    # it. A prefix that moves mid-episode does not degrade the
                    # measurements, it silently redefines them.
                    step = _new_step(call_id, base, role, layout=layout,
                                     rendered=rendered,
                                     prefix_tokens=target.prefix_tokens,
                                     prefix_sha256=head_hash)
                    step["status"] = StepStatus.ERROR
                    step["error_detail"] = (
                        f"prefix drift: the rendered system head changed under "
                        f"a live {role} target (pinned "
                        f"{target.prefix_sha256[:12]}..., now "
                        f"{head_hash[:12]}...). §4 requires it byte-identical "
                        f"for the target's lifetime")
                    step["t_end"] = utc_now()
                    self._record(step)
                    raise PrefixDrift(step["error_detail"])

                # The token length is a SEPARATE round trip, so it is measured
                # once and pinned. With the hash checked on every call, a
                # matching hash already implies a matching length.
                if target.prefix_tokens is None:
                    try:
                        head = await target.client.tokenize(head_text,
                                                             add_special=True)
                    except Exception:  # noqa: BLE001 -- diagnostic, never fatal
                        pass
                    else:
                        target.prefix_tokens = len(head)

        if len(tokens) > target.slot_capacity_tokens:
            step = _new_step(call_id, base, role, layout=layout, rendered=rendered,
                              prefix_tokens=target.prefix_tokens,
                              prefix_sha256=target.prefix_sha256)
            step["status"] = StepStatus.REJECTED
            step["error_detail"] = (
                f"prompt ({len(tokens)} tokens) exceeds slot capacity "
                f"({target.slot_capacity_tokens} tokens)"
            )
            self._record(step)
            raise DispatchError(step["error_detail"])

        # R13: this window's own never-reused slot, acquired AFTER admission
        # so a prompt rejected pre-flight (nothing sent, nothing prefilled)
        # does not burn one. Held for every attempt of this call: a retry
        # re-sends the SAME document, which is same-document reuse and
        # measured clean, and a fresh slot per attempt would drain the pool
        # three times as fast as the budget assumes.
        #
        # The gate was taken by `query()` before any of the pre-flight above,
        # and this call has counted as in flight throughout, so the pool it is
        # drawing from cannot be replaced underneath it.
        window = window_key(chunk, call_id)
        try:
            slot = self.slots.acquire(window)
        except SlotPoolExhausted as exc:
            step = _new_step(call_id, base, role, layout=layout, rendered=rendered,
                              prefix_tokens=target.prefix_tokens,
                              prefix_sha256=target.prefix_sha256)
            step["status"] = StepStatus.ERROR
            step["error_detail"] = str(exc)
            self._record(step)
            raise
        try:
            answer, parsed = await self._attempts_loop(
                target, role, call_id, base, layout, rendered, sent, slot,
                seed=seed, n_predict=n_predict)
        except asyncio.CancelledError:
            raise
        except Exception:
            # WHY this slot was consumed, not merely THAT it was (§5 C4). A
            # pool drained by failures is not a healthy pool that ran out of
            # windows, and rotating on it would relaunch a FAILED server --
            # the one thing §5 C4 has always forbidden.
            self.slots.mark_failed(window)
            raise
        self.slots.mark_answered(window)
        if parsed is None:
            return answer
        # The span check, IN PROCESS AND FREE: whitespace-normalized substring
        # match of each quoted span against the chunk that was actually sent.
        # Zero model calls, no judge, no second opinion -- and, measured, no
        # great power either: §10 R5 puts its catch rate at 7/66 = 11%, because
        # 89% of the leaf's wrong answers quote a genuine in-chunk span
        # belonging to a DIFFERENT entity. It ships because the A/B has to price
        # the whole envelope, and it is reported as what it is.
        return envelope_payload(parsed, chunk=chunk)

    async def _attempts_loop(self, target: DispatchTarget, role: str, call_id: str,
                              base: int, layout: str, rendered: str, sent: str,
                              slot: int, *, seed: int,
                              n_predict: int) -> tuple[str, "EnvelopeParse | None"]:
        """The retry loop for one call, on one already-acquired slot.

        `seed` and `n_predict` are already resolved by `_query_once` (this
        call's overrides, or the target's values) and are required rather than
        defaulted here, for the same reason `ServerClient.completion` requires
        them: a decoding parameter that can be forgotten is one that will be.

        Returns the answer verbatim plus, when this target asks for the JSON
        envelope, the parsed envelope that came with it. A malformed envelope is
        retried HERE, through this same loop and this same backoff, rather than
        in a wrapper of its own: §5 says "one retry on parse failure through
        C4's existing retry machinery", and a second loop outside this one would
        give a format failure its own retry budget on top of the transport's --
        the call would then cost up to max_attempts^2 dispatches and the A/B's
        cost column would be measuring the scaffold.
        """
        last_exc: Exception | None = None
        last_raw = ""
        for attempt in range(self._retries.max_attempts):
            step = _new_step(call_id, base + attempt, role, layout=layout,
                              rendered=rendered,
                              prefix_tokens=target.prefix_tokens,
                              prefix_sha256=target.prefix_sha256)
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
                        rendered, n_predict=n_predict,
                        temperature=target.temperature, top_p=target.top_p,
                        seed=seed, stream=True, id_slot=slot)
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
                # R13's slot assertion, and the reason the whole mitigation is
                # honest rather than merely intended: an out-of-range id_slot
                # is silently reassigned with HTTP 200 (measured: asked 200,
                # got 72), so without this the scaffold believes it holds a
                # virgin slot while sharing a used one. A mismatch is a
                # contaminated answer -- status=error, not a warning -- and it
                # is NOT retried: the same request would ask for the same slot
                # and learn nothing, while the answer in hand cannot be
                # trusted. `slot_id` records what actually served.
                #
                # THE DETECTOR RUNS FIRST, and on this path above all others.
                # The answer that came off a slot C4 did not choose is the one
                # answer in the system with the highest prior probability of
                # carrying another document's content -- a foreign slot is
                # R13's reproducing condition stated exactly ("a slot that has
                # held one document injects it into answers about the next").
                # Recording `leak_detected=None` (NOT CHECKED) here, as this
                # did, was therefore backwards: the check was skipped precisely
                # where it was most likely to fire, and §8's per-arm
                # contamination count -- the S4 monitor R13 made binding --
                # would have been blind to the events most likely to be leaks.
                # The answer is still discarded and still not retried; what
                # the trace gains is the verdict on what was in it.
                verdict = self.leak_verdict(result.content, sent)
                step["leak_detected"] = verdict.detected
                step["leak_detail"] = verdict.detail
                step["response_text"] = result.content
                last_raw = result.content
                if result.slot_id != slot:
                    step["status"] = StepStatus.ERROR
                    step["error_detail"] = (
                        f"slot assertion failed: asked for id_slot {slot}, the "
                        f"server answered on id_slot {result.slot_id} (HTTP 200, "
                        "silently reassigned). That slot may have held other "
                        "documents, so the answer is discarded (R13)")
                    self._record(step)
                    raise SlotMismatch(step["error_detail"])

                parsed: EnvelopeParse | None = None
                if target.envelope:
                    parsed = envelope_parse(result.content)
                    if not parsed.ok:
                        # A format failure, not a transport failure -- but it
                        # takes the same road, because the remedy is the same
                        # (draw again) and because giving it a road of its own
                        # would double the retry budget. The step is ERROR so
                        # the attempt is visible and countable in the trace;
                        # `response_text` above already carries what could not
                        # be parsed.
                        step["status"] = StepStatus.ERROR
                        step["error_detail"] = parsed.error
                        step["t_end"] = utc_now()
                        self._record(step)
                        if attempt < self._retries.max_attempts - 1:
                            backoff = self._retries.backoff_s[
                                min(attempt, len(self._retries.backoff_s) - 1)]
                            await asyncio.sleep(backoff)
                            continue
                        raise EnvelopeParseError(
                            f"leaf did not return a valid envelope in "
                            f"{self._retries.max_attempts} attempts "
                            f"(role={role!r}, call_id={call_id!r}): {parsed.error}",
                            raw=result.content)

                self._record(step)
                return result.content, parsed
        # unreachable: the loop above always returns or raises.
        raise DispatchError(f"exhausted retries with no result: {last_exc}"
                            + (f" (last output: {last_raw[:120]!r})" if last_raw else ""))


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
        self._chunk_index: ChunkIndex | None = None

    @property
    def last_step(self) -> dict[str, Any] | None:
        return self.steps[-1] if self.steps else None

    def set_corpus(self, chunks: Any) -> None:
        """Same entry point as the real dispatcher's, so the episode runner
        wires C2's chunks the same way in a dry run and the steps a dry run
        writes carry the same leak columns. There is no slot pool here: a
        MockDispatcher never touches a server, so it has no slot to reuse."""
        self._chunk_index = ChunkIndex.from_chunks(chunks) if chunks else None

    async def count_tokens(self, text: str, *, role: str = "leaf") -> int:
        """A dry run has no server to ask, so this is a stated approximation,
        not a measurement: ~4 chars per token, monotonic in prefix length
        (which C2's binary search requires) and deterministic (which the
        chunker's determinism contract requires). Dry-run chunk boundaries
        therefore do NOT match a real run's, and no dry-run episode may be
        scored -- which is already true for other reasons (`dry_run=true`)."""
        return (len(text) + 3) // 4

    async def query(self, prompt: str, *, role: str, call_id: str,
                     chunk: str | None = None, seed: int | None = None,
                     n_predict: int | None = None) -> str:
        # `seed` and `n_predict` are accepted and IGNORED, for interface parity
        # with `LLMDispatcher.query`: a dry run replays fixtures and decodes
        # nothing, so there is no draw for a seed to steer and no decode for a
        # budget to bound -- but every caller passes both, and a mock that
        # rejected either keyword would make the dry-run path diverge from the
        # real one at exactly the call site the dry run exists to exercise.
        #
        # Keyed on the COMPOSED user string, so a fixture keyed by (role,
        # prompt-hash) still matches whichever form the model used to build
        # the same request (§5 dry-run mode).
        composed = compose_leaf_user(prompt, chunk)
        key = f"{role}:{hashlib.sha256(composed.encode('utf-8')).hexdigest()}"
        step = _new_step(call_id, 0, role,
                          layout=(LAYOUT_QUESTION_ONLY if chunk is None
                                  else LAYOUT_CHUNK_QUESTION))
        async with self.semaphore:
            if key not in self._fixtures:
                step["status"] = StepStatus.ERROR
                step["error_detail"] = f"MockDispatcher: no fixture for {key}"
                self.steps.append(step)
                raise DispatchError(step["error_detail"])
            answer = self._fixtures[key]
            step["status"] = StepStatus.OK
            if self._chunk_index is not None:
                verdict = self._chunk_index.foreign(answer, sent=composed)
                step["leak_detected"] = verdict.detected
                step["leak_detail"] = verdict.detail
            self.steps.append(step)
            return answer
