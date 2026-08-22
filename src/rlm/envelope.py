"""The leaf JSON envelope -- parse, validate, verify spans (spec §5, §10 R5).

WHY THIS EXISTS, as a number. The S2 sweep asked the leaf about a fact that was
not in the chunk, 39 times across every chunk size from 1K to 32K. It answered
anyway **37 times (95%)**, and the rate is the one figure in `milestones/s2/RESULTS.md`
that is FLAT across the size axis. Worse, the wrong answers are not fabrications
but MISATTRIBUTIONS: 59 of 66 quote an identifier that genuinely occurs in the
chunk, belonging to a different entity. So the spec's original defence -- check
that the leaf's quoted evidence appears in the chunk -- catches 7/66 = 11%
(§10 R5). **A span check proves text was copied, never that it answers the
question**, and this module must not be read as claiming otherwise.

The field that might actually work is `abstain`: an explicit, cheap, structured
channel for "not here", separated from the answer text so the scaffold reads a
BOOLEAN rather than pattern-matching prose for refusal phrases. That is the
hypothesis the S2 A/B tests (`milestones/s2/REFUSAL-AB.md`); this module is its instrument,
and the evidence half ships with it so the A/B prices the whole envelope --
format cost included -- rather than a flattering half of it.

WHAT IS DELIBERATELY NOT HERE.

  * **No server-side grammar or `json_schema`.** That is a separate, optional,
    A/B'd flag and it is never trusted: llama.cpp has documented silent
    fail-open on schema-parse failure (§13), which would turn "the envelope
    parsed" into "the server said nothing was wrong". Validation is
    scaffold-side, here, in process, on the bytes that came back.
  * **No model calls.** Span verification is a whitespace-normalized substring
    test. Zero tokens, no second opinion, no judge.
  * **No retry policy.** C4 owns retries (`rlm.dispatcher`); this module
    returns a verdict and a reason, and the reason is what lands in
    `steps.error_detail`.
  * **No scoring.** Whether `abstain=True` beside a substantive `answer` counts
    as a refusal is a SCORING question and belongs to the scorer
    (`milestones/s2/run_sweep.py`), which must apply the same rule to the plain-text arms.
    This module reports what the model emitted.

Isolated by the §5 dependency rule (`tests/test_import_rules.py`): stdlib only,
so the trace/analysis side can re-derive envelope verdicts offline from stored
answers without constructing an HTTP client.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

#: The contract, verbatim from §5: `{"answer": str, "evidence": [str],
#: "abstain": bool}`. All three are REQUIRED -- an envelope missing `abstain` is
#: exactly the envelope whose one interesting field is absent.
REQUIRED_FIELDS: tuple[str, ...] = ("answer", "evidence", "abstain")

#: `steps.error_detail` is read by humans and by a retry log line. A model's
#: whole malformed output dumped there would bury the trace.
ERROR_CAP = 200

_WS_RE = re.compile(r"\s+")
#: A leading `<think>...</think>`. `enable_thinking` is off for the leaf by
#: default (config `scaffold.leaf.enable_thinking`), but S1 (F3) measured leaf
#: replies consisting of nothing but a think block, so a parser that dies on one
#: would be scoring the sampler. Independent of `rlm.rootclient.strip_reasoning`
#: on purpose: that module is C4-adjacent and this one must import nothing.
_THINK_RE = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Envelope:
    """One validated leaf envelope. `extras` names keys the model added that the
    contract does not mention -- kept rather than rejected (a model that also
    reports a confidence has not failed to abstain) but recorded, so "the
    envelope parsed" never quietly means "and it was the envelope we asked
    for"."""

    answer: str
    evidence: tuple[str, ...]
    abstain: bool
    extras: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParseResult:
    """`envelope` XOR `error`. `salvaged` is True when the object had to be
    dug out of surrounding prose rather than being the whole reply -- the A/B
    reports raw and salvaged compliance separately, because a parser generous
    enough to hide a format failure would be measuring itself."""

    envelope: Envelope | None
    error: str | None
    raw: str
    salvaged: bool = False

    @property
    def ok(self) -> bool:
        return self.envelope is not None

    def __bool__(self) -> bool:
        return self.ok


def normalize_ws(text: str) -> str:
    """THE PINNED NORMALIZATION for the span check: every whitespace run becomes
    one space. Nothing else.

    Case is PRESERVED. §5's rule is "whitespace-normalized substring match" and
    case-folding would only make an already near-inert check (11%, §10 R5) more
    permissive, while destroying its ability to separate `ENT-4A1` from
    `ent-4a1`. Punctuation and unicode are untouched for the same reason: every
    widening of this function is a widening of what counts as evidence.
    """
    return _WS_RE.sub(" ", text or "")


def verify_evidence(evidence: Sequence[str], *, chunk: str | None) -> tuple[bool | None, ...]:
    """One verdict per span: is it a substring of the chunk, modulo whitespace?

    `None` means NOT CHECKED, and it is not the same claim as False. `chunk=None`
    is `llm_query`'s single-string form, where the scaffold cannot see where the
    document ended and therefore has checked nothing -- recording False there
    would read as "checked and failed", the mistake `rlm.leakcheck` refuses to
    make with its own tri-state verdict.

    An empty or whitespace-only span is False, never True: `""` is a substring
    of every chunk, so trusting it would verify an envelope that quoted nothing.
    """
    if chunk is None:
        return tuple(None for _ in evidence)
    haystack = normalize_ws(chunk)
    verdicts: list[bool | None] = []
    for span in evidence:
        needle = normalize_ws(span).strip()
        verdicts.append(bool(needle) and needle in haystack)
    return tuple(verdicts)


def parse(raw: str) -> ParseResult:
    """Parse and validate one leaf reply into an `Envelope`.

    Permissive about WRAPPING (a code fence, a reasoning block, surrounding
    prose) and strict about CONTENT -- the same split the sweep's scorer makes,
    and for the same reason: a leaf that formats loosely is a prompt problem, a
    leaf that reports `"abstain": "true"` as a string is a corrupted
    measurement.
    """
    text = (raw or "").strip()
    if not text:
        return ParseResult(None, "envelope parse failed: empty output", raw)

    text = _THINK_RE.sub("", text, count=1).strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()

    salvaged = False
    obj, err = _load_object(text)
    if obj is None:
        span = _first_object_span(text)
        if span is not None:
            obj, err = _load_object(span)
            salvaged = obj is not None
    if obj is None:
        return ParseResult(None, _cap(f"envelope parse failed: no JSON object "
                                      f"in the reply ({err})"), raw)

    invalid = _validate(obj)
    if invalid:
        return ParseResult(None, _cap(f"envelope invalid: {invalid}"), raw)

    return ParseResult(
        Envelope(
            answer=obj["answer"],
            evidence=tuple(obj["evidence"]),
            abstain=obj["abstain"],
            extras=tuple(sorted(set(obj) - set(REQUIRED_FIELDS))),
        ),
        None,
        raw,
        salvaged=salvaged,
    )


def payload(result: ParseResult, *, chunk: str | None) -> dict:
    """The JSON-serializable object `llm_query` hands back inside the sandbox
    when the envelope is on.

    Everything here is a fact the root may branch on, and nothing here is a
    judgement: `abstain` is what the model said, `evidence_verified` is what the
    substring test found, `raw` is what came off the wire. Whether an abstention
    beside a substantive answer counts as a refusal is a SCORING rule and lives
    with the scorer, which has to apply the same rule to the plain-text arms.

    `evidence_ok` is tri-state for the same reason `rlm.leakcheck`'s verdict is:
    None means nothing was checked -- no chunk (the single-string call form) or
    no spans to check (an abstention) -- and must never be read as a pass.
    """
    env = result.envelope
    assert env is not None, "payload() is for a successful parse"
    verified = verify_evidence(env.evidence, chunk=chunk)
    ok: bool | None
    if not verified or any(v is None for v in verified):
        ok = None
    else:
        ok = all(verified)
    return {
        "answer": env.answer,
        "evidence": list(env.evidence),
        "abstain": env.abstain,
        "evidence_verified": list(verified),
        "evidence_ok": ok,
        "extras": list(env.extras),
        "salvaged": result.salvaged,
        "raw": result.raw,
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _load_object(text: str) -> tuple[dict | None, str]:
    try:
        obj = json.loads(text)
    except ValueError as exc:
        return None, str(exc).split("\n")[0]
    if not isinstance(obj, dict):
        return None, f"top level is {type(obj).__name__}, not an object"
    return obj, ""


def _first_object_span(text: str) -> str | None:
    """The first balanced `{...}` in `text`, STRING-AWARE.

    A `rfind("}")` would do for well-formed replies and fail on the ones that
    matter: corpus text quoted into `evidence` routinely contains braces and
    escaped quotes, and truncating an object at the wrong brace turns a
    recoverable reply into a spurious parse failure -- which the A/B would
    record as the model's fault.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _validate(obj: dict) -> str:
    """The first contract violation, named by field, or "" when the object is
    a valid envelope. Field order is `REQUIRED_FIELDS` so the reason is stable
    across runs and groupable in the report."""
    for field in REQUIRED_FIELDS:
        if field not in obj:
            return f"missing required field {field!r}"
    if not isinstance(obj["answer"], str):
        return (f"field 'answer' must be a string, got "
                f"{type(obj['answer']).__name__}")
    evidence = obj["evidence"]
    if not isinstance(evidence, list):
        return (f"field 'evidence' must be a list of strings, got "
                f"{type(evidence).__name__}")
    for i, span in enumerate(evidence):
        if not isinstance(span, str):
            return (f"field 'evidence'[{i}] must be a string, got "
                    f"{type(span).__name__}")
    # `isinstance(True, int)` is True in Python, so the bool check must come
    # first and the int check must exclude bools -- otherwise `"abstain": 0`
    # would pass as False and every arm would look like it abstains on demand.
    if not isinstance(obj["abstain"], bool):
        return (f"field 'abstain' must be a JSON boolean, got "
                f"{type(obj['abstain']).__name__}")
    return ""


def _cap(detail: str) -> str:
    detail = " ".join(detail.split())
    if len(detail) > ERROR_CAP:
        detail = detail[:ERROR_CAP - 3] + "..."
    return detail
