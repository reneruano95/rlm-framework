#!/usr/bin/env python3
"""R13 minimal reproducer: cross-request content leakage on a shared llama.cpp slot.

TWO COPIES, DELIBERATELY. This file exists byte-identically at BOTH
`milestones/s2/r13_repro.py` and `upstream/r13_repro.py`, and each is cited independently:
the s2 copy by `milestones/s2/R13.md`, `milestones/s2/R13-mitigations.md`, `src/rlm/leakcheck.py:38`
and three `milestones/s2/audit/*.py` scripts; the upstream copy by `upstream/README.md`.
Removing either breaks a citation, so both stay. KEEP THEM IDENTICAL -- apply
any change to both in the same commit. Their sha256 equality is the only thing
standing between these two files and silent divergence.

SELF-CONTAINED ON PURPOSE. This module shares no code with `milestones/s2/run_sweep.py` or
`milestones/s2/leafcall.py` -- the harness that found the defect cannot also be the
independent evidence that it is real. Only the standard library and `httpx` are
imported; nothing from `rlm.*` is touched. The prompts are built here, the
chat template is applied by the server's own `/apply-template`, and the
classifier is a plain substring test against a UUID minted seconds earlier.

THE EXPERIMENT
--------------
Two documents, one server slot.

  Document A plants  <entity_A> -> <uuid_A>   (both invented, fresh per trial)
  Document B plants  <entity_B> -> <uuid_B>   and contains no trace of A

Prompt A is sent, then prompt B is sent asking for **A's** entity -- a string
that appears nowhere in B's document. A server with no cross-request state must
answer "NOT IN DOCUMENT". A server that carries residue from prompt A can
answer with `uuid_A`, which it can only have obtained from the previous request.

Because both UUIDs are generated at trial time, a leaked UUID is not a model
prior, a training-set artefact, or a scoring bug: it is the previous request's
bytes.

CONDITIONS (`--conditions`, comma separated, `all` for every one)
  same_slot_cache    A and B on one pinned slot, cache_prompt=true
  same_slot_nocache  A and B on one pinned slot, cache_prompt=false
  diff_slot          A on slot 0, B on slot 1, same process
  fresh_process      B alone in a brand-new server process (floor control)
  aba                A, B, then a third prompt asking about B's fact
  save_restore       A, then POST /slots/N?action=save + restore, then B
  seed_change        as same_slot_cache but B draws a different sampling seed
  n_keep0            as same_slot_cache but B sends n_keep=0

Every call records the verbatim output, `timings.cache_n` (tokens reused from
the prompt cache), the slot the server actually used, `timings.prompt_n`, and
wall/prefill/decode timings, appended as one JSON object per line to `--out`.

WHICH MODE TO USE -- READ THIS FIRST
------------------------------------
The synthetic design above is the clean experiment and it is the one you want
if it fires, because a per-trial UUID makes a hit unarguable. **It did not
fire**: 110 trials across 522-8,551 token documents, both `cache_prompt`
settings, both slot arrangements, A->B->A, changed seeds, `n_keep: 0` and
save/restore all came back at zero. The effect needs long, low-entropy,
near-duplicate prose, so reproduction runs through `--replay-fixtures`, which
reads the corpus in `milestones/s2/fixtures/` as DATA and drives it with this file's
transport and classifier.

The decisive arrangement is `--replay-fixtures --paired-virgin`: every prompt
goes out twice in the same process at the same moment, once to a pinned slot
that accumulates documents and once to a slot that has only ever held that one
document. Weights, sampler, prompt bytes and timing are identical across the
pair, so the only variable left is slot history. Measured 24/54 vs 0/54,
p = 4.4e-9. See `milestones/s2/R13.md`.

USAGE
  # the decisive paired run
  uv run --python 3.12 python milestones/s2/r13_repro.py --port 8081 \
      --replay-fixtures milestones/s2/fixtures --replay-trials 3 --paired-virgin \
      --temperature 0.3 --n-predict 512 --slot 0 --out milestones/s2/results/r13.jsonl

  # does an explicit slot erase clear it? (server needs --slot-save-path)
  uv run --python 3.12 python milestones/s2/r13_repro.py --port 8081 \
      --replay-fixtures milestones/s2/fixtures --erase-between ...

  # the synthetic design, and the size search that found no floor
  uv run --python 3.12 python milestones/s2/r13_repro.py --port 8081 --trials 5
  uv run --python 3.12 python milestones/s2/r13_repro.py --port 8081 \
      --conditions same_slot_nocache --sizes 512,1024,2048,4096 --trials 4

  # architecture control: confirm the layout, then run the same replay
  uv run --python 3.12 python milestones/s2/r13_repro.py --port 8082 --arch-report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

SYSTEM = (
    "You answer questions strictly from the DOCUMENT given in the user message. "
    "The document is the only source of truth. If the answer does not appear in "
    "the document, reply with exactly: NOT IN DOCUMENT\n"
    "Answer with the value only. No explanation."
)

# Neutral filler vocabulary. Deliberately bland and repetitive across documents
# so that the *only* distinguishing content in either document is its planted
# fact -- if the model reproduces a fact, it cannot have inferred it from style.
_NOUNS = "pallet crate ledger bay dock manifest seal ticket lane rack bin lot".split()
_VERBS = "logged staged cleared held routed sealed weighed rotated scanned".split()
_ADJS = "damp cold sealed partial dry outbound inbound bonded interim".split()

_SYL_A = "bar cor dre fel gan hal jor kel mor nar pel quor ral sten tor vel wyn zar".split()
_SYL_B = "bleck dorn feld garth holm keth lume mont nell osk plith rond shaw thal vurn".split()
_SYL_C = "wick mere ridge gate stow field haven crest reach ford".split()


def _entity(rng: random.Random) -> str:
    """An invented proper noun that cannot collide with anything in training."""
    return (
        rng.choice(_SYL_A).capitalize()
        + rng.choice(_SYL_B)
        + rng.choice(_SYL_C)
        + " "
        + rng.choice(["Repository", "Exchange", "Vault", "Registry", "Depot"])
    )


def _filler_line(rng: random.Random) -> str:
    return (
        f"{rng.randint(1000, 9999)} {rng.choice(_ADJS)} {rng.choice(_NOUNS)} "
        f"{rng.choice(_VERBS)} at {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} "
        f"on {rng.choice(_NOUNS)} {rng.randint(1, 40)}."
    )


def build_document(
    rng: random.Random, entity: str, key: str, n_lines: int, depth: float = 0.95
) -> str:
    """Filler lines with the planted fact inserted at `depth` through them.

    Placement is not incidental, and it is the one knob that must be swept.
    `depth=0.95` puts the fact next to the question, inside this leaf's
    measured ~1000-token retrieval horizon: the model can answer its own
    document, so a wrong answer cannot be blamed on retrieval failure. Shallow
    depths push the fact past that horizon, which is where the parent sweep's
    cross-request answers were observed -- if leakage needs a model that has
    already failed to find its own fact, only the shallow cells will show it.
    """
    lines = [_filler_line(rng) for _ in range(n_lines)]
    at = min(len(lines), max(0, int(round(depth * len(lines)))))
    lines.insert(at, f"Custody key for the {entity}: {key}.")
    return "\n".join(lines)


def compose_user(document: str, question_entity: str) -> str:
    """Document first, question LAST -- the layout the server is asked to cache."""
    return (
        "DOCUMENT:\n"
        f"{document}\n\n"
        f"QUESTION: What is the custody key for the {question_entity}?"
    )


#: The enumeration probe. A targeted question lets a contaminated model hide
#: behind a terse refusal ("NOT IN DOCUMENT") that happens to be correct; the
#: contamination only becomes visible when the model volunteers what it thinks
#: the document DOES contain. Asking it to list everything turns a yes/no probe
#: into a dump of whatever the slot believes it is holding, which is far more
#: sensitive at the same token cost.
ENUM_SYSTEM = (
    "You answer questions strictly from the DOCUMENT given in the user message. "
    "List what is actually there. Do not invent entries."
)


def compose_enum(document: str) -> str:
    return (
        "DOCUMENT:\n"
        f"{document}\n\n"
        "QUESTION: List every custody key in this document, each with the name of "
        "the entity it belongs to. List them all."
    )


# --------------------------------------------------------------------------
# server transport
# --------------------------------------------------------------------------


@dataclass
class Call:
    """One /completion round trip, flattened for JSONL."""

    condition: str
    trial: int
    label: str  # "A", "B", or "A2"
    doc_tokens: int
    id_slot_requested: int | None
    cache_prompt: bool
    seed: int
    depth: float | None = None  # filled in by main() after each trial
    slot_id: int | None = None
    prompt_n: int | None = None
    cache_n: int | None = None
    tokens_cached: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None
    wall_s: float | None = None
    raw_output: str = ""
    stop_type: str | None = None
    error: str | None = None
    # trial bookkeeping -- what was planted where
    entity_a: str = ""
    uuid_a: str = ""
    entity_b: str = ""
    uuid_b: str = ""
    asked_about: str = ""
    verdict: str = ""


class Server:
    """Thin client over the llama.cpp server HTTP API."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def wait_healthy(self, deadline_s: float = 900.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < deadline_s:
            try:
                r = self.client.get("/health", timeout=5.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"{self.base_url} never became healthy")

    def props(self) -> dict[str, Any]:
        return self.client.get("/props").json()

    def render(self, user: str, *, thinking: bool = False, system: str | None = None) -> str:
        """Apply the model's own chat template server-side.

        `enable_thinking=false` closes the `<think>` block the Qwen template
        opens by default. Without it the model spends its budget narrating and
        the answer never arrives inside a sane `n_predict`; the sweep this
        reproduces also ran with direct answers.
        """
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system or SYSTEM},
                {"role": "user", "content": user},
            ]
        }
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = self.client.post("/apply-template", json=body)
        r.raise_for_status()
        return r.json()["prompt"]

    def tokenize(self, text: str) -> int:
        r = self.client.post("/tokenize", json={"content": text})
        r.raise_for_status()
        return len(r.json()["tokens"])

    def completion(
        self,
        prompt: str,
        *,
        id_slot: int | None,
        cache_prompt: bool,
        seed: int,
        n_predict: int = 96,
        temperature: float = 0.0,
        n_keep: int | None = None,
    ) -> tuple[dict[str, Any], float]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "cache_prompt": cache_prompt,
            "stream": False,
        }
        if id_slot is not None:
            body["id_slot"] = id_slot
        if n_keep is not None:
            body["n_keep"] = n_keep
        t0 = time.monotonic()
        r = self.client.post("/completion", json=body)
        wall = time.monotonic() - t0
        r.raise_for_status()
        return r.json(), wall

    def slot_action(self, slot: int, action: str, filename: str) -> tuple[int, str]:
        try:
            r = self.client.post(
                f"/slots/{slot}?action={action}", json={"filename": filename}
            )
            return r.status_code, r.text[:400]
        except httpx.HTTPError as exc:  # pragma: no cover - diagnostic path
            return -1, str(exc)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

_REFUSAL = re.compile(
    r"not in document|no information|does not (appear|contain|mention)|"
    r"is not (present|mentioned|found|in the)|cannot be determined|no mention|"
    r"not (mentioned|present|found|listed|provided)",
    re.I,
)


def classify(out: str, own_uuid: str, other_uuid: str, other_entity: str) -> str:
    """LEAK is defined ONLY by the OTHER prompt's UUID appearing in this answer.

    The UUID is minted per trial, so there is exactly one way for it to be in
    this response: the server carried it over from the previous request.

    Entity NAMES are deliberately not evidence. The probe question names A's
    entity, so any restatement of the question echoes it -- an entity-name test
    fires on well-behaved refusals and is worthless. Only the UUID, which the
    probe prompt never contains, discriminates.

    A prefix match of 8 hex characters counts as a hit because this leaf
    truncates identifiers (the parent sweep logged fragments like
    `ENT-78837 / 28ba`), and 8 hex characters is still 1 in 4 billion.
    """
    text = out.strip()
    head = other_uuid.replace("-", "")[:8] if other_uuid else ""
    normal = re.sub(r"[^0-9a-f]", "", text.lower())
    if other_uuid and (other_uuid.lower() in text.lower() or head in normal):
        return "LEAK_UUID"
    own_head = own_uuid.replace("-", "")[:8] if own_uuid else ""
    if own_uuid and (own_uuid.lower() in text.lower() or own_head in normal):
        # Answered with the fact from its OWN document. For a probe asking about
        # the other document's entity this is still wrong -- misattribution --
        # but it is not cross-request leakage.
        return "OWN_FACT"
    if _REFUSAL.search(text):
        return "REFUSAL"
    if not text:
        return "EMPTY"
    return "OTHER"


# --------------------------------------------------------------------------
# trial plumbing
# --------------------------------------------------------------------------


#: Set once from argv by main(). Kept module level so every condition samples
#: identically -- a leak rate is only comparable across conditions if the
#: sampler is held fixed.
SAMPLING: dict[str, Any] = {"n_predict": 96, "temperature": 0.0}


@dataclass
class Trial:
    entity_a: str
    uuid_a: str
    doc_a: str
    entity_b: str
    uuid_b: str
    doc_b: str


def make_trial(rng: random.Random, n_lines: int, depth: float = 0.95) -> Trial:
    ea, eb = _entity(rng), _entity(rng)
    while eb == ea:
        eb = _entity(rng)
    ua, ub = str(uuid.uuid4()), str(uuid.uuid4())
    return Trial(
        entity_a=ea,
        uuid_a=ua,
        doc_a=build_document(rng, ea, ua, n_lines, depth),
        entity_b=eb,
        uuid_b=ub,
        doc_b=build_document(rng, eb, ub, n_lines, depth),
    )


def _record(
    srv: Server,
    sink: list[Call],
    *,
    condition: str,
    trial_i: int,
    label: str,
    user: str,
    t: Trial,
    asked_about: str,
    id_slot: int | None,
    cache_prompt: bool,
    seed: int,
    doc_tokens: int,
    n_keep: int | None = None,
    own_uuid: str = "",
    other_uuid: str = "",
    other_entity: str = "",
    system: str | None = None,
) -> Call:
    call = Call(
        condition=condition,
        trial=trial_i,
        label=label,
        doc_tokens=doc_tokens,
        id_slot_requested=id_slot,
        cache_prompt=cache_prompt,
        seed=seed,
        entity_a=t.entity_a,
        uuid_a=t.uuid_a,
        entity_b=t.entity_b,
        uuid_b=t.uuid_b,
        asked_about=asked_about,
    )
    try:
        prompt = srv.render(user, system=system)
        data, wall = srv.completion(
            prompt,
            id_slot=id_slot,
            cache_prompt=cache_prompt,
            seed=seed,
            n_keep=n_keep,
            n_predict=SAMPLING["n_predict"],
            temperature=SAMPLING["temperature"],
        )
    except Exception as exc:  # noqa: BLE001 - a failed call is a recorded fact
        call.error = f"{type(exc).__name__}: {exc}"
        call.verdict = "ERROR"
        sink.append(call)
        return call

    timings = data.get("timings", {}) or {}
    call.raw_output = data.get("content", "")
    call.slot_id = data.get("id_slot")
    call.prompt_n = timings.get("prompt_n")
    call.cache_n = timings.get("cache_n")
    call.tokens_cached = data.get("tokens_cached", timings.get("cache_n"))
    call.prompt_ms = timings.get("prompt_ms")
    call.predicted_n = timings.get("predicted_n")
    call.predicted_ms = timings.get("predicted_ms")
    call.wall_s = round(wall, 4)
    call.stop_type = data.get("stop_type")
    call.verdict = classify(call.raw_output, own_uuid, other_uuid, other_entity)
    sink.append(call)
    return call


# --------------------------------------------------------------------------
# conditions
# --------------------------------------------------------------------------


def cond_two_prompt(
    srv: Server,
    sink: list[Call],
    t: Trial,
    trial_i: int,
    *,
    name: str,
    slot_a: int | None,
    slot_b: int | None,
    cache_a: bool,
    cache_b: bool,
    seed_a: int,
    seed_b: int,
    doc_tokens: int,
    n_keep_b: int | None = None,
    between: Any = None,
) -> None:
    """A, then B asking about A's entity. The core of every condition."""
    _record(
        srv,
        sink,
        condition=name,
        trial_i=trial_i,
        label="A",
        user=compose_user(t.doc_a, t.entity_a),
        t=t,
        asked_about=t.entity_a,
        id_slot=slot_a,
        cache_prompt=cache_a,
        seed=seed_a,
        doc_tokens=doc_tokens,
        own_uuid=t.uuid_a,
        other_uuid=t.uuid_b,
        other_entity=t.entity_b,
    )
    if between is not None:
        between()
    # Probe C first: the enumeration probe is the sensitive instrument, and it
    # must run before the targeted probe so the targeted probe's own wording
    # cannot prime the enumeration.
    _record(
        srv,
        sink,
        condition=name,
        trial_i=trial_i,
        label="C_enum",
        user=compose_enum(t.doc_b),
        t=t,
        asked_about="(enumerate)",
        id_slot=slot_b,
        cache_prompt=cache_b,
        seed=seed_b,
        doc_tokens=doc_tokens,
        n_keep=n_keep_b,
        own_uuid=t.uuid_b,
        other_uuid=t.uuid_a,
        other_entity=t.entity_a,
        system=ENUM_SYSTEM,
    )
    _record(
        srv,
        sink,
        condition=name,
        trial_i=trial_i,
        label="B",
        user=compose_user(t.doc_b, t.entity_a),  # <-- asks about A's entity
        t=t,
        asked_about=t.entity_a,
        id_slot=slot_b,
        cache_prompt=cache_b,
        seed=seed_b,
        doc_tokens=doc_tokens,
        n_keep=n_keep_b,
        own_uuid=t.uuid_b,
        other_uuid=t.uuid_a,  # a hit here is A's key surfacing inside B
        other_entity=t.entity_a,
    )


def cond_aba(
    srv: Server, sink: list[Call], t: Trial, trial_i: int, *, slot: int, doc_tokens: int
) -> None:
    """A -> B -> A2, where A2 re-sends document A but asks about B's entity.

    Order effects separate two mechanisms. If only B leaks A, the state is
    residue that decays or is overwritten. If A2 also leaks B, the slot is
    blending whatever it has most recently seen in either direction.
    """
    cond_two_prompt(
        srv,
        sink,
        t,
        trial_i,
        name="aba",
        slot_a=slot,
        slot_b=slot,
        cache_a=True,
        cache_b=True,
        seed_a=1000 + trial_i,
        seed_b=2000 + trial_i,
        doc_tokens=doc_tokens,
    )
    _record(
        srv,
        sink,
        condition="aba",
        trial_i=trial_i,
        label="A2",
        user=compose_user(t.doc_a, t.entity_b),  # A's doc, asking about B's entity
        t=t,
        asked_about=t.entity_b,
        id_slot=slot,
        cache_prompt=True,
        seed=3000 + trial_i,
        doc_tokens=doc_tokens,
        own_uuid=t.uuid_a,
        other_uuid=t.uuid_b,
        other_entity=t.entity_b,
    )


# --------------------------------------------------------------------------
# fresh-process control
# --------------------------------------------------------------------------


def launch_server(
    exe: str, model: str, port: int, extra: list[str], log_dir: Path
) -> subprocess.Popen[bytes]:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / f"srv{port}.out.log", "wb")
    err = open(log_dir / f"srv{port}.err.log", "wb")
    env = dict(os.environ)
    env.setdefault("ROCBLAS_USE_HIPBLASLT", "1")
    cmd = [exe, "--host", "127.0.0.1", "--port", str(port), "-m", model, *extra]
    return subprocess.Popen(cmd, stdout=out, stderr=err, env=env)


def cond_fresh_process(
    sink: list[Call],
    t: Trial,
    trial_i: int,
    *,
    exe: str,
    model: str,
    port: int,
    extra: list[str],
    log_dir: Path,
    doc_tokens: int,
) -> None:
    """Send ONLY prompt B into a server that has never seen prompt A.

    This is the floor: it proves the classifier cannot fire on model priors,
    the entity naming scheme, or anything intrinsic to document B.
    """
    proc = launch_server(exe, model, port, extra, log_dir)
    try:
        srv = Server(f"http://127.0.0.1:{port}")
        srv.wait_healthy()
        _record(
            srv,
            sink,
            condition="fresh_process",
            trial_i=trial_i,
            label="B",
            user=compose_user(t.doc_b, t.entity_a),
            t=t,
            asked_about=t.entity_a,
            id_slot=0,
            cache_prompt=True,
            seed=4000 + trial_i,
            doc_tokens=doc_tokens,
            own_uuid=t.uuid_b,
            other_uuid=t.uuid_a,
            other_entity=t.entity_a,
        )
        srv.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()


# --------------------------------------------------------------------------
# fixture replay -- reproduce the ORIGINAL observation with independent code
# --------------------------------------------------------------------------
#
# The synthetic two-prompt design above is the clean experiment, but if it
# comes back negative the next question is whether the original observation
# reproduces AT ALL. That question is only answerable against the original
# inputs. This mode reads the sweep's fixture corpus as DATA -- the chunk text
# and the manifest's questions -- and drives it with the transport, prompt
# assembly and classifier in this file. No sweep code is imported, so a hit
# here is a property of the server and a miss here rules out the server.

_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|ENT-\d{4,6}", re.I)
# Invented proper nouns in this corpus are long CamelCase-ish coinages
# ("Prylfennwick", "Korrjostholm"). Ordinary English words are excluded by
# requiring the token to be absent from every chunk but one.
_PROPER = re.compile(r"\b[A-Z][a-z]{6,}\b")


def load_fixtures(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    texts = {
        p.name.replace(".chunk.txt", ""): p.read_text(encoding="utf-8")
        for p in root.glob("*.chunk.txt")
    }
    return manifest, texts


def foreign_strings(answer: str, own: str, texts: dict[str, str], own_cell: str) -> list[tuple[str, str]]:
    """Strings in `answer` that live in some OTHER cell's chunk and not in this one."""
    hits: list[tuple[str, str]] = []
    for tok in set(_ID.findall(answer)) | set(_PROPER.findall(answer)):
        if tok.lower() in own.lower():
            continue
        for cell, txt in texts.items():
            if cell != own_cell and tok.lower() in txt.lower():
                hits.append((tok, cell))
                break
    return hits


def run_replay(
    srv: Server,
    sink: list[Call],
    root: Path,
    cells: list[str],
    trials: int,
    *,
    slot: int,
    cache_prompt: bool,
    temperature: float,
    n_predict: int,
    name: str,
    erase_between: bool = False,
    paired_virgin: bool = False,
) -> list[dict[str, Any]]:
    """Replay the corpus cell by cell on one pinned slot.

    With `paired_virgin`, every prompt is additionally sent to a slot that has
    never held any chunk but this one -- the "one fresh slot per chunk"
    mitigation. Both arms see byte-identical prompts in the same process at the
    same moment, so the pair differs in exactly one thing: whether the slot has
    previously held a different document.
    """
    manifest, texts = load_fixtures(root)
    findings: list[dict[str, Any]] = []
    order = ["literal", "paraphrase", "absent"]
    for cell_index, cell in enumerate(cells):
        spec = manifest["cells"][cell]
        chunk = texts[cell]
        if erase_between:
            # Requires --slot-save-path on the server; without it every /slots
            # action answers 501 regardless of which action was asked for.
            code, body = srv.slot_action(slot, "erase", "unused.bin")
            print(f"[erase slot {slot}] {code} {body[:120]}")
        for trial in range(1, trials + 1):
            for qtype in order:
                q = spec["questions"][qtype]
                user = f"DOCUMENT:\n{chunk}\n\nQUESTION: {q['question']}"
                call = Call(
                    condition=name,
                    trial=trial,
                    label=f"{cell}:{qtype}",
                    doc_tokens=spec["measured_tokens"],
                    id_slot_requested=slot,
                    cache_prompt=cache_prompt,
                    seed=trial,
                    depth=spec["position"],
                    asked_about=q.get("entity", ""),
                )
                try:
                    prompt = srv.render(user)
                    data, wall = srv.completion(
                        prompt,
                        id_slot=slot,
                        cache_prompt=cache_prompt,
                        seed=trial,
                        n_predict=n_predict,
                        temperature=temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    call.error = f"{type(exc).__name__}: {exc}"
                    call.verdict = "ERROR"
                    sink.append(call)
                    continue
                tm = data.get("timings", {}) or {}
                call.raw_output = data.get("content", "")
                call.slot_id = data.get("id_slot")
                call.prompt_n = tm.get("prompt_n")
                call.cache_n = tm.get("cache_n")
                call.tokens_cached = data.get("tokens_cached", tm.get("cache_n"))
                call.prompt_ms = tm.get("prompt_ms")
                call.predicted_n = tm.get("predicted_n")
                call.predicted_ms = tm.get("predicted_ms")
                call.wall_s = round(wall, 4)
                call.stop_type = data.get("stop_type")
                fg = foreign_strings(call.raw_output, chunk, texts, cell)
                call.verdict = "FOREIGN" if fg else "OWN_OR_NONE"
                sink.append(call)
                if fg:
                    findings.append(
                        {
                            "cell": cell,
                            "qtype": qtype,
                            "trial": trial,
                            "cache_n": call.cache_n,
                            "prompt_n": call.prompt_n,
                            "foreign": fg,
                            "output": call.raw_output[:200],
                        }
                    )
                    print(f"  FOREIGN {cell}/{qtype}/t{trial} cache_n={call.cache_n} {fg} :: {call.raw_output[:90]!r}")

                if not paired_virgin:
                    continue
                # Same prompt, same moment, same process -- but on a slot that
                # has only ever held this chunk.
                vslot = 1 + cell_index
                vcall = Call(
                    condition=name + "__virgin_per_chunk",
                    trial=trial,
                    label=f"{cell}:{qtype}",
                    doc_tokens=spec["measured_tokens"],
                    id_slot_requested=vslot,
                    cache_prompt=cache_prompt,
                    seed=trial,
                    depth=spec["position"],
                    asked_about=q.get("entity", ""),
                )
                try:
                    vdata, vwall = srv.completion(
                        prompt,
                        id_slot=vslot,
                        cache_prompt=cache_prompt,
                        seed=trial,
                        n_predict=n_predict,
                        temperature=temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    vcall.error = f"{type(exc).__name__}: {exc}"
                    vcall.verdict = "ERROR"
                    sink.append(vcall)
                    continue
                vtm = vdata.get("timings", {}) or {}
                vcall.raw_output = vdata.get("content", "")
                vcall.slot_id = vdata.get("id_slot")
                vcall.prompt_n = vtm.get("prompt_n")
                vcall.cache_n = vtm.get("cache_n")
                vcall.wall_s = round(vwall, 4)
                vfg = foreign_strings(vcall.raw_output, chunk, texts, cell)
                vcall.verdict = "FOREIGN" if vfg else "OWN_OR_NONE"
                sink.append(vcall)
        print(f"[replay] {cell} done ({len(findings)} foreign so far)")
    return findings


def run_slot_ab(
    srv: Server,
    sink: list[Call],
    root: Path,
    *,
    cell_a: str,
    cell_b: str,
    prime: int,
    seeds: int,
    slot: int,
    virgin_slot: int,
    temperature: float,
    n_predict: int,
) -> None:
    """The sharpest form of the experiment: ONE prompt, two slots, same process.

    Prompt B is held byte-identical across every arm. The only thing that
    varies is which slot answers it and what that slot was asked to do first.
    That isolates per-slot residue from every other explanation at once --
    the model, the weights, the sampler, the prompt and the process are all
    held fixed, so a difference between arms cannot be attributed to any of
    them. `cache_prompt=false` and an explicit `erase` are then applied to the
    contaminated slot to see whether either of the two documented ways of
    telling the server to forget actually clears it.
    """
    manifest, texts = load_fixtures(root)
    qa = manifest["cells"][cell_a]["questions"]["literal"]
    qb = manifest["cells"][cell_b]["questions"]["literal"]
    user_a = f"DOCUMENT:\n{texts[cell_a]}\n\nQUESTION: {qa['question']}"
    user_b = f"DOCUMENT:\n{texts[cell_b]}\n\nQUESTION: {qb['question']}"
    prompt_a, prompt_b = srv.render(user_a), srv.render(user_b)
    key_a = qa["expected"]
    assert key_a and key_a not in prompt_b, "cell A's key must be absent from prompt B"

    def one(arm: str, id_slot: int, cache_prompt: bool, seed: int) -> None:
        data, wall = srv.completion(
            prompt_b,
            id_slot=id_slot,
            cache_prompt=cache_prompt,
            seed=seed,
            n_predict=n_predict,
            temperature=temperature,
        )
        tm = data.get("timings", {}) or {}
        out = data.get("content", "")
        c = Call(
            condition=arm,
            trial=seed,
            label="B",
            doc_tokens=manifest["cells"][cell_b]["measured_tokens"],
            id_slot_requested=id_slot,
            cache_prompt=cache_prompt,
            seed=seed,
            slot_id=data.get("id_slot"),
            prompt_n=tm.get("prompt_n"),
            cache_n=tm.get("cache_n"),
            tokens_cached=data.get("tokens_cached", tm.get("cache_n")),
            prompt_ms=tm.get("prompt_ms"),
            predicted_n=tm.get("predicted_n"),
            predicted_ms=tm.get("predicted_ms"),
            wall_s=round(wall, 4),
            raw_output=out,
            stop_type=data.get("stop_type"),
            uuid_a=key_a,
            uuid_b=qb["expected"] or "",
            entity_a=qa.get("entity", ""),
            entity_b=qb.get("entity", ""),
            asked_about=qb.get("entity", ""),
        )
        c.verdict = "LEAK_UUID" if key_a.lower() in out.lower() else classify(
            out, qb["expected"] or "", "", ""
        )
        sink.append(c)

    # Arm 0: the virgin slot answers prompt B first, before anything has ever
    # touched it. This is the floor and it must come first.
    for s in range(seeds):
        one("virgin_slot", virgin_slot, True, 1000 + s)

    # Contaminate: slot `slot` is made to hold cell A.
    for k in range(prime):
        srv.completion(prompt_a, id_slot=slot, cache_prompt=True, seed=k, n_predict=n_predict, temperature=temperature)

    for s in range(seeds):
        one("primed_slot_cached", slot, True, 1000 + s)
    for s in range(seeds):
        one("primed_slot_nocache", slot, False, 2000 + s)
    code, body = srv.slot_action(slot, "erase", "unused.bin")
    print(f"[erase slot {slot}] {code} {body[:120]}")
    for s in range(seeds):
        one("primed_slot_after_erase", slot, True, 3000 + s)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def summarize(calls: Iterable[Call]) -> str:
    rows: dict[tuple[str, int, float | None, str], list[Call]] = {}
    for c in calls:
        if c.label == "A":
            continue  # A is the plant, not the probe
        rows.setdefault((c.condition, c.doc_tokens, c.depth, c.label), []).append(c)
    lines = [
        "| condition | doc tokens | depth | probe | n | LEAK_UUID | OWN_FACT | REFUSAL | OTHER | ERROR | leak rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (cond, size, dep, label), cs in sorted(
        rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0, kv[0][3])
    ):
        n = len(cs)
        lu = sum(c.verdict == "LEAK_UUID" for c in cs)
        le = sum(c.verdict == "OWN_FACT" for c in cs)
        rf = sum(c.verdict == "REFUSAL" for c in cs)
        er = sum(c.verdict == "ERROR" for c in cs)
        ot = n - lu - le - rf - er
        lines.append(
            f"| {cond} | {size} | {dep} | {label} | {n} | {lu} | {le} | {rf} | {ot} | {er} | "
            f"{lu / n:.0%} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

ALL_CONDITIONS = [
    "same_slot_cache",
    "same_slot_nocache",
    "diff_slot",
    "aba",
    "save_restore",
    "seed_change",
    "n_keep0",
    "fresh_process",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="server root; overrides --port")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--sizes",
        default="256",
        help="comma-separated approximate document token sizes to sweep",
    )
    ap.add_argument("--conditions", default="all")
    ap.add_argument(
        "--depth",
        type=float,
        default=0.95,
        help="fraction through the document at which the planted fact sits "
        "(0.95 = inside the retrieval horizon, 0.1 = far from the question)",
    )
    ap.add_argument("--slot", type=int, default=0, help="slot to pin for same-slot conditions")
    ap.add_argument("--slot-b", type=int, default=1, help="second slot for diff_slot")
    ap.add_argument("--out", default="milestones/s2/results/r13_repro.jsonl")
    ap.add_argument("--rng-seed", type=int, default=13)
    ap.add_argument("--n-predict", type=int, default=96)
    # fresh-process control
    ap.add_argument("--server-exe", default="tools/llamacpp-rocm/llama-server.exe")
    ap.add_argument("--model", default=None, help="gguf path, required for fresh_process")
    ap.add_argument("--fresh-port", type=int, default=8099)
    ap.add_argument(
        "--server-flags",
        default="-c 8192 -np 2 -fa on -lm none --no-kv-unified --cont-batching",
        help="flags for servers this script launches itself",
    )
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--arch-report", action="store_true", help="print /props and exit")
    # fixture replay
    ap.add_argument("--replay-fixtures", default=None, help="dir with manifest.json + *.chunk.txt")
    ap.add_argument(
        "--replay-cells",
        default="s2-1024-p50,s2-2048-p50,s2-4096-p50,s2-8192-p50,s2-16384-p50,s2-32768-p50",
    )
    ap.add_argument("--replay-trials", type=int, default=3)
    # one-prompt / two-slot isolation
    ap.add_argument("--slot-ab", action="store_true", help="run the one-prompt two-slot isolation")
    ap.add_argument("--cell-a", default="s2-2048-p50", help="chunk used to contaminate the slot")
    ap.add_argument("--cell-b", default="s2-4096-p50", help="chunk the held-fixed probe prompt uses")
    ap.add_argument("--prime", type=int, default=4, help="how many times cell A is sent to the slot")
    ap.add_argument("--virgin-slot", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--erase-between",
        action="store_true",
        help="POST /slots/N?action=erase before each replay cell (needs --slot-save-path server-side)",
    )
    ap.add_argument(
        "--paired-virgin",
        action="store_true",
        help="also send every replay prompt to a slot that has only ever held that chunk",
    )
    ap.add_argument("--label", default="", help="free-text tag written into every record")
    args = ap.parse_args(argv)

    SAMPLING["n_predict"] = args.n_predict
    SAMPLING["temperature"] = args.temperature
    base = args.base_url or f"http://127.0.0.1:{args.port}"
    srv = Server(base)
    srv.wait_healthy()

    if args.arch_report:
        props = srv.props()
        print(json.dumps(props, indent=2)[:4000])
        srv.close()
        return 0

    if args.slot_ab:
        calls: list[Call] = []
        run_slot_ab(
            srv,
            calls,
            Path(args.replay_fixtures or "milestones/s2/fixtures"),
            cell_a=args.cell_a,
            cell_b=args.cell_b,
            prime=args.prime,
            seeds=args.trials,
            slot=args.slot,
            virgin_slot=args.virgin_slot,
            temperature=args.temperature,
            n_predict=args.n_predict,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for c in calls:
                rec = asdict(c)
                rec["label_tag"] = args.label
                rec["base_url"] = base
                rec["ts"] = time.time()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print()
        print(summarize(calls))
        for c in calls:
            if c.verdict == "LEAK_UUID":
                print(f"LEAK {c.condition} seed={c.seed} slot={c.slot_id} cache_n={c.cache_n}: {c.raw_output.strip()[:160]!r}")
        srv.close()
        return 0

    if args.replay_fixtures:
        calls: list[Call] = []
        findings = run_replay(
            srv,
            calls,
            Path(args.replay_fixtures),
            args.replay_cells.split(","),
            args.replay_trials,
            slot=args.slot,
            cache_prompt="nocache" not in args.conditions,
            temperature=args.temperature,
            n_predict=args.n_predict,
            name=args.conditions if args.conditions != "all" else "replay",
            erase_between=args.erase_between,
            paired_virgin=args.paired_virgin,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for c in calls:
                rec = asdict(c)
                rec["label_tag"] = args.label
                rec["base_url"] = base
                rec["ts"] = time.time()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nreplay: {len(calls)} calls, {len(findings)} with foreign strings")
        print(json.dumps(findings, indent=1)[:6000])
        srv.close()
        return 0

    conditions = ALL_CONDITIONS if args.conditions == "all" else args.conditions.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calls: list[Call] = []
    save_dir = os.environ.get("R13_SLOT_DIR", ".")

    depth = args.depth
    for size_tokens in sizes:
        # Tokens-per-line is tokenizer specific, so measure it instead of
        # assuming it: build once, tokenize, rescale, and report what was
        # actually sent. `--sizes` is a target; `doc_tokens` in the table is
        # the truth.
        n_lines = max(1, size_tokens // 20)
        for _ in range(4):
            probe = make_trial(random.Random(args.rng_seed), n_lines, depth)
            measured = srv.tokenize(compose_user(probe.doc_a, probe.entity_a))
            if abs(measured - size_tokens) <= max(24, size_tokens * 0.05):
                break
            n_lines = max(1, int(n_lines * size_tokens / max(measured, 1)))
        print(
            f"[size {size_tokens} depth {depth}] {n_lines} filler lines "
            f"-> {measured} prompt tokens"
        )

        for i in range(args.trials):
            rng = random.Random(args.rng_seed * 1000 + size_tokens * 17 + i)
            t = make_trial(rng, n_lines, depth)

            if "same_slot_cache" in conditions:
                cond_two_prompt(
                    srv, calls, t, i, name="same_slot_cache",
                    slot_a=args.slot, slot_b=args.slot, cache_a=True, cache_b=True,
                    seed_a=100 + i, seed_b=100 + i, doc_tokens=measured,
                )
            if "same_slot_nocache" in conditions:
                cond_two_prompt(
                    srv, calls, t, i, name="same_slot_nocache",
                    slot_a=args.slot, slot_b=args.slot, cache_a=False, cache_b=False,
                    seed_a=200 + i, seed_b=200 + i, doc_tokens=measured,
                )
            if "diff_slot" in conditions:
                cond_two_prompt(
                    srv, calls, t, i, name="diff_slot",
                    slot_a=args.slot, slot_b=args.slot_b, cache_a=True, cache_b=True,
                    seed_a=300 + i, seed_b=300 + i, doc_tokens=measured,
                )
            if "seed_change" in conditions:
                cond_two_prompt(
                    srv, calls, t, i, name="seed_change",
                    slot_a=args.slot, slot_b=args.slot, cache_a=True, cache_b=True,
                    seed_a=500 + i, seed_b=999_000 + i, doc_tokens=measured,
                )
            if "n_keep0" in conditions:
                cond_two_prompt(
                    srv, calls, t, i, name="n_keep0",
                    slot_a=args.slot, slot_b=args.slot, cache_a=True, cache_b=True,
                    seed_a=600 + i, seed_b=600 + i, doc_tokens=measured, n_keep_b=0,
                )
            if "save_restore" in conditions:
                fn = f"r13_{size_tokens}_{i}.bin"

                def _cycle() -> None:
                    sc, body = srv.slot_action(args.slot, "save", fn)
                    print(f"  save -> {sc} {body[:120]}")
                    sc, body = srv.slot_action(args.slot, "restore", fn)
                    print(f"  restore -> {sc} {body[:120]}")

                cond_two_prompt(
                    srv, calls, t, i, name="save_restore",
                    slot_a=args.slot, slot_b=args.slot, cache_a=True, cache_b=True,
                    seed_a=700 + i, seed_b=700 + i, doc_tokens=measured,
                    between=_cycle,
                )
            if "aba" in conditions:
                cond_aba(srv, calls, t, i, slot=args.slot, doc_tokens=measured)
            if "fresh_process" in conditions:
                if not args.model:
                    print("fresh_process requires --model; skipping", file=sys.stderr)
                else:
                    cond_fresh_process(
                        calls, t, i,
                        exe=args.server_exe, model=args.model, port=args.fresh_port,
                        extra=args.server_flags.split(),
                        log_dir=Path(args.log_dir or "."), doc_tokens=measured,
                    )

            for c in calls:
                if c.depth is None:
                    c.depth = depth

    with out_path.open("a", encoding="utf-8") as fh:
        for c in calls:
            rec = asdict(c)
            rec["label_tag"] = args.label
            rec["base_url"] = base
            rec["ts"] = time.time()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(summarize(calls))
    print()
    for c in calls:
        if c.verdict.startswith("LEAK"):
            print(
                f"LEAK {c.condition} trial={c.trial} label={c.label} slot={c.slot_id} "
                f"cache_n={c.cache_n} asked={c.asked_about!r}\n"
                f"  planted-in-A: {c.uuid_a}\n"
                f"  planted-in-B: {c.uuid_b}\n"
                f"  output: {c.raw_output.strip()[:300]!r}"
            )
    srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
