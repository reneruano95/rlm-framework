"""S2 — CACHE INSTRUMENT: is `timings.cache_n` a measurement or a claim?

THE BLOCKER THIS EXISTS TO CLEAR (`s2/DISTANCE.md` §5, §10 R8, §7 #2's
INSTRUMENT WARNING). The layout cache probe reported `cache_n` 1,378 / 1,682 /
1,374 for a second document arriving on a slot that already held the first,
with prefill collapsing to ~50 ms — against a TRUE shared token prefix of
311 / 311 / **3**. Layout C cannot reuse 1,374 tokens of a document with which
it shares three. Every token-weighted target in §7 #3 — gate (a) prefix
integrity, gate (b) >80% repeated-chunk reuse, gate (c) the root-turn monitor —
was written against that counter, so none of them can be scored until the
counter is either vindicated or replaced.

WHAT COUNTS AS TRUTH HERE, AND WHY IT IS NOT `cache_n`. For any two prompts the
number of tokens that CAN be reused is a property of the token sequences, not
of the server: it is the length of their longest common token prefix. Both
sequences are tokenized by `/tokenize` with `add_special=true` on the same
server that will serve the call, because that is measured to match what
`/completion` counts exactly (`rlm.dispatcher.ServerClient.tokenize`: pre-flight
vs served 284/285, 474/475, 1274/1275 — a constant +1 which `add_special` fixes).
The tokenized string is the EXACT rendered prompt that is then POSTed, control
tokens already neutralised — not a re-composition of it, because a truth
computed on different bytes than were sent is not a truth.

THE HYPOTHESIS THE PRIOR PROBE COULD NOT SEE. It compared each call against the
PREVIOUS PROMPT ON THE SAME SLOT, which silently assumes the prompt cache is
per-slot. §4 says exactly that ("the prompt cache is **per-slot** — there is no
cross-slot sharing"). But b10375 ships a HOST prompt cache on by default
(`--cache-idle-slots`, `--cache-ram 8192`) that saves idle slot states to host
RAM and can RESTORE them onto a different slot — the same subsystem
`s2/OCCUPANCY.md` measured as the entire latency-vs-occupancy effect. If that
is the carrier, `cache_n` is honest and the TRUTH MODEL was wrong: the reuse
ceiling is the best common prefix against EVERY prompt the process has served,
not against the last one on this slot. This run measures both truths on every
call, and settles it with a server flag: `--cache-ram 0` removes the host cache,
so under that condition cross-slot reuse must vanish if the host cache is the
carrier and must persist if it is not.

CASES (each on its own never-reused slot; within a case the slot IS reused,
which is the thing R13 forbids and the thing being measured — every answer is
still passed through R13's foreign-identifier detector and every hit reported):

  identical          P, then byte-identical P            truth = whole prompt
  prefix-only        docA, then docB                     truth = rendered head
  diverge            docA, then docA with its tail cut   truth = head + shared body
  requery            docA/q1, then docA/q2               truth = head + doc  <- gate (b)
  virgin             docA on a virgin slot, once         truth = 0           <- gate (a)
  three-docs         docA, docB, docC                    truth = head each step
  cross-slot         docA on slot s1, docA on VIRGIN s2  truth(prev) = 0, truth(history) = all
  layoutC-elsewhere  layout C: docD on s1; docE then docD on s2  <- the 1,374-vs-3 case

CALIBRATION. A cold-prefill baseline at four prompt lengths on virgin slots
that have never seen the document, three replicates each, gives tokens/second
with a spread — which is what turns `timings.prompt_ms` into a token-valued
reuse estimate with error bars rather than a vibe.

The server is launched and shut down by this script, one launch per condition,
with the §4 pinned flags and ROCBLAS_USE_HIPBLASLT=1.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `uv run s2/run_cache_instrument.py`
    sys.path.insert(0, str(REPO_ROOT))

from rlm.config import load_config  # noqa: E402
from rlm.dispatcher import (  # noqa: E402
    chat_control_markers,
    compose_leaf_user,
    neutralise_control_tokens,
)
from rlm.leakcheck import ChunkIndex  # noqa: E402
from s2.run_occupancy import (  # noqa: E402
    BASE_FLAGS,
    FILLER,
    TimedLeaf,
    buckets,
    launch,
    shutdown,
)

S2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = S2_DIR / "results"
RUNS_PATH = RESULTS_DIR / "cache_instrument.jsonl"

#: Short on purpose. This experiment prices PREFILL accounting; decode is pure
#: cost here, and 32 tokens is more than enough for the leaf to state an
#: identifier and for R13's detector to have something to find.
N_PREDICT = 32

QUESTION_1 = "What is the record identifier stated in this document?"
#: A DIFFERENT question about the SAME document — gate (b)'s exact case. It has
#: to be a different string or `requery` degenerates into `identical`.
QUESTION_2 = ("Does this document state a record identifier? Answer with the "
              "identifier if it does.")

#: The four calibration prompt lengths, in CHUNK tokens. The largest is bounded
#: by one slot's capacity at the pinned `-np 128 -c 327680` (2,560 tokens) minus
#: the rendered head (~311) and the question.
CALIBRATION_TOKENS = (320, 640, 1280, 1900)

#: The window the re-derived geometry is heading for (§7 #2 v0.2.9). Every case
#: document is this long, so the tables price the configuration that will ship
#: rather than the one being retired.
CASE_CHUNK_TOKENS = 640


# --------------------------------------------------------------------------- #
# Corpus — deterministic, and every document carries exactly one identifier so
# R13's detector has an unambiguous foreign string to find.
# --------------------------------------------------------------------------- #


def doc_identifier(key: str) -> str:
    h = hashlib.sha256(f"s2-cacheinst-{key}".encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def doc_text(key: str, sentences: int) -> str:
    """One document: neutral filler with its identifier stated once near the top.

    Seeded by `key`, so two documents with different keys share only the filler
    vocabulary, never a long token prefix — which is what makes `prefix-only`'s
    truth equal to the rendered head and not more.
    """
    rng = random.Random(hashlib.sha256(key.encode()).digest())
    body = [rng.choice(FILLER) for _ in range(sentences)]
    body.insert(1, f"Record identifier: {doc_identifier(key)}.")
    return "\n".join(body)


async def doc_of_tokens(leaf: "TokenLeaf", key: str, target: int) -> str:
    """A document whose CHUNK BODY is `target` tokens, +-8, measured on the
    server that will serve it. Bisection on sentence count, then a trim: the
    experiment's whole point is that token counts are exact, so a document
    described as 640 tokens has to be 640 tokens."""
    lo, hi = 1, 8
    while len((await leaf.tokens(doc_text(key, hi), add_special=False))) < target:
        hi *= 2
        if hi > 4096:
            raise RuntimeError("document target unreachable")
    while lo < hi:
        mid = (lo + hi) // 2
        n = len(await leaf.tokens(doc_text(key, mid), add_special=False))
        if n < target:
            lo = mid + 1
        else:
            hi = mid
    text = doc_text(key, lo)
    lines = text.split("\n")
    while len(lines) > 2 and len(await leaf.tokens("\n".join(lines),
                                                  add_special=False)) > target + 8:
        lines.pop()
    return "\n".join(lines)


def truncate_doc(text: str, keep_fraction: float) -> str:
    """`text` with its tail replaced by different filler — a document that
    shares a long body prefix with its parent and then diverges.

    Replaced rather than deleted so the two documents are the same LENGTH: a
    shorter second prompt would let a reuse figure look right for the wrong
    reason (everything shared, nothing after it).
    """
    lines = text.split("\n")
    keep = max(2, int(len(lines) * keep_fraction))
    rng = random.Random(hashlib.sha256(f"diverge-{text[:64]}".encode()).digest())
    tail = [rng.choice(FILLER) for _ in range(len(lines) - keep)]
    return "\n".join(lines[:keep] + tail)


# --------------------------------------------------------------------------- #
# The transport: `run_occupancy.TimedLeaf` plus the one thing this experiment
# needs that a latency experiment did not — the TOKEN LIST of the exact bytes
# sent, and the two instruction layouts §4 names.
# --------------------------------------------------------------------------- #


@dataclass
class TokenLeaf(TimedLeaf):
    """A `TimedLeaf` that hands back token sequences, not just counts."""

    prefix_body_tokens: int | None = None

    async def tokens(self, text: str, *, add_special: bool = False) -> list[int]:
        resp = await self.client().post(
            f"{self.base_url}/tokenize",
            json={"content": text, "add_special": add_special})
        resp.raise_for_status()
        toks = resp.json().get("tokens", [])
        if text and not toks:
            raise RuntimeError("/tokenize returned 0 tokens for non-empty input")
        return toks

    def compose_layout(self, *, question: str, chunk: str,
                       layout: str = "A") -> list[dict[str, str]]:
        """§4's shipped layout A through production's own composer; layout C
        (`[chunk][prefix][question]`, no system head) composed exactly as
        `s2.leafcall.PinnedLeafCaller.compose` does, because the 1,374-vs-3
        anomaly was recorded under C and a paraphrase of C would not reproduce
        it."""
        if layout == "A":
            return self.compose(question=question, chunk=chunk)
        body = f"{chunk}\n\n{self.system_prefix}\n\n{question}"
        return [{"role": "user",
                 "content": neutralise_control_tokens(body, self.markers)}]


def lcp(a: list[int], b: list[int]) -> int:
    """Longest common token prefix — the ONLY number in this experiment that
    does not come from the server's own accounting."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------- #
# The script of calls.
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    case: str
    step: int
    slot_key: str
    doc_key: str
    question: str
    layout: str = "A"
    #: What this step is FOR, in one word, so the report can group without
    #: re-deriving intent from the case name.
    role: str = ""
    #: Chunk-body tokens for this step's document. Only `requery-len` varies it.
    chunk_tokens: int = 0

    def tokens_target(self) -> int:
        return self.chunk_tokens or CASE_CHUNK_TOKENS


def case_script(rep: int) -> list[Step]:
    """One replicate's worth of steps. `rep` disjoints the documents AND the
    slots, so no replicate can see another's cache."""
    p = f"r{rep}"
    S = Step
    steps: list[Step] = [
        S("identical", 0, f"{p}-identical", f"{p}-A", QUESTION_1, role="cold"),
        S("identical", 1, f"{p}-identical", f"{p}-A", QUESTION_1, role="repeat"),

        S("prefix-only", 0, f"{p}-prefixonly", f"{p}-B", QUESTION_1, role="cold"),
        S("prefix-only", 1, f"{p}-prefixonly", f"{p}-C", QUESTION_1, role="newdoc"),

        S("diverge", 0, f"{p}-diverge", f"{p}-D", QUESTION_1, role="cold"),
        S("diverge", 1, f"{p}-diverge", f"{p}-D/tail", QUESTION_1, role="diverged"),

        S("requery", 0, f"{p}-requery", f"{p}-E", QUESTION_1, role="cold"),
        S("requery", 1, f"{p}-requery", f"{p}-E", QUESTION_2, role="requery"),

        S("virgin", 0, f"{p}-virgin", f"{p}-F", QUESTION_1, role="firstsight"),

        S("three-docs", 0, f"{p}-three", f"{p}-G", QUESTION_1, role="cold"),
        S("three-docs", 1, f"{p}-three", f"{p}-H", QUESTION_1, role="newdoc"),
        S("three-docs", 2, f"{p}-three", f"{p}-I", QUESTION_1, role="newdoc"),

        # The decisive test of the host-prompt-cache hypothesis: the SAME
        # document, second time on a slot that has never served anything.
        S("cross-slot", 0, f"{p}-cross-a", f"{p}-J", QUESTION_1, role="cold"),
        S("cross-slot", 1, f"{p}-cross-b", f"{p}-J", QUESTION_1, role="virgin-slot-repeat"),

        # `s2/DISTANCE.md` §5's layout-C row, reconstructed: the document was
        # served ELSEWHERE first, then arrives behind a different document on
        # another slot, sharing ~3 tokens with what that slot last held.
        S("layoutC-elsewhere", 0, f"{p}-lc-a", f"{p}-K", QUESTION_1, "C", "elsewhere"),
        S("layoutC-elsewhere", 1, f"{p}-lc-b", f"{p}-L", QUESTION_1, "C", "cold"),
        S("layoutC-elsewhere", 2, f"{p}-lc-b", f"{p}-K", QUESTION_1, "C", "seen-elsewhere"),

        # `cross-slot` (above) and `layoutC-elsewhere` differ in exactly two
        # things: whether a task intervened between the document's two sightings
        # and whether the receiving slot was virgin. The smoke run had the first
        # restore and the second not, so both are separated here: an intervening
        # task, and a VIRGIN receiving slot. `--cache-idle-slots` saves idle
        # slots "on new task", so if the lag is the mechanism this restores and
        # `cross-slot` does not.
        S("cross-slot-lag", 0, f"{p}-lag-a", f"{p}-M", QUESTION_1, role="cold"),
        S("cross-slot-lag", 1, f"{p}-lag-b", f"{p}-N", QUESTION_1, role="intervening"),
        S("cross-slot-lag", 2, f"{p}-lag-c", f"{p}-M", QUESTION_1,
          role="virgin-slot-lagged"),
    ]
    return steps


#: Where the second prompt diverges from the first, as a fraction of the
#: document's LINES. The smoke run's two partial-match cases reported 454 of a
#: true 693 and 457 of a true 954 — two very different truths landing on almost
#: the same reported figure, which is the signature of a QUANTISED reuse rule
#: rather than of a proportional one. One pair of calls per point, each pair on
#: its own never-reused slot, decides between them.
DIVERGE_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def lenprobe_script(rep: int) -> list[Step]:
    """`requery` at four prompt LENGTHS, one never-reused slot per pair.

    The case suite holds prompt length fixed at the candidate 640-token window,
    which cannot separate two rules that fit its numbers equally well: "reuse is
    capped at a fixed number of tokens" and "reuse is capped at the prompt
    length minus roughly one `-ub` batch". Four lengths separate them in one
    pass — the first predicts a flat `cache_n`, the second a `cache_n` that
    rises with the prompt.
    """
    p = f"L{rep}"
    steps: list[Step] = []
    for tk in CALIBRATION_TOKENS:
        key = f"{p}-{tk}"
        steps.append(Step("requery-len", 0, f"{p}-rq{tk}", key, QUESTION_1,
                          role=f"cold@{tk}", chunk_tokens=tk))
        steps.append(Step("requery-len", 1, f"{p}-rq{tk}", key, QUESTION_2,
                          role=f"requery@{tk}", chunk_tokens=tk))
    return steps


def diverge_script(rep: int) -> list[Step]:
    p = f"r{rep}"
    steps: list[Step] = []
    for i, frac in enumerate(DIVERGE_POINTS):
        key = f"{p}-DS{i}"
        steps.append(Step("diverge-sweep", 0, f"{p}-ds{i}", key, QUESTION_1,
                          role=f"cold@{frac:.1f}"))
        steps.append(Step("diverge-sweep", 1, f"{p}-ds{i}", f"{key}/keep{frac:.1f}",
                          QUESTION_1, role=f"diverged@{frac:.1f}"))
    return steps


# --------------------------------------------------------------------------- #
# The run.
# --------------------------------------------------------------------------- #


async def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(Path(args.config))
    leaf_cfg = cfg.servers.leaf
    exe = str(Path(leaf_cfg.backend_dir) / "llama-server.exe")
    prefix = cfg.prompt_registry().load().leaf_prefix()
    sampling = cfg.scaffold.sampling.leaf

    leaf = TokenLeaf(
        base_url=f"http://127.0.0.1:{args.port}",
        system_prefix=prefix,
        max_predict=N_PREDICT,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        seed=1,
        enable_thinking=cfg.scaffold.leaf.enable_thinking,
    )

    extra = args.extra.split() if args.extra else []
    argv = [exe, "-m", str(leaf_cfg.model), "--port", str(args.port),
            "-c", str(args.ctx), "-np", str(args.np), *BASE_FLAGS, *extra]
    # Last `-ub` on the command line wins, which is how a condition overrides
    # BASE_FLAGS' pinned 512.
    ub_in_force = int(argv[len(argv) - 1 - argv[::-1].index("-ub") + 1])

    proc: subprocess.Popen | None = None
    records: list[dict[str, Any]] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_PATH.open("a", encoding="utf-8")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        if not args.no_launch:
            print(f"[{args.condition}] launching: {' '.join(argv)}", flush=True)
            proc = await launch(argv, Path(args.log), leaf)
        elif not await leaf.health_ok():
            raise SystemExit("no healthy server and --no-launch given")
        await leaf.prepare()
        leaf.prefix_body_tokens = len(await leaf.tokens(prefix, add_special=False))
        print(f"[{args.condition}] rendered head {leaf.prefix_tokens} tokens, "
              f"prefix body {leaf.prefix_body_tokens}", flush=True)

        # ---- corpus, built on the server that will serve it ----------------
        docs: dict[str, str] = {}
        needed: set[tuple[str, int]] = set()
        script: list[Step] = []
        for rep in range(1, args.reps + 1):
            script.extend(case_script(rep))
        for rep in range(1, args.diverge_reps + 1):
            script.extend(diverge_script(rep))
        for rep in range(1, args.len_reps + 1):
            script.extend(lenprobe_script(rep))
        for st in script:
            if "/" in st.doc_key:      # derived below from its parent
                continue
            needed.add((st.doc_key, st.tokens_target()))
        for i in range(args.cal_reps):
            for tk in CALIBRATION_TOKENS:
                needed.add((f"cal{i}-{tk}", tk))
        for key, tk in sorted(needed):
            docs[key] = await doc_of_tokens(leaf, key, tk)
        # Derived documents: the same parent with its TAIL replaced, so the two
        # share a body prefix and then diverge at a known point.
        for st in script:
            if "/" not in st.doc_key or st.doc_key in docs:
                continue
            parent, suffix = st.doc_key.split("/", 1)
            frac = 0.6 if suffix == "tail" else float(suffix.removeprefix("keep"))
            docs[st.doc_key] = truncate_doc(docs[parent], frac)
        index = ChunkIndex.from_chunks(docs)
        print(f"[{args.condition}] corpus: {len(docs)} documents", flush=True)

        # ---- slot allocation: never reused across cases or replicates ------
        slots: dict[str, int] = {}
        next_slot = 0

        def slot_for(key: str) -> int:
            nonlocal next_slot
            if key not in slots:
                if next_slot >= args.np:
                    raise RuntimeError("slot pool exhausted")
                slots[key] = next_slot
                next_slot += 1
            return slots[key]

        # ---- history, the two truths --------------------------------------
        history: list[tuple[str, list[int]]] = []   # (slot_key, tokens) in order
        ordinal = 0

        async def one_call(*, case: str, step: int, slot_key: str, doc_key: str,
                           question: str, layout: str, role: str, rep: int,
                           chunk_tokens: int) -> dict[str, Any]:
            return await one_call_messages(
                messages=leaf.compose_layout(question=question,
                                             chunk=docs[doc_key], layout=layout),
                case=case, step=step, slot_key=slot_key, doc_key=doc_key,
                question_id=("q1" if question == QUESTION_1 else "q2"),
                role=role, rep=rep, chunk_tokens=chunk_tokens, layout=layout,
                sent=f"{docs[doc_key]}\n\n{question}")

        async def one_call_messages(*, messages: list[dict[str, str]], case: str,
                                    step: int, slot_key: str, doc_key: str,
                                    question_id: str, role: str, rep: int,
                                    chunk_tokens: int, layout: str = "A",
                                    sent: str | None = None) -> dict[str, Any]:
            nonlocal ordinal
            slot = slot_for(slot_key)
            rendered = await leaf.render(messages)
            toks = await leaf.tokens(rendered, add_special=True)

            prev_same_slot = next((t for k, t in reversed(history)
                                   if k == slot_key), None)
            truth_prev_slot = lcp(toks, prev_same_slot) if prev_same_slot else 0
            truth_best_slot = max(
                [lcp(toks, t) for k, t in history if k == slot_key] or [0])
            truth_best_any = max([lcp(toks, t) for _, t in history] or [0])

            rec_t = await leaf.ask(rendered, id_slot=slot)
            ordinal += 1
            verdict = index.foreign(
                rec_t["content"],
                sent=sent if sent is not None else "\n\n".join(
                    m["content"] for m in messages))
            rec: dict[str, Any] = {
                "run_id": run_id, "condition": args.condition,
                "extra": args.extra, "np": args.np, "ctx": args.ctx,
                # The reuse law's one free parameter is `-ub`, so the flag that
                # was actually in force is a first-class field, not something a
                # reader has to parse back out of `extra`.
                "ub": ub_in_force,
                "ordinal": ordinal, "rep": rep, "case": case, "step": step,
                "role": role, "layout": layout,
                "slot_key": slot_key, "requested_slot": slot,
                "returned_slot": rec_t["id_slot_returned"],
                "slot_ok": rec_t["id_slot_returned"] == slot,
                "doc_key": doc_key, "question_id": question_id,
                "chunk_tokens_target": chunk_tokens,
                "prompt_tokens_true": len(toks),
                "prompt_sha256": hashlib.sha256(
                    rendered.encode("utf-8")).hexdigest(),
                "head_tokens": leaf.prefix_tokens,
                "prefix_body_tokens": leaf.prefix_body_tokens,
                "truth_lcp_prev_same_slot": truth_prev_slot,
                "truth_lcp_best_same_slot": truth_best_slot,
                "truth_lcp_best_any_slot": truth_best_any,
                "reported_cache_n": rec_t["cache_n"],
                "reported_prompt_n": rec_t["prompt_n"],
                "bookkeeping_ok": (rec_t["cache_n"] + rec_t["prompt_n"]
                                   == len(toks)),
                "prompt_ms": rec_t["prompt_ms"],
                "predicted_n": rec_t["predicted_n"],
                "predicted_ms": rec_t["predicted_ms"],
                "stop_type": rec_t["stop_type"],
                "truncated": rec_t["truncated"],
                "content": rec_t["content"],
                "leak_detected": verdict.detected,
                "leak_detail": verdict.detail,
                "answer_correct": doc_identifier(doc_key.split("/")[0])
                                  in rec_t["content"],
                **buckets(rec_t),
            }
            history.append((slot_key, toks))
            out.write(json.dumps(rec) + "\n")
            out.flush()
            records.append(rec)
            print(f"[{args.condition}] {case:18s} s{step} {role:19s} "
                  f"slot {slot:3d}{'' if rec['slot_ok'] else ' !!MISMATCH'} "
                  f"cache_n {rec_t['cache_n']:5d} | true(prev-slot) "
                  f"{truth_prev_slot:5d} true(any) {truth_best_any:5d} | "
                  f"prefill {rec_t['prompt_ms']:8.1f} ms of "
                  f"{len(toks)} tok"
                  f"{' LEAK' if verdict.detected else ''}", flush=True)
            return rec

        # ---- phase 1: cold-prefill calibration -----------------------------
        # First, and on documents no other phase uses, so every one of these is
        # genuinely first-sight for the process as well as for the slot.
        for i in range(args.cal_reps):
            for tk in CALIBRATION_TOKENS:
                key = f"cal{i}-{tk}"
                await one_call(case="calibration", step=i, slot_key=f"cal-{key}",
                               doc_key=key, question=QUESTION_1, layout="A",
                               role="cold-baseline", rep=i + 1, chunk_tokens=tk)

        # ---- phase 2: the cases --------------------------------------------
        for rep in range(1, args.reps + 1):
            for st in case_script(rep):
                await one_call(case=st.case, step=st.step, slot_key=st.slot_key,
                               doc_key=st.doc_key, question=st.question,
                               layout=st.layout, role=st.role, rep=rep,
                               chunk_tokens=st.tokens_target())

        # ---- phase 3: where does a PARTIAL match land? ---------------------
        for rep in range(1, args.diverge_reps + 1):
            for st in diverge_script(rep):
                await one_call(case=st.case, step=st.step, slot_key=st.slot_key,
                               doc_key=st.doc_key, question=st.question,
                               layout=st.layout, role=st.role, rep=rep,
                               chunk_tokens=st.tokens_target())

        # ---- phase 5: §7 #3 (c)'s case — a conversation that GROWS ---------
        # The root is the same hybrid architecture (§7 #3c), so its turn-by-turn
        # growth can be measured here without a second server: a rendered
        # [system, user] prompt is a strict token prefix of the rendered
        # [system, user, assistant, user'] that follows it, which is exactly
        # what "only the new observation should prefill" means. This is an
        # EXTENSION, not a divergence — the distinction the checkpoint law makes
        # everything turn on.
        for rep in range(1, args.root_turns_reps + 1):
            slot_key = f"rt{rep}"
            rt_key = f"rt-doc{rep}"
            if rt_key not in docs:
                docs[rt_key] = await doc_of_tokens(leaf, rt_key,
                                                   CASE_CHUNK_TOKENS)
                index = ChunkIndex.from_chunks(docs)
            history_msgs: list[dict[str, str]] = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": neutralise_control_tokens(
                    compose_leaf_user(QUESTION_1, docs[rt_key]), leaf.markers)},
            ]
            for turn in range(args.root_turns):
                rec = await one_call_messages(
                    messages=list(history_msgs), case="root-turn", step=turn,
                    slot_key=slot_key, doc_key=rt_key, question_id="turn",
                    role=f"turn{turn}", rep=rep,
                    chunk_tokens=CASE_CHUNK_TOKENS)
                history_msgs.append({"role": "assistant", "content": rec["content"]})
                history_msgs.append({"role": "user", "content":
                                     f"Restate that answer. Attempt {turn + 1}."})

        # ---- phase 4: does the cap move with the prompt LENGTH? ------------
        for rep in range(1, args.len_reps + 1):
            for st in lenprobe_script(rep):
                await one_call(case=st.case, step=st.step, slot_key=st.slot_key,
                               doc_key=st.doc_key, question=st.question,
                               layout=st.layout, role=st.role, rep=rep,
                               chunk_tokens=st.tokens_target())
    finally:
        out.close()
        await leaf.aclose()
        shutdown(proc)
        print(f"[{args.condition}] server shut down", flush=True)

    return {"condition": args.condition, "records": records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--np", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=327680)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--diverge-reps", type=int, default=1)
    ap.add_argument("--len-reps", type=int, default=0)
    ap.add_argument("--root-turns", type=int, default=4,
                    help="turns per root-turn conversation (§7 #3c)")
    ap.add_argument("--root-turns-reps", type=int, default=0)
    ap.add_argument("--cal-reps", type=int, default=-1)
    ap.add_argument("--log", default="traces/logs/cacheinst.log")
    ap.add_argument("--no-launch", action="store_true")
    args = ap.parse_args()
    if args.cal_reps < 0:
        args.cal_reps = args.reps
    result = asyncio.run(run_condition(args))
    recs = result["records"]
    bad = [r for r in recs if not r["slot_ok"]]
    leaks = [r for r in recs if r["leak_detected"]]
    book = [r for r in recs if not r["bookkeeping_ok"]]
    print(f"\n[{args.condition}] {len(recs)} calls, {len(bad)} slot mismatches, "
          f"{len(leaks)} leaks, {len(book)} bookkeeping violations")


if __name__ == "__main__":
    main()
