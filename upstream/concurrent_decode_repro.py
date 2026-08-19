#!/usr/bin/env python3
"""llama-server: answer quality collapses when requests are decoded concurrently.

Self-contained reproducer -- standard library plus `httpx`. It generates its own
corpus, applies the chat template through the server's own /apply-template, and
scores by substring against a UUID that is literally present in the document, so
a wrong answer is unambiguous.

THE EXPERIMENT

  N documents, each ~1,000 tokens of neutral filler stating exactly one
  UUID-shaped record identifier near the top. One question per document:
  "What is the record identifier stated in this document?" -- the answer is
  present verbatim in the text.

  The same N calls are issued at concurrency 1, 2, 4 and 8. Nothing changes
  between conditions except how many are in flight.

  Every call is pinned to a slot no other call in the run has used, so that
  slot reuse cannot contribute (this is why the server needs -np >= total
  calls). Correctness is `identifier in answer`.

  A server whose decode is independent per sequence must score the same at
  every concurrency.

DEGENERACY RULES (reported alongside, because the failures are not wrong
answers so much as broken decodes):
  STUB      < 12 non-space characters, or predicted_n <= 2
  LOOP      a word repeated >= 4 times consecutively
  CHARLOOP  a character repeated >= 6 times consecutively
  NONLATIN  a codepoint above Latin Extended-B (corpus and prompt are ASCII)
  EOSCUT    stopped on eos mid-phrase (ends in neither terminal punctuation
            nor a complete identifier)

USAGE

  python concurrent_decode_repro.py --base http://127.0.0.1:8081 --out r.jsonl

  Launch the server with enough slots for every call to get a fresh one, e.g.
  32 calls x 4 levels = 128:  -np 128 -c 327680
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import string
import sys
import time
import uuid

import httpx

SYSTEM = ("You answer questions strictly from the DOCUMENT in the user message. "
          "Answer with only the value asked for.")

POOL = [
    "The tallyman recorded the consignment against the standing account.",
    "No amendment was entered before the cut-off on the following day.",
    "A duplicate waybill was retained by the wharf office for reference.",
    "The stated weight was confirmed twice against the yard standard.",
    "Delivery was witnessed by the duty overseer and countersigned.",
    "The parcel remained in bonded storage pending further instruction.",
    "A short note was appended to the margin of the receiving column.",
    "The carrier acknowledged the transfer without further comment.",
]

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def make_document(index, target_tokens, ntokens):
    """Deterministic per index: one identifier near the top, neutral filler."""
    rng = random.Random(1000 + index)
    ident = str(uuid.UUID(int=rng.getrandbits(128)))
    lines = ["Record %04d of the wharf register." % (index + 1),
             "The record identifier stated for this entry is %s." % ident]
    while True:
        lines.append(rng.choice(POOL))
        if len(lines) % 12 == 0 and ntokens("\n".join(lines)) >= target_tokens:
            break
        if len(lines) > 4000:
            break
    return {"index": index, "ident": ident, "text": "\n".join(lines)}


def degeneracy(answer, predicted_n, stop_type, ident):
    flags = []
    stripped = "".join(answer.split())
    if len(stripped) < 12 or (predicted_n is not None and predicted_n <= 2):
        flags.append("STUB")
    words = answer.split()
    for i in range(len(words) - 3):
        if words[i] and all(words[i + k] == words[i] for k in range(1, 4)):
            flags.append("LOOP")
            break
    for i in range(len(answer) - 5):
        if answer[i] not in " \n" and all(answer[i + k] == answer[i] for k in range(1, 6)):
            flags.append("CHARLOOP")
            break
    if any(ord(ch) > 0x024F for ch in answer):
        flags.append("NONLATIN")
    if stop_type == "eos":
        tail = answer.rstrip()
        if tail and tail[-1] not in ".!?\"'" and not UUID_RE.search(tail[-40:]):
            flags.append("EOSCUT")
    return flags


class Server:
    def __init__(self, base, timeout=900.0):
        self.base = base
        self.sync = httpx.Client(base_url=base, timeout=timeout)

    def props(self):
        r = self.sync.get("/props")
        r.raise_for_status()
        return r.json()

    def ntokens(self, text):
        r = self.sync.post("/tokenize", json={"content": text})
        r.raise_for_status()
        return len(r.json()["tokens"])

    def render(self, user):
        r = self.sync.post("/apply-template", json={
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return r.json()["prompt"]


async def one_call(client, prompt, slot, seed):
    r = await client.post("/completion", json={
        "prompt": prompt, "n_predict": 64, "temperature": 0.3, "top_p": 0.9,
        "seed": seed, "cache_prompt": True, "id_slot": slot, "stream": False})
    r.raise_for_status()
    return r.json()


async def run_level(base, prompts, slots, concurrency, seed):
    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async with httpx.AsyncClient(base_url=base, timeout=900.0,
                                 limits=httpx.Limits(max_connections=concurrency + 4)) as client:
        async def worker(i):
            async with sem:
                t0 = time.time()
                d = await one_call(client, prompts[i], slots[i], seed)
                results[i] = (d, time.time() - t0)

        await asyncio.gather(*(worker(i) for i in range(len(prompts))))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--out", default="concurrent_decode_results.jsonl")
    ap.add_argument("--calls", type=int, default=32)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--doc-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    srv = Server(args.base)
    props = srv.props()
    build = props.get("build_info", "?")
    slots_avail = props.get("total_slots", 0)
    levels = [int(x) for x in args.levels.split(",")]
    need = args.calls * len(levels)
    if slots_avail < need:
        print("need >= %d slots so every call gets a fresh one (server reports %d); "
              "relaunch with -np %d" % (need, slots_avail, need), file=sys.stderr)
        return 2

    print("build %s, %d slots -- building %d documents" % (build, slots_avail, args.calls),
          flush=True)
    docs = [make_document(i, args.doc_tokens, srv.ntokens) for i in range(args.calls)]
    prompts = [srv.render("DOCUMENT:\n%s\n\nQUESTION: What is the record identifier "
                          "stated in this document?" % d["text"]) for d in docs]
    print("  document tokens ~%d, rendered ~%d"
          % (srv.ntokens(docs[0]["text"]), srv.ntokens(prompts[0])), flush=True)

    out = open(args.out, "w", encoding="utf-8")
    slot_cursor = 0
    summary = []
    for conc in levels:
        slots = list(range(slot_cursor, slot_cursor + args.calls))
        slot_cursor += args.calls
        t0 = time.time()
        results = asyncio.run(run_level(args.base, prompts, slots, conc, args.seed))
        wall = time.time() - t0
        correct = degen = 0
        for i, (d, dt) in enumerate(results):
            ans = d["content"].strip()
            ok = docs[i]["ident"] in ans
            flags = degeneracy(ans, d.get("tokens_predicted"), d.get("stop_type"),
                               docs[i]["ident"])
            correct += ok
            degen += bool(flags)
            out.write(json.dumps({
                "concurrency": conc, "call": i, "slot_requested": slots[i],
                "slot_served": d.get("id_slot"), "correct": ok, "degenerate": flags,
                "predicted_n": d.get("tokens_predicted"), "stop_type": d.get("stop_type"),
                "cache_n": d.get("timings", {}).get("cache_n"),
                "answer": ans[:300], "wall_s": round(dt, 2), "build": build}) + "\n")
        out.flush()
        rate = correct / len(results)
        summary.append((conc, correct, len(results), degen, wall, correct / wall))
        print("concurrency %-2d  correct %3d/%-3d (%5.1f%%)  degenerate %3d  "
              "wall %6.1fs  correct/s %.3f"
              % (conc, correct, len(results), 100 * rate, degen, wall, correct / wall),
              flush=True)
    out.close()

    print("\nbuild %s" % build)
    print("%-12s %-14s %-12s %-10s %s" % ("concurrency", "correct", "degenerate",
                                          "wall (s)", "correct/s"))
    for conc, c, n, dg, wall, cps in summary:
        print("%-12d %-14s %-12d %-10.1f %.3f" % (conc, "%d/%d" % (c, n), dg, wall, cps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
