# Tokenizer checks — double-BOS and the preflight off-by-one

**Date:** 2026-08-14 · **Build:** b10375 · **Model:** Qwen3.6-35B-A3B-UD-Q4_K_M
**Scripts:** `s2/bos_check.py`, `s2/offby1_check.py` · **Server:** `-c 32768 -np 4 --cache-ram 0 --no-cache-idle-slots`

Prompted by the question of whether this project's problems are tokenizer-level.
Two specific, cheap hypotheses were testable; both come back negative, and the
second **overturns a finding already recorded in the spec**.

## 1. Double BOS — NOT PRESENT

If `/completion` prepends a beginning-of-sequence token *and* the chat template
already emits one, the model sees a malformed sequence start. This is a known
llama.cpp quality-killer and had never been checked here.

| check | result |
|---|---|
| `chat_template` contains `bos_token` | **False** (8,057-char template) |
| first token of the rendered prompt | id **248045** = `<\|im_start\|>` — not a BOS |
| `tokenize(add_special=False)` first 5 | `[248045, 8678, 198, 2523, 8385]` |
| `tokenize(add_special=True)` first 5 | `[248045, 8678, 198, 2523, 8385]` — identical |

**Verdict: no double BOS.** This tokenizer emits no BOS at all, so `add_special`
is a **no-op** — the ChatML control token `<|im_start|>` does the framing work a
BOS would do elsewhere.

## 2. The preflight off-by-one — DOES NOT REPRODUCE

Spec v0.2.x recorded a "constant +1" between the preflight count and what the
server served (284/474/1274 preflight vs 285/475/1275 served), attributed to
`/tokenize` defaulting to `add_special=false` while `/completion` counts with it
true, and fixed by passing `add_special=True` in the preflight.

Measured directly against the rendered prompt actually sent:

| chunk words | tok(add_special=False) | tok(add_special=True) | served (`prompt_n + cache_n`) | served − tok(True) |
|---:|---:|---:|---:|---:|
| 5 | 35 | 35 | 35 | **0** |
| 50 | 165 | 165 | 165 | **0** |
| 500 | 1,915 | 1,915 | 1,915 | **0** |

**Verdict: there is no off-by-one, and there was no `add_special` mechanism that
could have produced one** — the flag changes nothing on this tokenizer. The
recorded +1 does not reproduce against the rendered string.

Stated carefully, because the original numbers (284/474/1274) differ from any
measured here: this does not prove the earlier observation was fabricated. The
likeliest reading is that it compared a count taken over a *different string*
than the one served — e.g. the raw user prompt rather than the template-rendered
prompt, which differ by exactly the template wrapper. What it does establish is
that **the admission boundary was never off by one**, and that the
`add_special=True` change fixed a non-problem. That change is harmless and is
retained (it is correct in principle, and would matter on a tokenizer that does
emit a BOS), but the spec's claim that "a rendered prompt of exactly
`slot_capacity_tokens` is admitted and then occupies cap+1" is **withdrawn**.

## 3. Incidental: the server has its own admission backstop

An over-length request is refused cleanly rather than truncated:

```
HTTP 400 {"error":{"code":400,
  "message":"request (8915 tokens) exceeds the available context size (8192 tokens)...",
  "type":"exceed_context_size_error","n_prompt_tokens":8915,"n_ctx":8192}}
```

The scaffold's C4 preflight is therefore a *courtesy* check that produces a
better trace row, not the only thing standing between a long prompt and a
corrupted slot. Worth knowing: if the preflight is ever wrong, the failure mode
is a clean 400, not silent truncation.

## What this does and does not say about the tokenizer hypothesis

It rules out the two concrete mechanisms by which tokenization could have caused
this project's symptoms. It does **not** rule out the tokenizer as a factor in
R14, whose signature — a correct decode emitting its own true identifier with
tokens dropped mid-string — remains the one symptom in this project that is
genuinely token-level. But R14 is concurrency-dependent, and tokenization is a
deterministic per-request transform, so any tokenizer involvement there would be
downstream of a scheduling or batching defect rather than its cause.

The stronger remaining hypothesis for the ~1,000-token horizon is architectural:
this leaf is hybrid, only ~25% of its layers carry true KV attention, and the
rest compress history into a **fixed-size, context-independent** recurrent state
(measured 62.8 MiB/slot regardless of context length). A fixed-capacity state
holding an ever-longer prompt is the shape that produces a hard horizon, and it
would degrade facts and instructions identically — which is exactly the
"one horizon, two distances" result in §4. **The discriminating test is the
distance ladder on a verified full-attention model** (`gemma-4-12B`, already
confirmed full-attention as the R13 control): if the cliff moves or vanishes,
the horizon belongs to the model and `fallback_leaf` becomes the lever.
