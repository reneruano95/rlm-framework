# Server: a reused slot answers with a previous request's document content (survives cold prefill and `action=erase`)

## Summary

On `llama-server`, a slot that has previously held one prompt can inject that
prompt's content into the response to a later, unrelated prompt routed to the
same slot. The effect:

* survives a full re-prefill in which the server itself reports
  `timings.cache_n == 0`;
* survives `POST /slots/{id}?action=erase`, which returns 200 and reports a
  correct non-zero `n_erased`;
* grows with the number of distinct prompts the slot has previously held;
* disappears completely when each prompt is given a slot that has held nothing
  else — same process, same weights, same prompt bytes, same sampler;
* occurs on both a hybrid attention/SSM model and a pure-attention model, so it
  does not appear to be related to recurrent-state handling.

Paired measurement, one process, byte-identical prompts issued to both arms:

| arm | responses containing another prompt's content | n |
|---|---|---|
| one shared pinned slot | 24 | 54 |
| one slot per document | 0 | 54 |

Fisher exact two-sided p = 4.4e-9.

## Build

```
b10375-ba360efe1
```
(`build_info` from `GET /props`.) Backend: HIP/ROCm, Windows 11, gfx1151.

## Models

Reproduced on both:

* `Qwen3.6-35B-A3B` Q4_K_M — GGUF arch `qwen35moe`, `block_count = 40`,
  `full_attention_interval = 4`, `ssm.state_size = 128`, `ssm.conv_kernel = 4`
  (hybrid attention + SSM).
* `gemma-4-12B-it` Q8_0 — GGUF arch `gemma4`, `block_count = 48`, no `ssm.*`
  keys. Loader at `-lv 4` prints `n_layer = 48`, `is_swa_any = 1`,
  `n_swa = 1024`, and allocates only
  `llama_kv_cache_iswa: creating non-SWA KV cache, size = 40960 cells` plus
  `creating SWA KV cache, size = 1536 cells`. No recurrent state cache.

The pure-attention model leaks at least as much as the hybrid (39/54 vs 34/54 in
the unpaired runs), which is why I do not think this is the known
hybrid/recurrent prompt-cache issue.

## Exact command

```
llama-server \
  --host 127.0.0.1 --port 8081 \
  -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -c 327680 -np 8 \
  -ctk q8_0 -ctv q8_0 \
  -fa on -ub 512 -b 2048 \
  -lm none --no-kv-unified --cont-batching \
  --slot-save-path ./slots
```

`--slot-save-path` is only needed for the `action=erase` part of the test. The
leak reproduces without it.

## Steps to reproduce

Take N documents of increasing length, each containing exactly one unique
identifier — I used 6 synthetic documents of 1,021 / 2,045 / 4,094 / 8,190 /
16,382 / 32,766 tokens, each containing one UUID that appears in no other
document. For each document in turn, send 9 `POST /completion` requests with
`"id_slot": 0` and `"cache_prompt": true`, asking a question about that
document's own content. Prompt layout is document first, question last.

Then check every response for identifiers that occur in a *different* document.

Minimal script (stdlib + `httpx`):

```python
import httpx, json, sys

BASE = "http://127.0.0.1:8081"
SYSTEM = ("You answer questions strictly from the DOCUMENT given in the user "
          "message. If the answer does not appear in the document, reply with "
          "exactly: NOT IN DOCUMENT")

c = httpx.Client(base_url=BASE, timeout=600)

def render(user):
    r = c.post("/apply-template", json={
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False},
    })
    r.raise_for_status()
    return r.json()["prompt"]

def ask(prompt, slot, seed):
    r = c.post("/completion", json={
        "prompt": prompt, "n_predict": 512, "temperature": 0.3,
        "seed": seed, "cache_prompt": True, "id_slot": slot, "stream": False,
    })
    r.raise_for_status()
    d = r.json()
    return d["content"], d["timings"].get("cache_n"), d.get("id_slot")

# docs: list of (name, text, unique_id, question) ordered shortest first
docs = json.load(open(sys.argv[1]))

for i, d in enumerate(docs):
    prompt = render(f"DOCUMENT:\n{d['text']}\n\nQUESTION: {d['question']}")
    for trial in range(3):
        for slot, arm in ((0, "shared"), (1 + i, "fresh")):
            out, cache_n, got = ask(prompt, slot, trial)
            foreign = [o["id"] for o in docs
                       if o["id"] != d["id"] and o["id"] not in d["text"]
                       and o["id"] in out]
            flag = "LEAK" if foreign else "    "
            print(f"{flag} {arm:6s} {d['name']:14s} t{trial} "
                  f"slot={got} cache_n={cache_n} foreign={foreign}")
```

The corpus I used is attached (6 files, ~360 KB total). It is procedurally
generated, low-entropy, repetitive prose. I could not reproduce the effect on
higher-entropy synthetic documents — see "What did not reproduce" below — so
using this corpus, or something like it, matters.

## Expected

Each response should reference only the document it was sent. A slot's previous
occupant should be invisible.

## Actual

From the fourth document onward, most responses on the shared slot enumerate
content belonging to the first and second documents. The same prompt sent to a
fresh slot in the same process, at the same moment, does not.

Document 4 (8,190 tokens), same prompt bytes, same reported `cache_n = 7798`:

```
slot 0 (has previously held documents 1-3):
  The provided text does not contain a custody note for the "Hurnshawfield
  Bureau." It contains custody notes for the "Prylfennwick Trust,"
  "Orstlornholm Trust," "Quinfennsted Trust," and "Selkdaleridge Bureau."
  Therefore, the requested key is not present in the excerpt.

slot 4 (has only ever held document 4):
  The archive key issued to the Hurnshawfield Bureau
```

"Prylfennwick" and "Orstlornholm" occur only in document 1. "Quinfennsted" and
"Selkdaleridge" occur only in document 2. Neither string is in document 4 and
neither is in the question.

Document 2 (2,045 tokens), asked for an identifier that is in no document:

```
slot 0: 1251d802-86aa-4e75-96be-aefc175c1e8e   <- document 1's UUID
slot 2: 0f3aac07-d1fe-460c-907f-53ddb57cc797   <- its own document's UUID
```

The earliest occurrence is always the second document — i.e. two documents on
one slot is enough. Rate by position in the sequence, three independent runs
agreeing:

| document | leaked / 9 (hybrid) | leaked / 9 (pure attention) |
|---|---|---|
| 1st | 0 | 0 |
| 2nd | 3 | 3 |
| 3rd | 6 | 9 |
| 4th | 8 | 9 |
| 5th | 8 | 9 |
| 6th | 9 | 9 |

## Two observations that may narrow it down

**1. It is not the prompt cache.** The first request against each new document
reports `timings.cache_n == 0` — nothing reused, whole prompt prefilled — and
2 of 6 such requests on the hybrid model, and 4 of 6 on the pure-attention
model, still returned earlier documents' content.

**2. It is not what `action=erase` clears.** Issuing
`POST /slots/0?action=erase` before each document returns 200 with a truthful
`n_erased` (1231, 2221, 4271, 8364, 16562 — matching the resident prompt
lengths), and the leak rate is unchanged: 33/54 with erase, 34/54 without.

Together these suggest per-slot state that is neither counted by `cache_n` nor
released by `erase`.

## Control

The control is in the same process rather than a separate run, so that weights,
sampler, prompt bytes and timing are all held fixed: every prompt is issued
twice, once to the accumulating slot 0 and once to a slot that has only ever
held that document. 24/54 vs 0/54 (p = 4.4e-9). A fresh server process per
document is likewise clean.

## What did not reproduce

I could not reproduce this with a smaller, cleaner test: two documents of 522 to
8,551 tokens built from randomly generated filler with one freshly minted UUID
each, across `cache_prompt` true and false, same slot and different slot, A→B→A
ordering, changed `seed`, `n_keep: 0`, and `save`/`restore` — 110 trials, zero
leaks. The effect needed the long, repetitive, low-entropy corpus attached, and
it shows up much more readily under a question that asks the model to enumerate
what the document contains than under a question with a single expected answer.

I have not looked at the source, so I have no hypothesis about which code path
is responsible.

## Minor: `action=erase` reports 501 for a misleading reason

Without `--slot-save-path`, `POST /slots/{id}?action=erase` returns
`501 This server does not support slots action. Start it with --slot-save-path`.
`erase` neither reads nor writes a file, so gating it on the save path is
surprising, and the message sends you looking for a build without the feature. I
originally concluded the endpoint was unimplemented in this build because of it.
Happy to open that as a separate issue if it is worth one.
