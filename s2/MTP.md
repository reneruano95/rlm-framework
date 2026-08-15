# S5 root swap → Qwen3.8-27B, and §7 #4 MTP measured at 2.11×

**Date:** 2026-08-15 · **Scripts:** `s2/gguf_compare.py`, `s2/mtp_bench.py`
**Raw:** `s2/results/mtp_*_t0.0.jsonl` · **Server:** vulkan build, port 8080,
`-c 32768 -np 1 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 -lm none
--no-context-shift`

§9 S5 names this swap in advance — *"swap the root to a new model (target:
Qwen3.8-27B … this slice may activate early)"* — so this is a pre-registered
slice activated early, and it is done the way S5 requires: **by editing
`config.yaml` only**.

## 1. Day-one checklist (§9 S5, all nine items)

Items 1–5 and 8–9 were answered from GGUF metadata alone, with no model load:

| # | item | answer |
|---|---|---|
| 1 | dense or MoE? | dense (no `expert_count`) |
| 2 | attention layout | `arch = qwen35`, `ssm.inner_size 6144 / state_size 128 / conv_kernel 4` — **identical to the incumbent**, so R7's "different architecture" hazard did not materialise |
| 3 | MTP head? | **present** — `nextn_predict_layers = 1`, `block_count` 65 vs 64 |
| 4 | chat template | **DIFFERS**, 8,057 → 8,952 chars |
| 5 | tokenizer identity | **IDENTICAL** — `gpt2`/`qwen35` pre, bos 248044, eos 248046, vocab and merges hashes equal |
| 6 | licence | same family terms as the incumbent |
| 7 | GGUF quality | **released 2026-08-14 — one day old** |
| 8 | mmproj present but unused? | yes, present on disk, deliberately not loaded |
| 9 | RLM-post-trained? | no |

Two of these carry consequences worth stating separately.

**Item 5 is the quiet win.** An identical tokenizer means the measured 3.7727
chars/token calibration, C2's chunk geometry, `max_subcalls`, the window/stride
arithmetic and every token count in §7 and §8 **carry over unchanged**. A
different tokenizer would have re-priced all of them. Only `padding_token_id`
moves (248055 → 248044) and it is unused here.

**Item 7 is a live risk, not a formality.** §9 S5's own checklist says community
quants stabilise over ~2 weeks; this one is a day old. The benchmark freeze at
the end of S2 must not be taken against a quant that may be silently re-uploaded,
so the file sha256 is recorded in `config.yaml` and here:

```
e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520  Qwen3.8-27B-Q4_K_M.gguf
5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0  Qwen3.6-27B-Q4_K_M.gguf  (incumbent)
```

Q4_K_M is **15.66 GB, the same size as the incumbent**, so dual residency and
§4's headroom arithmetic are unchanged.

Item 4 has one consequence in code: the rendered root head moves, so the root
prefix hash changes. R3's drift detector (v0.3.7) re-pins it on the first call —
which is exactly the case it was built for.

## 2. §7 #4 — MTP on the root

This was **unmeasurable before the swap**: the incumbent's Q4_K_M carried no MTP
head (only its Q6_K/Q8_0 MTP variant did, which the root does not run). The build
supports it — `--spec-type` accepts a comma-separated list including
`draft-mtp` — and the root is already at `--parallel 1`, MTP's requirement.

Median decode over 8 runs (4 root-shaped prompts × 2 reps), temperature 0,
`n_predict 256`, `cache_prompt: false`:

| arm | flags | median tok/s | vs base |
|---|---|---:|---:|
| base | none | 12.87 | 1.00× |
| **mtp2** | `--spec-type draft-mtp --spec-draft-n-max 2` | **27.17** | **2.11×** |
| mtp6 | same, `--spec-draft-n-max 6 --spec-draft-p-min 0.75` | 26.33 | 2.05× |
| mtp6-ngram | `--spec-type draft-mtp,ngram-mod` + ngram params | 18.04 | 1.40× |

**§7 #4's target is ≥1.4×. `draft-mtp` at n-max 2 clears it by 1.5×**, on the
serial, decode-bound part of every episode.

Two things the arms isolate:

* **Draft length is not the lever here.** n-max 6 (2.05×) and n-max 2 (2.11×) are
  within noise of each other, so §7 #4's pre-registered bound of ≤2 costs
  nothing and is kept.
* **`ngram-mod` COSTS about a third of the gain on this workload** (1.40× against
  2.11×), and it is the ngram stage rather than the longer draft that does it —
  the `draft-mtp`-only arm at the same n-max 6 keeps the full speedup. The
  combined arm also showed a systematic rep0 ≫ rep1 pattern on every prompt
  (24.1→11.0, 32.4→13.0, 32.0→17.2, 18.9→13.9) that neither other arm shows.

  **This is not a general verdict on ngram.** These prompts generate *fresh* code
  from short instructions with `cache_prompt: false`, so the ngram matcher has
  almost nothing to match against and its lookups are pure overhead. ngram
  speculation pays when output re-emits text from a long shared context —
  iterative edits, rewriting, agentic loops. **The RLM root's multi-turn path is
  exactly that shape and is NOT tested here**, so ngram deserves a separate
  measurement against a multi-turn root transcript before being dismissed.

### The losslessness question, and why it cannot be answered this way

The obvious check — does speculation change the answer? — was attempted by
comparing bytes at temperature 0 against the unspeculated arm. **It does not
work on this stack, and the control says so: the baseline does not reproduce
ITSELF, 0/4 prompts.** §8 already anticipated this ("continuous batching means a
fixed seed does not guarantee bitwise reproducibility — seeds pin sampling
identity, not numerics"), and production root sampling is 0.7/0.8 anyway, so
byte-identity was never available as a gate.

An early draft of this file labelled both MTP arms "LOSSY" on that comparison.
That was wrong and is withdrawn: 0/8 against a baseline that is itself 0/4 is a
measurement of ambient nondeterminism, not of speculation. The observed
divergence is also benign in kind — base and mtp2 share a 545-character prefix
and then choose two equivalent ways to write the same sort.

**So MTP ships for speed with its quality claim explicitly unverified.** R4's
actual requirement is *unchanged benchmark success*, which is an S4 measurement
and cannot be taken before the benchmark exists. That debt is recorded in
`config.yaml` beside the flag rather than left implicit.

## 3. What the swap invalidates

Stated rather than discovered later:

* **S1's gate result does not transfer.** Its 3/3 credited root-as-programmer on
  Qwen3.6-27B. Re-running S1's fixtures on the new root is owed before that
  claim is repeated.
* **The root prompt A/B (root.v3) was decided against the incumbent.** §9 S1
  closed that A/B at two variants and it must not be reopened casually — but the
  pin was made for a different model. The honest position: root.v3 is now
  *inherited*, not *validated*, and the freeze at the end of S2 should say so.
* **§7 #3c's root-turn `cache_n` monitor** and S0 item 5(b) — the owed
  confirming pass on port 8080 — must be taken on the new root, not the old one.
* Nothing about the **leaf** changes. Every S2 leaf measurement stands.

## 4. Config, and one gap it closed

`servers.root.mtp` existed and was validated (`mtp=true` requires
`parallel == 1`) but **was never emitted into the launch line** — `launch_argv`
does not reference it, so setting it would have been a silent no-op. Rather than
teach `launch_argv` to invent flags (its docstring forbids exactly that: *"a flag
invented in code is a flag `config_snapshot` cannot record"*), the flags live in
`extra_flags`, which is snapshotted verbatim, and a new cross-field validator
makes the declaration and the flags unable to disagree in either direction.
Four tests cover it.
