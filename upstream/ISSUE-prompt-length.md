# HIP: a value more than a few hundred tokens back cannot be repeated; Vulkan on the same GPU and the same commit is clean

## Summary

On the HIP/ROCm backend, a 12-hex-digit marker planted near the top of a prompt
and asked for at the bottom **cannot be repeated back** once it is more than a
few hundred tokens away. The same build's **Vulkan** backend — same commit, same
GPU, same model file, same flags, byte-identical prompts — repeats it correctly
at every distance tested, out to 2,936 tokens.

Repeating a string you were just handed is the least a language model can do:
nothing to reason about, nothing to count, no world knowledge. A wrong answer
here is a serving fault, not a capability limit.

`Qwen2.5-7B-Instruct-Q8_0`, `b10375-ba360efe1`, AMD Radeon 8060S (gfx1151):

| marker tokens before the question | 80 | 584 | 1,200 | 1,760 | 2,936 |
|---|---|---|---|---|---|
| **HIP / ROCm** | ok | WRONG | **HTTP 500** | WRONG | **HTTP 500** |
| **Vulkan** | ok | ok | ok | ok | ok |

Both backends pass the 80-token positive control, so the model and the flags are
not the problem. HIP fails every longer cell, twice with a server error.

**This is not model-specific.** Three families, two attention architectures,
three parameter scales, on the same box:

| model | attention | HIP / ROCm | Vulkan |
|---|---|---|---|
| `Qwen3.6-35B-A3B` Q4_K_M | hybrid, 10 of 40 layers global | **0/144** past 1,025 tokens | **240/240** across 600–1,500 |
| `Qwen2.5-7B-Instruct` Q8_0 | uniform global | **0/25**, plus 23 stream failures | **48/48** |
| `Meta-Llama-3.1-8B-Instruct` Q8_0 | uniform global | **0/48, every one degenerate** | **48/48** |

`Meta-Llama-3.1-8B-Instruct-Q8_0` at ~1,200 tokens on HIP answers a counting
question with `"The end of the end of the end of the end of the of the end of…"`
and answers coherently on Vulkan at every length.

## Reproducer

```bash
# HIP build
llama-server -m Qwen2.5-7B-Instruct-Q8_0.gguf --host 127.0.0.1 --port 8081 \
  -c 32768 -np 4 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048
python prompt_length_repro.py --base http://127.0.0.1:8081

# Vulkan build of the SAME commit, same command line, re-run
```

`prompt_length_repro.py` is standard library only, needs no corpus, and ships a
positive control at 80 tokens that must pass before anything else is read.

## What is ruled out

| candidate | test | result |
|---|---|---|
| the model | two further families, one hybrid and two uniform-attention | all three fail on HIP, all three clean on Vulkan |
| model architecture | uniform-global-attention controls, verified from GGUF metadata (no `ssm.*`, no `sliding`/`window` keys) | fail on HIP |
| KV cache quantization | `-ctk f16 -ctv f16` | identical failure |
| the flash-attention kernel | `-fa off` (paired with `f16`, since quantized V requires FA) | identical failure |
| hipBLASLt | `ROCBLAS_USE_HIPBLASLT=0` | identical failure — 0/36 |
| CPU fallback on the Vulkan side | server log | `Vulkan0 (AMD Radeon 8060S)`, 41/41 layers offloaded, KV and recurrent buffers identical to HIP's |
| a scoring artifact | the answer is an exact string the harness generated from a seed | it appears in no training corpus |
| prompt-length alone | a needle 300 tokens back in a 1,400-token prompt | **clean on HIP** — it is the DISTANCE back, not the total length |

## What it is not

Not the known gfx1151 issues. Those are performance (#13565), VMM and loading
(#15018, #19482), and crashes (ROCm #5534). This one returns HTTP 200 with a
confidently wrong answer, which is worse: nothing downstream can tell.

## Relationship to the concurrent-decode report

`ISSUE-concurrent-decode.md` reports the same backend producing the same
degenerate output — character loops, truncated identifiers, three-token stubs —
when two requests are decoded concurrently, with Vulkan clean at the same
concurrency. **These look like one defect with two dials: concurrency and
distance-back.** Both are HIP-only on this box, both survive every flag we tried,
and both produce degenerate text rather than an error.

## Environment

- llama.cpp `b10375 (ba360efe1)`. **Both backend builds report the same commit**,
  so this is not a version difference.
- Windows 11, AMD Ryzen AI MAX+ 395 "Strix Halo", Radeon 8060S iGPU (gfx1151),
  128 GB unified memory.
- HIP build: the ggml-org `win-rocm` release zip with AMD ROCm 7.14 runtime DLLs
  (hipblas / rocblas / rocsolver / hipblaslt plus gfx1151 Tensile) placed beside
  `llama-server.exe`. This is a hand-assembled stack; discussion #20856's
  known-good gfx1151 configuration is ROCm 7.2.0 built with
  `GGML_HIP_NO_VMM=ON`, and we have not tested that one.
- Vulkan build: the ggml-org `win-vulkan` release zip of the same commit.

## What we cannot say

The mechanism inside HIP. Not hipBLASLt, not KV precision, not the FA kernel,
not CPU fallback — but which kernel, and why the onset falls where it does, we
do not know. The onset is model-dependent (~584 tokens for Qwen2.5-7B, ~1,000
for the hybrid) which suggests a shape- or size-dependent kernel path rather
than a fixed window.
