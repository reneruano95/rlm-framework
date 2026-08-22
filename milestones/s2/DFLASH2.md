# DFlash2 on the root: 1.46× the incumbent MTP, measured on this box

**Date:** 2026-08-19 · **Scripts:** `milestones/s2/mtp_bench.py` (unmodified), `milestones/s2/dflash2_bench.py` · **Raw:** `milestones/s2/results/mtp_d2-*_t0.0.jsonl`, `milestones/s2/results/mtp_d2-*_t0.7.jsonl`
**Server:** PR-#27342 Vulkan build, port 8080, `-c 32768 -np 1 -ctk q8_0 -ctv q8_0
-fa on -ub 512 -b 2048 -lv 4 -lm none --no-context-shift`

**ADOPTED 2026-08-19.** The root now ships
`--spec-type draft-dflash --spec-draft-n-max 4` against the
`tools\llamacpp-vulkan-dflash2` build. §7 was written first, as the case for
*not* adopting; it is kept verbatim below because every item in it is still
true, and the ones that survive adoption are now debts this config carries
rather than reasons it was refused. §8 records what the swap changed in code.

## 1. Why this is even a question

DFlash2 shipped 2026-08-18 (Z Lab / Inco AI) with a drafter for **exactly the
model the S5 root swap landed on** — `Qwen/Qwen3.8-27B`. That is a coincidence
worth naming: DFlash2's other day-one drafter is for Meta Muse Glimmer-30B, and
DFlash **1** never had a Qwen3.8 drafter at all. Had the root still been
Qwen3.6-27B this would have been a DFlash-1 question with a different answer.

DFlash2 is a block-diffusion drafter: it predicts a whole block in **one**
forward pass and keeps top-k candidates at every position, then a 2.0M-parameter
selector traces one path through them. That matters here specifically, because
§7 #4's measured MTP profile is the signature of the cost DFlash2 removes —
MTP drafts autoregressively, so its n-max 6 arm (26.33 t/s) was a **wash**
against n-max 2 (27.17 t/s): the extra draft tokens cost as much as they bought.

## 2. What it required, and the provenance risk that creates

DFlash2 support is **not in any released llama.cpp binary**. It is
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342), opened 2026-08-18,
**still open**, one commit (`5ecbe1a`), `mergeable_state: unstable`. The DFlash2
GGUF carries 4 KV keys (`dflash.conv_kernel_size`, `conv_group_size`,
`selector_rank`, `selector_top_k`) and 7 tensors (`blk.N.attn_conv_base/proj`,
`blk.N.ffn_conv_base/proj`, `selector_predecessor/successor/hidden`) that only
that PR knows, so **b10375 and b10488 cannot load the checkpoint at all** — this
is not a "works but slower" situation, there is no fallback path.

So every number below comes from a **source build of an unmerged PR**, which is a
genuine departure from this project's pinned-release discipline and was the
single biggest argument against adopting it (§7 #3); it is now a standing debt:

| | pinned production build | this measurement build |
|---|---|---|
| version | `10375 (ba360efe1)` | `0.1.2-dev (build 1, commit 5ecbe1a)` |
| compiler | Clang 20.1.8 | MSVC 19.51.36252.0 |
| provenance | official `bXXXXX` release zip | `D:\AI\src\llama.cpp-dflash2`, built here |

**The build is controlled for.** A new `base` and a new `mtp2` arm were
re-measured on the PR build rather than reusing the 2026-08-15 numbers, because a
build-to-build delta would otherwise be scored as a DFlash2 effect. It is not one:

| arm | pinned b10375 (2026-08-15) | PR build (2026-08-19) | delta |
|---|---:|---:|---:|
| no speculation | 12.87 t/s | 12.84 t/s | −0.2% |
| `draft-mtp` n-max 2 | 27.17 t/s | 26.89 / 27.03 t/s | −0.5% |

Two independent MTP replicates (n=8 and n=12) land within 0.5% of the pinned
build. The measurement rig reproduces the incumbent, so the DFlash2 delta is a
DFlash2 delta.

Backend note: the PR touches **no backend file** — no CUDA, Vulkan, HIP or CPU
kernel — and builds its graph from stock ggml ops (`get_rows`, `mul_mat`, `top_k`,
`tanh`, `pad`, `cast`, views/reshapes). Upstream CI for this PR is green on both
`gpu-vulkan-*` and `gpu-rocm`. On this box the loaded graph reports
`graph splits = 2`, i.e. no wholesale CPU fallback.

## 3. Result, temperature 0 (the §7 #4 metric)

4 root-shaped prompts × reps, median decode t/s. `-r` arms are independent
re-measurements, not re-analyses of the same rows.

| arm | n | decode t/s | vs no-spec | vs MTP | mean acc len |
|---|---:|---:|---:|---:|---:|
| no speculation | 8 | 12.84 | 1.00× | 0.48× | — |
| `draft-mtp` n-max 2 | 8 | 26.89 | 2.09× | 1.00× | 2.51 |
| `draft-mtp` n-max 2 (`-r`) | 12 | 27.03 | 2.11× | 1.01× | 2.52 |
| `draft-dflash` n-max 2 | 8 | 26.80 | 2.09× | 1.00× | 2.52 |
| `draft-dflash` n-max 3 | 12 | 31.98 | 2.49× | 1.19× | 3.07 |
| **`draft-dflash` n-max 4** | 8 | **35.56** | **2.77×** | **1.32×** | — |
| **`draft-dflash` n-max 4 (`-r`)** | 12 | **34.35** | **2.68×** | **1.28×** | 3.47 |
| `draft-dflash` n-max 5 | 12 | 34.75 | 2.71× | 1.29× | 3.83 |
| `draft-dflash` n-max 6 | 12 | 32.02 | 2.49× | 1.19× | 4.24 |
| `draft-dflash` n-max 7 | 8 | 24.40 | 1.90× | 0.91× | 4.41 |

## 4. Result, production sampling (temp 0.7 / top_p 0.8)

This is `scaffold.sampling.root`, i.e. what the root actually runs. §7 #4 was
only ever measured at temperature 0, so this table is new information about the
shipped configuration and not merely a robustness check.

| arm | n | decode t/s | vs no-spec | vs MTP | mean acc len |
|---|---:|---:|---:|---:|---:|
| no speculation | 12 | 12.85 | 1.00× | 0.49× | — |
| `draft-mtp` n-max 2 | 12 | 26.18 | 2.04× | 1.00× | 2.46 |
| **`draft-dflash` n-max 4** | 12 | **38.34** | **2.98×** | **1.46×** | 3.61 |
| `draft-dflash` n-max 5 | 12 | 33.60 | 2.62× | 1.28× | 3.77 |

**At the sampling the root actually uses, DFlash2 at n-max 4 is 1.46× the
incumbent and 2.98× no speculation.** The gain is larger at 0.7 than at 0.0
(1.46× vs 1.28–1.32×), which is the opposite of the usual expectation that
speculation degrades as sampling gets less greedy.

The `prefill` column is deliberately omitted: these prompts are 64–81 tokens, so
prefill t/s here measures fixed per-request overhead, not prefill throughput. It
was flat (18.8–20.2 t/s) across every arm, so nothing was traded for the decode
gain — but this rig cannot make a claim about long-prompt prefill either way.

## 5. Why the optimum is 4, not the recommended 7

The model card and the PR both suggest `--spec-draft-n-max 7` (the maximum:
`n_draft_max = block_size − 1`, and `dflash.block_size = 8`). On this box 7 is
the **worst** DFlash2 setting and loses to MTP. Per-position acceptance from the
server's own `-lv 4` log explains it — the code-shaped prompts, one arm each:

```
n-max 4   (0.983, 0.864, 0.797, 0.678)
n-max 7   (0.930, 0.860, 0.791, 0.698, 0.605, 0.558, 0.488)
```

Acceptance at position 7 is still ~0.5, so the draft is not collapsing. The cost
is that both the drafter batch and the target verification batch are `n-max + 1`:

| | work per step | mean accepted | accepted per unit work |
|---|---:|---:|---:|
| n-max 4 | 5 tokens | 3.47 | **0.69** |
| n-max 7 | 8 tokens | 4.41 | 0.55 |

On a bandwidth-bound APU the marginal verified token is not free, so the setting
that maximises **acceptance length** is not the setting that maximises
**throughput**. The plateau is broad (n=4 and n=5 within 1.2% of each other at
temp 0) with a cliff at 7, so n-max 4 is a peak with margin on both sides, not a
knife-edge.

The prose prompt is where n-max 7 dies — its per-position profile decays to
`(0.648, 0.407, 0.185, 0.083, 0.019, 0.009, 0.000)`, i.e. positions 5–7 are pure
waste. Root turns are code-shaped, which is the regime DFlash2 wins in
(per-prompt at n-max 4, temp 0: plan 33.56, loop 37.98, reduce 40.21, prose 20.00).

## 6. This contradicts the published Strix Halo prior

Two prior Strix Halo reports have DFlash **v1** on Vulkan **losing** to MTP —
Qwen3.6-27B at 12.05–16.07 t/s against MTP's 16.73–20.85, and Gemma4-31B where
DFlash was *slower than no speculation at all* (9.73 vs 10.30) at 10–14%
acceptance. The stated mechanism was that the MTP head streams with the target's
weights for free while a separate ~1 GB drafter pays a per-cycle tax.

That mechanism is real and is still being paid here — it is why DFlash2 at
n-max 2 exactly ties MTP (26.80 vs 26.89) rather than beating it. What changed is
that DFlash2 can spend a **larger** block for one drafter pass, so it buys 3.47
accepted tokens where MTP's autoregressive drafting could only afford 2.51. The
prior was measured on a drafter that could not do that. **It should not be
generalised from DFlash 1 to DFlash 2 on this hardware.**

## 7. What is NOT established, and what adoption would need

1. **Losslessness is unverified, exactly as it is for MTP.** The byte-identity
   control is uninformative on this stack: the no-speculation arm does not
   reproduce **itself** at temperature 0 — 0/4 prompts self-identical, and no arm
   measured here exceeded 1/4 — which `milestones/s2/mtp_bench.py` already documents. So the
   near-zero cross-arm identity is not evidence of loss. R4's real gate is
   unchanged benchmark success, and that is an S4 measurement, not this one.
2. **The S4 gate passed against `draft-mtp`.** Changing the root's speculative
   decoding changes `config_snapshot`, so the S4 numbers would no longer describe
   the shipped root. Adoption is a re-run, not an edit.
3. **The build is an unmerged PR.** Adopting it would replace a pinned, hashed
   release binary with a locally-built branch that can be force-pushed or
   abandoned. The PR also carries an unfixed vision/M-RoPE defect (irrelevant
   here — the root is text-only and the mmproj is deliberately not loaded) and a
   reported **`-np > 1` collapse** (also irrelevant here — the root is `-np 1` —
   but it means the leaf could never use this).
4. **Memory cost, measured on this box:** drafter weights 1,079.61 MiB + draft KV
   50.00 MiB (SWA-capped at 2,560 cells × 5 layers) + draft compute 517.92 MiB,
   and the target's own compute buffer grows 164.28 → 416.75 MiB because it now
   verifies batches. Call it **~1.9 GiB** against §4's dual-residency headroom of
   19.5% free on the 64 GiB carve — which it takes to roughly 16.5%, still above
   the stated 15% floor but with the margin materially thinned. MTP costs zero
   here, since its head is already inside the root quant.

**Recommendation as written on 2026-08-19:** keep `draft-mtp` pinned; re-open
when PR #27342 merges and ships in a `bXXXXX` release. **Overruled the same day**
— the swap was made deliberately, with (1)–(4) understood and accepted. They are
now standing debts, not settled questions:

- (1) and (2) mean **S4 is owed a re-run.** The gate passed against `draft-mtp`;
  its numbers describe a root this config no longer launches.
- (3) means the root's `build_info` is a branch commit, not a release tag. If
  `z-lab:dflash2` is force-pushed or the PR is abandoned, the revert is two
  lines: `backend_dir` back to `tools\llamacpp-vulkan`, flags back to
  `--spec-type draft-mtp --spec-draft-n-max 2`, `mtp: true` / `dflash: false`.
- (4) means the memory floor is the thing to re-check first if the leaf is ever
  re-sized; ~16.5% free against a stated 15% floor.

## 8. What adoption changed outside `config.yaml`

Two code changes were required, and the second was a live defect the swap
exposed rather than a preference.

**`rlm/config.py` — the declaration guard would have gone vacuous.**
`servers.root.mtp` is a declaration that a cross-field validator keeps in sync
with `extra_flags`, so `config_snapshot` can never describe a run that did not
happen. That check only knew the token `draft-mtp`, so `mtp: false` plus DFlash
flags **satisfies it** while reintroducing exactly the silent lie it exists to
prevent. There is now a `dflash` declaration guarded identically, the
single-slot rule covers both (DFlash2 is measured upstream to collapse at
`-np > 1`), and — since DFlash needs a second file on disk, unlike MTP whose
head is inside the target quant — a missing or stale `-md` path is refused at
config load rather than minutes into a launch.

**`rlm/cli.py` — the §4 handshake silently rebound to the drafter.**
`parse_launch_log` recovers the KV cache types D27 proved `/props` cannot report.
A speculative launch builds **two** contexts, so the log carries two
`llama_kv_cache:` lines and two `flash_attn =` lines — target first, drafter
second — and the parse loop took the **last** of each. Attaching the drafter
therefore moved §4's assertion off the target and onto the draft cache, which
defaults to f16, and `rlm validate` reported the root's cache types as
`f16/f16`: a false negative on a correctly-launched server. Caught by running
the real handshake against the real log, not by the unit tests, which had no
two-context sample. First occurrence now wins and the drafter is recorded under
`draft_*` instead of shadowing the target:

```
target : type_k q8_0  type_v q8_0  kv 1088.00 MiB  32768 cells  16 layers
draft  : type_k f16   type_v f16   kv   50.00 MiB   2560 cells   5 layers
```

Verified end to end by launching the root through `serverproc.launch_argv` from
the shipped config: health OK, `n_max=4`, `draft_n=139 / accepted=91` on a live
completion, D27 handshake passes.

## 9. Reproducing

```
git -C D:\AI\src\llama.cpp-dflash2 log --oneline -1     # 5ecbe1a support DFlash2
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Drafter sha/size: 1,143,006,752 bytes, `dflash` arch, `block_size 8`,
`selector_top_k 16`, `target_layers [6, 20, 34, 48, 62]`, selector codebooks over
248,320 vocab — the same tokenizer as the root, which is why it drops in at all.

Server line = the pinned root line plus:

```
-md D:\AI\models\z-lab\Qwen3.8-27B-DFlash2-GGUF\Qwen3.8-27B-DFlash2-Q4_K_M.gguf
--spec-type draft-dflash --spec-draft-n-max 4
```

DFlash2 needs no extra flag beyond that: `common_speculative_impl_draft_dflash`
auto-detects it from the checkpoint's `dflash.selector_top_k` key.
