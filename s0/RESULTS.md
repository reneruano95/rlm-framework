# S0 Kill Gate — On-Box Results

**Date:** 2026-08-12/13 (single overnight session)
**Spec:** rlm-runtime-spec-v0.2.2 §9 S0
**Box:** GMKtec NucBox EVO-X2 — Ryzen AI MAX+ 395 (16C/32T), 128 GB LPDDR5X-8533, Radeon 8060S (gfx1151), Windows 11 Pro 26200, Adrenalin 32.0.31035.1003 (2026-07-23), VGM 64 GB dedicated / 64 GB system, pagefile fixed 64 GB.
**llama.cpp:** b10375 (ba360efe1), ggml-org Windows release zips; leaf = win-rocm-7.14 + AMD ROCm 7.14.0 wheel DLL graft (hipblas/rocblas/rocsolver/hipblaslt + gfx1151 Tensile/kpack); root = win-vulkan (AMD proprietary driver).
**Batch flags (pinned, stated per spec):** `-ub 512 -b 2048` on every measurement below. Load mode `-lm none` (the `--no-mmap` successor). KV `q8_0` K+V, `-fa on` everywhere.
**Env:** leaf runs with `ROCBLAS_USE_HIPBLASLT=1`.
**Raw records:** `s0/results/raw.jsonl` (every run, timestamped). Fixture: `s0/fixtures/fix32k.txt` = exactly 32,768 leaf tokens (deterministic seed 1).

## Verdict

**GATE PASSED.** Leaf 32K-prompt prefill, median of 3 cold runs, single slot, ROCm: **949.9 t/s** vs the pre-registered exact threshold of 500 t/s (margin: 1.90×). Backend split confirmed: leaf=ROCm, root=Vulkan.

## Item 1 — Leaf prefill @ 32K, single slot, per backend (median of 3 cold runs)

| Backend | Run 1 | Run 2 | Run 3 | **Median** |
|---|---|---|---|---|
| ROCm/HIP (chosen) | 949.87 | 953.79 | 943.87 | **949.87 t/s** |
| Vulkan (AMD prop.) | 800.35 | 836.22 | 839.74 | **836.22 t/s** |

All runs verified cold (`cache_n=0`, full `prompt_n=32776`). ROCm +13.6% over Vulkan at d=32K — same direction and nearly the same size as the Linux prior (+13%). Both backends individually clear the 500 t/s gate. On-box beats the Linux community prior (807 ROCm) by +18%.

## Item 2 — Concurrency scaling (leaf ROCm, §4 topology `-np 8 -c 327680 --no-kv-unified`)

| k | Aggregate t/s | Wall (s) | Per-req min/max t/s |
|---|---|---|---|
| 1 | 943.4 | 34.7 | 945 / 945 |
| 2 | 913.9 | 71.7 | 462 / 852 |
| 4 | 884.9 | 148.2 | 431 / 835 |
| 8 | 946.7 | 277.0 | 422 / 939 |

**Finding (new, not in any community dataset): aggregate prefill does NOT scale with slots — one 32K stream already saturates the iGPU (~950 t/s is the machine's total prefill budget).** Concurrency time-slices it; per-request rates spread 422–939 t/s within a wave. Consequence for §7 #1: parallel dispatch buys scheduling/latency-hiding and decode-prefill overlap via continuous batching, not throughput multiplication. The S2 fan-out gate target ("≥80% of S0 8-slot scaling") resolves to: an 8-chunk wave must complete in ≤ (8×32K)/(0.8×947) s ≈ 346 s.

## Item 3 — Decode, both servers (fresh / at depth / contended)

| Measurement | t/s | Spec prior |
|---|---|---|
| Leaf decode fresh (single stream) | 55.08 | 50–60 ✓ |
| Leaf decode @ 32K depth | 46.25 | 44–49 ✓ |
| Root decode fresh, uncontended (leaf resident, idle) | 12.35 | ~12 ✓ |
| Root decode @ 28K depth, uncontended | 12.00 | 9–12 ✓ (high end) |
| **Root decode fresh, CONTENDED (verified 8 leaf slots prefilling)** | **8.52 / 9.96** (2 runs) | unmeasured anywhere |
| Root prefill @ 28K (Vulkan) | 208.0 | — |

**Contention finding: worst-case root decode retains 69–81% of uncontended speed during full 8-slot leaf prefill.** Concurrent operation is viable; phase alternation is an optimization, not a requirement.

## Item 4 — Leaf KV cost per slot (§4 topology)

GPU dedicated-memory delta at launch: 24.86 GB total; minus 22.13 GB weights → **2.73 GB for 8×40960 slots of q8_0 KV + recurrent state + compute buffers ≈ 0.34 GB/slot**. Matches the pre-registered ≈0.45 GB/slot budget envelope; the 1.5 GB/slot misconfiguration signal is absent. f16 silent-fallback ruled out by arithmetic (would be ≈2× KV). Dual-resident total (item 3 config): leaf 27.3 GB + root 16.7 GB = **43.96 GB in the 64 GB carve** (20 GB headroom).

## Item 5 — `cache_n` sanity on the hybrid architecture (R8 probe)

**(a) Leaf warm-slot re-query (identical 32K prompt, slot 0):** run 2 reused **cache_n = 32,764 of 32,768** tokens (4 reprocessed; 77 ms vs 37,365 ms cold — 485×). Token-weighted reuse per §6 formula `cache_n/(cache_n+prompt_n)` = **0.99988**.

**(b) Root 3-turn conversation (checkpoint-extend pattern):** turn 2 `cache_n` 2,115 vs expected ≥2,116; turn 3 `cache_n` 4,219 vs expected ≥4,220 — the 1-token gap is the standard final-token re-eval. **Per-turn reuse ≈ 100%; §7 #3c monitor armed.**

R8 status: on build b10375, both the identical-re-query and checkpoint-extend cases work on this hybrid (Gated DeltaNet) architecture. The R8 risk row stays (upstream regressions have recurred), but S0 measures it as currently healthy on both servers.

## Item 6 — 10-minute sustained 8-slot prefill soak (R9)

3 full waves in the 600 s window: 979.3 → 984.5 → 1013.4 t/s (**drift +3.49%, i.e. slightly faster — warm-up, no thermal degradation**). Package temp (ACPI TZ01, 15 s cadence, 55 samples): 55.1 °C start → **stable ~96 °C plateau** (max 96.1). The box sits at its thermal ceiling without losing throughput. R9 consequence: temps must keep being logged per episode; blocked S4 scheduling stays justified.

## Item 7 — B1 shakedown (leaf `--parallel 1 -c 262144`)

256K-window KV + buffers residency: **1.52 GB** (vs ~2.8 GB expected — hybrid arch even cheaper than the corrected math; classic all-layer KV would be ~10×).
200K-token prefill: **320.1 t/s** (second sample under the decode run: 316.7) — ≈10.4 min per full-window pass.
Decode @ 200K depth: **18.24 t/s**.
**Thesis-relevant consequence (to be quality-tested in S4):** the same 200K tokens processed as shallow 32K chunks at ~947 t/s aggregate take ~3.5 min of leaf prefill vs ~10.4 min for one deep pass — attention depth-decay is the RLM arm's structural wall-clock advantage on this hardware (~3×), before any quality comparison.

## Item 8 — Power sampling (Windows collector validated)

Windows exposes AMD RAPL via the `Energy Meter` counter set. Calibration under load: `Δ(rapl_package0_pkg\Energy)/Δt` vs `\Power` counter → ratio 278,040 ≈ 1/3.6e-9, i.e. **Energy is in picowatt-hours, Power in mW; cross-validated at 117.7 W vs 117.6 W (0.1%)**. Package draw under full dual-server load: **~117.6 W**. Collector: 1 Hz `Get-Counter '\Energy Meter(rapl_package0_pkg)\Energy'`; `energy_j = ΔpWh × 3.6e-9`.
Poller overhead A/B (3×32K prefill each, same B1 server, same fixture): poller OFF median **926.68** t/s vs poller ON (verified sampling, 56 rows) median **931.89** t/s → delta +0.56%, inside run-to-run noise. **energy_j ENABLED.** Mean package power during 32K prefill: 117.2 W. Note for the record: the first poller-on attempt was discarded — its sampler process died silently at launch (no CSV), so that arm never actually sampled; the rerun verified row growth before and after the runs.

## Addendum — aggregate decode & root backend A/B (measured same night)

**Aggregate leaf decode, 8 concurrent short-prompt streams (§4 topology):** **143.0 t/s** total (8 × 20.3 t/s, perfectly uniform) vs 55.1 single-stream — 2.6× continuous-batching gain, matching the community np=8 figure (~146). Single-stream k=1 reference on the same server: 47.9 t/s wall-inclusive (55.05 server-timed).

**Root backend A/B (Q4_K_M 27B, `-c 32768 -np 1`):**

| Backend | Decode fresh | Decode @28K | Prefill @28K |
|---|---|---|---|
| Vulkan (pinned) | 12.35 | **12.00** | 208.0 |
| ROCm | 12.41 | 11.05 | 290.1 |

Fresh decode ties; Vulkan holds +8.6% decode at depth, which is the root's actual operating regime; ROCm's +39% prefill does not outweigh the decode-bound role. **Pin confirmed: root=Vulkan, leaf=ROCm.**

## §7/§9 consequences (applied same day)

- §7 numbers replaced in place in ARCHITECTURE.md; backend pins recorded with the on-box A/B.
- **Benchmark task count (pre-registered §8 rule, fixed now): N = 20, margin +2.** Projection arithmetic: assume mean corpus 600K chars ≈ 150K leaf tokens; per-episode wall from S0 rates ≈ RLM 8 min / B2 5 min / B1 7.5 min / B3 7 min → mean ≈ 7 min; full S4 at N=30 = 360 episodes ≈ 42 h + escalation reserve (≤32 episodes ≈ 3.7 h) + 180 relaunches ≈ 4.5 h ≈ **50 h central — under 60 h**; but at a pessimistic mix (mean 1M chars ≈ 250K tokens) the same arithmetic gives ≈ **70 h — over budget**. The rule's "keep under 60 h" is not robustly satisfied across the plausible task-size range, so the default stands: **20 tasks**. (N=20 pessimistic ≈ 46 h — inside budget even at the worst mix.)
- `max_wall_clock` re-derivation (C5): formula fixed as ≈ 3 × (task_leaf_tokens / 947 t/s) + root-loop allowance; per-task-size-class table lands with S1 traces. The 15-min default remains valid for tasks ≤ 250K tokens.
- §7 #1 re-framed: fan-out optimizes scheduling/overlap, not aggregate throughput. S2 gate (c) target resolves to an effective 8-chunk wave rate ≥ 758 t/s (≥80% of the measured 947).
- energy_j enabled (validated Windows RAPL collector; §8 cost scorecard gets joules).
