# R13 — Slot-count memory bill, measured

**Date:** 2026-08-13 · **Spec:** ARCHITECTURE.md v0.2.6 (§4 KV/slot arithmetic, §5 C4 slot discipline + pool rotation, §10 R13)
**Box:** GMKtec NucBox EVO-X2, Ryzen AI MAX+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5X, VGM **64 GiB dedicated** carve, Windows 11 Pro 26200.
**Build:** `b10375-ba360efe1`, ROCm/HIP, `ROCBLAS_USE_HIPBLASLT=1`.
**Leaf flags (constant):** `-m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8081 -ctk q8_0 -ctv q8_0 -fa on -ub 512 -b 2048 -lm none --no-kv-unified --cont-batching`, plus `-np N -c 327680` and `-lv 4`.

**Method.** Token budget held constant at `-c 327680` for every configuration, so **slot count is the only variable** (llama-server splits `-c` across slots: per-slot ctx = 327,680 / N). Residency read the way S0 read it — delta of the Windows `\GPU Adapter Memory(*)\Dedicated Usage` counter across launch, summed over adapter instances, median of 6 samples at 0.7 s spacing before and after `/health` returns `ok`. Idle desktop baseline was **2,774,360,064 B = 2.584 GiB**, byte-identical on every run. Cross-checked against the server's own `-lv 4` allocation lines. Every server started here was stopped; the counter returned to 2.585 GiB at the end.

Every configuration launched. No launch refusals, no fallbacks, no spill to shared memory (`common_params_fit_impl` reported ≥44 GiB of device memory still free even at `-np 256`).

---

## 1. Residency vs slot count

| `-np` | per-slot ctx | measured residency (delta) | | implied marginal cost/slot vs `-np 8` | Δ vs 8-slot baseline |
|---:|---:|---:|---|---:|---:|
| 8 | 40,960 | 26,701,148,160 B | **24.867 GiB** | — (baseline) | — |
| 32 | 10,240 | 28,285,149,184 B | **26.343 GiB** | 62.98 MiB | +1.476 GiB |
| 64 | 5,120 | 31,048,683,520 B | **28.916 GiB** | 74.03 MiB ‡ | +4.049 GiB |
| 128 | 2,560 | 34,621,362,176 B | **32.244 GiB** | 62.95 MiB | +7.377 GiB |
| 256 † | 1,280 | 43,523,563,520 B | **40.534 GiB** | 64.69 MiB | +15.667 GiB |

† `-np 256` measured for bracketing only; rejected in §4 on two independent grounds.
‡ See the anomaly note below — the `-np 64` figure carries a reproducible +629 MiB that no logged buffer accounts for.

The `-np 8` number reproduces S0 exactly (S0 recorded 24.86; that figure was GiB, not GB, and S0's "leaf 27.3 GiB" was the *absolute* counter value including the desktop baseline — 29.475 GB = 27.45 GiB here. Unit labels in S0 §item 4 should be read as GiB).

## 2. Server-log cross-check (`-lv 4`)

| `-np` | model buf | KV buf (`llama_kv_cache`) | **RS buf (`llama_memory_recurrent`)** | RS/slot | compute buf |
|---:|---:|---:|---:|---:|---:|
| 8 | 20,583.34 MiB | 3,400.00 MiB | 502.50 MiB | **62.8125 MiB** | 692.00 MiB |
| 32 | 20,583.34 MiB | 3,400.00 MiB | 2,010.00 MiB | **62.8125 MiB** | 692.00 MiB |
| 64 | 20,583.34 MiB | 3,400.00 MiB | 4,020.00 MiB | **62.8125 MiB** | 692.00 MiB |
| 128 | 20,583.34 MiB | 3,400.00 MiB | 8,040.00 MiB | **62.8125 MiB** | 692.00 MiB |
| 256 | 20,583.34 MiB | 3,400.00 MiB | 16,080.00 MiB | **62.8125 MiB** | 1,124.00 MiB |

The counter delta and the log agree to within tens of MiB at `-np` 32/128/256:

- `np32 − np8` = 1,510.6 MiB measured vs 1,507.5 MiB of RS growth (Δ = 3.1 MiB).
- `np128 − np8` = 7,553.1 MiB measured vs 7,537.5 MiB of RS growth (Δ = 15.6 MiB).
- `np256 − np8` = 16,043.1 MiB measured vs 15,577.5 MiB RS + 432.0 MiB compute-buffer step = 16,009.5 MiB (Δ = 33.6 MiB).
- **`np64 − np8` = 4,146.1 MiB measured vs 3,517.5 MiB of RS growth (Δ = +628.6 MiB, unexplained).** Two independent launches returned a byte-identical `AFTER_BYTES` of 33,823,043,584, so this is deterministic, not sampling noise; no logged buffer changed. Recorded as a reproducible allocator artifact at this size. It penalises `-np 64` specifically and is another small reason not to prefer it.

**The two axes are now cleanly separated, which is what §4 needs:**

- **KV is priced by the TOKEN budget, not by slot count.** 3,400.00 MiB at every `-np`, because `-c` was held at 327,680. Measured **10,880 B = 10.625 KiB per token** at q8_0 with 10/40 KV layers.
- **Recurrent state is priced by SLOT COUNT and is context-INDEPENDENT.** Exactly 62.8125 MiB per slot at every configuration, from 40,960 tok/slot down to 1,280 tok/slot.
- The compute buffer is flat at 692 MiB through `-np 128` and steps to 1,124 MiB by `-np 256`.

## 3. Does the ~63 MB/slot estimate hold?

**Yes — it holds, and it was slightly conservative.** The spec's arithmetic (30 Gated-DeltaNet layers × `ssm.inner_size` 4096 × `ssm.state_size` 128 × 4 B = 62,914,560 B) predicts the **S** component exactly: the log reports `S (f32)` = 480.00 MiB at 8 cells = **60.000 MiB/slot = 62,914,560 B**. Dead on.

What the estimate omitted is the **conv rolling state**: `R (f32)` = 22.50 MiB at 8 cells = **2.8125 MiB/slot** (`ssm.conv_kernel` 4 × `inner_size` 4096, plus grouping). Total per-slot recurrent state is therefore:

> **62.8125 MiB/slot = 65,867,776 B ≈ 65.9 MB** — 4.7% above the ~63 MB estimate.

The spec's downstream figures survive: predicted "128 slots ≈ 8.1 GB" → measured 8,040 MiB = 8.43 GB; predicted leaf "~35 GiB" → measured 34.83 GiB absolute (32.244 GiB delta); predicted dual-residency "~52 GiB" → measured 51.53 GiB. §4's arithmetic was right to within ~1%. It can now be restated as a measurement.

Note also: the `-lv 4` log shows the same 62.813 MiB quantum reappearing as the size of a **context checkpoint** (`created context checkpoint N of 32 … size = 62.813 MiB`) — checkpoints are full copies of a slot's recurrent state. These are *not* a per-slot GPU bill: driving 32 distinct slots at `-np 128` grew dedicated memory by only **99.5 MiB total**, and the growth was already complete after 8 slots (+0.098 GiB at 8 slots vs +0.097 GiB at 32). Checkpoint storage lands in the host-side prompt cache (`cache state: … 199.334 MiB` per distinct prompt, hard limit 8,192 MiB) — host RAM, outside the 64 GiB carve. Budget ~0.1 GiB of GPU slack for it and nothing more.

## 4. Dual-residency and the largest pool that fits

Leaf delta measured here + root **16.7 GiB** (S0 item 4 delta, unchanged) + **2.584 GiB** idle desktop baseline, against the **64 GiB** dedicated carve:

| `-np` | leaf | + root | + desktop | **total resident** | **free in carve** | margin |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 24.867 | 41.567 | 44.151 | **44.15 GiB** | 19.85 GiB | 31.0% |
| 32 | 26.343 | 43.043 | 45.627 | **45.63 GiB** | 18.37 GiB | 28.7% |
| 64 | 28.916 | 45.616 | 48.200 | **48.20 GiB** | 15.80 GiB | 24.7% |
| **128** | **32.244** | 48.944 | 51.528 | **51.53 GiB** | **12.47 GiB** | **19.5%** |
| 192 (projected) | ~36.15 | 52.85 | 55.43 | ~55.4 GiB | ~8.6 GiB | 13.4% |
| 256 | 40.534 | 57.234 | 59.818 | **59.82 GiB** | 4.18 GiB | 6.5% |

**Stated safety margin: ≥15% of the carve (≥9.6 GiB) must remain free** with both servers resident. Rationale: it holds §4's reserved uses (B1's 256K single-slot profile at 1.52 GiB, an A/B candidate during S5 swaps) without a re-plan, and it absorbs the ~0.1 GiB checkpoint slack plus the ~0.6 GiB allocator artifact measured at `-np 64` in case a similar step appears at another size.

**The largest pool that fits is `-np 128`,** at 19.5% margin. It is the largest for two independent reasons, either of which is sufficient:

1. **Memory.** `-np 192` lands at ~13.4% and `-np 256` at 6.5%, both under the 15% floor.
2. **Window geometry.** At the fixed 327,680-token budget, per-slot context is 327,680/N. The v0.2.5 recommended geometry (window 1024 / stride 768) needs ≈1,024 window + 311 prefix + ~50 question + 512 `max_predict` ≈ **1,900 tokens per slot**. `-np 128` gives 2,560 (34% headroom); `-np 192` gives 1,706 and `-np 256` gives 1,280 — **both below the requirement.** A larger pool could only be bought by raising `-c`, which re-inflates the KV bill at 10.625 KiB/token.

## 5. Throughput — does a big pool cost anything?

Identical 1,033-token prompt (first 4,555 chars of `milestones/s0/fixtures/fix32k.txt`), `n_predict 128`, `cache_prompt: false`, `temperature 0`, `seed 1`, serial requests, 5 reps, server-reported `timings`.

| `-np` | prefill t/s (median) | prefill t/s (mean) | decode t/s (median) | decode t/s (mean) |
|---:|---:|---:|---:|---:|
| 8 | **1,251.80** | 1,253.38 | **54.959** | 54.982 |
| 64 | 1,248.41 | 1,200.52 | 54.847 | 54.883 |
| **128** | **1,234.65** | 1,203.24 | **55.025** | 55.051 |

**`-np 128` vs `-np 8`: prefill −1.37%, decode +0.12%.** (Means are dragged down at 64/128 by a single first-rep warm-up outlier each; medians are the honest comparison. Per-rep spread within a configuration is ≤2.5%.)

**A big pool is effectively free on throughput.** This is consistent with S0's finding that aggregate prefill is flat across slots — the slots are allocation, not parallelism. The rotation trade therefore does **not** need repricing against a throughput loss; the only cost of a big pool is the 7.38 GiB of recurrent state.

## 6. Rotation cost, measured

Full cycle timed end-to-end: `Stop-Process -Force` + `WaitForExit` → spawn → poll `/health` until `ok` → `GET /props` (with a `total_slots == -np` assertion). Same flags as production, `-lv 4` omitted.

| condition | `-np` | n | stop | spawn | health | props | **total (median)** | range |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| warm OS cache | 128 | 6 | 0.207 s | 0.005 s | 5.41 s | 0.002 s | **5.652 s** | 5.535–5.819 |
| warm OS cache | 64 | 4 | 0.197 s | 0.005 s | 5.21 s | 0.002 s | **5.454 s** | 5.403–5.680 |
| cold OS cache | 128 | 1 | 0.032 s | 0.018 s | 9.807 s | 0.004 s | **9.861 s** | — |

Cold was produced by streaming 103.9 GiB of unrelated GGUFs through the page cache (40.2 s) to evict the 20.6 GiB leaf model; no admin rights on this box, so RAMMap/standby-list purge was unavailable and this is the honest substitute. Model load dominates the difference (+4.2 s ≈ 20.6 GiB at ~5 GB/s from NVMe).

**The spec's ~6.7 s assumption is confirmed as conservative — correct it to 5.65 s warm.** Cold is 9.9 s, but cold does not apply to rotation: a rotation happens seconds after the previous generation of the same server was serving the same file, so the model is always in page cache. Cold applies only to the first launch of a session.

**Rotation overhead per 200K corpus** (261 windows at window 1024 / stride 768; `ceil(261/N)` server generations, one fewer rotation):

| `-np` | generations | rotations | **rotation wall-clock** |
|---:|---:|---:|---:|
| 8 | 33 | 32 | **180.9 s** |
| 32 | 9 | 8 | 45.2 s |
| 64 | 5 | 4 | 21.8 s |
| **128** | **3** | **2** | **11.3 s** |

This is the case for the big pool, and it is much stronger than §4 anticipated. The spec priced `-np 128` at "2 rotations ≈ 13.4 s" (now 11.3 s) and `-np 64` at 4 rotations "+13.5 s" (now +10.5 s vs 128). But the alternative that actually matters is the status quo: **at `-np 8` the R13 mitigation costs 181 s of pure server restart per 200K corpus** — three minutes of dead time per episode, every episode. Going to 128 removes 94% of that for 7.38 GiB and −1.4% prefill.

## 7. Recommendation

> **`servers.leaf.parallel: 128`** — with `-c 327680` retained (2,560 tok/slot), measured 32.244 GiB leaf residency, 51.53 GiB dual-resident of the 64 GiB carve (19.5% margin), −1.4% prefill / +0.1% decode vs `-np 8`, and 2 rotations × 5.65 s = 11.3 s per 200K corpus.

Do not go above 128: `-np 192`/`-np 256` breach both the 15% margin floor and the ≈1,900 tok/slot the 1024/768 window geometry requires. The `-np 64` fallback named in §4 is measurable but strictly worse here — it costs 4 rotations instead of 2, and it is the one configuration carrying a reproducible +629 MiB of unaccounted allocation.

Consequential edits this measurement licenses (not applied here — `config.yaml` is owned by the code agent):

- §4: replace the estimated "≈63 MB/slot in f32" with **62.8125 MiB/slot measured (S 60.000 MiB + R 2.8125 MiB)**, and state the two axes explicitly — KV at 10.625 KiB/token priced by `-c`, recurrent state at 62.8125 MiB/slot priced by `-np`. Drop the "measure it before pinning `-np 128`" caveat; it is measured.
- §4/§5 C4: correct the rotation assumption from ~6.7 s to **5.65 s warm** (11.3 s per 200K corpus at `-np 128`).
- S0 item 4 unit labels: the leaf residency figures are GiB, not GB.

---

### Raw artefacts

Scratchpad (session-local): `runs/np{8,32,64,64b,128,128ck,256}.{log,err,props.json}`, `runs/np{8,64,128}.thru.csv`, `runs/rotate.{warm,warm-np64,cold}.csv`. Key values are transcribed in full above; the logs are not committed because they contain absolute local paths and 40 MB of per-token trace.
