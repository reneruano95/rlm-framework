# What window/stride overlap does to B2 and B3

**Date:** 2026-08-13 · **Spec:** ARCHITECTURE.md v0.2.6 (§7 #2 geometry, §8 baselines + chunk-size lock, §5 C2)
**Status:** statement of consequences. Nothing here changes §8; it exists so §8's pre-registered comparison is restated **deliberately** rather than drifting under a chunker change made for the RLM arm.

---

## 1. Why this note exists

§8 defines two of the three controls in terms of the RLM arm's own chunker:

> **B2:** deterministic map-reduce — scaffold-only chunking (**the C2 chunker, verbatim**) → leaf summaries → root final.
> **B3:** deterministic BM25-RAG single shot — **the C2 chunker verbatim**; chunks indexed with DuckDB's FTS extension (BM25) … filled to a pre-registered 80% of window, restoring original document order.

and the chunk-size lock leans on exactly that shared dependency:

> any post-S4 chunk-size change re-runs **all three chunked arms** (B2 and B3 share the C2 chunker verbatim, **so the controls stay controlled**).

"Verbatim" was written when C2 produced a **partition** — `chunks` tiled the corpus exactly once. Since §7 #2 (v0.2.5) the shipped geometry is **window 1,024 / stride 768**, so `chunks` is a **covering, not a partition**: consecutive windows share text and **(size − stride)/stride = 256/768 = 33.3% of the corpus by tokens appears in two windows**. Inheriting that "verbatim" transfers a semantic neither baseline was specified against. The changes below are consequences of the geometry, not choices anyone made for the baselines.

Arithmetic throughout uses a 200,000-**token** corpus (§7 #2's own reference size) and the shipped per-call head of 311 prefix + ~50 question tokens.

| | partition @ 1,024 | overlap 1,024 / 768 | change |
|---|---:|---:|---:|
| windows | `ceil(200,000/1,024)` = **196** | `ceil((200,000−1,024)/768)+1` = **261** | **+33.2%** |
| prompt tokens per call | 1,024 + 361 = 1,385 | 1,024 + 361 = 1,385 | 0 |
| total prompt tokens (one pass) | 271,460 | **361,485** | **+33.2%** |
| leaf-process generations at `-np 128` | 2 (1 rotation) | 3 (2 rotations) | +5.65 s |

(§7 #2's headline "~24% more prefill" is the **per-call** comparison against non-overlapping **768**-chunking at equal sub-call count: 1,385 vs 1,129 tokens = +22.7%. The +33.2% above is the comparison the baselines actually face — same window size, more windows.)

---

## 2. B2 — deterministic map-reduce

**Three changes, all mechanical, one of them a possible outcome flip.**

1. **Cost, +33.2%.** B2 makes one leaf call per window, so its sub-call count, prompt tokens and leaf wall-clock all rise by the window-count ratio. Its cost column in the §8 cost scorecard is not comparable to any B2 figure computed before v0.2.5.

2. **The reduce step now sees duplicated content.** B2's root final receives one summary per window; a third of the corpus is summarised twice. For extraction-shaped tasks that is harmless duplication, but for the **aggregation** category — the one §8 says must "force coverage and punish sampling" — a reduce that sums per-window counts over-counts by construction. This is the same defect the pinned strategy template carried and which `prompts/strat-aggregation.v2.md` now corrects for the RLM arm. **B2's reducer is scaffold code, not a prompt, so it must carry the same correction or B2 is measured with a known-wrong reduction.**

3. **The root window gets tighter, and this one can change an outcome.** 261 summaries against 196, at ~100 tokens each, is ~26,100 vs ~19,600 tokens into a 32,768-token root window. C5 kills at 90% (29,491) and §6 records `context_exhausted`. The partition version sits at 60% of the window; the overlap version at 80%, with the margin now depending on summary length. A B2 arm that starts scoring `context_exhausted` is scoring **failures** (§8: `context_exhausted` counts as a failure for every arm) — for a reason that has nothing to do with map-reduce being a weak strategy.

4. **R13 exposure rises with call count.** §8 (v0.2.6) names the arms' split as two chunked-and-exposed (RLM, B2) versus two single-shot-and-spared (B1, B3). B2 makes 33% more leaf calls, hence 33% more never-reused slots and one extra rotation per 200K corpus. The per-arm contamination count the verdict now reports must be read **per call**, not as an arm total, or B2 will look dirtier than RLM purely for making more calls.

---

## 3. B3 — BM25-RAG single shot

B3 is a **single-shot** arm: it never touches the leaf's slot pool, so none of R13 applies to it. Overlap reaches it through the **index**, and this is the part that is not merely a cost change.

1. **The document collection is now 33% larger and partly duplicated.** 261 index documents instead of 196, with a third of the corpus present twice.

2. **BM25's IDF is distorted by position, not by relevance.** IDF is computed over document frequency: a term that happens to lie in an overlap region appears in **two** documents rather than one, so its `df` doubles and its IDF falls. Whether a term is down-weighted therefore depends on **where the window boundaries fell**, which is an artifact of the stride. Both the corpus-level statistics and the resulting ranking change; B3's retrieval is not the same function it was under a partition.

3. **Window-fill selection wastes budget on text it has already selected.** The pre-registered rule is "fill to 80% of the B1 256K window, restoring original document order", with **no tunable k**. Under overlap, two adjacent selected windows contribute 25% duplicated tokens, so the 80% fill contains strictly less unique corpus than it did — and "restoring original document order" produces a prompt with visibly repeated passages, which is also an adversarial-ish input shape nobody chose.

4. **The honest repair is span-level dedup, and it is a rule change that must be pre-registered, not slipped in.** Selecting ranked windows and dropping text already selected (by character span in `context`) restores "80% of window = 80% of window's worth of unique corpus" without introducing a tunable. It is still a change to a pre-registered selection rule and belongs in §8 explicitly.

---

## 4. The decision §8 has to state

Two coherent options. Neither is free, and the point of this note is that one of them gets chosen on the record.

**(A) Keep "verbatim" literal — B2 and B3 run the shipped overlapping chunker.**
Preserves the property the chunk-size lock is built on (one chunker, so the controls stay controlled under any future geometry change). Costs: B2 +33.2% and a root-window risk that can manufacture `context_exhausted` failures; B3 needs the span-dedup rule above, and its IDF statistics are geometry-dependent. Both consequences must be written into §8 before the run.

**(B) Pin the controls to `stride == size` (a partition) and say so.**
Keeps B2's cost profile and B3's BM25 statistics as originally specified — B3 in particular stops having a retrieval function that depends on stride. Costs: the controls no longer share the RLM arm's chunker, so the lock's "so the controls stay controlled" clause is false as written and must be replaced by an explicit statement that the arms differ in geometry by design. It also hands RLM an advantage the controls do not get (the horizon property §7 #2 measured as worth 38/39 vs 0/39 retrieval), which is precisely the "sweeping RLM's main lever while the controls get no analogous tuning" hazard the lock exists to prevent.

**Recommendation: (A), with the three §8 edits named.** The lock's whole purpose is that a geometry change moves all chunked arms together, and (B) breaks that to protect a cost baseline. What (A) requires is not a re-design, only that §8 says so out loud:

- **§8 B2:** state that B2's reducer must deduplicate by identity and never sum per-window counts, for the same reason `strat-aggregation.v2.md` does, and record B2's cost baseline as computed under the shipped geometry.
- **§8 B3:** restate the selection rule as "fill to 80% of the window with ranked windows, dropping text already selected (span-level dedup against `context`), restoring original document order" — still no tunable k — and note that BM25 IDF is computed over overlapping documents.
- **§8 cost scorecard / contamination monitor:** B2's token and wall-clock figures are +33.2% versus any pre-v0.2.5 projection, and its R13 contamination count is to be reported per leaf call.

---

## 5. What is NOT affected

- **B1** takes no chunker at all (single shot, head+tail truncation to fit the window). Unchanged.
- **The `max_subcalls` budget** already carries the overlap geometry (522 = 261 windows × 2 questions); it was re-derived for it in v0.2.5. See the separate finding on the 1M-char end of §8's needle range, where 522 is **not** sufficient.
- **The §7 #2 distance-cliff and false-positive findings.** They were measured per-chunk on `s2/results/sweep.jsonl`, which audits clean for R13 (117 calls, 0 foreign identifiers), and no part of them depends on whether windows tile or overlap.
