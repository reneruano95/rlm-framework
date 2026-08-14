<!-- changelog (prompts/strat-aggregation.v2.md)
CHANGELOG (one line per version, newest last):
v2 | 2026-08-13 | v2 = strat-aggregation.v1.md with its COUNTING guidance corrected for overlapping windows, and nothing else. v1 was written against a partitioning chunker, where `chunks` tiled the corpus exactly once; since spec v0.2.5 §7 #2 the shipped geometry is window 1,024 / stride 768, so `chunks` are OVERLAPPING windows and every token appears in up to two of them. Three of v1's five steps were wrong under that geometry and are rewritten: step 3's reduction (deduplicating "across chunks" and then counting silently double-counts every item that falls in an overlap, which is ~25% of the corpus by construction — the correction states that per-chunk counts are never summed and that occurrence counts are resolved against `context`, the only non-repeating view); step 3's boundary-stitching tip (v1 told the root to inspect the tail of each chunk and the head of the next for a truncated item — under overlap that item is already whole in the neighbouring window, and stitching manufactures a phantom second copy of it, so the tip is inverted); and step 5's coverage report (v1's "how many chunks answered" reads as corpus coverage, which it is not when windows overlap). Steps 1, 2 and 4 — the code-first decision, the full-coverage map, and the pinned R12/R5 evidence-span check — are byte-identical to v1. v1 is NOT modified: it is the S2-recorded text for every run made before this date.
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: aggregation

The answer depends on **every** chunk. Sampling is failure, and one missed item is a wrong answer.

**`chunks` overlaps.** The windows are cut at a stride shorter than their own length, so consecutive windows share text and roughly a quarter of the corpus appears in two of them. That is deliberate — it is what keeps every part of the corpus close enough to the end of some window to be retrievable — but it means `chunks` is not a partition and the count of things in `chunks` is not the count of things in the corpus. `context` is the corpus; `chunks` is a covering of it.

1. Decide first whether code alone suffices. If the items are literally identifiable — a token, a pattern, a delimiter, a field name — count them with `re` and `collections.Counter` over `chunks` and skip sub-calls entirely. Cross-check the count with a second, independently written pattern before you trust it.

   Count over `context`, not over `chunks`. A pattern applied to `chunks` counts the overlaps twice.

2. Otherwise map once over all of `chunks`. One `asyncio.gather`, the same question for every chunk, no exceptions and no early stopping. Ask for an enumerable answer — one item per line, or `NONE` — never for a number. Counting is the reduction step's job, not the sub-model's.
3. Reduce in Python, and reduce **by identity, never by addition**. Parse each answer into items, normalize them, then take the union across chunks. Never sum per-chunk counts and never sum per-chunk list lengths: an item that lies in an overlap is reported by both windows, so a sum over-counts by construction rather than by accident.

   If the question is about **distinct** things — how many different X, which X appear — the union is the answer.

   If the question is about **occurrences** — how many times X appears — resolve them against `context`, which is the only non-repeating view of the corpus: use the map to learn *what* to look for, then locate and count the occurrences positionally (`re.finditer` over `context`, or `context.find` from the previous hit). Two occurrences of the same string are two only if they sit at different offsets in `context`.

   Do **not** stitch the tail of one chunk to the head of the next. Under a partitioning chunker that was the standard repair for an item cut in half; with overlapping windows an item cut by one window's edge is already intact in its neighbour, so stitching invents a second copy of an item you have already counted.

4. Verify every extracted item against the chunk that reported it:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

Drop items that do not verify and print how many you dropped. A large drop count means the question was ambiguous, not that the corpus is empty — re-ask those chunks with a sharper question.

5. Report coverage before answering. Print how many windows answered, how many returned `NONE`, and how many items survived verification — and read those numbers as being about windows, not about the corpus: windows overlap, so "12 of 14 answered" is not "86% of the corpus seen", and two windows agreeing on an item is one item, not corroboration. If any window errored, timed out, or was skipped, the aggregate is not complete: re-ask that window rather than answering from a partial map.
