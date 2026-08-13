<!-- changelog (prompts/strat-aggregation.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. Strategy block for declared category `aggregation`. Extraction-shaped: carries the REPL-prescan tip (the category deliberately contains regex-solvable tasks) and the pinned R12/R5 evidence-span check (shared block, byte-identical across every template that carries it).
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: aggregation

The answer depends on **every** chunk. Sampling is failure, and one missed item is a wrong answer.

1. Decide first whether code alone suffices. If the items are literally identifiable — a token, a pattern, a delimiter, a field name — count them with `re` and `collections.Counter` over `chunks` and skip sub-calls entirely. Cross-check the count with a second, independently written pattern before you trust it.
2. Otherwise map once over all of `chunks`. One `asyncio.gather`, the same question for every chunk, no exceptions and no early stopping. Ask for an enumerable answer — one item per line, or `NONE` — never for a number. Counting is the reduction step's job, not the sub-model's.
3. Reduce in Python. Parse each answer into items, normalize them, deduplicate across chunks, then count or aggregate. Items split across a chunk boundary are the standard failure here: inspect the tail of each chunk and the head of the next for a truncated item before you finalize.
4. Verify every extracted item against the chunk that reported it:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

Drop items that do not verify and print how many you dropped. A large drop count means the question was ambiguous, not that the corpus is empty — re-ask those chunks with a sharper question.

5. Report coverage before answering. Print how many chunks answered, how many returned `NONE`, and how many items survived verification. If any chunk errored, timed out, or was skipped, the aggregate is not complete: re-ask that chunk rather than answering from a partial map.
