<!-- changelog (prompts/strat-linear-semantic.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. Strategy block for declared category `linear_semantic` (benchmark v2, spec §14). Aggregation over labelled records: one map over all `chunks` with a single `asyncio.gather`, one label word per record id, then a Python reduce by identity over `context`. Carries the pinned R12/R5 evidence-span check (shared `verifies` snippet).
NOTE: appended verbatim to the selected root system prompt; the `rlm` arm's own block, paired with a `-nosubcalls` twin that never names the sub-model. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: linear-semantic

The corpus is a register of labelled records. The label each record needs is a judgment about what it means, not a string you can grep for.

1. Locate the records in code first. The record boundaries, ids and any structural fields are literal text: find them with `re` over `chunks` and print how many records you found. This is a search problem before it is anything else.
2. The kind of thing a Query asks about is a judgment about meaning; no pattern over the words settles it. Code locates the records; the sub-model labels them.
3. Map once over all of `chunks`. One `asyncio.gather` over `llm_query(question, chunk=chunks[i])` for every chunk, no exceptions and no early stopping. Ask for exactly one label word per record id it can see — never a running count, never a summary.
4. Reduce by identity over `context`. Build a dict from record id to label in Python. If two chunks report the same record id, keep the value that occurs in `context` for that id and drop the other; never average or vote between them.
5. Verify every label's evidence against the chunk that produced it:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

A label without a quoted span that verifies is not evidence, it is a guess: re-ask that record with a narrower question before you trust it.

6. Report coverage before answering. Print how many records were found, how many were labelled, and how many labels verified. A record that never got mapped is a missing label, not a default one — re-ask it rather than assuming.
