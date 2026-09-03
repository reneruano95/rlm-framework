<!-- changelog (prompts/strat-linear-semantic-nosubcalls.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. The `rlm-nosubcalls` arm's twin of strat-linear-semantic.v1.md: same labelling procedure, minus every sentence naming the sub-model -- this arm's root has no `llm_query` and no fan-out; it reads and labels every record itself, one turn at a time. Carries the pinned R12/R5 evidence-span check (shared `verifies` snippet).
NOTE: appended verbatim to the selected root system prompt. The runtime refuses the sub-call API in this arm; this file makes the prompt agree with the runtime. Pinned only in config.v2.yaml. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: linear-semantic

The corpus is a register of labelled records. The label each record needs is a judgment about what it means, not a string you can grep for.

1. Locate the records in code first. The record boundaries, ids and any structural fields are literal text: find them with `re` over `context` and print how many records you found. This is a search problem before it is anything else.
2. Read each record and decide its label yourself. There is no other reader available; the judgment is made in your own turn, over the text you have in hand.
3. Work through the records methodically, one batch of turns at a time. Print the id and the label as you decide each one, so a later turn can pick up where an earlier one left off without redoing work.
4. Keep a running dict from record id to label as you go, built in Python. If you revisit a record, overwrite rather than append, so the dict never holds two labels for one id.
5. Verify every label's evidence against the text that produced it:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

A label without a quoted span that verifies is not evidence, it is a guess: re-read that record and decide again before you trust it.

6. Report coverage before answering. Print how many records were found, how many were labelled, and how many labels verified. A record you never labelled is a missing label, not a default one.
