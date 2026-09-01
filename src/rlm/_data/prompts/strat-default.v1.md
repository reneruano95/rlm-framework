<!-- changelog (prompts/strat-default.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. Strategy block for ad-hoc tasks with no declared category. Generic loop; carries the REPL-prescan tip and the pinned R12/R5 evidence-span check (shared block, byte-identical across every template that carries it) as a conditional step, since the answer shape is unknown here.
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: default

No category-specific procedure applies to this task, so use the generic loop.

1. Characterize before deciding. First cell: `len(chunks)`, total characters, the first few hundred characters of one chunk, and whatever structure is visible — headers, delimiters, repeated fields, file paths. Choose a plan from what you actually see, not from what the question implies.
2. State the plan, then execute it. One short paragraph naming the turns and the calls each will issue, then one cell per turn.
3. Try code before sub-calls. If a scan, a count or a parse over `chunks` answers the question outright, or narrows it to a few chunks, that is the whole first phase and it costs nothing.
4. Prefer one fan-out to many turns. Whatever must be read by the sub-model, ask it of all relevant chunks at once with `asyncio.gather`, then reduce the results in Python.
5. Verify whatever can be verified. Where the answer is, or contains, a span that must occur in the corpus, check it before submitting:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

Where the answer is not a span — a judgement, a summary, a count — say what it rests on, and sanity-check it in code where you can: recount, rescan, or derive it a second way and compare.

6. Submit deliberately. Print the value, look at it, then call `final_answer`.
