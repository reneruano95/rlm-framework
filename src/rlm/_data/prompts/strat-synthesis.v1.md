<!-- changelog (prompts/strat-synthesis.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. Strategy block for declared category `synthesis`. Extraction-shaped at the citation level: carries the pinned R12/R5 evidence-span check (shared block, byte-identical across every template that carries it) applied to each cited support quote.
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: synthesis

Several documents, one answer, and every claim in it has to be supported by them.

1. Brief each document once. One `asyncio.gather` over `chunks`, the same request to each: a short structured brief on the task's dimensions, a fixed small number of labelled lines, each line ending with a verbatim quoted snippet from that document as its support.
2. Keep the briefs small. They all have to fit in your own window at reduce time. Cap the line count in the request; if a brief comes back long, re-ask that one chunk with a tighter cap instead of trimming it yourself.
3. Merge in code, then compose. Group the briefs by dimension in Python and print the grouped structure. Write the synthesis from that structure, not from a re-read of the raw briefs.
4. Every claim carries a source. Each statement in the final answer must trace to a chunk index whose quoted support actually occurs in that chunk:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

An unverified quote invalidates the claim it supports: drop the claim, or re-ask that chunk for a quote it can actually produce.

5. Name the disagreements. Where documents conflict, say so and attribute both sides. A smoothed-over consensus that no document actually states is the characteristic failure of this category, and it is worse than reporting the conflict.
