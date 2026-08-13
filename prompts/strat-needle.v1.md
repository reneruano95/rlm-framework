<!-- changelog (prompts/strat-needle.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. Strategy block for declared category `needle`. Extraction-shaped: carries the REPL-prescan tip and the pinned R12/R5 evidence-span check (shared block, byte-identical across every template that carries it).
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: needle

Exactly one short fact is somewhere in there. Find *where* in code, then read only there.

1. Scan in code first. Write the fact's surface form as a regex or a keyword set and run it over `chunks`. Literal ids, codes, names, numbers, dates and quoted phrases are found this way for zero sub-calls. Print the matching chunk indices and the matched strings, never their surroundings.
2. Widen before you fan out. No hits means relax the pattern — case-insensitive, a shorter stem, a partial token, a synonym set, `str.find` on the distinctive word — and rescan. Rescanning is free; a full fan-out is not.
3. Confirm each surviving candidate with one sub-call. Ask the narrow question against that chunk, chunk first and question last, and tell the sub-model to reply `NONE` when the fact is absent.
4. Fan out only if the scan fails outright. If the fact is paraphrased and no pattern reaches it, ask every chunk the same question in one `asyncio.gather`, then keep the non-`NONE` answers and treat each as a candidate.
5. Verify the span before you submit. The answer must actually occur in the chunk that produced it:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

A sub-answer that does not verify is a confabulation, not evidence: drop it, re-ask with a tighter question, or report the fact as absent. Never submit an unverified span as though it were quoted.

6. If several chunks return different answers, prefer the one that verifies. If more than one verifies, say so in the answer rather than picking silently.
