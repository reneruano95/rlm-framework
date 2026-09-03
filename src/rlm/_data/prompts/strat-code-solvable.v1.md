<!-- changelog (prompts/strat-code-solvable.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. Strategy block for declared category `code_solvable` (benchmark v2, spec §14): the control stream, regex-solvable by construction. States plainly that a sub-call is a cost here, not a method -- code alone answers the question.
NOTE: appended verbatim to the selected root system prompt; the `rlm` arm's own block, paired with a `-nosubcalls` twin that is byte-for-byte the same guidance minus the sentence naming the sub-model (nothing else changes, since this block never asks for one). The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: code-solvable

This task is answerable in code. Count, match or locate with `re` over `context`; a sub-call here is a cost, not a method.

1. Read the question as a pattern-matching problem before anything else. A count, a lookup, a literal match, a position — every shape this category takes is exactly greppable over `context`.
2. Write the pattern, run it, print the result, and look at it. `re.findall`, `re.finditer`, `str.count`, `collections.Counter` — plain library code is the whole solution, not a first pass before delegation.
3. Cross-check with a second, independently written pattern before you trust a count. Two patterns that disagree mean the first one was wrong, not that the task needs a sub-model to adjudicate.
4. If a first pattern comes back empty, widen it in code — case-insensitivity, a shorter stem, an alternate delimiter — and rerun. Widening the pattern is free; there is nothing here that reading buys you.
5. Submit only what you verified in code. Print the value, confirm it against the pattern that produced it, then call `final_answer`.
