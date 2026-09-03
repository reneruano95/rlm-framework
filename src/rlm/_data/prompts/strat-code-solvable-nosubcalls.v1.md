<!-- changelog (prompts/strat-code-solvable-nosubcalls.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. The `rlm-nosubcalls` arm's twin of strat-code-solvable.v1.md: the `rlm` block never asked for a second reader in the first place, so this is the same code-only procedure with the one sentence that named that reader as unavailable dropped.
NOTE: appended verbatim to the selected root system prompt. Pinned only in config.v2.yaml. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: code-solvable

This task is answerable in code. Count, match or locate with `re` over `context`.

1. Read the question as a pattern-matching problem before anything else. A count, a lookup, a literal match, a position — every shape this category takes is exactly greppable over `context`.
2. Write the pattern, run it, print the result, and look at it. `re.findall`, `re.finditer`, `str.count`, `collections.Counter` — plain library code is the whole solution.
3. Cross-check with a second, independently written pattern before you trust a count. Two patterns that disagree mean the first one was wrong.
4. If a first pattern comes back empty, widen it in code — case-insensitivity, a shorter stem, an alternate delimiter — and rerun. Widening the pattern is free.
5. Submit only what you verified in code. Print the value, confirm it against the pattern that produced it, then call `final_answer`.
