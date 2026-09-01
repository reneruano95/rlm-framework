<!-- changelog (prompts/strat-codeqa.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-08-13 | Initial. Strategy block for declared category `code_qa`. Extraction-shaped: carries the REPL-prescan tip (source code is exactly greppable) and the pinned R12/R5 evidence-span check (shared block, byte-identical across every template that carries it).
NOTE: appended verbatim to the selected root system prompt; identical in both S1 A/B arms. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: code QA

A repository, flattened into `chunks`. Source code is exactly greppable, so most of this task is a search problem, not a reading problem.

1. Grep first, always. Symbol names, `def` / `class` / `struct` / `func` declarations, imports, decorators, call sites, config keys and file-path headers are all exact strings. Locate them with `re` over `chunks` before any sub-call, and print chunk indices plus the matched lines only.
2. Distinguish definition from use. Search for the declaration form and the call form separately and count both. "Where is X defined" and "what calls X" are different scans, and comparing their counts tells you whether you have the whole picture or only part of it.
3. Follow the chain in code. Callers, imports, re-exports and inheritance are more scans, not more sub-calls. Build the map in Python first; only then decide what actually has to be read.
4. Spend sub-calls on semantics, not location. Send one candidate chunk with a specific question — what this function returns, what this condition guards, why this branch exists — and require an answer that quotes the deciding line verbatim.
5. Verify the quoted line against its chunk:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

A quoted line that does not occur in the chunk means the sub-model reconstructed plausible-looking code from memory. That is the most common failure on this category. Drop it and re-ask.

6. Answer at the level asked — a name, a file, a line, a short explanation. Do not reproduce large code blocks; you would only see them truncated anyway.
