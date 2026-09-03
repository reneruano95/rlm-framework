<!-- changelog (prompts/strat-interactive-nosubcalls.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. The `rlm-nosubcalls` arm's twin of strat-interactive.v1.md: same `env` navigation, minus every sentence naming a second reader -- this arm's root reads every window it opens itself.
NOTE: appended verbatim to the selected root system prompt. The runtime refuses the sub-call API in this arm; this file makes the prompt agree with the runtime. `context` and `chunks` are still empty in this category -- the corpus lives behind `env`, exactly as in the `rlm` arm. Pinned only in config.v2.yaml. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: interactive

`context` and `chunks` are empty in this category; the corpus lives behind `env`. Every call is an action and is counted. You have `env` and Python; read what you need through `env.window` and reason over it in code.

1. `hits = await env.search(term)` finds candidate documents and windows by term, cheaply. Search before you open anything — a well-chosen term narrows the whole task to a handful of documents.
2. `meta = await env.open(doc_id)` opens one document and returns its structure (title, window count) without its text. Open each document at most once; a second `open` of a document you already have is a wasted action.
3. `text = await env.window(doc_id, i)` reads one window of one document. This is the only call that returns document text, and its result is capped scaffold-side — plan which windows you actually need before you request them, rather than sweeping every index.
4. Read the window yourself, in the turn where you fetched it. There is no other reader available; extract or decide what the question needs directly from `text`, in Python or in your own reasoning.
5. Keep what you have learned in a Python variable as you go, so a later turn never has to re-fetch a window you already read.
6. Verify what you extracted against the window it came from:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

7. Report your action count before answering. Print how many `search`, `open` and `window` calls you made and against how many documents. An episode that re-opens documents or re-reads windows it already has has planned poorly, not thoroughly.
