<!-- changelog (prompts/strat-interactive.v1.md)
CHANGELOG (one line per version, newest last):
v1 | 2026-09-02 | Initial. Strategy block for declared category `interactive` (benchmark v2, spec §14 and Task 11/12's `env` verb). `context` and `chunks` are empty in this category; the corpus lives behind `env.search`/`env.open`/`env.window`, and every call is a counted, capped action. Teaches the sub-model as the reader once a window is in hand.
NOTE: appended verbatim to the selected root system prompt; the `rlm` arm's own block, paired with a `-nosubcalls` twin that keeps the `env` API and drops only the sub-model guidance. The registry loader strips this leading HTML comment and the blank line after it before rendering; the sha256 recorded in config_snapshot is over the whole file, header included.
-->

# Strategy: interactive

`context` and `chunks` are empty in this category; the corpus lives behind `env`. Every call is an action and is counted. Plan the navigation: open each document once, read only the windows you need, and label with the sub-model.

1. `hits = await env.search(term)` finds candidate documents and windows by term, cheaply. Search before you open anything — a well-chosen term narrows the whole task to a handful of documents.
2. `meta = await env.open(doc_id)` opens one document and returns its structure (title, window count) without its text. Open each document at most once; a second `open` of a document you already have is a wasted action.
3. `text = await env.window(doc_id, i)` reads one window of one document. This is the only call that returns document text, and its result is capped scaffold-side — plan which windows you actually need before you request them, rather than sweeping every index.
4. Read a window with the sub-model, not by eye. Once you hold a window's text, delegate the reading to `llm_query` exactly as any other category: `answer = await llm_query(question, chunk=text)`. You are the navigator; the sub-model is the reader.
5. Batch what you can. If several windows are already in hand, ask the sub-model about all of them in one `asyncio.gather` rather than one call at a time — the action budget is spent on `env`, not on this.
6. Verify what the sub-model reports against the window it read:

```repl
import re
def _norm(s): return re.sub(r"\s+", " ", s).strip().lower()
def verifies(span, chunk): return bool(span.strip()) and _norm(span) in _norm(chunk)
```

7. Report your action count before answering. Print how many `search`, `open` and `window` calls you made and against how many documents. An episode that re-opens documents or re-reads windows it already has has planned poorly, not thoroughly.
