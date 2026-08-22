# runs/ — where a bench run writes

`rlm bench` defaults here: `runs/ledger.jsonl` and `runs/RESULTS.md`.

**Not `milestones/`, deliberately.** That directory is the evidence archive for
gates already taken. A default that landed there meant the next run appended a
new grid into a closed ledger, and regenerated a report over one whose
hand-written half lives below `NARRATIVE_MARKER` — silently, because
`verdict.regenerate()` guards on `path.exists()`.

Contents are gitignored: this is runtime state, not source and not evidence.
When a run becomes evidence, it moves to `milestones/<gate>/` with the document
that interprets it.

To resume a historical grid, name it: `rlm bench --resume --ledger
milestones/s4/results/ledger.jsonl`.
