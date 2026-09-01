# docs/ — three genres, and which is which

This directory holds three kinds of document. The convention was real before it was
written down; this file writes it down.

| directory | genre | what belongs here |
|---|---|---|
| `research/` | **evidence** | Findings and audits. Answers *what is true*. Dated, and once published, treated as a record — a research brief describes what was known on its date and is not retro-edited. |
| `superpowers/specs/` | **design** | A design that consumes evidence and proposes a shape. Answers *what we should build and why*. Reviewed and approved before a plan is written. |
| `superpowers/plans/` | **execution** | A task-by-task implementation plan for an approved design. Answers *in what order, and how do we know each step worked*. |

The pipeline is `research → spec → plan → code`. A document that skips a stage is
usually a document in the wrong genre.

Milestone evidence used to live in `milestones/`, which was **deleted from the
working tree on 2026-08-22**. It is still in git: read any of it with
`git show 4e75b53:milestones/s2/R13.md`. The ~176 citations to it across
`ARCHITECTURE.md`, `CHANGELOG.md`, `config.yaml` and `src/rlm` docstrings were
deliberately not stripped — they are the evidence trail behind every gate
verdict, and `checks/test_citations.py` checks that each one is still
retrievable from that commit.

## Plan checkboxes are not a status signal

There are 171 unchecked `- [ ]` boxes across three plans whose work shipped. This
project never ticked plan checkboxes as tasks landed, and retro-ticking them would
fabricate a per-task history nobody recorded.

**Ground truth for what shipped is `ARCHITECTURE.md` §9's gate status lines and
`CHANGELOG.md`.** Each plan now carries a dated status header saying what actually
happened to it; read that, not the boxes.

## Two documents are not what their location suggests

- `superpowers/plans/2026-08-13-capa1-probe-recipes.md` is a **measurement record**,
  not a plan — 3,197 lines of Windows-sandbox recipes executed on the target box. It
  sits under `plans/` because it is the companion its sibling plan names by filename,
  and the recipes are still live. It carries a genre note.
- `superpowers/plans/2026-08-20-delegation-arm.md` says `Status: planned, not
  implemented` for an arm that **ships**. Its header now corrects this, and records
  that the 5-arm run it planned was stopped after 6 of 90 blocks.

## Coupled documents

`superpowers/specs/2026-08-22-long-horizon-agent-design.md` marks figures `[R]` when
they come from `research/2026-08-22-avo-arc-agi-3-dossier.md` and `[V]` when they were
re-verified against this repo. The two travel together: committing the design without
the dossier leaves every `[R]` citation dangling.
