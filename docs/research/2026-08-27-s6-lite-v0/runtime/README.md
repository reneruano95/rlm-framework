# Gate runtime state, recovered from WSL before it was torn down

**Date recovered:** 2026-08-31 · **Source:** `/home/spike/gate/` in WSL2 Ubuntu, the sandboxed user
the prime-agent spike ran as. **Status:** these files existed in NO version control. They are the
runtime side of the three decisions recorded in `../decisions/`.

prime-agent was a one-off spike to establish that local models can drive a recursive harness. It is
not part of the destination architecture, and the WSL tree is disposable. These files are not: the
ledger is the only record of whether an artifact actually reached the model, per episode.

**Integrity.** Copied byte-for-byte and verified against the originals:
`ledger.jsonl` sha256 `e1b0b093751278ae1ce00c247505b772175fb4c0a2b676837e7e9b0d8cc3f801`
on both sides. `SHA256SUMS` in this directory covers every file.

Two things were already in the repo and are not duplicated here: `/home/spike/gate/split.json`
(sha `78cc7985...`, byte-identical to `bench/splits/s6lite-v0.json` and its recorded `.sha256`),
and the two runners, which hash identically to `gate/run_decision.sh` and `gate/run_episode.sh`
once CRLF is normalised (`a1dcb001...`, `c90b1d3c...`). **The mirror had not drifted for code.**
Drift ran the other way only: the repo carries 20 scripts in `gate/checks/`, WSL had 18 —
`diag_stall.sh` and `resume_pc03.sh` exist only in the repo.

---

## Contents

| file | what it is |
|---|---|
| `ledger.jsonl` | 311 rows, 2026-08-27T19:35:58Z → 2026-08-28T10:52:00Z. The gate extension's own event log. |
| `accepted.json` | the artifact set handed to the extension. As recovered it is the EMPTY set — `{"entries": {"prompt":{},"memory":{},"skill":{},"subagent":{}}, "refinements": []}` — i.e. the state left behind after the last OFF episode. |
| `empty.json` | the canonical empty artifact set `run_episode.sh` writes before every OFF episode |
| `heldout.txt` | the nine held-out task ids: codeqa-04/05/07, needle-05/06/07/08, agg-06/07 |
| `train.txt` | the train-side task ids |
| `reliability.txt`, `verify.txt` | small run scratch files |

---

## What the ledger says, and it is not only confirmation

`gate/extension/rlmh-gate.ts` registers four hooks and **all four call `log()` when they fire** —
`session_before_refine` logs `before_refine` unconditionally at line 236, before its branch;
`before_agent_start` logs `prompt_filtered` every time; `tool_call` logs `identical_turn_blocked`
only on a block; `refine_complete` logs every time. So an absent event type means the hook did not
fire, not that it fired silently.

The ledger contains **exactly two event types**: `loaded` ×156 and `prompt_filtered` ×155.

**1. The delivery split is clean, and it is the evidence the decisions rest on.**

```
kept  stripped  accepted   episodes
   0         0         0         78     OFF arm — nothing delivered
   1         0         1         77     ON arm — one artifact delivered
```

Two states, no third. This is the per-episode proof that the ON arm received the artifact and the
OFF arm did not — the measurement `2026-08-28-s6-lite-v0-results.md` §1 depends on when it reports
`in_window=1` on every pc-01 ON episode.

**2. `stripped` is 0 in all 155 rows, and there are zero `stripped_entries` events.** The filter
never removed a pre-existing harness entry in any episode. It was never exercised against the case
it was built for.

**3. Zero `before_refine` rows — so `session_before_refine` never fired in 156 episodes.** The I1
veto is credited as a mechanism in the results document, and it was verified directly there (the
model called `refine.run(...)`, got `{'scheduled': True}`, and harness state did not change). But
across the three recorded decisions **the model never attempted a self-refine, so the veto never
had to act.** The guard is verified by its own test, not by the decision runs.

**4. Zero `identical_turn_blocked` rows.** The C5 repetition guard, restored for a harness that
ships none, **never blocked a cell in 156 episodes.** R15 did not manifest inside the gate runs at
the configured `MAX_IDENTICAL`. Its value there is insurance, not a measured save.

**5. Zero `refine_complete` rows** — consistent with 3, and the clean reading: no refinement ever
completed, so no unexpected write escaped the veto.

**6. 156 `loaded` against 155 `prompt_filtered`.** One episode loaded the extension and never
reached `before_agent_start`. The likeliest cause is recorded in `gate/run_decision.sh`: the
llama-server died mid-decision during pc-03 at 22/40 and every WSL process went with it. Not
reconciled against the episode directories.

---

## Why this was nearly lost

The gate wrote evidence to three places and only two were versioned: `decisions/pc-*/decision.json`
and `decisions/audit.jsonl` in the repo, and `ledger.jsonl` here. See
`2026-08-31-papers-verified.md` §9 for the other half of the same defect — that audit ledger holds
2 rows for 3 spent decisions, because `run_decision.sh` never calls `decide.py` and `--audit`
defaults to `None`.
