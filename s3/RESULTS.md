# S3 RESULTS — budgets + tracing gate

_Generated 2026-08-16T17:25:19+00:00 by `s3/run_s3.py --phase report` from `s3/results/runs.jsonl` and the trace store `s3/results/store/`. Numbers come from the store, not process memory._

## Verdict rule (stated before the numbers)

The gate passes iff ALL of: (1) the component suite is green; (2) the runaway-REPL task terminates as (budget_kill, wall_clock) on 3/3 attempts with identical (outcome, reason) and episode duration <= budget + 30s teardown slack; (3) the unbounded-sub-call task terminates as (budget_kill, max_subcalls) on 3/3 attempts with >0 dispatched calls and distinct call_ids <= 8; (4) after TerminateProcess on the scaffold mid-step: the sandbox tree is reaped by the Job Object, the store reopens with the episode row (NULL outcome) and every committed step intact, `sweep_orphan_blobs` finds no row referencing a missing blob, and the next `rlm run` tombstones the orphan as (error, orphaned_at_recovery); (5) the live sandbox pid shows no non-loopback endpoint; (6) `rlm replay` exits 0 with root_view_hash OK, message array OK, and a rendered transcript for EVERY episode in the store — budget-killed and tombstoned episodes included — with the lifecycle log deleted.

## S3 GATE: PASS

| check | result |
|---|---|
| component suite green (before and after phases) | PASS |
| runaway 3/3 (budget_kill, wall_clock), identical | PASS |
| subcalls 3/3 (budget_kill, max_subcalls), identical | PASS |
| hard-kill: job reaps sandbox | PASS |
| hard-kill: C6 durability (row+steps survive, no dangling blob rows) | PASS |
| hard-kill: post-restart tombstone (error, orphaned_at_recovery) | PASS |
| no-egress audit (§5 C1) | PASS |
| replay: every episode, lifecycle log deleted | PASS |

## Attempts

- `runaway` #1: PASS — (budget_kill, wall_clock), episode 19.867643 s
- `runaway` #2: PASS — (budget_kill, wall_clock), episode 19.926318 s
- `runaway` #3: PASS — (budget_kill, wall_clock), episode 19.840663 s
- `subcalls` #1: PASS — (budget_kill, max_subcalls), episode 0.014514 s, 8 calls / 8 distinct
- `subcalls` #2: PASS — (budget_kill, max_subcalls), episode 0.011087 s, 8 calls / 8 distinct
- `subcalls` #3: PASS — (budget_kill, max_subcalls), episode 0.01562 s, 8 calls / 8 distinct
- hard-kill durability: PASS — 1 step(s) committed, step0 ok, outcome_at_death=None, orphan blobs 0, pid match True
- tombstone: PASS — {'outcome': 'error', 'outcome_reason': 'orphaned_at_recovery'}, different instance: True
- egress audit: PASS — 3 loopback row(s): ['0.0.0.0:64135->0.0.0.0:0 Bound', '127.0.0.1:64135->127.0.0.1:64134 Established', '127.0.0.1:64134->127.0.0.1:64135 Established']

## Replay (lifecycle log deleted)

- `s3-runaway` (budget_kill, wall_clock): PASS (rc=0)
- `s3-runaway` (budget_kill, wall_clock): PASS (rc=0)
- `s3-runaway` (budget_kill, wall_clock): PASS (rc=0)
- `s3-subcalls` (budget_kill, max_subcalls): PASS (rc=0)
- `s3-subcalls` (budget_kill, max_subcalls): PASS (rc=0)
- `s3-subcalls` (budget_kill, max_subcalls): PASS (rc=0)
- `s3-hardkill` (error, orphaned_at_recovery): PASS (rc=0)
- `s3-recovery` (success): PASS (rc=0)
- `s3-hardkill` (error, orphaned_at_recovery): PASS (rc=0)
- `s3-recovery` (success): PASS (rc=0)
- `s3-hardkill` (error, orphaned_at_recovery): PASS (rc=0)
- `s3-recovery` (success): PASS (rc=0)

<!-- HAND-WRITTEN FINDINGS BELOW — regeneration preserves this -->

## Findings (hand-written)

**The scaffold needed zero changes.** Every check passed against `rlm/` as
committed, including the two shapes earlier spec versions had flagged as
unexercised risks: replay of a budget-killed episode (cancelled step, NULL
observation_view) and replay of a tombstoned episode. Both verify as-built.
The C6 durability promise held under a real `TerminateProcess`: the killed
episode lost exactly the in-flight step (step 2, the sleeping cell) and
nothing else — 1 committed step, 0 orphan blobs, no row referencing a
missing blob, WAL replay clean on the next open.

**Both mid-gate corrections were runner bugs, the arch-ladder class again
(an instrument measuring its own configuration):**

1. *The egress audit first measured the scaffold, not the sandbox.* The
   venv's `Scripts\python.exe` is a trampoline that re-execs the real
   interpreter as a child — exactly what `config.yaml`'s sandbox comment
   warns about from the inside; observed here from the outside:
   `Popen.pid`'s python child IS the scaffold (`scaffold_instance_id`
   matches it), and the sandbox is one level deeper (`episodes.sandbox_pid`
   matches THAT). The fixed audit walks the tree and cross-checks both pids
   against the episode row; both are exact.
2. *The first audit rule misread the Windows TCP table.* asyncio's
   self-pipe renders as THREE rows: two crossed loopback Established rows
   plus a remote-less `0.0.0.0:<pair-port> -> 0.0.0.0:0 Bound` shadow of
   one pair socket. A control run (bare `asyncio.new_event_loop()` in the
   base interpreter, no sandbox) reproduces the identical shape, so the
   rule accepts exactly that and nothing else — a real remote, a Listen
   state, or a port outside the pair still fails.

**`sweep_orphan_blobs` ran for the first time.** It existed in `rlm/trace.py`
with zero callers; this gate wired it in as the dangling-row detector, both
immediately post-kill and post-recovery. Clean both times.

**Caveat, stated rather than implied:** mock-dispatcher episodes exercise
the ROOT half of the per-step `root_view_hash` check only — `MockDispatcher`
stores no rendered leaf request, so replay's leaf-blob rehash has nothing to
rehash in this store. The leaf half runs under the real dispatcher (S1/S2
episodes) and its rendering is covered by the dispatcher's own tests.

**Incidental observation for R6:** during the sleep window the sandbox's TCP
table contained nothing but its own self-pipe — no connection to either
server port, consistent with §5 C1's measured claim that the bridge pipe is
the only channel and cross-process loopback is denied.
