#!/bin/bash
# One gate decision: the held-out split, both arms, blocked with ON and OFF adjacent.
#
#   usage: run_decision.sh <decision-id> <accepted.json> [task-list-file] [reps]
#
# Scheduling follows ARCHITECTURE.md §8: "runs execute in (task, seed) blocks
# adjacent in time across all arms, so R9 thermal drift cancels within each paired
# comparison." Here the block is (task, rep) and the two arms are ON and OFF, run
# back to back. Arm order alternates by rep so a systematic within-block ordering
# effect cannot masquerade as an artifact effect.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"

DEC="$1"; ACCEPTED="$2"
LIST="${3:-$HOME/gate/heldout.txt}"
REPS="${4:-3}"
ROOT="$HOME/runs/gate/$DEC"
# RESUME. Set RLMH_RESUME=1 to keep completed episodes and run only what is missing.
# Needed the hard way 2026-08-28: the llama-server died mid-decision (pc-03 at 22/40,
# every WSL process gone with it, no crash line in the server log -- consistent with
# the host sleeping rather than with upstream #23181) and a plain re-run would have
# discarded 22 good episodes. An episode counts as done when it has BOTH wall.txt and
# a non-empty answer file or a recorded exit -- a half-written directory is redone.
if [ "${RLMH_RESUME:-0}" = "1" ] && [ -d "$ROOT" ]; then
  echo "RESUMING $DEC: $(find "$ROOT" -name wall.txt | wc -l) episodes already on disk"
else
  rm -rf "$ROOT"
fi
mkdir -p "$ROOT"
cp "$ACCEPTED" "$ROOT/candidate.json" 2>/dev/null || echo "(no candidate file: OFF-only run)"
sha256sum "$ACCEPTED" 2>/dev/null | cut -d' ' -f1 > "$ROOT/candidate.sha256" || true

# The task list must match the CURRENT split. Found the hard way 2026-08-27: a list
# staged before the split was redrawn still named codeqa-06, which the redraw moved to
# train -- so a decision would have evaluated on a training task and nothing would have
# said so. The split file is the source of truth; the list is a cache of it.
SPLIT=/home/spike/gate/split.json
if [ -f "$SPLIT" ]; then
  EXPECT=$("$PRIME_AGENT_KERNEL_VENV/bin/python" -c "
import json,sys
d=json.load(open('$SPLIT'))
print(' '.join(r['task_id'] for r in d['held_out']))
")
  GOT=$(tr '
' ' ' < "$LIST" | sed 's/ *$//')
  if [ "$EXPECT" != "$GOT" ]; then
    # A SUBSET is allowed only when the caller names the design in
    # RLMH_SUBSET_REASON, and the reason is recorded with the decision. A focused
    # sub-experiment is legitimate -- 7 of the 9 held-out tasks pass 3/3 in both arms
    # every time, so on a reliability question they contribute noise and no signal --
    # but it must be a declared design, never a quiet swap. Anything that is not a
    # subset, or a subset with no stated reason, is still refused.
    SUBSET_OK=1
    for T in $GOT; do
      case " $EXPECT " in *" $T "*) ;; *) SUBSET_OK=0 ;; esac
    done
    if [ "$SUBSET_OK" = "1" ] && [ -n "${RLMH_SUBSET_REASON:-}" ]; then
      echo "SUBSET of the held-out side, declared: $RLMH_SUBSET_REASON" | tee -a "$ROOT/subset.txt"
    else
      echo "REFUSING: task list does not match the split's held_out side." >&2
      echo "  split: $EXPECT" >&2
      echo "  list : $GOT" >&2
      [ "$SUBSET_OK" = "1" ] && echo "  (it IS a subset -- set RLMH_SUBSET_REASON to declare the design)" >&2
      exit 2
    fi
  fi
fi

LOG="$ROOT/decision.log"
echo "=== decision $DEC start $(date -Is)" | tee -a "$LOG"
echo "    candidate: $ACCEPTED  sha=$(cat "$ROOT/candidate.sha256" 2>/dev/null | cut -c1-16)" | tee -a "$LOG"
echo "    tasks:     $(tr '\n' ' ' < "$LIST")" | tee -a "$LOG"
echo "    reps:      $REPS" | tee -a "$LOG"

for REP in $(seq 1 "$REPS"); do
  while read -r T; do
    [ -z "$T" ] && continue
    # alternate which arm leads, by rep
    if [ $((REP % 2)) -eq 1 ]; then ORDER="off on"; else ORDER="on off"; fi
    for ARM in $ORDER; do
      W="$ROOT/$ARM/$T/rep$REP"
      if [ "${RLMH_RESUME:-0}" = "1" ] && [ -s "$W/wall.txt" ] && [ -s "$W/exit.txt" ]; then
        echo "[$T $ARM rep$REP] SKIP (done: $(head -1 "$W/answer.txt" 2>/dev/null || echo '<no answer>'))" | tee -a "$LOG"
        continue
      fi
      # `< /dev/null` matters: without it the episode inherits the while-loop's stdin
      # and swallows the rest of the task list, so a decision silently runs only its
      # first task. Measured 2026-08-27 while verifying step 2.
      bash /home/spike/gate/run_episode.sh "$T" "$REP" "$ARM" "$ACCEPTED" "$ROOT" < /dev/null 2>&1 | tee -a "$LOG"
    done
  done < "$LIST"
done

echo "=== decision $DEC done $(date -Is)" | tee -a "$LOG"
echo "    episodes: $(find "$ROOT" -name wall.txt | wc -l)   void: $(find "$ROOT" -name VOID | wc -l)" | tee -a "$LOG"
