#!/bin/bash
# Step 2 verification: the runner must reproduce the spike's Phase A numbers.
# Three tasks x 3 reps, OFF arm only (empty harness) = the spike's Phase A conditions.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"
mkdir -p "$HOME/gate"
printf 'codeqa-05\nneedle-05\nagg-07\n' > "$HOME/gate/verify.txt"
ROOT="$HOME/runs/gate/step2-verify"
rm -rf "$ROOT"; mkdir -p "$ROOT"
EMPTY="$HOME/gate/empty.json"
printf '%s\n' '{ "schema": 1, "entries": { "prompt": {}, "memory": {}, "skill": {}, "subagent": {} }, "refinements": [] }' > "$EMPTY"

echo "=== runner reproduction check: 3 tasks x 3 reps, empty harness"
for REP in 1 2 3; do
  while read -r T; do
    bash /home/spike/gate/run_episode.sh "$T" "$REP" "off" "$EMPTY" "$ROOT" < /dev/null
  done < "$HOME/gate/verify.txt"
done
echo
echo "=== episodes: $(find "$ROOT" -name wall.txt | wc -l)   void: $(find "$ROOT" -name VOID | wc -l)"
