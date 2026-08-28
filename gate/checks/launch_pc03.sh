#!/bin/bash
# pc-03: the reliability sub-experiment (spec §4.2b), artifact v2, on the two held-out
# tasks that actually have failure headroom. Declared as a subset, with the reason
# recorded alongside the decision.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"

printf 'agg-06\nagg-07\n' > "$HOME/gate/reliability.txt"

export RLMH_SUBSET_REASON="spec 4.2b reliability design: agg-06 and agg-07 are the only held-out tasks with failure headroom (7 of 9 pass 3/3 in both arms in pc-01 and pc-02); 10 reps per arm, scored on discordant pairs, threshold fixed before the run"

rm -rf "$HOME/runs/gate/pc-03"
nohup bash "$HOME/gate/run_decision.sh" pc-03 "$HOME/gate/artifacts/control-v2.json" \
  "$HOME/gate/reliability.txt" 10 > "$HOME/runs/pc-03.out" 2>&1 &
sleep 6
head -8 "$HOME/runs/pc-03.out"
