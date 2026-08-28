#!/bin/bash
# Resume pc-03 after the server died mid-decision. Keeps the 22 completed episodes.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"

export RLMH_RESUME=1
export RLMH_SUBSET_REASON="spec 4.2b reliability design: agg-06 and agg-07 are the only held-out tasks with failure headroom (7 of 9 pass 3/3 in both arms in pc-01 and pc-02); 10 reps per arm, scored on discordant pairs, threshold fixed before the run. RESUMED 2026-08-28 after the llama-server died at 00:34 with 22 of 40 episodes complete; the completed episodes are kept and the remainder run on a fresh server."

nohup bash "$HOME/gate/run_decision.sh" pc-03 "$HOME/gate/artifacts/control-v2.json" \
  "$HOME/gate/reliability.txt" 10 >> "$HOME/runs/pc-03.out" 2>&1 &
sleep 8
tail -6 "$HOME/runs/pc-03.out"
