#!/bin/bash
. "$HOME/.spike_env"
D=/home/spike/runs/gate/pc-02/on/sessions
echo "########## agg-07 ON rep1 — what the model did with the rule in front of it"
bash /opt/spike-scripts/profile_ep.sh "$D" "pc-02/on/agg-07/rep1"
echo
echo "########## the last two cells and the final text"
bash /opt/spike-scripts/dump_session.sh "$D" "pc-02/on/agg-07/rep1" 2>/dev/null | tail -50
