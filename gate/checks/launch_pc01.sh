#!/bin/bash
# Verify the stale-list guard fires, then launch the positive-control decision.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"

echo "=== negative test: a stale task list must be refused"
printf 'codeqa-06\ncodeqa-05\n' > /tmp/stale.txt
bash "$HOME/gate/run_decision.sh" stale-test "$HOME/gate/artifacts/positive-control.json" /tmp/stale.txt 1 2>&1 | head -5
echo "    exit=$?"

echo
echo "=== launching pc-01 with the correct list"
rm -rf "$HOME/runs/gate/pc-01"
nohup bash "$HOME/gate/run_decision.sh" pc-01 "$HOME/gate/artifacts/positive-control.json" > "$HOME/runs/pc-01.out" 2>&1 &
sleep 6
head -6 "$HOME/runs/pc-01.out"
