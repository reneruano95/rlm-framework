#!/bin/bash
echo "=== is the decision loop alive?"
pgrep -fa 'run_decision.sh' | head -3
echo "=== is an episode running?"
pgrep -fa 'run_episode.sh' | head -3
echo "=== prime-agent processes:"
pgrep -fa 'prime-agent|ipykernel' | head -5
echo "=== newest run dir + its files:"
D=$(ls -dt /home/spike/runs/gate/pc-03/*/*/rep* 2>/dev/null | head -1)
echo "  $D"
ls -la "$D" 2>/dev/null | head -12
echo "=== mtime of pc-03.out:"
stat -c '%y  %n' /home/spike/runs/pc-03.out
date
echo "=== server health from inside WSL:"
curl -s -m 5 http://127.0.0.1:8080/health || echo "  NO RESPONSE"
echo
echo "=== last stderr of the newest episode:"
tail -5 "$D/stderr.txt" 2>/dev/null
