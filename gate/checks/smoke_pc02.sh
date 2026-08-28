#!/bin/bash
# One ON + one OFF episode with the new delivery block, before committing 54 to it.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"
ROOT="$HOME/runs/gate/pc02-smoke"; rm -rf "$ROOT"
echo "=== settle-wait + delivery smoke (codeqa-04)"
bash "$HOME/gate/run_episode.sh" codeqa-04 1 off "$HOME/gate/artifacts/positive-control.json" "$ROOT" < /dev/null
bash "$HOME/gate/run_episode.sh" codeqa-04 1 on  "$HOME/gate/artifacts/positive-control.json" "$ROOT" < /dev/null
echo
echo "settle: off=$(cat "$ROOT/off/codeqa-04/rep1/daemon_settle.txt" 2>&1) on=$(cat "$ROOT/on/codeqa-04/rep1/daemon_settle.txt" 2>&1)"
echo "voids : $(find "$ROOT" -name VOID | wc -l)"
echo "--- did the directive block reach the model? (gate_block_chars on the ON arm)"
grep -o '"gate_block_chars":[0-9]*' "$ROOT/on/codeqa-04/rep1/ledger.jsonl" | head -2
grep -o '"gate_block_chars":[0-9]*' "$ROOT/off/codeqa-04/rep1/ledger.jsonl" | head -2
