#!/bin/bash
set -u
. "$HOME/.spike_env"
node "$HOME/gate/checks/test_gate_unit.mjs" "$HOME/prime-spike/extensions/rlmh-gate.ts"
