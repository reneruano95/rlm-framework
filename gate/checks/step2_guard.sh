#!/bin/bash
# Verify the identical-turn guard: the 3rd byte-identical ipython cell must be blocked,
# and a session that varies its cells must be untouched.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"
OUT="$HOME/runs/step2-guard"; rm -rf "$OUT"; mkdir -p "$OUT" "$HOME/tmp/guard"; cd "$HOME/tmp/guard"
: > "$RLMH_LEDGER"

echo "########## A — a forced identical loop must be blocked at the 3rd cell"
prime-agent -p --offline --thinking off --session-dir "$OUT/a" \
  --autonomous --autonomous-max-continuations 1 --autonomous-max-turns 8 \
  --autonomous-max-tokens 60000 --autonomous-timeout-ms 400000 \
  "Run this exact ipython cell four times in a row, unchanged, one per turn, without editing a single character: print('LOOPTEST')" 2>&1 | tail -6
echo "  --- blocks recorded:"
grep '"event":"identical_turn_blocked"' "$RLMH_LEDGER" || echo "  (none)"

echo
echo "########## B — a normal varying session must see no block"
: > "$RLMH_LEDGER"
cp "$HOME/tasks/codeqa-05/corpus.txt" .
prime-agent -p --offline --thinking off --session-dir "$OUT/b" \
  --autonomous --autonomous-gate "test -s $HOME/tmp/guard/answer.txt" \
  --autonomous-max-continuations 1 --autonomous-max-turns 25 \
  --autonomous-max-tokens 300000 --autonomous-timeout-ms 900000 \
  "$(cat "$HOME/prompts/codeqa-05.txt")" 2>&1 | tail -3
echo "  --- blocks recorded (expect none):"
grep -c '"event":"identical_turn_blocked"' "$RLMH_LEDGER" || echo 0
echo "  --- answer: [$(head -1 answer.txt 2>/dev/null || echo none)]"
