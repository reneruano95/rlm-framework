#!/bin/bash
# One gate episode.  usage: run_episode.sh <task> <rep> <arm:on|off> <accepted.json|none> <run-root>
#
# Installs the arm's artifact set FIRST -- into both the gate's accepted file (which
# the extension filters against) and prime-agent's global store (which it renders
# from) -- then runs one episode and records the harness sha256 the episode actually
# saw. Spec §5 (4): a held-out run whose harness sha does not match the accepted one
# is void, so the sha is captured per episode rather than per block.
#
# ON THE WORD "REP", NOT "SEED". prime-agent exposes no seed pin, so the repetition
# index is NOT the scaffold's pinned seed and does not make a cell reproducible. It
# is a repetition under the same conditions. §8's blocked (task, seed) design is
# reproduced in its scheduling -- ON and OFF adjacent in time so thermal drift
# cancels within the pair -- but not in its determinism, and no report may call
# these seeds.
set -u
. "$HOME/.spike_env"; . "$HOME/.gate_env"

T="$1"; REP="$2"; ARM="$3"; ACCEPTED_SRC="$4"; ROOT="$5"
PY="$PRIME_AGENT_KERNEL_VENV/bin/python"
STORE="$PRIME_AGENT_CODING_AGENT_DIR/harness/harness_state.json"
EMPTY='{ "schema": 1, "entries": { "prompt": {}, "memory": {}, "skill": {}, "subagent": {} }, "refinements": [] }'

W="$ROOT/$ARM/$T/rep$REP"
rm -rf "$W"; mkdir -p "$W" "$ROOT/$ARM/sessions" "$(dirname "$STORE")"
cp "$HOME/tasks/$T/corpus.txt" "$W/corpus.txt"
cp "$HOME/prompts/$T.txt" "$W/prompt.txt"
printf '%s\n' "$T" > "$W/task.txt"
printf '%s\n' "$ARM" > "$W/arm.txt"
printf '%s\n' "$REP" > "$W/rep.txt"

# --- install the arm's artifact set -------------------------------------------
if [ "$ARM" = "on" ] && [ "$ACCEPTED_SRC" != "none" ] && [ -f "$ACCEPTED_SRC" ]; then
  cp "$ACCEPTED_SRC" "$RLMH_ACCEPTED"
  cp "$ACCEPTED_SRC" "$STORE"
else
  printf '%s\n' "$EMPTY" > "$RLMH_ACCEPTED"
  printf '%s\n' "$EMPTY" > "$STORE"
fi
HSHA=$(sha256sum "$STORE" | cut -d' ' -f1)
ASHA=$(sha256sum "$RLMH_ACCEPTED" | cut -d' ' -f1)
printf '%s\n' "$HSHA" > "$W/harness.sha256"
printf '%s\n' "$ASHA" > "$W/accepted.sha256"

# --- run ----------------------------------------------------------------------
curl -s http://127.0.0.1:8080/metrics > "$W/metrics.pre"
LEDGER_BEFORE=$(wc -l < "$RLMH_LEDGER" 2>/dev/null || echo 0)
cd "$W"
S=$(date +%s.%N)
timeout 1500 prime-agent -p --offline --thinking off \
  --autonomous --autonomous-gate "test -s $W/answer.txt" \
  --autonomous-max-continuations 1 --autonomous-max-turns 25 \
  --autonomous-max-tokens 300000 --autonomous-timeout-ms 1300000 \
  --session-dir "$ROOT/$ARM/sessions" --cwd "$W" \
  "$(cat "$W/prompt.txt")" > "$W/stdout.txt" 2> "$W/stderr.txt"
RC=$?
printf '%s\n' "$RC" > "$W/exit.txt"
echo "$(date +%s.%N) - $S" | bc > "$W/wall.txt"
curl -s http://127.0.0.1:8080/metrics > "$W/metrics.post"

# --- the void check: did the episode run under the harness the arm intended? ---
HSHA_AFTER=$(sha256sum "$STORE" | cut -d' ' -f1)
if [ "$HSHA" != "$HSHA_AFTER" ]; then
  printf 'VOID harness changed during the episode: %s -> %s\n' "$HSHA" "$HSHA_AFTER" > "$W/VOID"
fi
# what the extension actually did for this episode
tail -n +$((LEDGER_BEFORE + 1)) "$RLMH_LEDGER" 2>/dev/null > "$W/ledger.jsonl"
KEPT=$("$PY" -c "
import json,sys
kept=stripped=None
for line in open('$W/ledger.jsonl',encoding='utf-8'):
    line=line.strip()
    if not line: continue
    o=json.loads(line)
    if o.get('event')=='prompt_filtered': kept,stripped=o.get('kept'),o.get('stripped')
print(f'{kept},{stripped}')
" 2>/dev/null || echo "?,?")

printf '[%s %s rep%s] exit=%s wall=%ss answer=[%s] harness=%s in_window=%s %s\n' \
  "$T" "$ARM" "$REP" "$RC" "$(cat "$W/wall.txt")" \
  "$(head -1 "$W/answer.txt" 2>/dev/null || echo '<none>')" \
  "${HSHA:0:12}" "$KEPT" "$([ -f "$W/VOID" ] && echo VOID || echo '')"
