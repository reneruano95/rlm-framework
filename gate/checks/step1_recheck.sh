#!/bin/bash
# Re-run check 2 after the count-line fix, and add check 4: an ACCEPTED entry must survive.
. "$HOME/.spike_env"; . "$HOME/.gate_env"
G="$PRIME_AGENT_CODING_AGENT_DIR/harness/harness_state.json"
OUT="$HOME/runs/step1b"; rm -rf "$OUT"; mkdir -p "$OUT" "$HOME/tmp/s1b"; cd "$HOME/tmp/s1b"
PY="$PRIME_AGENT_KERNEL_VENV/bin/python"
: > "$RLMH_LEDGER"

echo "########## CHECK 2b — smuggled entry stripped AND its count line corrected"
prime-agent -p --offline --thinking off --session-dir "$OUT/c2b" \
  "Under the heading 'Continual Harness State' in your system prompt there are per-kind count lines. Report them verbatim, one per line, and nothing else." 2>&1 | tail -8
echo "  --- watchword leak?"
grep -qi 'ZEBRAQUARTZ\|SMUGGLED-ZEBRA' "$OUT"/c2b/*.jsonl 2>/dev/null && echo "  !! LEAKED" || echo "  OK — absent"

echo
echo "########## CHECK 4 — an ACCEPTED entry must reach the model (the filter is not a wall)"
"$PY" - "$RLMH_ACCEPTED" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["entries"]["prompt"]["rlmh-probe-canary"] = {
    "id": "rlmh-probe-canary", "kind": "prompt",
    "title": "Step 1 canary",
    "content": "When asked for the canary word, reply exactly LIGHTHOUSE-SEVEN.",
    "path": "00-gate/prompt/00", "scope": "global",
    "reference": {}, "arguments": {}, "metadata": {"rlmh": {"step": 1}},
    "source": "rlmh-gate", "created_at": "2026-08-27T00:00:00Z",
    "updated_at": "2026-08-27T00:00:00Z", "version": 1,
}
json.dump(d, open(p, "w"), indent=2)
print("  accepted set now holds:", sorted(k for kind in d["entries"].values() for k in kind))
PY
# the same entry must also exist in prime-agent's store, since the filter only removes
"$PY" - "$G" "$RLMH_ACCEPTED" <<'PY'
import json, sys
g, a = sys.argv[1], sys.argv[2]
gs, acc = json.load(open(g)), json.load(open(a))
gs["entries"]["prompt"]["rlmh-probe-canary"] = acc["entries"]["prompt"]["rlmh-probe-canary"]
json.dump(gs, open(g, "w"), indent=2)
print("  store now holds:", sorted(k for kind in gs["entries"].values() for k in kind))
PY
prime-agent -p --offline --thinking off --session-dir "$OUT/c4" \
  "What is the canary word? Reply with just the word." 2>&1 | tail -3
echo "  --- canary present, smuggled still absent?"
grep -qi 'LIGHTHOUSE-SEVEN' "$OUT"/c4/*.jsonl 2>/dev/null && echo "  OK — accepted entry reached the model" || echo "  !! accepted entry did NOT reach the model"
grep -qi 'ZEBRAQUARTZ' "$OUT"/c4/*.jsonl 2>/dev/null && echo "  !! smuggled entry leaked" || echo "  OK — smuggled entry still stripped"

echo
echo "########## LEDGER"
grep -E '"event":"(stripped_entries|prompt_filtered)"' "$RLMH_LEDGER" | tail -4
