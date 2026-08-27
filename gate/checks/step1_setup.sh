#!/bin/bash
# Step 1 setup: install the gate extension and reset settings to the v0 baseline.
set -e
SRC=/mnt/d/PROJECTS/rlm-halo-framework/gate/extension/rlmh-gate.ts
D=/home/spike/prime-spike

mkdir -p "$D/extensions" /home/spike/gate
tr -d '\r' < "$SRC" > "$D/extensions/rlmh-gate.ts"

cat > /home/spike/.gate_env <<'EOF'
export RLMH_ACCEPTED=$HOME/gate/accepted.json
export RLMH_LEDGER=$HOME/gate/ledger.jsonl
export RLMH_MARKER=
EOF
grep -q gate_env /home/spike/.bashrc || echo '. $HOME/.gate_env' >> /home/spike/.bashrc

# accepted set: empty to start (nothing is accepted yet)
cat > /home/spike/gate/accepted.json <<'EOF'
{ "schema": 1, "entries": { "prompt": {}, "memory": {}, "skill": {}, "subagent": {} }, "refinements": [] }
EOF
: > /home/spike/gate/ledger.jsonl

# settings back to the v0 baseline (Phase C left rlmMaxDepth at 2)
/home/spike/prime-spike/kernel-venv/bin/python - "$D/settings.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d.update({"rlmMaxDepth": 1, "defaultThinkingLevel": "off", "idleEvictionMinutes": "off"})
d["autoRefine"] = {"enabled": False}
d.setdefault("retry", {})["maxRetries"] = 3
d["retry"]["provider"] = {"timeoutMs": 3600000}
json.dump(d, open(p, "w"), indent=2)
print(json.dumps(d, indent=2))
PY

# clear any harness left from the spike so step 1 starts from nothing
rm -rf "$D/harness"
chown -R spike:spike "$D" /home/spike/gate /home/spike/.gate_env
echo "=== extension installed:"; ls -la "$D/extensions"
