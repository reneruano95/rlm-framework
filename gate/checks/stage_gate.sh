#!/bin/bash
# Runs as root: stage the gate's runtime scripts, its extension, its checks, and the
# scratchpad drivers. The spike user cannot read /mnt/d by design, so every copy
# out of the repo happens here.
set -e
G=/mnt/d/PROJECTS/rlm-halo-framework/gate
SCRATCH=/mnt/c/Users/Rene/AppData/Local/Temp/claude/D--PROJECTS-rlm-halo-framework/932c5923-0e9f-4e21-8c8b-8a2729a63343/scratchpad

mkdir -p /home/spike/gate /home/spike/gate/checks /opt/spike-scripts /home/spike/prime-spike/extensions

for f in "$G"/*.sh; do
  [ -f "$f" ] && tr -d '\r' < "$f" > "/home/spike/gate/$(basename "$f")"
done

for f in "$G"/checks/*; do
  [ -f "$f" ] && tr -d '\r' < "$f" > "/home/spike/gate/checks/$(basename "$f")"
done

tr -d '\r' < "$G/extension/rlmh-gate.ts" > /home/spike/prime-spike/extensions/rlmh-gate.ts

for f in "$SCRATCH"/*.sh; do
  [ -f "$f" ] && tr -d '\r' < "$f" > "/opt/spike-scripts/$(basename "$f")"
done

chown -R spike:spike /home/spike/gate /home/spike/prime-spike/extensions
chmod 755 /opt/spike-scripts/*.sh /home/spike/gate/*.sh 2>/dev/null || true
echo "gate/:        $(ls /home/spike/gate | tr '\n' ' ')"
echo "gate/checks/: $(ls /home/spike/gate/checks | tr '\n' ' ')"
