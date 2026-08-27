#!/bin/bash
D=/home/spike/runs/gate/step2-verify/off/sessions
for r in rep1 rep2 rep3; do
  echo "--- agg-07 $r"
  bash /opt/spike-scripts/profile_ep.sh "$D" "agg-07/$r"
  echo
done
echo "--- codeqa-05 rep1 (the control that reproduced)"
bash /opt/spike-scripts/profile_ep.sh "$D" "codeqa-05/rep1"
