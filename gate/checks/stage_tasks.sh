#!/bin/bash
# Runs as root: stage every v1 corpus and prompt the split needs, and write the
# split's task lists into the spike tree.
set -e
C=/mnt/d/PROJECTS/rlm-halo-framework/bench/corpora
P=/mnt/d/spike/prompts
SPLIT=/mnt/d/PROJECTS/rlm-halo-framework/bench/splits/s6lite-v0.json
PY=/home/spike/prime-spike/kernel-venv/bin/python

mkdir -p /home/spike/tasks /home/spike/prompts /home/spike/gate

# prompts, all 30
for f in "$P"/*.txt; do tr -d '\r' < "$f" > "/home/spike/prompts/$(basename "$f")"; done

# corpora: every task in the split (train + held-out)
"$PY" - "$SPLIT" > /tmp/split_tasks.txt <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for side in ("train", "held_out"):
    for r in d[side]:
        print(r["task_id"])
PY

while read -r T; do
  mkdir -p "/home/spike/tasks/$T"
  case "$T" in
    codeqa-*) SRC="$C/code-bundle.txt" ;;
    *)        SRC="$C/$T.txt" ;;
  esac
  [ -f "/home/spike/tasks/$T/corpus.txt" ] || cp "$SRC" "/home/spike/tasks/$T/corpus.txt"
done < /tmp/split_tasks.txt

# the split's task lists
"$PY" - "$SPLIT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for side, name in (("train", "train.txt"), ("held_out", "heldout.txt")):
    with open(f"/home/spike/gate/{name}", "w", newline="\n") as f:
        for r in d[side]:
            f.write(r["task_id"] + "\n")
    print(f"  {name}: {len(d[side])} tasks")
PY

cp "$SPLIT" /home/spike/gate/split.json
chown -R spike:spike /home/spike/tasks /home/spike/prompts /home/spike/gate
echo "corpora staged: $(ls /home/spike/tasks | wc -l) task dirs"
echo "missing corpora: $(while read -r T; do [ -s "/home/spike/tasks/$T/corpus.txt" ] || echo "$T"; done < /tmp/split_tasks.txt | tr '\n' ' ')"
echo "missing prompts: $(while read -r T; do [ -s "/home/spike/prompts/$T.txt" ] || echo "$T"; done < /tmp/split_tasks.txt | tr '\n' ' ')"
