#!/bin/bash
# Runs as root: copy a decision's results out to D: for scoring on the host.
set -e
DEC="${1:?usage: collect_dec.sh <decision-id>}"
SRC="/home/spike/runs/gate/$DEC"
DST="/mnt/d/spike/out/$DEC"
rm -rf "$DST"; mkdir -p "$DST"
cd "$SRC"
find . -type f ! -name corpus.txt -exec cp --parents {} "$DST"/ \;
echo "episodes copied: $(find "$DST" -name wall.txt | wc -l)"
echo "voids:           $(find "$DST" -name VOID | wc -l)"
find "$DST" -name VOID -printf '  %p\n' 2>/dev/null || true
