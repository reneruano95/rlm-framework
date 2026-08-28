#!/bin/bash
. "$HOME/.spike_env"
echo "=== prime-agent list (cost + output)"
S=$(date +%s.%N); prime-agent list 2>&1 | head -10; echo "  took $(echo "$(date +%s.%N) - $S" | bc)s"
echo
echo "=== prime-agent status"
S=$(date +%s.%N); prime-agent status 2>&1 | head -10; echo "  took $(echo "$(date +%s.%N) - $S" | bc)s"
echo
echo "=== prime-agent agents --all"
S=$(date +%s.%N); prime-agent agents --all 2>&1 | head -10; echo "  took $(echo "$(date +%s.%N) - $S" | bc)s"
