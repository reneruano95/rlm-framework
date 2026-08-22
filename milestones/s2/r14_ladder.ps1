# milestones/s2/R14.md -- the two-arm concurrency ladder.
#
# ARM A reproduces the defect under the ORIGINAL flags (llama-server b10375's
# host prompt cache at its default: --cache-idle-slots on, --cache-ram 8192).
# ARM B is the decisive arm: identical in every other respect, host prompt
# cache off by BOTH switches.
#
# Same fixtures, same question, same pinned leaf prefix, same sampling, same
# never-reused slot per call as the run that found R14 -- this is
# `milestones/s2/run_occupancy.py`, the runner the occupancy controls produced it with,
# not a new harness, so the comparison is like-for-like.
#
# n follows the original: 32 calls at concurrency 1/2/4, 128 at concurrency 8.
#
# `--extra=` and never `--extra `: argparse reads a following token that starts
# with `--` as an option rather than as this option's value.
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

$Out = "milestones/s2/results/r14.jsonl"
$CacheOff = "--cache-ram 0 --no-cache-idle-slots"

function Run-Arm {
  param([string]$Name, [string]$Extra, [int]$Conc, [int]$Calls)
  Write-Host "==== $Name (conc $Conc, n $Calls, extra='$Extra') ===="
  uv run --python 3.12 milestones/s2/run_occupancy.py --condition $Name --extra="$Extra" `
    --np 128 --ctx 327680 --calls $Calls --concurrency $Conc `
    --out $Out --log "traces/logs/r14-$Name.log"
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

foreach ($k in @(1, 2, 4)) { Run-Arm -Name "r14-A-conc$k" -Extra ""        -Conc $k -Calls 32 }
Run-Arm -Name "r14-A-conc8" -Extra ""        -Conc 8 -Calls 128
foreach ($k in @(1, 2, 4)) { Run-Arm -Name "r14-B-conc$k" -Extra $CacheOff -Conc $k -Calls 32 }
Run-Arm -Name "r14-B-conc8" -Extra $CacheOff -Conc 8 -Calls 128

Write-Host "==== R14 two-arm ladder complete ===="
