# milestones/s2/CACHE-INSTRUMENT.md's condition table, one server launch per condition.
# The workload is byte-identical in all three; only the host-prompt-cache flags
# move. Every server this starts is shut down by the runner itself.
#
# `--extra=$Extra`, never `--extra $Extra` (see milestones/s2/occupancy_conditions.ps1).
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

function Run-Cond {
  param([string]$Name, [string]$Extra = "")
  Write-Host "==== $Name ===="
  uv run --python 3.12 milestones/s2/run_cache_instrument.py --condition $Name --extra="$Extra" `
    --reps 3 --diverge-reps 1 --log "traces/logs/cacheinst-$Name.log"
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

# The shipped launch line: host prompt cache ON at its 8192 MiB default.
Run-Cond -Name "default"
# Two independent ways to switch the host cache off (milestones/s2/OCCUPANCY.md).
Run-Cond -Name "cram0"       -Extra "--cache-ram 0"
Run-Cond -Name "nocacheidle" -Extra "--no-cache-idle-slots"

Write-Host "==== all conditions complete ===="
