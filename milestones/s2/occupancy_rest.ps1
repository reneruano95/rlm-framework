# The conditions the first pass lost to the `--extra` quoting bug, re-run with
# `--extra=`. Same runner, same workload, same shutdown discipline.
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

function Run-Cond {
  param([string]$Name, [string]$Extra = "", [int]$Np = 128, [int]$Ctx = 327680,
        [int]$Calls = 128, [string]$Order = "asc", [int]$Conc = 1,
        [int]$Keepalive = 20, [int]$MaxConn = 100, [int]$ChunkTokens = 1024,
        [int]$Requery = 0)
  Write-Host "==== $Name ===="
  uv run --python 3.12 milestones/s2/run_occupancy.py --condition $Name --extra="$Extra" `
    --np $Np --ctx $Ctx --calls $Calls --order $Order --concurrency $Conc `
    --max-keepalive $Keepalive --max-connections $MaxConn --requery $Requery `
    --chunk-tokens $ChunkTokens --log "traces/logs/occ-$Name.log"
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Run-Cond -Name "nocacheidle" -Extra "--no-cache-idle-slots"
Run-Cond -Name "shuffle"     -Order shuffle
Run-Cond -Name "w640"        -ChunkTokens 640
Run-Cond -Name "np8"   -Np 8   -Ctx 20480  -Calls 8
Run-Cond -Name "np32"  -Np 32  -Ctx 81920  -Calls 8
Run-Cond -Name "np128" -Np 128 -Ctx 327680 -Calls 8
Run-Cond -Name "conc8-keepalive20" -Calls 32 -Conc 8 -Keepalive 20
Run-Cond -Name "conc8-keepalive1"  -Calls 32 -Conc 8 -Keepalive 1
Run-Cond -Name "warm-default" -Calls 8 -Requery 8
Run-Cond -Name "warm-cram0"   -Extra "--cache-ram 0" -Calls 8 -Requery 8

Write-Host "==== all conditions complete ===="
