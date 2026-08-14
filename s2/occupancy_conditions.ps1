# s2/OCCUPANCY.md's condition table, one server launch per condition.
# ONE factor moves per line; the workload (128 synthetic same-length documents,
# the pinned leaf prefix, the same question, the same sampling) is identical in
# every one. Every server this starts is shut down by the runner itself.
#
# `--extra=$Extra`, never `--extra $Extra`: argparse treats a following token
# that begins with `--` and contains no space as an option rather than as this
# option's value, so `--extra "--no-cache-idle-slots"` fails outright while
# `--extra "--cache-ram 0"` (which contains a space) silently does not.
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

function Run-Cond {
  param([string]$Name, [string]$Extra = "", [int]$Np = 128, [int]$Ctx = 327680,
        [int]$Calls = 128, [string]$Order = "asc", [int]$Conc = 1,
        [int]$Keepalive = 20, [int]$MaxConn = 100, [int]$ChunkTokens = 1024)
  Write-Host "==== $Name ===="
  uv run --python 3.12 s2/run_occupancy.py --condition $Name --extra="$Extra" `
    --np $Np --ctx $Ctx --calls $Calls --order $Order --concurrency $Conc `
    --max-keepalive $Keepalive --max-connections $MaxConn `
    --chunk-tokens $ChunkTokens --log "traces/logs/occ-$Name.log"
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

# HYPOTHESIS 4/5 -- the host prompt cache. Two independent ways to switch it off.
Run-Cond -Name "cram0"       -Extra "--cache-ram 0"
Run-Cond -Name "nocacheidle" -Extra "--no-cache-idle-slots"

# HYPOTHESIS 1 -- slot scanning by longest-common-prefix.
Run-Cond -Name "sps0"        -Extra "-sps 0"

# SLOT INDEX vs CALL ORDINAL: shuffled slot order decorrelates the two.
Run-Cond -Name "shuffle"     -Order shuffle

# POOL SIZE vs SLOTS-IN-USE: 8 calls (8 slots in use) on three pool sizes.
Run-Cond -Name "np8"   -Np 8   -Ctx 20480  -Calls 8
Run-Cond -Name "np32"  -Np 32  -Ctx 81920  -Calls 8
Run-Cond -Name "np128" -Np 128 -Ctx 327680 -Calls 8

# HYPOTHESIS 2 -- client-side pooling. Same 32 calls, 8 in flight, at httpx's
# default keepalive (20, i.e. above the concurrency) and at 1 (below it).
Run-Cond -Name "conc8-keepalive20" -Calls 32 -Conc 8 -Keepalive 20
Run-Cond -Name "conc8-keepalive1"  -Calls 32 -Conc 8 -Keepalive 1

# THE DECISION: the same two conditions at the candidate window geometry, so
# the 417-window and 261-window projections rest on measured per-call walls
# rather than on scaled ones.
Run-Cond -Name "w640"       -ChunkTokens 640
Run-Cond -Name "cram0-w640" -Extra "--cache-ram 0" -ChunkTokens 640

Write-Host "==== all conditions complete ===="
