# The check the recommendation depends on: does switching the host prompt
# cache off also cost the intra-window re-query (§7 #3 (d), the one cache lever
# R13 leaves intact)? 8 cold windows on 8 virgin slots, then the SAME document
# re-asked on its own slot -- same-document reuse, measured clean under R13.
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

foreach ($c in @(@{n="warm-default"; e=""}, @{n="warm-cram0"; e="--cache-ram 0"})) {
  Write-Host "==== $($c.n) ===="
  uv run --python 3.12 s2/run_occupancy.py --condition $c.n --extra $c.e `
    --np 128 --ctx 327680 --calls 8 --requery 8 --log "traces/logs/occ-$($c.n).log"
}
Write-Host "==== warm check complete ===="
