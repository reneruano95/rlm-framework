$ErrorActionPreference = 'Stop'
Set-Location D:\PROJECTS\rlm-halo-framework
foreach ($c in @(@{n='conc8-full'; e=''}, @{n='conc8-full-cram0'; e='--cache-ram 0'})) {
  Write-Host ('==== ' + $c.n + ' ====')
  uv run --python 3.12 s2/run_occupancy.py --condition $c.n --extra="$($c.e)" --np 128 --ctx 327680 --calls 128 --concurrency 8 --log ('traces/logs/occ-' + $c.n + '.log')
}
Write-Host '==== conc runs complete ===='
