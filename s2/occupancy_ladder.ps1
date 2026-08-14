$ErrorActionPreference = 'Stop'
Set-Location D:\PROJECTS\rlm-halo-framework
foreach ($k in @(1,2,3,4)) {
  Write-Host ('==== conc' + $k + '-ladder ====')
  uv run --python 3.12 s2/run_occupancy.py --condition ('conc' + $k + '-ladder') --extra='--cache-ram 0' --np 128 --ctx 327680 --calls 32 --concurrency $k --log ('traces/logs/occ-conc' + $k + '-ladder.log')
}
Write-Host '==== ladder complete ===='
