# milestones/s2/R14.md -- characterising the defect once the host-cache arm comes back
# negative. Every condition here holds the host prompt cache OFF (the shipped
# `config.yaml` setting, and arm B's setting), so each line moves exactly ONE
# factor against `r14-B-conc8` / `r14-B-conc2` as its control.
#
# H1  continuous batching  -- `--no-cont-batching`. If quality returns, the
#     defect is in batched decode and fan-out is a real, priceable trade.
# H4  sampling             -- temperature 0 (greedy). Degenerate text is often a
#     sampling pathology and this is the cheapest hypothesis on the list.
# H3  batch geometry       -- `-ub`/`-b` at fixed concurrency, both directions.
# H5  streamed abort       -- `--drain-stream` reads the SSE stream to its
#     natural end instead of breaking on the final event and closing the
#     connection. `milestones/s2/OCCUPANCY.md` §7 names this as one of the two candidates,
#     and `rlm/dispatcher.py` uses the same break-on-stop pattern, so a
#     positive here would be a production bug and not just a harness artefact.
$ErrorActionPreference = "Stop"
Set-Location D:\PROJECTS\rlm-halo-framework

$Out = "milestones/s2/results/r14.jsonl"
$CacheOff = "--cache-ram 0 --no-cache-idle-slots"

# NOT `-Args`: `$Args` is a PowerShell AUTOMATIC variable (the unbound-argument
# array), so a parameter of that name is silently ignored and binds to nothing.
# The first attempt at this battery used it and every condition launched with
# the DEFAULT flags -- a duplicate of the control wearing the treatment's name,
# which is the worst possible failure for a one-factor design. Caught by reading
# back the launch line the runner echoes. Hence `-Flags`, and hence the
# post-condition assertion below.
function Run-H {
  param([string]$Name, [int]$Conc, [int]$Calls, [string[]]$Flags = @(),
        [string]$MustContain = "")
  Write-Host "==== $Name (conc $Conc, n $Calls, flags: $($Flags -join ' ')) ===="
  $argv = @("run", "--python", "3.12", "milestones/s2/run_occupancy.py",
            "--condition", $Name, "--extra=$CacheOff",
            "--np", "128", "--ctx", "327680", "--calls", "$Calls",
            "--concurrency", "$Conc", "--out", $Out,
            "--log", "traces/logs/r14-$Name.log") + $Flags
  & uv @argv
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
  # The treatment must be visible in the RECORDED metadata, not merely intended.
  if ($MustContain) {
    $last = Get-Content $Out -Tail 1
    if ($last -notmatch [regex]::Escape($MustContain)) {
      throw "$Name did not record '$MustContain' -- the flag did not take effect"
    }
  }
}

# Ordered most-decisive-first, so a truncated battery still answers the biggest
# question. H1 leads because the ladder's own evidence points at decode: the
# concurrent failures include the TRUE identifier with tokens missing from the
# middle of it, which is a decode that lost tokens, not a model that guessed.

# H1 -- continuous batching, at the shipped value (8) and at the break point (2).
Run-H -Name "r14-H1-nocb-conc8" -Conc 8 -Calls 128 -Flags @("--no-cont-batching") -MustContain '"cont_batching": false'
Run-H -Name "r14-H1-nocb-conc2" -Conc 2 -Calls 32  -Flags @("--no-cont-batching") -MustContain '"cont_batching": false'

# H4 -- sampling. One flag, and it tests the commonest explanation for word salad.
Run-H -Name "r14-H4-temp0-conc8" -Conc 8 -Calls 128 -Flags @("--temperature", "0") -MustContain '"temperature": 0.0'

# H5 -- the streamed abort, at the shipped value and at the break point.
Run-H -Name "r14-H5-drain-conc8" -Conc 8 -Calls 128 -Flags @("--drain-stream") -MustContain '"drain_stream": true'
Run-H -Name "r14-H5-drain-conc2" -Conc 2 -Calls 32  -Flags @("--drain-stream") -MustContain '"drain_stream": true'

# H3 -- batch geometry, both directions, at fixed concurrency 8.
Run-H -Name "r14-H3-ub2048-conc8" -Conc 8 -Calls 128 -Flags @("--ub", "2048", "--batch", "2048") -MustContain '"ub": 2048'
Run-H -Name "r14-H3-ub128-conc8"  -Conc 8 -Calls 128 -Flags @("--ub", "128",  "--batch", "512") -MustContain '"ub": 128'

Write-Host "==== R14 hypothesis battery complete ===="
