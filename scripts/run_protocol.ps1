# Reproduce the full comparison matrix (baseline vs improvements).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_protocol.ps1 `
#       -RunId r20261114 `
#       -Python D:\anaconda\envs\motion_imitation_repro\python.exe
#
# -RunId     stamps every eval; the aggregate at the end filters on it, so
#            summaries left behind by earlier experiments cannot leak into the table.
# -Python    path to the interpreter of the env from docs/REPRODUCTION.md
#            (default "python", i.e. whatever is first on PATH).
#
# NOTE: $ErrorActionPreference stays "Continue" on purpose. TensorFlow prints
# FutureWarnings on stderr, and Windows PowerShell raises NativeCommandError for
# any stderr output from a native exe; with "Stop" that would abort the run even
# though python exited 0. Failures are detected via $LASTEXITCODE instead.

param(
    [string]$RunId = ("protocol-" + (Get-Date -Format "yyyyMMdd-HHmm")),
    [string]$Python = "python"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$seeds = 3
$eps = 3
$maxSteps = 600

function Run-Eval {
    param([string]$Out, [string]$Motion, [string]$Tag, [string[]]$Flags)
    $cliArgs = @("-m", "src.evaluate", "--motion", $Motion, "--tag", $Tag,
                 "--out", $Out, "--run-id", $RunId) + $Flags
    Write-Host ("== " + ($cliArgs -join " "))
    & $Python @cliArgs
    if ($LASTEXITCODE -ne 0) { throw "eval failed: $Motion $Tag" }
}

# 1) paired matrix: same seeds / same start phases, only the variant differs
foreach ($motion in @("dog_pace", "dog_spin")) {
    Run-Eval -Out "results\eval" -Motion $motion -Tag $motion -Flags @(
        "--episodes", "$eps", "--seeds", "$seeds", "--max-steps", "$maxSteps", "--random-phase")
    Run-Eval -Out "results\eval" -Motion $motion -Tag $motion -Flags @(
        "--episodes", "$eps", "--seeds", "$seeds", "--max-steps", "$maxSteps", "--random-phase", "--yaw-anchor")
    Run-Eval -Out "results\eval" -Motion $motion -Tag $motion -Flags @(
        "--episodes", "$eps", "--seeds", "$seeds", "--max-steps", "$maxSteps", "--random-phase", "--reward-v2")
    Run-Eval -Out "results\eval" -Motion $motion -Tag $motion -Flags @(
        "--episodes", "$eps", "--seeds", "$seeds", "--max-steps", "$maxSteps", "--random-phase", "--yaw-anchor", "--reward-v2")
}

# 2) deterministic phase grid (ablation over the start phase, one seed)
foreach ($motion in @("dog_pace", "dog_spin")) {
    Run-Eval -Out "results\eval" -Motion $motion -Tag $motion -Flags @(
        "--seeds", "1", "--episodes", "4", "--max-steps", "$maxSteps",
        "--start-phases", "0,0.25,0.5,0.75")
}

# 3) deterministic controls (single controlled episode, t=0 start)
Run-Eval -Out "results\det_dog_pace" -Motion "dog_pace" -Tag "dog_pace" -Flags @(
    "--seeds", "1", "--episodes", "1", "--max-steps", "$maxSteps", "--csv")
Run-Eval -Out "results\det_dog_spin" -Motion "dog_spin" -Tag "dog_spin" -Flags @(
    "--seeds", "1", "--episodes", "1", "--max-steps", "$maxSteps", "--csv")
Run-Eval -Out "results\det_spin_yaw" -Motion "dog_spin" -Tag "dog_spin" -Flags @(
    "--seeds", "1", "--episodes", "1", "--max-steps", "$maxSteps", "--yaw-anchor", "--csv")

# 4) one table for this run only
Write-Host "== aggregating run $RunId"
& $Python -m src.aggregate --run-id $RunId
if ($LASTEXITCODE -ne 0) { throw "aggregate failed" }
