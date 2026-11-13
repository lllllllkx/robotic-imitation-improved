# Reproduce the full comparison matrix (baseline vs improvements) sequentially.
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\run_protocol.ps1
# Requires the conda env from docs/REPRODUCTION.md to be active.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

foreach ($motion in @("dog_pace", "dog_spin")) {
    Write-Host "== $motion baseline (paper protocol: 3 seeds x 3 episodes)"

    python -m src.evaluate --motion $motion --episodes 3 --seeds 3 `
        --random-phase --out results\eval --tag $motion

    Write-Host "== $motion + yaw anchor"

    python -m src.evaluate --motion $motion --episodes 3 --seeds 3 `
        --random-phase --yaw-anchor --out results\eval --tag $motion
}

Write-Host "== aggregating"
python -m src.aggregate
