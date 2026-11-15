<#
.SYNOPSIS
  提交并推送本轮改动到 GitHub。

.DESCRIPTION
  每轮改动收尾的标准动作：git add -A -> 安全检查 -> commit -> push -> 校验远端 hash。
  安全检查拒绝三类东西进仓库：仓库根目录的临时文件（_ 开头）、logs/ 日志、单个超过 50 MB 的文件。

.PARAMETER Message
  提交信息（单段、单行用这个）。

.PARAMETER MessageFile
  UTF-8 文件里读提交信息；长段落、多行中文用这个，避免命令行引号炸掉。

.PARAMETER NoPush
  只提交不推送（用于想先本地看一眼 diff 的情况）。

.EXAMPLE
  .\scripts\push_changes.ps1 "fix: 跟踪指标改到与奖励同一时点"

.EXAMPLE
  .\scripts\push_changes.ps1 -MessageFile C:\tmp\msg.txt
#>
param(
  [Parameter(Mandatory = $true, ParameterSetName = "inline", Position = 0)][string]$Message,
  [Parameter(Mandatory = $true, ParameterSetName = "file")][string]$MessageFile,
  [string]$Remote = "origin",
  [string]$Branch = "main",
  [switch]$NoPush
)

if ($PSCmdlet.ParameterSetName -eq "file") {
  if (-not (Test-Path -LiteralPath $MessageFile)) { throw ("找不到提交信息文件: " + $MessageFile) }
  $Message = [IO.File]::ReadAllText($MessageFile, [Text.Encoding]::UTF8)
}

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

git add -A
if ($LASTEXITCODE -ne 0) { throw "git add failed" }

$staged = @(git diff --cached --name-only | Where-Object { $_ -ne "" })

if ($staged.Count -eq 0) {
  Write-Host "没有待提交的改动。"
} else {
  $bad = @()
  foreach ($f in $staged) {
    if ($f -match '^_') { $bad += ("{0}  (仓库根目录临时文件)" -f $f) }
    if ($f -match '^logs/') { $bad += ("{0}  (日志目录)" -f $f) }
    if ((Test-Path -LiteralPath $f) -and ((Get-Item -LiteralPath $f).Length -gt 50MB)) {
      $bad += ("{0}  (单文件超过 50 MB)" -f $f)
    }
  }
  if ($bad.Count -gt 0) {
    throw ("拒绝提交下列路径，请先加进 .gitignore 或移出仓库：`n  " + ($bad -join "`n  "))
  }

  $msgFile = Join-Path $env:TEMP ("commit-msg-" + [guid]::NewGuid().ToString("N") + ".txt")
  [IO.File]::WriteAllText($msgFile, $Message.TrimEnd() + "`n", (New-Object Text.UTF8Encoding $false))
  git commit -F $msgFile
  if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
  Remove-Item -LiteralPath $msgFile -Force
}

if ($NoPush) {
  Write-Host "已提交，未推送（-NoPush）。"
  return
}

git push $Remote $Branch
if ($LASTEXITCODE -ne 0) { throw "git push failed" }

$local = (git rev-parse HEAD).Trim()
$remoteLine = (git ls-remote $Remote ("refs/heads/" + $Branch))
$remoteHead = ($remoteLine -split "\s+")[0]
if ($local -ne $remoteHead) {
  throw ("推送未落地：本地 {0} / 远端 {1}" -f $local, $remoteHead)
}
Write-Host ("已推送 {0} -> {1}/{2}" -f $local, $Remote, $Branch)
