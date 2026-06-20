# tools/merge_daily.ps1
param(
  [string]$DailyDir = "logs",
  [string]$Pattern  = "live_meta_log_*.csv",
  [string]$Out      = "live_meta_master.csv",
  [switch]$IncludeMaster,
  [switch]$Dedupe
)

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)  # tools/
Set-Location ..

$py = ".\.venv\Scripts\python.exe"
$args = @("tools\merge_daily_logs.py", "--daily-dir", $DailyDir, "--pattern", $Pattern, "--out", $Out)
if ($IncludeMaster) { $args += "--include-master" }
if ($Dedupe)        { $args += "--dedupe" }

& $py @args
