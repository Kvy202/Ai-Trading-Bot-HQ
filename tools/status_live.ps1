# tools/status_live.ps1
# Robust live status viewer — detects writer/executor even if CommandLine missing
# Safe, null-tolerant, and uses approved verbs for compliance
$ErrorActionPreference = 'SilentlyContinue'

# ---------- Paths ----------
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root      = (Resolve-Path (Join-Path $scriptDir '..')).Path
$logsDir   = Join-Path $root 'logs'

$writerOut = Join-Path $logsDir 'live_writer.out'
$execOut   = Join-Path $logsDir 'live_executor.out'
$writerErr = Join-Path $logsDir 'live_writer.err'
$execErr   = Join-Path $logsDir 'live_executor.err'

$signals   = Join-Path $logsDir 'live_signals.csv'
$paper     = Join-Path $logsDir ("trades_paper_{0}.csv"  -f (Get-Date -Format yyyyMMdd))
$closed    = Join-Path $logsDir ("trades_closed_{0}.csv" -f (Get-Date -Format yyyyMMdd))
$heartbeat = Join-Path $logsDir 'heartbeat.json'
$stateJson = Join-Path $logsDir 'executor_state.json'
$lockWriter   = Join-Path $logsDir 'live_writer.lock'
$lockExecutor = Join-Path $logsDir 'live_executor.lock'

# ---------- Helpers ----------
function ConvertTo-RegexLiteral {
  param([string]$Text)
  return [regex]::Escape($Text)
}

function Test-ProcessArg {
  param($Process, [string]$Pattern)
  try {
    $cl = $Process.CommandLine
    if ([string]::IsNullOrWhiteSpace($cl)) { return $false }
    return ($cl -match $Pattern)
  } catch { return $false }
}

function Get-ProcsByPattern {
  param([string]$Pattern)
  $rootRx = ConvertTo-RegexLiteral $root
  $all = Get-CimInstance Win32_Process
  $live = $all | Where-Object { (Test-ProcessArg $_ $Pattern) -and (Test-ProcessArg $_ $rootRx) }
  if (-not $live) {
    $live = $all | Where-Object {
      $_.Name -match '^python(?:\.exe)?$' -and `
      (Test-ProcessArg $_ $rootRx -or ($_.ExecutablePath -like (Join-Path $root '.venv\Scripts\python.exe')))
    }
  }
  return ($live | Select-Object ProcessId, CreationDate, CommandLine)
}

function Get-ProcFromLock {
  param([string]$LockPath)
  if (-not (Test-Path $LockPath)) { return $null }
  try {
    $txt = (Get-Content $LockPath -Raw).Trim()
    if (-not $txt) { return $null }
    $pid = [int]($txt.Split(',')[0])
    $p = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($p) { return [PSCustomObject]@{ ProcessId=$p.Id; StartTime=$p.StartTime; Name=$p.Name; FromLock=$true } }
  } catch { return $null }
  return $null
}

# ---------- Display ----------
Write-Host "=== LIVE PROCESSES ===" -ForegroundColor Cyan
$rxLive = 'tools(\\|/)live_(writer|executor)\.py'
$rxDash = 'tools(\\|/)dashboard\.py'

$live = Get-ProcsByPattern $rxLive
$dash = Get-ProcsByPattern $rxDash

if ($live) {
  $live | Format-Table -AutoSize
} else {
  $w = Get-ProcFromLock $lockWriter
  $e = Get-ProcFromLock $lockExecutor
  if ($w -or $e) {
    Write-Host "(from lock files)"
    @($w,$e) | Where-Object {$_} | Format-Table -AutoSize
  } else {
    Write-Host "(no writer/executor)"
  }
}

if ($dash) {
  Write-Host "`nDashboard:"; $dash | Format-Table -AutoSize
}

# ---------- Health ----------
Write-Host "`n=== HEALTH ===" -ForegroundColor Cyan
if (Test-Path $heartbeat) {
  try {
    $hb = Get-Content $heartbeat -Raw | ConvertFrom-Json
    $ts = Get-Date $hb.ts
    $age = [int]((Get-Date) - $ts).TotalSeconds
    Write-Host ("heartbeat: {0} (age {1}s) component={2}" -f $ts.ToString("u"), $age, $hb.component)
  } catch { Write-Host "heartbeat: (unreadable)" }
}
if (Test-Path $stateJson) {
  try {
    $st = Get-Content $stateJson -Raw | ConvertFrom-Json
    Write-Host ("gate: thr={0} mode={1} adaptive={2} target_pass={3} window={4}" -f `
      $st.exec_thr, $st.exec_mode, $st.adaptive, $st.target_pass, $st.window)
  } catch { Write-Host "gate: (unreadable)" }
}

# ---------- Log tails ----------
Write-Host "`n=== LOG TAILS ===" -ForegroundColor Cyan
if (Test-Path $writerOut) { Write-Host "`n[writer.out]"; Get-Content $writerOut -Tail 15 }
if (Test-Path $writerErr) { Write-Host "`n[writer.err]"; Get-Content $writerErr -Tail 10 }
if (Test-Path $execOut)   { Write-Host "`n[executor.out]"; Get-Content $execOut -Tail 25 }
if (Test-Path $execErr)   { Write-Host "`n[executor.err]"; Get-Content $execErr -Tail 10 }

# ---------- Data tails ----------
Write-Host "`n=== SIGNALS (last 8) ===" -ForegroundColor Cyan
if (Test-Path $signals) { Get-Content $signals -Tail 8 } else { Write-Host "(no signals yet)" }

Write-Host "`n=== PAPER TRADES (today) ===" -ForegroundColor Cyan
if (Test-Path $paper) { Get-Content $paper -Tail 12 } else { Write-Host "(no paper trades today)" }

Write-Host "`n=== CLOSED TRADES (today) ===" -ForegroundColor Cyan
if (Test-Path $closed) { Get-Content $closed -Tail 12 } else { Write-Host "(no closed trades today)" }
