param(
  [switch]$AlsoDashboard  # stop dashboard too
)

$ErrorActionPreference = 'Continue'

# ---------- Paths ----------
$scriptDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$root          = (Resolve-Path (Join-Path $scriptDir '..')).Path
$logsDir       = Join-Path $root 'logs'
$executorLock  = Join-Path $logsDir 'live_executor.lock'
$writerLock    = Join-Path $logsDir 'live_writer.lock'

# Compile-time regex targets (Windows-style / cross-sep)
$rxLive = 'tools(\\|/)live_(writer|executor)\.py'
$rxDash = 'tools(\\|/)dashboard\.py'

# ---------- Helpers ----------
function Escape-Regex([string]$s) { [regex]::Escape($s) }

function Get-ProcsByRegex([string]$regex, [string]$scopePath) {
  $scopeRx = Escape-Regex $scopePath
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and ($_.CommandLine -match $regex) -and ($_.CommandLine -match $scopeRx)
  }
}

function Stop-ByPid([int]$TargetPid) {
  try {
    $null = Get-Process -Id $TargetPid -ErrorAction Stop
  } catch { return $false }
  try {
    Stop-Process -Id $TargetPid -Force -ErrorAction Stop
    Write-Host "[stop_live] stopped PID=${TargetPid}"
    return $true
  } catch {
    Write-Host "[stop_live] could not stop PID=${TargetPid}: $($_.Exception.Message)"
    return $false
  }
}

# Kill a proc AND its children (best-effort). NOTE: parameter name != 'pid'
function Stop-Tree([int]$ProcId, [int]$Depth = 0) {
  try {
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $ProcId } |
                Select-Object -ExpandProperty ProcessId
    foreach ($c in $children) { Stop-Tree -ProcId $c -Depth ($Depth + 1) }
  } catch {}
  $null = Stop-ByPid $ProcId
}

function Stop-ByPattern([string]$regex, [int]$retries = 4, [int]$sleepMs = 250) {
  $killed = @()
  for ($i = 0; $i -lt $retries; $i++) {
    $toKill = Get-ProcsByRegex -regex $regex -scopePath $root
    if (-not $toKill) { break }
    foreach ($p in $toKill) {
      Stop-Tree -ProcId $p.ProcessId
      $killed += $p.ProcessId
    }
    Start-Sleep -Milliseconds $sleepMs
  }
  return $killed
}

# Try PID from lock file (handles renamed scripts / odd states)
function Stop-FromLock([string]$lockPath) {
  if (-not (Test-Path $lockPath)) { return $false }
  try {
    $txt = (Get-Content $lockPath -Raw).Trim()
    if (-not $txt) { return $false }
    $pidVal = [int]($txt -split ',')[0]
    Stop-Tree -ProcId $pidVal | Out-Null
    return $true
  } catch {
    return $false
  }
}

# ---------- 1) Kill live writer & executor ----------
$k1 = Stop-ByPattern $rxLive -retries 6 -sleepMs 300
if (-not $k1 -or $k1.Count -eq 0) {
  Write-Host "[stop_live] no live_(writer|executor) found under $root."
}

# ---------- 2) Kill from lock files (if any) ----------
$lk = $false
$lk = (Stop-FromLock $executorLock) -or $lk
$lk = (Stop-FromLock $writerLock)   -or $lk
if ($lk) { Start-Sleep -Milliseconds 200 }

# ---------- 3) Optionally kill dashboard ----------
if ($AlsoDashboard) {
  $k2 = Stop-ByPattern $rxDash -retries 5 -sleepMs 300
  if (-not $k2 -or $k2.Count -eq 0) {
    Write-Host "[stop_live] no dashboard found under $root."
  }
}

# ---------- 4) Remove locks ----------
foreach ($lkPath in @($executorLock, $writerLock)) {
  try { Remove-Item $lkPath -ErrorAction SilentlyContinue } catch {}
}

# ---------- 5) Report survivors ----------
$survivors = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and ($_.CommandLine -match (Escape-Regex $root)) -and
  ( $_.CommandLine -match $rxLive -or ($AlsoDashboard -and $_.CommandLine -match $rxDash) )
}
if ($survivors) {
  Write-Host "[stop_live] WARNING: some processes are still alive:" -ForegroundColor Yellow
  $survivors | Select-Object ProcessId, CommandLine | Format-Table -AutoSize
} else {
  Write-Host "[stop_live] all target processes stopped."
}

Write-Host "[stop_live] done."
