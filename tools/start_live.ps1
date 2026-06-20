param(
  # --- Executor knobs (overridable per run) ---
  [double]$Plong      = 0.43,
  [ValidateSet('abs','raw')] [string]$Pmode = 'abs',
  [double]$RvMax      = 60,
  [int]$Poll          = 3,
  [int]$Cooldown      = 10,
  [int]$MaxSymbols    = 2,
  [double]$RiskUSD    = 100,
  [ValidateSet('both','long_only','short_only')] [string]$ExecSides = 'both',
  [switch]$OnePosition,
  [string]$SignalsPath = '.\logs\live_signals.csv',

  # --- Adaptive gate ---
  [switch]$Adaptive,
  [double]$TargetPass = 0.20,     # ~20% pass-rate
  [int]$WindowSignals = 180,      # learn from this many signals
  [double]$ThrMin = 0.42,
  [double]$ThrMax = 0.60,
  [double]$ThrAlpha = 0.20,       # EMA smoothing

  # --- General options ---
  [switch]$Force,         # stop existing writer/executor first
  [switch]$FreshLogs,     # touch fresh .out/.err (keeps paper/closed CSVs)
  [switch]$ShowStatus,    # show quick status after start
  [switch]$StartDashboard,# also start dashboard if not running

  # --- Threshold behavior toggle ---
  [switch]$RespectWriterThr # pass --respect-writer-thr to executor
)

$ErrorActionPreference = 'Stop'

# ----------------------------
# Helpers
# ----------------------------
function Resolve-FullPath([string]$p, [string]$baseDir) {
  if ([string]::IsNullOrWhiteSpace($p)) { return $null }
  if ([System.IO.Path]::IsPathRooted($p)) { return $p }
  return [System.IO.Path]::GetFullPath((Join-Path $baseDir $p))
}

function FmtFloat([double]$x) {
  return $x.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-LiveProcs {
  # match both tools\live_writer.py and tools\live_executor.py regardless of slash direction
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'tools(\\|/)live_(writer|executor)\.py(\s|$)'
  }
}

function Stop-LiveProcs {
  param([string]$logsDir)
  $tries = 0
  do {
    $procs = Get-LiveProcs
    if ($procs) {
      if ($tries -eq 0) {
        Write-Host "[start_live] Stopping existing live_* processes..." -ForegroundColor Yellow
      }
      foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {}
      }
      Start-Sleep -Milliseconds 500
    }
    $tries++
  } while (($procs) -and ($tries -lt 6))
  # clear any stale locks either way
  foreach ($lk in @('live_executor.lock','live_writer.lock')) {
    try { Remove-Item (Join-Path $logsDir $lk) -ErrorAction SilentlyContinue } catch {}
  }
}

function Start-DashboardIfNeeded {
  param([string]$pyExe, [string]$rootDir)
  $dash = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'tools(\\|/)dashboard\.py(\s|$)' }
  if (-not $dash) {
    Write-Host "[start_live] Launching dashboard..." -ForegroundColor Cyan
    Start-Process -FilePath $pyExe `
      -ArgumentList "tools\dashboard.py" `
      -WorkingDirectory $rootDir `
      -PassThru -WindowStyle Hidden | Out-Null
  }
}

# ----------------------------
# 0) Resolve paths
# ----------------------------
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root      = (Resolve-Path (Join-Path $scriptDir '..')).Path
Set-Location $root

$logsDir   = Join-Path $root 'logs'
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$writerOut = Join-Path $logsDir 'live_writer.out'
$writerErr = Join-Path $logsDir 'live_writer.err'
$execOut   = Join-Path $logsDir 'live_executor.out'
$execErr   = Join-Path $logsDir 'live_executor.err'

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  Write-Host "[start_live] ERROR: venv python not found at $py" -ForegroundColor Red
  exit 1
}

# ----------------------------
# 1) Optionally stop existing processes
# ----------------------------
$existing = Get-LiveProcs
if ($existing -and -not $Force) {
  Write-Host "[start_live] Found existing processes. Use -Force to stop them first." -ForegroundColor Yellow
  $existing | Select-Object ProcessId, CommandLine | Format-Table -AutoSize
  exit 0
}
if ($Force) { Stop-LiveProcs -logsDir $logsDir }

# ----------------------------
# 2) Export env so writer & dashboard reflect executor gate
# ----------------------------
$env:DL_P_LONG       = (FmtFloat $Plong)
$env:DL_P_LONG_MODE  = $Pmode
# keep writer cadence roughly aligned with executor poll
$env:DL_WRITER_SLEEP = [Math]::Max(1, $Poll)

# ----------------------------
# 3) Optionally rotate/touch logs
# ----------------------------
if ($FreshLogs) {
  "" | Out-File $writerOut -Encoding utf8
  "" | Out-File $writerErr -Encoding utf8
  "" | Out-File $execOut   -Encoding utf8
  "" | Out-File $execErr   -Encoding utf8
}

# ----------------------------
# 4) Start WRITER (background)
# ----------------------------
$writerArgs = @("tools\live_writer.py")
$writer = Start-Process -FilePath $py `
  -ArgumentList $writerArgs `
  -RedirectStandardOutput $writerOut `
  -RedirectStandardError  $writerErr `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# ----------------------------
# 5) Start EXECUTOR (background)
# ----------------------------
$signalsArg = Resolve-FullPath -p $SignalsPath -baseDir $root

$executorArgs = @(
  "tools\live_executor.py",
  "--signals", $signalsArg,
  "--plong",   (FmtFloat $Plong),
  "--pmode",   $Pmode,
  "--rv-max",  (FmtFloat $RvMax),
  "--poll",    $Poll.ToString(),
  "--cooldown",$Cooldown.ToString(),
  "--max-symbols", $MaxSymbols.ToString(),
  "--risk-usd",    (FmtFloat $RiskUSD),
  "--sides",       $ExecSides
)
if ($OnePosition) { $executorArgs += "--one-position" }
if ($Adaptive) {
  $executorArgs += @(
    "--adaptive",
    "--target-pass",     (FmtFloat $TargetPass),
    "--window-signals",  $WindowSignals.ToString(),
    "--thr-min",         (FmtFloat $ThrMin),
    "--thr-max",         (FmtFloat $ThrMax),
    "--thr-alpha",       (FmtFloat $ThrAlpha)
  )
}
if ($RespectWriterThr) { $executorArgs += "--respect-writer-thr" }

$executor = Start-Process -FilePath $py `
  -ArgumentList $executorArgs `
  -RedirectStandardOutput $execOut `
  -RedirectStandardError  $execErr `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# Post-start sanity: if somehow duplicates exist, kill extras
Start-Sleep -Milliseconds 300
$alive = Get-LiveProcs | Select-Object -ExpandProperty ProcessId
if ($alive.Count -gt 2) {
  Write-Host "[start_live] Found $($alive.Count) live_* procs; pruning extras..." -ForegroundColor Yellow
  $keep = @($writer.Id, $executor.Id)
  Get-LiveProcs | Where-Object { $keep -notcontains $_.ProcessId } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
  }
}

# Optionally start dashboard
if ($StartDashboard) { Start-DashboardIfNeeded -pyExe $py -rootDir $root }

# ----------------------------
# 6) Report
# ----------------------------
Write-Host "[start_live] Writer PID  : $($writer.Id)"
Write-Host "[start_live] Executor PID: $($executor.Id)"
Write-Host "[start_live] Logs:"
Write-Host "  $writerOut"
Write-Host "  $writerErr"
Write-Host "  $execOut"
Write-Host "  $execErr"
Write-Host ""
Write-Host "Tail commands (run in another console):" -ForegroundColor Cyan
Write-Host "  Get-Content .\logs\live_writer.out   -Tail 40 -Wait"
Write-Host "  Get-Content .\logs\live_executor.out -Tail 80 -Wait"
Write-Host "  Get-Content .\logs\trades_paper_`$(Get-Date -UFormat %Y%m%d).csv -Tail 20 -Wait"

if ($ShowStatus) {
  Start-Sleep -Seconds ([Math]::Max(4, $Poll + 1))
  Write-Host ""
  Write-Host "[status] Processes:" -ForegroundColor Cyan
  Get-LiveProcs | Select-Object ProcessId, CommandLine | Format-Table -AutoSize

  if (Test-Path $signalsArg) {
    Write-Host "`n[status] Latest signals:" -ForegroundColor Cyan
    Get-Content $signalsArg -Tail 6
  }

  $todayLocal = (Get-Date).ToString('yyyyMMdd')
  $todayUTC   = ([DateTime]::UtcNow).ToString('yyyyMMdd')

  $paperLocal = Join-Path $logsDir "trades_paper_$todayLocal.csv"
  $paperUTC   = Join-Path $logsDir "trades_paper_$todayUTC.csv"
  if (Test-Path $paperLocal) {
    Write-Host "`n[status] Paper trades (local):" -ForegroundColor Cyan
    Get-Content $paperLocal -Tail 10
  }
  if ((-not (Test-Path $paperLocal)) -and (Test-Path $paperUTC)) {
    Write-Host "`n[status] Paper trades (UTC):" -ForegroundColor Cyan
    Get-Content $paperUTC -Tail 10
  }

  $closedLocal = Join-Path $logsDir "trades_closed_$todayLocal.csv"
  $closedUTC   = Join-Path $logsDir "trades_closed_$todayUTC.csv"
  if (Test-Path $closedLocal) {
    Write-Host "`n[status] Closed trades (local):" -ForegroundColor Cyan
    Get-Content $closedLocal -Tail 10
  }
  if ((-not (Test-Path $closedLocal)) -and (Test-Path $closedUTC)) {
    Write-Host "`n[status] Closed trades (UTC):" -ForegroundColor Cyan
    Get-Content $closedUTC -Tail 10
  }
}
