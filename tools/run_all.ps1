param(
  [switch]$FreshLog,
  [switch]$Force,
  [double]$RvMax = 60,
  [int]$PollSec = 60,
  [string]$PMode,
  [string]$PLong,
  [string]$AllowOnly,
  [double]$PMin = 0.40,   # proxy selection threshold
  [switch]$Paper,         # run executor in paper mode (default)
  [switch]$Live,          # run executor in live mode (uses API keys)
  [switch]$Supervisor,    # also start the Supervisor API on port 8789
  [switch]$Watchdog,      # also start the process watchdog (loop every 60s)
  [switch]$Controller,    # also start the Telegram Controller Bot
  [switch]$Notifier,      # also start the Telegram Notifier Bot
  [switch]$Tier2          # also start the Tier 2 shadow data collector
)

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Resolve-Path "$base\.."
Set-Location $root

# refuse to start if already running (unless -Force)
$existing = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\live_(writer|proxy|executor).*\.py|tools\\live_proxy_loop\.ps1|tools\\telegram_(notifier|controller)\.py|tools\\watchdog\.py|tier2\\shadow_runner\.py' }
if ($existing -and -not $Force) {
  Write-Host "[run_all] already running:"
  $existing | Select-Object ProcessId, CommandLine
  Write-Host "[run_all] use -Force or run .\tools\stop_all.ps1 first."
  exit 0
}

# PYTHONPATH so 'ml_dl' imports work
$env:PYTHONPATH = ($root.Path)

# Pre-flight import check — catch namespace collisions before spawning anything.
# GIT_DIR is set to a nonexistent path only for this call so that ccxt's bundled
# toolz does not hang on "git describe --dirty" against the project repo.
# It is unset immediately after so the spawned processes are unaffected.
Write-Host "[run_all] running import sanity check..."
& ".\.venv\Scripts\python.exe" "tools\check_imports.py"
$importExit = $LASTEXITCODE
if ($importExit -ne 0) {
  Write-Host ""
  Write-Host "[run_all] ABORT: import check failed. Fix namespace collisions before starting."
  Write-Host "         Run:  python tools\check_imports.py   to see details."
  exit 1
}
Write-Host "[run_all] import check passed."

# artifacts — root model_artifacts only; subfolders (lstm/, tcn/, tx/) are Sep-2025 and must not be used
if (-not $env:DL_TX_MODEL_PATH)   { $env:DL_TX_MODEL_PATH   = "model_artifacts\dl_tx_latest.pt" }
if (-not $env:DL_TX_SCALER_PATH)  { $env:DL_TX_SCALER_PATH  = "model_artifacts\scaler_tx_latest.joblib" }
if (-not $env:DL_TCN_MODEL_PATH)  { $env:DL_TCN_MODEL_PATH  = "model_artifacts\dl_tcn_latest.pt" }
if (-not $env:DL_TCN_SCALER_PATH) { $env:DL_TCN_SCALER_PATH = "model_artifacts\scaler_tcn_latest.joblib" }
if (-not $env:DL_LSTM_MODEL_PATH) { $env:DL_LSTM_MODEL_PATH = "model_artifacts\dl_lstm_latest.pt" }
if (-not $env:DL_LSTM_SCALER_PATH){ $env:DL_LSTM_SCALER_PATH= "model_artifacts\scaler_lstm_latest.joblib" }

# writer gating knobs
if ($PMode)     { $env:DL_P_LONG_MODE = $PMode }     elseif (-not $env:DL_P_LONG_MODE) { $env:DL_P_LONG_MODE = "abs" }
if ($PLong)     { $env:DL_P_LONG      = $PLong }     elseif (-not $env:DL_P_LONG)      { $env:DL_P_LONG      = "0.08" }
if ($AllowOnly) { $env:DL_ALLOW_ONLY  = $AllowOnly } elseif (-not $env:DL_ALLOW_ONLY)  { $env:DL_ALLOW_ONLY  = "1" }

# general knobs
if (-not $env:DL_MAX_LOOKBACK_PAD) { $env:DL_MAX_LOOKBACK_PAD = "6000" }
if (-not $env:DL_SYMBOLS)          { $env:DL_SYMBOLS          = "BTCUSDT,ETHUSDT" }
if (-not $env:DL_TIMEFRAME)        { $env:DL_TIMEFRAME        = "1m" }
if (-not $env:DL_SEQ_LEN)          { $env:DL_SEQ_LEN          = "128" }
if (-not $env:DL_LOG_DIR)          { $env:DL_LOG_DIR          = "logs" }

# fresh log header if requested (master only; signals file is dynamic)
if ($FreshLog) {
  'ts,p_meta,thr,mode,rv_mean,allow,kinds_used' | Out-File .\live_meta_log.csv -Encoding utf8
}

# ensure logs dir exists
New-Item -ItemType Directory -Path .\logs -Force | Out-Null

# start writer
$writer = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "tools\live_writer.py" `
  -RedirectStandardOutput ".\logs\live_writer.out" `
  -RedirectStandardError  ".\logs\live_writer.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# start proxy loop with PMin
$proxy = Start-Process -FilePath "powershell.exe" `
  -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","tools\live_proxy_loop.ps1","-RvMax",$RvMax,"-PollSec",$PollSec,"-PMin",$PMin `
  -RedirectStandardOutput ".\logs\live_proxy.out" `
  -RedirectStandardError  ".\logs\live_proxy.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

# start executor
# --max-symbols is not hardcoded here; executor reads MAX_CONCURRENT from .env.
# --max-pos-usd is not a valid executor arg; per-trade sizing comes from .env (MAX_NOTIONAL_USDT).
# --paper / --live are now recognized by the executor; they override EXEC_PAPER/LIVE_MODE post-dotenv.
$execArgs = @("tools\live_executor.py",
              "--signals","logs\live_signals.csv",
              "--rv-max",$RvMax,
              "--plong",$env:DL_P_LONG,
              "--pmode",$env:DL_P_LONG_MODE)

if ($Live) {
  $execArgs += "--live"
} else {
  $execArgs += "--paper"
}

$executor = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList $execArgs `
  -RedirectStandardOutput ".\logs\live_executor.out" `
  -RedirectStandardError  ".\logs\live_executor.err" `
  -WorkingDirectory $root `
  -PassThru -WindowStyle Hidden

Write-Host "[run_all] started writer PID=$($writer.Id), proxy-loop PID=$($proxy.Id), executor PID=$($executor.Id)"

# optional: Supervisor API
if ($Supervisor) {
  $supScript = Join-Path $base "start_supervisor.ps1"
  $sup = Start-Process -FilePath "powershell.exe" `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$supScript `
    -RedirectStandardOutput ".\logs\supervisor.out" `
    -RedirectStandardError  ".\logs\supervisor.err" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
  Write-Host "[run_all] started supervisor PID=$($sup.Id)  (port 8789)"
}

# optional: watchdog (loop every 60s, auto-restart if all dead)
if ($Watchdog) {
  $wd = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "tools\watchdog.py","--loop","60","--restart","--quiet" `
    -RedirectStandardOutput ".\logs\watchdog.out" `
    -RedirectStandardError  ".\logs\watchdog.err" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
  Write-Host "[run_all] started watchdog PID=$($wd.Id)  (logs: .\logs\watchdog.log)"
}

# optional: Telegram Controller Bot (commands -> Supervisor API)
if ($Controller) {
  $ctrl = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "tools\telegram_controller.py" `
    -RedirectStandardOutput ".\logs\telegram_controller.out" `
    -RedirectStandardError  ".\logs\telegram_controller.err" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
  Write-Host "[run_all] started controller bot PID=$($ctrl.Id)  (logs: .\logs\telegram_controller.out)"
}

# optional: Telegram Notifier Bot (read-only health alerts)
if ($Notifier) {
  $ntfy = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "tools\telegram_notifier.py" `
    -RedirectStandardOutput ".\logs\telegram_notifier.out" `
    -RedirectStandardError  ".\logs\telegram_notifier.err" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
  Write-Host "[run_all] started notifier bot  PID=$($ntfy.Id)  (logs: .\logs\telegram_notifier.out)"
}

# optional: Tier 2 shadow data collector (no trading influence, shadow_only enforced)
if ($Tier2) {
  $env:TIER2_ENABLED     = "1"
  $env:TIER2_SHADOW_ONLY = "1"
  $t2 = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "tier2\shadow_runner.py" `
    -RedirectStandardOutput ".\logs\tier2_runner.out" `
    -RedirectStandardError  ".\logs\tier2_runner.err" `
    -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
  Write-Host "[run_all] started tier2 shadow runner PID=$($t2.Id)  (logs: .\logs\tier2_runner.out)"
}

Write-Host "[run_all] safe to close this window; processes keep running."
Write-Host "Logs: .\logs\live_writer.out, .\logs\live_proxy.out, .\logs\live_executor.out"
