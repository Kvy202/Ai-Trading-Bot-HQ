param([switch]$FreshLog, [switch]$Force, [double]$RvMax = 60, [int]$PollSec = 60)

$base = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$base\.."

# refuse to start if already running (unless -Force)
$existing = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\live_(writer|proxy).*\.py|tools\\live_proxy_loop\.ps1' }
if ($existing -and -not $Force) {
  Write-Host "[run_all] already running:"
  $existing | Select-Object ProcessId, CommandLine
  Write-Host "[run_all] use -Force or run .\tools\stop_all.ps1 first."
  exit 0
}

# ---- env (edit if you keep artifacts elsewhere) ----
$env:DL_TX_MODEL_PATH   = "model_artifacts\tx\dl_tx_latest.pt"
$env:DL_TX_SCALER_PATH  = "model_artifacts\tx\scaler_latest.joblib"
$env:DL_TCN_MODEL_PATH  = "model_artifacts\tcn\dl_tcn_latest.pt"
$env:DL_TCN_SCALER_PATH = "model_artifacts\tcn\scaler_latest.joblib"
$env:DL_LSTM_MODEL_PATH = "model_artifacts\lstm\dl_lstm_latest.pt"
$env:DL_LSTM_SCALER_PATH= "model_artifacts\lstm\scaler_latest.joblib"

$env:DL_ALLOW_ONLY = "0"
$env:DL_P_LONG_MODE = "abs"
$env:DL_P_LONG = "0.458314"
$env:DL_MAX_LOOKBACK_PAD = "6000"

# ---- fresh log header if requested ----
if ($FreshLog) {
  'ts,p_meta,thr,mode,rv_mean,allow,kinds_used' | Out-File .\live_meta_log.csv -Encoding utf8
}

# ---- ensure logs dir exists ----
New-Item -ItemType Directory -Path .\logs -Force | Out-Null

# ---- start writer (long-lived) ----
$writer = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "tools\live_writer.py" `
  -RedirectStandardOutput ".\logs\live_writer.out" `
  -RedirectStandardError  ".\logs\live_writer.err" `
  -PassThru -WindowStyle Hidden

# ---- start proxy loop (keeps re-running live_proxy) ----
$proxy = Start-Process -FilePath "powershell.exe" `
  -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","tools\live_proxy_loop.ps1","-RvMax",$RvMax,"-PollSec",$PollSec `
  -RedirectStandardOutput ".\logs\live_proxy.out" `
  -RedirectStandardError  ".\logs\live_proxy.err" `
  -PassThru -WindowStyle Hidden

Write-Host "[run_all] started writer PID=$($writer.Id) and proxy-loop PID=$($proxy.Id)"
Write-Host "[run_all] safe to close this window; processes keep running."
Write-Host "Logs: .\logs\live_writer.out and .\logs\live_proxy.out"
