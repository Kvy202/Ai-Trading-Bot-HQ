param(
  [int]$Port = 8787,
  [ValidateSet('flask','waitress')][string]$Mode = 'flask'
)
$root = Resolve-Path "$PSScriptRoot\.."
Set-Location $root
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "venv Python not found" -ForegroundColor Red; exit 1 }

# Ensure logs dir exists and CSVs won’t 404 the UI
New-Item -ItemType Directory -Path .\logs -Force | Out-Null
foreach ($f in @(".\logs\live_signals.csv",".\logs\trades_paper_$((Get-Date).ToString('yyyyMMdd')).csv",".\logs\trades_closed.csv")) {
  if (-not (Test-Path $f)) { "header" | Out-File $f -Encoding utf8 }
}

if ($Mode -eq 'flask') {
  $env:DASH_PORT = "$Port"
  Start-Process -FilePath $py -ArgumentList "tools\dashboard.py" -WorkingDirectory $root -WindowStyle Hidden
} else {
  Start-Process -FilePath $py -ArgumentList "-m waitress --listen=127.0.0.1:$Port tools.dashboard:app" -WorkingDirectory $root -WindowStyle Hidden
}

Start-Sleep 1
Write-Host "Dashboard started on http://127.0.0.1:$Port"
