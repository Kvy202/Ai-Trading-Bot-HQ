param([double]$RvMax = 60, [int]$PollSec = 60, [double]$PMin = 0.40)

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)  # tools/
Set-Location ..

$py = ".\.venv\Scripts\python.exe"

Write-Host "[proxy_loop] starting; rv-max=$RvMax p-min=$PMin poll=$PollSec s"

while ($true) {
  try {
    & $py tools\live_proxy.py --rv-max $RvMax --p-min $PMin 2>&1 | Write-Output
  } catch {
    Write-Host "[proxy_loop] error: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $PollSec
}
