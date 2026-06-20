Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\dashboard\.py' -or $_.CommandLine -match 'waitress.*tools\.dashboard:app' } |
  ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; Write-Host "Stopped PID=$($_.ProcessId)" } catch {}
  }
