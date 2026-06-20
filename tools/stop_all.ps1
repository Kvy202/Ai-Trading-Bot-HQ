# Stop all bot processes: writer, executor, proxy loop, notifier, controller, watchdog, tier2 runner
$procs = Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'tools\\live_(writer|proxy|executor).*\.py|tools\\live_proxy_loop\.ps1|tools\\telegram_(notifier|controller)\.py|tools\\watchdog\.py|tier2\\shadow_runner\.py' }

if ($procs) {
  foreach ($p in $procs) {
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      Write-Host "[stop_all] stopped PID=$($p.ProcessId)"
    } catch {
      Write-Host "[stop_all] could not stop PID=$($p.ProcessId): $($_.Exception.Message)"
    }
  }
} else {
  Write-Host "[stop_all] nothing to stop."
}
