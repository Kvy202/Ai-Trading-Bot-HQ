#!/bin/bash
# Stop all bot processes: writer, executor, proxy loop, notifier, controller, watchdog, tier2

PATTERN='tools/live_(writer|proxy|executor)[^/]*\.py|tools/live_proxy_loop\.sh|tools/telegram_(notifier|controller)\.py|tools/watchdog\.py|tier2/shadow_runner\.py'

mapfile -t pids < <(pgrep -f "$PATTERN" 2>/dev/null || true)

if [[ ${#pids[@]} -eq 0 ]]; then
    echo "[stop_all] nothing to stop."
    exit 0
fi

for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "[stop_all] stopped PID=$pid" || echo "[stop_all] could not stop PID=$pid"
    fi
done
