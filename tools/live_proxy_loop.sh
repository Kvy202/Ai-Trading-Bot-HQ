#!/bin/bash
# Proxy loop: continuously runs live_proxy.py with the given thresholds.
# Args: [RV_MAX] [POLL_SEC] [P_MIN]  (positional, all optional)

RV_MAX="${1:-60}"
POLL_SEC="${2:-60}"
P_MIN="${3:-0.40}"

cd "$(dirname "$0")/.."
PY=".venv/bin/python"

echo "[proxy_loop] starting; rv-max=$RV_MAX p-min=$P_MIN poll=${POLL_SEC}s"

while true; do
    "$PY" tools/live_proxy.py --rv-max "$RV_MAX" --p-min "$P_MIN" || true
    sleep "$POLL_SEC"
done
