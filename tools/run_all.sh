#!/bin/bash
set -euo pipefail
# Run all bot processes: writer, proxy-loop, executor, and optional components.
# Usage: bash tools/run_all.sh [options]
#
# Options:
#   --fresh-log           Overwrite live_meta_log.csv header
#   --force               Start even if processes are already running
#   --rv-max N            RV cap for proxy/executor (default 60)
#   --poll-sec N          Proxy loop poll interval in seconds (default 60)
#   --p-mode MODE         P_LONG mode: abs|raw (default abs)
#   --p-long N            P_LONG threshold (default 0.08)
#   --allow-only N        DL_ALLOW_ONLY value (default 1)
#   --p-min N             Proxy selection min threshold (default 0.40)
#   --paper               Run executor in paper mode (default)
#   --live                Run executor in live mode (uses API keys)
#   --supervisor          Also start Supervisor API on port 8789
#   --watchdog            Also start process watchdog (loop every 60s)
#   --controller          Also start Telegram Controller Bot
#   --notifier            Also start Telegram Notifier Bot
#   --tier2               Also start Tier 2 shadow data collector

FRESH_LOG=0; FORCE=0; RV_MAX=60; POLL_SEC=60
P_MODE=""; P_LONG=""; ALLOW_ONLY=""; P_MIN=0.40
MODE_LIVE=0
SUPERVISOR=0; WATCHDOG=0; CONTROLLER=0; NOTIFIER=0; TIER2=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fresh-log)   FRESH_LOG=1; shift ;;
        --force)       FORCE=1; shift ;;
        --rv-max)      RV_MAX="$2"; shift 2 ;;
        --poll-sec)    POLL_SEC="$2"; shift 2 ;;
        --p-mode)      P_MODE="$2"; shift 2 ;;
        --p-long)      P_LONG="$2"; shift 2 ;;
        --allow-only)  ALLOW_ONLY="$2"; shift 2 ;;
        --p-min)       P_MIN="$2"; shift 2 ;;
        --paper)       MODE_LIVE=0; shift ;;
        --live)        MODE_LIVE=1; shift ;;
        --supervisor)  SUPERVISOR=1; shift ;;
        --watchdog)    WATCHDOG=1; shift ;;
        --controller)  CONTROLLER=1; shift ;;
        --notifier)    NOTIFIER=1; shift ;;
        --tier2)       TIER2=1; shift ;;
        *) echo "[run_all] unknown arg: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

# Refuse to start if already running (unless --force)
if [[ $FORCE -eq 0 ]]; then
    existing=$(pgrep -f 'tools/live_(writer|proxy|executor)[^/]*\.py|tools/live_proxy_loop\.sh' 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        echo "[run_all] already running PIDs: $existing"
        echo "[run_all] use --force or run bash tools/stop_all.sh first."
        exit 0
    fi
fi

export PYTHONPATH="$ROOT"

# Pre-flight import check
echo "[run_all] running import sanity check..."
if ! "$PY" tools/check_imports.py; then
    echo ""
    echo "[run_all] ABORT: import check failed."
    echo "         Run:  python tools/check_imports.py  to see details."
    exit 1
fi
echo "[run_all] import check passed."

# Model artifact paths — default to root model_artifacts only
: "${DL_TX_MODEL_PATH:=model_artifacts/dl_tx_latest.pt}"
: "${DL_TX_SCALER_PATH:=model_artifacts/scaler_tx_latest.joblib}"
: "${DL_TCN_MODEL_PATH:=model_artifacts/dl_tcn_latest.pt}"
: "${DL_TCN_SCALER_PATH:=model_artifacts/scaler_tcn_latest.joblib}"
: "${DL_LSTM_MODEL_PATH:=model_artifacts/dl_lstm_latest.pt}"
: "${DL_LSTM_SCALER_PATH:=model_artifacts/scaler_lstm_latest.joblib}"
: "${DL_ADV_MODEL_PATH:=model_artifacts/dl_adv_latest.pt}"
: "${DL_ADV_SCALER_PATH:=model_artifacts/scaler_adv_latest.joblib}"
export DL_TX_MODEL_PATH DL_TX_SCALER_PATH DL_TCN_MODEL_PATH \
       DL_TCN_SCALER_PATH DL_LSTM_MODEL_PATH DL_LSTM_SCALER_PATH \
       DL_ADV_MODEL_PATH DL_ADV_SCALER_PATH

# Writer gating knobs
[[ -n "$P_MODE" ]]     && export DL_P_LONG_MODE="$P_MODE"
[[ -n "$P_LONG" ]]     && export DL_P_LONG="$P_LONG"
[[ -n "$ALLOW_ONLY" ]] && export DL_ALLOW_ONLY="$ALLOW_ONLY"
: "${DL_P_LONG_MODE:=abs}";  export DL_P_LONG_MODE
: "${DL_P_LONG:=0.08}";      export DL_P_LONG
: "${DL_ALLOW_ONLY:=1}";     export DL_ALLOW_ONLY

# General knobs
: "${DL_MAX_LOOKBACK_PAD:=6000}"; export DL_MAX_LOOKBACK_PAD
: "${DL_SYMBOLS:=BTCUSDT,ETHUSDT}"; export DL_SYMBOLS
: "${DL_TIMEFRAME:=1m}";      export DL_TIMEFRAME
: "${DL_SEQ_LEN:=64}";        export DL_SEQ_LEN
: "${DL_LOG_DIR:=logs}";      export DL_LOG_DIR

# Fresh log header
[[ $FRESH_LOG -eq 1 ]] && echo 'ts,p_meta,thr,mode,rv_mean,allow,kinds_used' > live_meta_log.csv

mkdir -p logs

# Start writer
nohup "$PY" tools/live_writer.py \
    >> logs/live_writer.out 2>> logs/live_writer.err &
WRITER_PID=$!

# Start proxy loop
nohup bash tools/live_proxy_loop.sh "$RV_MAX" "$POLL_SEC" "$P_MIN" \
    >> logs/live_proxy.out 2>> logs/live_proxy.err &
PROXY_PID=$!

# Start executor
EXEC_ARGS=("tools/live_executor.py"
           "--signals" "logs/live_signals.csv"
           "--rv-max" "$RV_MAX"
           "--plong"  "$DL_P_LONG"
           "--pmode"  "$DL_P_LONG_MODE")
[[ $MODE_LIVE -eq 1 ]] && EXEC_ARGS+=("--live") || EXEC_ARGS+=("--paper")

nohup "$PY" "${EXEC_ARGS[@]}" \
    >> logs/live_executor.out 2>> logs/live_executor.err &
EXEC_PID=$!

echo "[run_all] started writer PID=$WRITER_PID, proxy-loop PID=$PROXY_PID, executor PID=$EXEC_PID"

# Optional components
if [[ $SUPERVISOR -eq 1 ]]; then
    nohup "$PY" supervisor/server.py >> logs/supervisor.out 2>> logs/supervisor.err &
    echo "[run_all] started supervisor PID=$!  (port 8789)"
fi

if [[ $WATCHDOG -eq 1 ]]; then
    nohup "$PY" tools/watchdog.py --loop 60 --restart --quiet \
        >> logs/watchdog.out 2>> logs/watchdog.err &
    echo "[run_all] started watchdog PID=$!  (logs: logs/watchdog.log)"
fi

if [[ $CONTROLLER -eq 1 ]]; then
    nohup "$PY" tools/telegram_controller.py \
        >> logs/telegram_controller.out 2>> logs/telegram_controller.err &
    echo "[run_all] started controller bot PID=$!"
fi

if [[ $NOTIFIER -eq 1 ]]; then
    nohup "$PY" tools/telegram_notifier.py \
        >> logs/telegram_notifier.out 2>> logs/telegram_notifier.err &
    echo "[run_all] started notifier bot PID=$!"
fi

if [[ $TIER2 -eq 1 ]]; then
    export TIER2_ENABLED=1 TIER2_SHADOW_ONLY=1
    nohup "$PY" tier2/shadow_runner.py \
        >> logs/tier2_runner.out 2>> logs/tier2_runner.err &
    echo "[run_all] started tier2 shadow runner PID=$!"
fi

echo "[run_all] safe to close this terminal; processes keep running."
echo "Logs: logs/live_writer.out  logs/live_proxy.out  logs/live_executor.out"
