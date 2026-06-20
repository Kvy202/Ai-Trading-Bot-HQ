#!/bin/bash
# AWS/Linux setup for the AI trading bot.
# Run once as the deploy user (e.g. ubuntu) after cloning the repo.
# Usage: bash setup.sh [--no-systemd]

set -euo pipefail

INSTALL_SYSTEMD=1
[[ "${1:-}" == "--no-systemd" ]] && INSTALL_SYSTEMD=0

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== Trading bot setup: $INSTALL_DIR ==="

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[setup] installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-dev \
    build-essential git libssl-dev libffi-dev

# ── 2. Python virtual environment ─────────────────────────────────────────────
echo "[setup] creating .venv..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip wheel

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo "[setup] installing Python packages (this takes a few minutes)..."
# torch+cpu variant is already pinned in requirements.txt
"$INSTALL_DIR/.venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── 4. Runtime directories ────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR/logs"

# ── 5. Script permissions ─────────────────────────────────────────────────────
chmod +x "$INSTALL_DIR/tools/run_all.sh" \
         "$INSTALL_DIR/tools/stop_all.sh" \
         "$INSTALL_DIR/tools/live_proxy_loop.sh"

# ── 6. .env sanity check ──────────────────────────────────────────────────────
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    echo ""
    echo "[setup] WARNING: .env not found — copy .env.example and fill in your keys:"
    echo "  cp $INSTALL_DIR/.env.example $INSTALL_DIR/.env"
    echo "  nano $INSTALL_DIR/.env"
fi

# ── 7. Systemd services ───────────────────────────────────────────────────────
if [[ $INSTALL_SYSTEMD -eq 1 ]]; then
    echo "[setup] installing systemd services..."

    for svc in bot-writer bot-executor bot-proxy; do
        sed "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
            "$INSTALL_DIR/systemd/${svc}.service" \
            | sudo tee "$SYSTEMD_DIR/${svc}.service" > /dev/null
        echo "  installed $SYSTEMD_DIR/${svc}.service"
    done

    sudo cp "$INSTALL_DIR/systemd/trading-bot.target" "$SYSTEMD_DIR/trading-bot.target"
    sudo systemctl daemon-reload
    sudo systemctl enable trading-bot.target bot-writer.service bot-executor.service bot-proxy.service

    echo ""
    echo "[setup] systemd installed. Start with:"
    echo "  sudo systemctl start trading-bot.target"
    echo ""
    echo "  Status:  sudo systemctl status bot-writer bot-executor bot-proxy"
    echo "  Logs:    journalctl -fu bot-writer  (or check $INSTALL_DIR/logs/)"
    echo "  Stop:    sudo systemctl stop trading-bot.target"
fi

echo ""
echo "=== Setup complete ==="
echo "Manual start (no systemd): bash $INSTALL_DIR/tools/run_all.sh --live"
