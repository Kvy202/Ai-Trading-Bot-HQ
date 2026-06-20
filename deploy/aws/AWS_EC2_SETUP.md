# AWS EC2 Setup — Hyperliquid AI Trading Bot (NEW instance)

> ⚠️ This guide provisions a **brand-new, separate** EC2 instance for the
> Hyperliquid build. **Do not reuse, modify, or stop the old Bitget production
> instance.** Different instance, key pair, security group, and IAM role.

> 🔒 Default posture here is **Hyperliquid testnet + paper trading** (no real
> money). Real-money trading stays locked behind the full guardrail set
> (see [docs/SAFETY_CONTROLS.md](../../docs/SAFETY_CONTROLS.md)).

---

## 1. Launch the instance

- **AMI:** Ubuntu Server 22.04/24.04 LTS (x86_64).
- **Type:** `t3.small` is enough for paper/testnet (CPU-only torch inference).
- **Key pair:** create a NEW key pair (do not reuse the Bitget instance's).
- **Security group (new):**
  - Inbound: SSH (22) from **your IP only**.
  - Inbound: (optional) supervisor port from your IP only — otherwise none.
  - Outbound: allow HTTPS (443) — needed to reach the Hyperliquid API.
- **IAM role:** none required unless you use SSM/Secrets Manager (§5).

## 2. Base packages

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip git
```

## 3. Clone + virtualenv

```bash
cd ~ && git clone <your-repo-url> bot && cd bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt        # includes hyperliquid-python-sdk
```

## 4. Configure (.env) — testnet + paper

```bash
cp deploy/aws/env.hyperliquid.example .env
chmod 600 .env
nano .env     # fill HL_ACCOUNT_ADDRESS + HL_AGENT_PRIVATE_KEY
```

- `HL_ACCOUNT_ADDRESS` = your **main wallet public address**.
- `HL_AGENT_PRIVATE_KEY` = an **agent/API wallet** key (Hyperliquid UI → API →
  create/approve an agent wallet). An approved agent can sign but **cannot
  withdraw**. Never put your main wallet's private key here.
- Keep `LIVE_TRADING=false`, `PAPER_TRADING=true`, `HL_TESTNET=true` for the demo.

## 5. (Optional) Secrets via SSM / Secrets Manager

Instead of a plaintext `.env`, store secrets centrally and render `.env` at boot.

```bash
# SSM Parameter Store (SecureString); instance needs ssm:GetParameter perms.
aws ssm get-parameter --with-decryption --name /atb/hl_agent_private_key \
  --query Parameter.Value --output text
```

Pattern: keep non-secret knobs in `.env`/`config/run.json`, fetch only the secret
values (`HL_AGENT_PRIVATE_KEY`, `CONFIRM_LIVE_TRADING`) from SSM/Secrets Manager
in a small `ExecStartPre` script that writes them with `chmod 600`. Never log the
fetched values.

## 6. Smoke test (paper, then testnet)

```bash
source .venv/bin/activate
# Sanity: confirm the guardrail resolves to PAPER with the safe defaults.
python -c "from runtime.settings import Settings; from runtime.guardrails import resolve_trading_mode; \
print(resolve_trading_mode(Settings.from_env()).describe())"

# Run writer + executor in two shells (or via systemd below). Paper by default.
python tools/live_writer.py
python tools/live_executor.py --signals logs/live_signals.csv
```

Check `logs/live_executor.out`, `logs/trades_paper_*.csv`, and the heartbeat
files. Test the kill switch: `touch run/V2_PAUSE` blocks new entries within ~3 s;
`rm run/V2_PAUSE` resumes.

## 7. systemd services

```bash
export INSTALL_DIR=/home/ubuntu/bot
for f in hl-writer hl-executor hl-trading-bot; do
  sed "s#__INSTALL_DIR__#${INSTALL_DIR}#g" deploy/aws/${f}.service 2>/dev/null \
    || sed "s#__INSTALL_DIR__#${INSTALL_DIR}#g" deploy/aws/${f}.target
done
sudo cp deploy/aws/hl-writer.service deploy/aws/hl-executor.service \
        deploy/aws/hl-trading-bot.target /etc/systemd/system/
# (edit the three files to replace __INSTALL_DIR__ first, or use the sed loop)
sudo systemctl daemon-reload
sudo systemctl enable --now hl-trading-bot.target
systemctl status hl-executor.service
journalctl -u hl-executor.service -f
```

## 8. Logging, monitoring, restart

- Logs live in `~/bot/logs/`. Install rotation: `sudo cp deploy/aws/logrotate-atb.conf
  /etc/logrotate.d/atb` (edit the path inside first).
- `Restart=always` in the units auto-restarts on crash.
- Optional: ship logs to CloudWatch with the CloudWatch agent.
- Health: `tools/bot_health_check.py` and the heartbeat JSON files in `logs/`.

## 9. Going to real money (only if you truly intend to)

Real-money trading is **off** until you set, on the instance only:

```env
ENVIRONMENT=production
HL_TESTNET=false
LIVE_TRADING=true
PAPER_TRADING=false
CONFIRM_LIVE_TRADING=I_UNDERSTAND_LIVE_TRADING
```

…and provide valid mainnet credentials. Missing any one of these keeps the bot in
paper mode. For the university demonstration, **stay on testnet + paper.**
