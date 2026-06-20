#!/usr/bin/env python
"""
tools/telegram_controller.py
=============================
Controller Bot — receives Telegram commands and dispatches them to the
Supervisor API via authenticated HTTP calls (JWT + HMAC).

Commands:
  /start             register chat_id, show help
  /help              list commands
  /status            GET /supervisor/status
  /metrics           GET /supervisor/metrics
  /pause [reason]    POST /supervisor/pause
  /resume [reason]   POST /supervisor/resume  (PAPER: no approval needed)
  /reduce_risk       POST /supervisor/reduce_risk
  /conservative      POST /supervisor/conservative_mode
  /logs [N]          GET /supervisor/logs  (last N entries, default 5)
  /health            GET /supervisor/health

Token env var: TELEGRAM_CONTROLLER_TOKEN  (falls back to TELEGRAM_BOT_TOKEN)

Usage:
    python tools/telegram_controller.py
    python tools/telegram_controller.py --port 8789
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac as _hmac
import json
import os
import secrets
import sys
import time
import threading
from pathlib import Path

try:
    import requests as _req
except ImportError:
    print("ERROR: 'requests' not installed -- run: pip install requests")
    sys.exit(1)

_ROOT = Path(__file__).resolve().parents[1]
_ENV  = _ROOT / ".env"
_CHAT_ID_FILE = _ROOT / "logs" / "controller_chat_id.txt"


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    if not _ENV.exists():
        return
    for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# JWT (stdlib)
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_jwt(secret: str, ttl: int = 600) -> str:
    now = int(time.time())
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps({"sub": "telegram-controller", "iat": now, "exp": now + ttl},
                                  separators=(",", ":")).encode())
    msg = f"{header}.{payload}"
    sig = _hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return f"{msg}.{_b64url(sig)}"


# ---------------------------------------------------------------------------
# HMAC request signing
# ---------------------------------------------------------------------------

def _sign(method: str, path: str, body: bytes, hmac_secret: str) -> dict:
    ts    = f"{time.time():.3f}"
    nonce = secrets.token_hex(16)
    bhash = hashlib.sha256(body).hexdigest()
    canon = f"{method}\n{path}\n\n{ts}\n{nonce}\n{bhash}".encode()
    sig   = _hmac.new(hmac_secret.encode(), canon, hashlib.sha256).hexdigest()
    return {"X-Timestamp": ts, "X-Nonce": nonce, "X-Signature": sig}


# ---------------------------------------------------------------------------
# Supervisor client
# ---------------------------------------------------------------------------

class SupervisorClient:
    def __init__(self, base_url: str, jwt_secret: str, hmac_secret: str) -> None:
        self.base        = base_url.rstrip("/")
        self.jwt_secret  = jwt_secret
        self.hmac_secret = hmac_secret

    def _auth_headers(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {_make_jwt(self.jwt_secret)}",
             "Content-Type": "application/json"}
        if extra:
            h.update(extra)
        return h

    def get(self, path: str) -> tuple[int, dict]:
        try:
            r = _req.get(self.base + path, headers=self._auth_headers(), timeout=10)
            return r.status_code, r.json()
        except _req.exceptions.ConnectionError:
            return 0, {"error": "supervisor not reachable"}
        except Exception as exc:
            return -1, {"error": str(exc)}

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        raw   = json.dumps(body).encode()
        extra = _sign("POST", path, raw, self.hmac_secret)
        try:
            r = _req.post(self.base + path, data=raw,
                          headers=self._auth_headers(extra), timeout=10)
            return r.status_code, r.json()
        except _req.exceptions.ConnectionError:
            return 0, {"error": "supervisor not reachable"}
        except Exception as exc:
            return -1, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Telegram long-polling helpers
# ---------------------------------------------------------------------------

class ControllerBot:
    def __init__(self, tg_token: str, supervisor: SupervisorClient) -> None:
        self.tg_base   = f"https://api.telegram.org/bot{tg_token.strip()}"
        self.sup       = supervisor
        self.chat_id: int | None = None
        self._offset   = 0
        self._stop     = threading.Event()
        self._load_chat_id()

    def _load_chat_id(self) -> None:
        cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if cid.lstrip("-").isdigit():
            self.chat_id = int(cid)
            return
        if _CHAT_ID_FILE.exists():
            try:
                raw = _CHAT_ID_FILE.read_text().strip()
                if raw.lstrip("-").isdigit():
                    self.chat_id = int(raw)
            except Exception:
                pass

    def _save_chat_id(self, cid: int) -> None:
        self.chat_id = cid
        _CHAT_ID_FILE.parent.mkdir(exist_ok=True)
        try:
            _CHAT_ID_FILE.write_text(str(cid))
        except Exception:
            pass

    def send(self, text: str, chat_id: int | None = None) -> None:
        cid = chat_id or self.chat_id
        if not cid:
            return
        try:
            _req.post(self.tg_base + "/sendMessage",
                      json={"chat_id": cid, "text": text},
                      timeout=15)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _handle(self, cid: int, text: str) -> None:
        parts = text.strip().split()
        if not parts or not parts[0].startswith("/"):
            return
        cmd  = parts[0].lower().split("@")[0]   # strip @botname suffix
        args = parts[1:]

        if cmd == "/start":
            self._save_chat_id(cid)
            self.send(_HELP, cid)

        elif cmd == "/help":
            self.send(_HELP, cid)

        elif cmd == "/health":
            sc, data = self.sup.get("/supervisor/health")
            self.send(_fmt_health(sc, data), cid)

        elif cmd == "/status":
            sc, data = self.sup.get("/supervisor/status")
            self.send(_fmt_status(sc, data), cid)

        elif cmd == "/metrics":
            sc, data = self.sup.get("/supervisor/metrics")
            self.send(_fmt_metrics(sc, data), cid)

        elif cmd == "/pause":
            reason = " ".join(args) or "Telegram /pause"
            sc, data = self.sup.post("/supervisor/pause", {"reason": reason})
            self.send(_fmt_cmd(sc, data, "pause"), cid)

        elif cmd == "/resume":
            reason = " ".join(args) or "Telegram /resume"
            sc, data = self.sup.post("/supervisor/resume", {"reason": reason})
            self.send(_fmt_cmd(sc, data, "resume"), cid)

        elif cmd == "/reduce_risk":
            sc, data = self.sup.post("/supervisor/reduce_risk",
                                     {"reason": "Telegram /reduce_risk"})
            self.send(_fmt_cmd(sc, data, "reduce_risk"), cid)

        elif cmd == "/conservative":
            sc, data = self.sup.post("/supervisor/conservative_mode",
                                     {"reason": "Telegram /conservative"})
            self.send(_fmt_cmd(sc, data, "conservative_mode"), cid)

        elif cmd == "/logs":
            n = int(args[0]) if args and args[0].isdigit() else 5
            sc, data = self.sup.get("/supervisor/logs")
            self.send(_fmt_logs(sc, data, n), cid)

        else:
            self.send(f"Unknown command: {cmd}\n" + _HELP, cid)

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"[controller] polling Telegram (chat_id={self.chat_id})")
        while not self._stop.is_set():
            try:
                r = _req.get(self.tg_base + "/getUpdates",
                             params={"timeout": 50, "offset": self._offset + 1},
                             timeout=60)
                j = r.json()
                if not j.get("ok"):
                    time.sleep(2)
                    continue
                for upd in j.get("result", []):
                    self._offset = int(upd["update_id"])
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue
                    cid  = msg["chat"]["id"]
                    text = msg.get("text") or ""
                    if text:
                        self._handle(cid, text)
            except Exception:
                time.sleep(3)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_health(sc: int, d: dict) -> str:
    if sc == 0:
        return "Supervisor not reachable."
    if "error" in d:
        return f"Error: {d['error']}"
    return (f"Health\n"
            f"  mode:      {d.get('mode', '?')}\n"
            f"  paused:    {d.get('paused', '?')}\n"
            f"  risk_mode: {d.get('risk_mode', '?')}\n"
            f"  status:    {d.get('status', '?')}")


def _fmt_status(sc: int, d: dict) -> str:
    if sc == 0:
        return "Supervisor not reachable."
    if "error" in d:
        return f"Error: {d['error']}"
    return (f"Status\n"
            f"  paused:      {d.get('paused', '?')}\n"
            f"  mode:        {d.get('mode', '?')}\n"
            f"  risk_mode:   {d.get('risk_mode', '?')}\n"
            f"  open_symbols:{d.get('open_symbols', [])}\n"
            f"  exec_thr:    {d.get('exec_thr', '?')}")


def _fmt_metrics(sc: int, d: dict) -> str:
    if sc == 0:
        return "Supervisor not reachable."
    if "error" in d:
        return f"Error: {d['error']}"
    return (f"Metrics\n"
            f"  realized_today: {d.get('realized_sum_today', '?'):.4f} USDT\n"
            f"  unrealized_sum: {d.get('unrealized_sum', '?'):.4f} USDT\n"
            f"  drawdown_pct:   {d.get('drawdown_pct', '?'):.2f}%\n"
            f"  risk_mode:      {d.get('risk_mode', '?')}")


def _fmt_cmd(sc: int, d: dict, cmd: str) -> str:
    if sc == 0:
        return "Supervisor not reachable."
    if "error" in d:
        return f"Error: {d['error']}"
    if d.get("status") == "accepted":
        return f"Accepted: {cmd} dispatched to executor."
    if d.get("status") == "approval_required":
        aid = d.get("approval_id", "?")
        return (f"Approval required (LIVE mode).\n"
                f"Confirm via:\n  POST /supervisor/approve/{aid}")
    return f"Response ({sc}): {json.dumps(d)[:300]}"


def _fmt_logs(sc: int, d: dict, n: int) -> str:
    if sc == 0:
        return "Supervisor not reachable."
    if "error" in d:
        return f"Error: {d['error']}"
    entries = d.get("entries", [])[-n:]
    if not entries:
        return "Audit log is empty."
    lines = [f"Last {len(entries)} audit entries:"]
    for e in entries:
        lines.append(f"  [{e.get('ts', '?')[:19]}] {e.get('command', '?')} "
                     f"by {e.get('actor', '?')} -> {e.get('decision', '?')}")
    return "\n".join(lines)


_HELP = (
    "Controller Bot commands:\n"
    "/status       — bot + risk state\n"
    "/health       — supervisor health\n"
    "/metrics      — PnL + drawdown\n"
    "/pause [msg]  — pause trading\n"
    "/resume [msg] — resume trading\n"
    "/reduce_risk  — switch to reduced risk\n"
    "/conservative — switch to conservative mode\n"
    "/logs [N]     — last N audit entries (default 5)\n"
    "/help         — this message"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="Telegram Controller Bot")
    p.add_argument("--port", type=int,
                   default=int(os.getenv("SUPERVISOR_PORT", "8789")))
    p.add_argument("--url", default=None)
    args = p.parse_args()

    tg_token    = os.getenv("TELEGRAM_CONTROLLER_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    jwt_secret  = os.getenv("SUPERVISOR_JWT_SECRET", "")
    hmac_secret = os.getenv("SUPERVISOR_HMAC_SECRET", "")

    if not tg_token:
        print("ERROR: set TELEGRAM_CONTROLLER_TOKEN (or TELEGRAM_BOT_TOKEN) in .env")
        sys.exit(1)
    if not jwt_secret or not hmac_secret:
        print("ERROR: SUPERVISOR_JWT_SECRET and SUPERVISOR_HMAC_SECRET must be set in .env")
        sys.exit(1)

    sup_url = args.url or f"http://127.0.0.1:{args.port}"
    sup     = SupervisorClient(sup_url, jwt_secret, hmac_secret)
    bot     = ControllerBot(tg_token, sup)
    bot.run()


if __name__ == "__main__":
    main()
