"""Secure Supervisor API — integration boundary for Trady.

Runs as a separate process on SUPERVISOR_PORT (default 8789).
The executor (tools/live_executor.py) remains the authoritative
execution engine; this API provides a narrow, authenticated window
for Trady to observe state and issue soft commands.

Quick-start
-----------
1. Generate secrets (run once, store in .env):
       python -c "import secrets; print(secrets.token_hex(32))"
   Add to .env:
       SUPERVISOR_JWT_SECRET=<output>
       SUPERVISOR_HMAC_SECRET=<output>

2. Generate a token for the Trady backend:
       python tools/gen_supervisor_token.py --sub trady-backend --days 90

3. Start the API:
       .\\tools\\start_supervisor.ps1

Security model
--------------
- All endpoints require a signed HS256 JWT (Bearer token).
- State-changing endpoints (POST) additionally require an HMAC-SHA256
  request signature with timestamp + nonce for replay protection.
- Resuming live trading requires a two-step approval workflow.
- Exchange credentials never appear in any response.
- Rate limiting is applied per actor+IP on every endpoint.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from flask import Flask, g, jsonify, request

from .auth import require_auth
from .audit import log_action, tail_audit
from .approvals import confirm, consume, create, get
from .commands import dispatch, read_ack
from .rate_limit import check as rate_check

app = Flask(__name__)

_BASE      = Path(__file__).resolve().parents[1]
_LOGS      = _BASE / "logs"
_STATE     = _LOGS / "executor_state.json"
_EXEC_ERR  = _LOGS / "live_executor.err"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _actor() -> str:
    return g.jwt_payload.get("sub", "unknown")


def _device() -> str | None:
    return g.jwt_payload.get("device")


def _client_ip() -> str:
    # Only trust X-Forwarded-For when a trusted reverse proxy is explicitly configured.
    # Without this guard, clients can spoof the header and bypass per-IP rate limits.
    if os.environ.get("SUPERVISOR_TRUSTED_PROXY"):
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
    return request.remote_addr or "?"


def _rate_guard(limit: str = "default"):
    """Return a 429 response if the caller is rate-limited, else None."""
    key = f"{_actor()}:{_client_ip()}"
    if not rate_check(key, limit):
        return jsonify({"error": "rate limit exceeded"}), 429
    return None


def _read_state() -> dict:
    try:
        return json.loads(_STATE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _safe_state(state: dict) -> dict:
    """Return only the fields that are safe to expose to Trady."""
    return {
        "ts":           state.get("ts"),
        "mode":         state.get("mode"),
        "paused":       state.get("paused", False),
        "risk_mode":    state.get("risk_mode", "normal"),
        "exec_thr":     state.get("exec_thr"),
        "adaptive":     state.get("adaptive"),
        "open_symbols": list(state.get("open_positions", {}).keys()),
    }


def _tail_err(n: int = 30) -> list[str]:
    if not _EXEC_ERR.exists():
        return []
    lines = _EXEC_ERR.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


# ---------------------------------------------------------------------------
# Read-only endpoints (JWT only — no HMAC required)
# ---------------------------------------------------------------------------

@app.get("/supervisor/health")
@require_auth
def health():
    _rl = _rate_guard()
    if _rl:
        return _rl
    state = _read_state()
    ack   = read_ack()
    return jsonify({
        "status":    "ok",
        "ts":        time.time(),
        "mode":      state.get("mode", "unknown"),
        "paused":    state.get("paused", False),
        "risk_mode": state.get("risk_mode", "normal"),
        "last_ack":  ack,
    })


@app.get("/supervisor/status")
@require_auth
def status():
    _rl = _rate_guard()
    if _rl:
        return _rl
    return jsonify(_safe_state(_read_state()))


@app.get("/supervisor/metrics")
@require_auth
def metrics():
    _rl = _rate_guard()
    if _rl:
        return _rl
    state = _read_state()
    return jsonify({
        "ts":                 time.time(),
        "unrealized_sum":     state.get("unrealized_sum", 0.0),
        "realized_sum_today": state.get("realized_sum_today", 0.0),
        "drawdown_pct":       state.get("drawdown_pct", 0.0),
        "open_symbols":       list(state.get("open_positions", {}).keys()),
        "risk_mode":          state.get("risk_mode", "normal"),
        "paused":             state.get("paused", False),
    })


@app.get("/supervisor/positions")
@require_auth
def positions():
    _rl = _rate_guard()
    if _rl:
        return _rl
    state = _read_state()
    return jsonify({
        "ts":        time.time(),
        "positions": state.get("open_positions", {}),
    })


@app.get("/supervisor/risk_mode")
@require_auth
def risk_mode():
    _rl = _rate_guard()
    if _rl:
        return _rl
    state = _read_state()
    return jsonify({
        "ts":        time.time(),
        "risk_mode": state.get("risk_mode", "normal"),
        "paused":    state.get("paused", False),
    })


@app.get("/supervisor/logs")
@require_auth
def logs():
    _rl = _rate_guard()
    if _rl:
        return _rl
    n = min(int(request.args.get("n", 50)), 200)
    return jsonify({"ts": time.time(), "entries": tail_audit(n)})


@app.get("/supervisor/alerts")
@require_auth
def alerts():
    _rl = _rate_guard()
    if _rl:
        return _rl
    n = min(int(request.args.get("n", 30)), 200)
    return jsonify({"ts": time.time(), "alerts": _tail_err(n)})


# ---------------------------------------------------------------------------
# Command endpoints (JWT + HMAC required)
# ---------------------------------------------------------------------------

@app.post("/supervisor/pause")
@require_auth
def pause_bot():
    _rl = _rate_guard("pause")
    if _rl:
        return _rl
    actor  = _actor()
    device = _device()
    prior  = _read_state()

    cmd = dispatch("pause", actor)
    log_action(
        actor=actor, device=device, command="pause", approval_id=None,
        prior_state={"paused": prior.get("paused"), "risk_mode": prior.get("risk_mode")},
        result_state={"paused": True},
        decision="accepted",
    )
    return jsonify({"status": "accepted", "command": "pause", "issued_at": cmd["issued_at"]})


@app.post("/supervisor/resume")
@require_auth
def resume_bot():
    _rl = _rate_guard("pause")
    if _rl:
        return _rl
    actor  = _actor()
    device = _device()
    state  = _read_state()

    approval_id = None
    # Resuming live trading always requires a two-step approval.
    if state.get("mode", "").upper() == "LIVE":
        body        = request.get_json(silent=True) or {}
        approval_id = body.get("approval_id")
        if not approval_id:
            rec = create("resume_live", actor, {})
            log_action(
                actor=actor, device=device, command="resume_live",
                approval_id=rec["id"],
                prior_state={"paused": True},
                result_state={"paused": True},
                decision="approval_required",
            )
            return jsonify({
                "status":      "approval_required",
                "approval_id": rec["id"],
                "expires_at":  rec["expires_at"],
                "message":     "Confirm via POST /supervisor/approve/{approval_id}",
            }), 202

        rec = get(approval_id)
        if not rec or rec["status"] != "confirmed":
            return jsonify({"error": "approval not found or not yet confirmed"}), 403

    cmd = dispatch("resume", actor, approval_id=approval_id if state.get("mode", "").upper() == "LIVE" else None)
    # Single-use: consume the approval immediately after dispatch so it cannot be reused.
    if approval_id:
        consume(approval_id)
    log_action(
        actor=actor, device=device, command="resume", approval_id=None,
        prior_state={"paused": state.get("paused")},
        result_state={"paused": False, "risk_mode": "normal"},
        decision="accepted",
    )
    return jsonify({"status": "accepted", "command": "resume", "issued_at": cmd["issued_at"]})


@app.post("/supervisor/reduce_risk")
@require_auth
def reduce_risk():
    _rl = _rate_guard()
    if _rl:
        return _rl
    actor  = _actor()
    device = _device()
    prior  = _read_state()

    cmd = dispatch("reduce_risk", actor)
    log_action(
        actor=actor, device=device, command="reduce_risk", approval_id=None,
        prior_state={"risk_mode": prior.get("risk_mode")},
        result_state={"risk_mode": "reduced"},
        decision="accepted",
    )
    return jsonify({"status": "accepted", "command": "reduce_risk", "issued_at": cmd["issued_at"]})


@app.post("/supervisor/conservative_mode")
@require_auth
def conservative_mode():
    _rl = _rate_guard()
    if _rl:
        return _rl
    actor  = _actor()
    device = _device()
    prior  = _read_state()

    cmd = dispatch("conservative_mode", actor)
    log_action(
        actor=actor, device=device, command="conservative_mode", approval_id=None,
        prior_state={"risk_mode": prior.get("risk_mode")},
        result_state={"risk_mode": "conservative"},
        decision="accepted",
    )
    return jsonify({"status": "accepted", "command": "conservative_mode", "issued_at": cmd["issued_at"]})


@app.post("/supervisor/emergency_stop")
@require_auth
def emergency_stop():
    _rl = _rate_guard("emergency")
    if _rl:
        return _rl
    actor  = _actor()
    device = _device()
    state  = _read_state()

    # Emergency stop via API is restricted to paper/demo to prevent accidental
    # live position flattening through a potentially-compromised Trady session.
    if state.get("mode", "").upper() == "LIVE":
        return jsonify({
            "error": "Emergency stop via supervisor API is restricted to paper/demo mode. "
                     "Use POST /supervisor/pause for live mode, or intervene directly.",
        }), 403

    cmd = dispatch("emergency_stop", actor)
    log_action(
        actor=actor, device=device, command="emergency_stop", approval_id=None,
        prior_state={"mode": state.get("mode"), "paused": state.get("paused")},
        result_state={"paused": True, "risk_mode": "conservative"},
        decision="accepted",
    )
    return jsonify({"status": "accepted", "command": "emergency_stop", "issued_at": cmd["issued_at"]})


# ---------------------------------------------------------------------------
# Approval workflow
# ---------------------------------------------------------------------------

@app.post("/supervisor/approve/<approval_id>")
@require_auth
def approve(approval_id: str):
    _rl = _rate_guard("approval")
    if _rl:
        return _rl
    actor = _actor()
    rec   = confirm(approval_id, actor)
    if rec is None:
        return jsonify({"error": "approval not found"}), 404
    return jsonify(rec)


@app.get("/supervisor/approve/<approval_id>")
@require_auth
def approval_status(approval_id: str):
    _rl = _rate_guard()
    if _rl:
        return _rl
    rec = get(approval_id)
    if rec is None:
        return jsonify({"error": "approval not found"}), 404
    return jsonify(rec)


# ---------------------------------------------------------------------------
# Entry point (dev only — use waitress in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SUPERVISOR_PORT", 8789))
    app.run(host="0.0.0.0", port=port, debug=False)
