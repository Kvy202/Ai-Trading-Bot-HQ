"""Two-step approval workflow for high-risk supervisor commands.

An approval token is created when a dangerous command is requested.
The caller must explicitly confirm it (via a second API call) before
the command is dispatched to the executor.  Unconfirmed approvals
expire after APPROVAL_TTL seconds.
"""
from __future__ import annotations

import secrets
import threading
import time

APPROVAL_TTL = 300.0  # 5 minutes

# Commands that require an explicit approval before execution.
REQUIRE_APPROVAL: frozenset[str] = frozenset({
    "resume_live",
    "increase_leverage",
    "flatten_all",
    "modify_strategy",
    "add_symbols",
    "disable_safety",
    "change_drawdown_limits",
    "switch_to_live",
})

_lock = threading.Lock()
_store: dict[str, dict] = {}


def _prune() -> None:
    """Expire stale pending and unexercised confirmed approvals (call while holding _lock)."""
    now = time.time()
    for rec in _store.values():
        if rec["status"] in ("pending", "confirmed") and rec["expires_at"] < now:
            rec["status"] = "expired"


def create(command: str, actor: str, params: dict) -> dict:
    approval_id = secrets.token_hex(16)
    now = time.time()
    record: dict = {
        "id":         approval_id,
        "command":    command,
        "actor":      actor,
        "params":     params,
        "status":     "pending",
        "created_at": now,
        "expires_at": now + APPROVAL_TTL,
    }
    with _lock:
        _prune()
        _store[approval_id] = record
    return dict(record)


def confirm(approval_id: str, actor: str) -> dict | None:
    with _lock:
        _prune()
        rec = _store.get(approval_id)
        if rec is None:
            return None
        if rec["status"] in ("expired", "confirmed", "consumed"):
            return dict(rec)
        rec["status"]       = "confirmed"
        rec["confirmed_by"] = actor
        rec["confirmed_at"] = time.time()
        # Shorten the remaining window: confirmed approval must be used within 2 minutes.
        rec["expires_at"]   = min(rec["expires_at"], time.time() + 120.0)
    return dict(rec)


def consume(approval_id: str) -> bool:
    """Mark an approval as consumed (single-use enforcement).

    Must be called immediately after the command is dispatched. Returns True if
    the approval was in confirmed state and was successfully consumed.
    """
    with _lock:
        _prune()
        rec = _store.get(approval_id)
        if rec is None or rec["status"] != "confirmed":
            return False
        rec["status"]      = "consumed"
        rec["consumed_at"] = time.time()
    return True


def get(approval_id: str) -> dict | None:
    with _lock:
        _prune()
        rec = _store.get(approval_id)
        return dict(rec) if rec else None
