"""IPC command dispatcher.

Writes supervisor commands to logs/supervisor_cmd.json, which
live_executor.py polls each loop cycle and applies atomically.
The executor then writes an ACK to logs/supervisor_ack.json.

Every command record includes:
  cmd_sig         -- HMAC authenticating the command (executor verifies)
  approval_marker -- HMAC proving a confirmed approval exists (LIVE resume only)

Design: last-write-wins. The executor compares issued_at against its
last-consumed timestamp to skip duplicates.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .auth import sign_approval_marker, sign_command

_lock = threading.Lock()

_BASE    = Path(__file__).resolve().parents[1]
CMD_FILE = _BASE / "logs" / "supervisor_cmd.json"
ACK_FILE = _BASE / "logs" / "supervisor_ack.json"

VALID_COMMANDS: frozenset[str] = frozenset({
    "pause",
    "resume",
    "reduce_risk",
    "conservative_mode",
    "emergency_stop",
})


def dispatch(command: str, actor: str,
             approval_id: str | None = None,
             params: dict | None = None) -> dict:
    if command not in VALID_COMMANDS:
        raise ValueError(f"unknown command: {command!r}")

    issued_at = time.time()
    record: dict = {
        "command":     command,
        "actor":       actor,
        "approval_id": approval_id,
        "params":      params or {},
        "issued_at":   issued_at,
        "cmd_sig":     sign_command(command, actor, issued_at),
    }

    # Approval marker is a signed proof that a confirmed approval exists for this
    # specific dispatch. The executor uses this to validate LIVE mode resumes
    # without needing access to the in-memory approval store.
    if approval_id and command == "resume":
        record["approval_marker"] = sign_approval_marker(approval_id, actor, command, issued_at)

    with _lock:
        CMD_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CMD_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(tmp, CMD_FILE)

    return record


def read_ack() -> dict | None:
    try:
        return json.loads(ACK_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
