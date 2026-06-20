"""Append-only supervisor audit log.

Every supervisory action is written as a single JSON line to
logs/supervisor_audit.jsonl.  The file is never rewritten — only
appended — so it acts as an immutable record of who did what and when.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_AUDIT_LOG = Path(__file__).resolve().parents[1] / "logs" / "supervisor_audit.jsonl"


def log_action(
    *,
    actor: str,
    device: str | None,
    command: str,
    approval_id: str | None,
    prior_state: dict,
    result_state: dict,
    decision: str,
) -> None:
    entry = {
        "ts":           time.time(),
        "actor":        actor,
        "device":       device,
        "command":      command,
        "approval_id":  approval_id,
        "prior_state":  prior_state,
        "result_state": result_state,
        "decision":     decision,
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with _lock:
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)


def tail_audit(n: int = 50) -> list[dict]:
    if not _AUDIT_LOG.exists():
        return []
    raw_lines = _AUDIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
    entries: list[dict] = []
    for raw in raw_lines[-n:]:
        try:
            entries.append(json.loads(raw))
        except Exception:
            continue
    return entries
