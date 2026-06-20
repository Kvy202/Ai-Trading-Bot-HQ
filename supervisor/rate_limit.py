"""Per-key token-bucket rate limiter (in-memory).

Limits are expressed as (capacity, window_seconds).  Tokens refill
automatically as time passes — no background thread needed.

Keys are typically composed as "{actor}:{ip}" so limits apply per
device, user, and IP simultaneously.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_buckets: dict[str, dict] = {}

# (max_requests, window_seconds)
_LIMITS: dict[str, tuple[int, int]] = {
    "default":   (20, 60),
    "pause":     (5,  60),
    "emergency": (3,  60),
    "approval":  (10, 60),
}


def check(key: str, limit_name: str = "default") -> bool:
    """Return True if the request is within rate limits, False if it should be rejected."""
    capacity, window = _LIMITS.get(limit_name, _LIMITS["default"])
    now = time.monotonic()
    with _lock:
        b = _buckets.setdefault(key, {"tokens": float(capacity), "last": now})
        elapsed = now - b["last"]
        b["tokens"] = min(float(capacity), b["tokens"] + elapsed * capacity / window)
        b["last"] = now
        if b["tokens"] >= 1.0:
            b["tokens"] -= 1.0
            return True
        return False
