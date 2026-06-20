import os, time
from datetime import datetime, timezone
from typing import Dict, Any

# Enable/disable via env vars (no code changes needed)
FOREGROUND_LOG = bool(int(os.getenv("FOREGROUND_LOG", "1")))
FG_SYNC_THROTTLE_S = int(os.getenv("FG_SYNC_THROTTLE_S", "60"))  # min seconds between sync prints

_last_sync: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fg_setup() -> None:
    """One-time notify when foreground logging is enabled."""
    if not FOREGROUND_LOG:
        return
    print(f"[fg] {_now_iso()} foreground logging enabled", flush=True)


def fg_emit(event: Dict[str, Any]) -> None:
    """Emit a compact, human-friendly console line for key events.

    Call this from your code right after you persist the event to heartbeat.json.
    Safe to call even if FOREGROUND_LOG is off (it no-ops).
    """
    if not FOREGROUND_LOG:
        return

    et = (event.get("event") or "event").lower()
    ts = event.get("ts") or _now_iso()

    if et == "sync_positions":
        global _last_sync
        now = time.time()
        if now - _last_sync < FG_SYNC_THROTTLE_S:
            return
        _last_sync = now
        cnt = event.get("count")
        mode = event.get("mode")
        suffix = f" mode={mode}" if mode else ""
        print(f"[fg] {ts} sync_positions count={cnt}{suffix}", flush=True)
        return

    if et == "order":
        sym = event.get("symbol", "?")
        side = event.get("side", "?")
        qty = event.get("qty")
        reason = event.get("reason")
        stop = event.get("stop")
        take = event.get("take")
        bits = [f"order {sym} {side}"]
        if qty is not None:
            bits.append(f"qty={qty}")
        if stop is not None and take is not None:
            bits.append(f"sl={stop} tp={take}")
        if reason:
            bits.append(reason)
        print("[fg] " + ts + " " + " ".join(bits), flush=True)
        return

    if et in ("boot", "shutdown", "heartbeat"):
        note = event.get("note")
        suffix = (" " + str(note)) if note else ""
        print(f"[fg] {ts} {et}{suffix}", flush=True)
        return

    # default fallback (kept intentionally terse)
    print(f"[fg] {ts} {et}", flush=True)
