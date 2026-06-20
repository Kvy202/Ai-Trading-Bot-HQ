"""
Safe live/paper signal executor for the AI trading bot.

Reads live signals from logs/live_signals.csv and executes either paper trades
or real Bitget USDT-M futures market orders, depending on .env:

    LIVE_MODE=1 and EXEC_PAPER=0   -> real orders
    anything else                  -> paper only

Expected signal columns (minimum):
    ts,symbol,px,p_meta,rv_mean,allow,thr,mode

Extra columns are ignored.

Changelog vs previous version:
- Portfolio cap is now also enforced on scale-in (was only checked on new entries).
- --pmode is now actually applied in threshold_pass (was only used by the
  adaptive thresholder, which made the env knob misleading).
- Duplicate-price guard now uses a relative tolerance instead of float equality.
- log_err also writes to stderr to match the writer and aid debugging.
- Bad/zero/negative prices in signals are rejected before any sizing math.
- Symbol normalization for live mode: signal files use 'BTCUSDT' but ccxt
  Bitget unified swaps are 'BTC/USDT:USDT'. The Broker now builds a market
  map on init and translates shorthand to the right unified symbol before
  placing orders. Paper mode is unchanged.
- Position state is persisted to executor_state.json (atomic write) and
  reloaded on startup. In live mode, state is reconciled against the
  exchange via fetch_positions so a restart doesn't forget open trades or
  double up. Controlled by EXEC_RESTORE_STATE (default 1).
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import hmac as _hmac
import json
import math
import os
import signal as signal_module  # renamed to avoid clashing with SignalRow var "signal"
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None  # custom load_dotenv below handles it

try:
    import ccxt  # type: ignore
except Exception:  # ccxt is optional in paper mode
    ccxt = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# If this file lives inside a `tools/` subfolder, the project base is one level up.
_THIS = Path(__file__).resolve()
BASE_DIR = _THIS.parents[1] if _THIS.parent.name == "tools" else _THIS.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Repo root must be importable so `exchanges` / `runtime` resolve when this file
# is run directly as `python tools/live_executor.py` (then sys.path[0] is tools/).
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Shared position type + the exchange-adapter layer. These imports are light and
# offline-safe: the factory imports the concrete adapters (ccxt / Hyperliquid SDK)
# lazily, and settings/guardrails are stdlib-only.
from exchanges.types import Position
from exchanges.factory import make_adapter
from runtime.settings import Settings, scrub
from runtime.guardrails import resolve_trading_mode

OUT_LOG = LOGS_DIR / "live_executor.out"
ERR_LOG = LOGS_DIR / "live_executor.err"
RUNTIME_LOG = LOGS_DIR / "runtime.log"
LOCK_PATH = LOGS_DIR / "live_executor.lock"
STATE_JSON = LOGS_DIR / "executor_state.json"
HEARTBEAT_JSON      = LOGS_DIR / "heartbeat.json"
SUPERVISOR_CMD_FILE = LOGS_DIR / "supervisor_cmd.json"
SUPERVISOR_ACK_FILE = LOGS_DIR / "supervisor_ack.json"
CLOSED_MASTER_CSV   = LOGS_DIR / "trades_closed.csv"

# V2 optional risk controls (v2/risk_controls.py). Stays None unless main()
# initializes it; every use is None-guarded + try/except so a V2 failure can
# never break the V1 loop. No module-level import: tools.live_executor must
# stay importable offline (tools/test_fixes_123.py) and without the v2/ package.
_V2_RISK = None

# ---------------------------------------------------------------------------
# Env / config helpers
# ---------------------------------------------------------------------------

def load_dotenv(path: Optional[Path] = None, override: bool = True) -> None:
    """Load .env, overriding stale OS env values by default."""
    env_path = path or (BASE_DIR / ".env")
    if not env_path.exists():
        return
    if _dotenv_load is not None:
        _dotenv_load(env_path, override=override)
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if override or key not in os.environ:
            os.environ[key] = val


def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_bool(name: str, default: bool = False) -> bool:
    val = env_str(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, str(default))))
    except Exception:
        return int(default)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def log(msg: str) -> None:
    line = f"[{utc_ts()}] {msg}\n"
    write_line(OUT_LOG, line)
    write_line(RUNTIME_LOG, line)
    if env_bool("FOREGROUND_LOG", False):
        print(line, end="", flush=True)


def log_err(msg: str) -> None:
    line = f"[{utc_ts()}] {msg}\n"
    write_line(ERR_LOG, line)
    write_line(RUNTIME_LOG, line)
    # Always echo errors to stderr so they show up in service logs / terminals.
    print(line, end="", file=sys.stderr, flush=True)


def write_heartbeat(event: str, **fields: Any) -> None:
    data = {"ts": utc_ts(), "event": event, **fields}
    try:
        HEARTBEAT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

PAPER_HEADER = ["ts", "symbol", "side", "price", "qty", "reason", "mode", "order_id"]
CLOSED_HEADER = ["ts", "symbol", "closed_side", "qty", "entry_avg", "exit_price",
                 "realized_pnl", "reason"]


def paper_path_for_day(d: date) -> Path:
    return LOGS_DIR / f"trades_paper_{d:%Y%m%d}.csv"


def closed_path_for_day(d: date) -> Path:
    return LOGS_DIR / f"trades_closed_{d:%Y%m%d}.csv"


def ensure_header(path: Path, header: Iterable[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(list(header))


def append_csv(path: Path, header: Iterable[str], row: Iterable[Any]) -> None:
    ensure_header(path, header)
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(list(row))


def record_trade(path: Path, row: List[Any]) -> None:
    append_csv(path, PAPER_HEADER, row)


def record_closed_trade(ts: str, symbol: str, closed_side: str, qty: float, entry_avg: float,
                        exit_price: float, realized_pnl: float, reason: str) -> None:
    row = [ts, symbol, closed_side, qty, entry_avg, exit_price, realized_pnl, reason]
    append_csv(CLOSED_MASTER_CSV, CLOSED_HEADER, row)
    append_csv(closed_path_for_day(datetime.now(timezone.utc).date()), CLOSED_HEADER, row)
    # V2 hook: single choke point through which every close passes (TP/SL/flip/
    # time-stop/restart-close). Feeds the daily SL-count / loss / DD counters.
    if _V2_RISK is not None:
        try:
            _V2_RISK.on_close(symbol, reason, realized_pnl)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def single_instance_lock(stale_sec: int = 900) -> None:
    if LOCK_PATH.exists():
        try:
            pid_s, ts_s = LOCK_PATH.read_text(encoding="utf-8").strip().split(",", 1)
            old_pid = int(pid_s)
            old_ts = float(ts_s)
            if old_pid != os.getpid() and pid_alive(old_pid) and (time.time() - old_ts) < stale_sec:
                log_err(f"lock: another live_executor is running pid={old_pid}; exiting")
                sys.exit(0)
            log("lock: replacing stale/dead lock")
        except Exception:
            log("lock: replacing unreadable lock")
    LOCK_PATH.write_text(f"{os.getpid()},{time.time()}", encoding="utf-8")


def unlock() -> None:
    try:
        if LOCK_PATH.exists():
            pid_s = LOCK_PATH.read_text(encoding="utf-8").split(",", 1)[0]
            if int(pid_s) == os.getpid():
                LOCK_PATH.unlink()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Signal parsing / gating
# ---------------------------------------------------------------------------

@dataclass
class SignalRow:
    ts: str
    symbol: str
    price: float
    p_meta: float
    rv_mean: float
    allow: int
    thr: float
    mode: str  # "abs" or "raw" - as written by the live writer
    kinds_used: Optional[str] = None  # None = old-format row (col absent); "" = present but empty


def _is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def parse_signal_line(line: str) -> Optional[SignalRow]:
    try:
        parts = next(csv.reader([line]))
        if len(parts) < 8:
            return None
        # Skip header rows if present
        if parts[0].lower() == "ts" or parts[1].lower() == "symbol":
            return None
        ts, symbol, px, p_meta, rv_mean, allow, thr, mode = parts[:8]
        # kinds_used is column 9 (index 8); None means old-format row without the column.
        kinds_used: Optional[str] = parts[8].strip() if len(parts) > 8 else None
        price = float(px) if str(px).strip() else 0.0
        p = float(p_meta)
        rv = float(rv_mean)
        t = float(thr)
        # Reject NaN/inf in any numeric field - these would poison the executor.
        if not all(_is_finite_number(v) for v in (price, p, rv, t)):
            return None
        return SignalRow(
            ts=ts.strip(),
            symbol=symbol.strip(),
            price=price,
            p_meta=p,
            rv_mean=rv,
            allow=1 if str(allow).strip().lower() in {"1", "true", "yes", "y"} else 0,
            thr=t,
            mode=(mode or "abs").strip().lower(),
            kinds_used=kinds_used,
        )
    except Exception:
        return None


def mode_value(p_meta: float, mode: str) -> float:
    """Apply 'abs' or 'raw' interpretation to p_meta."""
    return abs(p_meta) if (mode or "abs").lower() == "abs" else p_meta


def threshold_pass(sig: SignalRow, exec_thr: float, exec_mode: str,
                   respect_writer_thr: bool) -> Tuple[bool, str]:
    """Decide whether a signal passes the entry gate.

    The mode used here is the executor's configured mode (--pmode / EXEC_PMODE),
    NOT whatever the writer happened to put in the signal row. This means
    operators can run the executor in 'abs' mode even if the writer logs raw,
    or vice versa, and the env knob actually has an effect.
    """
    if sig.allow != 1:
        return False, "allow=0"
    # Defense-in-depth: if the writer wrote allow=1 but no model contributed,
    # kinds_used will be an empty string. Reject those rows. Old-format rows
    # (kinds_used is None = column was absent) are passed through unchanged.
    if sig.kinds_used is not None and not sig.kinds_used:
        return False, "empty_kinds_used"
    val = mode_value(sig.p_meta, exec_mode)
    eff = max(sig.thr, exec_thr) if respect_writer_thr else exec_thr
    if val < eff:
        return False, (f"below_thr({val:.4f}<{eff:.4f}) "
                       f"writer_thr={sig.thr:.4f} exec_thr={exec_thr:.4f}")
    return True, ""


def side_allowed(p_meta: float, sides: str) -> bool:
    sides = (sides or "both").lower()
    return sides == "both" or (sides == "long_only" and p_meta >= 0) or (sides == "short_only" and p_meta < 0)


def symbol_allowed(symbol: str, whitelist: List[str]) -> bool:
    return not whitelist or symbol in whitelist


def read_recent_signals(path: Path, last_ts: Dict[str, str], window: int) -> List[SignalRow]:
    """Return at most one most-recent signal per symbol from the last `window` lines."""
    lines = tail_lines(path, window)
    if not lines:
        return []
    newest: Dict[str, SignalRow] = {}
    for raw in reversed(lines):
        sig = parse_signal_line(raw)
        if not sig:
            continue
        
        if last_ts.get(sig.symbol) == sig.ts:
            continue
        newest.setdefault(sig.symbol, sig)
    out = list(newest.values())
    out.sort(key=lambda s: s.ts)
    return out


def tail_lines(path: Path, n: int) -> List[str]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        data = b""
        block = 8192
        while end > 0 and data.count(b"\n") <= n + 1:
            take = min(block, end)
            end -= take
            f.seek(end)
            data = f.read(take) + data
    return [ln.decode("utf-8", errors="ignore").strip() for ln in data.splitlines() if ln.strip()][-n:]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round(max(0.0, min(1.0, q)) * (len(vals) - 1)))
    return vals[idx]


def adaptive_threshold(path: Path, window: int, pmode: str, target_pass: float,
                       thr_min: float, thr_max: float, prev: float, alpha: float) -> Optional[float]:
    """Roll an EMA of the percentile threshold needed to hit target_pass over the window."""
    vals: List[float] = []
    for raw in tail_lines(path, window):
        sig = parse_signal_line(raw)
        if sig:
            vals.append(abs(sig.p_meta) if pmode == "abs" else sig.p_meta)
    if len(vals) < 12:
        return None
    raw_thr = percentile(vals, 1.0 - target_pass)
    smoothed = alpha * raw_thr + (1.0 - alpha) * prev
    return max(thr_min, min(thr_max, smoothed))


# Bias guard time window: ignore allowed signals older than this many seconds.
# 1800 s = 30 min. Post-promotion, old-model signals age out in at most one window.
_BIAS_WINDOW_SEC: float = 1800.0


def _parse_signal_ts(ts_str: str) -> Optional[float]:
    """Parse a signal timestamp string to a UTC Unix timestamp. Returns None on failure."""
    try:
        s = ts_str.strip().replace("+0000", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        try:
            dt = datetime.strptime(ts_str.strip()[:19], "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            return None


def check_side_bias(
    path: Path,
    window: int = 1000,
    bias_threshold: float = 0.95,
    max_age_sec: Optional[float] = _BIAS_WINDOW_SEC,
    min_wall_sec: Optional[float] = None,
) -> Tuple[Optional[str], bool, int, int, int]:
    """Return (warning_or_None, is_biased, n_total, n_long, n_short) for recent *allowed* signals.

    is_biased is True when >= bias_threshold fraction are the same side.
    Requires at least 50 allowed rows in the (filtered) window before firing.

    max_age_sec: ignore signals whose timestamp is older than this many seconds ago
                 (default 1800 s = 30 min, so old pre-promotion signals age out naturally).
    min_wall_sec: ignore signals with a parsed timestamp before this Unix time
                  (pass executor start time to prevent stale history from re-locking on restart).
    """
    now_wall = time.time()
    sides: List[str] = []
    for raw in tail_lines(path, window):
        sig = parse_signal_line(raw)
        if sig and sig.allow == 1:
            if max_age_sec is not None or min_wall_sec is not None:
                sig_t = _parse_signal_ts(sig.ts)
                if sig_t is not None:
                    if max_age_sec is not None and (now_wall - sig_t) > max_age_sec:
                        continue
                    if min_wall_sec is not None and sig_t < min_wall_sec:
                        continue
            sides.append("long" if sig.p_meta >= 0 else "short")
    n = len(sides)
    short_n = sum(1 for s in sides if s == "short")
    long_n = n - short_n
    if n < 50:
        return None, False, n, long_n, short_n
    if short_n / n >= bias_threshold:
        return (f"SIDE_BIAS: {short_n}/{n} ({short_n/n:.0%}) recent allowed signals are SHORT "
                f"- model may be poorly calibrated or overfit to recent bearish data"), True, n, long_n, short_n
    if long_n / n >= bias_threshold:
        return (f"SIDE_BIAS: {long_n}/{n} ({long_n/n:.0%}) recent allowed signals are LONG "
                f"- model may be poorly calibrated or overfit to recent bullish data"), True, n, long_n, short_n
    return None, False, n, long_n, short_n

# ---------------------------------------------------------------------------
# Position / order execution
# ---------------------------------------------------------------------------

# Position now lives in exchanges/types.py (shared with the adapter layer) and is
# imported at the top of this module. SupervisorState stays executor-local.
@dataclass
class SupervisorState:
    paused: bool = False
    risk_mode: str = "normal"  # "normal" | "reduced" | "conservative"
    last_cmd_ts: float = 0.0


_RISK_NOTIONAL_MULT: Dict[str, float] = {
    "normal":       1.0,
    "reduced":      0.5,
    "conservative": 0.25,
}


def pnl_on_close(pos: Position, price: float) -> float:
    """GROSS pnl (no costs). Use net_pnl_on_close for realistic paper P&L."""
    return (price - pos.avg) * pos.qty if pos.side == "long" else (pos.avg - price) * pos.qty


def apply_slippage(price: float, action: str, slippage_bps: float) -> float:
    """Worsen a fill price by `slippage_bps` (per side) — always adverse to us.

    Buys (BUY / BUY_TO_COVER) fill HIGHER; sells (SELL / SELL_SHORT) fill LOWER.
    This makes paper fills resemble real taker fills instead of optimistic mid.
    """
    if price <= 0 or not slippage_bps:
        return price
    s = float(slippage_bps) / 1e4
    a = (action or "").upper()
    if a in ("BUY", "BUY_TO_COVER"):
        return price * (1.0 + s)
    if a in ("SELL", "SELL_SHORT"):
        return price * (1.0 - s)
    return price


def fee_cost(notional: float, fee_bps: float) -> float:
    """Taker fee on ONE side, in quote currency (USDT)."""
    return abs(notional) * (float(fee_bps) / 1e4)


def net_pnl_on_close(pos: Position, exit_mid: float, action_close: str,
                     fee_bps: float, slippage_bps: float) -> Tuple[float, float]:
    """Return (net_pnl, exit_fill_price) for closing `pos` at observed `exit_mid`.

    Includes exit-side slippage and taker fees on BOTH entry and exit notional.
    Entry-side slippage is already baked into pos.avg when the position opened,
    so it is not re-applied here. The TP/SL *trigger* still uses the observed
    mid price; only the fill is made adverse.
    """
    exit_fill = apply_slippage(exit_mid, action_close, slippage_bps)
    gross = pnl_on_close(pos, exit_fill)
    fees = fee_cost(pos.avg * pos.qty, fee_bps) + fee_cost(exit_fill * pos.qty, fee_bps)
    return gross - fees, exit_fill


def qty_for(price: float, notional_usdt: float, min_notional: float, min_qty: float) -> float:
    if price <= 0 or notional_usdt <= 0:
        return 0.0
    if notional_usdt < min_notional:
        return 0.0
    qty = notional_usdt / price
    if qty < min_qty:
        return 0.0
    return round(qty, 8)


def prices_close(a: float, b: float, rel_tol: float = 1e-9) -> bool:
    """Relative-tolerance float compare for duplicate-fill detection."""
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) <= rel_tol * max(abs(a), abs(b))


def _verify_cmd_sig(data: dict) -> bool:
    """Verify the HMAC signature of a command file record.

    Returns True when SUPERVISOR_HMAC_SECRET is not set (local dev mode).
    Returns False when a secret IS configured but the signature is missing or wrong,
    so an attacker who can write to supervisor_cmd.json cannot forge commands.
    """
    secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")
    if not secret:
        return True  # dev mode: secret not configured -> unauthenticated IPC accepted
    sig = data.get("cmd_sig", "")
    if not sig:
        return False
    command   = data.get("command", "")
    actor     = data.get("actor", "")
    issued_at = float(data.get("issued_at", 0.0))
    expected  = _hmac.new(
        secret.encode(),
        f"{command}:{actor}:{issued_at:.3f}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return _hmac.compare_digest(expected, sig)


def _verify_approval_marker(data: dict) -> bool:
    """Verify the approval_marker in a LIVE-mode resume command.

    Returns True in dev mode (no secret). Returns False when secret IS set
    and the marker is absent, wrong, or doesn't match this specific dispatch.
    """
    secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")
    if not secret:
        return True  # dev mode
    approval_id = data.get("approval_id", "")
    marker      = data.get("approval_marker", "")
    if not approval_id or not marker:
        return False
    actor     = data.get("actor", "")
    command   = data.get("command", "")
    issued_at = float(data.get("issued_at", 0.0))
    expected  = _hmac.new(
        secret.encode(),
        f"approve:{approval_id}:{actor}:{command}:{issued_at:.3f}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return _hmac.compare_digest(expected, marker)


def poll_supervisor_cmd(sv: SupervisorState, mode_name: str) -> Optional[str]:
    """Read and apply a pending supervisor command. Mutates sv. Returns command name or None."""
    if not SUPERVISOR_CMD_FILE.exists():
        return None
    try:
        data = json.loads(SUPERVISOR_CMD_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None

    issued_at = float(data.get("issued_at", 0.0))
    if issued_at <= sv.last_cmd_ts:
        return None  # already consumed

    # Reject commands that lack a valid HMAC signature when a secret is configured.
    if not _verify_cmd_sig(data):
        log_err("SUPERVISOR cmd rejected: invalid cmd_sig (possible unauthorized file write)")
        sv.last_cmd_ts = issued_at  # advance ts so we don't log this every loop
        return None

    cmd   = str(data.get("command", ""))
    actor = str(data.get("actor", "unknown"))
    sv.last_cmd_ts = issued_at

    if cmd == "pause":
        sv.paused = True
        log(f"SUPERVISOR {cmd} actor={actor!r}")
    elif cmd == "resume":
        if mode_name == "LIVE":
            if not _verify_approval_marker(data):
                log_err("SUPERVISOR resume rejected in LIVE mode: missing or invalid approval_marker")
                return None
        sv.paused    = False
        sv.risk_mode = "normal"
        log(f"SUPERVISOR {cmd} actor={actor!r} approval={data.get('approval_id', 'n/a')}")
    elif cmd == "reduce_risk":
        sv.risk_mode = "reduced"
        log(f"SUPERVISOR {cmd} actor={actor!r}")
    elif cmd == "conservative_mode":
        sv.risk_mode = "conservative"
        log(f"SUPERVISOR {cmd} actor={actor!r}")
    elif cmd == "emergency_stop":
        if mode_name != "LIVE":
            sv.paused    = True
            sv.risk_mode = "conservative"
            log(f"SUPERVISOR {cmd} actor={actor!r}")
        else:
            log_err("SUPERVISOR emergency_stop rejected in LIVE mode -- use pause instead")
            return None
    else:
        log_err(f"SUPERVISOR unknown command {cmd!r} from {actor!r}")
        return None

    try:
        ack = {
            "command":    cmd,
            "actor":      actor,
            "issued_at":  issued_at,
            "applied_at": time.time(),
            "paused":     sv.paused,
            "risk_mode":  sv.risk_mode,
        }
        tmp = SUPERVISOR_ACK_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ack, indent=2), encoding="utf-8")
        os.replace(tmp, SUPERVISOR_ACK_FILE)
    except Exception:
        pass

    return cmd


# NOTE: The Bitget `Broker` class that used to live here has been extracted
# verbatim into exchanges/bitget_adapter.py (BitgetAdapter) and a Hyperliquid
# sibling (exchanges/hyperliquid_adapter.py) added. The executor now obtains an
# ExchangeAdapter from exchanges.factory.make_adapter() based on the EXCHANGE
# env var, with the live/testnet/sandbox decision made by
# runtime.guardrails.resolve_trading_mode(). The adapter exposes the same
# methods the loop already called (create_market_order / fetch_open_positions /
# fetch_current_price), so the trading loop below is unchanged.

# ---------------------------------------------------------------------------
# Per-tick helpers (kept small so the main loop reads top-to-bottom)
# ---------------------------------------------------------------------------

def maybe_rotate_paper_path(current_day: date, paper_path: Path) -> Tuple[date, Path]:
    """Rotate the per-day paper trades CSV when UTC date changes."""
    today = datetime.now(timezone.utc).date()
    if today != current_day:
        new_path = paper_path_for_day(today)
        ensure_header(new_path, PAPER_HEADER)
        return today, new_path
    return current_day, paper_path


def write_state_snapshot(mode_name: str, exec_thr: float, exec_mode: str, adaptive: bool,
                         positions: Dict[str, Position],
                         paused: bool = False, risk_mode: str = "normal") -> None:
    """Atomically write the executor's current state to STATE_JSON.

    Used both as a status file for dashboards and as a way to recover
    open positions after a restart (see load_positions_from_state).
    """
    try:
        payload = {
            "ts":             utc_ts(),
            "mode":           mode_name,
            "exec_thr":       exec_thr,
            "exec_mode":      exec_mode,
            "adaptive":       bool(adaptive),
            "paused":         paused,
            "risk_mode":      risk_mode,
            "open_positions": {k: v.__dict__ for k, v in positions.items()},
        }
        tmp = STATE_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_JSON)
    except Exception:
        # Snapshot is informational - never let it crash the loop.
        pass


def load_positions_from_state(path: Path) -> Dict[str, Position]:
    """Read positions back from STATE_JSON written by a previous run.

    Returns an empty dict if the file is missing, unreadable, or doesn't
    contain valid position data. Designed to be safe to call unconditionally
    on startup - it will never raise.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log_err(f"state_restore: cannot parse {path}: {e}")
        return {}

    raw = data.get("open_positions", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Position] = {}
    for sym, pos_dict in raw.items():
        if not isinstance(pos_dict, dict):
            continue
        try:
            side = str(pos_dict.get("side", "")).lower()
            qty = float(pos_dict.get("qty", 0))
            avg = float(pos_dict.get("avg", 0))
        except Exception:
            continue
        # Sanity-check every field before trusting it.
        if side not in ("long", "short"):
            continue
        if qty <= 0 or not math.isfinite(qty):
            continue
        if avg <= 0 or not math.isfinite(avg):
            continue
        out[str(sym)] = Position(side=side, qty=qty, avg=avg)
    return out


def reconcile_live_positions(local: Dict[str, Position],
                             from_exchange: Dict[str, Position]) -> Dict[str, Position]:
    """In live mode, exchange positions are the source of truth.

    Strategy:
      - Any position on the exchange is included as-is.
      - Any local-only position is dropped (it must have been closed externally,
        liquidated, or never actually opened on the exchange).
      - Discrepancies are logged so the operator can investigate.
    """
    final: Dict[str, Position] = dict(from_exchange)

    # Log positions that the exchange reports but our local state doesn't know about.
    for sym, pos in from_exchange.items():
        loc = local.get(sym)
        if loc is None:
            log(f"state_restore: exchange has {sym} {pos.side} qty={pos.qty} avg={pos.avg} "
                f"(not in local state - adopting)")
        elif loc.side != pos.side or abs(loc.qty - pos.qty) > 1e-9 * max(loc.qty, pos.qty):
            log(f"state_restore: {sym} mismatch local={loc.side}/{loc.qty} "
                f"exchange={pos.side}/{pos.qty} - using exchange")

    # Log positions we thought we had but the exchange doesn't.
    for sym in local:
        if sym not in from_exchange:
            log(f"state_restore: local {sym} not on exchange - dropping")

    return final


def check_tp_sl(pos: Position, price: float, tp_pct: float, sl_pct: float) -> Tuple[bool, bool]:
    """Return (hit_tp, hit_sl) for the given position at `price`."""
    if pos.side == "long":
        hit_tp = tp_pct > 0 and price >= pos.avg * (1 + tp_pct)
        hit_sl = sl_pct > 0 and price <= pos.avg * (1 - sl_pct)
    else:
        hit_tp = tp_pct > 0 and price <= pos.avg * (1 - tp_pct)
        hit_sl = sl_pct > 0 and price >= pos.avg * (1 + sl_pct)
    return hit_tp, hit_sl


def portfolio_exposure(positions: Dict[str, Position]) -> float:
    return sum(p.qty * p.avg for p in positions.values())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe Bitget signal executor: paper by default, live only when explicitly enabled"
    )
    parser.add_argument("--signals", default=str(LOGS_DIR / "live_signals.csv"))
    parser.add_argument("--plong", type=float, default=env_float("DL_P_LONG", 0.55))
    parser.add_argument("--pmode", choices=["abs", "raw"],
                        default=env_str("DL_P_LONG_MODE", "abs") or "abs")
    parser.add_argument("--rv-max", type=float,
                        default=env_float("EXEC_RV_MAX", env_float("DL_MAX_RV", 0.02)))
    parser.add_argument("--poll", type=float, default=env_float("EXEC_POLL_SEC", 3.0))
    parser.add_argument("--cooldown", type=float, default=env_float("EXEC_COOLDOWN_SEC", 30.0))
    parser.add_argument("--sides", choices=["both", "long_only", "short_only"],
                        default=env_str("EXEC_SIDES", "both") or "both")
    parser.add_argument("--max-symbols", type=int, default=env_int("MAX_CONCURRENT", 1))
    parser.add_argument("--one-position", action="store_true",
                        default=env_bool("EXEC_ONE_POSITION", False))
    parser.add_argument("--notional-usdt", type=float,
                        default=env_float("PER_SYMBOL_NOTIONAL_USDT",
                                          env_float("MAX_NOTIONAL_USDT", 15.0)))
    parser.add_argument("--max-portfolio-usdt", type=float,
                        default=env_float("MAX_PORTFOLIO_EXPOSURE_USDT", 30.0))
    parser.add_argument("--min-notional", type=float, default=env_float("EXEC_MIN_NOTIONAL", 5.0))
    parser.add_argument("--min-qty", type=float, default=env_float("EXEC_MIN_QTY", 0.0))
    parser.add_argument("--tail", type=int, default=50)
    parser.add_argument("--tp-pct", type=float, default=env_float("EXEC_TP_PCT", 0.01))
    parser.add_argument("--sl-pct", type=float, default=env_float("EXEC_SL_PCT", 0.02))
    # Realistic-cost knobs (applied to paper P&L; also used to estimate live P&L
    # in our own bookkeeping). Basis points PER SIDE: 1 bp = 0.01%.
    parser.add_argument("--fee-bps", type=float, default=env_float("EXEC_FEE_BPS", 5.0),
                        help="Taker fee per side in basis points (5 = 0.05%%). "
                             "Charged on entry AND exit notional.")
    parser.add_argument("--slippage-bps", type=float, default=env_float("EXEC_SLIPPAGE_BPS", 2.0),
                        help="Adverse slippage per side in basis points (2 = 0.02%%). "
                             "Worsens every fill price; TP/SL triggers still use mid.")
    parser.add_argument("--flip-open", action="store_true", default=env_bool("EXEC_FLIP_OPEN", True))
    parser.add_argument("--flip-confirm-ticks", type=int,
                        default=env_int("EXEC_FLIP_CONFIRM_TICKS", 3),
                        help="Number of consecutive valid opposite-entry candidates (allow=1, above threshold, "
                             "within RV guard) required before a flip-close executes. Signals rejected by "
                             "allow=0, threshold, or RV guard do not reset the counter -- only a same-side "
                             "valid signal, a TP/SL close, or a fresh entry resets it. "
                             "(0 = immediate flip, original behaviour; 3 = default)")
    parser.add_argument("--scale-in", action="store_true", default=env_bool("EXEC_SCALE_IN", False))
    parser.add_argument("--respect-writer-thr", action="store_true",
                        default=env_bool("EXEC_RESPECT_WRITER_THR", True))
    parser.add_argument("--adaptive", action="store_true", default=env_bool("EXEC_ADAPTIVE", False))
    parser.add_argument("--target-pass", type=float, default=env_float("EXEC_TARGET_PASS", 0.20))
    parser.add_argument("--window-signals", type=int, default=env_int("EXEC_WINDOW_SIGNALS", 180))
    parser.add_argument("--thr-min", type=float, default=env_float("EXEC_THR_MIN", 0.40))
    parser.add_argument("--thr-max", type=float, default=env_float("EXEC_THR_MAX", 0.60))
    parser.add_argument("--thr-alpha", type=float, default=env_float("EXEC_THR_EMA_ALPHA", 0.20))
    parser.add_argument("--restore-state", action="store_true",
                        default=env_bool("EXEC_RESTORE_STATE", True),
                        help="On startup, restore open positions from the previous run "
                             "and (in live mode) sync against the exchange.")
    # These override EXEC_PAPER / LIVE_MODE from .env *after* dotenv is loaded,
    # so run_all.ps1 -Live / -Paper switches work even when .env says otherwise.
    parser.add_argument("--live", action="store_true", default=False,
                        help="Force live mode (overrides LIVE_MODE/EXEC_PAPER in .env)")
    parser.add_argument("--paper", action="store_true", default=False,
                        help="Force paper mode (overrides LIVE_MODE/EXEC_PAPER in .env)")
    parser.add_argument("--bias-guard", action="store_true",
                        default=env_bool("EXEC_BIAS_GUARD", True),
                        help="Suspend new entries when >=95%% of recent allowed signals are "
                             "one-sided. TP/SL exits and flip-closes remain active.")
    parser.add_argument("--no-bias-guard", dest="bias_guard", action="store_false")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    # Load run.json defaults first; load_dotenv (override=True) then wins.
    # Priority: shell env > .env > config/run.json
    try:
        from runtime.loader import apply_run_config as _apply_run_config
        _apply_run_config(BASE_DIR)
    except Exception:
        pass
    load_dotenv()  # custom loader -- re-applies any .env keys not yet in os.environ
    args = build_args(argv)

    # CLI flags override whatever .env said (load_dotenv already ran above).
    if args.live:
        os.environ["LIVE_MODE"] = "1"
        os.environ["EXEC_PAPER"] = "0"
    elif args.paper:
        os.environ["EXEC_PAPER"] = "1"
        os.environ["LIVE_MODE"] = "0"

    signals_path = Path(args.signals)
    if not signals_path.is_absolute():
        signals_path = (BASE_DIR / signals_path).resolve()

    # --- Trading-mode guardrail (safe by default) ---------------------------
    # resolve_trading_mode() is the SINGLE authority on whether real orders are
    # placed. It forces PAPER unless every required confirmation is present
    # (see runtime/guardrails.py). The legacy LIVE_MODE/EXEC_PAPER/BITGET_SANDBOX
    # knobs (pre-seeded by --live/--paper above) still feed into it for backward
    # compatibility, but a missing confirmation can only ever downgrade to paper.
    settings = Settings.from_env()
    decision = resolve_trading_mode(settings, cli_live=args.live, cli_paper=args.paper, log=log)
    live = decision.place_real_orders
    sandbox = decision.sandbox
    mode_name = decision.mode_name
    log(f"settings: {settings.summary()}")

    single_instance_lock()
    atexit.register(unlock)

    def stop_handler(signum: int, frame: Any) -> None:
        log(f"executor stopped by signal={signum}")
        unlock()
        raise SystemExit(0)

    signal_module.signal(signal_module.SIGINT, stop_handler)
    signal_module.signal(signal_module.SIGTERM, stop_handler)

    try:
        broker = make_adapter(
            exchange=settings.exchange,
            live=live,
            testnet=decision.testnet,
            sandbox=sandbox,
            log=log,
            log_err=log_err,
        )
    except Exception as e:
        # scrub() masks any secret/key-like text before it can reach the logs.
        log_err(f"FATAL adapter init: {scrub(str(e))}\n{scrub(traceback.format_exc())}")
        sys.exit(1)

    current_day = datetime.now(timezone.utc).date()
    paper_path = paper_path_for_day(current_day)
    ensure_header(paper_path, PAPER_HEADER)
    ensure_header(CLOSED_MASTER_CSV, CLOSED_HEADER)

    exec_thr = float(args.plong)
    exec_mode = (args.pmode or "abs").lower()

    last_signal_ts: Dict[str, str] = {}
    last_fill_time: Dict[str, float] = {}
    last_fill_price: Dict[str, float] = {}
    positions: Dict[str, Position] = {}
    active_symbols: set[str] = set()
    idle_no_file = 0
    idle_no_new = 0
    sv_state = SupervisorState()
    _bias_tick = 200         # check on first loop iteration, not after 10 min warmup
    _BIAS_EVERY = 200        # check every ~200 polls (~10 min at poll=3 s) when unlocked
    _BIAS_LOCKED_EVERY = 10  # re-check every ~10 polls (~30 s) while locked so it clears promptly
    _BIAS_MIN_CLEAR = 50     # minimum fresh allowed-signal samples required to declare bias cleared
    _bias_locked = False     # True while side bias >= 95%; blocks new entries only
    _start_ts = time.time()  # executor start time; bias check ignores signals before this
    # Flip confirmation: tracks (wanted_side, consecutive_tick_count) per symbol.
    # A flip-close only executes once the counter reaches args.flip_confirm_ticks.
    _flip_pending: Dict[str, Tuple[str, int]] = {}

    # --- Position recovery ---------------------------------------------------
    # Without this, restarting the executor while positions are open means it
    # forgets them: TP/SL won't fire, and the next entry signal will open a
    # second position on top of the forgotten one. The state file is written
    # atomically every tick so it should always be a consistent snapshot.
    if args.restore_state:
        restored = load_positions_from_state(STATE_JSON)
        if restored:
            log(f"state_restore: loaded {len(restored)} position(s) from {STATE_JSON.name}")

        if live and restored is not None:
            # Live mode: the exchange is the source of truth. Pull live positions
            # and reconcile. If fetch fails, fall back to local state with a warning.
            exchange_positions = broker.fetch_open_positions()
            if exchange_positions:
                positions = reconcile_live_positions(restored, exchange_positions)
            else:
                log("state_restore: no exchange positions returned - "
                    "using local state only (verify manually if you expected open positions)")
                positions = restored
        else:
            positions = restored

        active_symbols = set(positions.keys())
        for sym in positions:
            log(f"state_restore: tracking {sym} {positions[sym].side} "
                f"qty={positions[sym].qty} avg={positions[sym].avg}")

        # Immediately close any restored position already past its TP/SL at the
        # current market price. Without this, the position sits open until the
        # next writer signal arrives, which can be minutes — during which price
        # can move further against us (the EXIT_SL loss keeps growing).
        if live and positions:
            ts_now = utc_ts()
            for sym in list(positions.keys()):
                pos = positions[sym]
                price = broker.fetch_current_price(sym)
                if price is None:
                    log(f"restart_close: cannot fetch price for {sym} — will rely on signal TP/SL")
                    continue
                hit_tp, hit_sl = check_tp_sl(pos, price, args.tp_pct, args.sl_pct)
                if not (hit_tp or hit_sl):
                    continue
                action = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                try:
                    order_id = broker.create_market_order(sym, action, pos.qty, reduce_only=True)
                except Exception as exc:
                    log_err(f"restart_close: order failed for {sym}: {exc}")
                    continue
                pnl, exit_fill = net_pnl_on_close(pos, price, action,
                                                  args.fee_bps, args.slippage_bps)
                reason = f"EXIT_{'TP' if hit_tp else 'SL'}_RESTART pnl={pnl:.6f}"
                record_trade(paper_path, [ts_now, sym, action, exit_fill, pos.qty, reason, mode_name, order_id])
                record_closed_trade(ts_now, sym, action, pos.qty, pos.avg, exit_fill, pnl, reason)
                log(f"RESTART_CLOSE {mode_name} {action} {sym} qty={pos.qty} "
                    f"px={exit_fill:.8f} order={order_id} {reason}")
                positions.pop(sym)
                active_symbols.discard(sym)

    # --- V2 optional risk controls (all flags off by default) ---------------
    # Initialized AFTER the restart-close block so the counter rebuild from
    # today's trades_closed CSV includes any EXIT_*_RESTART rows written above.
    # On any failure (v2/ missing, bad env, bug) the executor runs pure V1.
    global _V2_RISK
    try:
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from v2.risk_controls import init_risk_controls as _v2_init
        _V2_RISK = _v2_init(BASE_DIR, LOGS_DIR, positions.keys(), log, log_err)
    except Exception as exc:
        log_err(f"v2_risk init failed - running without V2 risk controls: {exc}")
        _V2_RISK = None

    whitelist_env = env_str("EXEC_SYMBOL_WHITELIST") or env_str("SYMBOL_WHITELIST")
    whitelist = [x.strip() for x in whitelist_env.split(",") if x.strip()]

    log(f"START mode={mode_name} sandbox={sandbox} signals={signals_path} "
        f"exec_thr={exec_thr:.4f} pmode={exec_mode} adaptive={args.adaptive} "
        f"restore_state={args.restore_state} open_positions={len(positions)}")
    log(f"costs: fee_bps={args.fee_bps:.3f}/side slippage_bps={args.slippage_bps:.3f}/side "
        f"(paper P&L now nets fees + slippage; TP/SL trigger on mid)")
    write_heartbeat("executor_start", mode=mode_name, sandbox=sandbox,
                    exec_thr=exec_thr, pmode=exec_mode,
                    open_positions=len(positions))

    # Drain pre-start signals so a restart never replays stale rows as fresh entries.
    if signals_path.exists():
        for _s in read_recent_signals(signals_path, {}, args.tail):
            last_signal_ts[_s.symbol] = _s.ts

    while True:
        try:
            # --- Supervisor command poll ---
            sv_cmd = poll_supervisor_cmd(sv_state, mode_name)
            if sv_cmd:
                write_heartbeat("supervisor_cmd", mode=mode_name, command=sv_cmd,
                                paused=sv_state.paused, risk_mode=sv_state.risk_mode)

            # --- Day rollover & adaptive threshold update ---
            current_day, paper_path = maybe_rotate_paper_path(current_day, paper_path)

            if args.adaptive and signals_path.exists():
                new_thr = adaptive_threshold(
                    signals_path, args.window_signals, exec_mode, args.target_pass,
                    args.thr_min, args.thr_max, exec_thr, args.thr_alpha,
                )
                if new_thr is not None and abs(new_thr - exec_thr) > 1e-6:
                    exec_thr = float(new_thr)
                    log(f"ADAPT_THR exec_thr={exec_thr:.6f}")

            write_state_snapshot(mode_name, exec_thr, exec_mode, args.adaptive, positions,
                                 paused=sv_state.paused, risk_mode=sv_state.risk_mode)

            # --- Periodic side-bias monitor ---
            # While locked, re-check every _BIAS_LOCKED_EVERY polls so the lock
            # clears within ~30 s of the signal mix normalizing (e.g. after model
            # promotion). While unlocked, check at the normal 10-minute interval.
            _bias_interval = _BIAS_LOCKED_EVERY if _bias_locked else _BIAS_EVERY
            _bias_tick += 1
            if _bias_tick >= _bias_interval and signals_path.exists():
                _bias_tick = 0
                bias_warn, is_biased, _bn, _bl, _bs = check_side_bias(
                    signals_path, min_wall_sec=_start_ts
                )
                _bias_pct = round(max(_bl, _bs) / _bn, 3) if _bn > 0 else 0.0
                _bias_side = "LONG" if _bl >= _bs else "SHORT"
                if is_biased and not _bias_locked:
                    _bias_locked = True
                    log_err(bias_warn)
                    if args.bias_guard:
                        log_err("BIAS_LOCK: new entries suspended until bias normalizes (TP/SL exits still active)")
                    write_heartbeat("side_bias_alert", mode=mode_name, warning=bias_warn,
                                    entries_suspended=args.bias_guard,
                                    recent_long=_bl, recent_short=_bs,
                                    bias_pct=_bias_pct, bias_side=_bias_side, bias_locked=True)
                elif is_biased and _bias_locked:
                    write_heartbeat("side_bias_alert", mode=mode_name, warning=bias_warn,
                                    entries_suspended=args.bias_guard,
                                    recent_long=_bl, recent_short=_bs,
                                    bias_pct=_bias_pct, bias_side=_bias_side, bias_locked=True)
                elif not is_biased and _bias_locked:
                    if _bn < _BIAS_MIN_CLEAR:
                        # Not enough fresh samples -- old signals aged out but no healthy mix
                        # has returned yet. Keep the lock to avoid a spurious "cleared" state.
                        log(f"BIAS_STALE: only {_bn} recent allowed samples "
                            f"(need {_BIAS_MIN_CLEAR}) -- keeping bias lock")
                        log_err(f"BIAS_STALE: only {_bn} recent allowed samples "
                                f"(need {_BIAS_MIN_CLEAR}) -- keeping bias lock")
                        write_heartbeat("side_bias_stale", mode=mode_name,
                                        recent_long=_bl, recent_short=_bs,
                                        sample_count=_bn, min_required=_BIAS_MIN_CLEAR,
                                        bias_locked=True)
                    else:
                        _bias_locked = False
                        log("BIAS_CLEARED: side distribution normalized, resuming new entries")
                        log_err("BIAS_CLEARED: side distribution normalized, resuming new entries")
                        write_heartbeat("side_bias_cleared", mode=mode_name,
                                        recent_long=_bl, recent_short=_bs,
                                        bias_pct=_bias_pct, bias_side=_bias_side, bias_locked=False)

            # --- Wait for the signals file to exist ---
            if not signals_path.exists():
                idle_no_file += 1
                if idle_no_file % max(1, int(30 / max(args.poll, 1))) == 0:
                    log("waiting: live_signals.csv not found")
                write_heartbeat("waiting_for_signals", mode=mode_name)
                time.sleep(args.poll)
                continue

            new_signals = read_recent_signals(signals_path, last_signal_ts, args.tail)
            if not new_signals:
                idle_no_new += 1
                if idle_no_new % max(1, int(60 / max(args.poll, 1))) == 0:
                    log("idle: no new signals")
                write_heartbeat("idle", mode=mode_name, open_positions=len(positions),
                                bias_locked=_bias_locked)
                time.sleep(args.poll)
                continue

            idle_no_file = 0
            idle_no_new = 0

            # --- Process each new signal ---
            eff_notional = args.notional_usdt * _RISK_NOTIONAL_MULT.get(sv_state.risk_mode, 1.0)
            for sig in new_signals:
                last_signal_ts[sig.symbol] = sig.ts

                if not symbol_allowed(sig.symbol, whitelist):
                    log(f"SKIP {sig.symbol} reason=not_whitelisted")
                    continue

                price = sig.price
                if price <= 0 or not math.isfinite(price):
                    log(f"SKIP {sig.symbol} reason=bad_price price={price}")
                    continue

                pos = positions.get(sig.symbol)

                # 1) Exit on TP/SL before any new-entry logic.
                if pos:
                    hit_tp, hit_sl = check_tp_sl(pos, price, args.tp_pct, args.sl_pct)
                    if hit_tp or hit_sl:
                        action = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                        order_id = broker.create_market_order(sig.symbol, action, pos.qty, reduce_only=True)
                        pnl, exit_fill = net_pnl_on_close(pos, price, action,
                                                          args.fee_bps, args.slippage_bps)
                        reason = (f"EXIT_{'TP' if hit_tp else 'SL'} "
                                  f"p={sig.p_meta:.4f} rv={sig.rv_mean:.6f} pnl={pnl:.6f}")
                        record_trade(paper_path, [sig.ts, sig.symbol, action, exit_fill, pos.qty, reason, mode_name, order_id])
                        record_closed_trade(sig.ts, sig.symbol, action, pos.qty, pos.avg, exit_fill, pnl, reason)
                        log(f"TRADE {mode_name} {action} {sig.symbol} qty={pos.qty} px={exit_fill:.8f} order={order_id} {reason}")
                        positions.pop(sig.symbol, None)
                        active_symbols.discard(sig.symbol)
                        _flip_pending.pop(sig.symbol, None)
                        last_fill_time[sig.symbol] = time.time()
                        last_fill_price[sig.symbol] = price
                        continue

                # 1b) V2 time-stop: close positions held >= V2_TIME_STOP_MIN
                # minutes (wall clock). Runs AFTER TP/SL (which keeps priority)
                # and BEFORE the pause gates so a pause cannot trap a position
                # past its time limit. Cleanup mirrors the TP/SL exit exactly.
                if pos and _V2_RISK is not None:
                    try:
                        held_min = _V2_RISK.time_stop_due(sig.symbol)
                    except Exception:
                        held_min = None
                    if held_min is not None:
                        action = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                        order_id = broker.create_market_order(sig.symbol, action, pos.qty, reduce_only=True)
                        pnl, exit_fill = net_pnl_on_close(pos, price, action,
                                                          args.fee_bps, args.slippage_bps)
                        reason = (f"EXIT_TIME held_min={held_min:.1f} "
                                  f"p={sig.p_meta:.4f} rv={sig.rv_mean:.6f} pnl={pnl:.6f}")
                        record_trade(paper_path, [sig.ts, sig.symbol, action, exit_fill, pos.qty, reason, mode_name, order_id])
                        record_closed_trade(sig.ts, sig.symbol, action, pos.qty, pos.avg, exit_fill, pnl, reason)
                        log(f"TRADE {mode_name} {action} {sig.symbol} qty={pos.qty} px={exit_fill:.8f} order={order_id} {reason}")
                        positions.pop(sig.symbol, None)
                        active_symbols.discard(sig.symbol)
                        _flip_pending.pop(sig.symbol, None)
                        last_fill_time[sig.symbol] = time.time()
                        last_fill_price[sig.symbol] = price
                        continue

                # Supervisor pause: allow TP/SL exits above; block all new entries.
                if sv_state.paused:
                    log(f"SKIP {sig.symbol} reason=supervisor_paused")
                    continue

                # V2 entry gates (pause file / daily SL budget / daily loss /
                # daily DD): same semantics as supervisor pause - exits above
                # already ran; scale-ins, flips and fresh entries are blocked.
                if _V2_RISK is not None:
                    try:
                        _v2_blk = _V2_RISK.entry_block_reason()
                    except Exception:
                        _v2_blk = None
                    if _v2_blk:
                        log(f"SKIP {sig.symbol} reason={_v2_blk}")
                        continue

                # 2) Side filter (long_only / short_only / both).
                if not side_allowed(sig.p_meta, args.sides):
                    log(f"SKIP {sig.symbol} reason=side_blocked p={sig.p_meta:.4f} cfg={args.sides}")
                    continue

                # 3) Threshold gate uses executor's pmode (not the writer's mode field).
                ok, why = threshold_pass(sig, exec_thr, exec_mode, args.respect_writer_thr)
                if not ok:
                    log(f"SKIP {sig.symbol} reason={why} p={sig.p_meta:.4f}")
                    continue

                # 4) Volatility guard.
                if abs(sig.rv_mean) > args.rv_max:
                    log(f"SKIP {sig.symbol} reason=rv_guard rv={sig.rv_mean:.6f} abs>{args.rv_max:.6f}")
                    continue

                want = "long" if sig.p_meta >= 0 else "short"
                action_open = "BUY" if want == "long" else "SELL_SHORT"
                now_s = time.time()
                cooldown_ok = (now_s - last_fill_time.get(sig.symbol, 0.0)) >= args.cooldown

                # 5) Concurrency limits.
                if args.one_position and sig.symbol not in active_symbols and len(active_symbols) >= 1:
                    log(f"SKIP {sig.symbol} reason=one_position_active")
                    continue

                if sig.symbol not in active_symbols and len(active_symbols) >= args.max_symbols:
                    log(f"SKIP {sig.symbol} reason=max_symbols({args.max_symbols})")
                    continue

                # 6) Scale-in into an existing same-side position.
                pos = positions.get(sig.symbol)  # re-read in case it changed
                if pos and pos.side == want:
                    # Signal agrees with open position -- reset any stale flip-pending counter.
                    if sig.symbol in _flip_pending:
                        _flip_pending.pop(sig.symbol, None)
                    if not args.scale_in:
                        log(f"SKIP {sig.symbol} reason=already_{pos.side}")
                        continue
                    if args.bias_guard and _bias_locked:
                        log(f"SKIP {sig.symbol} reason=bias_locked(scale_in)")
                        continue
                    if not cooldown_ok:
                        log(f"SKIP {sig.symbol} reason=cooldown_scale")
                        continue
                    # Portfolio cap MUST be checked here too - scale-ins add exposure.
                    exposure = portfolio_exposure(positions)
                    if exposure + eff_notional > args.max_portfolio_usdt:
                        log(f"SKIP {sig.symbol} reason=portfolio_cap_scale "
                            f"exposure={exposure:.2f} cap={args.max_portfolio_usdt:.2f}")
                        continue
                    qty = qty_for(price, eff_notional, args.min_notional, args.min_qty)
                    if qty <= 0:
                        log(f"SKIP {sig.symbol} reason=no_qty_scale")
                        continue
                    order_id = broker.create_market_order(sig.symbol, action_open, qty, reduce_only=False)
                    entry_fill = apply_slippage(price, action_open, args.slippage_bps)
                    new_qty = pos.qty + qty
                    pos.avg = (pos.avg * pos.qty + entry_fill * qty) / new_qty
                    pos.qty = new_qty
                    reason = f"SCALE_IN p={sig.p_meta:.4f} rv={sig.rv_mean:.6f}"
                    record_trade(paper_path, [sig.ts, sig.symbol, action_open, entry_fill, qty, reason, mode_name, order_id])
                    log(f"TRADE {mode_name} {action_open} {sig.symbol} qty={qty} px={entry_fill:.8f} order={order_id} {reason}")
                    last_fill_time[sig.symbol] = now_s
                    last_fill_price[sig.symbol] = price
                    continue

                # 7) Flip: require N consecutive valid opposite-entry candidates before closing.
                #    "Valid" means allow=1 + threshold + RV guard all passed. Noise signals
                #    (allow=0, below threshold, RV fail) do not reset the counter.
                if pos and pos.side != want:
                    confirm_ticks = args.flip_confirm_ticks
                    if confirm_ticks > 0:
                        pending_side, pending_count = _flip_pending.get(sig.symbol, ("", 0))
                        if pending_side != want:
                            # First tick of this flip direction -- start counter.
                            pending_count = 1
                        else:
                            pending_count += 1
                        _flip_pending[sig.symbol] = (want, pending_count)
                        if pending_count < confirm_ticks:
                            log(f"FLIP_PENDING {sig.symbol} want={want} "
                                f"tick={pending_count}/{confirm_ticks} p={sig.p_meta:.4f}")
                            continue
                        # Counter reached -- execute flip, clear pending.
                        _flip_pending.pop(sig.symbol, None)
                    action_close = "SELL" if pos.side == "long" else "BUY_TO_COVER"
                    order_id = broker.create_market_order(sig.symbol, action_close, pos.qty, reduce_only=True)
                    pnl, exit_fill = net_pnl_on_close(pos, price, action_close,
                                                      args.fee_bps, args.slippage_bps)
                    reason = f"FLIP_CLOSE p={sig.p_meta:.4f} rv={sig.rv_mean:.6f} pnl={pnl:.6f}"
                    record_trade(paper_path, [sig.ts, sig.symbol, action_close, exit_fill, pos.qty, reason, mode_name, order_id])
                    record_closed_trade(sig.ts, sig.symbol, action_close, pos.qty, pos.avg, exit_fill, pnl, reason)
                    log(f"TRADE {mode_name} {action_close} {sig.symbol} qty={pos.qty} px={exit_fill:.8f} order={order_id} {reason}")
                    positions.pop(sig.symbol, None)
                    active_symbols.discard(sig.symbol)
                    last_fill_time[sig.symbol] = now_s
                    last_fill_price[sig.symbol] = price
                    if not args.flip_open:
                        continue
                    # Fall through to open the new side.
                    if args.bias_guard and _bias_locked:
                        log(f"SKIP {sig.symbol} reason=bias_locked(flip_open)")
                        continue

                # 8) Cooldown for fresh entries (skip if we just closed and flip_open=True).
                if not cooldown_ok and sig.symbol not in positions:
                    # If we just flipped this tick we *want* to open immediately, so
                    # only enforce cooldown when there was no recent flip-close.
                    last_fill = last_fill_time.get(sig.symbol, 0.0)
                    if last_fill != now_s:
                        log(f"SKIP {sig.symbol} reason=cooldown({args.cooldown:.0f}s)")
                        continue

                # 9) Duplicate-fill guard with relative tolerance (no float == comparison).
                last_px = last_fill_price.get(sig.symbol, 0.0)
                if (prices_close(last_px, price) and
                        (now_s - last_fill_time.get(sig.symbol, 0.0)) < args.cooldown * 2 and
                        sig.symbol not in positions):
                    log(f"SKIP {sig.symbol} reason=dup_price")
                    continue

                # 10) Bias-lock gate: block fresh entries when model is one-sided.
                if args.bias_guard and _bias_locked:
                    log(f"SKIP {sig.symbol} reason=bias_locked(entry)")
                    continue

                # 11) Portfolio cap on fresh entries.
                if sig.symbol not in positions:
                    exposure = portfolio_exposure(positions)
                    if exposure + eff_notional > args.max_portfolio_usdt:
                        log(f"SKIP {sig.symbol} reason=portfolio_cap "
                            f"exposure={exposure:.2f} cap={args.max_portfolio_usdt:.2f}")
                        continue

                qty = qty_for(price, eff_notional, args.min_notional, args.min_qty)
                if qty <= 0:
                    log(f"SKIP {sig.symbol} reason=no_qty notional={eff_notional:.2f} price={price:.8f}")
                    continue

                order_id = broker.create_market_order(sig.symbol, action_open, qty, reduce_only=False)
                entry_fill = apply_slippage(price, action_open, args.slippage_bps)
                reason = f"ENTRY p={sig.p_meta:.4f} rv={sig.rv_mean:.6f} eff_thr={exec_thr:.4f}"
                record_trade(paper_path, [sig.ts, sig.symbol, action_open, entry_fill, qty, reason, mode_name, order_id])
                positions[sig.symbol] = Position(want, qty, entry_fill)
                active_symbols.add(sig.symbol)
                _flip_pending.pop(sig.symbol, None)
                # V2 hook: start the time-stop clock (single entry site - the
                # flip-open path falls through to here too).
                if _V2_RISK is not None:
                    try:
                        _V2_RISK.on_entry(sig.symbol)
                    except Exception:
                        pass
                last_fill_time[sig.symbol] = now_s
                last_fill_price[sig.symbol] = price
                log(f"TRADE {mode_name} {action_open} {sig.symbol} qty={qty} px={entry_fill:.8f} order={order_id} {reason}")
                write_heartbeat("trade", mode=mode_name, symbol=sig.symbol,
                                side=action_open, qty=qty, price=entry_fill)

            time.sleep(args.poll)

        except Exception as exc:
            tb = traceback.format_exc()
            log_err(f"LOOP_ERROR {exc}\n{tb}")
            write_heartbeat("loop_error", mode=mode_name, error=str(exc))
            time.sleep(max(args.poll, 3.0))


if __name__ == "__main__":
    main()
