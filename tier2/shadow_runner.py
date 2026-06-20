"""
features/shadow_runner.py

Tier 2 shadow-mode data collection loop.

Runs all enabled collectors on a fixed interval and writes collected data to
the feature store (path from TIER2_STORE_PATH, default data/tier2_features.db).
Writes a runner heartbeat every cycle to logs/tier2_runner_heartbeat.json.

NO-AUTH GUARANTEE:
  The ccxt client built by this runner intentionally carries NO API credentials.
  It omits apiKey, secret, and password from the ccxt constructor — these fields
  are never read from .env.  Funding rates and open interest are public endpoints
  on all major perp exchanges.  If a collector accidentally tries to call an
  authenticated endpoint, ccxt will raise AuthenticationError rather than silently
  failing, which makes the gap visible in the heartbeat log.

This process has ZERO influence on live trading:
  - It does not write to logs/live_signals.csv.
  - It does not communicate with live_writer.py or live_executor.py.
  - TIER2_SHADOW_ONLY=1 is enforced at startup; the runner refuses to start
    if it is set to 0, guarding against accidental signal injection.

Usage:
    python features/shadow_runner.py [--interval 60] [--symbols BTCUSDT,ETHUSDT]

Config (read from config/run.json then .env, shell env takes priority):
    TIER2_ENABLED               1/0  — master switch (default 0 in run.json)
    TIER2_SHADOW_ONLY           1/0  — must be 1 (enforced, default 1)
    TIER2_STORE_PATH                 — DB path (default data/tier2_features.db)
    TIER2_COLLECT_INTERVAL_SEC       — seconds between cycles (default 60)
    TIER2_FUNDING_RATE          1/0  — enable funding rate collector (default 1)
    TIER2_OPEN_INTEREST         1/0  — enable open interest collector (default 1)
    DL_SYMBOLS                       — comma-separated symbols (e.g. BTCUSDT,ETHUSDT)
    EXCHANGE_ID                      — ccxt exchange id (default bitget)
    CCXT_TIMEOUT_MS                  — ccxt request timeout ms (default 30000)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

# --- path setup -----------------------------------------------------------
_FEATURES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FEATURES_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- config: run.json first, then .env (so .env always wins) --------------
try:
    from runtime.loader import apply_run_config as _apply_run_config
    _apply_run_config(_PROJECT_ROOT)
except Exception:
    pass

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(_PROJECT_ROOT / ".env", override=True)
except ImportError:
    pass

# --- local imports (after path and config setup) --------------------------
import ccxt

from tier2.feature_store import FeatureStore
from tier2.collectors.funding_rate import FundingRateCollector
from tier2.collectors.open_interest import OpenInterestCollector
from tier2.collectors.quality import QualityChecker

# --------------------------------------------------------------------------

_LOG = logging.getLogger("tier2.runner")
_LOGS = _PROJECT_ROOT / "logs"
_RUNNER_HB = _LOGS / "tier2_runner_heartbeat.json"

_STOP = False


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    val = _env(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes"}


def _build_public_client(exchange_id: str) -> Any:
    """Build a ccxt client for public endpoints only — NO API credentials.

    Intentionally omits apiKey / secret / password from the constructor so
    that any collector that accidentally calls an authenticated endpoint gets
    an AuthenticationError rather than silently using live credentials.
    """
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise ValueError(f"Unknown ccxt exchange: {exchange_id!r}")
    client = cls({
        "enableRateLimit": True,
        "timeout": _env_int("CCXT_TIMEOUT_MS", 30000),
        "options": {"defaultType": "swap"},
        # No apiKey / secret / password — public endpoints only.
    })
    # Explicit runtime assertion: verify no credentials leaked in.
    if getattr(client, "apiKey", None) or getattr(client, "secret", None):
        raise RuntimeError(
            "BUG: ccxt client has credentials after public-only construction. "
            "Tier 2 shadow runner must not hold API keys."
        )
    return client


def _write_runner_heartbeat(
    cycle: int,
    ok: bool,
    collectors_run: int,
    rows_total: int,
    db_counts: dict,
    error: str = "",
) -> None:
    _LOGS.mkdir(parents=True, exist_ok=True)
    payload = {
        "runner": "tier2_shadow",
        "ok": ok,
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "collectors_run": collectors_run,
        "rows_this_cycle": rows_total,
        "db_row_counts": db_counts,
        "error": error,
    }
    tmp = _RUNNER_HB.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(_RUNNER_HB)


def _handle_stop(signum, frame):  # noqa: ANN001
    global _STOP
    _LOG.info("received signal %s, stopping after current cycle", signum)
    _STOP = True


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tier 2 shadow-mode data collection loop",
    )
    p.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Collection interval in seconds (overrides TIER2_COLLECT_INTERVAL_SEC)",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol list (overrides DL_SYMBOLS)",
    )
    p.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (overrides TIER2_STORE_PATH; "
             "default: data/tier2_features.db)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Safety gate — refuse to start if shadow mode is disabled.
    if not _env_bool("TIER2_SHADOW_ONLY", default=True):
        _LOG.error(
            "TIER2_SHADOW_ONLY=0 detected.  The shadow runner must not run with "
            "shadow mode disabled (it has no signal-injection path anyway, but this "
            "guard prevents accidental misconfiguration).  Set TIER2_SHADOW_ONLY=1 "
            "and restart."
        )
        sys.exit(1)

    if not _env_bool("TIER2_ENABLED", default=True):
        _LOG.info("TIER2_ENABLED=0 — shadow runner disabled, exiting.")
        sys.exit(0)

    # Resolve symbols.
    raw_syms = args.symbols or _env("DL_SYMBOLS", "BTCUSDT,ETHUSDT")
    symbols: List[str] = [s.strip() for s in raw_syms.split(",") if s.strip()]
    if not symbols:
        _LOG.error("No symbols configured.  Set DL_SYMBOLS or pass --symbols.")
        sys.exit(1)

    # Resolve collection interval.
    interval_sec = args.interval or _env_int("TIER2_COLLECT_INTERVAL_SEC", 60)
    interval_sec = max(10, interval_sec)

    # Resolve exchange.
    exchange_id = _env("EXCHANGE_ID", "bitget")

    _LOG.info(
        "Tier 2 shadow runner starting — exchange=%s symbols=%s interval=%ds db=%s",
        exchange_id,
        symbols,
        interval_sec,
        args.db or "data/tier2_features.db",
    )

    # Build exchange client (public, no auth).
    try:
        exchange = _build_public_client(exchange_id)
    except Exception as exc:
        _LOG.error("Failed to build exchange client: %s", exc)
        sys.exit(1)

    # Open feature store.
    # Priority: --db CLI arg > TIER2_STORE_PATH env var > default (data/tier2_features.db).
    # FeatureStore._default_db() already reads TIER2_STORE_PATH, so passing None
    # here activates env-var control.  --db overrides everything.
    store = FeatureStore(db_path=args.db)
    _LOG.info("Feature store: %s  (override via --db or TIER2_STORE_PATH)", store.db_path())

    # Register collectors and quality checker.
    collectors = [
        FundingRateCollector(),
        OpenInterestCollector(),
    ]
    quality = QualityChecker()
    _LOG.info(
        "Collectors registered: %s",
        [c.name for c in collectors],
    )

    # Install signal handlers.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_stop)
        except (OSError, ValueError):
            pass  # not supported on Windows for SIGTERM in some contexts

    cycle = 0
    while not _STOP:
        cycle_start = time.monotonic()
        cycle += 1
        collectors_run = 0
        rows_total = 0
        cycle_error = ""

        try:
            for collector in collectors:
                if _STOP:
                    break
                ok = collector.run_once(exchange, symbols, store)
                if ok:
                    collectors_run += 1
            db_counts = store.row_counts()
            rows_total = sum(db_counts.values())
            # Quality check after every cycle (writes tier2_quality_heartbeat.json).
            quality.run_check(store, symbols)
        except Exception as exc:
            cycle_error = str(exc)
            _LOG.error("Cycle %d error: %s", cycle, exc)
            db_counts = {}

        _write_runner_heartbeat(
            cycle=cycle,
            ok=not cycle_error,
            collectors_run=collectors_run,
            rows_total=rows_total,
            db_counts=db_counts,
            error=cycle_error,
        )

        elapsed = time.monotonic() - cycle_start
        _LOG.info(
            "cycle=%d collectors_run=%d db_rows=%s elapsed=%.1fs",
            cycle,
            collectors_run,
            db_counts,
            elapsed,
        )

        sleep_for = max(0.0, interval_sec - elapsed)
        deadline = time.monotonic() + sleep_for
        while not _STOP and time.monotonic() < deadline:
            time.sleep(0.5)

    _LOG.info("shadow runner stopped after %d cycles.", cycle)


if __name__ == "__main__":
    main()
