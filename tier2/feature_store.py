"""
features/feature_store.py

Append-only SQLite store for Tier 2 shadow-mode alpha signals.

Schema (schema_version=2):

    metadata(key TEXT PRIMARY KEY, value TEXT)
    funding_rate(ts, exchange, symbol, rate, interval_hours, mark_px)
    open_interest(ts, exchange, symbol, oi_usd, oi_base, mark_px)

Dedup: timestamps are rounded to the nearest minute before insertion.
       UNIQUE(ts, exchange, symbol) on both data tables causes INSERT OR IGNORE
       to silently drop any row that would duplicate an already-stored snapshot
       (e.g. runner restart within the same minute, or parallel invocations).

DB path: controlled by the TIER2_STORE_PATH env var (default data/tier2_features.db).

This module NEVER writes to logs/live_signals.csv and has no effect on trading.

Schema migration: if the DB contains tables from schema_version 1 (no exchange
column, no UNIQUE index), they are dropped and recreated.  Shadow data is
ephemeral, so data loss on migration is acceptable.

CLI:
    python features/feature_store.py --summary
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# DB path resolution
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_VERSION = 2


def _default_db() -> Path:
    env = os.getenv("TIER2_STORE_PATH", "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else _PROJECT_ROOT / p
    return _PROJECT_ROOT / "data" / "tier2_features.db"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

_LOCKS: Dict[str, threading.Lock] = {}
_META_LOCK = threading.Lock()

_ALLOWED_TABLES = frozenset({"funding_rate", "open_interest"})


def _lock_for(db_path: Path) -> threading.Lock:
    key = str(db_path)
    with _META_LOCK:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _get_schema_version(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
    )
    if cur.fetchone() is None:
        return 0
    try:
        cur = conn.execute("SELECT value FROM metadata WHERE key='schema_version'")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS funding_rate (
            ts             TEXT NOT NULL,
            exchange       TEXT NOT NULL DEFAULT '',
            symbol         TEXT NOT NULL,
            rate           REAL,
            interval_hours REAL,
            mark_px        REAL,
            UNIQUE(ts, exchange, symbol)
        );
        CREATE INDEX IF NOT EXISTS ix_fr_sym_ts
            ON funding_rate (symbol, ts);

        CREATE TABLE IF NOT EXISTS open_interest (
            ts       TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT '',
            symbol   TEXT NOT NULL,
            oi_usd   REAL,
            oi_base  REAL,
            mark_px  REAL,
            UNIQUE(ts, exchange, symbol)
        );
        CREATE INDEX IF NOT EXISTS ix_oi_sym_ts
            ON open_interest (symbol, ts);
    """)
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
        (str(_SCHEMA_VERSION),),
    )
    conn.commit()


def _migrate_if_needed(conn: sqlite3.Connection) -> None:
    version = _get_schema_version(conn)
    if version >= _SCHEMA_VERSION:
        return
    # Pre-v2: no exchange column, no UNIQUE index.
    # Drop and recreate — shadow data is ephemeral.
    conn.executescript("""
        DROP TABLE IF EXISTS funding_rate;
        DROP TABLE IF EXISTS open_interest;
        DROP TABLE IF EXISTS metadata;
    """)
    conn.commit()
    _create_tables(conn)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _minute_ts() -> str:
    """Current UTC time rounded to the minute (used as the dedup key)."""
    dt = datetime.now(timezone.utc)
    return dt.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%MZ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FeatureStore:
    """Thread-safe append-only store for Tier 2 alpha features."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self._db = Path(db_path) if db_path else _default_db()
        self._lock = _lock_for(self._db)
        with self._lock:
            conn = _connect(self._db)
            try:
                _migrate_if_needed(conn)
                _create_tables(conn)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def append_funding_rate(
        self,
        symbol: str,
        rate: Optional[float],
        exchange: str = "",
        interval_hours: Optional[float] = None,
        mark_px: Optional[float] = None,
        ts: Optional[str] = None,
    ) -> bool:
        """Insert a funding-rate snapshot.  Returns False if deduped (already stored)."""
        row = (ts or _minute_ts(), exchange, symbol, rate, interval_hours, mark_px)
        with self._lock:
            conn = _connect(self._db)
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO funding_rate"
                    "(ts,exchange,symbol,rate,interval_hours,mark_px)"
                    " VALUES(?,?,?,?,?,?)",
                    row,
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def append_open_interest(
        self,
        symbol: str,
        oi_usd: Optional[float],
        exchange: str = "",
        oi_base: Optional[float] = None,
        mark_px: Optional[float] = None,
        ts: Optional[str] = None,
    ) -> bool:
        """Insert an OI snapshot.  Returns False if deduped (already stored)."""
        row = (ts or _minute_ts(), exchange, symbol, oi_usd, oi_base, mark_px)
        with self._lock:
            conn = _connect(self._db)
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO open_interest"
                    "(ts,exchange,symbol,oi_usd,oi_base,mark_px)"
                    " VALUES(?,?,?,?,?,?)",
                    row,
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def get_latest(
        self,
        table: str,
        symbol: str,
        n: int = 1,
        exchange: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the last n rows for symbol, newest first."""
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"Unknown table {table!r}")
        clause = "WHERE symbol=?"
        params: list = [symbol]
        if exchange is not None:
            clause += " AND exchange=?"
            params.append(exchange)
        with self._lock:
            conn = _connect(self._db)
            try:
                cur = conn.execute(
                    f"SELECT * FROM {table} {clause} ORDER BY ts DESC LIMIT ?",
                    (*params, n),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            finally:
                conn.close()

    def get_since(
        self,
        table: str,
        symbol: str,
        since_ts: str,
        exchange: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all rows for symbol at or after since_ts (ISO-8601 UTC), oldest first."""
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"Unknown table {table!r}")
        clause = "WHERE symbol=? AND ts>=?"
        params: list = [symbol, since_ts]
        if exchange is not None:
            clause += " AND exchange=?"
            params.append(exchange)
        with self._lock:
            conn = _connect(self._db)
            try:
                cur = conn.execute(
                    f"SELECT * FROM {table} {clause} ORDER BY ts ASC",
                    params,
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def row_counts(self) -> Dict[str, int]:
        with self._lock:
            conn = _connect(self._db)
            try:
                result: Dict[str, int] = {}
                for tbl in sorted(_ALLOWED_TABLES):
                    cur = conn.execute(f"SELECT COUNT(*) FROM {tbl}")
                    result[tbl] = cur.fetchone()[0]
                return result
            finally:
                conn.close()

    def latest_ts_per_table(self) -> Dict[str, Optional[str]]:
        with self._lock:
            conn = _connect(self._db)
            try:
                result: Dict[str, Optional[str]] = {}
                for tbl in sorted(_ALLOWED_TABLES):
                    cur = conn.execute(f"SELECT MAX(ts) FROM {tbl}")
                    row = cur.fetchone()
                    result[tbl] = row[0] if row else None
                return result
            finally:
                conn.close()

    def schema_version(self) -> int:
        with self._lock:
            conn = _connect(self._db)
            try:
                return _get_schema_version(conn)
            finally:
                conn.close()

    def db_path(self) -> Path:
        return self._db


# ---------------------------------------------------------------------------
# CLI  --  python features/feature_store.py --summary
# ---------------------------------------------------------------------------

def _summary() -> None:
    store = FeatureStore()
    counts = store.row_counts()
    latest = store.latest_ts_per_table()
    ver = store.schema_version()

    print(f"Tier 2 Feature Store: {store.db_path()}")
    print(f"Schema version      : {ver}")
    print()
    print(f"  {'Table':<20s} {'Rows':>8s}  {'Latest ts'}")
    print(f"  {'-'*20} {'-'*8}  {'-'*25}")
    for tbl in sorted(_ALLOWED_TABLES):
        n = counts.get(tbl, 0)
        ts = latest.get(tbl) or "n/a"
        print(f"  {tbl:<20s} {n:>8d}  {ts}")


def _cli(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Feature store diagnostics")
    p.add_argument("--summary", action="store_true", help="Print row counts and latest ts per table")
    args = p.parse_args(argv)
    if args.summary:
        _summary()
    else:
        p.print_help()


if __name__ == "__main__":
    _cli(sys.argv[1:])
