"""
features/collectors/quality.py

Tier 2 data quality checker.

Runs after each shadow collection cycle and checks:
  1. Stale data   — latest row for a (table, symbol) is older than STALE_S
  2. Missing data — expected symbol has zero rows in the store
  3. Null streak  — last NULL_WINDOW rows for a symbol all have value=NULL
  4. Zero streak  — last NULL_WINDOW rows for a symbol all have value=0.0
                    (advisory only — legitimate on some exchanges)

Writes an atomic heartbeat to logs/tier2_quality_heartbeat.json.
Has no effect on trading.

Config:
    TIER2_STALE_THRESHOLD_S  — stale threshold in seconds (default 300 = 5 min)
    TIER2_NULL_WINDOW        — rows to inspect for null/zero streaks (default 5)
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_LOG = logging.getLogger("tier2.quality")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS = _PROJECT_ROOT / "logs"
_QUALITY_HB = _LOGS / "tier2_quality_heartbeat.json"

UTC = timezone.utc

# Table → column name that holds the primary numeric value.
# open_interest uses oi_base because Bitget's OI endpoint returns the base-coin
# amount (openInterestAmount) but not the USD value — so oi_base is always
# populated, oi_usd is derived secondarily.
_VALUE_COL = {
    "funding_rate": "rate",
    "open_interest": "oi_base",
}


def _stale_s() -> int:
    try:
        return int(os.getenv("TIER2_STALE_THRESHOLD_S", "300"))
    except (TypeError, ValueError):
        return 300


def _null_window() -> int:
    try:
        return int(os.getenv("TIER2_NULL_WINDOW", "5"))
    except (TypeError, ValueError):
        return 5


def _age_s(ts_str: str) -> float:
    """Return seconds since ts_str (ISO-8601 UTC minute bucket)."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return (datetime.now(UTC) - dt).total_seconds()
    except Exception:
        return float("inf")


class QualityChecker:
    """Runs data quality checks against the Tier 2 feature store."""

    def check(
        self,
        store: Any,
        symbols: List[str],
    ) -> Dict[str, Any]:
        stale_thresh = _stale_s()
        null_win = _null_window()
        issues: List[str] = []
        advisories: List[str] = []
        per_table: Dict[str, Dict[str, Any]] = {}

        for table, val_col in _VALUE_COL.items():
            t_issues: List[str] = []
            t_advisories: List[str] = []
            t_sym: Dict[str, Any] = {}

            for symbol in symbols:
                rows = store.get_latest(table, symbol, n=null_win)

                if not rows:
                    t_issues.append(f"{symbol}: no data")
                    t_sym[symbol] = {"status": "missing"}
                    continue

                latest_ts = rows[0].get("ts", "")
                age = _age_s(latest_ts)
                sym_status: Dict[str, Any] = {
                    "latest_ts": latest_ts,
                    "age_s": int(age),
                }

                # --- stale check ---
                if age > stale_thresh:
                    t_issues.append(
                        f"{symbol}: stale ({int(age)}s > {stale_thresh}s)"
                    )
                    sym_status["stale"] = True

                # --- null streak (hard issue) ---
                vals = [r.get(val_col) for r in rows]
                null_count = sum(1 for v in vals if v is None)
                if null_count == len(rows):
                    t_issues.append(f"{symbol}: all-null streak ({null_win} rows)")
                    sym_status["all_null"] = True
                elif null_count > 0:
                    sym_status["null_count"] = null_count

                # --- zero streak (advisory) ---
                zero_count = sum(1 for v in vals if v is not None and v == 0.0)
                if zero_count == len(rows) and null_count == 0:
                    t_advisories.append(
                        f"{symbol}: all-zero streak ({null_win} rows) — may be normal"
                    )
                    sym_status["all_zero"] = True

                t_sym[symbol] = sym_status

            per_table[table] = {
                "issues": t_issues,
                "advisories": t_advisories,
                "symbols": t_sym,
            }
            issues.extend(t_issues)
            advisories.extend(t_advisories)

        return {
            "ok": len(issues) == 0,
            "ts": datetime.now(UTC).isoformat(),
            "stale_threshold_s": stale_thresh,
            "null_window": null_win,
            "issues": issues,
            "advisories": advisories,
            "per_table": per_table,
        }

    def run_check(self, store: Any, symbols: List[str]) -> Dict[str, Any]:
        """Run checks and write heartbeat.  Never raises."""
        try:
            result = self.check(store, symbols)
        except Exception as exc:
            _LOG.warning("quality check error: %s", exc)
            result = {
                "ok": False,
                "ts": datetime.now(UTC).isoformat(),
                "issues": [f"checker error: {exc}"],
                "advisories": [],
                "per_table": {},
            }
        self._write_heartbeat(result)
        if not result["ok"]:
            _LOG.warning(
                "Tier 2 quality issues: %s",
                "; ".join(result.get("issues", [])),
            )
        return result

    def _write_heartbeat(self, result: Dict[str, Any]) -> None:
        _LOGS.mkdir(parents=True, exist_ok=True)
        tmp = _QUALITY_HB.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        tmp.replace(_QUALITY_HB)
