"""
tier2/influence.py
==================
Phase 2A: Shadow Influence Engine.

Pure functions only - no state, no I/O, no executor/writer mutation.
Input:  symbol, side, p_centered, current threshold, latest funding/OI row.
Output: InfluenceResult (recommended threshold + reason).

Mode is controlled by TIER2_INFLUENCE_MODE in config/run.json:
  "off"    - default; this module computes nothing (fast-path return)
  "shadow" - compute and log to logs/tier2_influence_shadow.csv only
  "paper"  - future; requires backtest approval (n>=200, stable improvement)
  "live"   - blocked; not implemented

Safety rules (v0):
  - Only TIGHTEN, never loosen. delta is always >= 0.
  - Negative FR (shorts pay longs) -> raise SHORT threshold.
  - Positive FR (longs pay shorts) -> raise LONG threshold.
  - Max delta capped at TIER2_INFLUENCE_MAX_DELTA (default 0.02).
  - No trade is ever made easier by Tier 2.

Usage (shadow mode, called from live_writer.py):
    from tier2.influence import compute_influence, log_influence
    result = compute_influence(symbol, side, p_centered, base_thr, fr_row, oi_row)
    log_influence(result)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOGS = _PROJECT_ROOT / "logs"
_SHADOW_LOG = _LOGS / "tier2_influence_shadow.csv"
_SHADOW_HEADER = [
    "ts", "symbol", "side", "original_thr", "shadow_thr",
    "delta", "funding_rate", "oi_usd", "reason",
]

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InfluenceResult:
    ts: str
    symbol: str
    side: str              # "long" or "short"
    original_thr: float
    shadow_thr: float      # what threshold WOULD be in paper/live mode
    delta: float           # shadow_thr - original_thr (always >= 0 in v0)
    funding_rate: Optional[float]
    oi_usd: Optional[float]
    reason: str
    active: bool = False   # True only when mode == "paper" or "live" (not yet)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _mode() -> str:
    return os.getenv("TIER2_INFLUENCE_MODE", "off").strip().lower()


def _max_delta() -> float:
    try:
        return float(os.getenv("TIER2_INFLUENCE_MAX_DELTA", "0.02"))
    except (TypeError, ValueError):
        return 0.02


def _fr_threshold() -> float:
    """Funding rate magnitude above which we apply influence."""
    try:
        return float(os.getenv("TIER2_INFLUENCE_FR_THRESHOLD", "0.0005"))
    except (TypeError, ValueError):
        return 0.0005


# ---------------------------------------------------------------------------
# Core pure function
# ---------------------------------------------------------------------------

def compute_influence(
    symbol: str,
    side: str,
    p_centered: float,
    base_thr: float,
    fr_row: Optional[Dict[str, Any]],
    oi_row: Optional[Dict[str, Any]],
) -> InfluenceResult:
    """
    Compute the shadow threshold adjustment for a signal.

    Args:
        symbol:      compact symbol (e.g. "BTCUSDT")
        side:        "long" or "short"
        p_centered:  p_meta from writer (signed distance from 0.5)
        base_thr:    current effective threshold
        fr_row:      latest funding_rate row from FeatureStore (or None)
        oi_row:      latest open_interest row from FeatureStore (or None)

    Returns:
        InfluenceResult with delta=0 if no adjustment warranted.
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    fr = _safe_float(fr_row.get("rate") if fr_row else None)
    oi = _safe_float(oi_row.get("oi_usd") if oi_row else None)
    fr_thresh = _fr_threshold()
    max_delta = _max_delta()

    delta = 0.0
    reason = "no_adjustment"

    if fr is not None and abs(fr) >= fr_thresh:
        if fr > 0 and side == "long":
            # Positive FR: longs paying shorts -> crowded long side -> risk of unwind.
            # Tighten LONG threshold proportionally, capped at max_delta.
            raw = min(abs(fr) * 10, max_delta)
            delta = round(raw, 4)
            reason = f"positive_fr={fr:.5f}_tighten_long"

        elif fr < 0 and side == "short":
            # Negative FR: shorts paying longs -> crowded short side -> squeeze risk.
            # Tighten SHORT threshold proportionally, capped at max_delta.
            raw = min(abs(fr) * 10, max_delta)
            delta = round(raw, 4)
            reason = f"negative_fr={fr:.5f}_tighten_short"

    shadow_thr = round(base_thr + delta, 6)

    return InfluenceResult(
        ts=ts,
        symbol=symbol,
        side=side,
        original_thr=base_thr,
        shadow_thr=shadow_thr,
        delta=delta,
        funding_rate=fr,
        oi_usd=oi,
        reason=reason,
        active=False,  # v0: never active; log only
    )


# ---------------------------------------------------------------------------
# Shadow logger
# ---------------------------------------------------------------------------

def log_influence(result: InfluenceResult) -> None:
    """
    Append one row to the shadow log CSV.
    Called only when TIER2_INFLUENCE_MODE == "shadow".
    Never raises - silently skips on I/O error.
    """
    try:
        _LOGS.mkdir(parents=True, exist_ok=True)
        write_header = not _SHADOW_LOG.exists()
        with open(_SHADOW_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(_SHADOW_HEADER)
            w.writerow([
                result.ts,
                result.symbol,
                result.side,
                result.original_thr,
                result.shadow_thr,
                result.delta,
                result.funding_rate if result.funding_rate is not None else "",
                result.oi_usd if result.oi_usd is not None else "",
                result.reason,
            ])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Convenience: run one signal through the full shadow pipeline
# ---------------------------------------------------------------------------

_store_singleton: Any = None


def _get_store() -> Any:
    """Lazy singleton FeatureStore - created once, reused every tick."""
    global _store_singleton
    if _store_singleton is None:
        try:
            from tier2.feature_store import FeatureStore
            _store_singleton = FeatureStore()
        except Exception:
            pass
    return _store_singleton


def shadow_evaluate(
    symbol: str,
    side: str,
    p_centered: float,
    base_thr: float,
    store: Any = None,
) -> InfluenceResult:
    """
    Fetch latest Tier 2 rows from store, compute influence, log if shadow mode.
    Safe to call unconditionally - fast-paths when mode == "off".
    """
    mode = _mode()
    if mode == "off":
        return InfluenceResult(
            ts=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            symbol=symbol, side=side,
            original_thr=base_thr, shadow_thr=base_thr,
            delta=0.0, funding_rate=None, oi_usd=None,
            reason="mode_off", active=False,
        )

    active_store = store or _get_store()
    fr_rows = []
    oi_rows = []
    try:
        if active_store:
            fr_rows = active_store.get_latest("funding_rate", symbol, n=1)
            oi_rows = active_store.get_latest("open_interest", symbol, n=1)
    except Exception:
        pass

    result = compute_influence(
        symbol=symbol,
        side=side,
        p_centered=p_centered,
        base_thr=base_thr,
        fr_row=fr_rows[0] if fr_rows else None,
        oi_row=oi_rows[0] if oi_rows else None,
    )

    if mode == "shadow":
        log_influence(result)

    return result


# ---------------------------------------------------------------------------
# Backtest comparison helper
# ---------------------------------------------------------------------------

def backtest_shadow_impact(joined_csv: Path) -> Dict[str, Any]:
    """
    Read the bt_tier2_join.csv produced by backtest_join.py and compute
    what would have happened if the shadow thresholds had been applied.

    Returns a summary dict. Requires n >= 200 matched rows for conclusions.
    """
    import csv as _csv

    rows = []
    if not joined_csv.exists():
        return {"error": "joined CSV not found", "path": str(joined_csv)}

    with open(joined_csv, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    if not rows:
        return {"error": "empty joined CSV"}

    n = len(rows)

    # Detect if p_meta is present - closed-trade CSVs don't carry it,
    # so the computation would be meaningless (p=0 -> everything "skipped").
    has_p_meta = any(r.get("p_meta") not in (None, "", "0", "0.0") for r in rows[:20])
    if not has_p_meta:
        return {
            "n_matched": n,
            "sufficient": n >= 200,
            "note": "p_meta not in joined CSV - run backtest_join.py with signal-level join for impact analysis",
            "would_trade": None,
            "would_skip": None,
            "pnl_kept": None,
            "pnl_skipped": None,
        }

    would_trade = 0
    would_skip  = 0
    pnl_kept    = 0.0
    pnl_skipped = 0.0
    fr_thresh   = _fr_threshold()
    max_delta   = _max_delta()

    for r in rows:
        try:
            side = r.get("closed_side", "")
            norm_side = "long" if side in ("SELL", "long") else "short"
            base_thr = float(r.get("thr", 0.08) or 0.08)
            p = abs(float(r.get("p_meta", 0) or 0))
            fr = float(r["funding_rate"]) if r.get("funding_rate") else None
            pnl = float(r.get("realized_pnl", 0) or 0)

            delta = 0.0
            if fr is not None and abs(fr) >= fr_thresh:
                if fr > 0 and norm_side == "long":
                    delta = min(abs(fr) * 10, max_delta)
                elif fr < 0 and norm_side == "short":
                    delta = min(abs(fr) * 10, max_delta)

            shadow_thr = base_thr + delta
            if p >= shadow_thr:
                would_trade += 1
                pnl_kept += pnl
            else:
                would_skip += 1
                pnl_skipped += pnl
        except (ValueError, KeyError):
            continue

    return {
        "n_matched": n,
        "sufficient": n >= 200,
        "would_trade": would_trade,
        "would_skip": would_skip,
        "pnl_kept": round(pnl_kept, 4),
        "pnl_skipped": round(pnl_skipped, 4),
        "pnl_net_delta": round(pnl_kept - (pnl_kept + pnl_skipped), 4),
        "skip_rate_pct": round(would_skip / n * 100, 1) if n else 0,
    }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
