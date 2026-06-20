"""
async_scanner.py

Async, multi-symbol scanner. The pipeline:
  1. Build a universe of Bitget USDT-M perp symbols (top-N by quote volume)
  2. Sanitise each symbol with a probe fetch
  3. Fetch OHLCV in parallel
  4. Run feature build + scoring across a process pool
  5. Optionally fetch cross-venue last prices

Output:
  - JSONL stream to logs/scan.jsonl (one line per scored symbol)
  - Per-symbol dicts returned from run_scan()

Changelog vs the previous version
---------------------------------
- bool(int(env)) replaced everywhere with a robust _env_bool helper.
  The old pattern crashed at import if you set QUIET_MODE=true etc.,
  taking down the whole scanner because of an env typo.
- All asyncio.gather / as_completed loops now have wall-clock timeouts.
  The previous version would stall indefinitely if any single exchange
  call hung. There are now three timeouts:
      SCAN_SANITIZE_TIMEOUT_SEC (default 30)
      SCAN_FETCH_TIMEOUT_SEC    (default 60)
      SCAN_CROSS_TIMEOUT_SEC    (default 10)
- Failures are now logged with reason. Silent dropping of symbols and
  silent except clauses are replaced with logging that includes which
  symbol, which phase, and what went wrong. The behaviour is the same
  (errors don't crash the scan) but the operator can see what failed.
- Env vars read at call time, not import time. Same fix as risk_engine
  and trade.py.
- Cached client construction. Cross-venue clients used to be built and
  torn down PER CALL PER VENUE - with 25 results and 4 venues that's
  100 client-builds per scan. Now they're cached and reused.
- MODEL_PATH default now reflects the project layout. The old default
  pointed to models/model.pkl which didn't exist; users without
  MODEL_PATH set saw an empty scan with no error.
- SCAN_OUT is opened ONCE before the loop, not once per result.
- Cross-venue symbol resolution made less fragile: tries multiple known
  shapes (BTC/USDT:USDT -> BTC/USDT -> BTCUSDT) in order.
- Bare exception handlers got the contents of the exception logged. The
  scan still doesn't crash on a single bad symbol, but the operator can
  diagnose recurring failures.
- Cleaner shutdown: all cached clients can be released via shutdown().
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import ccxt.async_support as ccxt_async
import joblib

from features import build_features

_LOG = logging.getLogger("scanner")


# ---------------------------------------------------------------------------
# Env helpers (read at call time)
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    """Robust env -> bool. Accepts 1/true/yes/y/on (case-insensitive).

    Replaces the old `bool(int(os.getenv(...)))` pattern which crashed
    at import time if the value was non-numeric (e.g. 'true').
    """
    val = _env_str(name, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: str = "") -> List[str]:
    raw = _env_str(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Config getters - call these at function-entry, not module-load
# ---------------------------------------------------------------------------

def _cfg_timeframe() -> str:           return _env_str("TIMEFRAME", "5m")
def _cfg_lookback() -> int:            return _env_int("LOOKBACK_CANDLES", 2000)
def _cfg_topn() -> int:                return _env_int("SCAN_TOPN", 25)
def _cfg_min_notional() -> float:      return _env_float("SCAN_MIN_NOTIONAL", 100_000.0)
def _cfg_whitelist() -> List[str]:     return _env_list("SYMBOL_WHITELIST")
def _cfg_sandbox() -> bool:            return _env_bool("BITGET_SANDBOX", False)
def _cfg_model_path() -> str:
    # NEW DEFAULT: the project uses model_artifacts/, not models/.
    # Old default 'models/model.pkl' silently failed because that path
    # doesn't exist in the standard layout.
    return _env_str("MODEL_PATH", "model_artifacts/model.pkl")
def _cfg_http_workers() -> int:        return _env_int("HTTP_WORKERS", 16)
def _cfg_sanitize_workers() -> int:    return _env_int("SANITIZE_WORKERS", 16)
def _cfg_cross_venues() -> List[str]:
    return _env_list("ANALYZE_EXCHANGES", "binance,bybit,bitget,mexc")
def _cfg_fetch_cross() -> bool:        return _env_bool("FETCH_CROSS_TICKERS", False)
def _cfg_cpu_workers() -> int:         return _env_int("CPU_WORKERS", os.cpu_count() or 2)
def _cfg_runtime_log() -> str:         return _env_str("RUNTIME_LOG", "logs/runtime.log")
def _cfg_scan_out() -> str:            return _env_str("SCAN_OUT", "logs/scan.jsonl")
def _cfg_quiet() -> bool:              return _env_bool("QUIET_MODE", True)
def _cfg_spinner() -> bool:            return _env_bool("SPINNER", True)
def _cfg_color() -> bool:              return _env_bool("COLOR", True)
def _cfg_runtime_throttle_ms() -> int: return _env_int("RUNTIME_THROTTLE_MS", 800)
def _cfg_sanitize_timeout() -> float:  return _env_float("SCAN_SANITIZE_TIMEOUT_SEC", 30.0)
def _cfg_fetch_timeout() -> float:     return _env_float("SCAN_FETCH_TIMEOUT_SEC", 60.0)
def _cfg_cross_timeout() -> float:     return _env_float("SCAN_CROSS_TIMEOUT_SEC", 10.0)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


_last_write_ms = 0.0


def write_runtime_progress(done: int, total: int, last_symbol: str, phase: str = "scan") -> None:
    """Throttled progress writer. Honours RUNTIME_THROTTLE_MS."""
    global _last_write_ms
    now = time.time() * 1000.0
    if now - _last_write_ms < _cfg_runtime_throttle_ms():
        return
    _last_write_ms = now
    runtime_log = _cfg_runtime_log()
    _ensure_dir(runtime_log)
    try:
        with open(runtime_log, "a", encoding="utf-8") as f:
            f.write(f"{_ts_utc()} | {phase} {done}/{total} | {last_symbol}\n")
    except Exception as e:
        _LOG.debug("write_runtime_progress: %s", e)
    if not _cfg_quiet():
        print(f"{_ts_utc()} | {phase} {done}/{total} | {last_symbol}")


def _c(txt: str, code: str) -> str:
    if not _cfg_color():
        return txt
    return f"\033[{code}m{txt}\033[0m"


_SPIN = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


def spinner_batch(done: int, total: int, last: str, phase: str = "scan") -> None:
    """Optional spinner. Skipped entirely when QUIET_MODE or SPINNER=0."""
    if not _cfg_spinner() or _cfg_quiet():
        return
    spin = next(_SPIN)
    line = f"\r\033[2K{_c(spin, '90')} {_c(phase, '36')} {done}/{total} {_c(last, '33')}"
    print(line, end="", flush=True)


# ---------------------------------------------------------------------------
# Bitget client (uncached - opened/closed per phase)
# ---------------------------------------------------------------------------

async def _bitget_client() -> "ccxt_async.Exchange":
    ex = ccxt_async.bitget({
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap", "defaultSubType": "linear", "defaultSettle": "USDT"},
    })
    try:
        ex.set_sandbox_mode(_cfg_sandbox())
    except Exception as e:
        _LOG.debug("bitget set_sandbox_mode failed: %s", e)
    try:
        await ex.load_markets()
    except Exception:
        try:
            await ex.close()
        except Exception:
            pass
        raise
    return ex


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------

async def build_universe_bitget(topn: int, min_qvol: float) -> List[str]:
    """Top-N USDT-M linear perps by 24h quote volume."""
    ex = await _bitget_client()
    try:
        tickers = await ex.fetch_tickers()
        cands: List[Tuple[str, float]] = []
        for sym, m in (ex.markets or {}).items():
            if not m.get("contract", False):
                continue
            if (m.get("settle") or "").upper() != "USDT":
                continue
            if m.get("linear") is False:
                continue
            t = tickers.get(sym, {}) or {}
            last = t.get("last")
            qvol = t.get("quoteVolume")
            if qvol is None:
                base_vol = t.get("baseVolume")
                if base_vol is not None and last:
                    try:
                        qvol = float(base_vol) * float(last)
                    except (TypeError, ValueError):
                        qvol = 0.0
                else:
                    info = t.get("info", {}) or {}
                    try:
                        qvol = float(info.get("usdtVol") or 0.0)
                    except (TypeError, ValueError):
                        qvol = 0.0
            try:
                qvol_f = float(qvol or 0.0)
            except (TypeError, ValueError):
                qvol_f = 0.0
            if last and qvol_f >= min_qvol:
                cands.append((sym, qvol_f))
        cands.sort(key=lambda x: x[1], reverse=True)
        out = [s for s, _ in cands[:topn]]
        _LOG.info("universe: %d candidates met min_qvol=%.0f, kept top %d",
                  len(cands), min_qvol, len(out))
        return out
    finally:
        try:
            await ex.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OHLCV fetch with pagination
# ---------------------------------------------------------------------------

async def fetch_ohlcv_async(ex: "ccxt_async.Exchange", symbol: str,
                             timeframe: str, want: int) -> List[List[float]]:
    """Page through fetch_ohlcv until we have `want` bars."""
    per = 1000
    ms_per_bar = int(ex.parse_timeframe(timeframe) * 1000)
    end_ms = ex.milliseconds()
    start_ms = end_ms - (want + 200) * ms_per_bar
    since = start_ms

    rows: List[List[float]] = []
    last_seen_ts: Optional[int] = None
    pages = 0
    max_pages = 2000

    while len(rows) < want and pages < max_pages:
        part = await ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=per)
        if not part:
            break
        if last_seen_ts is not None and part[-1][0] <= last_seen_ts:
            break
        rows += part
        last_seen_ts = part[-1][0]
        since = last_seen_ts + ms_per_bar
        pages += 1
        if since >= end_ms - ms_per_bar:
            break
        await asyncio.sleep((ex.rateLimit or 250) / 1000)

    if not rows:
        # Last-resort: fetch the most recent `want` bars without pagination.
        rows = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(1000, want))
    return rows


# ---------------------------------------------------------------------------
# Symbol sanitisation
# ---------------------------------------------------------------------------

async def _probe_one(ex: "ccxt_async.Exchange", symbol: str, timeframe: str,
                      sem: asyncio.Semaphore) -> Tuple[str, bool, str]:
    """Returns (symbol, ok, reason)."""
    async with sem:
        try:
            await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            return symbol, True, ""
        except Exception as e:
            return symbol, False, type(e).__name__


async def sanitize_symbols(symbols: List[str], timeframe: str,
                            workers: int, timeout_sec: float) -> List[str]:
    """Drop symbols whose 1-bar fetch fails. Bounded by `timeout_sec`.

    Failures are LOGGED (with reason) instead of being silently ignored.
    """
    if not symbols:
        return []
    ex = await _bitget_client()
    sem = asyncio.Semaphore(max(1, workers))
    try:
        tasks = [asyncio.create_task(_probe_one(ex, s, timeframe, sem)) for s in symbols]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            _LOG.warning("sanitize: timed out after %.1fs; cancelling probes", timeout_sec)
            for t in tasks:
                if not t.done():
                    t.cancel()
            results = []
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        results.append(t.result())
                    except Exception as e:
                        results.append(("?", False, type(e).__name__))

        kept: List[str] = []
        failed: List[Tuple[str, str]] = []
        for r in results:
            if isinstance(r, tuple) and len(r) == 3:
                sym, ok, reason = r
                if ok:
                    kept.append(sym)
                else:
                    failed.append((sym, reason))
            else:
                failed.append(("?", type(r).__name__))
        if failed:
            _LOG.info("sanitize: dropped %d symbol(s); first few: %s",
                      len(failed), failed[:3])
        kept_set = set(kept)
        return [s for s in symbols if s in kept_set]
    finally:
        try:
            await ex.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Cross-venue last prices (with caching)
# ---------------------------------------------------------------------------

_CROSS_CLIENTS: Dict[str, "ccxt_async.Exchange"] = {}


async def _get_or_open_venue(ex_id: str) -> Optional["ccxt_async.Exchange"]:
    """Cached venue client. Returns None if the exchange isn't supported."""
    if ex_id in _CROSS_CLIENTS:
        return _CROSS_CLIENTS[ex_id]
    cls = getattr(ccxt_async, ex_id, None)
    if cls is None:
        _LOG.debug("cross-venue: unknown exchange %s", ex_id)
        return None
    opts: Dict[str, Any] = {"enableRateLimit": True, "timeout": 15000}
    if ex_id in ("bybit", "bitget", "mexc"):
        opts["options"] = {"defaultType": "swap"}
    ex = cls(opts)
    try:
        await ex.load_markets()
    except Exception as e:
        _LOG.debug("cross-venue %s load_markets failed: %s", ex_id, e)
        try:
            await ex.close()
        except Exception:
            pass
        return None
    _CROSS_CLIENTS[ex_id] = ex
    return ex


def _candidate_symbols_for_cross(base_symbol: str) -> List[str]:
    """Generate plausible symbol forms to try across venues.

    Order:
      1. The unified ccxt form passed in (e.g. 'BTC/USDT:USDT')
      2. Without the settle suffix    (e.g. 'BTC/USDT')
      3. Concatenated form            (e.g. 'BTCUSDT')
    """
    forms = [base_symbol]
    no_settle = base_symbol.split(":", 1)[0] if ":" in base_symbol else base_symbol
    if no_settle != base_symbol:
        forms.append(no_settle)
    if "/" in no_settle:
        forms.append(no_settle.replace("/", ""))
    seen, out = set(), []
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


async def cross_venue_last(base_symbol: str, venues: List[str],
                            timeout_sec: float = 10.0) -> Dict[str, Optional[float]]:
    """Fetch last price for `base_symbol` from each venue.

    Returns {venue: price_or_None}. Bounded by `timeout_sec` total.
    Uses cached clients (call shutdown() on exit to free them).
    """
    results: Dict[str, Optional[float]] = {v: None for v in venues}

    async def one(ex_id: str) -> None:
        ex = await _get_or_open_venue(ex_id)
        if ex is None:
            return
        try:
            chosen = None
            for cand in _candidate_symbols_for_cross(base_symbol):
                if cand in (ex.symbols or []):
                    chosen = cand
                    break
            if chosen is None:
                _LOG.debug("cross-venue %s: no symbol form matched for %s", ex_id, base_symbol)
                return
            t = await ex.fetch_ticker(chosen)
            last = t.get("last") if isinstance(t, dict) else None
            if last is not None:
                try:
                    val = float(last)
                    if math.isfinite(val) and val > 0:
                        results[ex_id] = val
                except (TypeError, ValueError):
                    pass
        except Exception as e:
            _LOG.debug("cross-venue %s fetch failed: %s", ex_id, e)

    tasks = [asyncio.create_task(one(v)) for v in venues]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                timeout=timeout_sec)
    except asyncio.TimeoutError:
        _LOG.debug("cross-venue: timed out after %.1fs", timeout_sec)
        for t in tasks:
            if not t.done():
                t.cancel()
    return results


async def shutdown() -> None:
    """Release any cached cross-venue clients. Call on bot shutdown."""
    for ex_id, ex in list(_CROSS_CLIENTS.items()):
        try:
            await ex.close()
        except Exception as e:
            _LOG.debug("shutdown %s: %s", ex_id, e)
    _CROSS_CLIENTS.clear()


# ---------------------------------------------------------------------------
# CPU worker: feature build + scoring
# ---------------------------------------------------------------------------
# These are module-level globals in the WORKER processes (each worker has
# its own). The model is loaded lazily on first use, then reused for the
# life of the worker process.

_MODEL = None
_FCOLS: Optional[List[str]] = None


def _cpu_features_and_score(symbol: str, rows: List[List[float]],
                             model_path: str) -> Optional[Dict[str, Any]]:
    """Build features and score one symbol.

    Returns either:
      - a result dict on success
      - {"symbol": ..., "error": "..."} on failure (so the parent can log)
      - None only when there are no rows to score
    """
    global _MODEL, _FCOLS
    if not rows:
        return None
    try:
        if _MODEL is None:
            bundle = joblib.load(model_path)
            _MODEL = bundle["model"]
            _FCOLS = bundle["features"]

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()

        feats = build_features(df)
        need = [c for c in ("ema_20", "ema_50", "atr_14", "close") if c not in feats.columns]
        if need:
            return {"symbol": symbol, "error": f"missing_features:{','.join(need)}"}

        if "trend_ok" not in feats.columns:
            feats["trend_ok"] = (feats["ema_20"] > feats["ema_50"]).astype(int)
        if "momentum_ok" not in feats.columns:
            feats["momentum_ok"] = (
                (feats["close"] > feats["ema_20"]) & (feats["ema_20"] > feats["ema_50"])
            ).astype(int)
        if "vol_ok" not in feats.columns:
            feats["vol_ok"] = (feats["atr_14"] / feats["close"] > 0.002).astype(int)

        last = feats.iloc[-1]
        X = feats[_FCOLS].iloc[[-1]]
        p_raw = float(_MODEL.predict_proba(X)[:, 1][0])
        price = float(last["close"])
        atr = float(last["atr_14"])
        mom_ok = bool(int(last["momentum_ok"]) == 1)
        trend_ok = bool(int(last["trend_ok"]) == 1)
        vol_ok = bool(int(last["vol_ok"]) == 1)
        rule_prob = 0.8 if mom_ok else 0.2
        p_ens = 0.7 * p_raw + 0.3 * rule_prob

        return {
            "symbol": symbol,
            "price": price,
            "atr": atr,
            "p_raw": p_raw,
            "p_ens": p_ens,
            "trend_ok": trend_ok,
            "vol_ok": vol_ok,
            "mom_ok": mom_ok,
            "n_bars": int(len(feats)),
            "ts_last": feats.index[-1].isoformat(),
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _effective_cpu_workers(requested: Optional[int]) -> int:
    if requested is None:
        requested = _cfg_cpu_workers()
    if requested <= 0:
        requested = os.cpu_count() or 2
    return max(1, int(requested))


async def run_scan(cpu_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run one full scan. Returns scored symbol dicts sorted by p_ens desc.

    Side effects:
      - Appends one JSON object per scored symbol to SCAN_OUT (jsonl)
      - Throttled progress lines to RUNTIME_LOG
      - Does NOT call shutdown() - the caller is responsible for that
        if they want to free cached cross-venue clients.
    """
    timeframe = _cfg_timeframe()
    lookback = _cfg_lookback()
    whitelist = _cfg_whitelist()
    fetch_timeout = _cfg_fetch_timeout()
    sanitize_timeout = _cfg_sanitize_timeout()
    cross_timeout = _cfg_cross_timeout()
    sanitize_workers = _cfg_sanitize_workers()
    http_workers = _cfg_http_workers()

    # 1) Universe
    if whitelist:
        universe = list(whitelist)
        _LOG.info("universe: using whitelist of %d symbols", len(universe))
    else:
        universe = await build_universe_bitget(_cfg_topn(), _cfg_min_notional())
    total_n = len(universe)

    # 2) Sanitise
    if sanitize_workers > 0 and universe:
        before = len(universe)
        universe = await sanitize_symbols(universe, timeframe,
                                           sanitize_workers, sanitize_timeout)
        total_n = len(universe)
        if total_n < before:
            _LOG.info("sanitize: %d -> %d symbols", before, total_n)

    if not universe:
        _LOG.warning("universe is empty after sanitise; returning no results")
        return []

    # 3) Async OHLCV fetch with timeout
    bitget = await _bitget_client()
    sem = asyncio.Semaphore(max(1, http_workers))

    async def fetch_one(sym: str) -> Tuple[str, Optional[List[List[float]]], str]:
        async with sem:
            try:
                rows = await fetch_ohlcv_async(bitget, sym, timeframe, lookback)
                return sym, rows, ""
            except Exception as e:
                return sym, None, f"{type(e).__name__}: {e}"

    all_rows: Dict[str, List[List[float]]] = {}
    fetch_failures: List[Tuple[str, str]] = []

    try:
        tasks = [asyncio.create_task(fetch_one(s)) for s in universe]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=fetch_timeout,
            )
        except asyncio.TimeoutError:
            _LOG.warning("fetch: timed out after %.1fs; cancelling in-flight tasks",
                         fetch_timeout)
            for t in tasks:
                if not t.done():
                    t.cancel()
            results = []
            for t in tasks:
                if t.done() and not t.cancelled():
                    try:
                        results.append(t.result())
                    except Exception as e:
                        results.append(("?", None, f"{type(e).__name__}: {e}"))

        done = 0
        for r in results:
            if isinstance(r, BaseException):
                _LOG.debug("fetch: task raised %s", r)
                continue
            if not (isinstance(r, tuple) and len(r) == 3):
                continue
            sym, rows, err = r
            done += 1
            spinner_batch(done, total_n, sym, phase="fetch")
            write_runtime_progress(done, total_n, sym, phase="fetch")
            if rows:
                all_rows[sym] = rows
            else:
                fetch_failures.append((sym, err or "empty"))
        if fetch_failures:
            _LOG.info("fetch: %d/%d symbols failed; first few: %s",
                      len(fetch_failures), total_n, fetch_failures[:3])
    finally:
        try:
            await bitget.close()
        except Exception:
            pass

    if not all_rows:
        _LOG.warning("no OHLCV data collected; returning no results")
        return []

    # 4) CPU features + scoring
    eff_workers = _effective_cpu_workers(cpu_workers)
    scan_out = _cfg_scan_out()
    _ensure_dir(scan_out)
    model_path = _cfg_model_path()

    results: List[Dict[str, Any]] = []
    score_errors: List[Tuple[str, str]] = []

    # Open the output file ONCE - the previous version reopened per result.
    with open(scan_out, "a", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=eff_workers) as pool:
            futs = {pool.submit(_cpu_features_and_score, s, all_rows[s], model_path): s
                    for s in all_rows}
            done = 0
            for future in as_completed(futs):
                sym = futs[future]
                try:
                    res = future.result()
                except Exception as e:
                    res = {"symbol": sym, "error": f"{type(e).__name__}: {e}"}
                done += 1
                spinner_batch(done, len(futs), sym, phase="score")
                write_runtime_progress(done, len(futs), sym, phase="score")
                if not res:
                    continue
                if "error" in res:
                    score_errors.append((res.get("symbol", sym), res["error"]))
                    continue
                results.append(res)
                try:
                    fout.write(json.dumps(res) + "\n")
                except Exception as e:
                    _LOG.debug("scan_out write failed: %s", e)

    if score_errors:
        _LOG.info("score: %d symbol(s) failed; first few: %s",
                  len(score_errors), score_errors[:3])

    # 5) Optional cross-venue prices
    if _cfg_fetch_cross() and results:
        venues = _cfg_cross_venues()

        async def fill_cross(item: Dict[str, Any]) -> None:
            base_sym = item["symbol"].split(":")[0]
            item["cross"] = await cross_venue_last(base_sym, venues, cross_timeout)

        cross_tasks = [asyncio.create_task(fill_cross(r)) for r in results]
        try:
            await asyncio.wait_for(
                asyncio.gather(*cross_tasks, return_exceptions=True),
                timeout=cross_timeout * 2.0,
            )
        except asyncio.TimeoutError:
            _LOG.warning("cross-venue: phase timed out; cancelling")
            for t in cross_tasks:
                if not t.done():
                    t.cancel()

    results.sort(key=lambda d: d.get("p_ens", 0.0), reverse=True)
    return results