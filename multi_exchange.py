# multi_exchange.py
from __future__ import annotations

import os, asyncio, math, json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import ccxt.async_support as ccxt_async  # asyncio version of ccxt

try:
    import aiohttp
except ImportError:
    aiohttp = None  # DEX analysis needs aiohttp; CEX works without it


@dataclass
class Quote:
    venue: str          # "binance", "bybit", "bitget", "mexc", or "dex:<source>"
    vwap: float         # volume-weighted average fill price in quote/base terms
    eff_bps: float      # effective taker fee in bps included in decision
    depth_ok: bool      # was there enough depth for the base_amount?
    reason: str = ""    # why it failed (if depth_ok=False or None)
    mtype: str = "spot" # "spot" | "swap" | "dex"


# ---- Config helpers ----
def _env_bool(name: str, default: bool = True) -> bool:
    v = os.getenv(name, "1" if default else "0").strip().lower()
    return v not in ("0", "false", "no", "off", "")


# Map ccxt defaultType per venue + market_type requested
_DEFAULTTYPE = {
    "binance": {"spot": "spot", "swap": "future"},  # USDT-M futures
    "bybit":   {"spot": "spot", "swap": "swap"},
    "bitget":  {"spot": "spot", "swap": "swap"},
    "mexc":    {"spot": "spot", "swap": "swap"},
}

# ---- CEX client factory (async) ----
async def _with_client(ex_id: str, market_type: str = "spot"):
    ex_id = ex_id.strip().lower()
    cls = getattr(ccxt_async, ex_id)
    default_type = _DEFAULTTYPE.get(ex_id, {}).get(market_type, market_type)

    # Per-exchange SSL verify toggle (helps if your OS trust store is noisy)
    verify = _env_bool(f"{ex_id.upper()}_VERIFY", True)

    opts = {
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": default_type},
        "verify": verify,
    }
    ex = cls(opts)
    ex.verify = verify

    try:
        # prefer loading markets for the type we want
        try:
            await ex.load_markets(False, {"type": default_type})
        except Exception:
            await ex.load_markets()
        return ex
    except Exception:
        try:
            await ex.close()
        except Exception:
            pass
        raise


# ---- Orderbook → VWAP helper ----
def _vwap_from_orderbook(ob: Dict[str, Any], side: str, base_amount: float) -> Tuple[Optional[float], bool]:
    levels = ob.get("asks" if side == "buy" else "bids", [])
    if not levels or base_amount <= 0:
        return None, False
    need = float(base_amount)
    cost = 0.0
    for price, qty in levels:
        qty = float(qty); price = float(price)
        take = min(need, qty)
        cost += take * price
        need -= take
        if need <= 1e-12:
            break
    if need > 1e-12:
        return None, False
    return float(cost / max(base_amount, 1e-12)), True


# ---- Quote one CEX venue ----
async def _quote_one_cex(
    ex_id: str,
    symbol: str,
    side: str,
    base_amount: float,
    default_bps: float,
    market_type: str = "spot",
) -> Optional[Quote]:
    try:
        ex = await _with_client(ex_id, market_type=market_type)
    except Exception:
        return None
    try:
        # ensure symbol exists
        if symbol not in getattr(ex, "symbols", []):
            return Quote(ex_id, math.inf, float("inf"), False, reason="no_symbol", mtype=market_type)

        ob = await ex.fetch_order_book(symbol, limit=50)
        vwap, depth_ok = _vwap_from_orderbook(ob, side, base_amount)
        if vwap is None:
            return Quote(ex_id, math.inf, float("inf"), False, reason="no_depth", mtype=market_type)

        # taker bps: prefer market metadata, else env default
        fee_bps = float(default_bps)
        try:
            mk = ex.markets.get(symbol, {})
            if mk and isinstance(mk.get("taker"), (int, float)):
                fee_bps = float(mk["taker"]) * 10000.0  # fraction->bps
        except Exception:
            pass

        return Quote(ex_id, float(vwap), float(fee_bps), True, mtype=market_type)
    except Exception:
        return None
    finally:
        try:
            await ex.close()
        except Exception:
            pass


# ---- Optional: DEX quote via aggregator (analysis only) ----
async def _quote_one_dex(
    symbol: str,
    side: str,
    base_amount: float,
) -> Optional[Quote]:
    """
    Requires these envs:
      DEX_ENABLED=1
      DEX_AGG_URL=https://api.0x.org/swap/v1/price
      DEX_CHAIN_ID=1
      DEX_TOKEN_MAP={"BTC":"WBTC","ETH":"WETH","USDT":"USDT","PEPE":"PEPE"}   (symbol->token symbol)
      DEX_ADDR_MAP={"WBTC":"0x2260...","WETH":"0xC02a...","USDT":"0xdAC1...","PEPE":"0x6982..."} (token->address)
      DEX_DECIMALS={"WBTC":8,"WETH":18,"USDT":6,"PEPE":18}
      DEX_SLIPPAGE_BPS=30
      DEX_GAS_USD=0.50  (fallback if aggregator doesn’t return gas)
    If these are missing, returns None (skipped).
    """
    if not _env_bool("DEX_ENABLED", False):
        return None
    if aiohttp is None:
        return None

    agg_url = os.getenv("DEX_AGG_URL", "").strip()
    if not agg_url:
        return None

    try:
        base_raw, quote_raw = symbol.split("/")[0], symbol.split("/")[1].split(":")[0]
    except Exception:
        return None

    # token symbol → canonical token symbol (WBTC/WETH/USDT/...)
    try:
        token_map = json.loads(os.getenv("DEX_TOKEN_MAP", "{}"))
        addr_map  = json.loads(os.getenv("DEX_ADDR_MAP", "{}"))
        dec_map   = json.loads(os.getenv("DEX_DECIMALS", "{}"))
    except Exception:
        return None

    base_tok  = token_map.get(base_raw)
    quote_tok = token_map.get(quote_raw)
    if not base_tok or not quote_tok:
        return None

    base_addr  = addr_map.get(base_tok)
    quote_addr = addr_map.get(quote_tok)
    base_dec   = int(dec_map.get(base_tok, 18))
    quote_dec  = int(dec_map.get(quote_tok, 18))
    if not base_addr or not quote_addr:
        return None

    # side: we are "buying base with quote" for side="buy"
    # use aggregator with buyAmount = base_amount in wei
    buy_amount_wei = int(round(float(base_amount) * (10 ** base_dec)))

    params = {
        # 0x-style params; for other aggregators, adapt here
        "buyToken": base_addr,
        "sellToken": quote_addr,
        "buyAmount": str(buy_amount_wei),
    }

    headers = {}
    if os.getenv("DEX_API_KEY"):
        headers["0x-api-key"] = os.getenv("DEX_API_KEY")

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(agg_url, params=params, headers=headers, timeout=10) as r:
                if r.status != 200:
                    return Quote("dex:agg", math.inf, float("inf"), False, reason=f"http_{r.status}", mtype="dex")
                data = await r.json()

        # prefer amounts to compute price precisely
        # data may have "sellAmount" and "buyAmount" as strings
        sell_amount = data.get("sellAmount")
        buy_amount  = data.get("buyAmount")
        if sell_amount and buy_amount:
            sell = float(sell_amount) / (10 ** quote_dec)
            buy  = float(buy_amount)  / (10 ** base_dec)
            if buy <= 0:
                return Quote("dex:agg", math.inf, float("inf"), False, reason="zero_buy", mtype="dex")
            vwap = sell / buy  # quote per base
        else:
            # fallback to "price" if provided
            vwap = float(data.get("price", "nan"))
            if not math.isfinite(vwap):
                return Quote("dex:agg", math.inf, float("inf"), False, reason="no_price", mtype="dex")

        # effective fee: include slippage + gas as bps on notional
        slippage_bps = float(os.getenv("DEX_SLIPPAGE_BPS", "30"))
        gas_usd = float(os.getenv("DEX_GAS_USD", "0.5"))
        # estimate notional in quote:
        notional_quote = vwap * float(base_amount)
        gas_bps = 0.0
        if notional_quote > 0:
            gas_bps = (gas_usd / notional_quote) * 10000.0
        eff_bps = slippage_bps + gas_bps

        return Quote("dex:agg", float(vwap), float(eff_bps), True, mtype="dex")
    except Exception:
        return None


# ---- Orchestrator (CEX + optional DEX) ----
async def best_cex_quote_async(
    exchanges: List[str],
    symbol: str,
    side: str,
    base_amount: float,
    default_bps: float,
    market_type: str = "spot",
    analyze_dex: bool = False,
) -> Optional[Quote]:
    tasks: List[asyncio.Task] = []

    # CEX tasks
    for e in exchanges:
        e = e.strip()
        if not e:
            continue
        tasks.append(
            asyncio.create_task(
                _quote_one_cex(e, symbol, side, base_amount, default_bps, market_type=market_type)
            )
        )

    # DEX (optional)
    if analyze_dex:
        tasks.append(asyncio.create_task(_quote_one_dex(symbol, side, base_amount)))

    if not tasks:
        return None

    results = await asyncio.gather(*tasks, return_exceptions=False)
    best: Optional[Tuple[Quote, float]] = None

    for q in results:
        if q is None or not q.depth_ok or not math.isfinite(q.vwap):
            continue

        if side == "buy":
            eff_price = q.vwap * (1.0 + q.eff_bps / 10000.0)
            better = lambda a, b: a < b
        else:  # sell
            eff_price = q.vwap * (1.0 - q.eff_bps / 10000.0)
            better = lambda a, b: a > b

        if best is None or better(eff_price, best[1]):
            best = (q, eff_price)

    return None if best is None else best[0]


def best_cex_quote(
    exchanges: List[str],
    symbol: str,
    side: str,
    base_amount: float,
    default_bps: float,
    market_type: str = "spot",
    analyze_dex: bool = False,
) -> Optional[Quote]:
    return asyncio.run(
        best_cex_quote_async(
            exchanges, symbol, side, base_amount, default_bps,
            market_type=market_type, analyze_dex=analyze_dex
        )
    )


def best_cex_quote_from_env(symbol: str, side: str, base_amount: float, market_type: str = "spot") -> Optional[Quote]:
    exs = os.getenv("ANALYZE_EXCHANGES", os.getenv("ROUTER_EXCHANGES", "binance,bybit,bitget,mexc")).split(",")
    default_bps = float(os.getenv("ROUTER_DEFAULT_TAKER_BPS", "10"))
    analyze_dex = _env_bool("ANALYZE_DEX", False)
    return best_cex_quote(exs, symbol, side, base_amount, default_bps, market_type=market_type, analyze_dex=analyze_dex)


__all__ = [
    "Quote",
    "best_cex_quote_async",
    "best_cex_quote",
    "best_cex_quote_from_env",
]
