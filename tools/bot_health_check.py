#!/usr/bin/env python3
"""
bot_health_check.py — Complete bot health diagnostic.
Run on EC2:  ~/bot/.venv/bin/python ~/bot/tools/bot_health_check.py
"""
import os, sys, json, csv, math, glob, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, deque

BASE          = Path(__file__).resolve().parent.parent
LOGS_DIR      = BASE / "logs"
SIGNALS_CSV   = LOGS_DIR / "live_signals.csv"
STATE_JSON    = LOGS_DIR / "executor_state.json"
HB_WRITER     = LOGS_DIR / "live_writer_heartbeat.json"
WR_ERR        = LOGS_DIR / "live_writer.err"
EX_ERR        = LOGS_DIR / "live_executor.err"
EX_OUT        = LOGS_DIR / "live_executor.out"
ENV_FILE      = BASE / ".env"
ARTIFACTS_DIR = BASE / "model_artifacts"

G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
NC = "\033[0m"    # reset
SEP  = "=" * 62
SEP2 = "-" * 62

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ok(msg):  print(f"  [{G} OK {NC}] {msg}")
def warn(msg):print(f"  [{Y}WARN{NC}] {msg}")
def err(msg): print(f"  [{R}FAIL{NC}] {msg}")
def info(msg):print(f"  [INFO] {msg}")

def load_env():
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

def _tail_text(path, nbytes=65536):
    """Read at most the last nbytes of a file (OOM-safe on huge logs)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            return fh.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

# ── Feature-width expectation (DL_ADD_SYMBOL_ID-aware) ─────────────────────────
def _feature_cols_len():
    try:
        if str(BASE) not in sys.path:
            sys.path.insert(0, str(BASE))
        from features import FEATURE_COLS
        return len(FEATURE_COLS)
    except Exception:
        return None

def _read_scaler_dims():
    dims = {}
    try:
        import joblib
    except ImportError:
        return dims
    for kind in ["adv", "lstm", "tcn", "tx"]:
        p = ARTIFACTS_DIR / f"scaler_{kind}_latest.joblib"
        if p.exists():
            try:
                dims[kind] = int(getattr(joblib.load(p), "n_features_in_", -1))
            except Exception:
                pass
    return dims

def resolve_feature_expectation(env):
    """Decide the expected feature width from DL_ADD_SYMBOL_ID, reconciled with
    the deployed scalers. Mirrors ml_dl.dl_ensemble.resolve_add_symbol_id so the
    health check agrees with what the writer actually does.

    Rules:
      - DL_ADD_SYMBOL_ID=0 -> expected_n = len(FEATURE_COLS)        (27, no symbol_id)
      - DL_ADD_SYMBOL_ID=1 -> expected_n = len(FEATURE_COLS) + 1    (28, with symbol_id)
      - unset              -> infer expected_n from the scaler n_features_in_
    """
    base = _feature_cols_len()
    scaler_dims = _read_scaler_dims()
    common_dim = None
    if scaler_dims:
        uniq = sorted(set(scaler_dims.values()))
        common_dim = uniq[0] if len(uniq) == 1 else None

    raw = env.get("DL_ADD_SYMBOL_ID")
    raw_set = raw not in (None, "", "NOT SET")
    add_sid = None
    expected_n = None
    if not raw_set:
        source = "scaler-inferred"
        if common_dim is not None:
            expected_n = common_dim
            if base is not None:
                add_sid = (common_dim == base + 1)
        elif base is not None:
            expected_n = base + 1  # last-resort default
    else:
        source = "env"
        add_sid = str(raw).strip().lower() not in ("0", "false", "no", "off")
        if base is not None:
            expected_n = base + (1 if add_sid else 0)
        elif common_dim is not None:
            expected_n = common_dim

    mismatch = None
    if expected_n is not None and common_dim is not None and expected_n != common_dim:
        want = "1" if (base is not None and common_dim == base + 1) else "0"
        mismatch = (f"env expects {expected_n} features but scalers expect {common_dim} "
                    f"— set DL_ADD_SYMBOL_ID={want} to match the deployed artifacts")
    return {
        "base": base, "raw": raw if raw_set else None, "add_sid": add_sid,
        "expected_n": expected_n, "source": source, "scaler_dims": scaler_dims,
        "common_dim": common_dim, "mismatch": mismatch,
    }

def fmt_pnl(v):
    s = f"{'+' if v >= 0 else ''}{v:.4f}"
    return f"{G}{s}{NC}" if v >= 0 else f"{R}{s}{NC}"

# ── 1. SYSTEMD SERVICES ───────────────────────────────────────────────────────
def check_services():
    print(f"\n{SEP}\n  1. SYSTEMD SERVICES\n{SEP2}")
    for svc in ["bot-writer", "bot-executor"]:
        try:
            r = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=5
            )
            status = r.stdout.strip()
            if status == "active":
                ok(f"{svc:<20} active (running)")
            else:
                err(f"{svc:<20} {status}  ← run: sudo systemctl restart {svc}")
        except Exception as e:
            warn(f"{svc}: cannot check — {e}")

# ── 2. MODEL ARTIFACTS ────────────────────────────────────────────────────────
def check_models(feat):
    print(f"\n{SEP}\n  2. MODEL ARTIFACTS\n{SEP2}")
    expected_tf = "5m"
    expected_feats = feat["expected_n"]
    base = feat["base"]
    add_sid = feat["add_sid"]
    sid_disp = {True: "True", False: "False", None: "?"}[add_sid]

    # Feature-width banner so the reason for the expectation is explicit.
    info(f"DL_ADD_SYMBOL_ID={feat['raw'] if feat['raw'] is not None else '(unset)'}  "
         f"add_symbol_id={sid_disp}  FEATURE_COLS={base if base is not None else '?'}  "
         f"expected_n={expected_feats if expected_feats is not None else '?'}  "
         f"(source: {feat['source']})")
    if feat["mismatch"]:
        err(f"DL_ADD_SYMBOL_ID mismatch: {feat['mismatch']}")

    kinds = ["adv", "lstm", "tcn", "tx"]
    any_fail = bool(feat["mismatch"])
    for kind in kinds:
        meta_path = ARTIFACTS_DIR / f"dl_{kind}_metadata.json"
        pt_path   = ARTIFACTS_DIR / f"dl_{kind}_latest.pt"
        sc_path   = ARTIFACTS_DIR / f"scaler_{kind}_latest.joblib"
        if not meta_path.exists():
            err(f"{kind}: metadata NOT FOUND"); any_fail = True; continue
        if not pt_path.exists():
            err(f"{kind}: .pt file NOT FOUND"); any_fail = True
        if not sc_path.exists():
            err(f"{kind}: scaler NOT FOUND"); any_fail = True
        try:
            m = json.loads(meta_path.read_text())
            tf  = m.get("timeframe", "?")
            nf  = m.get("n_features", 0)
            auc = m.get("val_auc", 0.0)
            trained = m.get("trained_at", "?")[:10]
            issues = []
            if tf != expected_tf:
                issues.append(f"tf={tf} (want {expected_tf})")
            if expected_feats is not None and nf != expected_feats:
                issues.append(f"n_features={nf} (want {expected_feats})")
            if auc < 0.55:
                issues.append(f"AUC={auc:.4f} LOW (<0.55)")
            line = f"{kind.upper():<5} tf={tf}  n_features={nf}  AUC={auc:.4f}  trained={trained}"
            if issues:
                warn(f"{line}  ⚠ {', '.join(issues)}")
                any_fail = True
            else:
                ok(line)
        except Exception as e:
            err(f"{kind}: cannot parse metadata — {e}"); any_fail = True

    # Scaler dim check against the resolved expected_n (NOT a hardcoded 28).
    if feat["scaler_dims"]:
        for kind, got in feat["scaler_dims"].items():
            if expected_feats is not None and got != expected_feats:
                err(f"Scaler dim {kind}: expects {got} features (want {expected_feats})")
                any_fail = True
    else:
        info("scaler dims unavailable (joblib missing or no scalers) — skipped")

    if not any_fail:
        ok(f"Model/scaler feature dims OK — all expect {expected_feats} features "
           f"(add_symbol_id={sid_disp}); timeframe OK; AUC >= 0.55")

# ── 3. .ENV SETTINGS ─────────────────────────────────────────────────────────
def check_settings(env, feat):
    print(f"\n{SEP}\n  3. .ENV SETTINGS\n{SEP2}")
    checks = [
        # key,                    good_values,              note
        ("LIVE_MODE",             ["0"],                    "0=paper  1=live"),
        ("EXEC_PAPER",            ["1"],                    "1=paper  0=live"),
        ("DL_TIMEFRAME",          ["5m"],                   "MUST match model training timeframe"),
        ("DL_P_LONG",             ["0.15"],                 "Signal threshold"),
        ("DL_MIN_AGREE",          ["2"],                    "Min models agreeing"),
        # DL_ADD_SYMBOL_ID is validated dynamically below (against the scaler width).
        ("EXEC_FEE_BPS",          ["5", None],              "Taker fee bps/side (default 5)"),
        ("EXEC_SLIPPAGE_BPS",     ["2", None],              "Slippage bps/side (default 2)"),
        ("EXEC_BIAS_GUARD",       ["1"],                    "Block biased-signal phases"),
        ("EXEC_FLIP_CONFIRM_TICKS",["20"],                  "Ticks before flip"),
        ("EXEC_COOLDOWN_SEC",     ["300"],                  "Min seconds between trades"),
        # 0.5%/0.5% since fd8128b: aligned with the ~0.3%/36-bar label barriers
        # the 5m models were trained on (1.5% TP was never reached: 0 TP hits
        # in 78 trades, all net loss came from EXIT_SL). See tools/sim_exits.py.
        ("EXEC_TP_PCT",           ["0.005"],                "Take-profit %"),
        ("EXEC_SL_PCT",           ["0.005"],                "Stop-loss %"),
        ("DL_ALLOW_ONLY",         ["1"],                    "MUST be 1 — 0 bypasses signal threshold gate"),
        ("LEVERAGE",              ["5"],                    "Futures leverage"),
        ("DL_BIAS_LSTM",          ["0.0", "0.00"],          "Bias correction (0 for new models)"),
        ("DL_BIAS_TCN",           ["0.0", "0.00"],          "Bias correction (0 for new models)"),
        ("DL_BIAS_TX",            ["0.0", "0.00"],          "Bias correction (0 for new models)"),
        ("DL_TEMP_LSTM",          ["1.0"],                  "Temperature scaling"),
        ("DL_TEMP_TX",            ["1.0"],                  "Temperature scaling"),
        ("DL_TEMP_ADV",           ["1.0", None],            "Temperature scaling"),
        ("DL_MODEL_WEIGHTS",      [None],                   "AUC-based weights (any non-empty value OK)"),
    ]
    issues = []
    for key, good_vals, note in checks:
        val = env.get(key, "NOT SET")
        if good_vals == [None]:   # any non-empty value is fine
            if val and val != "NOT SET":
                ok(f"{key:<30} = {val}")
            else:
                warn(f"{key:<30} = NOT SET  ({note})")
        elif None in good_vals:   # optional — skip if not set
            if val == "NOT SET":
                info(f"{key:<30} = not set (optional)")
            elif val in [v for v in good_vals if v is not None]:
                ok(f"{key:<30} = {val}")
            else:
                warn(f"{key:<30} = {val}  (expected: {[v for v in good_vals if v]})")
        else:
            if val in good_vals:
                ok(f"{key:<30} = {val}")
            else:
                warn(f"{key:<30} = {val}  (expected: {good_vals[0]})  # {note}")
                issues.append((key, good_vals[0]))

    # DL_ADD_SYMBOL_ID — validated against the deployed scaler width (not hardcoded).
    sid_val = env.get("DL_ADD_SYMBOL_ID", "NOT SET")
    if feat["mismatch"]:
        warn(f"{'DL_ADD_SYMBOL_ID':<30} = {sid_val}  (MISMATCH: {feat['mismatch']})")
        issues.append(("DL_ADD_SYMBOL_ID", "match scaler"))
    elif feat["expected_n"] is not None:
        ok(f"{'DL_ADD_SYMBOL_ID':<30} = {sid_val}  "
           f"(-> expected_n={feat['expected_n']}, matches deployed scalers)")
    else:
        info(f"{'DL_ADD_SYMBOL_ID':<30} = {sid_val}")

    # Mode summary
    live  = env.get("LIVE_MODE","0")
    paper = env.get("EXEC_PAPER","1")
    mode  = "PAPER" if (live == "0" or paper == "1") else "LIVE"
    col   = Y if mode == "LIVE" else G
    print(f"\n  {col}Mode: {mode}{NC}")
    return issues

# ── 4. WRITER HEARTBEAT ───────────────────────────────────────────────────────
def check_heartbeat():
    print(f"\n{SEP}\n  4. WRITER HEARTBEAT\n{SEP2}")
    if not HB_WRITER.exists():
        err("live_writer_heartbeat.json not found — writer may not be running"); return
    try:
        hb = json.loads(HB_WRITER.read_text())
        ts_str = hb.get("ts","")
        ts_str_clean = ts_str.replace("+0000","").replace("Z","")
        try:
            ts = datetime.fromisoformat(ts_str_clean).replace(tzinfo=timezone.utc)
            age = int((datetime.now(timezone.utc) - ts).total_seconds())
        except Exception:
            age = 9999
        syms   = hb.get("symbols", [])
        p_meta = hb.get("p_meta", 0)
        mode   = hb.get("mode", "?")
        allow  = hb.get("allow", "?")
        if age > 60:
            err(f"Heartbeat is {age}s old — writer may be stuck or crashed")
        else:
            ok(f"Last tick {age}s ago  symbols={syms}  p_meta={p_meta:.4f}  allow={allow}  mode={mode}")
    except Exception as e:
        err(f"Cannot parse heartbeat: {e}")

# ── 5. EXECUTOR STATE ─────────────────────────────────────────────────────────
def check_state():
    print(f"\n{SEP}\n  5. EXECUTOR STATE\n{SEP2}")
    if not STATE_JSON.exists():
        info("executor_state.json not found (no state persisted yet)"); return
    try:
        state = json.loads(STATE_JSON.read_text())
        positions = state.get("open_positions", {})
        if positions:
            for sym, pos in positions.items():
                side = pos.get("side","?")
                qty  = pos.get("qty", 0)
                avg  = pos.get("avg", 0)
                ok(f"Open: {sym}  {side.upper()}  qty={qty}  avg_entry={avg:.4f}")
        else:
            ok("Open positions: NONE (flat)")
        flip_pend = state.get("flip_pending", {})
        for sym, fp in flip_pend.items():
            info(f"Flip pending: {sym}  dir={fp.get('direction','?')}  ticks={fp.get('ticks',0)}")
        bias_skip = state.get("bias_skip_until", 0)
        if bias_skip and bias_skip > time.time():
            warn(f"BIAS_SKIP active: {int(bias_skip - time.time())}s remaining")
    except Exception as e:
        err(f"Cannot parse executor state: {e}")

# ── 6. SIGNAL QUALITY ─────────────────────────────────────────────────────────
def check_signals(n=300):
    print(f"\n{SEP}\n  6. SIGNAL QUALITY (last {n} rows)\n{SEP2}")
    if not SIGNALS_CSV.exists():
        err("live_signals.csv not found"); return
    rows = list(deque(csv.DictReader(open(SIGNALS_CSV)), maxlen=n))
    if not rows:
        err("No signals found"); return

    # Parse (symbol, p_meta) so we can assess each symbol independently now that
    # ETH and SOL are scored separately.
    parsed = []
    for r in rows:
        try:
            parsed.append((r.get("symbol", "?"), float(r.get("p_meta", "")), r))
        except Exception:
            pass
    if not parsed:
        err("No p_meta values in signals"); return
    p_vals = [p for _, p, _ in parsed]

    # Don't cry "collapse" before enough FRESH post-reset rows exist. A one-sided
    # run on a handful of rows is too-early, not a failure.
    COLLAPSE_ROWS = int(os.getenv("HEALTH_COLLAPSE_ROWS", "300"))
    LEAN_ROWS     = int(os.getenv("HEALTH_LEAN_ROWS", "150"))

    def bias_report(label, pv):
        tot = len(pv)
        ln = sum(1 for p in pv if p >= 0)
        lp = ln / tot * 100
        sp = 100 - lp
        dom = max(lp, sp)
        side = "LONG" if lp >= sp else "SHORT"
        msg = f"{label}: {side} {dom:.0f}%  (LONG {lp:.0f}% / SHORT {sp:.0f}%, {tot} rows)"
        if dom >= 95:
            if tot >= COLLAPSE_ROWS:
                err(f"{msg} — model collapsed!")
            elif tot >= LEAN_ROWS:
                warn(f"{msg} — one-sided; monitor (need {COLLAPSE_ROWS} rows to call collapse)")
            else:
                warn(f"{msg} — only {tot} fresh rows (<{LEAN_ROWS}); too early to judge")
        elif dom >= 80:
            warn(msg)
        else:
            ok(msg)

    # Overall, then per-symbol.
    bias_report("SIDE_BIAS (all)", p_vals)
    by_sym = {}
    for sym, p, _ in parsed:
        by_sym.setdefault(sym, []).append(p)
    if len(by_sym) > 1:
        for sym in sorted(by_sym):
            bias_report(f"  {sym}", by_sym[sym])

    # Allow rate
    allow_n = sum(1 for r in rows if r.get("allow","").strip() == "1")
    allow_pct = allow_n / len(rows) * 100
    if allow_pct < 5:
        warn(f"Allow rate: {allow_pct:.1f}% — threshold too tight or model weak")
    else:
        ok(f"Allow rate: {allow_pct:.1f}%  ({allow_n}/{len(rows)} signals pass threshold)")

    # p_meta stats
    avg_abs = sum(abs(p) for p in p_vals) / len(p_vals)
    avg_p   = sum(p_vals) / len(p_vals)
    p_min, p_max = min(p_vals), max(p_vals)
    ok(f"p_meta range: {p_min:+.3f} to {p_max:+.3f}  avg={avg_p:+.3f}  avg|p|={avg_abs:.3f}")
    if avg_abs < 0.08:
        warn("avg|p_meta| < 0.08 — model has very weak edge right now")

    # Price feed
    px_vals = [r.get("px","").strip() for r in rows]
    zero_px = sum(1 for p in px_vals if p in ("0","0.00000000",""))
    zero_pct = zero_px / len(rows) * 100
    if zero_pct >= 50:
        err(f"Price feed: {zero_pct:.0f}% of signals have px=0 — no trades possible!")
    elif zero_pct > 10:
        warn(f"Price feed: {zero_pct:.0f}% of signals have px=0 — intermittent")
    else:
        ok(f"Price feed: OK ({100-zero_pct:.0f}% of signals have valid price)")

    # Latest signal
    last = rows[-1]
    info(f"Latest: ts={last.get('ts','')[:19]}  side={last.get('side_hint','')}  "
         f"p_meta={last.get('p_meta','')}  allow={last.get('allow','')}  px={last.get('px','')[:10]}")

    # SIDE_BIAS warnings from executor log (tail-only, OOM-safe).
    if EX_OUT.exists():
        bias_lines = [l for l in _tail_text(EX_OUT).splitlines() if "SIDE_BIAS" in l]
        if bias_lines:
            warn(f"SIDE_BIAS warnings in recent executor log: {len(bias_lines)}")
            info(f"  Last: {bias_lines[-1][1:40]}")
        else:
            ok("No recent SIDE_BIAS warnings in executor log")

# ── 7. TRADE PERFORMANCE ──────────────────────────────────────────────────────
def check_trades():
    print(f"\n{SEP}\n  7. TRADE PERFORMANCE (all trades_closed_*.csv)\n{SEP2}")
    files = sorted(glob.glob(str(LOGS_DIR / "trades_closed_*.csv")))
    if not files:
        info("No trades_closed_*.csv files yet"); return

    trades = []
    for f in files:
        try:
            for row in csv.DictReader(open(f)):
                try:
                    trades.append({
                        "ts":     row["ts"],
                        "symbol": row.get("symbol",""),
                        "side":   row.get("closed_side",""),
                        "pnl":    float(row["realized_pnl"]),
                        "reason": row.get("reason","").split()[0],
                        "date":   os.path.basename(f).replace("trades_closed_","").replace(".csv",""),
                    })
                except: pass
        except: pass

    if not trades:
        info("No trades found in files"); return

    n   = len(trades)
    pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr  = len(wins)/n*100
    gw  = sum(t["pnl"] for t in wins)
    gl  = abs(sum(t["pnl"] for t in losses))
    pf  = gw/gl if gl > 0 else 999.0
    eq  = []
    run = 0.0
    for t in trades:
        run += t["pnl"]; eq.append(run)
    peak = eq[0]
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        max_dd = min(max_dd, e - peak)

    avg_win  = sum(t["pnl"] for t in wins)/len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses)/len(losses) if losses else 0

    print(f"  Total trades     : {n}")
    print(f"  Win rate         : {G if wr >= 50 else Y}{wr:.1f}%{NC}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net P&L          : {fmt_pnl(pnl)} USDT")
    print(f"  Profit factor    : {G if pf >= 1 else R}{pf:.3f}{NC}  (>1.0 = profitable)")
    print(f"  Max drawdown     : {R}{max_dd:.4f}{NC} USDT")
    print(f"  Avg win          : {fmt_pnl(avg_win)} USDT")
    print(f"  Avg loss         : {fmt_pnl(avg_loss)} USDT")

    # Exit reasons
    print(f"\n  Exit reasons:")
    for rt, cnt in Counter(t["reason"] for t in trades).most_common():
        rt_pnl = sum(t["pnl"] for t in trades if t["reason"] == rt)
        rt_avg = rt_pnl / cnt
        bar = f"{fmt_pnl(rt_pnl)}"
        print(f"    {rt:<15} {cnt:4d} trades   total={bar}   avg={fmt_pnl(rt_avg)}")

    # Per-day
    print(f"\n  Per-day P&L:")
    by_date = {}
    for t in trades:
        by_date.setdefault(t["date"],[]).append(t)
    for date in sorted(by_date):
        day = by_date[date]
        dpnl = sum(t["pnl"] for t in day)
        longs  = sum(1 for t in day if t["side"].upper() in ("SELL","SELL_SHORT"))
        shorts = sum(1 for t in day if t["side"].upper() in ("BUY_TO_COVER","BUY"))
        max_side = max(longs,shorts)/len(day)*100 if day else 0
        bias_warn = f"  {Y}← side {max_side:.0f}%{NC}" if max_side > 70 else ""
        print(f"    {date}  {len(day):3d} trades  P&L {fmt_pnl(dpnl)}  "
              f"L:{longs} S:{shorts}{bias_warn}")

    # Last 5 trades
    print(f"\n  Last 5 trades:")
    for t in trades[-5:]:
        print(f"    {t['ts'][:19]}  {t['symbol']:<8} {t['side']:<14} "
              f"{fmt_pnl(t['pnl'])}  {t['reason']}")

    # Verdict
    print(f"\n{SEP2}")
    if pf >= 1.1 and wr >= 50:
        print(f"  {G}VERDICT: Profitable — looking good{NC}")
    elif pf >= 0.95:
        print(f"  {Y}VERDICT: Marginal — monitor for 3-4 more days{NC}")
    elif pf < 0.85:
        print(f"  {R}VERDICT: Losing — check signal bias and TP/SL ratio{NC}")
    else:
        print(f"  {Y}VERDICT: Slightly below breakeven — give it more time{NC}")

# ── 8. RECENT ERRORS ─────────────────────────────────────────────────────────
def check_errors():
    print(f"\n{SEP}\n  8. RECENT ERRORS (last 24h)\n{SEP2}")
    cutoff = "2026"  # simple check: just show last few lines
    for label, path in [("Writer ERR", WR_ERR), ("Executor ERR", EX_ERR)]:
        if not path.exists():
            info(f"{label}: no file"); continue
        # Read only the tail to avoid OOM on large files
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 8192))
                tail_bytes = fh.read()
            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()
        except Exception as e:
            warn(f"{label}: cannot read — {e}"); continue
        # Last 5 non-empty lines
        recent = [l for l in lines if l.strip()][-5:]
        if recent:
            fatal = [l for l in recent if "FATAL" in l or "Error" in l or "error" in l]
            if fatal:
                err(f"{label}: recent errors found:")
                for l in fatal[-3:]:
                    print(f"    {l[:100]}")
            else:
                ok(f"{label}: no recent fatal errors")
        else:
            ok(f"{label}: empty (clean)")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{SEP}")
    print(f"  BOT HEALTH CHECK  —  {now_utc()}")
    print(SEP)

    env = load_env()
    feat = resolve_feature_expectation(env)
    check_services()
    check_models(feat)
    check_settings(env, feat)
    check_heartbeat()
    check_state()
    check_signals()
    check_trades()
    check_errors()

    print(f"\n{SEP}")
    print(f"  Run again anytime:  ~/bot/.venv/bin/python ~/bot/tools/bot_health_check.py")
    print(SEP)
    print()

if __name__ == "__main__":
    main()
