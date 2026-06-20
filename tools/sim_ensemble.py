#!/usr/bin/env python3
r"""
sim_ensemble.py — READ-ONLY ensemble-variant simulator.

Recomputes the blended signal over recent logs/live_meta_log.csv rows under
several weighting/subset variants and reports the side-bias of each, faithfully
replaying the live agreement gate (DL_MIN_AGREE) and abs-mode threshold
(DL_P_LONG). Writes nothing, trades nothing, changes no settings.

Run on EC2:
    ~/bot/.venv/bin/python ~/bot/tools/sim_ensemble.py
    ~/bot/.venv/bin/python ~/bot/tools/sim_ensemble.py --rows 4000 --horizon 60

If logs/live_models_by_symbol.csv exists (written by the writer since the
agree-gate fix), a PER-SYMBOL variant table is printed too — true per-symbol
variant effects (e.g. "ADV off" flipping ETH more than SOL). The
live_meta_log-based table remains the symbol-AVERAGED view. The optional P&L
block is a crude next-horizon directional proxy for the CURRENT ensemble only
(from live_signals.csv px), not the real TP/SL/flip strategy P&L.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import deque
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "logs"
META_CSV = LOGS / "live_meta_log.csv"
SIGNALS_CSV = LOGS / "live_signals.csv"
MODELS_BY_SYMBOL_CSV = LOGS / "live_models_by_symbol.csv"


def load_env():
    env = {}
    f = BASE / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def tail(path, n):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(deque(csv.DictReader(fh), maxlen=n))


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_weights(env):
    raw = env.get("DL_MODEL_WEIGHTS", "").strip()
    w = {}
    for part in raw.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                w[k.strip()] = max(0.0, float(v.strip()))
            except Exception:
                pass
    return w


def blend_row(pm, subset, weights, min_agree, equal=False):
    """Return (centered_p, gate) for one row's per-model dict pm over `subset`.
    Replays the live agree-gate (post-fix): only positive-weight models vote;
    if <min_agree voters agree on a side, p_long->0.5 -> centered 0.0 (FLAT,
    fails the allow threshold). Pre-fix this was 0.0 -> centered -0.5, a fake
    allowed SHORT."""
    present = {m: pm[m] for m in subset if m in pm and pm[m] is not None}
    if not present:
        return None, "none"
    if equal:
        w = {m: 1.0 for m in present}
    else:
        w = {m: max(0.0, weights.get(m, 0.0)) for m in present}
        if sum(w.values()) <= 0:
            w = {m: 1.0 for m in present}
    den = sum(w.values())
    p = sum(w[m] * present[m] for m in present) / den
    gate = "pass"
    voters = [m for m in present if w.get(m, 0.0) > 0.0] or list(present)
    if len(voters) >= min_agree:
        nb = sum(1 for m in voters if present[m] > 0.5)
        ns = sum(1 for m in voters if present[m] < 0.5)
        if nb < min_agree and ns < min_agree:
            p = 0.5
            gate = "suppressed"
    return p - 0.5, gate


def summarize(name, centered_gates, thr):
    rows = [(c, g) for c, g in centered_gates if c is not None]
    n = len(rows)
    if not n:
        return f"  {name:<26} (no data)"
    supp = sum(1 for c, g in rows if g == "suppressed")
    longs = sum(1 for c, g in rows if g != "suppressed" and c > 0)
    shorts = sum(1 for c, g in rows if g != "suppressed" and c < 0)
    flat = n - supp - longs - shorts
    allowed = sum(1 for c, g in rows if abs(c) >= thr)
    mean_c = st.fmean(c for c, g in rows)
    return (f"  {name:<26} LONG {100*longs/n:4.0f}%  SHORT {100*shorts/n:4.0f}%  "
            f"supp {100*supp/n:4.0f}%  flat {100*flat/n:3.0f}%   "
            f"allow {100*allowed/n:4.0f}%   mean_c {mean_c:+.3f}   n={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=60, help="ticks ahead for the P&L proxy")
    args = ap.parse_args()
    env = load_env()
    min_agree = int(env.get("DL_MIN_AGREE", "2"))
    thr = _f(env.get("DL_P_LONG", "0.15")) or 0.15
    weights = parse_weights(env)

    print("=" * 78 + "\n  ENSEMBLE-VARIANT SIMULATOR (read-only)\n" + "=" * 78)
    meta = tail(META_CSV, args.rows)
    if not meta:
        print(f"  {META_CSV} not found/empty"); return 1
    model_cols = [c for c in meta[0] if c.endswith("_p") and c != "p_meta"]
    models = [c[:-2] for c in model_cols]
    print(f"  rows={len(meta)}  models={models}  DL_MODEL_WEIGHTS={weights or '(equal)'}  "
          f"DL_MIN_AGREE={min_agree}  DL_P_LONG={thr}")
    print("  NOTE: per-model values are symbol-AVERAGED in live_meta_log -> variant")
    print("        side-bias below is the averaged view (see header caveat).")

    per_tick = []
    for r in meta:
        per_tick.append({m: _f(r.get(m + "_p")) for m in models})

    # Per-model context
    print("\n" + "-" * 78 + "\n  PER-MODEL (symbol-averaged)\n" + "-" * 78)
    for m in models:
        v = [d[m] for d in per_tick if d.get(m) is not None]
        if v:
            print(f"  {m.upper():<5} mean={st.fmean(v):.3f}  median={st.median(v):.3f}  "
                  f"std={(st.pstdev(v) if len(v)>1 else 0):.3f}  "
                  f"LONG={100*sum(1 for x in v if x>0.5)/len(v):.0f}%  "
                  f"min={min(v):.3f} max={max(v):.3f}")

    # Variants
    allm = set(models)
    variants = [
        ("current (all, weights)", allm, False),
        ("equal (all)", allm, True),
        ("ADV off (weights)", allm - {"adv"}, False),
        ("no TCN (weights)", allm - {"tcn"}, False),
        ("LSTM+TX (weights)", allm & {"lstm", "tx"}, False),
        ("TX only", allm & {"tx"}, False),
        ("ADV only", allm & {"adv"}, False),
        ("LSTM only", allm & {"lstm"}, False),
    ]
    print("\n" + "-" * 78 + "\n  VARIANT SIDE-BIAS (agree-gate + abs threshold replayed)\n" + "-" * 78)
    for name, subset, equal in variants:
        if not subset:
            continue
        cg = [blend_row(d, subset, weights, min_agree, equal) for d in per_tick]
        print(summarize(name, cg, thr))
    print("\n  'supp' = agree-gate suppressed -> neutral 0.5 -> centered 0.0 = FLAT, allow=0")
    print("          (post-fix; it used to emit -0.5, a fake allowed SHORT).")
    print("          High supp% = models disagree a lot. Zero-weight models do not vote.")

    # Per-symbol variant tables (only if the writer has logged per-symbol
    # per-model probabilities — written since the agree-gate fix).
    mbs = tail(MODELS_BY_SYMBOL_CSV, args.rows)
    if mbs and "symbol" in mbs[0]:
        mbs_models = [c[:-2] for c in mbs[0] if c.endswith("_p")]
        rows_by_sym = {}
        for r in mbs:
            rows_by_sym.setdefault(r.get("symbol", "?"), []).append(
                {m: _f(r.get(m + "_p")) for m in mbs_models})
        print("\n" + "-" * 78 + "\n  PER-SYMBOL VARIANT SIDE-BIAS (from live_models_by_symbol.csv)\n" + "-" * 78)
        for symn in sorted(rows_by_sym):
            ticks = rows_by_sym[symn]
            print(f"\n  [{symn}]  n={len(ticks)}")
            for m in mbs_models:
                v = [d[m] for d in ticks if d.get(m) is not None]
                if v:
                    print(f"    {m.upper():<5} mean={st.fmean(v):.3f}  median={st.median(v):.3f}  "
                          f"std={(st.pstdev(v) if len(v)>1 else 0):.3f}  "
                          f"LONG={100*sum(1 for x in v if x>0.5)/len(v):.0f}%")
            for name, subset, equal in variants:
                if not subset & set(mbs_models):
                    continue
                cg = [blend_row(d, subset & set(mbs_models), weights, min_agree, equal)
                      for d in ticks]
                print("  " + summarize(name, cg, thr))

    # Optional crude directional proxy for CURRENT ensemble (per symbol, from signals)
    sig = tail(SIGNALS_CSV, args.rows)
    if sig and "symbol" in sig[0] and "px" in sig[0]:
        print("\n" + "-" * 78 + f"\n  CURRENT-ENSEMBLE DIRECTIONAL PROXY  (next-{args.horizon}-tick, NOT strategy P&L)\n" + "-" * 78)
        bysym = {}
        for r in sig:
            p = _f(r.get("p_meta")); px = _f(r.get("px"))
            if p is None or px is None or px <= 0:
                continue
            bysym.setdefault(r.get("symbol", "?"), []).append((p, px))
        for symn in sorted(bysym):
            seq = bysym[symn]
            h = args.horizon
            edges, hits, n = 0.0, 0, 0
            for i in range(len(seq) - h):
                p, px0 = seq[i]; _, px1 = seq[i + h]
                if px0 <= 0:
                    continue
                r = (px1 - px0) / px0
                edges += (1 if p > 0 else -1) * r
                hits += 1 if ((p > 0) == (r > 0)) else 0
                n += 1
            if n:
                print(f"  {symn}  n={n}  hit-rate={100*hits/n:4.1f}%  "
                      f"mean directional ret/step={1e4*edges/n:+.2f} bps  (excl fees)")
        print("  Caveat: tick-spaced px, fixed horizon, no TP/SL/flip/fees -> indicative only.")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
