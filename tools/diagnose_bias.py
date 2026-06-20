#!/usr/bin/env python3
r"""
diagnose_bias.py — READ-ONLY analysis of the LONG-bias source.

Run on EC2:
    ~/bot/.venv/bin/python ~/bot/tools/diagnose_bias.py
    ~/bot/.venv/bin/python ~/bot/tools/diagnose_bias.py --rows 5000

Reads only logs/live_meta_log.csv (per-model probabilities, blended p_meta) and
logs/live_signals.csv (per-symbol centered signal). Writes nothing, trades
nothing, changes no settings. Answers:

  Q1/Q3/Q8  per-model raw-probability distribution + %LONG (p>0.5)
  Q2        which model drives the LONG bias
  Q4        per-tick vote distribution (4/4, 3/4, 2/4, 1/4, 0/4 LONG)
  Q5        is the ensemble weighting causing the bias?
  Q6        would removing TCN improve balance?
  Q7        is DL_P_LONG too permissive? (threshold = frequency, not direction)
  Q9        do ETH and SOL have different distributions?
  Q10       pointer to the highest-leverage change

NOTE on data: live_meta_log per-model columns are averaged across symbols, so the
per-MODEL view (Q1-Q8) is symbol-averaged; the per-SYMBOL view (Q9) uses the
blended signal in live_signals.csv. Per-model-per-symbol is not logged.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import deque, Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "logs"
META_CSV = LOGS / "live_meta_log.csv"
SIGNALS_CSV = LOGS / "live_signals.csv"


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


def tail_rows(path, n):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(deque(csv.DictReader(fh), maxlen=n))


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def hist(vals, lo=0.0, hi=1.0, bins=10, width=40):
    if not vals:
        return "  (no data)"
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in vals:
        b = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
        counts[b] += 1
    mx = max(counts) or 1
    out = []
    for i in range(bins):
        bar = "#" * int(width * counts[i] / mx)
        pct = 100 * counts[i] / len(vals)
        out.append(f"    [{edges[i]:+.2f},{edges[i+1]:+.2f})  {counts[i]:5d} {pct:4.0f}%  {bar}")
    return "\n".join(out)


def parse_weights(env, models):
    raw = env.get("DL_MODEL_WEIGHTS", "").strip()
    w = {}
    if raw:
        for part in raw.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                try:
                    w[k.strip()] = max(0.0, float(v.strip()))
                except Exception:
                    pass
    if not w:
        w = {m: 1.0 for m in models}
    w = {m: w.get(m, 0.0) for m in models}
    s = sum(w.values()) or 1.0
    return {m: w[m] / s for m in models}


def blended_long_rate(per_model_rows, weights):
    """Fraction of ticks whose weighted-mean p_long > 0.5."""
    longs = tot = 0
    for row in per_model_rows:
        num = den = 0.0
        for m, p in row.items():
            if p is None:
                continue
            num += weights.get(m, 0.0) * p
            den += weights.get(m, 0.0)
        if den <= 0:
            continue
        tot += 1
        if num / den > 0.5:
            longs += 1
    return (100 * longs / tot if tot else 0.0), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000, help="how many recent meta rows to analyse")
    args = ap.parse_args()
    env = load_env()

    print("=" * 70)
    print("  LONG-BIAS DIAGNOSIS (read-only)")
    print("=" * 70)

    meta = tail_rows(META_CSV, args.rows)
    if not meta:
        print(f"  live_meta_log.csv not found or empty at {META_CSV}")
        return 1
    cols = list(meta[0].keys())
    model_cols = [c for c in cols if c.endswith("_p") and c != "p_meta"]
    models = [c[:-2] for c in model_cols]
    print(f"  meta rows analysed : {len(meta)}   models detected: {models}")
    print(f"  DL_MODEL_WEIGHTS   : {env.get('DL_MODEL_WEIGHTS','(unset)')}")
    print(f"  DL_MIN_AGREE={env.get('DL_MIN_AGREE','?')}   DL_P_LONG={env.get('DL_P_LONG','?')}   "
          f"DL_BIAS_*={{lstm:{env.get('DL_BIAS_LSTM','?')}, tcn:{env.get('DL_BIAS_TCN','?')}, "
          f"tx:{env.get('DL_BIAS_TX','?')}, adv:{env.get('DL_BIAS_ADV','0.0')}}}")

    # Collect per-model probability series + per-tick model dict.
    series = {m: [] for m in models}
    per_tick = []
    for r in meta:
        d = {}
        for m in models:
            p = _f(r.get(m + "_p"))
            if p is not None:
                series[m].append(p)
                d[m] = p
        per_tick.append(d)

    # ---- Q1 / Q3 / Q8 : per-model distribution -----------------------------
    print("\n" + "-" * 70)
    print("  Q1/Q3/Q8 — per-model raw probability (p_long), %LONG = p>0.5")
    print("-" * 70)
    summary = {}
    for m in models:
        v = series[m]
        if not v:
            print(f"  {m.upper():<5}  (no data)"); continue
        long_pct = 100 * sum(1 for p in v if p > 0.5) / len(v)
        summary[m] = {
            "n": len(v), "mean": st.fmean(v), "median": st.median(v),
            "std": (st.pstdev(v) if len(v) > 1 else 0.0),
            "min": min(v), "max": max(v), "long_pct": long_pct,
        }
        s = summary[m]
        print(f"\n  {m.upper():<5}  n={s['n']}  mean={s['mean']:.3f}  median={s['median']:.3f}  "
              f"std={s['std']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}  LONG={long_pct:.0f}%")
        print(hist(v))

    # ---- Q2 : driver -------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Q2 — which model drives the LONG bias")
    print("-" * 70)
    if summary:
        ranked = sorted(summary.items(), key=lambda kv: kv[1]["long_pct"], reverse=True)
        for m, s in ranked:
            print(f"  {m.upper():<5}  LONG {s['long_pct']:5.0f}%   mean p {s['mean']:.3f}   "
                  f"(median dev from 0.5: {s['median']-0.5:+.3f})")
        worst, ws = ranked[0]
        balanced = [m for m, s in summary.items() if 40 <= s["long_pct"] <= 60]
        print(f"\n  => most LONG-biased: {worst.upper()} ({ws['long_pct']:.0f}% LONG). "
              f"Balanced (40-60%): {', '.join(b.upper() for b in balanced) or 'NONE'}.")
        if all(s["long_pct"] >= 70 for s in summary.values()):
            print("  => ALL models are LONG-biased -> systemic (calibration/regime), not one model.")

    # ---- Q4 : vote distribution -------------------------------------------
    print("\n" + "-" * 70)
    print("  Q4 — per-tick LONG-vote distribution (model votes p>0.5)")
    print("-" * 70)
    votes = Counter()
    nfull = 0
    for d in per_tick:
        if len(d) != len(models) or not models:
            continue
        nfull += 1
        votes[sum(1 for p in d.values() if p > 0.5)] += 1
    k = len(models)
    for nl in range(k, -1, -1):
        c = votes.get(nl, 0)
        pct = 100 * c / nfull if nfull else 0
        label = f"{nl}/{k} LONG ({k-nl}/{k} SHORT)"
        print(f"  {label:<22} {c:6d}  {pct:5.1f}%  {'#'*int(pct/2)}")
    print(f"  (ticks with all {k} models present: {nfull})")

    # ---- Q5 : weighting ----------------------------------------------------
    print("\n" + "-" * 70)
    print("  Q5 — is the ensemble weighting causing the bias?")
    print("-" * 70)
    w = parse_weights(env, models)
    eq = {m: 1.0 / len(models) for m in models}
    print("  normalized weights:  " + "  ".join(f"{m}={w[m]:.3f}" for m in models))
    wr, _ = blended_long_rate(per_tick, w)
    er, _ = blended_long_rate(per_tick, eq)
    print(f"  blended %LONG  weighted={wr:.0f}%   equal-weight={er:.0f}%   delta={wr-er:+.0f}pp")
    print("  weighted contribution to mean p (weight x mean):")
    for m in models:
        if m in summary:
            print(f"    {m.upper():<5} {w[m]:.3f} x {summary[m]['mean']:.3f} = {w[m]*summary[m]['mean']:.3f}")
    print("  => if weighted ~ equal-weight, weighting is NOT the cause.")

    # ---- Q6 : remove TCN ---------------------------------------------------
    print("\n" + "-" * 70)
    print("  Q6 — would removing TCN improve balance?")
    print("-" * 70)
    if "tcn" in models:
        rest = [m for m in models if m != "tcn"]
        w2 = parse_weights(env, rest)
        full_r, _ = blended_long_rate(per_tick, w)
        no_tcn_r, _ = blended_long_rate([{m: d[m] for m in rest if m in d} for d in per_tick], w2)
        print(f"  blended %LONG  with TCN={full_r:.0f}%   without TCN={no_tcn_r:.0f}%   "
              f"delta={no_tcn_r-full_r:+.0f}pp")
        if "tcn" in summary:
            print(f"  TCN itself: LONG {summary['tcn']['long_pct']:.0f}%, weight {w.get('tcn',0):.3f} "
                  f"(lowest by AUC). Removing only helps if TCN is MORE biased than the rest.")
    else:
        print("  TCN not present in logs.")

    # ---- Q7 : threshold (uses signals, centered) ---------------------------
    print("\n" + "-" * 70)
    print("  Q7 — is DL_P_LONG too permissive? (threshold gates FREQUENCY, not direction)")
    print("-" * 70)
    sig = tail_rows(SIGNALS_CSV, args.rows)
    cps = [(_f(r.get("p_meta")), r.get("symbol", "?")) for r in sig]
    cps = [(p, s) for p, s in cps if p is not None]
    if cps:
        cur = _f(env.get("DL_P_LONG", "0.15")) or 0.15
        print(f"  signals analysed: {len(cps)}  (p_meta is CENTERED, LONG = p>0)")
        print(f"  {'thr':>5} {'allowed%':>9} {'amongAllowed LONG%':>20} {'SHORT%':>8}")
        for thr in sorted({0.10, 0.15, 0.20, 0.25, 0.30, cur}):
            allowed = [(p, s) for p, s in cps if abs(p) >= thr]
            ap = 100 * len(allowed) / len(cps)
            lp = 100 * sum(1 for p, _ in allowed if p > 0) / len(allowed) if allowed else 0
            mark = "  <- current" if abs(thr - cur) < 1e-9 else ""
            print(f"  {thr:5.2f} {ap:8.0f}% {lp:19.0f}% {100-lp:7.0f}%{mark}")
        print("  => raising the threshold cuts trade COUNT but barely changes LONG/SHORT split:")
        print("     it filters weak signals, it does not de-bias direction.")
    else:
        print("  live_signals.csv not found/empty.")

    # ---- Q9 : ETH vs SOL ---------------------------------------------------
    print("\n" + "-" * 70)
    print("  Q9 — do ETH and SOL have different distributions? (blended, centered)")
    print("-" * 70)
    by_sym = {}
    for p, s in cps:
        by_sym.setdefault(s, []).append(p)
    for sym in sorted(by_sym):
        v = by_sym[sym]
        long_pct = 100 * sum(1 for p in v if p > 0) / len(v)
        print(f"\n  {sym}  n={len(v)}  mean={st.fmean(v):+.3f}  median={st.median(v):+.3f}  "
              f"LONG={long_pct:.0f}%")
        print(hist(v, lo=-0.5, hi=0.5))

    # ---- Q10 ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Q10 — highest-leverage change (read the numbers above):")
    print("   - If ALL models are LONG-biased -> recenter each with DL_BIAS_* via")
    print("     tools/calibrate_temperature.py (NON-retraining). DL_BIAS_* are 0 now.")
    print("   - If ONE model dominates -> down-weight/disable it in DL_MODEL_WEIGHTS.")
    print("   - Threshold (DL_P_LONG) changes frequency, not direction — not the bias fix.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
