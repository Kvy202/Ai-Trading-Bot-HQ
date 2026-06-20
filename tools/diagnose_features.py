#!/usr/bin/env python3
r"""
diagnose_features.py — READ-ONLY. Are the models saturating because the live
SCALED features are out-of-distribution (serving/scaler problem), or are the
models themselves degenerate (needs retraining)?

Run on EC2:
    ~/bot/.venv/bin/python ~/bot/tools/diagnose_features.py

For each symbol's live window it applies each model's scaler and reports the
per-feature z-stats. If live features match the training distribution, scaled
values are ~mean 0 / std 1. Features with |mean z| >> 0 are OUT OF DISTRIBUTION
and are what push the models to a saturated (one-sided) corner.

Verdict:
  - many OOD features / large |z|  -> serving/scaler issue (fixable, no retrain):
      re-fit or re-export the scaler in the serving env (kills the version
      warning too), or the live feature regime is outside training range.
  - features in-distribution but models still saturate -> the models are
      degenerate / regime-overfit -> retraining is justified.

Writes nothing, trades nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
try:
    from runtime.loader import apply_run_config
    apply_run_config(BASE)
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env", override=True)
except Exception:
    pass


def main() -> int:
    from features import canonical_feature_columns, FEATURE_COLS
    from ml_dl.dl_ensemble import (
        load_ensemble, refresh_live_features_per_symbol, predict_ensemble,
        common_scaler_dim, resolve_add_symbol_id,
    )
    from data import executor_symbol

    syms = [s.strip() for s in os.getenv(
        "DL_SYMBOLS", os.getenv("SYMBOL_WHITELIST", "ETHUSDT,SOLUSDT")).split(",") if s.strip()]
    tf = os.getenv("DL_TIMEFRAME", "5m")
    seq = int(os.getenv("DL_SEQ_LEN", "64"))

    print("=" * 72 + "\n  FEATURE-DISTRIBUTION DIAGNOSIS (read-only)\n" + "=" * 72)
    models, dev = load_ensemble(X_dim=len(FEATURE_COLS) + 1, device=None)
    scaler_dim = common_scaler_dim(models)
    add_sid = resolve_add_symbol_id(scaler_dim)
    cols = canonical_feature_columns(add_sid)
    print(f"  symbols={syms} tf={tf} seq={seq} add_symbol_id={add_sid} n_features={scaler_dim}")

    meta, windows = refresh_live_features_per_symbol(
        seq_len=seq, add_symbol_id=add_sid,
        lookback_pad=int(os.getenv("DL_MAX_LOOKBACK_PAD", "6000")),
        symbols=syms, timeframe=tf)

    OOD_Z = 2.0          # |mean z| above this = out-of-distribution feature
    worst_overall = 0
    for sym, win in windows.items():
        win = np.asarray(win, dtype=np.float32)
        print("\n" + "-" * 72 + f"\n  {sym}   window={win.shape}\n" + "-" * 72)
        # Per-model prediction (raw, DL_BIAS=0) to show the saturation alongside.
        per_model, agg = predict_ensemble(win, models, dev, None, symbol=sym)
        print("  per-model p_long: " +
              "  ".join(f"{k}={v[2]:.3f}" for k, v in sorted(per_model.items())))

        # Scaler distribution check (use the shared/first scaler; report per model
        # only if they differ materially).
        for k in sorted(models):
            sc = models[k]["scaler"]
            z = sc.transform(win)                       # [seq, F]
            zmean = z.mean(axis=0)
            frac_hi = float((np.abs(z) > 3).mean()) * 100.0
            ood = sorted(
                [(cols[i], float(zmean[i])) for i in range(len(cols)) if abs(zmean[i]) > OOD_Z],
                key=lambda t: -abs(t[1]))
            worst_overall = max(worst_overall, len(ood))
            head = ", ".join(f"{name}={z_:+.1f}sd" for name, z_ in ood[:6])
            print(f"    {k.upper():<5} |z|>3: {frac_hi:4.1f}%   OOD features (|mean z|>{OOD_Z:.0f}): "
                  f"{len(ood)}/{len(cols)}   {head}")

        # OOD mechanism: raw live value vs the training scaler's mean_/scale_.
        # If train_scale ~ 0 -> feature had ~no variance in training (re-fit scaler).
        # If live values are huge -> feature is exploding live (clip / fix feature).
        sc0 = models[sorted(models)[0]]["scaler"]
        mean_ = getattr(sc0, "mean_", None)
        scale_ = getattr(sc0, "scale_", None)
        zmean0 = sc0.transform(win).mean(axis=0)
        ood_idx = [i for i in range(len(cols)) if abs(zmean0[i]) > OOD_Z]
        if ood_idx and mean_ is not None and scale_ is not None:
            print("    OOD detail (raw live window vs training scaler):")
            print(f"      {'feature':<14}{'live_min':>12}{'live_max':>12}{'live_mean':>12}"
                  f"{'train_mean':>13}{'train_scale':>13}{'z(mean)':>10}")
            for i in ood_idx:
                col = win[:, i]
                print(f"      {cols[i]:<14}{col.min():12.4g}{col.max():12.4g}{col.mean():12.4g}"
                      f"{float(mean_[i]):13.4g}{float(scale_[i]):13.4g}{zmean0[i]:10.1f}")

    print("\n" + "=" * 72)
    print("  Read the per-model p_long lines above first: if they are spread/two-sided,")
    print("  the models are FINE and there is no saturation to explain.")
    if worst_overall >= 3:
        print("\n  If the models ARE saturated (one-sided p_long) AND you see OOD features above:")
        print("  VERDICT -> SERVING/SCALER issue (fixable WITHOUT retraining):")
        print("    - re-export each scaler in the serving sklearn version (kills the 1.8.0/1.7.2")
        print("      InconsistentVersionWarning), and/or the live regime is outside training range.")
        print("    - inspect the OOD features above against scaler.mean_/scale_.")
    else:
        print("\n  If the models ARE saturated (one-sided p_long) but features are in-distribution")
        print("  (few/no OOD above):")
        print("  VERDICT -> DEGENERATE / regime-overfit models. The inputs are clean, so the")
        print("  models themselves map everything to one side -> retraining is justified.")
        print("  (calibration can only recenter the median; it cannot restore lost variance.)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise SystemExit(2)
