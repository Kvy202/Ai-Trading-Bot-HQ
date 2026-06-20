# live_meta_ensemble.py
import os, time, joblib, numpy as np
from collections import deque
from datetime import datetime, timezone

from ml_dl.dl_ensemble import load_ensemble, predict_ensemble, refresh_live_features
from ml_dl.micro_gates import micro_gates
from data import load_prices_and_features


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def _b(env: str, default: bool = False) -> bool:
    return os.getenv(env, "1" if default else "0").lower() in ("1", "true", "yes", "y")


def _maybe_load_platt() -> dict:
    """
    Optionally load Platt calibrators for base models.
    Env:
      DL_PLATT_TCN, DL_PLATT_TX, DL_PLATT_LSTM  -> joblib LogisticRegression
      DL_PLATT_BEFORE_META (default 0)          -> if 1, apply Platt to base p before meta
    """
    out = {}
    for k, env in (("tcn", "DL_PLATT_TCN"), ("tx", "DL_PLATT_TX"), ("lstm", "DL_PLATT_LSTM")):
        pth = os.getenv(env, "")
        if pth and os.path.exists(pth):
            try:
                out[k] = joblib.load(pth)
            except Exception:
                pass
    return out


def _apply_platt(clf, p: float) -> float:
    # logistic regression on logit(p)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    x = np.log(p / (1.0 - p)).reshape(1, 1)
    q = clf.predict_proba(x)[0, 1]
    return float(np.clip(q, 1e-6, 1 - 1e-6))


def main():
    # ---------- runtime knobs ----------
    ADD_SYMBOL_ID = _b("DL_ADD_SYMBOL_ID", True)
    SEQ_LEN       = int(os.getenv("DL_SEQ_LEN", "32"))

    # gating: absolute, percentile, or calibrated (abs|pctl)
    MODE          = os.getenv("DL_P_LONG_MODE", "abs").lower()   # "abs" or "pctl"
    P_LONG        = float(os.getenv("DL_P_LONG", "0.43"))        # if pctl, 0.9 = 90th pct of rolling buffer
    BUF_MAX       = int(os.getenv("DL_P_LONG_BUF", "2000"))      # buffer for percentile mode
    WARM_MIN      = max(100, int(0.05 * BUF_MAX))                # min obs before pctl kicks in

    RV_MAX        = float(os.getenv("DL_MAX_RV",  "60"))
    RV_AGG        = os.getenv("DL_RV_AGG", "median").lower()     # "median" | "mean"
    RV_CAP_PCTL   = float(os.getenv("DL_RV_CAP_PCTL", "0"))      # e.g. 0.98 to hard-cap RV outliers
    SNOOZE        = int(os.getenv("DL_POLL_SECS", "15"))
    DEVICE        = os.getenv("DL_DEVICE", "cpu")

    # microstructure (policy enforced inside micro_gates; here we just log)
    MAX_SPREAD     = float(os.getenv("MAX_SPREAD", "0.30"))
    MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "0"))
    MIN_LIQ        = float(os.getenv("MIN_LIQ", "50000"))

    # optional CSV logging (for later win-rate/Sharpe over live)
    LOG_PATH       = os.getenv("DL_LOG_PATH", "")  # e.g. "live_meta_log.csv"

    # keep symbols/timeframe identical to training
    symbols   = os.getenv("SYMBOL_WHITELIST", "BTC/USDT:USDT,ETH/USDT:USDT,DOGE/USDT:USDT")
    timeframe = os.getenv("TIMEFRAME", "5m")
    sym_list  = [s.strip() for s in symbols.split(",") if s.strip()]

    # load meta head
    meta_path = os.getenv("DL_META_PATH", "model_artifacts/meta_logreg_h24_cls2.joblib")
    meta = joblib.load(meta_path)
    scaler_m, clf = meta["scaler"], meta["clf"]

    # meta schema
    kinds_from_artifact = meta.get("kinds_order")
    kinds_env = [k.strip() for k in os.getenv("META_KINDS", "").split(",") if k.strip()]
    features_per_kind = int(meta.get("features_per_kind", 3))  # expected [p_long, rv, ret]

    # learn feature dim with the same env settings (ensures ensemble input width)
    X0, _ = load_prices_and_features(
        symbols=sym_list, timeframe=timeframe,
        lookback=max(SEQ_LEN + 200, 5000),
        add_symbol_id=ADD_SYMBOL_ID, return_dfs=False
    )
    X_dim = X0.shape[1]

    # load base models
    models, device = load_ensemble(X_dim, device=DEVICE)

    # resolve kinds order
    if kinds_from_artifact:
        kinds_meta = list(kinds_from_artifact)
    elif kinds_env:
        kinds_meta = kinds_env
    else:
        kinds_meta = sorted(models.keys())  # matches meta_train’s default sort

    # optional platt (base p_long before meta)
    platt = _maybe_load_platt()
    use_platt_before_meta = _b("DL_PLATT_BEFORE_META", False)

    # sanity: ensure every kind in kinds_meta is loaded
    missing = [k for k in kinds_meta if k not in models]
    if missing:
        print(f"[{now()}] WARNING: meta expects {kinds_meta} but missing {missing} in loaded models {list(models.keys())}")

    print(f"[{now()}] live meta ready: {sorted(models.keys())} dev={device} | "
          f"META_KINDS={kinds_meta} | gates: MAX_SPREAD={MAX_SPREAD} MAX_SPREAD_BPS={MAX_SPREAD_BPS} MIN_LIQ={MIN_LIQ} | "
          f"mode={MODE} p_long={P_LONG} buf={BUF_MAX} | features_per_kind={features_per_kind} | rv_agg={RV_AGG} "
          f"| log_path={(LOG_PATH or 'disabled')}")

    # percentile buffers
    p_buf = deque(maxlen=BUF_MAX)
    rv_buf = deque(maxlen=max(512, BUF_MAX // 2))  # for optional RV cap

    # csv header
    if LOG_PATH and not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            cols = ["ts","p_meta","thr","mode","rv_mean","allow","kinds_used"]
            for k in kinds_meta:
                cols += [f"{k}_p", f"{k}_rv", f"{k}_ret"]
            cols += ["spread_summary","liq_summary"]
            f.write(",".join(cols) + "\n")

    try:
        while True:
            try:
                # refresh features
                _, xw = refresh_live_features(SEQ_LEN, ADD_SYMBOL_ID, lookback_pad=2000)

                # per-model predictions
                per_model, _ = predict_ensemble(xw, models, device)  # {kind: (ret, rv, p)}
                # optional platt before meta
                if use_platt_before_meta and len(platt):
                    tmp = {}
                    for k, (ret_hat, rv_hat, p_long) in per_model.items():
                        if k in platt:
                            p_long = _apply_platt(platt[k], p_long)
                        tmp[k] = (ret_hat, rv_hat, p_long)
                    per_model = tmp

                # meta features in declared order (p, rv, ret per model)
                feats, used_kinds = [], []
                for k in kinds_meta:
                    if k not in per_model:
                        continue
                    ret_hat, rv_hat, p_long = per_model[k]
                    feats.extend([p_long, rv_hat, ret_hat])
                    used_kinds.append(k)

                if not used_kinds:
                    print(f"{now()} ERROR no models available for META_KINDS={kinds_meta} | have={list(per_model.keys())}")
                    time.sleep(SNOOZE)
                    continue

                Xm = np.array(feats, dtype=np.float32).reshape(1, -1)

                # dimension sanity vs scaler
                expected_dim = getattr(scaler_m, "n_features_in_", None)
                if expected_dim is not None and Xm.shape[1] != expected_dim:
                    print(f"{now()} ERROR (meta dim): got {Xm.shape[1]} features but scaler expects {expected_dim} "
                          f"| used_kinds={used_kinds} features_per_kind={features_per_kind}")
                    time.sleep(SNOOZE)
                    continue

                Xm_s = scaler_m.transform(Xm)
                p_meta = float(np.clip(joblib.parallel_backend.__self__ if False else  # noop to keep joblib import
                                       0.0, 0.0, 1.0))  # placeholder to keep static analyzers happy
                p_meta = float(clf.predict_proba(Xm_s)[0, 1])

                # aggregate RV across used models
                rv_vals = np.array([per_model[k][1] for k in used_kinds], dtype=np.float32)
                rv_mean = float(np.median(rv_vals) if RV_AGG != "mean" else np.mean(rv_vals))

                # optional RV cap by rolling percentile (defensive against spikes)
                if RV_CAP_PCTL > 0 and len(rv_buf) >= 64:
                    cap = float(np.quantile(np.array(rv_buf, dtype=np.float32), RV_CAP_PCTL))
                    rv_mean = float(min(rv_mean, cap))

                # update buffers & compute threshold
                p_buf.append(p_meta); rv_buf.append(rv_mean)
                thr = (float(np.quantile(np.asarray(p_buf, dtype=np.float32), P_LONG))
                       if MODE == "pctl" and len(p_buf) >= WARM_MIN else P_LONG)

                # microstructure
                gates = micro_gates(sym_list)  # {sym: {ok, spread, liq, best_bid, best_ask}}
                all_ok = all(v["ok"] == 1 for v in gates.values())

                allow = (p_meta >= thr) and (rv_mean <= RV_MAX) and all_ok

                # logging
                base_str = " ".join([f"{k}:p={per_model[k][2]:.3f},rv={per_model[k][1]:.3f}" for k in used_kinds])
                gate_str = " ".join([f"{s.split(':')[0]}[ok={v['ok']},spr={v['spread']:.4f},liq={v['liq']:.0f}]"
                                     for s, v in gates.items()])
                print(f"{now()} META p={p_meta:.3f} thr={thr:.3f} mode={MODE} rv̄={rv_mean:.3f} allow={str(allow):5} | "
                      f"{gate_str} | {base_str}")

                if LOG_PATH:
                    spreads = ";".join([f"{s.split(':')[0]}:{v['spread']:.6f}" for s, v in gates.items()])
                    liqs    = ";".join([f"{s.split(':')[0]}:{v['liq']:.0f}"    for s, v in gates.items()])
                    row = [now(), f"{p_meta:.6f}", f"{thr:.6f}", MODE, f"{rv_mean:.6f}", int(allow), "|".join(used_kinds)]
                    for k in kinds_meta:
                        if k in per_model:
                            ret_hat, rv_hat, p_long = per_model[k]
                            row += [f"{p_long:.6f}", f"{rv_hat:.6f}", f"{ret_hat:.6f}"]
                        else:
                            row += ["", "", ""]
                    row += [spreads, liqs]
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(",".join(map(str, row)) + "\n")

                # if allow: place_order(...)

            except Exception as e:
                print(f"{now()} ERROR {e}")

            time.sleep(SNOOZE)

    except KeyboardInterrupt:
        print(f"{now()} bye.")


if __name__ == "__main__":
    main()
