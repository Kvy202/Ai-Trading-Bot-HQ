# live_ensemble.py
import os
import time
from datetime import datetime, timezone

from ml_dl.dl_ensemble import load_ensemble, predict_ensemble, refresh_live_features
from data import load_prices_and_features  # just to validate feature count at startup

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")

def main():
    # --------- config (env-driven) ----------
    ADD_SYMBOL_ID = os.getenv("DL_ADD_SYMBOL_ID", "1") in ("1","true","True")
    SEQ_LEN  = int(os.getenv("DL_SEQ_LEN", "64"))

    # gating
    P_LONG_MIN = float(os.getenv("DL_P_LONG", "0.55"))
    RV_MAX     = float(os.getenv("DL_MAX_RV",  "0.02"))
    SNOOZE_SEC = int(os.getenv("DL_POLL_SECS", "15"))   # how long to wait between polls (placeholder)

    # ensemble weights (optional)
    # ex: DL_W_TCN=0.6, DL_W_LSTM=0.2, DL_W_TX=0.2
    W = {
        "tcn": float(os.getenv("DL_W_TCN", "1.0")),
        "lstm": float(os.getenv("DL_W_LSTM", "0.0")),
        "tx": float(os.getenv("DL_W_TX", "0.0")),
    }

    # --------- startup: learn feature dimension ----------
    X0, _ = load_prices_and_features(lookback=SEQ_LEN+200, add_symbol_id=ADD_SYMBOL_ID)
    X_dim = X0.shape[1]

    # --------- load ensemble ----------
    models, device = load_ensemble(X_dim)
    print(f"[{now()}] ensemble loaded: kinds={list(models.keys())}  X_dim={X_dim}  device={device}")

    # --------- main loop (stub: poll every X seconds) ----------
    while True:
        try:
            # 1) refresh features
            _, xw = refresh_live_features(SEQ_LEN, ADD_SYMBOL_ID, lookback_pad=200)

            # 2) predict per-model and blended
            per_model, (ret_hat, rv_hat, p_long) = predict_ensemble(xw, models, device, weights=W)

            # 3) gating
            allow_entry = (p_long >= P_LONG_MIN) and (rv_hat <= RV_MAX)

            # 4) log/act
            pm_str = " | ".join([f"{k}: p={per_model[k][2]:.3f}, rv={per_model[k][1]:.3f}, r={per_model[k][0]:.4f}"
                                 for k in per_model])
            print(f"[{now()}] BLEND  p={p_long:.3f}  rv={rv_hat:.3f}  r={ret_hat:.4f}  allow={allow_entry}  ||  {pm_str}")

            # TODO: send to your bot/router:
            # if allow_entry: place_order(...)
        except Exception as e:
            print(f"[{now()}] ERROR: {e}")

        time.sleep(SNOOZE_SEC)

if __name__ == "__main__":
    main()
