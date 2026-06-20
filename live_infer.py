import os, time
import pandas as pd
from data import load_prices_and_features
from ml_dl.dl_infer import load_model, predict_next

def env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"

def main():
    # env / defaults
    kind         = os.getenv("DL_MODEL_KIND", "tcn")
    scaler_path  = os.getenv("DL_SCALER_PATH", "model_artifacts/scaler_latest.joblib")
    model_path   = os.getenv("DL_MODEL_PATH",  "model_artifacts/dl_tcn_latest.pt")
    add_symbol   = env_bool("DL_ADD_SYMBOL_ID", "1")   # keep consistent with training
    seq_len      = int(os.getenv("DL_SEQ_LEN", "64"))
    p_long_min   = float(os.getenv("DL_P_LONG", "0.55"))
    rv_max       = float(os.getenv("DL_MAX_RV",  "0.02"))
    poll_secs    = int(os.getenv("POLL_SECS", "5"))
    lookback     = int(os.getenv("SANITY_LOOKBACK", str(seq_len + 300)))

    # initial feature pull to discover X_dim
    X0, _ = load_prices_and_features(lookback=lookback, add_symbol_id=add_symbol)
    X_dim = X0.shape[1]

    scaler, model, device = load_model(
        kind=kind, in_dim=X_dim,
        scaler_path=scaler_path, model_path=model_path,
        device=os.getenv("DL_DEVICE", "auto")
    )
    print(f"[live] model loaded (kind={kind}, device={device}, X_dim={X_dim}, seq_len={seq_len})")
    print(f"[live] gates: p_long_min={p_long_min}  rv_max={rv_max}")

    last_ts = None
    while True:
        try:
            X, prices = load_prices_and_features(lookback=lookback, add_symbol_id=add_symbol)
            # use price index to detect new bar (falls back gracefully if no DatetimeIndex)
            cur_ts = None
            if hasattr(prices, "index") and len(prices.index):
                cur_ts = prices.index[-1]

            if cur_ts != last_ts:
                last_ts = cur_ts
                xw = X[-seq_len:, :]
                ret_hat, rv_hat, p_long = predict_next(xw, scaler, model, device)
                allow = (p_long >= p_long_min) and (rv_hat <= rv_max)

                print(f"[{pd.Timestamp.utcnow()}] ret={ret_hat:.6f} "
                      f"rv={rv_hat:.6f} p_long={p_long:.3f} allow_entry={allow}")

                # TODO: plug your exchange/order logic here if allow is True
                #       and your other guards (spread, cooldown, position limits) pass.

            time.sleep(poll_secs)
        except Exception as e:
            print("[live] error:", e)
            time.sleep(max(10, poll_secs))

if __name__ == "__main__":
    main()
