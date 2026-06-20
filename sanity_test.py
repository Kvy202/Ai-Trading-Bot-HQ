# sanity_test.py
import os
import sys
import traceback
import numpy as np

def _b(env: str, default: bool = False) -> bool:
    return os.getenv(env, "1" if default else "0").lower() in ("1","true","yes","y")

def _get_model_in_dim(model) -> int | None:
    """
    Try to infer the model's expected feature dimension.
    Works for our TCN/TX implementations by checking common attrs or first layer shape.
    """
    # Preferred: many of our models set this during __init__
    in_dim = getattr(model, "in_dim", None)
    if isinstance(in_dim, (int, np.integer)) and in_dim > 0:
        return int(in_dim)

    # Heuristic: look for a first conv/linear layer with weight [..., in_dim, ...]
    for attr in ("net", "encoder", "input_proj", "stem", "feature_proj"):
        layer = getattr(model, attr, None)
        if layer is None:
            continue
        # Try torch layers without importing torch types explicitly
        w = getattr(layer, "weight", None)
        if w is not None and hasattr(w, "shape") and len(w.shape) >= 2:
            try:
                return int(w.shape[1])
            except Exception:
                pass
        # Some models store submodules in lists/Sequential
        for maybe in getattr(layer, "_modules", {}).values():
            w2 = getattr(maybe, "weight", None)
            if w2 is not None and hasattr(w2, "shape") and len(w2.shape) >= 2:
                try:
                    return int(w2.shape[1])
                except Exception:
                    pass
    return None


def main():
    # --- env/config ---
    kind         = os.getenv("DL_MODEL_KIND", "tcn")                      # "tcn" | "tx" | "lstm"
    scaler_path  = os.getenv("DL_SCALER_PATH", "model_artifacts/scaler_latest.joblib")
    model_path   = os.getenv("DL_MODEL_PATH",  "model_artifacts/dl_tcn_latest.pt")
    add_sym_id   = _b("DL_ADD_SYMBOL_ID", True)
    seq_len      = int(os.getenv("DL_SEQ_LEN", "32"))
    lookback     = int(os.getenv("SANITY_LOOKBACK", "1500"))
    device_pref  = os.getenv("DL_DEVICE", "auto")                         # "auto" | "cpu" | "cuda"
    symbols      = os.getenv("SYMBOL_WHITELIST", "").strip() or None
    timeframe    = os.getenv("TIMEFRAME", "5m")

    print("=== sanity_test.py ===")
    print(f"kind={kind}  scaler={scaler_path}  model={model_path}")
    print(f"add_symbol_id={add_sym_id}  seq_len={seq_len}  lookback={lookback}  device={device_pref}")
    if symbols:
        print(f"symbols={symbols}  timeframe={timeframe}")

    # --- imports that require your venv (torch etc.) ---
    try:
        import torch
        from data import load_prices_and_features
        from ml_dl.dl_infer import load_model, predict_next
    except Exception:
        print("\n[ERROR] Failed to import modules. Are you in the virtual environment?")
        print("Hint (PowerShell):  & .\\.venv\\Scripts\\Activate.ps1")
        traceback.print_exc()
        sys.exit(1)

    if device_pref == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; falling back to CPU.")
        device_pref = "cpu"

    # --- load features (match training env: symbols/timeframe) ---
    try:
        X, prices = load_prices_and_features(
            symbols=[s.strip() for s in symbols.split(",")] if symbols else None,
            timeframe=timeframe,
            lookback=lookback,
            add_symbol_id=add_sym_id,
            return_dfs=False,
        )
    except Exception:
        print("\n[ERROR] load_prices_and_features() failed.")
        traceback.print_exc()
        sys.exit(1)

    if X.shape[0] < seq_len:
        print(f"\n[ERROR] Not enough rows: have {X.shape[0]}, need >= seq_len={seq_len}. "
              f"Increase SANITY_LOOKBACK or reduce DL_SEQ_LEN.")
        sys.exit(1)

    print(f"[data] X shape={X.shape}  prices={prices.shape if prices is not None else None}")

    # --- load model/scaler (trust scaler for feature count internally) ---
    try:
        scaler, model, device = load_model(
            kind=kind,
            in_dim=X.shape[1],           # load_model will override with scaler.n_features_in_ if present
            scaler_path=scaler_path,
            model_path=model_path,
            device=device_pref
        )
    except RuntimeError:
        print("\n[ERROR] Model load failed — common cause is feature-count mismatch.")
        print("       Check that your runtime features match training.")
        traceback.print_exc()
        sys.exit(1)
    except Exception:
        print("\n[ERROR] Model load failed.")
        traceback.print_exc()
        sys.exit(1)

    scaler_n = getattr(scaler, "n_features_in_", None)
    model_in = _get_model_in_dim(model)
    X_dim    = X.shape[1]

    print(f"[model] device={device}  seq_len={seq_len}")
    print(f"[dims]  X_dim={X_dim}  scaler_n_features_in_={scaler_n}  model_in_dim={model_in}")

    # sanity checks / warnings
    if scaler_n is not None and scaler_n != X_dim:
        print(f"[WARN] Runtime X_dim ({X_dim}) != scaler_n_features_in_ ({scaler_n}). "
              f"Your data pipeline may not match training features.")
    if model_in is not None and scaler_n is not None and model_in != scaler_n:
        print(f"[WARN] Model expects {model_in} features but scaler expects {scaler_n}. "
              f"Check your checkpoint vs. scaler pairing.")

    # --- last window & predict ---
    xw = X[-seq_len:, :]    # shape (seq_len, F)
    if xw.shape[0] != seq_len:
        print(f"\n[ERROR] Window wrong length: {xw.shape[0]} != {seq_len}")
        sys.exit(1)

    try:
        ret_hat, rv_hat, p_long = predict_next(xw, scaler, model, device)
    except Exception:
        print("\n[ERROR] predict_next() failed.")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Prediction ===")
    print(f"window:    {xw.shape}")
    print(f"ret_hat:   {ret_hat:.6f}  (predicted next-k log return)")
    print(f"rv_hat:    {rv_hat:.6f}  (predicted realized vol)")
    print(f"p_long:    {p_long:.6f}  (probability of positive return)")

    # optional gating preview using env thresholds
    p_long_min = float(os.getenv("DL_P_LONG", "0.55"))
    rv_max     = float(os.getenv("DL_MAX_RV",  "0.02"))
    allow      = (p_long >= p_long_min) and (rv_hat <= rv_max)
    print("\n=== Gating (preview) ===")
    print(f"DL_P_LONG={p_long_min}  DL_MAX_RV={rv_max}  -> allow_entry={allow}")

if __name__ == "__main__":
    main()
