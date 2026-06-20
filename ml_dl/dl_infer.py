# ml_dl/dl_infer.py
import os
from typing import Tuple

import joblib
import numpy as np
import torch

from .dl_models import TemporalConvNet, TinyTransformer
try:
    from .dl_models import TinyLSTM as LSTMBackbone
except Exception:
    try:
        from .dl_models import LSTMBackbone  # type: ignore
    except Exception:
        LSTMBackbone = None

try:
    from .dl_models_adv import AdvancedTransformer
except Exception:
    AdvancedTransformer = None


def _pick_device(pref: str = "auto") -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _build_model(kind: str, in_dim_eff: int):
    k = kind.lower()
    if k == "tcn":
        return TemporalConvNet(in_dim_eff)
    if k == "tx":
        return TinyTransformer(in_dim_eff)
    if k == "lstm":
        if LSTMBackbone is None:
            raise ImportError(
                "LSTM backbone not found. Define TinyLSTM or LSTMBackbone in ml_dl/dl_models.py."
            )
        return LSTMBackbone(in_dim_eff)
    if k in ("adv", "tft", "transformer"):
        if AdvancedTransformer is None:
            raise ImportError("AdvancedTransformer not found. Ensure ml_dl/dl_models_adv.py exists.")
        return AdvancedTransformer(in_dim_eff)
    raise ValueError(f"Unknown model kind: {kind!r} (expected 'tcn', 'tx', 'lstm', or 'adv')")


def load_model(kind: str, in_dim: int, scaler_path: str, model_path: str, device: str = "auto"):
    dev = _pick_device(device)

    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    scaler = joblib.load(scaler_path)
    in_dim_eff = int(getattr(scaler, "n_features_in_", in_dim) or in_dim)

    model = _build_model(kind, in_dim_eff)

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load model weights (shape mismatch).\n"
            f"  kind={kind}  effective_in_dim={in_dim_eff}\n"
            f"  scaler.n_features_in_={getattr(scaler, 'n_features_in_', None)}  "
            f"fallback_in_dim={in_dim}\n"
            "This usually means your inference feature set differs from training. "
            "Use the same scaler/feature pipeline as during training."
        ) from e

    model.eval()
    model.to(dev)
    return scaler, model, dev


@torch.no_grad()
def predict_next(
    x_window: np.ndarray,  # shape [seq_len, F]
    scaler,
    model,
    device: str = "cpu",
) -> Tuple[float, float, float]:
    """
    Forward pass on the latest sequence window.

    Model must return a dict with:
      - out['ret_reg']       : regression head for next-k return (shape [B] or [B,1])
      - out['rv_reg']        : regression head for next-k realized vol (shape [B] or [B,1])
      - out['ret_cls_logits']: classification logits (shape [B, 2])
    """
    # Scale with the training scaler
    x = scaler.transform(x_window.astype(np.float32, copy=False))  # [L, F]
    x_t = torch.from_numpy(x[None, ...]).to(device)                # [1, L, F]

    out = model(x_t)
    for k in ("ret_reg", "rv_reg", "ret_cls_logits"):
        if k not in out:
            raise KeyError(f"Model output missing '{k}'")

    ret_hat = float(out["ret_reg"].squeeze().detach().cpu().numpy())
    rv_hat  = float(out["rv_reg"].squeeze().detach().cpu().numpy())
    p_long  = float(torch.softmax(out["ret_cls_logits"], dim=-1)[0, 1].item())
    p_long  = float(np.clip(p_long, 1e-6, 1 - 1e-6))  # numerical guard

    return ret_hat, rv_hat, p_long


# --- High-level wrapper used by trade_multi_bitget.py ---
# Caches loaded (scaler, model, device) per (kind, scaler_path, model_path)
# so we don't reload from disk on every bar.
_INFER_CACHE: dict = {}


def infer_scores(
    X_windowed: np.ndarray,   # shape [L, F]
    scaler_path: str,
    model_path: str,
    kind: str = "tcn",
    device: str = "cpu",
) -> dict:
    """
    Convenience wrapper around load_model + predict_next.

    Returns a dict:
        {"p_long": float, "ret_hat": float, "rv_hat": float}

    On any failure, returns a dict of NaNs rather than raising, so the
    caller can fall back to the classical signal.
    """
    try:
        if X_windowed is None or np.asarray(X_windowed).ndim != 2:
            raise ValueError("X_windowed must be a 2-D array of shape [L, F]")

        in_dim_fallback = int(np.asarray(X_windowed).shape[1])
        key = (str(kind).lower(), str(scaler_path), str(model_path))
        cached = _INFER_CACHE.get(key)
        if cached is None:
            scaler, model, dev = load_model(
                kind=kind,
                in_dim=in_dim_fallback,
                scaler_path=scaler_path,
                model_path=model_path,
                device=device,
            )
            cached = (scaler, model, dev)
            _INFER_CACHE[key] = cached
        scaler, model, dev = cached

        ret_hat, rv_hat, p_long = predict_next(
            x_window=np.asarray(X_windowed, dtype=np.float32),
            scaler=scaler,
            model=model,
            device=dev,
        )
        return {"p_long": float(p_long), "ret_hat": float(ret_hat), "rv_hat": float(rv_hat)}
    except Exception:
        return {"p_long": float("nan"), "ret_hat": float("nan"), "rv_hat": float("nan")}
