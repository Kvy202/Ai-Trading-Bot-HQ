# dry_run_gate.py
import os, time, numpy as np, joblib
from ml_dl.dl_ensemble import load_ensemble, predict_ensemble
from data import load_prices_and_features

# --- config
SEQ_LEN   = int(os.getenv("DL_SEQ_LEN", "32"))
ADD_ID    = os.getenv("DL_ADD_SYMBOL_ID", "1").lower() in ("1","true","yes")
SYMS      = [s.strip() for s in os.getenv("SYMBOL_WHITELIST", "BTC/USDT:USDT").split(",") if s.strip()]
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
P_LONG_MIN= float(os.getenv("DL_P_LONG", "0.55"))
RV_MAX    = float(os.getenv("DL_MAX_RV", "60"))
SNOOZE    = int(os.getenv("DL_POLL_SECS", "5"))
DEVICE    = os.getenv("DL_DEVICE", "cpu")

meta = joblib.load(os.getenv("DL_META_PATH", "model_artifacts/meta_logreg_h24.joblib"))
scaler_m, clf = meta["scaler"], meta["clf"]

# preload features
X, _ = load_prices_and_features(symbols=SYMS, timeframe=TIMEFRAME, lookback=10_000, add_symbol_id=ADD_ID, return_dfs=False)
X_dim = X.shape[1]
models, device = load_ensemble(X_dim, device=DEVICE)

def fake_micro_gates(t):  # deterministic, repeatable “spread/liquidity” from t
    rng = np.random.default_rng(t)
    out = {}
    for s in SYMS:
        # spread ~ [0.02, 0.50], liq ~ [10k, 100k]
        spr = float(rng.uniform(0.02, 0.50))
        liq = float(rng.uniform(10_000, 100_000))
        ok  = int((spr <= float(os.getenv("MAX_SPREAD", "0.50"))) and (liq >= float(os.getenv("MIN_LIQ", "50000"))))
        out[s] = {"ok": ok, "spread": spr, "liq": liq}
    return out

# walk through last N windows as if live
for t in range(SEQ_LEN, len(X)):
    xw = X[t-SEQ_LEN:t, :]
    per_model, _ = predict_ensemble(xw, models, device, None)
    # meta features in consistent order
    feats = []
    for k in ["tcn","tx","lstm"]:
        if k not in per_model:  # skip if model not loaded
            continue
        ret_hat, rv_hat, p_long = per_model[k]
        feats.extend([p_long, rv_hat, ret_hat])
    if not feats:
        continue
    Xm_s = scaler_m.transform(np.array(feats, dtype=np.float32).reshape(1, -1))
    p_meta = float(clf.predict_proba(Xm_s)[0, 1])
    rv_mean = float(np.mean([v[1] for v in per_model.values()]))

    gates = fake_micro_gates(t)
    all_ok = all(v["ok"]==1 for v in gates.values())
    allow = (p_meta >= P_LONG_MIN) and (rv_mean <= RV_MAX) and all_ok

    gate_str = " ".join([f"{s.split(':')[0]}[ok={v['ok']},spr={v['spread']:.3f},liq={v['liq']:.0f}]" for s,v in gates.items()])
    base_str = " ".join([f"{k}:p={v[2]:.3f},rv={v[1]:.3f}" for k,v in per_model.items()])
    print(f"t={t} META p={p_meta:.3f} rv̄={rv_mean:.2f} allow={str(allow):5} | {gate_str} | {base_str}")

    # throttle a bit so output is readable
    if t % 50 == 0:
        time.sleep(0.2)
