# ml_dl/dl_train.py
import copy
import json
import math
import os as os_mod
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dl_models import TemporalConvNet, TinyTransformer, TinyLSTM
try:
    from .dl_models_adv import AdvancedTransformer
except Exception:
    AdvancedTransformer = None  # type: ignore

from .dl_metrics import auc, mse_mae, information_coefficient, calibration_ece
from .dl_labels import next_k_logret, next_k_rv, binarize_return, triple_barrier_label
from .dl_dataset import SeqDataset, load_prices_and_features
from .dl_walkforward import rolling_windows


# ---------------- model factory ----------------
def make_model(kind: str, in_dim: int):
    if kind == "tcn":
        return TemporalConvNet(in_dim)
    if kind == "tx":
        return TinyTransformer(in_dim)
    if kind == "lstm":
        return TinyLSTM(in_dim)
    if kind in ("adv", "tft", "transformer"):
        if AdvancedTransformer is None:
            raise ValueError("Advanced model not available. Ensure ml_dl/dl_models_adv.py exists.")
        return AdvancedTransformer(in_dim)
    raise ValueError("kind must be one of 'tcn', 'tx', 'lstm' or 'adv'/'tft'")


def _slice_len(s):
    if isinstance(s, slice):
        return s.stop - s.start
    return len(s)


def _align_global_to_X(T, prices, r, rv, y_cls):
    if not (len(r) == len(rv) == len(y_cls)):
        raise RuntimeError(f"Label arrays disagree: r={len(r)} rv={len(rv)} y={len(y_cls)}")
    if len(r) != T:
        r = r[-T:]
        rv = rv[-T:]
        y_cls = y_cls[-T:]
        if isinstance(prices, np.ndarray) and len(prices) >= T:
            prices = prices[-T:]
    return prices, r, rv, y_cls


def _align_fold_arrays(Xp, r, y, rv):
    n = min(len(Xp), len(r), len(y), len(rv))
    if not (len(Xp) == len(r) == len(y) == len(rv)):
        print(f"[WARN] aligning fold lengths: X={len(Xp)} r={len(r)} y={len(y)} rv={len(rv)} -> {n}")
    return Xp[:n], r[:n], y[:n], rv[:n]


# ---------------- training loop ----------------
def train_once(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: str,
    epochs: int = 40,
    patience: int = 8,
    lr: float = 1e-3,
    class_weights: Optional[torch.Tensor] = None,
) -> dict:
    crit_cls = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    best_auc, best_loss, best_state, bad = -1.0, math.inf, None, 0
    A = MSE = MAE = IC = ECE = float("nan")

    for ep in range(1, epochs + 1):
        # train
        model.train()
        for batch in loaders["train"]:
            x = batch["x"].to(device, non_blocking=True)
            y_rc = batch["y_ret_cls"].to(device, non_blocking=True)
            out = model(x)
            loss = crit_cls(out["ret_cls_logits"], y_rc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # val
        model.eval()
        with torch.no_grad():
            vloss, probs, ycls, rhat, rtrue = 0.0, [], [], [], []
            for batch in loaders["val"]:
                x = batch["x"].to(device, non_blocking=True)
                y_rr = batch["y_ret_reg"].to(device, non_blocking=True)
                y_rc = batch["y_ret_cls"].to(device, non_blocking=True)
                out = model(x)
                loss = crit_cls(out["ret_cls_logits"], y_rc)
                vloss += float(loss.item())
                p = out["ret_cls_logits"].softmax(-1)[:, 1]
                probs.append(p.cpu())
                ycls.append(y_rc.cpu())
                rhat.append(out["ret_reg"].cpu())
                rtrue.append(y_rr.cpu())

            if probs:
                probs_t = torch.cat(probs)
                ycls_t = torch.cat(ycls)
                rhat_t = torch.cat(rhat)
                rtrue_t = torch.cat(rtrue)
                A = auc(ycls_t, probs_t)
                MSE, MAE = mse_mae(rtrue_t, rhat_t)
                IC = information_coefficient(rtrue_t, rhat_t)
                ECE = calibration_ece(ycls_t, probs_t)
            else:
                vloss = math.inf

        # early stop on AUC (maximise) with loss as tiebreak
        improved = (A > best_auc) or (A == best_auc and vloss < best_loss)
        if improved:
            best_auc = A
            best_loss = vloss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "val_loss": float(best_loss),
        "val_auc": float(best_auc),
        "val_mse": float(MSE),
        "val_mae": float(MAE),
        "val_ic": float(IC),
        "val_ece": float(ECE),
    }


# ---------------- helpers ----------------
def _parse_symbols(s: str) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _derive_windows(T, seq_len, horizon, train_len, val_len, step):
    tr = train_len or max(int(0.7 * T), 5000)
    va = val_len or max(int(0.15 * T), 1500)
    st = step or max(int(0.1 * T), 1000)
    cushion = seq_len + horizon + 32
    need = tr + va + cushion
    if T < need:
        raise RuntimeError(
            f"Not enough rows (T={T}) for windows: train={tr}, val={va}, "
            f"seq={seq_len}, horizon={horizon}. "
            f"Increase --lookback or reduce --train-len/--val-len."
        )
    return tr, va, st


# ---------------- entry point ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["tcn", "tx", "lstm", "adv"], default=os_mod.getenv("DL_MODEL_KIND", "tcn"))
    parser.add_argument("--seq-len", type=int, default=int(os_mod.getenv("DL_SEQ_LEN", "64")))
    parser.add_argument("--horizon", type=int, default=int(os_mod.getenv("DL_HORIZON_K", "12")))
    parser.add_argument("--batch", type=int, default=int(os_mod.getenv("DL_BATCH", "256")))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default=os_mod.getenv("DL_SAVE_DIR", "model_artifacts"))
    parser.add_argument("--tag", type=str, default="latest")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--symbols", type=str, default=os_mod.getenv("SYMBOL_WHITELIST", ""))
    parser.add_argument("--timeframe", type=str, default=os_mod.getenv("TIMEFRAME", "1m"))
    parser.add_argument("--lookback", type=int, default=int(os_mod.getenv("LOOKBACK_CANDLES", "12000")))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-len", type=int, default=None)
    parser.add_argument("--val-len", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    # label design
    parser.add_argument("--label", choices=["triple", "simple"], default="triple",
                        help="'triple' = triple-barrier (recommended), 'simple' = fixed-horizon binary")
    parser.add_argument("--pt", type=float, default=float(os_mod.getenv("DL_LABEL_PT", "0.005")),
                        help="Profit target fraction for triple barrier (default 0.5%%)")
    parser.add_argument("--sl", type=float, default=float(os_mod.getenv("DL_LABEL_SL", "0.005")),
                        help="Stop-loss fraction for triple barrier (default 0.5%%)")
    parser.add_argument("--max-hold", type=int, default=int(os_mod.getenv("DL_LABEL_MAX_HOLD", "60")),
                        help="Max hold bars for triple barrier (default 60)")
    parser.add_argument("--tau", type=float, default=float(os_mod.getenv("DL_LABEL_TAU", "0.003")),
                        help="Return threshold for simple binary label")
    # quality gate
    parser.add_argument("--min-auc", type=float, default=float(os_mod.getenv("DL_MIN_AUC", "0.52")),
                        help="Minimum val_auc to save 'latest' artifacts (default 0.52)")
    args = parser.parse_args()

    device = "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    symbols = _parse_symbols(args.symbols)

    X, prices, sym_lengths = load_prices_and_features(
        symbols=symbols,
        timeframe=args.timeframe,
        lookback=args.lookback,
        add_symbol_id=True,
        return_dfs=False,
        return_symbol_lengths=True,
    )
    T, F = X.shape
    print(f"[data] X shape={X.shape} prices={prices.shape} symbols={symbols or 'default'} "
          f"sym_lengths={sym_lengths}")

    # labels
    if args.label == "triple":
        print(f"[label] triple_barrier pt={args.pt} sl={args.sl} max_hold={args.max_hold}")
        y_cls = triple_barrier_label(prices, pt=args.pt, sl=args.sl, max_hold=args.max_hold)
        r = next_k_logret(prices, args.max_hold)
    else:
        print(f"[label] simple binarize tau={args.tau} horizon={args.horizon}")
        r = next_k_logret(prices, args.horizon)
        y_cls = binarize_return(r, tau=args.tau)

    rv = next_k_rv(np.log(prices), args.horizon)
    prices, r, rv, y_cls = _align_global_to_X(T, prices, r, rv, y_cls)

    # Null rows at each symbol boundary to prevent two forms of cross-symbol contamination:
    #   PRE:  last max_hold rows of symbol N  -> forward label window crosses into symbol N+1 prices
    #   POST: first seq_len-1 rows of symbol N+1 -> backward feature window includes symbol N rows
    if len(sym_lengths) > 1:
        boundary = 0
        total_nulled = 0
        for slen in sym_lengths[:-1]:
            boundary += slen
            pre_start = max(0, boundary - args.max_hold)
            y_cls[pre_start:boundary] = np.nan
            r[pre_start:boundary] = np.nan
            total_nulled += boundary - pre_start
            post_end = min(len(y_cls), boundary + args.seq_len - 1)
            y_cls[boundary:post_end] = np.nan
            r[boundary:post_end] = np.nan
            total_nulled += post_end - boundary
        print(f"[label] nulled {total_nulled} rows at {len(sym_lengths) - 1} symbol "
              f"boundaries (pre={args.max_hold} post={args.seq_len - 1} per boundary)")

    pos = int(np.nansum(y_cls == 1))
    n = int(np.sum(np.isfinite(y_cls)))
    print(f"[label balance] n={n}  positives={pos}  frac={pos / max(n, 1):.3f}")

    os_mod.makedirs(args.save_dir, exist_ok=True)

    train_len, val_len, step = _derive_windows(
        T=T, seq_len=args.seq_len, horizon=args.horizon,
        train_len=args.train_len, val_len=args.val_len, step=args.step,
    )
    print(f"[windows] train_len={train_len}  val_len={val_len}  step={step}")

    # track best fold across all windows
    best_auc_overall = -1.0
    best_scaler = None
    best_state_dict = None
    folds = 0

    for tr, va in rolling_windows(T, train_len=train_len, val_len=val_len, step=step):
        folds += 1
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32, copy=False)
        Xva = scaler.transform(X[va]).astype(np.float32, copy=False)

        r_tr, y_tr, rv_tr = r[tr], y_cls[tr], rv[tr]
        r_va, y_va, rv_va = r[va], y_cls[va], rv[va]
        Xtr, r_tr, y_tr, rv_tr = _align_fold_arrays(Xtr, r_tr, y_tr, rv_tr)
        Xva, r_va, y_va, rv_va = _align_fold_arrays(Xva, r_va, y_va, rv_va)

        ds_tr = SeqDataset(Xtr, r_tr, y_tr, rv_tr, args.seq_len)
        ds_va = SeqDataset(Xva, r_va, y_va, rv_va, args.seq_len)
        print(f"[fold {folds}] train={len(ds_tr)} val={len(ds_va)} "
              f"rows tr={_slice_len(tr)} va={_slice_len(va)}")

        if len(ds_tr) == 0 or len(ds_va) == 0:
            print(f"[fold {folds}] SKIP - empty dataset after NaN drop")
            continue

        dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device == "cuda"))

        # Inverse-frequency class weights from SeqDataset's actual kept windows,
        # not from the raw fold slice (which may include rows SeqDataset dropped
        # due to NaN features or cross-symbol boundary nulling).
        y_ds = ds_tr.y[ds_tr.idx]
        n0 = int(np.sum(y_ds == 0))
        n1 = int(np.sum(y_ds == 1))
        if n0 > 0 and n1 > 0:
            total = n0 + n1
            cw = torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32)
            print(f"[fold {folds}] label balance: SHORT={n0} ({n0/total:.1%}) LONG={n1} ({n1/total:.1%}) "
                  f"-> class weights [{cw[0]:.3f}, {cw[1]:.3f}]")
        else:
            cw = None
            print(f"[fold {folds}] label balance: SHORT={n0} LONG={n1} (degenerate fold, no weighting)")

        model = make_model(args.kind, in_dim=F).to(device)
        res = train_once(model, {"train": dl_tr, "val": dl_va}, device,
                         epochs=args.epochs, patience=args.patience, lr=args.lr,
                         class_weights=cw)
        print(f"[VAL fold {folds}] auc={res['val_auc']:.4f} loss={res['val_loss']:.4f} "
              f"ic={res['val_ic']:.4f} ece={res['val_ece']:.4f}")

        # save per-fold artifacts regardless of AUC
        sfx = f"{args.tag}_{va.start}_{va.stop}"
        joblib.dump(scaler, os_mod.path.join(args.save_dir, f"scaler_{sfx}.joblib"))
        torch.save(model.state_dict(), os_mod.path.join(args.save_dir, f"dl_{args.kind}_{sfx}.pt"))

        # AUC gate — update "latest" only from best qualifying fold
        if res["val_auc"] < args.min_auc:
            print(f"[fold {folds}] SKIP latest - val_auc={res['val_auc']:.4f} < min_auc={args.min_auc}")
            continue

        if res["val_auc"] > best_auc_overall:
            best_auc_overall = res["val_auc"]
            best_scaler = scaler
            best_state_dict = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            print(f"[fold {folds}] NEW BEST auc={best_auc_overall:.4f}")

    if folds == 0:
        raise RuntimeError("No folds produced. Increase --lookback or reduce window sizes.")

    if best_scaler is None:
        print(f"[WARN] No fold passed min_auc={args.min_auc}. Model NOT saved as 'latest'.")
        print("       Lower --min-auc or improve training data/labels and retry.")
        return

    # save best fold as "latest" — shared fallback AND per-kind so each model
    # keeps its own correctly-paired scaler when multiple kinds are trained.
    joblib.dump(best_scaler, os_mod.path.join(args.save_dir, "scaler_latest.joblib"))
    joblib.dump(best_scaler, os_mod.path.join(args.save_dir, f"scaler_{args.kind}_latest.joblib"))
    best_model = make_model(args.kind, in_dim=F).to("cpu")
    best_model.load_state_dict(best_state_dict)
    torch.save(best_model.state_dict(), os_mod.path.join(args.save_dir, f"dl_{args.kind}_latest.pt"))

    # save metadata alongside the model
    # Persist the EXACT symbol->id mapping training used (id = position in the
    # symbol list). Serving reads this so it assigns the same id a model learned,
    # instead of a position-in-the-live-list id that silently differs. Derived
    # from `symbols`; both are saved for clarity and forward-compat.
    symbol_id_map = {s: i for i, s in enumerate(symbols)} if symbols else None

    meta = {
        "kind": args.kind,
        "seq_len": args.seq_len,
        "n_features": int(F),
        "symbols": symbols,
        "symbol_id_map": symbol_id_map,
        "timeframe": args.timeframe,
        "label": {
            "type": args.label,
            "pt": args.pt if args.label == "triple" else None,
            "sl": args.sl if args.label == "triple" else None,
            "max_hold": args.max_hold if args.label == "triple" else None,
            "tau": args.tau if args.label == "simple" else None,
            "horizon": args.horizon,
        },
        "val_auc": float(best_auc_overall),
        "min_auc_gate": args.min_auc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os_mod.path.join(args.save_dir, f"dl_{args.kind}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Saved best fold (auc={best_auc_overall:.4f}) as dl_{args.kind}_latest.pt")
    print(f"       Metadata: {meta_path}")


if __name__ == "__main__":
    main()
