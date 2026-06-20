# ml_dl/meta_train.py
import os, glob, joblib
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import argparse


"""
Train a stacking head that combines base models' [p_long, rv, ret] in a fixed order.

Usage:
  python -m ml_dl.meta_train --tags "h24_ret20cls2,tx_h24_cls2,lstm_h24_cls2" \
    --out model_artifacts/meta_logreg_h24_cls2.joblib

Env:
  DL_OOF_DIR=oof            # where oof_*.npz files live

Artifact contents:
  {
    "scaler": StandardScaler,
    "clf": LogisticRegression,
    "tags": [..],                 # your input tags
    "kinds_order": ["lstm","tcn","tx"],  # alphabetical by kind (stable)
    "schema": [("lstm","h24_ret20cls2"),("tcn","h24_ret20cls2"),("tx","tx_h24_cls2")],
    "features_per_kind": 3,       # p_long, rv, ret (in that order)
    "version": "v1"
  }

Live should build features in exactly this order:
  for kind in kinds_order:
     append [p_long(kind), rv(kind), ret(kind)]
"""


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tags",
        type=str,
        required=True,
        help="comma list of run tags to include, e.g. h24_ret20cls2,tx_h24_cls2,lstm_h24_cls2",
    )
    ap.add_argument("--out", type=str, default="model_artifacts/meta_logreg.joblib")
    ap.add_argument("--oof-dir", type=str, default=os.getenv("DL_OOF_DIR", "oof"))
    return ap.parse_args()


def _read_oof(path: str) -> Dict:
    """Return dict with unified keys: p_long, y, rv, ret, meta(kind/start/stop)."""
    z = np.load(path, allow_pickle=True)
    files = set(z.files)

    # required
    if "p_long" not in files or "y" not in files:
        raise KeyError(f"{path}: missing p_long and/or y")

    # tolerate old/new naming
    if "rv" in files:
        rv = z["rv"]
    elif "rv_hat" in files:
        rv = z["rv_hat"]
    else:
        raise KeyError(f"{path}: missing rv / rv_hat")

    if "ret" in files:
        ret = z["ret"]
    elif "ret_hat" in files:
        ret = z["ret_hat"]
    else:
        raise KeyError(f"{path}: missing ret / ret_hat")

    meta = {}
    if "meta" in files:
        try:
            meta = z["meta"].item()
        except Exception:
            meta = {}

    # fallback kind/start/stop from filename when missing
    base = os.path.basename(path)  # oof_{kind}_{tag}_{start}_{stop}.npz
    parts = base[:-4].split("_")   # drop .npz
    kind_guess = parts[1] if len(parts) >= 2 else meta.get("kind", "tcn")
    try:
        start_guess, stop_guess = int(parts[-2]), int(parts[-1])
    except Exception:
        start_guess = meta.get("start")
        stop_guess  = meta.get("stop")

    meta.setdefault("kind", kind_guess)
    if meta.get("start") is None: meta["start"] = start_guess
    if meta.get("stop")  is None: meta["stop"]  = stop_guess

    return dict(
        p_long=z["p_long"],
        y=z["y"],
        rv=rv,
        ret=ret,
        meta=meta,
    )


def _collect_by_fold(oof_dir: str, tags: List[str]) -> Dict[Tuple[int, int], List[Dict]]:
    """Return { (start,stop): [ {kind, tag, p_long, rv, ret, y}, ... ] }."""
    by_fold: Dict[Tuple[int, int], List[Dict]] = {}
    for tag in tags:
        for path in glob.glob(os.path.join(oof_dir, f"oof_*_{tag}_*.npz")):
            d = _read_oof(path)
            meta = d["meta"]
            kind = meta.get("kind", "tcn")
            start = meta.get("start")
            stop = meta.get("stop")
            if start is None or stop is None:
                raise RuntimeError(f"Cannot infer (start,stop) for {path}")
            key = (int(start), int(stop))
            by_fold.setdefault(key, []).append(
                dict(
                    tag=tag,
                    kind=kind,
                    p_long=d["p_long"],
                    rv=d["rv"],
                    ret=d["ret"],
                    y=d["y"],
                )
            )
    return by_fold


def main():
    args = parse()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    by_fold = _collect_by_fold(args.oof_dir, tags)

    # Build training matrix across aligned folds
    X_blocks, y_blocks = [], []
    skipped = 0
    fold_schemas = []  # store the (kind, tag) order used per included fold

    for (start, stop), items in sorted(by_fold.items()):
        # Stable order: alphabetical by kind, tie-break by tag string
        items_sorted = sorted(items, key=lambda d: (d["kind"], d["tag"]))

        # lengths must match across models within the fold
        Ls = [len(it["y"]) for it in items_sorted]
        if len(set(Ls)) != 1:
            skipped += 1
            continue
        L = Ls[0]

        # stack features in [p, rv, ret] per model, in the sorted order
        feats = []
        for it in items_sorted:
            feats.append(it["p_long"].reshape(L, 1))
            feats.append(it["rv"].reshape(L, 1))
            feats.append(it["ret"].reshape(L, 1))
        Xf = np.concatenate(feats, axis=1)
        yf = items_sorted[0]["y"].reshape(L)

        X_blocks.append(Xf)
        y_blocks.append(yf)
        fold_schemas.append([(it["kind"], it["tag"]) for it in items_sorted])

    if not X_blocks:
        raise RuntimeError(
            "No aligned OOF folds found. "
            "Ensure each tag was trained with identical windows (train/val/step) and produced OOF for the same folds."
        )

    # verify that schema (order of kinds/tags) is constant across included folds
    schema0 = fold_schemas[0]
    if any(s != schema0 for s in fold_schemas[1:]):
        # If this happens, it usually means you mixed runs with different model sets.
        raise RuntimeError(
            "Inconsistent (kind, tag) ordering across folds. "
            "Make sure all tags exist for all folds and were produced with the same settings."
        )

    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)

    # class balance info
    pos = int(np.sum(y == 1))
    n = int(len(y))
    print(f"[meta] folds_used={len(X_blocks)} folds_skipped={skipped}  n={n}  pos={pos}  frac_pos={pos/max(n,1):.3f}")
    print(f"[meta] feature_dim={X.shape[1]}  models={len(schema0)}  features_per_kind=3")

    # fit scaler + LR
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(Xs, y)

    # derive stable kinds_order (unique kinds in schema order)
    kinds_order = [k for k, _ in schema0]
    # reduce duplicates while preserving order
    seen = set()
    kinds_order = [k for k in kinds_order if not (k in seen or seen.add(k))]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "scaler": scaler,
        "clf": clf,
        "tags": tags,
        "kinds_order": kinds_order,        # e.g. ["lstm","tcn","tx"]
        "schema": schema0,                 # e.g. [("lstm","lstm_h24_cls2"), ("tcn","h24_ret20cls2"), ("tx","tx_h24_cls2")]
        "features_per_kind": 3,            # p_long, rv, ret
        "version": "v1",
    }
    joblib.dump(payload, args.out)
    print(f"[meta] trained on {X.shape[0]} rows, {X.shape[1]} features -> saved {args.out}")
    print(f"[meta] kinds_order={kinds_order} schema={schema0}")


if __name__ == "__main__":
    main()
