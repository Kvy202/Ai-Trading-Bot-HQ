import os, glob, numpy as np, argparse
import matplotlib.pyplot as plt

def sharpe_like(x: np.ndarray) -> float:
    x = x.astype("float64")
    return float(x.mean() / (x.std() + 1e-12)) if x.size else float("nan")

def bucketize(P, RET, qs):
    """Return dict: name -> (hit, avg, shp, n) for thresholds >= q."""
    out = {}
    for q, name in qs:
        m = (P >= q)
        if m.any():
            pay = RET[m]
            out[name] = (float((pay>0).mean()), float(pay.mean()), sharpe_like(pay), int(m.sum()))
        else:
            out[name] = (np.nan, np.nan, np.nan, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-dir", default="oof")
    ap.add_argument("--rv-max", type=float, default=60.0)
    ap.add_argument("--out-dir", default="metrics_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    per_kind = {}
    allP, allRET, allRV = [], [], []

    for p in sorted(glob.glob(os.path.join(args.oof_dir, "oof_*_*.npz"))):
        z = np.load(p, allow_pickle=True)
        base = os.path.basename(p)
        kind = base.split("_", 2)[1]  # oof_{kind}_...
        P   = z["p_long"].astype("float32")
        RV  = z["rv"].astype("float32")
        RET = z["ret"].astype("float32")
        m = np.isfinite(P) & np.isfinite(RV) & np.isfinite(RET) & (RV <= args.rv_max)
        if not m.any(): continue
        per_kind.setdefault(kind, {"P":[], "RET":[], "RV":[]})
        per_kind[kind]["P"].append(P[m])
        per_kind[kind]["RET"].append(RET[m])
        per_kind[kind]["RV"].append(RV[m])
        allP.append(P[m]); allRET.append(RET[m]); allRV.append(RV[m])

    if not allP:
        print("[oo_metrics] no usable OOF points."); return

    AP = np.concatenate(allP); ARET = np.concatenate(allRET)
    qs_raw = np.quantile(AP, [0.50, 0.70, 0.80, 0.90, 0.95])
    qs = [(qs_raw[0], "p50"), (qs_raw[1], "p70"), (qs_raw[2], "p80"), (qs_raw[3], "p90"), (qs_raw[4], "p95")]

    # Print table
    print("== Aggregate (RV<=%.1f) ==" % args.rv_max)
    agg = bucketize(AP, ARET, qs)
    for name,(hit,avg,shp,n) in agg.items():
        print(f"{name:>4s}  n={n:5d}  hit={hit:.3f}  avg={avg:.6f}  sharpe_like={shp:.3f}")

    for k,v in sorted(per_kind.items()):
        P  = np.concatenate(v["P"]); RET = np.concatenate(v["RET"])
        stat = bucketize(P, RET, qs)
        print(f"\n== {k} ==")
        for name,(hit,avg,shp,n) in stat.items():
            print(f"{name:>4s}  n={n:5d}  hit={hit:.3f}  avg={avg:.6f}  sharpe_like={shp:.3f}")

    # Charts on aggregate
    labels = [n for _,n in qs]
    xs = np.arange(len(labels))
    hits = [agg[n][0] for _,n in qs]
    avgs = [agg[n][1] for _,n in qs]
    shps = [agg[n][2] for _,n in qs]

    # hit
    plt.figure()
    plt.plot(xs, hits, marker="o")
    plt.xticks(xs, labels)
    plt.title("OOF: Hit-rate vs p-bucket")
    plt.xlabel("bucket threshold")
    plt.ylabel("hit-rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"oof_hit_vs_bucket.png"))
    plt.close()

    # avg
    plt.figure()
    plt.plot(xs, avgs, marker="o")
    plt.xticks(xs, labels)
    plt.title("OOF: Avg payoff vs p-bucket")
    plt.xlabel("bucket threshold")
    plt.ylabel("avg payoff")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"oof_avg_vs_bucket.png"))
    plt.close()

    # sharpe-like
    plt.figure()
    plt.plot(xs, shps, marker="o")
    plt.xticks(xs, labels)
    plt.title("OOF: Sharpe-like vs p-bucket")
    plt.xlabel("bucket threshold")
    plt.ylabel("sharpe-like")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir,"oof_sharpe_vs_bucket.png"))
    plt.close()

    print(f"\n[oo_metrics] saved charts to {args.out_dir}\\*.png")

if __name__ == "__main__":
    main()
