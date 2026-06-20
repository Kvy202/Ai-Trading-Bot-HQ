import os, csv, glob, numpy as np, argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("LIVE_LOG","logs/live_meta_log.csv"))
    ap.add_argument("--p-min", type=float, default=float(os.getenv("P_LONG_MIN","0.43")))
    ap.add_argument("--rv-max", type=float, default=float(os.getenv("RV_MAX","60")))
    ap.add_argument("--oof-dir", default="oof")
    args = ap.parse_args()

    oof_P, oof_RET, oof_RV = [], [], []
    for p in sorted(glob.glob(os.path.join(args.oof_dir,"oof_*_*.npz"))):
        z = np.load(p, allow_pickle=True)
        P = z["p_long"].astype("float32")
        RV= z["rv"].astype("float32")
        RET=z["ret"].astype("float32")
        m = np.isfinite(P) & np.isfinite(RV) & np.isfinite(RET)
        oof_P.append(P[m]); oof_RV.append(RV[m]); oof_RET.append(RET[m])
    if not oof_P:
        print("[WARN] no oof files"); sys.exit(0)

    oP  = np.concatenate(oof_P)
    oRV = np.concatenate(oof_RV)
    oRET= np.concatenate(oof_RET)
    mask = (oRV <= args.rv_max)
    oP, oRET = oP[mask], oRET[mask]

    qs = np.quantile(oP, [0.5, 0.7, 0.8, 0.9, 0.95])
    buckets = [(qs[0], "p50"), (qs[1], "p70"), (qs[2], "p80"), (qs[3], "p90"), (qs[4], "p95")]

    def oof_expectation_at(p):
        for q, name in buckets:
            if p >= q:
                sel = (oP >= q)
                if sel.any():
                    pay = oRET[sel]
                    return name, float((pay>0).mean()), float(pay.mean())
        sel = (oP >= qs[0])
        return ("below_p50",
                float((oRET[sel]>0).mean()) if sel.any() else float("nan"),
                float(oRET[sel].mean())     if sel.any() else float("nan"))

    trades=0; wins=0; pays=[]
    try:
        with open(args.log, newline="", encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                p_meta=float(row["p_meta"])
                rv=float(row["rv_mean"])
                allow = (str(row["allow"]).strip() in ("1","True","true","YES","yes"))
                if allow and rv<=args.rv_max and p_meta>=args.p_min:
                    trades+=1
                    _, hit_est, avg_est = oof_expectation_at(p_meta)
                    wins += (hit_est>0.5)
                    if np.isfinite(avg_est):
                        pays.append(avg_est)
    except FileNotFoundError:
        print(f"[live] no log found at {args.log}")
        sys.exit(0)

    if trades==0:
        print("[live-proxy] no allowed rows in log yet")
    else:
        wr = wins/max(1,trades)
        avg = (float(np.mean(pays)) if len(pays) else float("nan"))
        shp = (avg/float(np.std(pays)+1e-12)) if len(pays)>1 else float("nan")
        print(f"[live-proxy] trades={trades} approx_winrate={wr:.3f} exp_avg_pay={avg:.6f} sharpe_likeâ‰ˆ{shp:.3f}")

if __name__ == "__main__":
    import numpy as np
    main()
