#!/usr/bin/env python
import argparse, os, glob, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BUCKET_LABELS_DEFAULT = ["p50","p60","p70","p80","p90","p95"]

@dataclass
class BucketStats:
    n: int
    hit: float
    avg: float
    sharpe_like: float

def parse_args():
    ap = argparse.ArgumentParser(
        description="OOF + Live side-by-side metrics with threshold suggestion"
    )
    ap.add_argument("--oof-dir", required=True, help="Folder with oof_*.npz")
    ap.add_argument("--live-log", required=True, help="live_meta_log.csv from live_meta_ensemble.py")
    ap.add_argument("--rv-max", type=float, default=60.0, help="Cap realized vol for OOF filtering AND live rv_mean filtering")
    ap.add_argument("--out-dir", default="metrics_out_plus", help="Where to save CSVs/PNGs")
    ap.add_argument("--kinds", default="", help="Comma list of kinds to include (e.g., tx or lstm,tx). Empty = all.")
    ap.add_argument("--metric", choices=["avg","sharpe","hit"], default="sharpe",
                    help="Metric used for threshold suggestion: avg>0, sharpe>0, or hit>0.5")
    ap.add_argument("--min-live", type=int, default=10, help="Minimum live rows in a bucket to consider it viable")
    ap.add_argument("--buckets", default=",".join(BUCKET_LABELS_DEFAULT),
                    help="Bucket labels, low->high (default p50,p60,p70,p80,p90,p95)")
    ap.add_argument("--allow-only", action="store_true",
                    help="Only count live rows where allow==1")
    ap.add_argument("--emit-ps1", action="store_true",
                    help="Save a PowerShell script that exports the suggested threshold")
    return ap.parse_args()

def _kind_from_filename(path: str) -> str:
    base = os.path.basename(path)
    parts = base.split("_", 2)
    return parts[1] if len(parts) >= 2 else "unknown"

def load_oof(oof_dir: str, kinds_filter: Optional[List[str]], rv_max: float):
    """Load OOF arrays for p_long, ret, rv; returns dict: {kind: (P,RV,RET)} plus aggregate 'all'."""
    paths = sorted(glob.glob(os.path.join(oof_dir, "oof_*_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No OOF files in {oof_dir}")

    data_by_kind: Dict[str, List[Tuple[np.ndarray,np.ndarray,np.ndarray]]] = {}
    for p in paths:
        kind = _kind_from_filename(p)
        if kinds_filter and kind not in kinds_filter:
            continue
        z = np.load(p, allow_pickle=True)
        P   = z["p_long"].astype("float32")
        RV  = z["rv"].astype("float32")
        RET = z["ret"].astype("float32")
        m = np.isfinite(P) & np.isfinite(RV) & np.isfinite(RET) & (RV <= rv_max)
        if not m.any():
            continue
        data_by_kind.setdefault(kind, []).append((P[m], RV[m], RET[m]))

    if not data_by_kind:
        raise RuntimeError("No OOF files matched the filter and rv_max constraint.")

    stacked: Dict[str, Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}
    for k, lst in data_by_kind.items():
        P  = np.concatenate([a for a,_,_ in lst]) if lst else np.array([],dtype="float32")
        RV = np.concatenate([b for _,b,_ in lst]) if lst else np.array([],dtype="float32")
        R  = np.concatenate([c for _,_,c in lst]) if lst else np.array([],dtype="float32")
        stacked[k] = (P,RV,R)

    P_agg = np.concatenate([v[0] for v in stacked.values()])
    RV_agg= np.concatenate([v[1] for v in stacked.values()])
    R_agg = np.concatenate([v[2] for v in stacked.values()])
    stacked["all"] = (P_agg, RV_agg, R_agg)
    return stacked

def bucket_quantiles(P: np.ndarray, labels: List[str]) -> Dict[str, float]:
    qmap = {}
    for lab in labels:
        if not lab.startswith("p") or not lab[1:].isdigit():
            raise ValueError(f"Bad bucket label: {lab}")
        q = float(int(lab[1:])) / 100.0
        qmap[lab] = float(np.quantile(P, q))
    return qmap

def stats_for_slice(pay: np.ndarray) -> BucketStats:
    if pay.size == 0:
        return BucketStats(0, float("nan"), float("nan"), float("nan"))
    hit = float((pay > 0).mean())
    avg = float(pay.mean())
    std = float(pay.std()) + 1e-12
    shp = float(avg / std)
    return BucketStats(int(pay.size), hit, avg, shp)

def compute_oof_buckets(P: np.ndarray, R: np.ndarray, qmap: Dict[str,float]) -> Dict[str, BucketStats]:
    out: Dict[str, BucketStats] = {}
    for lab, thr in qmap.items():
        m = P >= thr
        out[lab] = stats_for_slice(R[m])
    return out

def load_live_counts(live_csv: str, qmap: Dict[str,float], rv_max: float, allow_only: bool) -> Dict[str,int]:
    if not os.path.isfile(live_csv):
        return {lab: 0 for lab in qmap.keys()} | {"below_p50": 0}

    counts = {lab: 0 for lab in qmap.keys()}
    below = 0
    with open(live_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                p = float(row["p_meta"])
                rv = float(row["rv_mean"])
                allow_raw = str(row.get("allow","0")).strip()
                allow = allow_raw in ("1","True","true","YES","yes")
            except Exception:
                continue
            if not np.isfinite(p) or not np.isfinite(rv):
                continue
            if rv > rv_max:
                continue
            if allow_only and not allow:
                continue
            placed = False
            for lab, thr in sorted(qmap.items(), key=lambda kv: kv[1], reverse=True):
                if p >= thr:
                    counts[lab] += 1
                    placed = True
                    break
            if not placed:
                below += 1
    counts["below_p50"] = below
    return counts

def suggest_threshold(metric: str, min_live: int,
                      oof_stats: Dict[str, BucketStats],
                      live_counts: Dict[str, int],
                      qmap: Dict[str, float]) -> Tuple[str, float, str]:
    def metric_val(bs: BucketStats) -> float:
        if metric == "avg":    return bs.avg
        if metric == "sharpe": return bs.sharpe_like
        if metric == "hit":    return bs.hit
        return float("nan")

    def metric_ok(bs: BucketStats) -> bool:
        if metric == "avg":    return (bs.avg > 0)
        if metric == "sharpe": return (bs.sharpe_like > 0)
        if metric == "hit":    return (bs.hit > 0.5)
        return False

    # Highest-first scan
    for lab in sorted(qmap.keys(), key=lambda k: qmap[k], reverse=True):
        bs = oof_stats[lab]
        if live_counts.get(lab, 0) >= min_live and metric_ok(bs):
            return lab, qmap[lab], f"Meets {metric} criterion with enough live coverage (n_live={live_counts.get(lab,0)})"

    # Fallback: best metric among buckets that have at least one live row
    viable = [(lab, metric_val(oof_stats[lab]), live_counts.get(lab,0))
              for lab in qmap.keys() if live_counts.get(lab,0) > 0]
    if viable:
        best = sorted(viable, key=lambda x: x[1], reverse=True)[0]
        return best[0], qmap[best[0]], f"No bucket passed the strict rule; chose best {metric} among buckets with live>0 (n_live={best[2]})"

    # Nothing? pick p50
    lab = "p50"
    return lab, qmap[lab], "No live coverage; defaulting to p50"

def save_csv(path: str, header: List[str], rows: List[List]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    kinds_filter = [k.strip() for k in args.kinds.split(",") if k.strip()] if args.kinds else None
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]

    # --- Load OOF ---
    stacked = load_oof(args.oof_dir, kinds_filter, args.rv_max)

    # Quantiles pool: if user filtered kinds, compute quantiles on that subset; else use global
    if kinds_filter:
        P_pool = np.concatenate([stacked[k][0] for k in stacked if k != "all"])
    else:
        P_pool = stacked["all"][0]
    qmap = bucket_quantiles(P_pool, buckets)

    # OOF aggregate + per-kind
    oof_stats_agg = compute_oof_buckets(stacked["all"][0], stacked["all"][2], qmap)
    oof_agg_rows = [[lab, oof_stats_agg[lab].n, oof_stats_agg[lab].hit,
                     oof_stats_agg[lab].avg, oof_stats_agg[lab].sharpe_like]
                    for lab in buckets]
    per_kind_rows = []
    for k, (P, _, R) in stacked.items():
        if k == "all": continue
        stats_k = compute_oof_buckets(P, R, qmap)
        for lab in buckets:
            bs = stats_k[lab]
            per_kind_rows.append([k, lab, bs.n, bs.hit, bs.avg, bs.sharpe_like])

    save_csv(out_dir / "oof_buckets_aggregate_plus.csv",
             ["bucket","n","hit","avg","sharpe_like"], oof_agg_rows)
    save_csv(out_dir / "oof_buckets_by_kind_plus.csv",
             ["kind","bucket","n","hit","avg","sharpe_like"], per_kind_rows)

    # --- Live coverage (rv + allow filter applied) ---
    live_counts = load_live_counts(args.live_log, qmap, args.rv_max, args.allow_only)
    live_rows = [[lab, live_counts.get(lab,0)] for lab in buckets] + [["below_p50", live_counts.get("below_p50",0)]]
    save_csv(out_dir / "live_bucket_counts_plus.csv", ["bucket","live_count"], live_rows)

    # --- Suggest threshold ---
    sugg_bucket, sugg_thr, reason = suggest_threshold(
        metric=args.metric, min_live=args.min_live,
        oof_stats=oof_stats_agg, live_counts=live_counts, qmap=qmap
    )

    # Console summary
    print(f"\n== OOF (aggregate) with RV<={args.rv_max} ==")
    for lab, n, hit, avg, shp in oof_agg_rows:
        print(f" {lab:>3}  n={n:5d}  hit={hit:.3f}  avg={avg:+.6f}  sharpe_like={shp:+.3f}")
    print("\n== Live coverage ==")
    for lab in buckets:
        print(f" {lab:>3}  live_n={live_counts.get(lab,0)}")
    print(f" below_p50 live_n={live_counts.get('below_p50',0)}")
    print(f"\nSuggested threshold (abs): {sugg_thr:.6f}  [{sugg_bucket}]")
    print(f"Reason: {reason}")
    print("Tip: set DL_P_LONG_MODE=abs and DL_P_LONG to that value. If using percentile mode, map this to the same bucket.\n")

    # Plots
    # live counts
    labs_for_bar = buckets + ["below_p50"]
    vals = [live_counts.get(l,0) for l in labs_for_bar]
    plt.figure(figsize=(8,6)); plt.bar(labs_for_bar, vals); plt.title("Live p_meta coverage")
    plt.tight_layout(); plt.savefig(out_dir / "live_counts_plus.png"); plt.close()

    # OOF metric bars
    def plot_bar(metric_name: str, filename: str):
        vals=[]
        for lab in buckets:
            bs = oof_stats_agg[lab]
            v = bs.avg if metric_name=="avg" else (bs.hit if metric_name=="hit" else bs.sharpe_like)
            vals.append(v)
        plt.figure(figsize=(8,6)); plt.bar(buckets, vals); plt.title(f"OOF {metric_name} by p-bucket")
        plt.tight_layout(); plt.savefig(out_dir / filename); plt.close()
    plot_bar("avg", "oof_avg_plus.png")
    plot_bar("hit", "oof_hit_plus.png")
    plot_bar("sharpe_like", "oof_sharpe_plus.png")

    # Side-by-side OOF vs normalized live
    choose = args.metric
    oof_vals = []
    for lab in buckets:
        bs = oof_stats_agg[lab]
        oof_vals.append(bs.avg if choose=="avg" else (bs.hit if choose=="hit" else bs.sharpe_like))
    x = np.arange(len(buckets)); w = 0.4
    max_cnt = max(1, max([live_counts.get(l,0) for l in buckets]))
    live_norm = [live_counts.get(l,0)/max_cnt for l in buckets]
    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, oof_vals, width=w, label=f"OOF {choose}")
    plt.bar(x + w/2, live_norm, width=w, label="Live coverage (normalized)")
    plt.xticks(x, buckets); plt.title(f"OOF {choose} vs Live coverage (norm)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / "oof_vs_live_plus.png"); plt.close()

    # Emit a one-click PowerShell script if requested
    if args.emit_ps1:
        kinds_env = args.kinds.replace(" ", "") if args.kinds else "lstm,tcn,tx"
        ps1 = out_dir / "apply_suggested_threshold.ps1"
        ps1.write_text(
f"""# Auto-generated by oo_metrics_plus.py
$env:DL_P_LONG_MODE = "abs"
$env:DL_P_LONG      = "{sugg_thr:.6f}"
$env:META_KINDS     = "{kinds_env}"
$env:DL_LOG_PATH    = "live_meta_log.csv"
Write-Host "Set DL_P_LONG={sugg_thr:.6f} (bucket {sugg_bucket})."
""",
            encoding="utf-8"
        )
        print(f"[emit] wrote {ps1}")

    # -------------------------------------------------------------
    # Additional outputs: coverage breakdown and thresholds JSON
    # -------------------------------------------------------------
    # Build a summary table combining OOF and live statistics.  This file
    # makes it easy to audit why certain buckets were selected.  Each row
    # contains the bucket label, OOF sample count (n), live sample count,
    # OOF hit-rate, average return and Sharpe-like ratio.  The final
    # ``below_p50`` row reports live samples that fall below the lowest
    # computed threshold; the OOF columns for that row are left as NaN.
    coverage_rows: List[List] = []
    for lab in buckets:
        bs = oof_stats_agg[lab]
        live_n = live_counts.get(lab, 0)
        coverage_rows.append([
            lab,
            bs.n,
            live_n,
            bs.hit,
            bs.avg,
            bs.sharpe_like,
        ])
    # Add below_p50 row
    coverage_rows.append([
        "below_p50",
        0,
        live_counts.get("below_p50", 0),
        float("nan"),
        float("nan"),
        float("nan"),
    ])
    save_csv(
        out_dir / "coverage_breakdown.csv",
        [
            "bucket",
            "oof_n",
            "live_n",
            "oof_hit",
            "oof_avg",
            "oof_sharpe",
        ],
        coverage_rows,
    )
    # Persist the suggested threshold and justification as JSON
    import json
    thr_info = {
        "suggested_bucket": sugg_bucket,
        "threshold_value": sugg_thr,
        "reason": reason,
    }
    with open(out_dir / "thresholds.json", "w", encoding="utf-8") as jf:
        json.dump(thr_info, jf, indent=2)

if __name__ == "__main__":
    main()
