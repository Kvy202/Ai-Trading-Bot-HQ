# tools/merge_daily_logs.py
import argparse, glob, os, sys
from datetime import datetime
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Merge daily live logs into one master CSV.")
    ap.add_argument("--daily-dir", default="logs",
                    help="Directory containing daily CSVs (default: logs)")
    ap.add_argument("--pattern", default="live_meta_log_*.csv",
                    help='Glob for daily files (default: "live_meta_log_*.csv")')
    ap.add_argument("--out", default="live_meta_master.csv",
                    help="Output CSV path (default: live_meta_master.csv)")
    ap.add_argument("--include-master", action="store_true",
                    help="Also include the current master live_meta_log.csv if present.")
    ap.add_argument("--dedupe", action="store_true",
                    help="Drop duplicate rows by (ts,p_meta,rv_mean,allow).")
    ap.add_argument("--quiet", action="store_true", help="Less printing")
    return ap.parse_args()

def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Make columns consistent across days."""
    if "allow" in df.columns:
        df["allow"] = df["allow"].astype(str).str.strip().str.lower().isin(["1","true","yes"]).astype(int)
    # Ensure ts is parseable; keep original string column for output order
    if "ts" in df.columns:
        df["_ts_dt"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    else:
        df["_ts_dt"] = pd.NaT
    return df

def main():
    args = parse_args()

    # Gather daily files
    daily_glob = os.path.join(args.daily_dir, args.pattern)
    files = sorted(glob.glob(daily_glob))
    if args.include_master and os.path.isfile("live_meta_log.csv"):
        files.append("live_meta_log.csv")

    if not files:
        print(f"[merge] no files found at {daily_glob}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for f in files:
        df = read_csv_safe(f)
        if df.empty:
            continue
        df = normalize(df)
        frames.append(df)

    if not frames:
        print("[merge] nothing to merge (all empty).", file=sys.stderr)
        sys.exit(1)

    # Union of all columns
    all_cols = []
    seen = set()
    for df in frames:
        for c in df.columns:
            if c not in seen:
                seen.add(c); all_cols.append(c)

    # Concat with unioned columns (missing -> NaN)
    merged = pd.concat(frames, ignore_index=True)[all_cols]

    # Sort by parsed datetime if available
    if "_ts_dt" in merged.columns:
        merged = merged.sort_values("_ts_dt", kind="stable")
        # Drop helper column
        merged = merged.drop(columns=["_ts_dt"], errors="ignore")

    # Optional de-dupe by a stable subset (ts + a few fields)
    if args.dedupe:
        keys = [c for c in ["ts","p_meta","rv_mean","allow"] if c in merged.columns]
        if keys:
            merged = merged.drop_duplicates(subset=keys, keep="last")

    # Write out
    out = args.out
    merged.to_csv(out, index=False, encoding="utf-8")
    if not args.quiet:
        print(f"[merge] wrote {len(merged):,} rows -> {out}")
        print(f"[merge] sources ({len(files)}):")
        for f in files:
            print("  -", f)

if __name__ == "__main__":
    main()
