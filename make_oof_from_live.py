import pandas as pd
import pathlib, re

live = pd.read_csv("live_meta_log.csv", header=None)
# Guess column layout from your sample
live.columns = ["ts","p_long","p50","mode","rv","zero",
                "model_type","meta","rv2","something","something2","something3",
                "something4","something5","something6","something7","sizes","vols"]

# Build a minimal OOF-like frame: ts, pred, side, model_type
oof = pd.DataFrame({
    "ts": live["ts"],
    "pred": live["p50"],         # using p50 as a stand-in for prediction
    "side": "long",              # adjust if you have a side column
    "model_type": live["model_type"]
})
out = pathlib.Path("oof_csv"); out.mkdir(exist_ok=True)
oof.to_csv(out / "live_as_oof.csv", index=False)
print("Wrote", out / "live_as_oof.csv", "rows:", len(oof))
