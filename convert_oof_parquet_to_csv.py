import os, glob, pandas as pd, pathlib
out = pathlib.Path("oof_csv"); out.mkdir(exist_ok=True)
for p in glob.glob("oof/**/*.parquet", recursive=True):
    try:
        df = pd.read_parquet(p)
        base = pathlib.Path(p).stem + ".csv"
        df.to_csv(out / base, index=False)
        print('OK ->', base)
    except Exception as e:
        print('FAIL', p, e)
