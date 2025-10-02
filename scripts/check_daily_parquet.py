from pathlib import Path
import pandas as pd

ROOT = Path("/Users/Finance/QuantETFUS_small")
m30_dir = ROOT / "data_enriched/30min"
daily_path = ROOT / "data_enriched/prices_enriched.parquet"

# tickers present in 30m folder
m30_tickers = sorted([p.stem.upper() for p in m30_dir.glob("*.parquet")])

# open daily parquet, list tickers & columns
D = pd.read_parquet(daily_path)
D.columns = [c.strip().lower() for c in D.columns]
daily_tickers = sorted(D["ticker"].astype(str).str.strip().str.upper().unique())

print("Daily _d columns sample:", [c for c in D.columns if c.endswith("_d")][:12])
print("Counts → 30m:", len(m30_tickers), "daily:", len(daily_tickers))

missing = [t for t in m30_tickers if t not in daily_tickers]
present = [t for t in m30_tickers if t in daily_tickers]
print("\nMissing in daily parquet (no merge will happen):", missing[:30], "… total:", len(missing))

# sanity: show last few daily rows for one “missing” and one “present” (if any)
if present:
    t = present[0]
    print(f"\nSample PRESENT {t}:")
    print(D[D["ticker"].astype(str).str.upper().eq(t)].tail(3)[["datetime","ticker","ema20_d","ema44_d","rsi14_d"]])
if missing:
    t = missing[0]
    print(f"\nSample MISSING {t}:")
    print(D[D["ticker"].astype(str).str.upper().eq(t)].tail(3))