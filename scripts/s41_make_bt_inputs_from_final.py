#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s41_make_bt_inputs_from_final.py
Create per-ticker DAILY *_bt.csv for s50 (daily engine) using the enriched parquet.

Reads
-----
- P.DATA_ENRICHED/prices_enriched.parquet   (from s30)  [override with --parquet]

Writes
------
- P.ROOT/backtest_data/daily/{TICKER}_bt.csv [override with --outdir]
- Columns: datetime, ticker, open, high, low, close, ema5, ema20, ema44, entry_signal
"""

from pathlib import Path
import sys
import argparse
import pandas as pd

# --- import centralized paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402

DEFAULT_PARQUET = P.DATA_ENRICHED / "prices_enriched.parquet"
DEFAULT_OUTDIR  = P.ROOT / "backtest_data" / "daily"

REQUIRED_BASE = ["datetime","ticker","open","high","low","close"]
EMA_NO_SUFFIX = ["ema5","ema20","ema44"]
EMA_D_SUFFIX  = ["ema5_d","ema20_d","ema44_d"]
SIGNAL_COL = "entry_signal"

def _ensure_emas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns ema5, ema20, ema44 exist.
    - If *_d versions exist, use them and rename to no-suffix.
    - Otherwise compute per-ticker EMAs from close.
    """
    have_d = all(c in df.columns for c in EMA_D_SUFFIX)
    have_plain = all(c in df.columns for c in EMA_NO_SUFFIX)

    if have_plain:
        return df

    out = df.copy()
    if have_d:
        out["ema5"]  = out["ema5_d"]
        out["ema20"] = out["ema20_d"]
        out["ema44"] = out["ema44_d"]
        return out

    # compute
    out = out.sort_values(["ticker","datetime"]).copy()
    def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
    out["ema5"]  = out.groupby("ticker", group_keys=False)["close"].apply(lambda s: _ema(s, 5))
    out["ema20"] = out.groupby("ticker", group_keys=False)["close"].apply(lambda s: _ema(s, 20))
    out["ema44"] = out.groupby("ticker", group_keys=False)["close"].apply(lambda s: _ema(s, 44))
    return out

def _default_signal(df: pd.DataFrame) -> pd.Series:
    """Fallback entry signal if none provided."""
    return (df["ema5"] > df["ema20"]) & (df["close"] > df["ema20"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET),
                    help="Path to prices_enriched.parquet (default: project DATA_ENRICHED)")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR),
                    help="Output folder for *_bt.csv (default: project backtest_data/daily)")
    args = ap.parse_args()

    FINAL = Path(args.parquet)
    OUTDIR = Path(args.outdir)

    if not FINAL.exists():
        raise SystemExit(f"[ERR] Missing {FINAL} — run s30 to build it first.")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FINAL)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize & basic checks
    missing = [c for c in REQUIRED_BASE if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] parquet missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime","ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str)

    # Ensure numeric OHLC
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure EMAs exist (use *_d if available; else compute)
    df = _ensure_emas(df)

    # Entry signal
    if SIGNAL_COL in df.columns:
        sig = df[SIGNAL_COL].astype(bool)
    else:
        sig = _default_signal(df)

    base = df[REQUIRED_BASE + EMA_NO_SUFFIX].copy()
    base[SIGNAL_COL] = sig

    # Per-ticker files
    tickers = base["ticker"].dropna().astype(str).unique()
    count = 0
    for t in tickers:
        g = base[base["ticker"] == t].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        if g.empty:
            continue
        out = OUTDIR / f"{t}_bt.csv"
        g.to_csv(out, index=False)
        count += 1

    print(f"[OK] Wrote {count} file(s) to {OUTDIR}")

if __name__ == "__main__":
    main()