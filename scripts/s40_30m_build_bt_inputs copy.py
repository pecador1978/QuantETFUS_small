#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s40_30m_build_bt_inputs.py
From raw 30m CSVs → per-ticker *_bt_30m.csv used by s50_30m backtests.

Context
-------
- Part of the 30-minute stream. Converts raw OHLC data into enriched 30m backtest inputs.
- Takes raw 30m data (downloaded from TradingView/IBKR into QuantShared/data_raw_ETF_*/30min) and standardizes it.
- Adds EMAs and computes the entry signal used later in s50_30m.
- Produces lightweight *_bt_30m.csv files per ticker.

Reads
-----
- P.DATA_RAW/30min/{TICKER}_30min_raw.csv
- or P.DATA_RAW/30min/{TICKER}_30min.csv
  Expected (case-insensitive): datetime, open, high, low, close [, volume]

Processing
----------
- Normalize datetime → tz-aware UTC, clean OHLC
- Compute EMAs: ema5, ema20, ema44, ema275
- Entry signal (cross-lift): (ema5>ema44 & ema20>ema44) crossing, with trend filter close>ema275

Writes
------
- P.ROOT/backtest_data/30min/{TICKER}_bt_30m.csv
  Columns: datetime,ticker,open,high,low,close,ema5,ema20,ema44,ema275,entry_signal
"""

from pathlib import Path
import sys
import os
import argparse
import pandas as pd

# ---- import centralized paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402

pd.set_option("future.no_silent_downcasting", True)

# ---------------- CLI / input dir resolution ----------------
def resolve_in_dir(bucket_arg: str | None) -> Path:
    # 0) If SHARED_RAW_BASE is set, use it
    env_shared = os.environ.get("SHARED_RAW_BASE", "").strip()
    if env_shared:
        cand = Path(env_shared) / "30min"
        if cand.exists():
            return cand

    # 1) Project-local shared layout
    shared = P.DATA_RAW / "30min"
    if shared.exists():
        return shared

    # 2) QuantShared flat layouts
    qs = getattr(P, "QUANTSHARED", None) or (P.ROOT.parent / "QuantShared")
    cand_us = qs / "data_raw_ETF_US" / "30min"
    cand_eu = qs / "data_raw_ETF"    / "30min"
    if cand_us.exists():
        return cand_us
    if cand_eu.exists():
        return cand_eu

    # 3) Bucketed fallback under project data_raw
    bucket = (
        (bucket_arg or "").strip()
        or os.environ.get("TARGET_BUCKET", "").strip()
        or next(
            (b for b in ("targeted_ETFs_US", "targeted_ETFs")
             if (P.DATA_RAW / b / "30min").exists()),
            ""
        )
    )
    if bucket:
        return P.DATA_RAW / bucket / "30min"

    raise SystemExit(
        "[ERR] Could not find 30min input folder.\n"
        f"  Tried project: {shared}\n"
        f"  Tried shared US: {cand_us}\n"
        f"  Tried shared EU: {cand_eu}\n"
        f"  And bucketed fallbacks under {P.DATA_RAW}/<bucket>/30min\n"
        "  (Tip: export SHARED_RAW_BASE=/Users/Finance/QuantShared/data_raw_ETF_US)"
    )

# ---------------- loaders & helpers ----------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize datetime column name
    if "datetime" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})

    req = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def make_entry_signal(g: pd.DataFrame) -> pd.Series:
    """
    Entry when 30m EMAs 'lift' above ema44 under an ema275 up-trend:
      - Filter: close > ema275
      - Trigger: (ema5>ema44 AND ema20>ema44) and previously NOT both above (cross-lift)
    """
    above_now = (g["ema5"] > g["ema44"]) & (g["ema20"] > g["ema44"])
    above_prev = above_now.shift(1).fillna(False).astype(bool)
    cross_lift = above_now & (~above_prev)
    trend_ok = g["close"] > g["ema275"]
    return (cross_lift & trend_ok)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Fallback bucket under data_raw/ if shared layout isn't found "
             "(e.g., targeted_ETFs_US). Default: use shared P.DATA_RAW/30min."
    )
    args = ap.parse_args()

    IN_DIR = resolve_in_dir(args.bucket)
    OUT_DIR = P.ROOT / "backtest_data" / "30min"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(IN_DIR.glob("*_30min_raw.csv")) + list(IN_DIR.glob("*_30min.csv")))
    if not files:
        raise SystemExit(f"[ERR] No input files in {IN_DIR}")

    print(f"[INFO] Input dir: {IN_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    count = 0
    errors = 0

    for f in files:
        try:
            tkr = f.name.replace("_30min_raw.csv", "").replace("_30min.csv", "")
            df = load_csv(f)

            # EMAs
            df["ema5"]   = ema(df["close"], 5)
            df["ema20"]  = ema(df["close"], 20)
            df["ema44"]  = ema(df["close"], 44)
            df["ema275"] = ema(df["close"], 275)

            # Entry signal
            df["entry_signal"] = make_entry_signal(df).astype(bool)

            # Output
            out = df[["datetime","open","high","low","close","ema5","ema20","ema44","ema275"]].copy()
            out.insert(1, "ticker", tkr)
            out["entry_signal"] = df["entry_signal"]

            out_path = OUT_DIR / f"{tkr}_bt_30m.csv"
            out.to_csv(out_path, index=False)
            count += 1
        except Exception as e:
            errors += 1
            print(f"[WARN] {f.name}: {e}")

    print(f"[OK] Wrote {count} file(s) to {OUT_DIR}  |  skipped with warnings: {errors}")

if __name__ == "__main__":
    main()