#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s30_inspect_enriched_parquet.py
Quick inspector + debug snapshots for prices_enriched.parquet

What it does
------------
- Loads: P.DATA_ENRICHED/prices_enriched.parquet (or --parquet override)
- Prints: row/column counts, date range, per-ticker stats
- Validates: daily TA feature naming convention (*_d)
- Reports (optional): columns list, null-counts, per-ticker coverage
- Debug (optional): small sample snapshot saved to /reports/debug_enriched

Usage:
  python scripts/s30_inspect_enriched_parquet.py
  python scripts/s30_inspect_enriched_parquet.py --save_reports --debug
  python scripts/s30_inspect_enriched_parquet.py --parquet /custom/file.parquet
"""

import sys, argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# --- project bootstrapping for common.paths/settings ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # EU/US aware

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _reports_dir() -> Path:
    # Prefer P.REPORTS_DIR if available, else fallback
    rep = getattr(P, "REPORTS_DIR", None)
    d = rep if rep else (P.ROOT / "reports")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _debug_dir() -> Path:
    d = _reports_dir() / "debug_enriched"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_csv(df: pd.DataFrame, stem: str) -> Path:
    p = _reports_dir() / f"{stem}_{_ts()}.csv"
    df.to_csv(p, index=False)
    print(f"[OK] Saved → {p}")
    return p

def save_debug_sample(df: pd.DataFrame, stem: str = "s30_sample", n: int = 200):
    snap = df.head(n).copy()
    pq = _debug_dir() / f"{stem}_{_ts()}.parquet"
    snap.to_parquet(pq, index=False)
    print(f"[OK] Debug snapshot → {pq} (rows={len(snap)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        type=str,
        default=str(P.DATA_ENRICHED / "prices_enriched.parquet"),
        help="Path to prices_enriched.parquet (default: P.DATA_ENRICHED/prices_enriched.parquet)",
    )
    ap.add_argument("--save_reports", action="store_true",
                    help="Write CSV reports (columns, nulls, coverage) to /reports")
    ap.add_argument("--debug", action="store_true",
                    help="Save a small debug snapshot parquet to /reports/debug_enriched")
    ap.add_argument("--show_head", type=int, default=0,
                    help="Print head(N) rows to console (0=skip)")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(f"[ERR] Not found: {path}")

    print(f"[INFO] Loading: {path}")
    df = pd.read_parquet(path)
    df.columns = [c.strip() for c in df.columns]

    # --- basic info ---
    print("\n=== BASIC INFO ===")
    print(f"Rows: {len(df):,}  | Cols: {df.shape[1]}")
    if "datetime" in df.columns:
        dts = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        print(f"Date range: {dts.min()} → {dts.max()}")
    if "ticker" in df.columns:
        uniq = df["ticker"].dropna().astype(str).nunique()
        print(f"Tickers: {uniq}")

    # --- columns overview ---
    cols = pd.DataFrame({"column": df.columns})
    def _tag(c: str) -> str:
        if c in ("datetime","ticker","open","high","low","close","volume","outlier_flag","source_d","_source"):
            return "base"
        if c.endswith("_d"):
            return "daily_feature"
        return "other"
    cols["class"] = cols["column"].map(_tag)
    print("\n=== COLUMNS (first 25) ===")
    print(cols.head(25).to_string(index=False))
    print(f"... total columns: {len(cols)}")

    # --- validate *_d naming for TA/labels ---
    print("\n=== NAMING VALIDATION (*_d) ===")
    daily_like_keys = ("ema", "rsi", "atr", "adx", "bb_", "macd_", "stoch_", "is_trending", "is_sideways", "ema_slope")
    daily_like = [c for c in df.columns if any(k in c for k in daily_like_keys)]
    missing_d_suffix = [c for c in daily_like if not c.endswith("_d")]
    if missing_d_suffix:
        print("[WARN] Columns likely daily features but missing '_d' suffix (first 30):")
        for c in missing_d_suffix[:30]:
            print("   -", c)
        if len(missing_d_suffix) > 30:
            print(f"   ... (+{len(missing_d_suffix)-30} more)")
    else:
        print("[OK] Daily TA/labels follow '*_d' convention.")

    # --- null counts ---
    print("\n=== NULL COUNTS (top 20 by %null) ===")
    nulls = df.isna().mean().sort_values(ascending=False)
    top_nulls = nulls.head(20).reset_index()
    top_nulls.columns = ["column", "null_frac"]
    try:
        print(top_nulls.to_string(index=False, formatters={"null_frac": "{:.2%}".format}))
    except Exception:
        # pandas version differences
        top_nulls["null_pct"] = (top_nulls["null_frac"] * 100).round(2)
        print(top_nulls.drop(columns=["null_frac"]).to_string(index=False))

    # --- per-ticker coverage ---
    print("\n=== PER-TICKER COVERAGE (first 15) ===")
    if {"ticker","datetime"}.issubset(df.columns):
        dd = df.copy()
        dd["datetime"] = pd.to_datetime(dd["datetime"], errors="coerce", utc=True)
        cov = (dd.groupby("ticker", dropna=False)
                 .agg(first_dt=("datetime","min"),
                      last_dt =("datetime","max"),
                      rows    =("datetime","size"))
                 .reset_index()
                 .sort_values("ticker"))
        print(cov.head(15).to_string(index=False))
    else:
        cov = pd.DataFrame(columns=["ticker","first_dt","last_dt","rows"])
        print("(ticker/datetime missing; coverage omitted)")

    # --- show head(N) if asked ---
    if args.show_head > 0:
        print(f"\n=== HEAD({args.show_head}) ===")
        try:
            print(df.head(args.show_head).to_string(index=False))
        except Exception:
            print(df.head(args.show_head))

    # --- save reports/debug if requested ---
    if args.save_reports:
        save_csv(cols, "s30_columns")
        try:
            save_csv(
                top_nulls.assign(null_pct=(top_nulls["null_frac"] * 100).round(2)).drop(columns=["null_frac"]),
                "s30_nulls"
            )
        except Exception:
            save_csv(top_nulls, "s30_nulls")
        if not cov.empty:
            save_csv(cov, "s30_ticker_coverage")

    if args.debug:
        save_debug_sample(df, stem="s30_sample", n=300)

    print("\n[OK] s30 inspection complete.")

if __name__ == "__main__":
    main()