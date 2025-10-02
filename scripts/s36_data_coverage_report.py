#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s36_coverage_report.py
Report the first available timestamp per ticker from prices_enriched.parquet.

Assumes you've already created:
  - data_raw/etf_prices_daily_master.csv via s20
  - data_enriched/prices_enriched.parquet via s30

Defaults:
- Input parquet: P.DATA_ENRICHED / "prices_enriched.parquet"
- Output CSV  : P.REPORTS_DIR / "s16_first_dates_parquet_<ts>.csv"

Usage:
  python scripts/s16_coverage_report.py
  python scripts/s16_coverage_report.py --parquet /path/to/prices_enriched.parquet
  python scripts/s16_coverage_report.py --ticker_col ticker --dt_col datetime
"""

from pathlib import Path
from datetime import datetime, timezone
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
OUTDIR = P.REPORTS_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

def _tsnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _auto_detect_cols(df: pd.DataFrame, ticker_col: str | None, dt_col: str | None) -> tuple[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}

    # ticker column
    if ticker_col:
        if ticker_col in df.columns:
            tcol = ticker_col
        elif ticker_col.lower() in cols_lower:
            tcol = cols_lower[ticker_col.lower()]
        else:
            raise SystemExit(f"[ERR] --ticker_col '{ticker_col}' not found. Columns: {list(df.columns)}")
    else:
        for cand in ("ticker", "symbol", "code", "tkr", "instrument"):
            if cand in cols_lower:
                tcol = cols_lower[cand]
                break
        else:
            raise SystemExit("[ERR] Could not detect ticker column (looked for: ticker/symbol/...).")

    # datetime column
    if dt_col:
        if dt_col in df.columns:
            dcol = dt_col
        elif dt_col.lower() in cols_lower:
            dcol = cols_lower[dt_col.lower()]
        else:
            raise SystemExit(f"[ERR] --dt_col '{dt_col}' not found. Columns: {list(df.columns)}")
    else:
        for cand in ("dt", "datetime", "timestamp", "time", "date", "datetime_utc", "ts"):
            if cand in cols_lower:
                dcol = cols_lower[cand]
                break
        else:
            raise SystemExit("[ERR] Could not detect datetime column (looked for: dt/datetime/timestamp/...).")

    return tcol, dcol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET),
                    help="Path to enriched parquet (default: DATA_ENRICHED/prices_enriched.parquet)")
    ap.add_argument("--ticker_col", type=str, default=None, help="Explicit ticker column name")
    ap.add_argument("--dt_col", type=str, default=None, help="Explicit datetime column name")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise SystemExit(
            f"[ERR] Missing parquet file: {parquet_path}\n"
            "      Run s20_make_master_daily.py then s30_enrich_to_parquet.py first."
        )

    print(f"[INFO] Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    tcol, dcol = _auto_detect_cols(df, args.ticker_col, args.dt_col)

    # normalize types
    df[tcol] = df[tcol].astype(str)
    df[dcol] = pd.to_datetime(df[dcol], utc=True, errors="coerce")

    first_dates = (
        df.groupby(tcol, dropna=False)[dcol]
          .min()
          .reset_index()
          .rename(columns={tcol: "ticker", dcol: "first_dt_utc"})
          .sort_values("first_dt_utc", kind="mergesort")
          .reset_index(drop=True)
    )

    ts = _tsnow()
    out_path = OUTDIR / f"s16_first_dates_parquet_{ts}.csv"
    first_dates.to_csv(out_path, index=False)

    print(f"[OK] First dates → {out_path}")
    # pretty print a small preview
    try:
        print(first_dates.to_string(index=False, max_rows=60))
    except Exception:
        print(first_dates.head(60).to_string(index=False))

if __name__ == "__main__":
    main()