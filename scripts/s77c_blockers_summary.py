#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s77c_blockers_summary.py — Summarize rule blockers across boards_ds

Reads the Parquet dataset written by s77/s77a (signals/boards_ds),
focuses on DO NOTHING rows, explodes reason_long / reason_short,
and produces:
  - Top reasons overall
  - Reasons by year
  - Reasons by ticker
  - Reason → metric snapshots (RSI, Donchian pos/width, ADX)

Outputs to signals/reports:
  - blockers_summary_<ts>.xlsx  (multiple sheets)
  - blockers_top_reasons_<ts>.csv
  - blockers_by_year_<ts>.csv
  - blockers_by_ticker_<ts>.csv

Usage examples:
  python3 scripts/s77c_blockers_summary.py
  python3 scripts/s77c_blockers_summary.py --since 2019-01-01 --until 2025-09-12
  python3 scripts/s77c_blockers_summary.py --top 30 --min-count 25
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import sys

# project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # type: ignore

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M")

def _read_boards_ds(ds_dir: Path) -> pd.DataFrame:
    if not ds_dir.exists():
        raise SystemExit(f"[ERR] Parquet dataset not found: {ds_dir}")
    # Read all partitions
    df = pd.read_parquet(ds_dir)  # pyarrow handles dir as dataset
    # Normalize
    if "board_day" in df.columns:
        # board_day may be str like '20250512'
        try:
            df["board_day"] = pd.to_datetime(df["board_day"], format="%Y%m%d")
        except Exception:
            df["board_day"] = pd.to_datetime(df["board_day"], errors="coerce")
    else:
        df["board_day"] = pd.NaT
    # Year column for grouping
    df["year"] = df["board_day"].dt.year
    # Ensure columns exist
    for c in ["reason_long", "reason_short", "ticker", "decision",
              "rsi14_d", "donchian_pos_pct", "donchian_width_pct",
              "adx14_d", "adx14_prev_d", "adx14_delta", "confidence_score"]:
        if c not in df.columns:
            df[c] = np.nan if c != "ticker" else ""
    return df

def _explode_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Return tall table with one row per (ticker, day, side, reason)."""
    x = df[df["decision"] == "DO NOTHING"].copy()

    # Prepare LONG reasons
    long_df = x[["ticker","board_day","year","reason_long","rsi14_d",
                 "donchian_pos_pct","donchian_width_pct",
                 "adx14_d","adx14_prev_d","adx14_delta","confidence_score"]].copy()
    long_df["side"] = "long"
    long_df["reason_str"] = long_df["reason_long"].fillna("")

    # Prepare SHORT reasons
    short_df = x[["ticker","board_day","year","reason_short","rsi14_d",
                  "donchian_pos_pct","donchian_width_pct",
                  "adx14_d","adx14_prev_d","adx14_delta","confidence_score"]].copy()
    short_df["side"] = "short"
    short_df["reason_str"] = short_df["reason_short"].fillna("")

    both = pd.concat([long_df, short_df], ignore_index=True)

    # Split and explode
    both["reason"] = both["reason_str"].str.split(",")
    tall = both.explode("reason")
    tall["reason"] = tall["reason"].fillna("").str.strip()
    tall = tall[tall["reason"] != ""].copy()

    # Optional: standardize common tokens if needed (keep as-is for now)
    return tall

def _summaries(tall: pd.DataFrame, top_n: int, min_count: int):
    # Overall top reasons
    top_all = (tall["reason"].value_counts()
               .rename_axis("reason").reset_index(name="count"))
    if min_count > 1:
        top_all = top_all[top_all["count"] >= min_count]
    if top_n:
        top_all = top_all.head(top_n)

    # By year
    by_year = (tall.groupby(["year","reason"])
               .size().reset_index(name="count")
               .sort_values(["year","count"], ascending=[True, False]))
    if min_count > 1:
        by_year = by_year[by_year["count"] >= min_count]

    # By ticker
    by_ticker = (tall.groupby(["ticker","reason"])
                 .size().reset_index(name="count")
                 .sort_values(["ticker","count"], ascending=[True, False]))
    if min_count > 1:
        by_ticker = by_ticker[by_ticker["count"] >= min_count]

    # Reason → metric snapshots (median, IQR)
    metrics = (tall.groupby("reason")
               .agg(
                   n=("reason","size"),
                   rsi_med=("rsi14_d","median"),
                   rsi_q25=("rsi14_d", lambda s: np.nanpercentile(s.dropna(), 25) if s.notna().any() else np.nan),
                   rsi_q75=("rsi14_d", lambda s: np.nanpercentile(s.dropna(), 75) if s.notna().any() else np.nan),
                   don_pos_med=("donchian_pos_pct","median"),
                   don_pos_q25=("donchian_pos_pct", lambda s: np.nanpercentile(s.dropna(), 25) if s.notna().any() else np.nan),
                   don_pos_q75=("donchian_pos_pct", lambda s: np.nanpercentile(s.dropna(), 75) if s.notna().any() else np.nan),
                   don_w_med=("donchian_width_pct","median"),
                   adx_med=("adx14_d","median"),
                   adx_delta_med=("adx14_delta","median"),
                   conf_med=("confidence_score","median"),
               )
               .reset_index()
               .sort_values("n", ascending=False))
    if min_count > 1:
        metrics = metrics[metrics["n"] >= min_count]

    return top_all, by_year, by_ticker, metrics

def main():
    ap = argparse.ArgumentParser(description="Summarize blockers from boards_ds.")
    ap.add_argument("--since", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--until", type=str, default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--top", type=int, default=30, help="Top N reasons to display (overall).")
    ap.add_argument("--min-count", type=int, default=5, help="Filter out reasons with < min-count in tables.")
    ap.add_argument("--out-prefix", type=str, default=None, help="Custom output filename prefix.")
    args = ap.parse_args()

    ds_dir = P.ROOT / "signals" / "boards_ds"
    out_dir = P.ROOT / "signals" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts()
    prefix = args.out_prefix or f"blockers_{ts}"

    print(f"[INFO] Reading Parquet dataset from {ds_dir}")
    df = _read_boards_ds(ds_dir)

    # Date range filters if provided
    if args.since:
        df = df[df["board_day"] >= pd.to_datetime(args.since)]
    if args.until:
        df = df[df["board_day"] <= pd.to_datetime(args.until)]

    # Build tall reasons
    tall = _explode_reasons(df)
    if tall.empty:
        print("[WARN] No blockers found (tall table empty).")
        sys.exit(0)

    # Summaries
    top_all, by_year, by_ticker, metrics = _summaries(tall, args.top, args.min_count)

    # Console quick peek
    print("\n=== Top Reasons (overall) ===")
    print(top_all.to_string(index=False))

    # Write CSVs
    top_csv = out_dir / f"{prefix}_top_reasons.csv"
    year_csv = out_dir / f"{prefix}_by_year.csv"
    tkr_csv = out_dir / f"{prefix}_by_ticker.csv"
    met_csv = out_dir / f"{prefix}_reason_metrics.csv"

    top_all.to_csv(top_csv, index=False)
    by_year.to_csv(year_csv, index=False)
    by_ticker.to_csv(tkr_csv, index=False)
    metrics.to_csv(met_csv, index=False)

    # Excel workbook
    xlsx = out_dir / f"{prefix}.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as xl:
        top_all.to_excel(xl, sheet_name="TopReasonsAll", index=False)
        by_year.to_excel(xl, sheet_name="ReasonsByYear", index=False)
        by_ticker.to_excel(xl, sheet_name="ReasonsByTicker", index=False)
        metrics.to_excel(xl, sheet_name="ReasonMetrics", index=False)

        # Basic autofit widths
        for sheet in ["TopReasonsAll","ReasonsByYear","ReasonsByTicker","ReasonMetrics"]:
            ws = xl.sheets[sheet]
            # crude autofit
            for col_idx, col_name in enumerate(df.columns):
                col_width = max(len(str(col_name)), int(df[col_name].astype(str).map(len).max()))
                ws.set_column(col_idx, col_idx, min(col_width + 2, 50))  # cap width at 50
                pass  # (xlsxwriter has no true autofit; we'll just set a sane width below)
            ws.set_column(0, 0, 26)   # reason / ticker
            ws.set_column(1, 8, 14)

    print(f"\n[OK] Wrote:\n  {xlsx}\n  {top_csv}\n  {year_csv}\n  {tkr_csv}\n  {met_csv}")

if __name__ == "__main__":
    main()