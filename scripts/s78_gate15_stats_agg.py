#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s78_gate15_stats_agg.py — Aggregate Gate-1.5 stats (multi-ticker rollup + confidence)

Purpose
-------
- Aggregate ticker-level Gate-1.5 stats (from s78_gate15_stats.py)
- Include confidence metrics and stretch benchmarks in the rollup
- Write out combined Parquet/CSV datasets for analytics and calibration

Inputs
------
- signals/analytics/gate15_stats.parquet (per-signal enriched stats)

Outputs
-------
- signals/analytics/gate15_stats_agg.parquet
- signals/analytics/gate15_stats_agg.csv

Usage
-----
  python scripts/s78_gate15_stats_agg.py
"""

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ---------- project paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P


def main():
    # Input
    in_path = P.SIGNALS_DIR / "analytics" / "gate15_stats.parquet"
    if not in_path.exists():
        raise SystemExit(f"[ERR] Missing input: {in_path}")

    df = pd.read_parquet(in_path)
    if df.empty:
        raise SystemExit("[ERR] gate15_stats is empty")

    # Separate numeric vs categorical
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c not in ("ticker",)]

    # Aggregate numeric: mean
    agg_num = df.groupby("ticker")[num_cols].mean().reset_index()

    # Confidence bucket distribution
    if "confidence_bucket" in df.columns:
        dist = (
            df.groupby(["ticker", "confidence_bucket"])
              .size()
              .unstack(fill_value=0)
              .reset_index()
        )
        # normalize to % of signals
        total = dist.drop(columns="ticker").sum(axis=1)
        for c in dist.columns:
            if c != "ticker":
                dist[c] = (dist[c] / total * 100).round(1)
        agg = pd.merge(agg_num, dist, on="ticker", how="left")
    else:
        agg = agg_num

    # Fill missing categorical with NaN-safe placeholders (for reference only, not aggregated)
    for c in cat_cols:
        if c not in agg.columns:
            agg[c] = np.nan

    # Outputs
    out_dir = P.SIGNALS_DIR / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "gate15_stats_agg.parquet"
    out_csv = out_dir / "gate15_stats_agg.csv"

    agg.to_parquet(out_parquet, index=False)
    agg.to_csv(out_csv, index=False)

    print(f"[OK] Aggregated stats → {out_parquet}")
    print(f"[OK] Aggregated stats → {out_csv}")


if __name__ == "__main__":
    main()