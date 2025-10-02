#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s77b_analyze_boards.py — Analyze Gate-1 boards history from boards_ds parquet.
Summarizes BUY/SELL counts and confidence scores by year & ticker.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "signals" / "boards_ds"

def main():
    if not DS.exists():
        raise SystemExit(f"[ERR] {DS} not found")

    print(f"[INFO] Reading Parquet dataset from {DS}")
    df = pd.read_parquet(DS)

    if df.empty:
        raise SystemExit("[ERR] boards_ds is empty")

    # Convert board_day to year
    if "board_day" in df.columns:
        df["board_day"] = pd.to_datetime(df["board_day"], format="%Y%m%d", errors="coerce")
        df["year"] = df["board_day"].dt.year
    else:
        raise SystemExit("[ERR] board_day column missing")

    # --- Summary 1: Decision counts by year ---
    print("\n=== Decisions by Year ===")
    print(df.groupby("year")["decision"].value_counts().unstack(fill_value=0))

    # --- Summary 2: Decision counts by ticker ---
    print("\n=== Decisions by Ticker ===")
    print(df.groupby("ticker")["decision"].value_counts().unstack(fill_value=0).sort_index())

    # --- Summary 3: Confidence score by year ---
    if "confidence_score" in df.columns:
        print("\n=== Confidence Score by Year ===")
        print(df.groupby("year")["confidence_score"].describe())

    # Save to Excel
    out = ROOT / "signals" / "boards_analysis.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.groupby("year")["decision"].value_counts().unstack(fill_value=0).to_excel(writer, sheet_name="Decisions_by_Year")
        df.groupby("ticker")["decision"].value_counts().unstack(fill_value=0).to_excel(writer, sheet_name="Decisions_by_Ticker")
        if "confidence_score" in df.columns:
            df.groupby("year")["confidence_score"].describe().to_excel(writer, sheet_name="Confidence_by_Year")

    print(f"\n[OK] Exported analysis → {out}")

if __name__ == "__main__":
    main()