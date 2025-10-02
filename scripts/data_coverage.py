#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_first_dates.py — Show first available date per ticker & frequency.

Reads tickers from ETF_list.xlsx (sheet=signalsUSD).
Scans QuantShared data_raw_ETF_US (change to EU if needed).
Prints the earliest 'date' for each ticker×bucket (daily, weekly, 30min).
"""

from pathlib import Path
import pandas as pd

# --- CONFIG ---
SHARED_BASE = Path("/Users/Finance/QuantShared")
BASE_DIR = SHARED_BASE / "data_raw_ETF_US"   # adjust for EU
ETF_LIST = SHARED_BASE / "ETF_list.xlsx"
SHEET = "signalsUSD"
BUCKETS = ["daily", "weekly", "30min"]
TICKER_COL = "Ticker"


def load_tickers(path: Path, sheet: str, col: str) -> list[str]:
    df = pd.read_excel(path, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    if col.lower() not in colmap:
        raise SystemExit(f"[ERR] Column '{col}' not found in {sheet}")
    s = df[colmap[col.lower()]].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t]


def get_first_date(path: Path):
    try:
        df = pd.read_csv(path, nrows=5)  # only need the head
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        return df["date"].min()
    except Exception:
        return None


def main():
    tickers = load_tickers(ETF_LIST, SHEET, TICKER_COL)
    print(f"[INFO] Checking first dates in {BASE_DIR} for {len(tickers)} tickers")
    header = ["Ticker"] + [f"{b}_first" for b in BUCKETS]
    print("\t".join(header))
    for t in tickers:
        row = [t]
        for b in BUCKETS:
            path = BASE_DIR / b / f"{t}_{b}_raw.csv"
            if not path.exists():
                row.append("MISSING")
                continue
            d = get_first_date(path)
            row.append(d.date().isoformat() if d is not None else "ERR")
        print("\t".join(row))


if __name__ == "__main__":
    main()