#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s00_check_first_dates.py — Show first & last available date per ticker × bucket.

- Uses QuantETF paths from common.paths (no hardcoded QuantShared).
- Buckets supported: daily, weekly, 30min (auto-skips if folder missing).
- Accepts alt filenames with or without '_raw' suffix.

Usage:
  python3 s00_check_first_dates.py
  python3 s00_check_first_dates.py --sheet signalsNY --buckets daily weekly
  ETF_SHEET=signalsNY python3 s00_check_first_dates.py
"""

from __future__ import annotations
from pathlib import Path
import argparse
import os
import pandas as pd

# ---- project-aware paths ----
try:
    from common.paths import P, default_etf_sheet
except Exception as e:
    raise SystemExit(f"[ERR] Cannot import common.paths: {e}")

# ---- defaults from paths / env ----
ETF_LIST: Path = P.ETF_LIST
DEFAULT_SHEET = os.environ.get("ETF_SHEET", "") or default_etf_sheet()

# Compute bucket directories safely (some repos don’t define weekly)
def _bucket_dirs():
    dirs = {}
    # prefer explicit shared dirs if present; else derive from SHARED_RAW_BASE
    base = getattr(P, "SHARED_RAW_BASE", None)
    d_30 = getattr(P, "SHARED_30M_DIR", None)
    d_1d = getattr(P, "SHARED_DAILY_DIR", None)
    d_1w = getattr(P, "SHARED_WEEKLY_DIR", None)

    if d_1d is None and base:
        d_1d = base / "daily"
    if d_1w is None and base:
        d_1w = base / "weekly"
    if d_30 is None and base:
        d_30 = base / "30min"

    if d_1d: dirs["daily"]  = Path(d_1d)
    if d_1w: dirs["weekly"] = Path(d_1w)
    if d_30: dirs["30min"]  = Path(d_30)
    return dirs

BUCKET_DIRS = _bucket_dirs()

# Column name that holds tickers in ETF_list.xlsx (case-insensitive options)
POSSIBLE_TICKER_COLS = ["ticker", "symbol", "etf"]

def load_tickers(path: Path, sheet: str, col_hint: str | None) -> list[str]:
    df = pd.read_excel(path, sheet_name=sheet)
    colmap = {c.lower().strip(): c for c in df.columns}
    col = None
    if col_hint and col_hint.lower() in colmap:
        col = colmap[col_hint.lower()]
    else:
        for c in POSSIBLE_TICKER_COLS:
            if c in colmap:
                col = colmap[c]
                break
    if not col:
        raise SystemExit(f"[ERR] No ticker column found in {path}:{sheet}. "
                         f"Looked for {POSSIBLE_TICKER_COLS} and hint '{col_hint}'.")
    s = df[col].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t]

def _find_csv(dirpath: Path, ticker: str, bucket: str) -> Path | None:
    """
    Return the best-matching CSV path for a given ticker/bucket.
    Tries both *_raw.csv and plain *.csv variants.
    """
    c1 = dirpath / f"{ticker}_{bucket}_raw.csv"
    if c1.exists(): return c1
    c2 = dirpath / f"{ticker}_{bucket}.csv"
    if c2.exists(): return c2
    # for 30min some feeds omit '_min'
    if bucket == "30min":
        c3 = dirpath / f"{ticker}_30m_raw.csv"
        c4 = dirpath / f"{ticker}_30m.csv"
        if c3.exists(): return c3
        if c4.exists(): return c4
    return None

def quick_first_last_count(csv_path: Path):
    """
    Efficiently get earliest/latest 'date' and total rows without loading whole file.
    Strategy:
      - Read the header to identify the 'date' column (case-insensitive).
      - Read first few rows to get a valid first date.
      - Stream the file in chunks to obtain the last valid date and row count.
    """
    if not csv_path or not csv_path.exists():
        return None, None, 0

    try:
        head = pd.read_csv(csv_path, nrows=5)
        date_col = next((c for c in head.columns if c.lower().strip() == "date"), None)
        if not date_col:
            # Try 'datetime' if 'date' is absent
            date_col = next((c for c in head.columns if c.lower().strip() == "datetime"), None)
        if not date_col:
            return None, None, 0

        # First valid date from the first small read
        first_part = pd.read_csv(csv_path, usecols=[date_col], nrows=200)
        first_dt = pd.to_datetime(first_part[date_col], errors="coerce", utc=True).dropna()
        first_dt = first_dt.iloc[0] if len(first_dt) else None

        # Stream chunks to get last valid date and total count
        last_dt = None
        total = 0
        for chunk in pd.read_csv(csv_path, usecols=[date_col], chunksize=100_000):
            total += len(chunk)
            dt = pd.to_datetime(chunk[date_col], errors="coerce", utc=True).dropna()
            if len(dt):
                last_dt = dt.iloc[-1]

        return first_dt, last_dt, int(total)
    except Exception:
        # Fallback: full single-column read
        try:
            colread = pd.read_csv(csv_path, usecols=lambda c: str(c).lower().strip() in ("date", "datetime"))
            dcol = "date" if "date" in colread.columns else ("datetime" if "datetime" in colread.columns else None)
            if not dcol:
                return None, None, 0
            s = pd.to_datetime(colread[dcol], errors="coerce", utc=True).dropna()
            return (s.min() if len(s) else None), (s.max() if len(s) else None), int(len(colread))
        except Exception:
            return None, None, 0

def main():
    ap = argparse.ArgumentParser(description="Show first/last dates per ticker×bucket.")
    ap.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Excel sheet (default: {DEFAULT_SHEET})")
    ap.add_argument("--buckets", nargs="+", default=["daily", "weekly", "30min"], help="Buckets to scan")
    ap.add_argument("--ticker_col", default=None, help="Ticker column hint (e.g., Ticker/Symbol)")
    args = ap.parse_args()

    # Validate and keep only buckets whose directories actually exist
    chosen = []
    for b in args.buckets:
        if b not in BUCKET_DIRS:
            print(f"[WARN] Unknown bucket '{b}' (available: {list(BUCKET_DIRS)}) — skipping.")
            continue
        if not BUCKET_DIRS[b] or not Path(BUCKET_DIRS[b]).exists():
            print(f"[WARN] Bucket '{b}' directory missing: {BUCKET_DIRS[b]} — skipping.")
            continue
        chosen.append(b)
    if not chosen:
        raise SystemExit("[ERR] No usable buckets found.")

    tickers = load_tickers(Path(ETF_LIST), args.sheet, args.ticker_col)

    print(f"[INFO] ETF_LIST         = {ETF_LIST}")
    print(f"[INFO] Sheet            = {args.sheet}")
    print(f"[INFO] Buckets          = {chosen}")
    print(f"[INFO] Tickers (unique) = {len(tickers)}")
    for b in chosen:
        print(f"[INFO] {b:6} dir        = {BUCKET_DIRS[b]}")
    print()

    # header
    header = ["Ticker"]
    for b in chosen:
        header += [f"{b}_first", f"{b}_last", f"{b}_rows"]
    print("\t".join(header))

    # rows
    for t in tickers:
        row = [t]
        for b in chosen:
            csv_dir = Path(BUCKET_DIRS[b])
            csv_path = _find_csv(csv_dir, t, b)
            if not csv_path:
                row += ["MISSING", "MISSING", "0"]
                continue
            first_dt, last_dt, count = quick_first_last_count(csv_path)
            row += [
                (first_dt.date().isoformat() if first_dt is not None else "ERR"),
                (last_dt.date().isoformat()  if last_dt is not None else "ERR"),
                str(count),
            ]
        print("\t".join(row))

if __name__ == "__main__":
    main()