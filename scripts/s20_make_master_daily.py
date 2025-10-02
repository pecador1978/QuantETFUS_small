#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s20_make_master_daily.py — Build a single master DAILY CSV from raw targeted ETFs (UTC)

- Read DAILY raw per-ticker CSVs from QuantShared/data_raw_ETF_US/daily (or EU, per common.paths).
- Concatenate into a single master CSV for s30.
- No macro/forex here (handled elsewhere).

Inputs
- ETF list from P.ETF_LIST (sheet resolved via CLI/env/settings with sensible fallbacks).
- Raw files:
    data_raw_ETF_*/daily/{TICKER}_daily_raw.csv   (IBKR schema: date,open,high,low,close,volume,...)

Output
- DATA_RAW/etf_prices_daily_master.csv
  Columns: datetime,ticker,open,high,low,close,volume,source  (UTC)
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# --- project bootstrapping ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # shared paths (handles QuantShared etc.)

def _resolve_sheet(cli_sheet: str | None) -> str:
    """
    Resolve ETF sheet in order of priority:
    1) CLI --sheet
    2) ENV ETF_SHEET
    3) common.settings.default_etf_sheet()
    4) common.paths.default_etf_sheet()
    5) 'signals'
    """
    if cli_sheet and cli_sheet.strip():
        return cli_sheet.strip()
    env_sheet = os.environ.get("ETF_SHEET", "").strip()
    if env_sheet:
        return env_sheet
    # try settings
    try:
        from common.settings import default_etf_sheet as _def_sheet_settings  # type: ignore
        return _def_sheet_settings()
    except Exception:
        pass
    # try paths
    try:
        from common.paths import default_etf_sheet as _def_sheet_paths  # type: ignore
        return _def_sheet_paths()
    except Exception:
        pass
    return "signals"

# --- where to READ daily raws from (QuantShared) ---
# Default to US; override at runtime:  export SHARED_RAW_BASE="/Users/Finance/QuantShared/data_raw_ETF_EU"
SHARED_BASE = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
DIR_TT_D = SHARED_BASE / "daily"  # <— READ here

# --- where to WRITE the master (keep inside the project) ---
OUT_FILE = P.DATA_RAW / "etf_prices_daily_master.csv"

def _banner(sheet: str):
    print(f"[s20] ROOT={P.ROOT}")
    print(f"[s20] DATA_RAW(project)={P.DATA_RAW}")
    print(f"[s20] SHARED_BASE={SHARED_BASE}")
    print(f"[s20] TT daily dir (read)={DIR_TT_D}")
    print(f"[s20] ETF list={P.ETF_LIST} (sheet={sheet})")

def read_tickers(xlsx: Path, sheet: str, col: str = "Ticker") -> list[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col.lower()
    if key not in colmap:
        raise ValueError(f"Column '{col}' not found in {xlsx} (sheet '{sheet}').")
    vals = df[colmap[key]].astype(str).str.strip().str.upper().tolist()
    return sorted({v for v in vals if v})

def load_ibkr_daily(path: Path, ticker: str) -> pd.DataFrame:
    """IBKR daily schema: date,open,high,low,close,volume,..."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    out = df.rename(columns={"date": "datetime"})[["datetime","open","high","low","close","volume"]]
    out["ticker"] = ticker
    out["source"] = "ibkr_daily_raw"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="", help="Override ETF sheet (e.g., signalsUSD). Else: env/settings/fallback.")
    args = ap.parse_args()

    sheet = _resolve_sheet(args.sheet)
    _banner(sheet)

    if not DIR_TT_D.exists():
        raise SystemExit(f"[ERR] Daily input folder not found: {DIR_TT_D}")

    tickers = read_tickers(P.ETF_LIST, sheet, "Ticker")
    if not tickers:
        raise SystemExit(f"[ERR] No tickers read from {P.ETF_LIST} (sheet {sheet}).")

    rows, missing = [], []
    for t in tickers:
        f = DIR_TT_D / f"{t}_daily_raw.csv"
        df = load_ibkr_daily(f, t)
        if df.empty:
            missing.append(t)
            continue
        rows.append(df)

    if missing:
        print(f"[WARN] Missing/empty daily files for {len(missing)} tickers (first 20): {missing[:20]}")

    if not rows:
        raise SystemExit("[ERR] No daily inputs found for any ticker.")

    master = pd.concat(rows, axis=0, ignore_index=True)
    master = master.dropna(subset=["datetime","close"])
    master["datetime"] = pd.to_datetime(master["datetime"], utc=True)
    for c in ["open","high","low","close","volume"]:
        master[c] = pd.to_numeric(master[c], errors="coerce")
    master = (master
              .sort_values(["ticker","datetime"])
              .drop_duplicates(subset=["ticker","datetime"]))
    master = master[["datetime","ticker","open","high","low","close","volume","source"]]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_FILE, index=False)
    print(f"[OK] Wrote {OUT_FILE} | rows={len(master):,} | tickers={master['ticker'].nunique()}")

if __name__ == "__main__":
    main()