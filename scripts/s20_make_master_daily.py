#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s20_make_master_daily.py — Build a master DAILY CSV from raw per-ticker CSVs

READS:
  <input_base>/daily/{TICKER}_daily_raw.csv
  - default input_base:
      1) --input-base if provided
      2) $SHARED_RAW_BASE
      3) common.paths.P.SHARED_RAW_BASE

WRITES (master CSV):
  <outbase>/etf_prices_daily_master.csv
  - default outbase:
      1) --outbase if provided
      2) $DATA_RAW_SHARED or $SHARED_RAW_BASE
      3) common.paths.P.SHARED_RAW_BASE

Universe:
  --excel (or $ETF_LIST_XLSX, else P.ETF_LIST)
  --sheet (or $ETF_SHEET, else default_etf_sheet())
"""

from __future__ import annotations
from pathlib import Path
import os
import argparse
import pandas as pd

from common.paths import P, default_etf_sheet


# ---------- helpers ----------
def _env_path(key: str) -> str | None:
    v = os.environ.get(key)
    return v if v and str(v).strip() else None

def _resolve_input_base(cli_val: str | None) -> Path:
    if cli_val and cli_val.strip():
        return Path(cli_val).expanduser().resolve()
    if _env_path("SHARED_RAW_BASE"):
        return Path(_env_path("SHARED_RAW_BASE")).expanduser().resolve()
    return Path(P.SHARED_RAW_BASE).expanduser().resolve()

def _resolve_out_base(cli_val: str | None) -> Path:
    if cli_val and cli_val.strip():
        return Path(cli_val).expanduser().resolve()
    # prefer project-scoped DATA_RAW_SHARED if provided
    for k in ("DATA_RAW_SHARED", "SHARED_RAW_BASE"):
        v = _env_path(k)
        if v:
            return Path(v).expanduser().resolve()
    return Path(P.SHARED_RAW_BASE).expanduser().resolve()

def _resolve_excel(cli_val: str | None) -> Path:
    if cli_val and cli_val.strip():
        return Path(cli_val).expanduser().resolve()
    if _env_path("ETF_LIST_XLSX"):
        return Path(_env_path("ETF_LIST_XLSX")).expanduser().resolve()
    return Path(P.ETF_LIST).expanduser().resolve()

def _resolve_sheet(cli_val: str | None) -> str:
    if cli_val and cli_val.strip():
        return cli_val.strip()
    if _env_path("ETF_SHEET"):
        return _env_path("ETF_SHEET").strip()
    return default_etf_sheet()

def _banner(input_base: Path, out_base: Path, daily_dir: Path, excel: Path, sheet: str):
    print(f"[s20] INPUT_BASE         : {input_base}")
    print(f"[s20] INPUT daily dir    : {daily_dir}")
    print(f"[s20] OUT_BASE (master)  : {out_base}")
    print(f"[s20] ETF list           : {excel} (sheet={sheet})")
    print(f"[s20] P.SHARED_RAW_BASE  : {P.SHARED_RAW_BASE}")
    print(f"[s20] P.DATA_ENRICHED    : {P.DATA_ENRICHED}")

def _read_tickers(xlsx: Path, sheet: str, prefer_col: str | None = None) -> list[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    cols = {c.strip().lower(): c for c in df.columns}
    if prefer_col and prefer_col.lower() in cols:
        chosen = cols[prefer_col.lower()]
    else:
        chosen = None
        for key in ("ticker", "symbol", "etf"):
            if key in cols:
                chosen = cols[key]; break
        if chosen is None:
            chosen = df.columns[0]
    s = df[chosen].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t and t not in {"NAN", "NONE"}]

def _load_ibkr_daily(path: Path, ticker: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df.rename(columns={"date": "datetime"})[
        ["datetime", "open", "high", "low", "close", "volume"]
    ].copy()
    out["ticker"] = ticker
    out["source"] = "ibkr_daily_raw"
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="", help="Path to ETF_list.xlsx (default: env or P.ETF_LIST)")
    ap.add_argument("--sheet", default="", help="Excel sheet for universe (default: env or project heuristic)")
    ap.add_argument("--col", default="", help="Preferred ticker column (Ticker/Symbol/ETF). Optional.")
    ap.add_argument("--input-base", dest="input_base", default=None,
                    help="Root containing daily/ and 30min/ (default: env SHARED_RAW_BASE or P.SHARED_RAW_BASE)")
    ap.add_argument("--input_base", dest="input_base_alt", default=None, help=argparse.SUPPRESS)  # legacy spelling
    ap.add_argument("--outbase", dest="out_base", default=None,
                    help="Where to write master CSV (default: DATA_RAW_SHARED or SHARED_RAW_BASE or P.SHARED_RAW_BASE)")
    args = ap.parse_args()

    # normalize legacy alias
    if getattr(args, "input_base_alt", None) and not args.input_base:
        args.input_base = args.input_base_alt

    excel = _resolve_excel(args.excel)
    sheet = _resolve_sheet(args.sheet)
    input_base = _resolve_input_base(args.input_base)
    out_base   = _resolve_out_base(args.out_base)
    daily_dir  = input_base / "daily"

    _banner(input_base, out_base, daily_dir, excel, sheet)

    if not excel.exists():
        raise SystemExit(f"[ERR] ETF list not found: {excel}")
    if not daily_dir.exists():
        raise SystemExit(f"[ERR] Daily input folder not found: {daily_dir}")

    tickers = _read_tickers(excel, sheet, args.col or None)
    if not tickers:
        raise SystemExit(f"[ERR] No tickers read from {excel} (sheet {sheet}).")

    rows, missing = [], []
    for t in tickers:
        f = daily_dir / f"{t}_daily_raw.csv"
        df = _load_ibkr_daily(f, t)
        if df.empty:
            missing.append(t); continue
        rows.append(df)

    if missing:
        print(f"[WARN] Missing/empty daily files for {len(missing)} tickers (first 20): {missing[:20]}")

    if not rows:
        raise SystemExit("[ERR] No daily inputs found for any ticker.")

    master = pd.concat(rows, ignore_index=True)
    master = master.dropna(subset=["datetime", "close"]).copy()
    master["datetime"] = pd.to_datetime(master["datetime"], utc=True)
    for c in ("open", "high", "low", "close", "volume"):
        master[c] = pd.to_numeric(master[c], errors="coerce")

    master = (master
              .sort_values(["ticker", "datetime"])
              .drop_duplicates(subset=["ticker", "datetime"]))[
              ["datetime", "ticker", "open", "high", "low", "close", "volume", "source"]]

    out_file = (out_base / "etf_prices_daily_master.csv").expanduser().resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_file, index=False)
    print(f"[OK] Wrote {out_file} | rows={len(master):,} | tickers={master['ticker'].nunique()}")

if __name__ == "__main__":
    main()