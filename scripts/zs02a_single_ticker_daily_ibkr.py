#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s02a_single_ticker_daily_ibkr.py — FULL overwrite daily download (IBKR schema)

!!! TRADABLE TICKERS ONLY !!!

- Pulls full window (default 10Y) of DAILY bars for one or many tickers.
- IBKR-format CSV: date,open,high,low,close,volume,average,barCount (UTC).
- Output: <project>/data_raw/<TARGET_BUCKET>/daily/{TICKER}_daily_raw.csv

Usage examples:
  # one ticker (default AIGA if none given)
  python scripts/s02a_single_ticker_daily_ibkr.py --ticker AIGA

  # multiple tickers (comma or space separated)
  python scripts/s02a_single_ticker_daily_ibkr.py --tickers "AIGA BATT DGTL"

  # force MIDPOINT & 15Y window
  python scripts/s02a_single_ticker_daily_ibkr.py --tickers AIGA,BATT --what MIDPOINT --duration "15 Y"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ib_insync import IB, Contract, util

# ---------------- bootstrapping ----------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P                     # project-aware paths
from common.settings import TARGET_BUCKET      # US_small aware

# ---------------- config ----------------
HARDCODED_TICKERS: List[str] = ["AIGA"]  # <- edit this list any time

DEFAULT_DURATION = "10 Y"
DEFAULT_USE_RTH = 1
DEFAULT_CCY = "USD" if PROJECT_ROOT.name.endswith(("US_small", "US")) else "EUR"

# write DAILY bars to the shared raw store (US). For EU, set env SHARED_RAW_BASE to .../data_raw_ETF_EU
SHARED_BASE = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
OUT_DIR = SHARED_BASE / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ticker mapping
MAPPING_CSV = Path("/Users/Finance/QuantShared/ticker_mapping.csv")

# ---------------- helpers ----------------
def parse_tickers(single: str | None, many: str | None) -> List[str]:
    if single:
        return [single.strip().upper()]
    if many:
        raw = many.replace(",", " ").split()
        return [t.strip().upper() for t in raw if t.strip()]
    return [t.upper() for t in HARDCODED_TICKERS]

def load_mapping(path_csv: Path) -> Dict[str, Dict[str, str]]:
    if not path_csv.exists():
        return {}
    df = pd.read_csv(path_csv, sep=";")
    need = {"Ticker","SecType","Exchange","Currency","PrimaryExchange"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"{path_csv} must contain columns: {need}")
    mp: Dict[str, Dict[str, str]] = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType": (str(r.get("SecType","STK")).strip() or "STK").upper(),
            "Exchange": str(r.get("Exchange","SMART")).strip() or "SMART",
            "Currency": (str(r.get("Currency", DEFAULT_CCY)).strip() or DEFAULT_CCY).upper(),
            "PrimaryExchange": str(r.get("PrimaryExchange","")).strip()
        }
    return mp

def make_contract(ticker: str, mapping: Dict[str, Dict[str, str]]) -> Contract:
    m = mapping.get(ticker, {})
    c = Contract()
    c.symbol = ticker
    c.secType = m.get("SecType", "STK")
    c.exchange = m.get("Exchange", "SMART")
    c.currency = m.get("Currency", DEFAULT_CCY)
    if m.get("PrimaryExchange"):
        c.primaryExchange = m["PrimaryExchange"]
    return c

def normalize_ib_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in out.columns:
            out[c] = pd.NA
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["date","open","high","low","close","volume","average","barCount"]].dropna(subset=["date"]).sort_values("date")

def fetch_full_daily(ib: IB, contract: Contract, duration: str, what: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",             # now
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    return normalize_ib_df(util.df(bars)) if bars else normalize_ib_df(pd.DataFrame())

def try_trades_midpoint(ib: IB, contract: Contract, duration: str, use_rth: bool, force_what: str | None) -> tuple[pd.DataFrame,str]:
    if force_what:
        df = fetch_full_daily(ib, contract, duration, force_what.upper(), use_rth)
        return df, force_what.upper()
    for what in ("TRADES","MIDPOINT"):
        df = fetch_full_daily(ib, contract, duration, what, use_rth)
        if not df.empty:
            return df, what
    return pd.DataFrame(), "TRADES"

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=913)
    ap.add_argument("--ticker", default=None, help="Single ticker (e.g., AIGA)")
    ap.add_argument("--tickers", default=None, help="List: comma or space separated (e.g., 'AIGA,BATT IUCS')")
    ap.add_argument("--duration", default=DEFAULT_DURATION)
    ap.add_argument("--use_rth", type=int, default=DEFAULT_USE_RTH)
    ap.add_argument("--what", default="", help="Force TRADES or MIDPOINT; empty = try TRADES→MIDPOINT")
    args = ap.parse_args()

    tickers = parse_tickers(args.ticker, args.tickers)
    mapping = load_mapping(MAPPING_CSV)

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    for i, t in enumerate(tickers, 1):
        try:
            qc = ib.qualifyContracts(make_contract(t, mapping))
            if not qc:
                print(f"[{i}/{len(tickers)}] {t}: [SKIP] could not qualify (check mapping/permissions).")
                continue
            con = qc[0]
            df, used = try_trades_midpoint(ib, con, args.duration, bool(args.use_rth), args.what.strip().upper() or None)
            if df.empty:
                print(f"[{i}/{len(tickers)}] {t}: no data (TRADES/MIDPOINT).")
                continue
            out_path = OUT_DIR / f"{t}_daily_raw.csv"
            df.to_csv(out_path, index=False)
            print(f"[{i}/{len(tickers)}] {t}: WROTE {len(df)} rows via {used} → {out_path}")
            ib.sleep(0.25)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")

    ib.disconnect()
    print("[DONE] s02a daily full download complete.")

if __name__ == "__main__":
    main()