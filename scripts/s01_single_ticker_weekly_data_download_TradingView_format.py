#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s01_single_ticker_weekly_data_download.py

!!! MARKET TRENDS TICKERS ONLY !!!

- Downloads ~10 years of WEEKLY bars for one ticker (default: XLE)
- TradingView CSV schema: time,open,high,low,close,Volume
- Output: /Users/Finance/QuantShared/market_trends/{TICKER}_TV_weekly.csv
- Robust: tries TRADES first, then MIDPOINT if TRADES is empty
- London listing by default (SMART + PrimaryExchange=LSEETF). Flip Currency if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from ib_insync import IB, Contract, util

# ---------------- Config ----------------
OUTPUT_DIR = Path("/Users/Finance/QuantShared/market_trends")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default mapping (adjust Currency to "GBP" if your line is GBP)
DEFAULT_MAP: Dict[str, Any] = {
    "SecType": "STK",
    "Exchange": "SMART",
    "PrimaryExchange": "LSEETF",
    "Currency": "USD",
    "whatToShow": "TRADES",
}

# ---------------- Helpers ----------------
def make_contract(ticker: str, mp: Dict[str, Any]) -> Contract:
    c = Contract()
    c.symbol = ticker.upper()
    c.secType = mp.get("SecType", "STK")
    c.exchange = mp.get("Exchange", "SMART")
    c.currency = mp.get("Currency", "USD")
    pe = mp.get("PrimaryExchange", "")
    if pe:
        c.primaryExchange = pe
    return c

def what_list(mp: Dict[str, Any]) -> List[str]:
    first = str(mp.get("whatToShow", "TRADES")).upper()
    return [first, "MIDPOINT"] if first != "MIDPOINT" else ["MIDPOINT", "TRADES"]

def fetch_weekly_10y_with_fallback(ib: IB, contract: Contract, whats: List[str], rth: bool, duration: str) -> pd.DataFrame:
    for what in whats:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",            # now
            durationStr=duration,      # e.g., "10 Y"
            barSizeSetting="1 week",   # <-- WEEKLY
            whatToShow=what,
            useRTH=rth,
            formatDate=1,
            keepUpToDate=False,
        )
        if not bars:
            continue
        df = util.df(bars).rename(columns={"date": "datetime"})
        if df.empty:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        for k in ["open", "high", "low", "close", "volume"]:
            if k not in df.columns:
                df[k] = pd.NA
            df[k] = pd.to_numeric(df[k], errors="coerce")
        out = (
            df[["datetime", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["datetime"])
            .sort_values("datetime")
        )
        if not out.empty:
            return out
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

def to_tradingview_csv(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    out["time"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"]).rename(columns={"volume": "Volume"})
    out = out[["time", "open", "high", "low", "close", "Volume"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)         # TWS=7496, Gateway=4001
    ap.add_argument("--client_id", type=int, default=914)
    ap.add_argument("--ticker", default="XLE")
    ap.add_argument("--use_rth", type=int, default=1)         # 1=RTH, 0=all sessions
    ap.add_argument("--currency", default=DEFAULT_MAP["Currency"])  # override to GBP if needed
    ap.add_argument("--duration", default="10 Y")             # tweak if you want longer/shorter
    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    out_path = OUTPUT_DIR / f"{ticker}_TV_weekly.csv"         # <-- WEEKLY filename

    # prepare mapping (allow currency override)
    mp = dict(DEFAULT_MAP)
    mp["Currency"] = args.currency.strip().upper()

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.\n")

    try:
        contr = make_contract(ticker, mp)
        q = ib.qualifyContracts(contr)
        if not q:
            print(f"[WARN] {ticker}: could not qualify contract with mapping {mp}.")
            return
        contr = q[0]

        df = fetch_weekly_10y_with_fallback(
            ib, contr, whats=what_list(mp), rth=bool(args.use_rth), duration=args.duration
        )
        if df.empty:
            print(f"[WARN] {ticker}: no weekly data (tried {what_list(mp)}).")
            return

        # compute first/last BEFORE converting to TV schema
        first = df["datetime"].dt.strftime("%Y-%m-%d").min()
        last  = df["datetime"].dt.strftime("%Y-%m-%d").max()

        to_tradingview_csv(df, out_path)
        print(f"[OK]  {ticker:<6} → {out_path}  rows={len(df)}  first={first}  last={last}")

    except Exception as e:
        print(f"[ERR] {ticker}: {e}")
    finally:
        ib.disconnect()
        print("\n[DONE] Single-ticker weekly download.")

if __name__ == "__main__":
    main()