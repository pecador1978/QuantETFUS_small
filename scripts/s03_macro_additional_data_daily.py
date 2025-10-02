#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s03_full_download_market_trends.py — FULL 10Y daily download to TradingView CSVs

- Connects to IBKR via ib_insync
- Downloads 10 years of DAILY bars for a fixed list of tickers
- Writes TradingView schema CSVs: time,open,high,low,close,Volume
- Output dir: /Users/Finance/QuantShared/market_trends
- File name: {TICKER}_TV_daily.csv (overwrites)

Edit the TICKERS list or the PER_TICKER mapping below if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from ib_insync import IB, Contract, util


# ---------- CONFIG ----------
import os
OUTPUT_DIR = Path(os.environ.get("SHARED_TRENDS_DIR", "/Users/Finance/QuantShared/market_trends"))

# Your requested set (deduped)
TICKERS = [
    "QQQ","SPYD","VIX","VIX9D","MOVE","VVIX","VIX3M","DBC","DIA","DXY","GDX","SLV",
    "KRE","XHB","EEM","EFA","FEZ","IEI","JNK","SPHB","LQD","IHYG","CEW"
]

# Per-ticker IB contract hints (edit if IB rejects qualification on your account)
# Keys you can set: SecType, Exchange, PrimaryExchange, Currency, whatToShow
# Most ETFs: STK / SMART / USD / TRADES
# CBOE volatility indices: IND / CBOE / USD / TRADES
# MOVE and DXY may vary by permissions; defaults below are common but adjust if needed.
PER_TICKER: Dict[str, Dict[str, Any]] = {
    # ETFs (US)
    "QQQ":  {"SecType": "STK", "Exchange": "SMART", "PrimaryExchange": "NASDAQ", "Currency": "USD", "whatToShow": "TRADES"},
    "SPYD": {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "DBC":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "DIA":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "GDX":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "SLV":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "KRE":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "XHB":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "EEM":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "EFA":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "FEZ":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "IEI":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "JNK":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "SPHB": {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "LQD":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},
    "IHYG": {"SecType": "STK", "Exchange": "LSE",   "Currency": "USD", "whatToShow": "TRADES"},  # adjust if you hold a different listing
    "CEW":  {"SecType": "STK", "Exchange": "SMART", "Currency": "USD", "whatToShow": "TRADES"},

    # Vol indices (CBOE)
    "VIX":   {"SecType": "IND", "Exchange": "CBOE", "Currency": "USD", "whatToShow": "TRADES"},
    "VIX3M": {"SecType": "IND", "Exchange": "CBOE", "Currency": "USD", "whatToShow": "TRADES"},
    "VIX9D": {"SecType": "IND", "Exchange": "CBOE", "Currency": "USD", "whatToShow": "TRADES"},
    "VVIX":  {"SecType": "IND", "Exchange": "CBOE", "Currency": "USD", "whatToShow": "TRADES"},

    # Dollar index (mapping differs across accounts; this is a common one)
    # If qualification fails, try "DX-Y.NYB" as a futures index proxy or use your preferred DXY proxy ETF (UUP).
    "DXY":   {"SecType": "IND", "Exchange": "ICEUS", "Currency": "USD", "whatToShow": "TRADES"},

    # MOVE (ICE BofA MOVE Index) — availability differs; you might need to substitute a proxy if unavailable.
    "MOVE":  {"SecType": "IND", "Exchange": "CBOE", "Currency": "USD", "whatToShow": "TRADES"},
}

# ---------- helpers ----------

def _make_contract(ticker: str) -> Contract:
    m = PER_TICKER.get(ticker.upper(), {})
    c = Contract()
    c.symbol  = ticker
    c.secType = m.get("SecType", "STK")
    c.exchange = m.get("Exchange", "SMART")
    c.currency = m.get("Currency", "USD")
    pe = m.get("PrimaryExchange", "")
    if pe:
        c.primaryExchange = pe
    return c

def _what_to_show(ticker: str) -> str:
    return PER_TICKER.get(ticker.upper(), {}).get("whatToShow", "TRADES")

def _fetch_daily_10y(ib: IB, contract: Contract, what: str, rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",              # now
        durationStr="10 Y",
        barSizeSetting="1 day",
        whatToShow=what,
        useRTH=rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    df = util.df(bars).rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for k in ["open","high","low","close","volume"]:
        if k not in df.columns:
            df[k] = pd.NA
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df[["datetime","open","high","low","close","volume"]].dropna(subset=["datetime"]).sort_values("datetime")

def _to_tradingview_csv(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    # TV daily files usually hold dates (no timezone) as YYYY-MM-DD in a 'time' column
    out["time"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"]).rename(columns={"volume": "Volume"})
    out = out[["time", "open", "high", "low", "close", "Volume"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)   # TWS=7496, Gateway=4001 (default)
    ap.add_argument("--client_id", type=int, default=913)
    ap.add_argument("--use_rth", type=int, default=1, help="1=RegularTradingHours only, 0=all sessions")
    ap.add_argument("--out_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--tickers", nargs="*", default=TICKERS, help="Override list of tickers")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tickers = [t.strip().upper() for t in args.tickers if t.strip()]

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.\n")

    for tkr in tickers:
        out_path = out_dir / f"{tkr}_TV_daily.csv"
        try:
            contr = _make_contract(tkr)
            qualified = ib.qualifyContracts(contr)
            if not qualified:
                print(f"[WARN] {tkr}: could not qualify contract with mapping {PER_TICKER.get(tkr, {})}. Skipping.")
                continue

            what = _what_to_show(tkr)
            df = _fetch_daily_10y(ib, qualified[0], what=what, rth=bool(args.use_rth))

            if df.empty:
                print(f"[WARN] {tkr}: no data returned (whatToShow={what}).")
                continue

            _to_tradingview_csv(df, out_path)
            print(f"[OK]  {tkr:<6} → {out_path}  rows={len(df)}  first={df['datetime'].min().date()}  last={df['datetime'].max().date()}")

        except Exception as e:
            print(f"[ERR] {tkr}: {e}")

        # polite pacing
        ib.sleep(0.25)

    ib.disconnect()
    print("\n[DONE] Full 10Y market_trends download complete.")

if __name__ == "__main__":
    main()