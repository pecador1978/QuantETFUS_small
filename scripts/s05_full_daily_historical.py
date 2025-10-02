#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s05_full_daily_historical.py — build full daily history for US_small universe

- Reads tickers from /Users/Finance/QuantShared/ETF_list.xlsx → sheet "US_small"
- Pulls DAILY bars from IBKR (TRADES → MIDPOINT → BID_ASK fallback)
- Saves into: /Users/Finance/QuantShared/data_raw_ETF_US/daily/{TICKER}_daily_raw.csv
- Skips tickers that already exist in RAW_DIR (so it won’t re-download).
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys, os
import pandas as pd
from ib_insync import IB, Contract, Stock, util
from contextlib import contextmanager

# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ETF_XLSX = Path("/Users/Finance/QuantShared/ETF_list.xlsx")
ETF_SHEET = "signalsUSD"
ETF_COL = "Ticker"

# <<< changed destination >>>
RAW_DIR = Path("/Users/Finance/QuantShared/data_raw_ETF_US/daily")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------- mute 162 ----------
@contextmanager
def mute_162(ib: IB):
    def _handler(reqId, code, msg, contract):
        if code == 162:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ---------- helpers ----------
def load_universe(xlsx: Path, sheet: str, col: str) -> list[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col.lower()
    if key not in colmap:
        raise SystemExit(f"[ERR] column '{col}' not found in {xlsx}:{sheet}")
    s = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in df.columns: df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def _end_str_utc(ts: datetime) -> str:
    return ts.strftime("%Y%m%d %H:%M:%S") + " UTC"

def try_contract(ib: IB, symbol: str) -> Contract | None:
    """
    Prefer LSEETF/LSE in USD; then SMART pinned to the London primary; then secType='ETF';
    finally discover via reqContractDetails and pick a USD listing on LSEETF/LSE.
    """
    attempts: list[Contract] = [
        Stock(symbol, 'LSEETF', 'USD'),
        Stock(symbol, 'LSE',    'USD'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSEETF'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSE'),
        Contract(symbol=symbol, secType='ETF', exchange='LSEETF', currency='USD'),
        Contract(symbol=symbol, secType='ETF', exchange='SMART',  currency='USD', primaryExchange='LSEETF'),
    ]

    for con in attempts:
        try:
            got = ib.qualifyContracts(con)
            if got:
                return got[0]
        except Exception:
            pass

    # Discovery fallback
    try:
        # Ask LSEETF first
        cds = ib.reqContractDetails(Stock(symbol, 'LSEETF', 'USD'))
        for cd in cds:
            c = cd.contract
            px = (c.primaryExchange or c.exchange or '').upper()
            if (c.currency or '').upper() == 'USD' and px in {'LSEETF', 'LSE'}:
                return c

        # Then SMART and filter results
        cds = ib.reqContractDetails(Stock(symbol, 'SMART', 'USD'))
        for cd in cds:
            c = cd.contract
            px = (c.primaryExchange or c.exchange or '').upper()
            if (c.currency or '').upper() == 'USD' and px in {'LSEETF', 'LSE'}:
                return c
    except Exception:
        pass

    return None

def fetch_daily(ib: IB, con: Contract, start: datetime, end: datetime) -> pd.DataFrame:
    rows = []
    cursor = end
    start_floor = start
    with mute_162(ib):
        while cursor >= start_floor:
            begin = max(start_floor, cursor - timedelta(days=360))  # safe ~1y chunks
            duration_days = (cursor - begin).days + 1
            got = None
            for what in ("TRADES","MIDPOINT","BID_ASK"):
                try:
                    bars = ib.reqHistoricalData(
                        con,
                        endDateTime=_end_str_utc(cursor),
                        durationStr=f"{duration_days} D",
                        barSizeSetting="1 day",
                        whatToShow=what,
                        useRTH=True,
                        formatDate=1,
                        keepUpToDate=False,
                    )
                    if bars:
                        df = normalize(util.df(bars))
                        if not df.empty:
                            got = df
                            break
                except Exception:
                    continue
            if got is not None and not got.empty:
                rows.append(got)
            cursor = begin - timedelta(days=1)
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start) & (out["date"] <= end)].reset_index(drop=True)

# ---------- main ----------
def main():
    tickers = load_universe(ETF_XLSX, ETF_SHEET, ETF_COL)
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=88, readonly=True)
    print("[IB] Connected.")
    START = datetime(2010,1,1, tzinfo=timezone.utc)
    END   = datetime.now(timezone.utc)

    for i, t in enumerate(tickers, 1):
        out_path = RAW_DIR / f"{t}_daily_raw.csv"
        if out_path.exists():
            print(f"[{i}/{len(tickers)}] {t}: SKIP (already exists)")
            continue
        try:
            con = try_contract(ib, t)
            if not con:
                print(f"[{i}/{len(tickers)}] {t}: no qualified contract")
                continue
            df = fetch_daily(ib, con, START, END)
            if df.empty:
                print(f"[{i}/{len(tickers)}] {t}: no rows")
                continue
            df.to_csv(out_path, index=False)
            print(f"[{i}/{len(tickers)}] {t}: wrote {len(df)} rows → {out_path}")
            ib.sleep(0.3)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")

    ib.disconnect()
    print("[DONE] Daily build complete.")

if __name__ == "__main__":
    main()