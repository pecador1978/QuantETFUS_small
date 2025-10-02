#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s00_probe_30m_range.py — Single-ticker 30m download over a fixed date window (paged)

Usage:
  /Users/Finance/QuantETFUS/venv/bin/python /Users/Finance/QuantETFUS/scripts/s00_probe_30m_range.py \
    --ticker IOGP --start 2018-01-01 --end 2024-12-31
Optional:
  --what TRADES|MIDPOINT|BID_ASK  (force stream)
  --rth  1|0                       (force RTH on/off)
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import sys, os
import pandas as pd
from contextlib import contextmanager
from ib_insync import IB, Contract, util

# --- project bootstrap ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P, default_etf_sheet

TARGET_BUCKET = "targeted_ETFs_US"
OUT_DIR = P.DATA_RAW / TARGET_BUCKET / "30min_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_CSV = str(P.CONFIG_DIR / "ticker_mapping.csv")

@contextmanager
def _mute_162(ib: IB):
    """Silence HMDS code 162 during probe & fetch."""
    def _handler(reqId, code, msg, contract):
        if code == 162:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

def load_mapping(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        return {}
    df = pd.read_csv(path_csv, sep=";")
    mp = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType":         (str(r.get("SecType","STK")).strip() or "STK").upper(),
            "Exchange":        str(r.get("Exchange","SMART")).strip() or "SMART",
            "Currency":        str(r.get("Currency","USD")).strip().upper() or "USD",
            "PrimaryExchange": str(r.get("PrimaryExchange","")).strip()
        }
    return mp

def make_contract(ticker: str, mapping: dict) -> Contract:
    m = mapping.get(ticker.upper(), {})
    c = Contract()
    c.symbol   = ticker.upper()
    c.secType  = m.get("SecType","STK")
    c.exchange = m.get("Exchange","SMART")   # SMART ok; IB will qualify (e.g., LSEETF)
    c.currency = m.get("Currency","USD")     # switch to GBP in mapping if needed
    if m.get("PrimaryExchange"):
        c.primaryExchange = m["PrimaryExchange"]
    return c

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date","open","high","low","close","volume","average","barCount"]].dropna(subset=["date"])

def _probe_stream(ib: IB, con: Contract):
    """Try TRADES→MIDPOINT→BID_ASK with RTH=1→0 over a tiny 3-day window; return first (what,rth) that yields rows."""
    trials = [("TRADES", True), ("TRADES", False),
              ("MIDPOINT", True), ("MIDPOINT", False),
              ("BID_ASK", True), ("BID_ASK", False)]
    end_dt = datetime.now(timezone.utc).replace(microsecond=0)
    start_dt = end_dt - timedelta(days=2)
    with _mute_162(ib):
        for what, rth in trials:
            df = _fetch_paged(ib, con, start_dt, end_dt, what, rth, page_days=3)
            if not df.empty:
                return what, rth
    return None, None

def _fetch_paged(ib: IB, con: Contract, start_dt: datetime, end_dt: datetime,
                 what: str, use_rth: bool, page_days: int = 60) -> pd.DataFrame:
    """Fetch [start,end] in pages (default 60D) and stitch."""
    rows = []
    cursor = end_dt.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    start_floor = start_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    with _mute_162(ib):
        while cursor >= start_floor:
            begin = max(start_floor, cursor - timedelta(days=page_days - 1))
            duration_days = (cursor - begin).days + 1
            end_str = cursor.strftime("%Y%m%d %H:%M:%S")
            duration_str = f"{max(1, duration_days)} D"
            try:
                bars = ib.reqHistoricalData(
                    con,
                    endDateTime=end_str,
                    durationStr=duration_str,
                    barSizeSetting="30 mins",
                    whatToShow=what,
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception:
                bars = []
            if bars:
                df = _normalize(util.df(bars))
                if not df.empty:
                    mask = (df["date"] >= begin) & (df["date"] <= cursor)
                    rows.append(df.loc[mask])
            cursor = begin - timedelta(seconds=1)
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=77)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--what", default="", help="Force: TRADES|MIDPOINT|BID_ASK")
    ap.add_argument("--rth", type=int, default=-1, help="Force: 1 (RTH) or 0 (all). -1 = auto")
    args = ap.parse_args()

    # Parse dates (UTC)
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_dt < start_dt:
        raise SystemExit("end < start")

    mapping = load_mapping(MAPPING_CSV)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)

    con = make_contract(args.ticker, mapping)
    q = ib.qualifyContracts(con)
    if not q:
        print(f"[SKIP] could not qualify contract for {args.ticker}")
        ib.disconnect(); return
    con = q[0]

    # Choose stream
    if args.what and args.rth in (0,1):
        what, use_rth = args.what.upper(), bool(args.rth)
        print(f"[FORCED] using {what} RTH={int(use_rth)}")
    else:
        what, use_rth = _probe_stream(ib, con)
        if not what:
            print(f"[{args.ticker}] NO 30m stream available (TRADES/MIDPOINT/BID_ASK).")
            ib.disconnect(); return
        print(f"[PROBE OK] {what} RTH={int(use_rth)}")

    # Fetch full range (paged) and save
    df = _fetch_paged(ib, con, start_dt, end_dt, what, use_rth, page_days=60)
    if df.empty:
        print(f"[{args.ticker}] No rows returned for {args.start}..{args.end} ({what} RTH={int(use_rth)})")
        ib.disconnect(); return

    out_name = f"{args.ticker.upper()}_{args.start}_{args.end}_30m.csv"
    out_path = OUT_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"[{args.ticker}] OK: {len(df)} rows via {what} RTH={int(use_rth)} → {out_path}")

    ib.disconnect()

if __name__ == "__main__":
    main()