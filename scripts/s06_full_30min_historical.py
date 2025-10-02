#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_full_30min_historical.py — Seed 30m history ONLY for tickers missing locally (USD-only, London-first)

- Universe: ETF_list.xlsx, sheet 'signalsUSD' (auto-detects Ticker/Symbol column).
- For each ticker, check:
    /Users/Finance/QuantShared/data_raw_ETF_US/30min/{TICKER}_30min_raw.csv
  If it exists → SKIP. If missing → fetch ~10y of 30m bars and write CSV.

- Qualification order (USD only): LSEETF → LSE → SMART (with London primary when possible).
- Stream probe: TRADES/MIDPOINT/BID_ASK × RTH{1,0}, head-timestamp then tiny hist fallback.
- Paged fetch: 180-day pages, yesterday 23:59:59 UTC back ~N years (default 10y).
- Clean logging: suppress IBKR 162/200/321 chatter.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import argparse, time
import pandas as pd
from ib_insync import IB, Contract, Stock, util

# ----- Paths -----
RAW_DIR = Path("/Users/Finance/QuantShared/data_raw_ETF_US/30min")
RAW_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH  = Path("/Users/Finance/QuantShared/ETF_list.xlsx")
EXCEL_SHEET = "signalsUSD"
PREFERRED_COLS = ("Ticker", "Symbol", "ticker", "symbol")

# ----- Clean logging (mute spammy IBKR warnings) -----
SUPPRESS_CODES = {162, 200, 321}

@contextmanager
def mute_ibkr_warnings(ib: IB):
    """Suppress common HMDS/validation noise."""
    def _handler(reqId, errorCode, errorString, misc=None):
        if errorCode in SUPPRESS_CODES:
            return
        print(f"[IB {errorCode}] {errorString}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ----- Excel universe -----
def load_universe_from_excel(xlsx: Path, sheet: str) -> list[str]:
    if not xlsx.exists():
        raise SystemExit(f"[ERR] Universe file missing: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if df is None or df.empty:
        raise SystemExit(f"[ERR] Universe sheet empty: {xlsx.name}:{sheet}")
    cols_map = {c.strip().lower(): c for c in df.columns}
    chosen = next((cols_map[c.lower()] for c in PREFERRED_COLS if c.lower() in cols_map), df.columns[0])
    s = df[chosen].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t and t not in {"NAN", "NONE"}]

# ----- IB helpers -----
def try_qualify_london_usd(ib: IB, symbol: str) -> Contract | None:
    """
    Prefer LSEETF/LSE in USD; then SMART pinned to LSEETF/LSE; then secType='ETF';
    finally discover via reqContractDetails and filter to USD + (LSEETF/LSE).
    """
    attempts: list[Contract] = [
        Stock(symbol, 'LSEETF', 'USD'),
        Stock(symbol, 'LSE',    'USD'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSEETF'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSE'),
        Contract(symbol=symbol, secType='ETF', exchange='LSEETF', currency='USD'),
        Contract(symbol=symbol, secType='ETF', exchange='SMART',  currency='USD', primaryExchange='LSEETF'),
    ]
    with mute_ibkr_warnings(ib):
        for con in attempts:
            try:
                got = ib.qualifyContracts(con)
                if got:
                    return got[0]
            except Exception:
                pass
        # Discovery fallback
        try:
            for venue in ('LSEETF', 'SMART'):
                cds = ib.reqContractDetails(Stock(symbol, venue, 'USD'))
                for cd in cds:
                    c = cd.contract
                    px = (c.primaryExchange or c.exchange or '').upper()
                    if (c.currency or '').upper() == 'USD' and px in {'LSEETF', 'LSE'}:
                        return c
        except Exception:
            pass
    return None

def normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns: df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def read_existing_csv(path_csv: Path) -> bool:
    return path_csv.exists() and path_csv.is_file()

def save_csv(path_csv: Path, df: pd.DataFrame):
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    out = df.drop_duplicates(subset=["date"]).sort_values("date")
    out.to_csv(path_csv, index=False)

def probe_stream(ib: IB, con: Contract, hist_days: int = 3):
    """Head timestamp, then small historical probe → returns (what, use_rth)."""
    trials = [
        ("TRADES", True), ("TRADES", False),
        ("MIDPOINT", True), ("MIDPOINT", False),
        ("BID_ASK", True), ("BID_ASK", False),
    ]
    end_str = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
    with mute_ibkr_warnings(ib):
        # head timestamp probe
        for what, rth in trials:
            try:
                ts = ib.reqHeadTimeStamp(con, whatToShow=what, useRTH=rth, formatDate=1, endDateTime=end_str)
                if ts:
                    return what, rth
            except Exception:
                continue
        # tiny historical probe
        end_dt = datetime.now(timezone.utc)
        for what, rth in trials:
            try:
                bars = ib.reqHistoricalData(
                    con,
                    endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
                    durationStr=f"{max(1, hist_days)} D",
                    barSizeSetting="30 mins",
                    whatToShow=what,
                    useRTH=rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
                if bars:
                    df = normalize_hist_df(util.df(bars))
                    if not df.empty:
                        return what, rth
            except Exception:
                continue
    return None, None

def fetch_paged(ib: IB, con: Contract, what: str, use_rth: bool,
                years: float = 10.0, page_days: int = 180, sleep_s: float = 0.3) -> pd.DataFrame:
    """Fetch ~years ending yesterday 23:59:59 UTC, paging backward by page_days."""
    end_dt = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    start_dt = end_dt - timedelta(days=int(365.25 * years))
    rows = []
    cursor = end_dt
    with mute_ibkr_warnings(ib):
        while cursor >= start_dt:
            begin = max(start_dt, cursor - timedelta(days=page_days - 1))
            duration_days = (cursor - begin).days + 1
            try:
                bars = ib.reqHistoricalData(
                    con,
                    endDateTime=cursor.strftime("%Y%m%d %H:%M:%S"),
                    durationStr=f"{max(1, duration_days)} D",
                    barSizeSetting="30 mins",
                    whatToShow=what,
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception:
                bars = []
            if bars:
                df = normalize_hist_df(util.df(bars))
                if not df.empty:
                    mask = (df["date"] >= begin) & (df["date"] <= cursor)
                    got = df.loc[mask]
                    if not got.empty:
                        rows.append(got)
            cursor = begin - timedelta(seconds=1)
            time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)

# ----- main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=57)
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--probe_days", type=int, default=3)
    ap.add_argument("--sleep_ms", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--excel", default=str(EXCEL_PATH))
    ap.add_argument("--sheet", default=str(EXCEL_SHEET))
    args = ap.parse_args()

    tickers = load_universe_from_excel(Path(args.excel), args.sheet)
    if args.only:
        allow = {t.upper() for t in args.only}
        tickers = [t for t in tickers if t in allow]
    if args.skip:
        block = {t.upper() for t in args.skip}
        tickers = [t for t in tickers if t not in block]
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    seeded = 0
    for i, t in enumerate(tickers, 1):
        out_path = RAW_DIR / f"{t}_30min_raw.csv"
        if read_existing_csv(out_path):
            print(f"[{i}/{len(tickers)}] {t}: exists → SKIP")
            continue

        try:
            con = try_qualify_london_usd(ib, t)
            if not con:
                print(f"[{i}/{len(tickers)}] {t}: SKIP (no USD contract on LSE/LSEETF/SMART)")
                continue

            what, use_rth = probe_stream(ib, con, hist_days=args.probe_days)
            if not what:
                print(f"[{i}/{len(tickers)}] {t}: SKIP (no 30m stream found)")
                continue

            print(f"[{i}/{len(tickers)}] {t}: seeding ~{args.years}y via {what} RTH={int(use_rth)}")
            df = fetch_paged(
                ib, con, what, use_rth,
                years=args.years,
                page_days=180,
                sleep_s=max(args.sleep_ms, 0) / 1000.0
            )
            if df.empty:
                print(f"[{i}/{len(tickers)}] {t}: no rows returned; not creating file.")
                continue

            save_csv(out_path, df)
            print(f"[{i}/{len(tickers)}] {t}: wrote {len(df)} rows → {out_path}")
            seeded += 1

            ib.sleep(max(args.sleep_ms, 0) / 1000.0)

        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")

    ib.disconnect()
    print(f"[DONE] Scanned {len(tickers)} | Newly seeded {seeded} | Dir → {RAW_DIR}")

if __name__ == "__main__":
    main()