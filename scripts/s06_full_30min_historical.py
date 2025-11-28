#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_full_30min_historical.py — Seed 30-minute history ONLY for tickers missing locally.

- Universe: ETF_list.xlsx (sheet from env ETF_SHEET or paths.default_etf_sheet())
- For each ticker, if {OUTDIR}/{TICKER}_30min_raw.csv exists → SKIP, else fetch ~10y of 30m bars and write CSV.
- Venue/currency preference is driven by env (no LSE hard-coding):
    MAPPING_PRIMARY_EXCH_SEGMENTS="NASDAQ,NYE,ARCA"   # example for US
    MAPPING_PREFERRED_CCY="USD,EUR"                   # highest priority first
- If config/ticker_mapping.csv exists, it is used to build the SMART contract first
  (Exchange=SMART, Currency, PrimaryExchange). If it fails, discovery fallback runs.

Output dir defaults to P.SHARED_30M_DIR (project-aware).
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from typing import List, Optional, Tuple

import os, argparse, time, sys
import pandas as pd
from ib_insync import IB, Contract, Stock, ContractDetails, util

# ---------- project-aware imports ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P                     # SHARED_30M_DIR, CONFIG_DIR, etc.
from common.settings import ETF_LIST_PATH_STR, DEFAULT_ETF_SHEET

# ---------- env-driven venue/currency preferences ----------
ENV_SEGMENTS = os.environ.get("MAPPING_PRIMARY_EXCH_SEGMENTS", "")
EXCHANGE_SEGMENTS: tuple[str, ...] = tuple(
    s.strip().upper() for s in ENV_SEGMENTS.split(",") if s.strip()
) or ("LSEETF", "LSE", "LSEETP", "LSEIOB")  # safe default

ENV_CCY = os.environ.get("MAPPING_PREFERRED_CCY", "")
PREFERRED_CCY: tuple[str, ...] = tuple(
    c.strip().upper() for c in ENV_CCY.split(",") if c.strip()
) or ("USD", "GBP")

# ---------- defaults (paths) ----------
RAW_DIR_DEFAULT = P.SHARED_30M_DIR
MAPPING_CSV     = P.CONFIG_DIR / "ticker_mapping.csv"   # typically semicolon CSV
ETF_XLSX        = Path(ETF_LIST_PATH_STR).resolve()
ETF_SHEET       = DEFAULT_ETF_SHEET
PREFERRED_COLS  = ("Ticker", "Symbol", "ticker", "symbol")

# ---------- logging / noise control ----------
SUPPRESS_CODES = {162, 200, 321}

@contextmanager
def mute_ibkr_warnings(ib: IB):
    def _handler(reqId, code, msg, misc=None):
        if code in SUPPRESS_CODES:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ---------- universe ----------
def load_universe_from_excel(xlsx: Path, sheet: str) -> List[str]:
    if not xlsx.exists():
        raise SystemExit(f"[ERR] Universe file missing: {xlsx}")
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if df is None or df.empty:
        raise SystemExit(f"[ERR] Universe sheet empty: {xlsx.name}:{sheet}")
    cols_map = {c.strip().lower(): c for c in df.columns}
    chosen = next((cols_map[c.lower()] for c in PREFERRED_COLS if c.lower() in cols_map), df.columns[0])
    s = df[chosen].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t and t not in {"NAN", "NONE"}]

# ---------- CSV utils ----------
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

# ---------- contract resolution ----------
def _contract_from_mapping(ticker: str) -> Optional[Contract]:
    """Build a SMART contract from mapping CSV row if available."""
    if not MAPPING_CSV.exists():
        return None
    # robust read: accept ; or , and handle BOM
    try:
        df = pd.read_csv(MAPPING_CSV, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(MAPPING_CSV, sep=";")
    cols = {c.lower(): c for c in df.columns}
    req = {"ticker","currency","primaryexchange"}
    if not req.issubset(set(cols.keys())):
        return None
    row = df[df[cols["ticker"]].astype(str).str.strip().str.upper() == ticker]
    if row.empty:
        return None
    r = row.iloc[0]
    currency = str(r[cols["currency"]]).upper() if pd.notna(r[cols["currency"]]) else (PREFERRED_CCY[0] if PREFERRED_CCY else "USD")
    primary  = str(r[cols["primaryexchange"]]).upper() if pd.notna(r[cols["primaryexchange"]]) else (EXCHANGE_SEGMENTS[0] if EXCHANGE_SEGMENTS else "")
    return Stock(symbol=ticker, exchange="SMART", currency=currency, primaryExchange=primary)

def _is_primary_px(cd: ContractDetails) -> bool:
    return (cd.contract.primaryExchange or "").upper() in EXCHANGE_SEGMENTS

def _discover_contract(ib: IB, ticker: str) -> Optional[Contract]:
    """Venue-agnostic discovery using env-preferred segments/currencies."""
    # Try SMART + (segment, currency) combos first
    for cur in PREFERRED_CCY:
        for seg in EXCHANGE_SEGMENTS:
            probe = Stock(ticker, 'SMART', cur, primaryExchange=seg)
            try:
                cds = ib.reqContractDetails(probe)
                for cd in cds:
                    c = cd.contract
                    if (c.secType or "").upper() in {"STK","ETF"} and _is_primary_px(cd):
                        return c
            except Exception:
                continue
    # Fallback: plain SMART with preferred ccy
    for cur in PREFERRED_CCY:
        try:
            got = ib.qualifyContracts(Stock(ticker, 'SMART', cur))
            if got:
                return got[0]
        except Exception:
            continue
    return None

def resolve_contract(ib: IB, ticker: str) -> Optional[Contract]:
    """Prefer mapping; else discover."""
    con = _contract_from_mapping(ticker)
    if con:
        try:
            got = ib.qualifyContracts(con)
            if got:
                return got[0]
        except Exception:
            pass
    return _discover_contract(ib, ticker)

# ---------- probing + fetch ----------
def probe_stream(ib: IB, con: Contract, hist_days: int = 3) -> Tuple[Optional[str], Optional[bool]]:
    trials = [
        ("TRADES", True), ("TRADES", False),
        ("MIDPOINT", True), ("MIDPOINT", False),
        ("BID_ASK", True), ("BID_ASK", False),
    ]
    end_dt = datetime.now(timezone.utc)
    end_str = end_dt.strftime("%Y%m%d %H:%M:%S")
    with mute_ibkr_warnings(ib):
        # Head timestamp probe
        for what, rth in trials:
            try:
                ts = ib.reqHeadTimeStamp(con, whatToShow=what, useRTH=rth, formatDate=1, endDateTime=end_str)
                if ts:
                    return what, rth
            except Exception:
                continue
        # Tiny historical probe
        for what, rth in trials:
            try:
                bars = ib.reqHistoricalData(
                    con,
                    endDateTime=end_str,
                    durationStr=f"{max(hist_days,1)} D",
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
                    durationStr=f"{duration_days} D",
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
                    rows.append(df)
            cursor = begin - timedelta(seconds=1)
            time.sleep(max(sleep_s, 0.0))

    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)

# ---------- main ----------
def main():
    global EXCHANGE_SEGMENTS, PREFERRED_CCY

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
    ap.add_argument("--excel", default=str(ETF_XLSX))
    ap.add_argument("--sheet", default=str(ETF_SHEET))
    ap.add_argument("--outdir", default=str(RAW_DIR_DEFAULT))
    ap.add_argument("--ticker_col", default="Ticker")
    ap.add_argument("--segments", default=",".join(EXCHANGE_SEGMENTS),
                    help="PrimaryExchange segments to prefer, comma-separated (env MAPPING_PRIMARY_EXCH_SEGMENTS)")
    ap.add_argument("--ccy", default=",".join(PREFERRED_CCY),
                    help="Preferred currencies, comma-separated (env MAPPING_PREFERRED_CCY)")
    args = ap.parse_args()

    # allow ad-hoc overrides
    if args.segments:
        EXCHANGE_SEGMENTS = tuple(s.strip().upper() for s in args.segments.split(",") if s.strip())
    if args.ccy:
        PREFERRED_CCY = tuple(c.strip().upper() for c in args.ccy.split(",") if c.strip())

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

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
        out_path = out_root / f"{t}_30min_raw.csv"
        if read_existing_csv(out_path):
            print(f"[{i}/{len(tickers)}] {t}: exists → SKIP")
            continue

        try:
            con = resolve_contract(ib, t)
            if not con:
                print(f"[{i}/{len(tickers)}] {t}: SKIP (no qualified contract)")
                continue

            what, use_rth = probe_stream(ib, con, hist_days=args.probe_days)
            if not what:
                print(f"[{i}/{len(tickers)}] {t}: SKIP (no 30m stream found)")
                continue

            print(f"[{i}/{len(tickers)}] {t}: seeding ~{args.years}y via {what} RTH={int(bool(use_rth))}")
            df = fetch_paged(
                ib, con, what, bool(use_rth),
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
    print(f"[DONE] Frame=30m | Scanned {len(tickers)} | Newly seeded {seeded} | Dir → {out_root}")

if __name__ == "__main__":
    main()