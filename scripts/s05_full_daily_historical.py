#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s05_full_daily_historical.py — build full DAILY history for the project universe (NY)

Key points
----------
- Prefers IBKR ADJUSTED_LAST (corporate-action adjusted) for entire span.
- If ADJUSTED_LAST unavailable, falls back: TRADES → MIDPOINT → BID_ASK.
- Writes CSV: <outdir>/{TICKER}_daily_raw.csv with columns:
    date, open, high, low, close, volume, average, barCount, __what
- Honors ticker mapping in config/ticker_mapping.csv (SMART + primaryExchange + currency).
- Venues/currencies are driven by env; not hard-coded to LSE.

Env knobs (optional)
--------------------
- MAPPING_PRIMARY_EXCH_SEGMENTS="NASDAQ,NYE,ARCA"
- MAPPING_PREFERRED_CCY="USD,EUR"
- ETF_TICKER_COL="Ticker"

Usage examples
--------------
python scripts/s05_full_daily_historical.py --years 20 --force
python scripts/s05_full_daily_historical.py --only SPY,QQQ --force
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os, sys, argparse
from typing import List, Optional

import pandas as pd
from ib_insync import IB, Contract, Stock, ContractDetails, util
from contextlib import contextmanager

# ---------- project-aware imports ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # SHARED_DAILY_DIR, CONFIG_DIR, etc.
from common.settings import ETF_LIST_PATH_STR, DEFAULT_ETF_SHEET

# ---------- env-driven venue/currency preferences ----------
ENV_SEGMENTS = os.environ.get("MAPPING_PRIMARY_EXCH_SEGMENTS", "")
EXCHANGE_SEGMENTS: tuple[str, ...] = tuple(
    s.strip().upper() for s in ENV_SEGMENTS.split(",") if s.strip()
) or ("NASDAQ","NYE","ARCA")

ENV_CCY = os.environ.get("MAPPING_PREFERRED_CCY", "")
PREFERRED_CCY: tuple[str, ...] = tuple(
    c.strip().upper() for c in ENV_CCY.split(",") if c.strip()
) or ("USD","EUR")

# ---------- defaults (paths) ----------
RAW_DIR_DEFAULT = P.SHARED_DAILY_DIR
MAPPING_CSV     = P.CONFIG_DIR / "ticker_mapping.csv"   # ; or , is fine
ETF_XLSX        = Path(ETF_LIST_PATH_STR).resolve()
ETF_SHEET       = DEFAULT_ETF_SHEET
ETF_COL         = os.environ.get("ETF_TICKER_COL", "Ticker")

# ---------- IB noise filter ----------
@contextmanager
def mute_162(ib: IB):
    def _handler(reqId, code, msg, contract):
        # 162 = "Historical Market Data Service error message" (noisy)
        if code == 162:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ---------- helpers ----------
def load_universe(xlsx: Path, sheet: str, col: str) -> List[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col.lower()
    if key not in colmap:
        raise SystemExit(f"[ERR] column '{col}' not found in {xlsx}:{sheet}")
    s = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t and t not in {"NAN","NONE"}]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    if "date" not in df.columns:
        # ib_insync util.df(...) for historical bars always has 'date'
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in out.columns: out[c] = pd.NA
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out

def _end_str_utc(ts: datetime) -> str:
    return ts.strftime("%Y%m%d %H:%M:%S") + " UTC"

# ---------- contract resolution ----------
def _contract_from_mapping(ticker: str) -> Optional[Stock]:
    if not MAPPING_CSV.exists():
        return None
    try:
        df = pd.read_csv(MAPPING_CSV, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(MAPPING_CSV, sep=";")
    cols = {c.lower(): c for c in df.columns}
    if not {"ticker","currency","primaryexchange"}.issubset(cols.keys()):
        return None
    row = df[df[cols["ticker"]].astype(str).str.strip().str.upper() == ticker]
    if row.empty:
        return None
    r = row.iloc[0]
    currency = str(r[cols["currency"]]).upper() if pd.notna(r[cols["currency"]]) else PREFERRED_CCY[0]
    primary  = str(r[cols["primaryexchange"]]).upper() if pd.notna(r[cols["primaryexchange"]]) else ""
    return Stock(
        symbol=ticker,
        exchange="SMART",
        currency=currency or PREFERRED_CCY[0],
        primaryExchange=primary or (EXCHANGE_SEGMENTS[0] if EXCHANGE_SEGMENTS else "")
    )

def _is_primary_px(cd: ContractDetails) -> bool:
    return (cd.contract.primaryExchange or "").upper() in EXCHANGE_SEGMENTS

def _discover_contract(ib: IB, ticker: str) -> Optional[Stock]:
    # Try SMART + (segment, currency) combos
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
    con = _contract_from_mapping(ticker)
    if con:
        try:
            got = ib.qualifyContracts(con)
            if got:
                return got[0]
        except Exception:
            pass
    return _discover_contract(ib, ticker)

# ---------- data pulls ----------
def _req_hist(ib: IB, con: Contract, *, end: str, duration: str, what: str) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        con,
        endDateTime=end,
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what,
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )
    return normalize(util.df(bars) if bars else pd.DataFrame())

def fetch_adjusted_full(ib: IB, con: Contract, years: int) -> pd.DataFrame:
    """
    Best-effort pull of ADJUSTED_LAST for the whole span.
    IB limitation: ADJUSTED_LAST often rejects explicit endDateTime with code 321.
    Workaround: call with endDateTime="" (i.e., 'now') and a long duration.
    """
    duration = f"{int(years)} Y"
    with mute_162(ib):
        try:
            # Primary attempt: empty endDateTime (works for adjusted)
            df = _req_hist(ib, con, end="", duration=duration, what="ADJUSTED_LAST")
            if not df.empty:
                df["__what"] = "ADJUSTED_LAST"
                return df
        except Exception as e:
            print(f"[INFO] ADJUSTED_LAST (end='') failed: {e}")

        # Fallback attempt: try with explicit end for shorter spans (some servers allow)
        try:
            df = _req_hist(ib, con, end=_end_str_utc(datetime.now(timezone.utc)), duration=duration, what="ADJUSTED_LAST")
            if not df.empty:
                df["__what"] = "ADJUSTED_LAST"
                return df
        except Exception as e:
            print(f"[INFO] ADJUSTED_LAST (explicit end) failed: {e}")

    return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount","__what"])

def fetch_fallback_chunked(ib: IB, con: Contract, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Chunked pull for non-adjusted (TRADES → MIDPOINT → BID_ASK) over the full period.
    """
    rows = []
    cursor = end
    start_floor = start
    with mute_162(ib):
        while cursor >= start_floor:
            begin = max(start_floor, cursor - timedelta(days=360))  # ~1y per request
            duration_days = (cursor - begin).days + 1
            got = None
            for what in ("TRADES","MIDPOINT","BID_ASK"):
                try:
                    df = _req_hist(
                        ib, con,
                        end=_end_str_utc(cursor),
                        duration=f"{duration_days} D",
                        what=what
                    )
                    if not df.empty:
                        df["__what"] = what
                        got = df
                        break
                except Exception:
                    continue
            if got is not None and not got.empty:
                rows.append(got)
            cursor = begin - timedelta(days=1)

    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount","__what"])

    out = (pd.concat(rows, ignore_index=True)
             .sort_values("date")
             .drop_duplicates(subset=["date"], keep="last")
             .reset_index(drop=True))
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=88)
    ap.add_argument("--years", type=float, default=20.0, help="Years of daily history to fetch (max ask).")
    ap.add_argument("--excel", default=str(ETF_XLSX))
    ap.add_argument("--sheet", default=str(ETF_SHEET))
    ap.add_argument("--ticker_col", default=str(ETF_COL))
    ap.add_argument("--outdir", default=str(RAW_DIR_DEFAULT))
    ap.add_argument("--only", default="", help="Comma-separated tickers (skip universe).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing CSVs instead of skipping.")
    args = ap.parse_args()

    out_root = Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.only.strip():
        tickers = [t.strip().upper() for t in args.only.split(",") if t.strip()]
    else:
        tickers = load_universe(Path(args.excel), args.sheet, args.ticker_col)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    END   = datetime.now(timezone.utc)
    START = END - timedelta(days=int(round(365.25 * args.years)))

    for i, t in enumerate(tickers, 1):
        out_path = out_root / f"{t}_daily_raw.csv"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(tickers)}] {t}: SKIP (exists). Use --force to overwrite.")
            continue

        try:
            con = resolve_contract(ib, t)
            if not con:
                print(f"[{i}/{len(tickers)}] {t}: no qualified contract")
                continue

            # 1) Try full-span ADJUSTED_LAST
            adj = fetch_adjusted_full(ib, con, int(args.years))

            if not adj.empty:
                df = adj
            else:
                # 2) Fallback to non-adjusted (chunked)
                print(f"[{i}/{len(tickers)}] {t}: ⚠ no ADJUSTED_LAST; falling back to TRADES/MIDPOINT/BID_ASK")
                df = fetch_fallback_chunked(ib, con, START, END)

            if df.empty:
                print(f"[{i}/{len(tickers)}] {t}: no rows")
                continue

            # Ensure __what is present (constant if adjusted)
            if "__what" not in df.columns:
                df["__what"] = "UNKNOWN"

            df.to_csv(out_path, index=False)
            last_dt = df["date"].max()
            w = df["__what"].iloc[-1]
            print(f"[{i}/{len(tickers)}] {t}: wrote {len(df):,} rows → {out_path} | what={w} last={last_dt.date() if pd.notna(last_dt) else 'NA'}")
            ib.sleep(0.3)

        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR {e}")

    ib.disconnect()
    print("[DONE] Daily build complete.")

if __name__ == "__main__":
    main()