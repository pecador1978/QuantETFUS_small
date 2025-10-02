#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_full_30min_backfill_initial_2y.py (v7-robust)
- London venues supported. No US remapping.
- Robust probe per ticker:
    1) HeadTimeStamp: TRADES/MIDPOINT/BID_ASK × RTH{1,0}
    2) If head is empty -> tiny historical probe (3 D) for same sequence
  First combo that returns data is used for the whole operation.
- Two modes:
    A) Default: prepend ~N years before earliest existing rows (safe 180D pages)
    B) If --start/--end supplied: fetch that fixed window and APPEND to the CSV (safe 60D pages)
- Dedup + sort; 162 messages muted during probes so it never gets “stuck”.
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import sys
import os
import argparse
import pandas as pd
from contextlib import contextmanager
from ib_insync import IB, Contract, util

# ---------- bootstrap ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- shared paths/settings ----------
from common.paths import P  # keep using project config paths for lists/mapping

# write 30m bars to the shared raw store (US). For EU, set env SHARED_RAW_BASE to .../data_raw_ETF_EU
SHARED_BASE = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
DIR_30M = SHARED_BASE / "30min"
DIR_30M.mkdir(parents=True, exist_ok=True)

# inputs (unchanged location inside the project repo)
EXCEL_LIST       = str(P.ETF_LIST)
EXCEL_SHEET      = ("signalsUSD" if PROJECT_ROOT.name.endswith(("US_small", "US")) else "signals")
EXCEL_TICKER_COL = "Ticker"
# ticker mapping
MAPPING_CSV = Path("/Users/Finance/QuantShared/ticker_mapping.csv")

# ---------- helpers ----------
@contextmanager
def mute_162(ib: IB):
    """Mute HMDS 162 for the scope (handles tuple or object signatures)."""
    def _handler(*args, **kwargs):
        code = None; msg = None
        if len(args) >= 2 and isinstance(args[1], int):
            code = args[1]; msg = args[2] if len(args) >= 3 else ""
        else:
            err = args[0] if args else None
            code = getattr(err, "errorCode", None)
            msg  = getattr(err, "errorMsg", None)
        if code == 162:
            return
        if code is not None:
            print(f"[IB {code}] {msg}")
        else:
            print("[IB] Error:", args, kwargs)
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

def load_universe(xlsx: str, sheet: str, col: str):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col.lower()
    if key not in colmap:
        raise SystemExit(f"[ERR] Column '{col}' not found in {xlsx}:{sheet}")
    s = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in s.unique().tolist() if t]

def load_mapping(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        return {}
    df = pd.read_csv(path_csv, sep=";")
    need = {"Ticker","SecType","Exchange","Currency","PrimaryExchange"}
    if not set(need).issubset(df.columns):
        raise SystemExit(f"[ERR] {path_csv} must contain columns: {need}")
    mp = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType":         (str(r.get("SecType","STK")).strip() or "STK").upper(),
            "Exchange":        str(r.get("Exchange","SMART")).strip() or "SMART",
            "Currency":        str(r.get("Currency","USD")).strip().upper() or "USD",  # set GBP here if needed
            "PrimaryExchange": str(r.get("PrimaryExchange","")).strip()
        }
    return mp

def make_contract(ticker: str, mapping: dict) -> Contract:
    m = mapping.get(ticker, {})
    c = Contract()
    c.symbol          = ticker
    c.secType         = m.get("SecType", "STK")
    c.exchange        = m.get("Exchange", "SMART")
    c.currency        = m.get("Currency", "USD")
    if m.get("PrimaryExchange"):
        c.primaryExchange = m["PrimaryExchange"]  # e.g., LSEETF
    return c

def normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns: df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date","open","high","low","close","volume","average","barCount"]].dropna(subset=["date"])

def fetch_chunk(ib: IB, con: Contract, end_dt_utc: datetime, duration: str, what: str, use_rth: bool) -> pd.DataFrame:
    end_str = end_dt_utc.strftime("%Y%m%d %H:%M:%S")  # UTC-ish with formatDate=1
    bars = ib.reqHistoricalData(
        con,
        endDateTime=end_str,
        durationStr=duration,
        barSizeSetting="30 mins",
        whatToShow=what,           # TRADES | MIDPOINT | BID_ASK
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()
    return normalize_hist_df(util.df(bars))

def read_existing_30m(path_csv: Path) -> pd.DataFrame:
    if not path_csv.exists():
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df = pd.read_csv(path_csv)
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime":"date"})
        else:
            return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in df.columns: df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def save_merge(path_csv: Path, existing: pd.DataFrame, newdf: pd.DataFrame) -> int:
    if existing.empty:
        merged = newdf.copy()
    else:
        merged = pd.concat([existing, newdf], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path_csv, index=False)
    return len(merged) - (0 if existing.empty else len(existing))

# ---------- probe logic ----------
def probe_stream(ib: IB, con: Contract, hist_days: int = 3):
    """
    1) HeadTimeStamp for TRADES/MIDPOINT/BID_ASK × RTH{1,0}
    2) If all empty -> 3-day historical probe for same sequence
    Return (what, use_rth, origin) or None if nothing available.
    """
    trials = [
        ("TRADES", True), ("TRADES", False),
        ("MIDPOINT", True), ("MIDPOINT", False),
        ("BID_ASK", True), ("BID_ASK", False),
    ]
    # 1) head timestamps
    with mute_162(ib):
        end_str = datetime.now(timezone.utc).strftime("%Y%m%d %H:%M:%S")
        for what, rth in trials:
            try:
                ts = ib.reqHeadTimeStamp(con, whatToShow=what, useRTH=rth, formatDate=1, endDateTime=end_str)
                if ts:
                    return (what, rth, "head")
            except Exception:
                continue
    # 2) tiny historical window
    with mute_162(ib):
        end_dt = datetime.now(timezone.utc)
        for what, rth in trials:
            try:
                df = fetch_chunk(ib, con, end_dt, duration=f"{hist_days} D", what=what, use_rth=rth)
                if not df.empty:
                    return (what, rth, "hist")
            except Exception:
                continue
    return None

# ---------- range fetch (append mode) ----------
def fetch_window_paged(ib: IB, con: Contract, start_dt: datetime, end_dt: datetime,
                       what: str, use_rth: bool, page_days: int = 60) -> pd.DataFrame:
    rows = []
    cursor = end_dt.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=timezone.utc)
    start_floor = start_dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    while cursor >= start_floor:
        begin = max(start_floor, cursor - timedelta(days=page_days - 1))
        duration_days = (cursor - begin).days + 1
        duration_str = f"{max(1, duration_days)} D"
        end_str = cursor.strftime("%Y%m%d %H:%M:%S")
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
        if bars:
            df = normalize_hist_df(util.df(bars))
            if not df.empty:
                mask = (df["date"] >= begin) & (df["date"] <= cursor)
                rows.append(df.loc[mask])
        cursor = begin - timedelta(seconds=1)
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)

# ---------- prepend mode (years back) ----------
def prepend_backfill_30m(ib: IB, con: Contract, path_csv: Path,
                         years_back: float, duration_str: str,
                         what_primary: str, what_fallback: str,
                         use_rth: bool, sleep_s: float, max_chunks: int = 40) -> int:
    existing = read_existing_30m(path_csv)
    if existing.empty:
        end_dt = datetime.now(timezone.utc)
        target_start = end_dt - timedelta(days=int(365.25 * years_back))
    else:
        earliest = existing["date"].min()
        end_dt = earliest - timedelta(seconds=1)
        target_start = earliest - timedelta(days=int(365.25 * years_back))

    all_new, chunks, last_batch = [], 0, None
    while end_dt > target_start and chunks < max_chunks:
        try:
            df = fetch_chunk(ib, con, end_dt, duration_str, what_primary, use_rth)
            if df.empty and what_fallback:
                df = fetch_chunk(ib, con, end_dt, duration_str, what_fallback, use_rth)
        except Exception:
            end_dt -= timedelta(days=30); chunks += 1; time.sleep(sleep_s); continue

        if df.empty:
            end_dt -= timedelta(days=30); chunks += 1; time.sleep(sleep_s); continue

        df = df[df["date"] <= end_dt].copy()
        if df.empty:
            end_dt -= timedelta(days=30); chunks += 1; time.sleep(sleep_s); continue

        all_new.append(df)
        end_dt = df["date"].min() - timedelta(seconds=1)
        chunks += 1; time.sleep(sleep_s)

        if last_batch is not None and len(df) == last_batch and len(df) < 3:
            break
        last_batch = len(df)

    if not all_new:
        return 0

    older = pd.concat(all_new, ignore_index=True).dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"])
    added = save_merge(path_csv, existing, older)
    return added

# ---------- main ----------
def main():
    print(f"[RUNNING FILE] {Path(__file__).resolve()} :: v7-robust")

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=57)
    ap.add_argument("--years", type=float, default=2.0, help="Years to backfill before earliest existing bar.")
    ap.add_argument("--duration", default="180 D", help="Chunk duration for prepend mode.")
    ap.add_argument("--probe_days", type=int, default=3, help="Tiny historical probe window when head-timestamp empty.")
    ap.add_argument("--sleep_ms", type=int, default=300)
    ap.add_argument("--limit_tickers", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    # NEW: fixed range append mode
    ap.add_argument("--start", default="", help="YYYY-MM-DD (optional: enables fixed-range APPEND mode)")
    ap.add_argument("--end",   default="", help="YYYY-MM-DD (optional: enables fixed-range APPEND mode)")
    args = ap.parse_args()

    tickers = load_universe(EXCEL_LIST, EXCEL_SHEET, EXCEL_TICKER_COL)
    if args.only:
        onlyset = {t.upper() for t in args.only}
        tickers = [t for t in tickers if t in onlyset]
    if args.skip:
        skipset = {t.upper() for t in args.skip}
        tickers = [t for t in tickers if t not in skipset]
    if args.limit_tickers and args.limit_tickers > 0:
        tickers = tickers[: args.limit_tickers]

    mapping = load_mapping(MAPPING_CSV)

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    # Determine mode
    use_range = bool(args.start and args.end)
    if use_range:
        # parse dates as UTC
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_dt < start_dt:
            raise SystemExit("end < start")

    total_new = 0
    for i, t in enumerate(tickers, 1):
        out_path = DIR_30M / f"{t}_30min_raw.csv"

        base = make_contract(t, mapping)
        q = ib.qualifyContracts(base)
        if not q:
            print(f"[{i}/{len(tickers)}] {t}: [SKIP] could not qualify contract.")
            continue
        con = q[0]

        # Probe stream (Head TS -> tiny hist)
        choice = probe_stream(ib, con, hist_days=args.probe_days)
        if choice is None:
            print(f"[{i}/{len(tickers)}] {t}: [SKIP 30m NO DATA] Head+Hist probe found nothing.")
            continue
        what, rth, origin = choice
        print(f"[{i}/{len(tickers)}] {t}: [PROBE OK] {what} RTH={int(rth)} via {origin}")

        with mute_162(ib):
            if use_range:
                # Range APPEND mode
                df = fetch_window_paged(ib, con, start_dt, end_dt, what, rth, page_days=60)
                if df.empty:
                    print(f"[{i}/{len(tickers)}] {t}: No rows for {args.start}..{args.end} ({what} RTH={int(rth)})")
                    continue
                existing = read_existing_30m(out_path)
                added = save_merge(out_path, existing, df)
                print(f"[{i}/{len(tickers)}] {t}: APPENDED {len(df)} rows ({what} RTH={int(rth)}) → {out_path} (+{added} net)")
                total_new += max(0, added)
            else:
                # Prepend YEARS mode
                fallback = "MIDPOINT" if what == "TRADES" else "TRADES"
                added = prepend_backfill_30m(
                    ib, con, out_path,
                    years_back=args.years,
                    duration_str=args.duration,
                    what_primary=what,
                    what_fallback=fallback,
                    use_rth=rth,
                    sleep_s=max(args.sleep_ms, 0) / 1000.0,
                )
                print(f"[{i}/{len(tickers)}] {t}: +{added} older bars via {what} RTH={int(rth)} → {out_path}")
                total_new += max(0, added)

        ib.sleep(max(args.sleep_ms, 0) / 1000.0)

    ib.disconnect()
    print(f"[DONE] s06 complete. Tickers={len(tickers)} | Net rows added={total_new}")

if __name__ == "__main__":
    main()