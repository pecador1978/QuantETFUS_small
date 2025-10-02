#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_full_30min_historical.py — selected tickers, London-first, 2016 → yesterday

- Hardcoded universe (your list below).
- Contract qualification order:
    1) LSEETF, GBP
    2) LSEETF, USD
    3) LSE,    GBP
    4) LSE,    USD
    5) mapping.csv (if present)
    6) SMART fallback
- 30m bars, paged (60 D) with adaptive per-page fallback:
    TRADES/MIDPOINT/BID_ASK × RTH{1,0}
- Output:
  data_raw/targeted_ETFs_US/30min_test/{TICKER}_2016-01-01_<END>_30m.csv
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
import sys, os
import pandas as pd
from contextlib import contextmanager
from ib_insync import IB, Contract, util

# ---------- project bootstrap ----------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- shared raw store ----------
# US default; for EU set env SHARED_RAW_BASE="/Users/Finance/QuantShared/data_raw_ETF_EU"
SHARED_BASE = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
OUT_DIR = SHARED_BASE / "30min_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ticker mapping ----------
MAPPING_CSV = Path("/Users/Finance/QuantShared/ticker_mapping.csv")

# ---------- your exact (deduped) list ----------
TICKERS = [
    "SGLD","AIGA"
]

# ---------- silence 162 ----------
@contextmanager
def mute_162(ib: IB):
    def _handler(reqId, code, msg, contract):
        if code == 162:  # HMDS no data
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ---------- helpers ----------
def load_mapping(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        return {}
    df = pd.read_csv(path_csv, sep=";")
    mp = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType":         (str(r.get("SecType","STK")).strip() or "STK"),
            "Exchange":        (str(r.get("Exchange","SMART")).strip() or "SMART"),
            "Currency":        (str(r.get("Currency","USD")).strip().upper() or "USD"),
            "PrimaryExchange": str(r.get("PrimaryExchange","")).strip(),
        }
    return mp

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns: df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")

def _end_str_utc(ts: datetime) -> str:
    return ts.strftime("%Y%m%d %H:%M:%S") + " UTC"

def try_qualify(ib: IB, symbol: str, mapping: dict) -> Contract | None:
    """Prefer LSEETF/LSE (GBP→USD), then mapping, then SMART."""
    attempts: list[Contract] = []

    # 1) LSEETF GBP → USD
    for ccy in ("GBP","USD"):
        c = Contract(symbol=symbol, secType="STK", exchange="LSEETF", primaryExchange="LSEETF", currency=ccy)
        attempts.append(c)
    # 2) LSE GBP → USD
    for ccy in ("GBP","USD"):
        c = Contract(symbol=symbol, secType="STK", exchange="LSE", primaryExchange="LSE", currency=ccy)
        attempts.append(c)
    # 3) mapping row
    if symbol in mapping:
        m = mapping[symbol]
        c = Contract(symbol=symbol,
                     secType=m.get("SecType","STK"),
                     exchange=m.get("Exchange","SMART"),
                     currency=m.get("Currency","USD"))
        if m.get("PrimaryExchange"): c.primaryExchange = m["PrimaryExchange"]
        attempts.append(c)
    # 4) SMART fallback
    attempts.append(Contract(symbol=symbol, secType="STK", exchange="SMART", currency="USD"))
    attempts.append(Contract(symbol=symbol, secType="STK", exchange="SMART", currency="GBP"))

    for c in attempts:
        try:
            q = ib.qualifyContracts(c)
            if q:
                return q[0]
        except Exception:
            continue
    return None

def _fetch_one_window(ib: IB, con: Contract, begin: datetime, end: datetime,
                      what: str, use_rth: bool) -> pd.DataFrame:
    duration_days = (end - begin).days + 1
    try:
        bars = ib.reqHistoricalData(
            con,
            endDateTime=_end_str_utc(end),
            durationStr=f"{max(1, duration_days)} D",
            barSizeSetting="30 mins",
            whatToShow=what,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
    except Exception:
        bars = []
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df = normalize(util.df(bars))
    if df.empty:
        return df
    return df[(df["date"] >= begin) & (df["date"] <= end)][["date","open","high","low","close","volume","average","barCount"]]

def fetch_window_paged_adaptive(ib: IB, con: Contract,
                                start_dt: datetime, end_dt: datetime,
                                seed_what: str, seed_rth: bool,
                                page_days: int = 60) -> pd.DataFrame:
    order = ["TRADES", "MIDPOINT", "BID_ASK"]
    alts = [w for w in order if w != seed_what]
    tries_template = [
        (seed_what, seed_rth),
        (seed_what, not seed_rth),
        (alts[0],  seed_rth),
        (alts[0],  not seed_rth),
        (alts[1],  seed_rth),
        (alts[1],  not seed_rth),
    ]
    rows = []
    cursor = end_dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    start_floor = start_dt.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)

    with mute_162(ib):
        while cursor >= start_floor:
            begin = max(start_floor, cursor - timedelta(days=page_days - 1))
            got = None
            for what, rth in tries_template:
                df = _fetch_one_window(ib, con, begin, cursor, what, rth)
                if not df.empty:
                    got = df
                    seed_what, seed_rth = what, rth  # keep best combo going
                    break
            if got is not None and not got.empty:
                rows.append(got)
            cursor = begin - timedelta(seconds=1)

    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
    return out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)

def probe_stream(ib: IB, con: Contract):
    trials = [
        ("TRADES", True), ("TRADES", False),
        ("MIDPOINT", True), ("MIDPOINT", False),
        ("BID_ASK", True), ("BID_ASK", False),
    ]
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(days=3)
    with mute_162(ib):
        for what, rth in trials:
            df = _fetch_one_window(ib, con, start, now, what, rth)
            if not df.empty:
                return what, rth
    return None, None

# ---------- main ----------
def main():
    print(f"[RUNNING] {Path(__file__).name}")
    START = datetime(2016, 1, 1, tzinfo=timezone.utc)
    END   = (datetime.now(timezone.utc) - timedelta(days=1)).replace(hour=23, minute=59, second=59)

    mapping = load_mapping(MAPPING_CSV)
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=187, readonly=True)
    print("[IB] Connected.")

    total = len(TICKERS)
    for i, t in enumerate(TICKERS, 1):
        try:
            con = try_qualify(ib, t, mapping)
            if not con:
                print(f"[{i}/{total}] {t}: SKIP (no qualified LSE contract)")
                continue

            seed_what, seed_rth = probe_stream(ib, con)
            if not seed_what:
                print(f"[{i}/{total}] {t}: SKIP (no 30m stream)")
                continue
            print(f"[{i}/{total}] {t}: seed {seed_what} RTH={int(seed_rth)} on {getattr(con,'primaryExchange','')}")

            df = fetch_window_paged_adaptive(ib, con, START, END, seed_what, seed_rth, page_days=60)
            if df.empty:
                print(f"[{i}/{total}] {t}: no rows {START.date()}..{END.date()}")
                continue

            out_path = OUT_DIR / f"{t}_{START.date()}_{END.date()}_30m.csv"
            df.to_csv(out_path, index=False)
            print(f"[{i}/{total}] {t}: wrote {len(df)} rows → {out_path}")

            ib.sleep(0.3)

        except Exception as e:
            print(f"[{i}/{total}] {t}: ERROR {e}")

    ib.disconnect()
    print(f"[DONE] {total} tickers processed. Output → {OUT_DIR}")

if __name__ == "__main__":
    main()