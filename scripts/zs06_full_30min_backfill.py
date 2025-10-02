#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_full_30min_backfill.py — FULL history rebuild of 30‑minute bars (UTC)

- Paginates backward in safe windows (default 90 days) with whatToShow=TRADES,
  falling back to MIDPOINT if TRADES is empty for that window.
- Overwrites existing 30m_raw files (clean slate). After the first run, keep using s11 for appends.

Inputs:
  - ETF_list.xlsx (auto sheet: signals / signalsUSD unless overridden with --sheet)
  - Optional contract mapping: <project>/config/ticker_mapping.csv (semicolon-separated)

Outputs:
  <project>/data_raw/<TARGET_BUCKET>/30min/{TICKER}_30min_raw.csv  (UTC)
  columns: date,open,high,low,close,volume,average,barCount
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import os
import sys
import pandas as pd
from ib_insync import IB, Contract, util

# ---------------- bootstrapping: make 'common' importable ----------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # .../QuantETF*/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                        # .../QuantETF*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------- shared paths & settings ----------------
from common.paths import P  # keep using project config for lists/mapping

# currency hint by project name
DEFAULT_CCY = "USD" if PROJECT_ROOT.name.endswith(("US_small", "US")) else "EUR"

# write 30m bars to the shared raw store (US). For EU, set env SHARED_RAW_BASE to .../data_raw_ETF_EU
SHARED_BASE = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
DIR_30M = SHARED_BASE / "30min"
DIR_30M.mkdir(parents=True, exist_ok=True)

# inputs (unchanged, live in the project repo)
EXCEL_LIST       = str(P.ETF_LIST)
EXCEL_SHEET      = ("signalsUSD" if PROJECT_ROOT.name.endswith(("US_small", "US")) else "signals")
EXCEL_TICKER_COL = "Ticker"
# ticker mapping
MAPPING_CSV = Path("/Users/Finance/QuantShared/ticker_mapping.csv")

# ---------------- helpers ----------------
def _fmt_ib_time(ts: datetime) -> str:
    return ts.strftime("%Y%m%d %H:%M:%S")  # IBKR expected format

def load_tickers_from_excel(path_xlsx: str, sheet: str, col_name: str) -> list[str]:
    df = pd.read_excel(path_xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col_name.lower()
    if key not in colmap:
        raise ValueError(f"Column '{col_name}' not found in {path_xlsx} (sheet '{sheet}').")
    tickers = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in tickers.unique().tolist() if t]

def load_mapping(path_csv: str) -> dict:
    """Optional mapping for IBKR contracts; CSV is semicolon-separated."""
    if not os.path.isfile(path_csv):
        return {}
    df = pd.read_csv(path_csv, sep=";")
    need = {"Ticker","SecType","Exchange","Currency","PrimaryExchange"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{path_csv} must contain columns: {need}")
    mp = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType":         (str(r.get("SecType", "STK")).strip() or "STK").upper(),
            "Exchange":        str(r.get("Exchange", "SMART")).strip() or "SMART",
            "Currency":        (str(r.get("Currency", DEFAULT_CCY)).strip() or DEFAULT_CCY).upper(),
            "PrimaryExchange": str(r.get("PrimaryExchange", "")).strip() or ""
        }
    return mp

def make_contract(ticker: str, mapping: dict) -> Contract:
    m = mapping.get(ticker.upper(), {})
    c = Contract()
    c.symbol          = ticker
    c.secType         = m.get("SecType", "STK")
    c.exchange        = m.get("Exchange", "SMART")
    c.currency        = m.get("Currency", DEFAULT_CCY)
    pe = m.get("PrimaryExchange", "")
    if pe:
        c.primaryExchange = pe
    return c

def _normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","open","high","low","close","volume","average","barCount"]]

def fetch_30m_window(ib: IB, contract: Contract, end_ts: datetime, duration: str,
                     use_rth: bool) -> pd.DataFrame:
    # Try TRADES first then MIDPOINT within this window
    for what in ("TRADES", "MIDPOINT"):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=_fmt_ib_time(end_ts),
            durationStr=duration,           # e.g., "90 D"
            barSizeSetting="30 mins",
            whatToShow=what,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
        if bars:
            df = _normalize_hist_df(util.df(bars))
            if not df.empty:
                df["__what"] = what
                return df
    return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount","__what"])

def backfill_30m(ib: IB, contract: Contract, years: int, use_rth: bool,
                 window_days: int, max_empty_windows: int) -> pd.DataFrame:
    """
    Page backward from 'now' until we cover ~years or we hit too many empty windows.
    """
    end = datetime.now(timezone.utc).replace(microsecond=0)
    earliest = end - timedelta(days=int(years * 365.25) + 5)

    rows = []
    last_edge = end
    empty_streak = 0
    guard = 0

    duration = f"{window_days} D"  # safe window for 30m bars

    while last_edge > earliest and guard < 1000:
        guard += 1
        df = fetch_30m_window(ib, contract, end_ts=last_edge, duration=duration, use_rth=use_rth)
        if df.empty:
            empty_streak += 1
            if empty_streak >= max_empty_windows:
                break
            # step back a full window to move on
            last_edge = last_edge - timedelta(days=window_days)
            continue

        empty_streak = 0
        rows.append(df)

        new_last = pd.to_datetime(df["date"].min(), utc=True) - timedelta(days=1)
        if new_last >= last_edge:
            # safety stop; shouldn't happen but avoids infinite loop
            break
        last_edge = new_last

    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "__what" in out.columns:
        out = out.drop(columns=["__what"])
    return out

def save_overwrite(path_csv: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df.to_csv(path_csv, index=False)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=57)
    ap.add_argument("--years", type=int, default=2, help="How many years of 30m history to rebuild.")
    ap.add_argument("--use_rth", type=int, default=1, help="1 = RTH bars; 0 = all sessions")
    ap.add_argument("--window_days", type=int, default=90, help="Paging window size in days.")
    ap.add_argument("--max_empty_windows", type=int, default=6, help="Stop if we hit this many empty windows.")
    ap.add_argument("--excel", default=str(EXCEL_LIST))
    ap.add_argument("--sheet", default=str(EXCEL_SHEET))
    ap.add_argument("--ticker_col", default=str(EXCEL_TICKER_COL))
    ap.add_argument("--mapping", default=str(MAPPING_CSV))
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (0 = all)")
    ap.add_argument("--sleep_ms", type=int, default=200)
    args = ap.parse_args()

    tickers = load_tickers_from_excel(args.excel, args.sheet, args.ticker_col)
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    mapping = load_mapping(args.mapping)

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    for i, t in enumerate(tickers, 1):
        try:
            c = make_contract(t, mapping)
            qualified = ib.qualifyContracts(c)
            if not qualified:
                print(f"[{i}/{len(tickers)}] {t}: could not qualify contract; skipping.")
                continue
            qc = qualified[0]

            df = backfill_30m(
                ib, qc,
                years=args.years,
                use_rth=bool(args.use_rth),
                window_days=args.window_days,
                max_empty_windows=args.max_empty_windows
            )

            if df.empty:
                print(f"[{i}/{len(tickers)}] {t}: no history returned.")
                continue

            out_path = str(DIR_30M / f"{t}_30min_raw.csv")
            save_overwrite(out_path, df)
            print(f"[{i}/{len(tickers)}] {t}: wrote {len(df)} rows → {out_path}")
            ib.sleep(args.sleep_ms / 1000.0)
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR: {e}")

    ib.disconnect()
    print("[DONE] s06_full_30min_rebuild complete.")

if __name__ == "__main__":
    main()