#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s10_ibkr_download_daily.py — Append-only DAILY updater (IBKR via ib_insync, UTC-only)

- Universe: ETF_list.xlsx (sheet default comes from env ETF_SHEET or 'signalsUSD').
- Output (append-only): <QuantShared>/data_raw_ETF_US/daily/{TICKER}_daily_raw.csv
- Uses project ticker_mapping.csv when available to qualify contracts (prefers ConId/conId).
- Fallback: London-USD qualification (LSEETF/LSE → SMART pinned), same logic as s06/s07.
- Mutes noisy IB 162/200/321 chatter.

Environment (optional)
- ETF_LIST_XLSX      : path to ETF_list.xlsx
- ETF_SHEET          : sheet name (default 'signalsUSD')
- SHARED_RAW_BASE    : base dir for raw data (default /Users/Finance/QuantShared/data_raw_ETF_US)
- TICKER_MAPPING_CSV : mapping CSV (default /Users/Finance/QuantETFUS_small/config/ticker_mapping.csv)
- DEFAULT_CCY        : default currency if mapping missing (default 'USD')
"""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import os
import argparse
import pandas as pd

from ib_insync import IB, Contract, Stock, util

# ===== Paths & defaults =====
SHARED_BASE   = Path(os.environ.get("SHARED_RAW_BASE", "/Users/Finance/QuantShared/data_raw_ETF_US"))
DAILY_DIR     = SHARED_BASE / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EXCEL = Path(os.environ.get("ETF_LIST_XLSX", "/Users/Finance/QuantShared/ETF_list.xlsx"))
DEFAULT_SHEET = os.environ.get("ETF_SHEET", "signalsUSD")

# IMPORTANT: default to the *project* mapping file
DEFAULT_MAPPING = Path(os.environ.get(
    "TICKER_MAPPING_CSV",
    "/Users/Finance/QuantETFUS_small/config/ticker_mapping.csv"
))

DEFAULT_CCY   = os.environ.get("DEFAULT_CCY", "USD")

print(f"[INFO] Excel={DEFAULT_EXCEL} | sheet={DEFAULT_SHEET} | out={DAILY_DIR}")

# ===== Logging / IB warning muter =====
@contextmanager
def mute_ibkr_warnings(ib: IB, suppress_codes=(162, 200, 321)):
    """
    Mute common noisy messages:
      162 HMDS query returned no data
      200 No security definition found
      321 Error validating request
    Still prints other unexpected errors with their code.
    """
    def _handler(reqId, code, msg, *a, **k):
        try:
            code_int = int(code)
        except Exception:
            code_int = None
        if code_int in suppress_codes:
            return
        print(f"[IB {code}] {msg}")
    ib.errorEvent += _handler
    try:
        yield
    finally:
        ib.errorEvent -= _handler

# ===== Excel & CSV helpers =====
def load_tickers_from_excel(path_xlsx: str | Path, sheet: str, col_name: str = "Ticker") -> list[str]:
    df = pd.read_excel(path_xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col_name.lower()
    if key not in colmap:
        raise ValueError(f"Column '{col_name}' not found in {path_xlsx} (sheet '{sheet}').")
    tickers = df[colmap[key]].astype(str).str.strip().str.upper()
    return [t for t in tickers.unique().tolist() if t]

def load_mapping(path_csv: str | Path) -> dict[str, Dict[str, Any]]:
    """
    Optional mapping CSV (semicolon-delimited).
    Recommended columns:
      Ticker;ConId;SecType;Exchange;Currency;PrimaryExchange
    (accepts 'conId' or 'ConId')
    """
    p = Path(path_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p, sep=";")
    if "Ticker" not in df.columns:
        raise ValueError(f"{p} must have a 'Ticker' column (semicolon-separated).")

    # normalize column presence / casing
    if "ConId" in df.columns and "conId" not in df.columns:
        df["conId"] = df["ConId"]
    for col in ["conId","SecType","Exchange","Currency","PrimaryExchange"]:
        if col not in df.columns:
            df[col] = ""

    mp: dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        if not t:
            continue
        mp[t] = {
            "conId":           str(r.get("conId", "")).strip(),
            "SecType":         (str(r.get("SecType", "")).strip() or "STK").upper(),
            "Exchange":        str(r.get("Exchange", "")).strip() or "SMART",
            "Currency":        (str(r.get("Currency", "")).strip() or DEFAULT_CCY).upper(),
            "PrimaryExchange": str(r.get("PrimaryExchange", "")).strip(),
        }
    return mp

# ===== Contract builders =====
def contract_from_map(ticker: str, mapping: dict[str, Dict[str, Any]]) -> Optional[Contract]:
    m = mapping.get(ticker.upper())
    if not m:
        return None
    # Prefer conId if supplied (most reliable)
    conId = m.get("conId", "")
    if conId and str(conId).isdigit():
        c = Contract()
        c.conId = int(conId)
        return c
    # Otherwise build from fields
    c = Contract()
    c.symbol          = ticker
    c.secType         = m.get("SecType", "STK") or "STK"
    c.exchange        = m.get("Exchange", "SMART") or "SMART"
    c.currency        = m.get("Currency", DEFAULT_CCY) or DEFAULT_CCY
    pe = m.get("PrimaryExchange", "")
    if pe:
        c.primaryExchange = pe
    return c

def try_qualify_london_usd(ib: IB, symbol: str) -> Optional[Contract]:
    """
    Prefer LSEETF/LSE in USD; then SMART pinned to LSEETF/LSE; then ETF secType;
    finally discover via reqContractDetails and filter to USD + (LSEETF/LSE).
    Matches s06/s07 so behavior is consistent.
    """
    attempts: list[Contract] = [
        Stock(symbol, 'LSEETF', 'USD'),
        Stock(symbol, 'LSE',    'USD'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSEETF'),
        Stock(symbol, 'SMART',  'USD', primaryExchange='LSE'),
        Contract(symbol=symbol, secType='ETF', exchange='LSEETF', currency='USD'),
        Contract(symbol=symbol, secType='ETF', exchange='SMART', currency='USD', primaryExchange='LSEETF'),
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
        for base in (Stock(symbol, 'LSEETF', 'USD'), Stock(symbol, 'SMART', 'USD')):
            cds = ib.reqContractDetails(base)
            for cd in cds:
                c = cd.contract
                px = (c.primaryExchange or c.exchange or '').upper()
                if (c.currency or '').upper() == 'USD' and px in {'LSEETF', 'LSE'}:
                    return c
    except Exception:
        pass
    return None

# ===== Data fetch & append =====
def fetch_daily_df(ib: IB, contract: Contract, duration: str, what: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",               # now
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df = util.df(bars)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","average","barCount"]:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date","open","high","low","close","volume","average","barCount"]]

def append_only(path_csv: str | Path, df_new: pd.DataFrame) -> int:
    """Append only newer rows (by 'date' col), keep UTC."""
    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"], utc=True)
    p = Path(path_csv)
    if p.exists():
        old = pd.read_csv(p)
        if "date" not in old.columns:
            raise ValueError(f"{p} missing 'date' column")
        old["date"] = pd.to_datetime(old["date"], utc=True, errors="coerce")
        last_dt = old["date"].max()
        df_new = df_new[df_new["date"] > last_dt] if pd.notna(last_dt) else df_new
        merged = pd.concat([old, df_new], ignore_index=True)
    else:
        merged = df_new
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date")
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)
    return len(df_new)

# ===== main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=60)
    ap.add_argument("--duration", default="10 Y", help="IB durationStr (e.g., '10 Y')")
    ap.add_argument("--what", default="TRADES", help="TRADES / MIDPOINT / BID_ASK")
    ap.add_argument("--use_rth", type=int, default=1, help="1 = RTH only, 0 = all hours")
    ap.add_argument("--excel", default=str(DEFAULT_EXCEL))
    ap.add_argument("--sheet", default=str(DEFAULT_SHEET))
    ap.add_argument("--ticker_col", default="Ticker")
    ap.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    ap.add_argument("--outdir", default=str(DAILY_DIR))
    ap.add_argument("--sleep_ms", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tickers = load_tickers_from_excel(args.excel, args.sheet, args.ticker_col)
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
    mapping = load_mapping(args.mapping)

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    with mute_ibkr_warnings(ib):
        for i, t in enumerate(tickers, 1):
            try:
                # 1) mapping (prefer ConId) → qualify; fallback to London-USD discovery
                con = contract_from_map(t, mapping)
                if con:
                    q = ib.qualifyContracts(con)
                    if not q:
                        con = try_qualify_london_usd(ib, t)
                    else:
                        con = q[0]
                else:
                    con = try_qualify_london_usd(ib, t)

                if not con:
                    print(f"[{i}/{len(tickers)}] {t}: [SKIP] could not qualify")
                    continue

                df = fetch_daily_df(ib, con, args.duration, args.what, bool(args.use_rth))
                if df.empty:
                    print(f"[{i}/{len(tickers)}] {t}: [SKIP] no data")
                    continue

                out_path = Path(args.outdir) / f"{t}_daily_raw.csv"
                added = append_only(out_path, df)
                print(f"[{i}/{len(tickers)}] {t}: +{added} → {out_path}")
                ib.sleep(max(args.sleep_ms, 0) / 1000.0)

            except Exception as e:
                print(f"[{i}/{len(tickers)}] {t}: [ERR] {e}")

    ib.disconnect()
    print("[DONE] s10_ibkr_download_daily (append-only, UTC) complete.")

if __name__ == "__main__":
    main()