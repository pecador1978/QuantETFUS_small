#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s14_forex_update.py — Append-only daily USDEUR via EURUSD inversion (UTC, clone-friendly)

- Uses QuantShared paths via common.paths
- Pulls EURUSD (IDEALPRO), inverts to USDEUR, appends only new rows
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from ib_insync import IB, Forex, util

# paths (QuantShared)
SHARED_FOREX_DIR = Path(os.environ.get("SHARED_FOREX_DIR", "/Users/Finance/QuantShared/forex"))
FOREX_DIR = SHARED_FOREX_DIR
FOREX_DIR.mkdir(parents=True, exist_ok=True)

def _append_only_csv(path: Path, df_new: pd.DataFrame, date_col: str = "date") -> int:
    df_new = df_new.copy()
    df_new[date_col] = pd.to_datetime(df_new[date_col], utc=True)
    if path.exists():
        old = pd.read_csv(path)
        if date_col not in old.columns: raise ValueError(f"{path} missing '{date_col}'")
        old[date_col] = pd.to_datetime(old[date_col], utc=True)
        last_dt = old[date_col].max()
        df_new = df_new[df_new[date_col] > last_dt] if pd.notna(last_dt) else df_new
        out = pd.concat([old, df_new], ignore_index=True)
    else:
        out = df_new
    out = out.drop_duplicates(subset=[date_col]).sort_values(date_col)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return len(df_new)

def _fetch_eurusd(ib: IB, duration_str: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        Forex("EURUSD", exchange="IDEALPRO"),
        endDateTime="",
        durationStr=duration_str,
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns: df[c]=pd.NA
        df[c]=pd.to_numeric(df[c], errors="coerce")
    return df[["date","open","high","low","close","volume"]].dropna(subset=["date"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=64)
    ap.add_argument("--symbol", default="USDEUR", help="Output symbol; USDEUR is EURUSD inverted")
    ap.add_argument("--from_pair", default="EURUSD", help="Base pair to invert")
    ap.add_argument("--use_rth", type=int, default=1)
    ap.add_argument("--lookback_years_if_empty", type=int, default=10)
    ap.add_argument("--outfile", default=None)
    args = ap.parse_args()

    symbol = args.symbol.upper()
    out_path = Path(args.outfile) if args.outfile else (FOREX_DIR / f"{symbol}_daily.csv")

    # duration: initial N years or incremental days
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "date" in existing.columns and not existing.empty:
            last = pd.to_datetime(existing["date"], utc=True, errors="coerce").max()
            days = (datetime.now(timezone.utc) - last).days if pd.notna(last) else args.lookback_years_if_empty * 365
            duration = f"{max(1, days)} D"
        else:
            duration = f"{args.lookback_years_if_empty} Y"
    else:
        duration = f"{args.lookback_years_if_empty} Y"

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    try:
        if symbol == "USDEUR" and args.from_pair.upper() == "EURUSD":
            eurusd = _fetch_eurusd(ib, duration, bool(args.use_rth))
            if eurusd.empty:
                print("[WARN] EURUSD returned no rows."); return
            inv = eurusd.copy()
            for c in ["open","high","low","close"]:
                inv[c] = inv[c].where(inv[c] == 0, 1.0 / inv[c])
                inv.loc[inv[c] == 0, c] = pd.NA
            added = _append_only_csv(out_path, inv, date_col="date")
            print(f"[OK] {symbol}: +{added} rows → {out_path}")
        else:
            print(f"[ERR] Only USDEUR via EURUSD inversion is implemented.")
    finally:
        ib.disconnect()
        print("[IB] Disconnected.")

if __name__ == "__main__":
    main()