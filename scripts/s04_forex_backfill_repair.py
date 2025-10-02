#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s04_forex_backfill_repair.py — Full-history backfill + integrity audit/repair for FX (UTC, IBKR 7496)

What it does
- Works on one or more FX pairs; default = USDEUR (built from EURUSD via inversion).
- If the CSV does not exist → downloads full history and creates it.
- If the CSV exists:
    * --append  : append ONLY missing days at the end (safe default)
    * --audit   : check for gaps/duplicates; refill any gaps by fetching those date spans
    * --rebuild : ignore existing file and rebuild FULL history from IBKR, then overwrite

Output (per pair)
- Writes to the SHARED folder via common.paths.P.FOREX_DIR
  e.g., /Users/Finance/QuantShared/forex/{PAIR}_daily.csv
  Columns: date, open, high, low, close, volume, average, barCount  (all in UTC)

Notes
- IBKR contract source uses *IBKR-style* pairs (EURUSD, GBPUSD, USDJPY, etc.).
  If you request USDEUR, the script fetches EURUSD and **inverts** OHLC to USD/EUR.
- Historical data: whatToShow = MIDPOINT, useRTH = True (mirrors your other FX scripts).

Usage
  python scripts/s15_forex_backfill_repair.py --pairs USDEUR --mode append
  python scripts/s15_forex_backfill_repair.py --pairs USDEUR,USDJPY --mode audit
  python scripts/s15_forex_backfill_repair.py --pairs EURUSD --mode rebuild --years 20
"""

from ib_insync import IB, Forex, util
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import math
import os

# QuantShared-based forex folder (override with SHARED_FOREX_DIR if needed)
SHARED_FOREX_DIR = Path(os.environ.get("SHARED_FOREX_DIR", "/Users/Finance/QuantShared/forex"))
OUT_DIR = SHARED_FOREX_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Pair helpers ----------
def ib_src_for(pair: str) -> tuple[str, bool]:
    """
    Return (ib_symbol, invert) for the requested six-letter pair.
    Uses a whitelist of common IBKR direct pairs to avoid wrong inversions.
    """
    pair = pair.upper()
    if len(pair) != 6:
        raise ValueError("Pair must be 6 letters like USDEUR / EURUSD / USDJPY.")

    # Common direct pairs on IBKR (expand if you need more)
    DIRECT = {
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    }

    base, quote = pair[:3], pair[3:]
    direct = base + quote
    flipped = quote + base

    if direct in DIRECT:
        return direct, False
    if flipped in DIRECT:
        # e.g., USDEUR -> EURUSD (invert OHLC)
        return flipped, True

    # Fallback: assume direct is available (don’t invert by default)
    return direct, False

def invert_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["open","high","low","close"]:
        df[c] = df[c].apply(lambda x: (1/x) if (x and x != 0) else None)
    # Ensure low<=high after inversion (rare, but double-check)
    lo = df[["open","high","low","close"]].min(axis=1)
    hi = df[["open","high","low","close"]].max(axis=1)
    df["low"], df["high"] = lo, hi
    return df

# ---------- IBKR fetch ----------
def fetch_daily_fx(ib: IB, symbol: str, end_dt: datetime, duration_str: str) -> pd.DataFrame:
    """
    Fetch daily bars for an IBKR Forex(symbol like 'EURUSD') from end_dt back by duration_str.
    Returns columns: ['date','open','high','low','close','volume','average','barCount'] in UTC.
    """
    bars = ib.reqHistoricalData(
        Forex(symbol, exchange="IDEALPRO"),
        endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
        durationStr=duration_str,
        barSizeSetting="1 day",
        whatToShow="MIDPOINT",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False
    )
    if not bars:
        return pd.DataFrame(columns=["date","open","high","low","close","volume","average","barCount"])
    df = util.df(bars)
    # normalize types
    for col in ["open","high","low","close","volume","average","barCount"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df[["date","open","high","low","close","volume","average","barCount"]]

# ---------- File ops ----------
def load_existing(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} missing 'date' column")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df

def save_csv(path: Path, df: pd.DataFrame):
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    df.to_csv(path, index=False)

# ---------- Modes ----------
def mode_append(pair: str, out_path: Path, ib: IB, max_back_years: int):
    ib_symbol, invert = ib_src_for(pair)
    end_dt = datetime.now()
    existing = load_existing(out_path)
    if existing is not None and not existing.empty:
        last_dt = existing["date"].max()
        # If already up-to-date, exit quietly
        if (end_dt.date() - last_dt.date()).days <= 0:
            print(f"{pair}: up-to-date (no new rows).")
            return
        duration_days = (end_dt.date() - last_dt.date()).days
        duration_str = f"{duration_days} D"
        df_new = fetch_daily_fx(ib, ib_symbol, end_dt, duration_str)
        if invert:
            df_new = invert_ohlc(df_new)
        df_all = pd.concat([existing, df_new]).drop_duplicates(subset=["date"]).sort_values("date")
        save_csv(out_path, df_all)
        print(f"{pair}: appended +{max(0, len(df_all)-len(existing))} rows → {out_path.name}")
    else:
        # Full history fetch (first time)
        duration_str = f"{max_back_years} Y"
        df_full = fetch_daily_fx(ib, ib_symbol, end_dt, duration_str)
        if invert:
            df_full = invert_ohlc(df_full)
        save_csv(out_path, df_full)
        print(f"{pair}: created {out_path.name} rows={len(df_full)}")

def mode_rebuild(pair: str, out_path: Path, ib: IB, max_back_years: int):
    ib_symbol, invert = ib_src_for(pair)
    end_dt = datetime.now()
    df_full = fetch_daily_fx(ib, ib_symbol, end_dt, f"{max_back_years} Y")
    if invert:
        df_full = invert_ohlc(df_full)
    save_csv(out_path, df_full)
    print(f"{pair}: rebuilt {out_path.name} rows={len(df_full)}")

def mode_audit(pair: str, out_path: Path, ib: IB):
    """
    Detect gaps/dupes, fill gaps by fetching the missing date ranges.
    """
    existing = load_existing(out_path)
    if existing is None or existing.empty:
        print(f"{pair}: file empty or missing — run with --append or --rebuild.")
        return

    # Build continuous daily index from min→max
    dates = pd.date_range(existing["date"].min().date(),
                          existing["date"].max().date(),
                          freq="D", tz="UTC")
    present = pd.Series(True, index=existing["date"].dt.normalize())
    missing = [d for d in dates if d not in present.index]

    dupes = existing["date"].duplicated().sum()
    if not missing and dupes == 0:
        print(f"{pair}: audit OK — no gaps, no dupes.")
        return

    ib_symbol, invert = ib_src_for(pair)
    end_dt = datetime.now()

    # Fill gaps by minimal fetch ranges
    patched = existing.copy()
    if missing:
        # group consecutive missing dates into spans
        spans = []
        start = missing[0]
        prev = start
        for d in missing[1:]:
            if (d - prev).days == 1:
                prev = d
            else:
                spans.append((start, prev))
                start = d
                prev = d
        spans.append((start, prev))

        for a,b in spans:
            # fetch a buffer around the span (to ensure coverage)
            span_days = (b - a).days + 1
            duration_days = span_days + 10
            duration_str = f"{duration_days} D"
            # end at b + 1 day, so the range includes b
            end_for_span = (b + timedelta(days=1)).to_pydatetime()
            df_patch = fetch_daily_fx(ib, ib_symbol, end_for_span, duration_str)
            if invert:
                df_patch = invert_ohlc(df_patch)
            patched = pd.concat([patched, df_patch], ignore_index=True)

    # Deduplicate
    patched = patched.drop_duplicates(subset=["date"]).sort_values("date")
    save_csv(out_path, patched)
    print(f"{pair}: audit fixed — gaps filled={len(missing)}, dupes_removed={dupes}. Saved {out_path.name}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=65)
    ap.add_argument("--pairs", default="USDEUR", help="Comma-separated list, e.g. USDEUR,USDGBP,EURUSD")
    ap.add_argument("--mode", choices=["append","audit","rebuild"], default="append")
    ap.add_argument("--years", type=int, default=15, help="Max lookback years for full fetch/rebuild")
    args = ap.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    try:
        for pair in pairs:
            out_path = OUT_DIR / f"{pair}_daily.csv"
            if args.mode == "append":
                mode_append(pair, out_path, ib, args.years)
            elif args.mode == "rebuild":
                mode_rebuild(pair, out_path, ib, args.years)
            else:
                mode_audit(pair, out_path, ib)
    finally:
        ib.disconnect()
        print("[DONE] s15_forex_backfill_repair complete.")

if __name__ == "__main__":
    main()