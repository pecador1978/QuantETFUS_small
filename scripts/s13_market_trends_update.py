#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s13_market_trends_update.py — Append + build ratios/composites (UTC)

Part A — Update/initialize raw daily series (TradingView schema):
- Reads allowed tickers from shared ETF_list.xlsx (sheet 'market_trends', col 'Ticker')
- Skips ratio/synthetic files (names containing 'ratio', '_X_', '_composite', '_spy_ratio').
- Preserves TradingView schema: time,open,high,low,close,Volume
- Robust IBKR qualification with mapping & fallbacks

Part B — Build ratios/composites

New:
- --ignore_list 1 : process ALL *_TV_daily.csv in the source dir (bypass Excel gating)
- --only A B ...  : process only these tickers (whitelist)
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
from ib_insync import IB, Contract, Stock, util

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # scripts/ -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P  # shared-aware paths (QuantShared)

# =========================
# Config (folders & lists)
# =========================

SRC_DIR     = str(P.MARKET_TRENDS_DIR)            # e.g., /Users/Finance/QuantShared/market_trends
LIST_XLSX   = str(P.ETF_LIST)                     # e.g., /Users/Finance/QuantShared/ETF_list.xlsx
LIST_SHEET  = "market_trends"
LIST_COL    = "Ticker"
MAPPING_CSV = str(P.CONFIG_DIR / "ticker_mapping.csv")  # semicolon-separated

VIX_NAMES = {"VIX", "VIX3M", "VIX9D", "VVIX"}
STK_PRIMARY_EXCHANGES = ["ARCA", "NASDAQ", "NYSE", "BATS", "ISLAND", "AMEX"]

SEC_WHAT_DEFAULT = {
    "IND":  "TRADES",
    "STK":  "TRADES",
    "FUT":  "TRADES",
    "CASH": "MIDPOINT",
}

# ---- Ratios (DXY replaced by UUP) ----
RATIO_SPECS: List[str] = [
    "DBC/SPY",
    "DBC/TLT",
    "DIA/QQQ",
    "EEM/SPY",
    "EFA/SPY",
    "FEZ/SPY",
    "GLD/SPY",
    "GLD/TLT",
    "HYG/IEI",
    "JNK/LQD",
    "QQQ/IWM",
    "QQQ/SPY",
    "SPHB/SPY",
    "XLP/SPY",
    "XLY/XLP",
    "HYG/TLT",
    "EMB/LQD",
    "LQD/TLT",
    "RSP/SPY",
    "RSP/QQQ",
    "SPY/TLT",
    "QQQ/TLT",
    "CEW/UUP",
]

# ---- Composites (equal weight) ----
COMPOSITES: Dict[str, List[str]] = {
    "XLU_XLV_XLP_composite": ["XLU", "XLV", "XLP"]
}

# =========================
# Utilities
# =========================

def pick_what_to_show(sec_type: str, override: Optional[str]) -> str:
    if override and override.strip():
        return override.strip().upper()
    return SEC_WHAT_DEFAULT.get((sec_type or "").upper(), "TRADES")


def load_market_trend_tickers(xlsx: str, sheet: str, col: str) -> set[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    colmap = {c.lower(): c for c in df.columns}
    key = col.lower()
    if key not in colmap:
        raise SystemExit(f"Column '{col}' not found in {xlsx} (sheet '{sheet}').")
    tickers = df[colmap[key]].astype(str).str.strip().str.upper()
    return {t for t in tickers.tolist() if t}


def load_mapping(path_csv: str) -> dict:
    if not os.path.isfile(path_csv):
        return {}
    df = pd.read_csv(path_csv, sep=";")
    need = {"Ticker","SecType","Exchange","Currency","PrimaryExchange"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"{path_csv} must contain columns: {need}")
    mp = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        mp[t] = {
            "SecType": (str(r.get("SecType","STK")).strip().upper() or "STK"),
            "Exchange": (str(r.get("Exchange","SMART")).strip() or "SMART"),
            "Currency": (str(r.get("Currency","USD")).strip().upper() or "USD"),
            "PrimaryExchange": str(r.get("PrimaryExchange","")).strip() or ""
        }
    return mp


def make_contract_from_map(ticker: str, mapping: dict) -> Contract:
    m = mapping.get(ticker.upper(), {})
    c = Contract()
    c.symbol  = ticker
    c.secType = m.get("SecType","STK")
    c.exchange = m.get("Exchange","SMART")
    c.currency = m.get("Currency","USD")
    pe = m.get("PrimaryExchange","")
    if pe:
        c.primaryExchange = pe
    return c


def qualify_with_fallbacks(ib: IB, ticker: str, mapping: dict) -> Optional[Contract]:
    # 1) mapping first
    if ticker in mapping:
        c = make_contract_from_map(ticker, mapping)
        qc = ib.qualifyContracts(c)
        if qc:
            return qc[0]

    # 2) special indices: VIX family
    if ticker in VIX_NAMES:
        for ex in ["CBOE", "CFE"]:
            c = Contract(symbol=ticker, secType="IND", exchange=ex, currency="USD")
            qc = ib.qualifyContracts(c)
            if qc:
                return qc[0]
        return None

    # 3) assume ETF/stock; iterate primary exchanges
    for pe in STK_PRIMARY_EXCHANGES:
        c = Stock(symbol=ticker, exchange="SMART", currency="USD", primaryExchange=pe)
        qc = ib.qualifyContracts(c)
        if qc:
            return qc[0]

    # 4) last-ditch: plain SMART
    c = Stock(symbol=ticker, exchange="SMART", currency="USD")
    qc = ib.qualifyContracts(c)
    if qc:
        return qc[0]

    return None


def fetch_daily_df(ib: IB, contract: Contract, duration: str, what_to_show: str, use_rth: bool) -> pd.DataFrame:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting="1 day",
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    df = util.df(bars).rename(columns={"date":"datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for k in ["open","high","low","close","volume"]:
        if k not in df.columns:
            df[k] = pd.NA
        df[k] = pd.to_numeric(df[k], errors="coerce")
    return df[["datetime","open","high","low","close","volume"]]


def should_skip_filename(fname: str) -> bool:
    name = fname.lower()
    return ("ratio" in name) or ("_x_" in name) or ("_spy_ratio" in name) or ("_composite" in name)


def ensure_tv_schema(df_new: pd.DataFrame) -> pd.DataFrame:
    out = df_new.copy()
    out["time"] = out["datetime"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["datetime"]).rename(columns={"volume":"Volume"})
    out = out[["time","open","high","low","close","Volume"]]
    return out


def update_one(path_csv: str, ib: IB, mapping: dict, duration: str, cli_what: str, use_rth: bool) -> str:
    base = os.path.basename(path_csv)
    m = re.match(r"^([A-Za-z0-9\.\-]+)_TV_daily\.csv$", base)
    if not m:
        return f"{base}: skip (name pattern mismatch)"
    ticker = m.group(1).upper()

    qc = qualify_with_fallbacks(ib, ticker, mapping)
    if not qc:
        return f"{ticker}: could not qualify (set mapping or check permissions)."

    sec_type = (qc.secType or "STK").upper()
    what_to_show = pick_what_to_show(sec_type, cli_what)

    df_new = fetch_daily_df(ib, qc, duration, what_to_show, use_rth)
    if df_new.empty:
        return f"{ticker} [{sec_type}/{what_to_show}]: no data returned."

    expected = ["time","open","high","low","close","Volume"]

    # init file if missing
    if not os.path.isfile(path_csv):
        out = ensure_tv_schema(df_new)
        out.to_csv(path_csv, index=False)
        return f"{ticker} [{sec_type}/{what_to_show}]: created file → {base} (+{len(out)} rows)"

    # append path
    old = pd.read_csv(path_csv)
    for col in expected:
        if col not in old.columns:
            return f"{ticker} [{sec_type}/{what_to_show}]: bad columns in {base} (needs {expected})."

    old["time"] = pd.to_datetime(old["time"], utc=True, errors="coerce")
    last_dt = old["time"].max()

    df_new2 = df_new[df_new["datetime"] > last_dt] if pd.notna(last_dt) else df_new

    merged = pd.concat(
        [
            old.rename(columns={"time":"datetime","Volume":"volume"})[
                ["datetime","open","high","low","close","volume"]
            ],
            df_new2,
        ],
        axis=0, ignore_index=True
    ).drop_duplicates(subset=["datetime"]).sort_values("datetime")

    out = ensure_tv_schema(merged)
    out.to_csv(path_csv, index=False)

    added = max(0, len(out) - len(old))
    return f"{ticker} [{sec_type}/{what_to_show}]: +{added} rows → {base}"


# =========================
# Ratios / Composite build
# =========================

def _read_tv_csv(dirpath: Path, ticker: str) -> pd.DataFrame:
    f = dirpath / f"{ticker}_TV_daily.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing input: {f}")
    df = pd.read_csv(f)
    expected = ["time", "open", "high", "low", "close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{f.name}: missing columns {missing}; expected {expected}")
    # Normalize
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["time", "open", "high", "low", "close"]].dropna(subset=["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def _write_tv_csv(df: pd.DataFrame, out_path: Path) -> None:
    out = df.copy()
    out = out.reset_index()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d")
    out["Volume"] = ""
    out = out[["time","open","high","low","close","Volume"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def _safe_div(n: pd.Series, d: pd.Series) -> pd.Series:
    return np.where((d.isna()) | (d == 0), np.nan, n / d)


def build_ratio(dirpath: Path, a: str, b: str) -> pd.DataFrame:
    A = _read_tv_csv(dirpath, a)
    B = _read_tv_csv(dirpath, b)
    merged = A.join(B, how="inner")
    out = pd.DataFrame(index=merged.index)
    for col in ["open","high","low","close"]:
        out[col] = _safe_div(merged[(a, col)], merged[(b, col)])
    out.index.name = "time"
    return out


def build_composite(dirpath: Path, members: List[str]) -> pd.DataFrame:
    dfs = [_read_tv_csv(dirpath, t) for t in members]
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d, how="inner")
    out = pd.DataFrame(index=merged.index)
    for col in ["open","high","low","close"]:
        cols = [(t, col) for t in members]
        out[col] = merged[cols].mean(axis=1)
    out.index.name = "time"
    return out


def rebuild_ratios_and_composites(srcdir: Path) -> None:
    # Ratios
    for spec in RATIO_SPECS:
        try:
            a, b = [x.strip().upper() for x in spec.split("/")]
        except Exception:
            print(f"[RATIO SKIP] Bad spec: {spec}")
            continue
        out_path = srcdir / f"{a}_{b}_ratio_TV_daily.csv"
        try:
            df = build_ratio(srcdir, a, b)
            _write_tv_csv(df, out_path)
            print(f"[RATIO] {a}/{b}: rows={len(df)} → {out_path.name}")
        except FileNotFoundError as e:
            print(f"[RATIO MISS] {e}")
        except Exception as e:
            print(f"[RATIO ERR] {a}/{b}: {e}")

    # Composites
    for name, members in COMPOSITES.items():
        out_path = srcdir / f"{name}_TV_daily.csv"
        try:
            df = build_composite(srcdir, [m.upper() for m in members])
            _write_tv_csv(df, out_path)
            print(f"[COMP ] {name}: rows={len(df)} → {out_path.name}")
        except FileNotFoundError as e:
            print(f"[COMP MISS] {e}")
        except Exception as e:
            print(f"[COMP ERR]  {name}: {e}")


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496)
    ap.add_argument("--client_id", type=int, default=63)
    ap.add_argument("--duration", default="10 Y")
    ap.add_argument("--what", default="", help="Override whatToShow (TRADES/MIDPOINT/etc). Empty = auto by secType")
    ap.add_argument("--use_rth", type=int, default=1)
    ap.add_argument("--srcdir", default=SRC_DIR)
    ap.add_argument("--list_xlsx", default=LIST_XLSX)
    ap.add_argument("--list_sheet", default=LIST_SHEET)
    ap.add_argument("--list_col", default=LIST_COL)
    ap.add_argument("--mapping", default=MAPPING_CSV)
    ap.add_argument("--sleep_ms", type=int, default=200)
    # NEW controls
    ap.add_argument("--ignore_list", type=int, default=0,
                    help="1 = process all *_TV_daily.csv files in srcdir (do not gate by ETF_list)")
    ap.add_argument("--only", nargs="*", default=[],
                    help="Whitelist of tickers to process (e.g. --only AIGA SPY)")
    args = ap.parse_args()

    srcdir = Path(args.srcdir)

    # Gating list
    if int(args.ignore_list):
        wanted = None
        print(f"[INFO] ignore_list=1 → processing ALL *_TV_daily.csv in {srcdir}")
    else:
        wanted = load_market_trend_tickers(args.list_xlsx, args.list_sheet, args.list_col)
        print(f"[INFO] Using list: {args.list_xlsx} | sheet={args.list_sheet} | tickers={len(wanted)}")

    only_set = {t.upper() for t in args.only} if args.only else set()
    if only_set:
        print(f"[INFO] Only processing: {sorted(only_set)}")

    mapping = load_mapping(args.mapping)

    ib = IB()
    print(f"[IB] Connecting {args.host}:{args.port} (clientId={args.client_id}) …")
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print("[IB] Connected.")

    # Part A — update/init raw TV CSVs
    for fname in sorted(os.listdir(srcdir)):
        if not fname.endswith("_TV_daily.csv"):
            continue
        if should_skip_filename(fname):
            print(f"{fname}: skip (ratio/composite)")
            continue

        tkr = fname.split("_TV_daily.csv")[0].upper()

        # --only whitelist first
        if only_set and tkr not in only_set:
            print(f"{fname}: skip (not in --only)")
            continue

        # Excel list gate (unless ignore_list=1)
        if (wanted is not None) and (tkr not in wanted):
            print(f"{fname}: skip (ticker not in ETF_list.xlsx → {args.list_sheet})")
            continue

        path = srcdir / fname
        try:
            msg = update_one(str(path), ib, mapping, args.duration, args.what, bool(args.use_rth))
            print(msg)
        except Exception as e:
            print(f"{fname}: ERROR {e}")
        ib.sleep(args.sleep_ms / 1000.0)

    ib.disconnect()
    print("[IB] Disconnected.")

    # Part B — rebuild ratios & composites
    rebuild_ratios_and_composites(srcdir)

    print("[DONE] s13_market_trends_update complete.")


if __name__ == "__main__":
    main()