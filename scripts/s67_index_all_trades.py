#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s67_index_all_trades.py — Build a UNIQUE trades index from ALL trades files (no champion tagging).

- Scans: <PARAM_RESULTS>/trades/<TICKER>/trades_*.csv
- Infers ticker from folder / filename if column is missing.
- Collapses fills -> 1 row per trade (per entry_dt).
- Dedupes using a stable hash (ticker|entry_dt|exit_dt|legs).

Outputs (timestamped):
  <ROOT>/merged_datasets/s67_index_<ts>.csv
  <ROOT>/merged_datasets/s67_index_<ts>.parquet
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import glob
import hashlib
import sys

import numpy as np
import pandas as pd

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # ROOT, PARAM_RESULTS, etc.

PARAM_RESULTS_DEFAULT = P.PARAM_RESULTS
TRADES_ROOT_DEFAULT   = P.PARAM_RESULTS / "trades"
OUT_DIR_DEFAULT       = P.ROOT / "merged_datasets"
OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _infer_ticker_from_path(path: Path) -> str | None:
    # prefer folder name
    parent = path.parent.name.strip()
    if parent and parent.lower() != "trades":
        return parent
    # fallback: trades_<TICKER>_combo...
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].lower() == "trades":
        return parts[1]
    return None

def _read_csv_minimal(path: Path) -> pd.DataFrame | None:
    # Read minimally and standardize columns
    try:
        probe = pd.read_csv(path, nrows=0)
        if probe is None:
            return None
        cols = list(probe.columns.astype(str))
        want = ["ticker","entry_dt","exit_dt","leg","qty","ret_pct","fee_eur"]
        usecols = [c for c in cols if c in want] or None
        df = pd.read_csv(path, usecols=usecols)
        if df is None or df.empty:
            return None

        if "ticker" not in df.columns:
            tkr = _infer_ticker_from_path(path)
            if not tkr:
                return None
            df.insert(0, "ticker", tkr)

        df["ticker"] = df["ticker"].astype(str)
        if "leg" not in df.columns:
            df["leg"] = pd.NA
        else:
            df["leg"] = df["leg"].astype(str)

        for c in ("entry_dt","exit_dt"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        for c in ("qty","ret_pct","fee_eur"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "entry_dt" not in df.columns or df["entry_dt"].isna().all():
            return None

        keep = [c for c in ["ticker","entry_dt","exit_dt","leg","qty","ret_pct","fee_eur"] if c in df.columns]
        return df[keep]
    except Exception:
        return None

def _collapse_fills_to_trade(f: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple fills belonging to the same trade (same ticker+entry_dt)."""
    if f is None or f.empty or "ticker" not in f.columns or "entry_dt" not in f.columns:
        return pd.DataFrame(columns=["ticker","entry_dt","exit_dt","ret_trade","pnl_eur","num_fills","legs"])

    f = f.copy()
    f["entry_dt"] = pd.to_datetime(f["entry_dt"], utc=True, errors="coerce")
    f["trade_id"] = f["ticker"].astype(str) + "|" + f["entry_dt"].astype(str)

    def _agg(g):
        gross = (pd.to_numeric(g.get("ret_pct", 0.0), errors="coerce") * pd.to_numeric(g.get("qty", 0.0), errors="coerce")).sum()
        fees  = pd.to_numeric(g.get("fee_eur", 0.0), errors="coerce").fillna(0.0).sum()
        unit_capital = 10_000.0
        ret_trade = gross - (fees / unit_capital)
        pnl_eur = ret_trade * unit_capital
        legs = ",".join(sorted(set(g["leg"].astype(str).dropna().tolist())))
        return pd.Series({
            "ticker": str(g["ticker"].iloc[0]),
            "entry_dt": pd.to_datetime(g["entry_dt"].min(), utc=True, errors="coerce"),
            "exit_dt":  pd.to_datetime(g.get("exit_dt").max(), utc=True, errors="coerce") if "exit_dt" in g.columns else pd.NaT,
            "ret_trade": ret_trade,
            "pnl_eur": pnl_eur,
            "num_fills": len(g),
            "legs": legs
        })

    return f.groupby("trade_id", as_index=False).apply(_agg, include_groups=False).reset_index(drop=True)

def _hash_trade_key(ticker: str, entry_dt: pd.Timestamp, exit_dt: pd.Timestamp | None, legs: str | None) -> str:
    ent = "" if pd.isna(entry_dt) else entry_dt.isoformat()
    ex  = "" if (exit_dt is None or pd.isna(exit_dt)) else exit_dt.isoformat()
    lg  = "" if legs is None else str(legs)
    raw = f"{ticker}|{ent}|{ex}|{lg}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Index ALL trades into a unique, deduped master list.")
    ap.add_argument("--trades_root", type=str, default=str(TRADES_ROOT_DEFAULT),
                    help="Directory with per-ticker trades: PARAM_RESULTS/trades/<TICKER>/trades_*.csv")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    trades_root = Path(args.trades_root)
    out_dir     = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # All candidate files
    files = [Path(p) for p in glob.glob(str(trades_root / "*" / "trades_*.csv"))]
    print(f"[s67-index] Scanning {len(files):,} trades files under {trades_root}")

    uniq: dict[str, dict] = {}
    skipped = 0
    processed = 0

    for p in files:
        processed += 1
        df = _read_csv_minimal(p)
        if df is None or df.empty:
            skipped += 1
            continue

        tdf = _collapse_fills_to_trade(df)
        if tdf is None or tdf.empty:
            skipped += 1
            continue

        for _, r in tdf.iterrows():
            h = _hash_trade_key(str(r["ticker"]), r["entry_dt"], r.get("exit_dt"), r.get("legs"))
            if h not in uniq:
                uniq[h] = {
                    "ticker": str(r["ticker"]),
                    "entry_dt": r["entry_dt"],
                    "exit_dt":  r.get("exit_dt"),
                    "ret_trade": float(r.get("ret_trade", np.nan)),
                    "pnl_eur":   float(r.get("pnl_eur",   np.nan)),
                    "num_fills": int(r.get("num_fills", 0)),
                    "legs":      str(r.get("legs", "")),
                    "source_file": p.name,
                }

        if processed % 10000 == 0:
            print(f"[s67-index] …processed {processed:,} files, unique trades so far: {len(uniq):,} (skipped {skipped:,})")

    if not uniq:
        raise SystemExit("[ERR] No valid trades found to index.")

    idx = pd.DataFrame(list(uniq.values())).sort_values(["ticker","entry_dt"]).reset_index(drop=True)

    ts = _ts()
    index_csv = out_dir / f"s67_index_{ts}.csv"
    index_pq  = out_dir / f"s67_index_{ts}.parquet"
    idx.to_csv(index_csv, index=False)
    idx.to_parquet(index_pq, index=False)

    print(f"[OK] Index → {index_csv}  (unique trades: {len(idx):,}; files skipped: {skipped:,})")

if __name__ == "__main__":
    main()