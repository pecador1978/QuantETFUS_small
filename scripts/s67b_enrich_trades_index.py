#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s67b_enrich_trades_index.py — Enrich a UNIQUE trades index with features (vectorized, lag-safe).

Inputs:
  - latest or given: <ROOT>/merged_datasets/s67_index_*.csv   (from s67_index_all_trades.py)

Feature sources (all optional; merged if present):
  - ETF daily (wide by ticker, *_d cols):     <ROOT>/data_enriched/prices_enriched.parquet
  - Macro/FX (wide, 1 row per date):          <ROOT>/data_enriched/macro_forex_enriched.parquet
  - 30m context per ticker parquet files:     <ROOT>/data_enriched/30min/{TICKER}.parquet

Key safeguards:
  - ETF-daily features are merged with a configurable lag (--daily_lag_days, default 1)
    so entries on day D only see daily data from D-1 (prevents lookahead).
  - Any columns with name containing 'vermeulen' (case-insensitive) are dropped.

Outputs:
  <ROOT>/merged_datasets/trades_enriched_<ts>.csv
  <ROOT>/merged_datasets/trades_enriched_<ts>.parquet
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import sys
import numpy as np
import pandas as pd

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P

OUT_DIR_DEFAULT      = P.ROOT / "merged_datasets"
ETF_DAILY_PQ_DEFAULT = P.DATA_ENRICHED / "prices_enriched.parquet"
MACRO_PQ_DEFAULT     = P.DATA_ENRICHED / "macro_forex_enriched.parquet"
M30_DIR_DEFAULT      = P.DATA_ENRICHED / "30min"

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _latest_index() -> Path | None:
    cands = sorted((P.ROOT / "merged_datasets").glob("s67_index_*.csv"))
    return cands[-1] if cands else None

def _load_etf_daily(path: Path) -> pd.DataFrame | None:
    """Return daily features with columns: ['ticker','date', *_d ...] or None if missing."""
    if not path.exists():
        print(f"[WARN] ETF daily parquet not found: {path} (skipping ETF daily).")
        return None
    df = pd.read_parquet(path)

    # detect ticker/datetime columns (case-insensitive)
    low = {c.lower(): c for c in df.columns}
    tcol = next((low[k] for k in ["ticker","symbol","tkr","asset","instrument","code"] if k in low), None)
    dcol = next((low[k] for k in ["datetime","dt","date","timestamp","time","ts","datetime_utc","time_utc"] if k in low), None)
    if not tcol or not dcol:
        print(f"[WARN] ETF daily parquet missing ticker/datetime columns: {path} (skipping).")
        return None

    df = df.rename(columns={tcol:"ticker", dcol:"datetime"}).copy()
    df["ticker"] = df["ticker"].astype(str)
    df["date"]   = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.floor("D")
    keep = ["ticker","date"] + [c for c in df.columns if str(c).endswith("_d")]
    df = df[keep].drop_duplicates(subset=["ticker","date"]).sort_values(["ticker","date"])
    return df

def _macro_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long macro (date,ticker,value-cols) → wide by date."""
    dtcand = next((c for c in ["date","datetime","dt","day"] if c in df_long.columns), None)
    if not dtcand or "ticker" not in df_long.columns:
        raise ValueError("Macro long must have a date-like column and 'ticker'")
    df = df_long.rename(columns={dtcand:"date"}).copy()
    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.floor("D")
    df["ticker"] = df["ticker"].astype(str)

    feat_cols = [c for c in df.columns if c not in ("date","ticker")]
    frames = []
    for f in feat_cols:
        piv = df.pivot_table(index="date", columns="ticker", values=f, aggfunc="last")
        piv = piv.rename(columns=lambda t: f"{t}_{f}")
        frames.append(piv)
    wide = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df["date"].drop_duplicates().sort_values())
    wide = wide.sort_index().reset_index()
    return wide

def _load_macro(path: Path) -> pd.DataFrame | None:
    """Return macro-wide with 'date' as the key, or None if not available."""
    if not path.exists():
        print(f"[WARN] Macro parquet not found: {path} (skipping macro).")
        return None
    m = pd.read_parquet(path)

    # already wide by date?
    if "date" in m.columns and "ticker" not in m.columns:
        out = m.copy()
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.floor("D")
        out = out.sort_values("date").drop_duplicates(subset=["date"])
        return out

    # long → wide
    if "ticker" in m.columns:
        try:
            return _macro_long_to_wide(m)
        except Exception as e:
            print(f"[WARN] Macro long->wide failed: {e} (skipping macro).")
            return None

    print(f"[WARN] Macro parquet unexpected schema; skipping.")
    return None

def _load_all_m30(m30_dir: Path, tickers: list[str]) -> pd.DataFrame | None:
    """Load all available 30m parquet files for the given tickers and return one concat dataframe:
       columns: ['ticker','dt', m30_* features]."""
    frames = []
    for tkr in sorted(set(tickers)):
        f = m30_dir / f"{tkr}.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        dtcand = next((c for c in ["datetime","dt","time","timestamp","ts"] if c in df.columns), None)
        if not dtcand:
            continue
        df = df.rename(columns={dtcand:"dt"}).copy()
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        df = df.sort_values("dt")
        rename = {c: (c if str(c).startswith("m30_") else f"m30_{c}") for c in df.columns if c != "dt"}
        df = df.rename(columns=rename)
        df.insert(0, "ticker", tkr)
        frames.append(df[["ticker","dt"] + [c for c in df.columns if c not in ("ticker","dt")]])
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["dt"]).sort_values(["ticker","dt"])
    return out

def main():
    ap = argparse.ArgumentParser(description="Enrich a UNIQUE trades index with ETF/Macro/30m features (vectorized, lag-safe).")
    ap.add_argument("--index", type=str, default=None,
                    help="Path to s67_index_*.csv (default: latest in merged_datasets)")
    ap.add_argument("--etf_daily", type=str, default=str(ETF_DAILY_PQ_DEFAULT))
    ap.add_argument("--macro", type=str, default=str(MACRO_PQ_DEFAULT))
    ap.add_argument("--m30_dir", type=str, default=str(M30_DIR_DEFAULT))
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--skip_macro", action="store_true", default=False)
    ap.add_argument("--skip_m30", action="store_true", default=False)
    ap.add_argument("--limit_tickers", type=int, default=0, help="For testing: limit to first N tickers.")
    ap.add_argument("--daily_lag_days", type=int, default=1,
                    help="Lag ETF-daily features by N days before merging (default 1).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    index_path = Path(args.index) if args.index else _latest_index()
    if not index_path or not index_path.exists():
        raise SystemExit("[ERR] Index not found. Run s67_index_all_trades.py first.")

    print(f"[s67b] Enriching index: {index_path.name}")

    idx = pd.read_csv(index_path)
    if idx.empty:
        raise SystemExit("[ERR] Index is empty.")
    # standardize times/keys
    idx["entry_dt"] = pd.to_datetime(idx["entry_dt"], utc=True, errors="coerce")
    idx["exit_dt"]  = pd.to_datetime(idx.get("exit_dt"),  utc=True, errors="coerce")
    idx["date"]     = idx["entry_dt"].dt.floor("D")
    idx = idx.dropna(subset=["ticker","entry_dt"]).copy()
    idx["ticker"] = idx["ticker"].astype(str)

    # Optional: limit tickers for quicker test runs
    if args.limit_tickers and args.limit_tickers > 0:
        keep = sorted(idx["ticker"].unique())[: args.limit_tickers]
        idx = idx[idx["ticker"].isin(keep)].copy()
        print(f"[s67b] Limiting to first {len(keep)} tickers: {keep}")

    # ----- ETF daily (merge on (ticker, date_lagged)) -----
    etf_daily = _load_etf_daily(Path(args.etf_daily))
    if etf_daily is not None and not etf_daily.empty:
        # make a lagged date key on the INDEX side (so we don't touch source parquet)
        if args.daily_lag_days and args.daily_lag_days > 0:
            idx["date_for_daily"] = (idx["date"] - pd.to_timedelta(args.daily_lag_days, unit="D")).dt.floor("D")
        else:
            idx["date_for_daily"] = idx["date"]

        etf_daily = etf_daily.rename(columns={"date": "date_for_daily"})
        idx = idx.merge(etf_daily, on=["ticker","date_for_daily"], how="left")
        idx = idx.drop(columns=["date_for_daily"], errors="ignore")
        print(f"[s67b] Merged ETF daily features (lag={args.daily_lag_days}): "
              f"{len([c for c in idx.columns if str(c).endswith('_d')])} columns.")
    else:
        print("[s67b] ETF daily features skipped or empty.")

    # ----- Macro (as-of join on date) -----
    if not args.skip_macro:
        macro_wide = _load_macro(Path(args.macro))
        if macro_wide is not None and not macro_wide.empty:
            macro_wide = macro_wide.sort_values("date")
            idx = pd.merge_asof(
                idx.sort_values("date"),
                macro_wide.sort_values("date"),
                on="date",
                direction="backward",
                allow_exact_matches=True,
            ).sort_values(["ticker","entry_dt"]).reset_index(drop=True)
            print(f"[s67b] Merged macro features: {len(macro_wide.columns) - 1} columns.")
        else:
            print("[s67b] Macro features skipped or empty.")
    else:
        print("[s67b] Macro merge skipped by flag.")

    # ----- 30m context per ticker (as-of join on entry_dt by ticker) -----
    if not args.skip_m30:
        m30_all = _load_all_m30(Path(args.m30_dir), idx["ticker"].unique().tolist())
        if m30_all is not None and not m30_all.empty:
            idx = idx.sort_values(["ticker","entry_dt"]).reset_index(drop=True)
            m30_all = m30_all.sort_values(["ticker","dt"]).reset_index(drop=True)
            idx = pd.merge_asof(
                idx,
                m30_all.rename(columns={"dt":"m30_dt"}),
                left_on="entry_dt", right_on="m30_dt",
                by="ticker",
                direction="backward",
                allow_exact_matches=True,
            )
            idx = idx.drop(columns=["m30_dt"], errors="ignore")
            m30_added = [c for c in idx.columns if str(c).startswith("m30_")]
            print(f"[s67b] Merged 30m features: {len(m30_added)} columns.")
        else:
            print("[s67b] No 30m parquet features found; skipping 30m.")
    else:
        print("[s67b] 30m merge skipped by flag.")

    # ---- Drop any Vermeulen-related columns ----
    verm_cols = [c for c in idx.columns if "vermeulen" in str(c).lower()]
    if verm_cols:
        idx = idx.drop(columns=verm_cols, errors="ignore")
        print(f"[s67b] Dropped Vermeulen columns: {len(verm_cols)}")

    # Drop all-NaN feature cols (purely empty)
    empty_cols = idx.columns[idx.isna().all(axis=0)]
    if len(empty_cols):
        idx = idx.drop(columns=list(empty_cols))
        print(f"[s67b] Dropped all-NaN columns: {len(empty_cols)}")

    ts = _ts()
    trades_csv = out_dir / f"trades_enriched_{ts}.csv"
    trades_pq  = out_dir / f"trades_enriched_{ts}.parquet"

    idx.to_csv(trades_csv, index=False)
    try:
        idx.to_parquet(trades_pq, index=False)
    except Exception:
        pass

    print(f"[OK] Trades enriched → {trades_csv}  (rows: {len(idx):,})")

if __name__ == "__main__":
    main()