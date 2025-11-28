#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s36_coverage_report.py — Coverage of enriched DAILY parquet and 30m parquet layer + reconciliation.

Outputs (to P.REPORTS_DIR):
- s36_first_dates_daily_<ts>.csv
- s36_30m_catalog_<ts>.csv
- s36_reconcile_m30_vs_daily_<ts>.csv

Usage:
  python scripts/s36_coverage_report.py
  python scripts/s36_coverage_report.py --parquet /path/to/prices_enriched.parquet --m30_dir /path/to/30min
  python scripts/s36_coverage_report.py --ticker_col ticker --dt_col datetime
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys
import argparse
import pandas as pd

# --- import centralized paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa

DEFAULT_PARQUET = P.DATA_ENRICHED / "prices_enriched.parquet"
DEFAULT_M30_DIR = P.DATA_ENRICHED / "30min"
OUTDIR = P.REPORTS_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

def _tsnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _auto_detect_cols(df: pd.DataFrame, ticker_col: str | None, dt_col: str | None) -> tuple[str, str]:
    cols_lower = {c.lower(): c for c in df.columns}

    # ticker column
    if ticker_col:
        if ticker_col in df.columns:
            tcol = ticker_col
        elif ticker_col.lower() in cols_lower:
            tcol = cols_lower[ticker_col.lower()]
        else:
            raise SystemExit(f"[ERR] --ticker_col '{ticker_col}' not found. Columns: {list(df.columns)}")
    else:
        for cand in ("ticker", "symbol", "code", "tkr", "instrument"):
            if cand in cols_lower:
                tcol = cols_lower[cand]
                break
        else:
            raise SystemExit("[ERR] Could not detect ticker column (looked for: ticker/symbol/...).")

    # datetime column
    if dt_col:
        if dt_col in df.columns:
            dcol = dt_col
        elif dt_col.lower() in cols_lower:
            dcol = cols_lower[dt_col.lower()]
        else:
            raise SystemExit(f"[ERR] --dt_col '{dt_col}' not found. Columns: {list(df.columns)}")
    else:
        for cand in ("dt", "datetime", "timestamp", "time", "date", "datetime_utc", "ts"):
            if cand in cols_lower:
                dcol = cols_lower[cand]
                break
        else:
            raise SystemExit("[ERR] Could not detect datetime column (looked for: dt/datetime/timestamp/...).")

    return tcol, dcol

def _normalize_ticker_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace("\u00A0", " ", regex=False)  # NBSP
         .str.replace(r"[^A-Za-z0-9._-]", "", regex=True)
         .str.upper()
    )

def _first_dates_daily(parquet_path: Path, ticker_col: str | None, dt_col: str | None) -> pd.DataFrame:
    if not parquet_path.exists():
        raise SystemExit(
            f"[ERR] Missing parquet file: {parquet_path}\n"
            "      Run daily enrich first (e.g., s30) to produce prices_enriched.parquet."
        )
    print(f"[INFO] Loading DAILY parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    tcol, dcol = _auto_detect_cols(df, ticker_col, dt_col)

    df[tcol] = _normalize_ticker_series(df[tcol])
    df[dcol] = pd.to_datetime(df[dcol], utc=True, errors="coerce")

    grp = df.groupby(tcol, dropna=False)[dcol]
    out = (
        pd.DataFrame({
            "first_dt_utc": grp.min(),
            "last_dt_utc":  grp.max(),
            "rows":         grp.size()
        })
        .reset_index()
        .rename(columns={tcol: "ticker"})
        .sort_values("first_dt_utc", kind="mergesort")
        .reset_index(drop=True)
    )
    return out

def _catalog_30m_parquets(m30_dir: Path) -> pd.DataFrame:
    if not m30_dir.exists():
        print(f"[WARN] 30m directory not found: {m30_dir}")
        return pd.DataFrame(columns=["ticker", "first_dt_utc", "last_dt_utc", "rows", "path"])

    files = sorted(m30_dir.glob("*.parquet"))
    if not files:
        print(f"[WARN] No 30m parquets in: {m30_dir}")
        return pd.DataFrame(columns=["ticker", "first_dt_utc", "last_dt_utc", "rows", "path"])

    rows = []
    for p in files:
        tkr = p.stem.strip().upper()
        try:
            # Load only datetime column to keep it light
            dts = pd.read_parquet(p, columns=["datetime"])
            dts = pd.to_datetime(dts["datetime"], utc=True, errors="coerce")
            rows.append({
                "ticker": tkr,
                "first_dt_utc": dts.min(),
                "last_dt_utc":  dts.max(),
                "rows": int(len(dts)),
                "path": str(p)
            })
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
            rows.append({
                "ticker": tkr,
                "first_dt_utc": pd.NaT,
                "last_dt_utc":  pd.NaT,
                "rows": 0,
                "path": str(p)
            })
    out = pd.DataFrame(rows).sort_values(["ticker"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET),
                    help="Path to enriched DAILY parquet (default: DATA_ENRICHED/prices_enriched.parquet)")
    ap.add_argument("--m30_dir", type=str, default=str(DEFAULT_M30_DIR),
                    help="Directory with 30m parquets (default: DATA_ENRICHED/30min)")
    ap.add_argument("--ticker_col", type=str, default=None, help="Explicit DAILY ticker column name")
    ap.add_argument("--dt_col", type=str, default=None, help="Explicit DAILY datetime column name")
    args = ap.parse_args()

    daily_path = Path(args.parquet)
    m30_dir    = Path(args.m30_dir)

    # DAILY coverage
    daily_cov = _first_dates_daily(daily_path, args.ticker_col, args.dt_col)
    ts = _tsnow()
    daily_csv = OUTDIR / f"s36_first_dates_daily_{ts}.csv"
    daily_cov.to_csv(daily_csv, index=False)
    print(f"[OK] DAILY coverage → {daily_csv} (tickers={len(daily_cov)})")

    # 30m catalog
    m30_cov = _catalog_30m_parquets(m30_dir)
    m30_csv = OUTDIR / f"s36_30m_catalog_{ts}.csv"
    m30_cov.to_csv(m30_csv, index=False)
    print(f"[OK] 30m catalog → {m30_csv} (tickers={len(m30_cov)})")

    # Reconciliation
    daily_tickers = set(_normalize_ticker_series(daily_cov["ticker"]))
    m30_tickers   = set(_normalize_ticker_series(m30_cov["ticker"])) if len(m30_cov) else set()

    only_in_30m = sorted(list(m30_tickers - daily_tickers))
    only_in_daily = sorted(list(daily_tickers - m30_tickers))

    recon_rows = []
    for t in only_in_30m:
        recon_rows.append({"ticker": t, "status": "in_30m_only"})
    for t in only_in_daily:
        recon_rows.append({"ticker": t, "status": "in_daily_only"})

    if recon_rows:
        recon = (
            pd.DataFrame(recon_rows)
            .sort_values(["status", "ticker"])
            .reset_index(drop=True)
        )
    else:
        # keep a proper CSV with headers even when perfectly reconciled
        recon = pd.DataFrame(columns=["ticker", "status"])

    recon_csv = OUTDIR / f"s36_reconcile_m30_vs_daily_{ts}.csv"
    recon.to_csv(recon_csv, index=False)

    msg = (f"[OK] Reconciliation → {recon_csv} "
        f"(30m_only={len(only_in_30m)}, daily_only={len(only_in_daily)})")
    if recon.empty:
        msg += " — perfect match (no differences)."
    print(msg)

    # Pretty preview
    try:
        print("\n[Preview] DAILY coverage (head):")
        print(daily_cov.head(12).to_string(index=False))
        if len(m30_cov):
            print("\n[Preview] 30m catalog (head):")
            print(m30_cov.head(12).to_string(index=False))
        if len(recon):
            print("\n[Preview] Reconciliation (up to 20 rows):")
            print(recon.head(20).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()