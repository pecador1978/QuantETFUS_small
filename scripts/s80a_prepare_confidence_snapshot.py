#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s80a_prepare_confidence_snapshot.py — Summarize latest Gate-2 confidence per ticker
-----------------------------------------------------------------------------------
- Mirrors s80 defaults (PROJECT_ROOT, paths).
- Reads gate2_confidence_30m.csv (from s80). If missing in project, auto-finds the
  newest gate2_confidence_30m*.csv in QuantShared.
- Builds 1-row-per-ticker snapshot with:
    * latest confidence_score + timestamp
    * percentile vs full history
    * z-score vs full history
    * peer rank and decile for the latest bar
- Writes snapshot CSV (default: alongside the input, or --output if provided).
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# -------- Defaults aligned with s80 --------
PROJECT_ROOT   = Path("/Users/Finance/QuantETFUS_small")
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "data_enriched" / "30min"  # not used here, kept for symmetry
DEFAULT_GLOB       = "*.parquet"                               # not used here, kept for symmetry
DEFAULT_OUTPUT_S80 = PROJECT_ROOT / "data_enriched" / "gate2_confidence_30m.csv"

# QuantShared fallback root
QUANTSHARED_ROOT = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs" / "QuantShared"

def find_input_csv(user_input: str | None) -> Path:
    """Resolve input CSV path: --input > project default > QuantShared newest."""
    if user_input:
        p = Path(user_input).expanduser()
        if not p.exists():
            raise SystemExit(f"[ERR] --input not found: {p}")
        return p

    # 1) Project default (s80 default output)
    if DEFAULT_OUTPUT_S80.exists():
        return DEFAULT_OUTPUT_S80

    # 2) QuantShared fallback: newest matching file
    candidates: list[Path] = []
    qs_dir = QUANTSHARED_ROOT / "data_enriched"
    if qs_dir.exists():
        candidates += list(qs_dir.glob("gate2_confidence_30m*.csv"))
        candidates += list(qs_dir.glob("gate2_confidence_30m*.csv.gz"))

    if not candidates:
        raise SystemExit(
            "[ERR] No gate2_confidence_30m*.csv found.\n"
            f"  Checked: {DEFAULT_OUTPUT_S80}\n"
            f"  And: {qs_dir}\n"
            "Tip: run s80_confidence_gate2.py first, or pass --input /full/path.csv"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def pct_rank(arr: np.ndarray, v: float) -> float:
    if arr.size == 0 or not np.isfinite(v):
        return np.nan
    return float((np.sum(arr <= v) / arr.size) * 100.0)

def zscore(arr: np.ndarray, v: float) -> float:
    if arr.size < 2 or not np.isfinite(v):
        return np.nan
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return np.nan
    return (v - mu) / sd

def build_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    # dtypes & clean
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "ticker", "confidence_score"])
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")

    rows = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("datetime")
        if len(g) < 5:
            continue
        last = g.iloc[-1]
        v = float(last["confidence_score"])
        hist = g["confidence_score"].to_numpy(dtype=float)

        rows.append({
            "ticker": str(tkr),
            "datetime_latest": last["datetime"],
            "confidence_latest": v,
            "pct_hist": pct_rank(hist, v),
            "z_hist": zscore(hist, v),
            "n_obs": int(hist.size),
        })

    snap = pd.DataFrame(rows)
    if snap.empty:
        raise SystemExit("[ERR] No snapshots produced (insufficient history).")

    # Peer rank/decile based on latest confidence
    snap = snap.sort_values("confidence_latest", ascending=False).reset_index(drop=True)
    snap["rank_today"] = snap["confidence_latest"].rank(ascending=False, method="min").astype(int)
    snap["decile_today"] = pd.qcut(
        snap["confidence_latest"].rank(method="first"),
        10, labels=False, duplicates="drop"
    ) + 1

    # Column order
    cols = [
        "ticker","datetime_latest","confidence_latest",
        "pct_hist","z_hist","rank_today","decile_today","n_obs"
    ]
    return snap[cols]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare Gate-2 historical snapshot for dashboard.")
    ap.add_argument("--input", help="Optional explicit path to gate2_confidence_30m*.csv or .csv.gz")
    ap.add_argument("--output", help="Optional output CSV path (default: alongside input as gate2_confidence_snapshot.csv)")
    return ap.parse_args()

def main():
    args = parse_args()
    src = find_input_csv(args.input)
    print(f"[INFO] Using input: {src}")

    # Read only required columns
    need = ["datetime","ticker","confidence_score"]
    header = pd.read_csv(src, nrows=0)
    usecols = [c for c in need if c in header.columns]
    if len(usecols) < 3:
        raise SystemExit("[ERR] Input CSV missing required columns: datetime, ticker, confidence_score")

    df = pd.read_csv(src, usecols=usecols, low_memory=False)

    snap = build_snapshot(df)

    # Output path
    out = Path(args.output).expanduser() if args.output else (src.parent / "gate2_confidence_snapshot.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    snap.to_csv(out, index=False, float_format="%.6f")
    print(f"✅ Snapshot saved → {out} | tickers={snap['ticker'].nunique()}")

if __name__ == "__main__":
    main()