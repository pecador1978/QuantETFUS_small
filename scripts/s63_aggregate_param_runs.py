#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s63_aggregate_param_runs.py — aggregate + dedupe + rank parameter-sweep results
from multiple runs, and (optionally) enrich with per-ticker data coverage stats
from an enriched prices parquet.

Clone-friendly: paths are obtained from common.paths.P so it works in
QuantETF and QuantETFUS_small without edits.

Outputs (written to P.PARAM_RESULTS):
  - s63_master_all_rows_YYYYMMDD_HHMM.csv
  - s63_combo_stability_YYYYMMDD_HHMM.csv
  - s63_topN_per_ticker_YYYYMMDD_HHMM.csv
  - s63_param_heatmap_YYYYMMDD_HHMM.csv
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---- single-threaded math (avoid oversubscription)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---- import shared paths (project-aware)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # ROOT, PARAM_RESULTS, CONFIG_DIR, DATA_ENRICHED, etc.

# ---------- defaults via P ----------
PARAM_DIR_DEFAULT = P.PARAM_RESULTS
CONFIG_DEFAULT    = P.CONFIG_DIR / "s60_parameters.json"
PARQUET_DEFAULT   = P.DATA_ENRICHED / "prices_enriched.parquet"
OUT_PREFIX = "s63"

# ---------- helpers ----------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _load_param_keys(config_path: Path) -> list[str]:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    # preserve insertion order as in JSON (Python 3.7+ dicts are ordered)
    return list(cfg.keys())

def _find_ranked_csvs(param_dir: Path, filename_contains: str | None) -> list[Path]:
    """Find per-ticker ranked files (exclude global and *_ALL.csv).
       If filename_contains is truthy, only include names containing it."""
    ranked: list[Path] = []
    for p in param_dir.glob("param_results_*_*.csv"):
        n = p.name
        if n.startswith("param_results_ALL_"):
            continue
        if n.endswith("_ALL.csv"):
            continue
        if filename_contains:
            if filename_contains not in n:
                continue
        ranked.append(p)
    return ranked

def _parse_ticker_and_suffix(path: Path) -> tuple[str, str]:
    body = path.stem.replace("param_results_", "", 1)
    if "_" not in body:
        return body, "UNKNOWN"
    ticker, suffix = body.split("_", 1)
    return ticker, suffix

def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def _as_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if len(s) else np.nan

def _safe_std(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.std(ddof=0)) if len(s) else np.nan

def _param_heatmap(stability_df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    if stability_df.empty or not param_cols:
        return pd.DataFrame(columns=["param","value","avg_mean_return"])
    rows = []
    for col in param_cols:
        if col not in stability_df.columns:
            continue
        g = (stability_df
             .groupby(col, dropna=False)["mean_return"]
             .mean()
             .reset_index()
             .rename(columns={col: "value", "mean_return": "avg_mean_return"}))
        g.insert(0, "param", col)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["param","value","avg_mean_return"])

def _load_coverage_from_parquet(parquet_path: Path,
                                ticker_col: str | None,
                                datetime_col: str | None) -> pd.DataFrame:
    if not parquet_path.exists():
        print(f"[WARN] Parquet not found: {parquet_path} — coverage columns will be empty.")
        return pd.DataFrame(columns=["ticker","first_data_date","coverage_years"])

    df = pd.read_parquet(parquet_path)

    tick_col = ticker_col or ("ticker" if "ticker" in df.columns else None)
    time_candidates = ["datetime", "date", "time", "timestamp"]
    dt_col = datetime_col or next((c for c in time_candidates if c in df.columns), None)

    if tick_col is None or dt_col is None:
        print(f"[WARN] Could not detect ticker/datetime columns in {parquet_path}. "
              f"Have columns: {list(df.columns)}. Coverage will be empty.")
        return pd.DataFrame(columns=["ticker","first_data_date","coverage_years"])

    dts = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
    dd = pd.DataFrame({"ticker": df[tick_col], "__dt": dts}).dropna(subset=["__dt"])
    if dd.empty:
        print("[WARN] No valid datetimes in parquet after coercion; coverage empty.")
        return pd.DataFrame(columns=["ticker","first_data_date","coverage_years"])

    cov = (dd.groupby("ticker")["__dt"]
             .agg(first_ts="min", last_ts="max")
             .reset_index())
    cov["first_data_date"] = cov["first_ts"]
    cov["coverage_years"] = (cov["last_ts"] - cov["first_ts"]).dt.total_seconds() / (365.25*24*3600)
    return cov[["ticker","first_data_date","coverage_years"]]

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--param_dir", type=str, default=str(PARAM_DIR_DEFAULT),
                    help="Directory containing per-ticker ranked CSVs from runs.")
    ap.add_argument("--config", type=str, default=str(CONFIG_DEFAULT),
                    help="Path to s60_parameters.json to detect parameter columns.")
    ap.add_argument("--top_n", type=int, default=5,
                    help="Top-N per ticker in aggregated ranking output.")
    ap.add_argument("--min_trades", type=int, default=0,
                    help="Drop rows with num_trades < min_trades before aggregation.")
    ap.add_argument("--filename_contains", type=str, default="",
                    help="Only ingest ranked files whose names contain this substring. "
                         "Use '' (empty) to include all.")
    ap.add_argument("--parquet_file", type=str, default=str(PARQUET_DEFAULT),
                    help="Parquet with enriched prices for coverage stats (optional).")
    ap.add_argument("--parquet_ticker_col", type=str, default=None)
    ap.add_argument("--parquet_datetime_col", type=str, default=None)
    args = ap.parse_args()

    param_dir   = Path(args.param_dir)
    config_path = Path(args.config)
    parquet_path= Path(args.parquet_file)
    ts = _ts()

    if not param_dir.exists():
        raise SystemExit(f"[ERR] param_dir not found: {param_dir}")
    if not config_path.exists():
        raise SystemExit(f"[ERR] config file not found: {config_path}")

    # 1) Parameter columns (from the grid json)
    param_cols = _load_param_keys(config_path)

    # 2) Per-ticker ranked CSVs
    ranked_files = _find_ranked_csvs(param_dir, args.filename_contains.strip() or None)
    if not ranked_files:
        raise SystemExit(
            f"[ERR] No ranked per-ticker CSVs matched in {param_dir} "
            f"(filter='{args.filename_contains}')."
        )
    print(f"[INFO] Scanning: {param_dir}")
    print(f"[INFO] Filename filter: '{args.filename_contains or '(disabled)'}'")
    print(f"[INFO] Matched {len(ranked_files)} file(s)")

    # 3) Load all ranked rows + metadata
    all_rows = []
    for p in sorted(ranked_files):
        try:
            tkr, suffix = _parse_ticker_and_suffix(p)
            df = pd.read_csv(p)
            if df is None or df.empty:
                continue

            # normalize numeric types
            for col in ("total_return","win_rate","end_capital"):
                if col in df.columns:
                    df[col] = _as_float(df[col])
            if "num_trades" in df.columns:
                df["num_trades"] = _as_int(df["num_trades"])
            if "rank" in df.columns:
                df["rank"] = _as_int(df["rank"])

            if args.min_trades and "num_trades" in df.columns:
                df = df[df["num_trades"].fillna(0) >= args.min_trades]
                if df.empty:
                    continue

            df["ticker"] = tkr
            df["run_id"] = suffix
            df["source_file"] = p.name
            all_rows.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    if not all_rows:
        raise SystemExit("[ERR] No valid ranked rows loaded after filtering.")

    master = pd.concat(all_rows, ignore_index=True)

    out_master = param_dir / f"{OUT_PREFIX}_master_all_rows_{ts}.csv"
    master.to_csv(out_master, index=False)
    print(f"[OK] Wrote master rows → {out_master}")

    # 4) Ensure param columns exist
    for c in param_cols:
        if c not in master.columns:
            master[c] = np.nan

    # 5) Stability by (ticker + full param set) — robust to NaNs
    grp_keys = ["ticker"] + param_cols
    gobj = master.groupby(grp_keys, dropna=False)

    stability = gobj.agg(
        count_runs      = ("run_id", "nunique"),
        observations    = ("run_id", "size"),
        mean_return     = ("total_return", _safe_mean),
        median_return   = ("total_return", "median"),
        best_return     = ("total_return", "max"),
        worst_return    = ("total_return", "min"),
        std_return      = ("total_return", _safe_std),
        mean_win_rate   = ("win_rate", _safe_mean),
        mean_num_trades = ("num_trades", _safe_mean),
        run_ids         = ("run_id", lambda s: ",".join(sorted(pd.Series(s).astype(str).unique()))),
    ).reset_index()

    # 6) Optional: robustness score for ranking
    # Higher better: mean_return, count_runs, mean_num_trades; lower better: std_return
    stability["score"] = (
        stability["mean_return"].fillna(-1e9)
        + 0.05  * stability["count_runs"].fillna(0)
        - 0.25  * stability["std_return"].fillna(0)
        + 0.0005* stability["mean_num_trades"].fillna(0)
    )

    # 7) Best/top-N per ticker
    stability_sorted = stability.sort_values(
        by=["ticker", "score", "mean_return", "count_runs", "mean_num_trades", "mean_win_rate"],
        ascending=[True, False, False, False, False, False]
    )
    topN_per_ticker = (stability_sorted
                       .groupby("ticker", as_index=False)
                       .head(max(int(args.top_n), 1))
                       .reset_index(drop=True))
    topN_per_ticker["rank"] = (
        topN_per_ticker.groupby("ticker")["score"].rank(method="first", ascending=False).astype(int)
    )

    # 8) Param heatmap (marginal effect of each param value)
    heatmap = _param_heatmap(stability, param_cols)

    # 9) Coverage from parquet (optional)
    coverage = _load_coverage_from_parquet(
        parquet_path=parquet_path,
        ticker_col=args.parquet_ticker_col,
        datetime_col=args.parquet_datetime_col,
    )
    if not coverage.empty:
        for dfname, d in (("stability", stability),
                          ("topN_per_ticker", topN_per_ticker)):
            merged = d.merge(coverage, on="ticker", how="left")
            if dfname == "stability":
                stability = merged
            else:
                topN_per_ticker = merged

    # 10) Save outputs
    out_stab = param_dir / f"{OUT_PREFIX}_combo_stability_{ts}.csv"
    out_topN = param_dir / f"{OUT_PREFIX}_topN_per_ticker_{ts}.csv"
    out_heat = param_dir / f"{OUT_PREFIX}_param_heatmap_{ts}.csv"

    stability.to_csv(out_stab, index=False)
    topN_per_ticker.to_csv(out_topN, index=False)
    heatmap.to_csv(out_heat, index=False)

    print(f"[OK] Wrote stability     → {out_stab}")
    print(f"[OK] Wrote topN/ticker   → {out_topN}")
    print(f"[OK] Wrote param heatmap → {out_heat}")
    print(f"[OK] Wrote master rows   → {out_master}")

if __name__ == "__main__":
    main()