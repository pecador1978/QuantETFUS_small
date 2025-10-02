#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s63a_pick_top5_distinct_roi.py
Pick Top-N per ticker by ROI (total_return) across ALL sweeps,
but ensure ROI values are distinct within each ticker (dedupe on ROI).

- Scans P.PARAM_RESULTS for ranked per-ticker CSVs named:
    param_results_{TICKER}_{SUFFIX}.csv
  Skips:
    - param_results_ALL_*.csv (global boards)
    - *_ALL.csv (unfiltered per-ticker dumps)

- Within each ticker:
    1) Sort by ROI desc, then num_trades desc, then win_rate desc
    2) Dedupe by rounded ROI (tolerance configurable) to get distinct ROI levels
    3) Take Top-N and re-rank 1..N

Outputs → P.PARAM_RESULTS/s63a_topN_distinct_roi_{TS}.csv
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, os, sys, json
import pandas as pd
import numpy as np

# single-threaded math
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# project-aware paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # type: ignore

PARAM_DIR = P.PARAM_RESULTS
CONFIG = P.CONFIG_DIR / "s60_parameters.json"

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

def _is_ranked_file(p: Path) -> bool:
    n = p.name
    if not n.startswith("param_results_"): return False
    if n.startswith("param_results_ALL_"): return False
    if n.endswith("_ALL.csv"): return False
    return n.endswith(".csv")

def _find_ranked(param_dir: Path, filt: str | None) -> list[Path]:
    files = [p for p in param_dir.glob("param_results_*_*.csv") if _is_ranked_file(p)]
    if filt:
        files = [p for p in files if filt in p.name]
    return sorted(files)

def _load_param_keys(config_path: Path) -> list[str]:
    try:
        with open(config_path, "r") as f:
            return list(json.load(f).keys())
    except Exception:
        return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param_dir", type=str, default=str(PARAM_DIR))
    ap.add_argument("--config", type=str, default=str(CONFIG))
    ap.add_argument("--filename_contains", type=str, default="",
                    help="Only include per-ticker ranked files whose names contain this substring. '' = all.")
    ap.add_argument("--top_n", type=int, default=5, help="Top-N per ticker after ROI dedupe.")
    ap.add_argument("--min_trades", type=int, default=0, help="Filter out rows with num_trades < this before ranking.")
    ap.add_argument("--roi_round_dp", type=int, default=6, help="Dedup tolerance: round ROI to this many decimals.")
    args = ap.parse_args()

    param_dir = Path(args.param_dir)
    if not param_dir.exists():
        raise SystemExit(f"[ERR] param_dir not found: {param_dir}")

    files = _find_ranked(param_dir, args.filename_contains.strip() or None)
    if not files:
        raise SystemExit(f"[ERR] No per-ticker ranked CSVs matched in {param_dir} (filter='{args.filename_contains}').")

    # Load all per-ticker ranked rows
    chunks = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue
        if df is None or df.empty:
            continue
        # normalize key metrics
        for c in ("total_return","win_rate"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        if "num_trades" in df.columns:
            df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce")

        # infer ticker from filename: param_results_{TICKER}_{SUFFIX}.csv
        base = p.stem[len("param_results_"):]
        ticker = base.split("_", 1)[0] if "_" in base else base
        df["ticker"] = ticker
        df["source_file"] = p.name

        # min_trades filter
        if args.min_trades and "num_trades" in df.columns:
            df = df[df["num_trades"].fillna(0) >= args.min_trades]
            if df.empty:
                continue

        chunks.append(df)

    if not chunks:
        raise SystemExit("[ERR] No data loaded after filtering.")

    all_rows = pd.concat(chunks, ignore_index=True)

    # Sort by objective then dedupe ROI within ticker
    # (Use rounding to 'roi_round_dp' to avoid float jitter)
    if "total_return" not in all_rows.columns:
        raise SystemExit("[ERR] Column 'total_return' not present in inputs.")

    all_rows["__roi_round"] = all_rows["total_return"].round(int(args.roi_round_dp))

    # Strong sort before dedupe
    sort_cols = ["ticker", "total_return", "num_trades", "win_rate"]
    sort_asc  = [True, False, False, False]
    ranked = all_rows.sort_values(by=sort_cols, ascending=sort_asc)

    # Keep first occurrence of each rounded ROI per ticker
    deduped = (ranked
               .drop_duplicates(subset=["ticker","__roi_round"], keep="first")
               .copy())

    # Take Top-N distinct-ROI rows per ticker
    topN = (deduped
            .groupby("ticker", as_index=False, group_keys=False)
            .head(max(int(args.top_n), 1))
            .copy())

    # Re-rank 1..N within ticker
    topN["rank"] = topN.groupby("ticker").cumcount() + 1

    # Order output columns: rank, ticker, metrics, then params (if present)
    metric_cols = [c for c in ["total_return","num_trades","win_rate","source_file"] if c in topN.columns]
    # Include known param keys if available
    param_keys = _load_param_keys(Path(args.config))
    param_cols = [c for c in param_keys if c in topN.columns]
    # Fallback: include any other columns that look like params
    if not param_cols:
        guess_params = [c for c in topN.columns if c not in (["ticker","rank","__roi_round"] + metric_cols)]
        param_cols = guess_params

    out_cols = ["ticker","rank"] + metric_cols + param_cols
    out = topN[out_cols].reset_index(drop=True)

    # Save
    ts = _ts()
    out_file = param_dir / f"s63a_top{args.top_n}_distinct_roi_{ts}.csv"
    out.to_csv(out_file, index=False)
    print(f"[OK] Wrote Top-{args.top_n} distinct ROI per ticker → {out_file}")

if __name__ == "__main__":
    main()