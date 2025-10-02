#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s69_quality_check.py
Audit per-ticker sample sizes and assign Quality levels for ML robustness.

What this script does
---------------------
- Loads train/val/test splits from <ROOT>/data_final_training/{train,val,test}.{csv,parquet}
  (CSV preferred; falls back to Parquet if CSV missing).
- Counts total samples per ticker (n_total).
- Simulates a balanced 60/20/20 split to estimate per-phase sample sizes.
- Classifies each ticker into (defaults match previous version):
    High Confidence : train_est ≥ 200 and val_est ≥ 100 and test_est ≥ 100
    Medium (OK)     : train_est ≥ 150 and val_est ≥  80 and test_est ≥  80
    Risky           : n_total ≥ 300 but below Medium thresholds
    No Go           : n_total < 300
- Adds a guidance note per ticker.
- Saves results → <ROOT>/param_results/per_ticker_quality.csv
- Prints a summary by Quality category.

Run AFTER:  s68_split_dataset.py
Run BEFORE: downstream model training (s70…)
"""

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import pandas as pd

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # expects P.ROOT etc.

# ---- Defaults (can be overridden by CLI) ----
DATA_DIR_DEFAULT = P.ROOT / "data_final_training"
OUT_DIR_DEFAULT  = P.ROOT / "param_results"
OUT_FILE_DEFAULT = OUT_DIR_DEFAULT / "per_ticker_quality.csv"

R_TRAIN, R_VAL, R_TEST = 0.60, 0.20, 0.20  # simulated split proportions

# ---------- helpers ----------
def _load_df_any(path_csv: Path, name: str) -> pd.DataFrame:
    """
    Try CSV first; if not present, try a same-name Parquet.
    """
    if path_csv.exists():
        return pd.read_csv(path_csv)
    pq = path_csv.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    raise SystemExit(f"[ERR] Missing {name} file (neither CSV nor Parquet): {path_csv} / {pq}")

def _classify_quality(n_total: int, tr: int, va: int, te: int,
                      min_total: int,
                      hi_train: int, hi_val: int, hi_test: int,
                      ok_train: int, ok_val: int, ok_test: int) -> str:
    if n_total < min_total:
        return "No Go"
    if tr >= hi_train and va >= hi_val and te >= hi_test:
        return "High Confidence"
    if tr >= ok_train and va >= ok_val and te >= ok_test:
        return "Medium (OK)"
    return "Risky"

def _note(quality: str) -> str:
    return {
        "High Confidence": "Trust signal + ML projection",
        "Medium (OK)":     "Signal + ML OK; reduce size",
        "Risky":           "Signal only; ignore ML",
        "No Go":           "Signal only; manual confirmation",
    }.get(quality, "")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-ticker quality audit after s68 split.")
    p.add_argument("--data_dir", default=str(DATA_DIR_DEFAULT),
                   help="Dir with {train,val,test}.{csv,parquet} (default: <ROOT>/data_final_training)")
    p.add_argument("--out_file", default=str(OUT_FILE_DEFAULT),
                   help="Output CSV path (default: <ROOT>/param_results/per_ticker_quality.csv)")

    # Thresholds (defaults match your previous logic)
    p.add_argument("--min_total", type=int, default=300,
                   help="Minimum total samples to avoid 'No Go' (default: 300)")
    p.add_argument("--hi_train", type=int, default=200)
    p.add_argument("--hi_val",   type=int, default=100)
    p.add_argument("--hi_test",  type=int, default=100)
    p.add_argument("--ok_train", type=int, default=150)
    p.add_argument("--ok_val",   type=int, default=80)
    p.add_argument("--ok_test",  type=int, default=80)

    # Proportions (only if you want to simulate other ratios; defaults 60/20/20)
    p.add_argument("--ratio_train", type=float, default=R_TRAIN)
    p.add_argument("--ratio_val",   type=float, default=R_VAL)
    # test ratio is implied (1 - train - val)

    return p.parse_args()

# ---------- main ----------
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "train.csv"
    val_csv   = data_dir / "val.csv"
    test_csv  = data_dir / "test.csv"

    print("[s69_quality_check] Loading splits…")
    df_tr = _load_df_any(train_csv, "TRAIN")
    df_va = _load_df_any(val_csv,   "VAL")
    df_te = _load_df_any(test_csv,  "TEST")

    for name, df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
        if "ticker" not in df.columns:
            raise SystemExit(f"[ERR] {name} missing required column 'ticker'")

    # Current counts
    n_tr_cur = df_tr.groupby("ticker").size().rename("n_train_cur")
    n_va_cur = df_va.groupby("ticker").size().rename("n_val_cur")
    n_te_cur = df_te.groupby("ticker").size().rename("n_test_cur")

    # Global totals
    all_df = pd.concat([df_tr[["ticker"]], df_va[["ticker"]], df_te[["ticker"]]], axis=0)
    n_total = all_df.groupby("ticker").size().rename("n_total").reset_index()

    # Estimate balanced split based on provided ratios
    r_tr = float(args.ratio_train)
    r_va = float(args.ratio_val)
    if r_tr < 0 or r_va < 0 or (r_tr + r_va) >= 1.0:
        raise SystemExit("[ERR] Bad ratios: ensure 0 <= train,val and train+val < 1.0")

    n_total["train_est"] = (n_total["n_total"] * r_tr).astype(int)
    n_total["val_est"]   = (n_total["n_total"] * r_va).astype(int)
    n_total["test_est"]  = n_total["n_total"] - n_total["train_est"] - n_total["val_est"]

    # Merge current counts
    out = (n_total
           .merge(n_tr_cur.reset_index(), on="ticker", how="left")
           .merge(n_va_cur.reset_index(), on="ticker", how="left")
           .merge(n_te_cur.reset_index(), on="ticker", how="left"))

    for c in ["n_train_cur", "n_val_cur", "n_test_cur"]:
        out[c] = out[c].fillna(0).astype(int)

    # Quality + guidance note
    out["quality"] = out.apply(
        lambda r: _classify_quality(
            int(r["n_total"]),
            int(r["train_est"]), int(r["val_est"]), int(r["test_est"]),
            args.min_total,
            args.hi_train, args.hi_val, args.hi_test,
            args.ok_train, args.ok_val, args.ok_test
        ),
        axis=1
    )
    out["guidance_note"] = out["quality"].map(_note)

    # Order & save
    cols = ["ticker", "n_total", "train_est", "val_est", "test_est",
            "n_train_cur", "n_val_cur", "n_test_cur", "quality", "guidance_note"]
    out = out[cols].sort_values(["quality","n_total"], ascending=[True, False]).reset_index(drop=True)

    out.to_csv(out_file, index=False)
    print(f"[OK] Wrote {out_file}")

    # Summary
    summary = out["quality"].value_counts().reset_index()
    summary.columns = ["quality","count"]
    total_tickers = len(out)
    summary["share_%"] = (summary["count"]/total_tickers*100).round(1)

    print("\n[Summary by Quality]")
    for _, r in summary.iterrows():
        print(f"  {r['quality']:<16} {r['count']:>3} ({r['share_%']:>4.1f}%)")
    print(f"\n[Totals] tickers={total_tickers}, samples={int(out['n_total'].sum())}")

if __name__ == "__main__":
    main()