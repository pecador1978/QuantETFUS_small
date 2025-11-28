#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s83_model_quality_gate2.py — Derive Gate-2 model trust quality per ticker
--------------------------------------------------------------------------
Reads per-ticker model performance stats (from s82) and computes:
- trust_score ∈ [0,1]
- quality category: High / Medium / Low
Writes: config/gate2_ticker_quality.json

Scoring:
  trust_score = 0.6 * norm(AUC) + 0.3 * (1 - norm(Brier)) + 0.1 * norm(log(rows))
  (each term min–max normalized across tickers)
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

# --- Project root resolution ---
SCRIPT_DIR    = Path(__file__).resolve().parent
DEFAULT_ROOT  = SCRIPT_DIR.parents[1]
PROJECT_ROOT  = Path(os.environ.get("PROJECT_ROOT", DEFAULT_ROOT))

DEFAULT_INPUT  = PROJECT_ROOT / "reports" / "gate2_per_ticker_metrics.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "config"  / "gate2_ticker_quality.json"

# -------- Helpers --------
def _pick_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"[ERR] Missing column for {name}. Tried: {candidates}")

def _normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(hi, lo):
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def compute_quality(df: pd.DataFrame) -> pd.DataFrame:
    # flexible column mapping (handles s82 output)
    auc_col    = _pick_col(df, ["AUC", "auc_mean"], "AUC")
    brier_col  = _pick_col(df, ["Brier", "brier_mean"], "Brier")
    rows_col   = _pick_col(df, ["rows", "n_rows", "count"], "rows")

    out = df.copy()
    out["auc_n"]       = _normalize(out[auc_col])
    out["brier_n"]     = _normalize(out[brier_col])
    out["log_rows"]    = np.log1p(pd.to_numeric(out[rows_col], errors="coerce"))
    out["log_rows_n"]  = _normalize(out["log_rows"])

    out["trust_score"] = (
        0.6 * out["auc_n"] +
        0.3 * (1 - out["brier_n"]) +
        0.1 * out["log_rows_n"]
    ).clip(0, 1)

    def label(q):
        if q >= 0.66: return "High"
        if q >= 0.33: return "Medium"
        return "Low"

    out["quality"] = out["trust_score"].apply(label)

    # keep a tidy view with the actual source columns
    out = out.assign(AUC_src=out[auc_col], Brier_src=out[brier_col], rows_src=out[rows_col])
    cols = ["ticker", "AUC_src", "Brier_src", "rows_src", "trust_score", "quality"]
    return out[cols].rename(columns={"AUC_src":"AUC", "Brier_src":"Brier", "rows_src":"rows"})

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="Compute Gate-2 model trust per ticker.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="CSV from s82 with per-ticker metrics (supports auc_mean/brier_mean).")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Output JSON path for gate2_ticker_quality.json.")
    return ap.parse_args()

# -------- Main --------
def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(f"[ERR] Missing input metrics file: {in_path}")

    df = pd.read_csv(in_path)
    if "ticker" not in df.columns:
        raise SystemExit("[ERR] Missing 'ticker' column in input CSV.")

    dfq = compute_quality(df)

    result = {
        str(r["ticker"]): {
            "trust_score": round(float(r["trust_score"]), 3),
            "quality": r["quality"]
        }
        for _, r in dfq.iterrows()
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Gate-2 model quality JSON saved → {out_path}")
    print(
        dfq.sort_values("trust_score", ascending=False)
           .to_string(index=False)
    )

if __name__ == "__main__":
    main()