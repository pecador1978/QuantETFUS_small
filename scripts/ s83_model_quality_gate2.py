#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s83_model_quality_gate2.py — Derive Gate-2 model trust quality per ticker
--------------------------------------------------------------------------
Reads per-ticker model performance stats (from s82) and computes:
- trust_score ∈ [0,1]
- quality category: High / Medium / Low
Saves to config/gate2_ticker_quality.json for downstream use (e.g. s90 decisions).

Scoring logic:
    trust_score = 0.6 * norm_AUC  + 0.3 * (1 - norm_Brier)  + 0.1 * norm_log_rows
    where each term is min-max normalized across tickers
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# -------- Defaults --------
PROJECT_ROOT = Path("/Users/Finance/QuantETFUS_small")
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "gate2_model_per_ticker.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "config" / "gate2_ticker_quality.json"

# -------- Helpers --------
def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return s
    low, high = np.nanmin(s), np.nanmax(s)
    if np.isclose(high, low):
        return pd.Series(0.5, index=s.index)
    return (s - low) / (high - low)

def compute_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["AUC", "Brier", "rows"]:
        if col not in df.columns:
            raise SystemExit(f"[ERR] Missing column: {col}")

    # Normalize each component
    df["auc_n"] = normalize(df["AUC"])
    df["brier_n"] = normalize(df["Brier"])
    df["log_rows"] = np.log1p(df["rows"])
    df["log_rows_n"] = normalize(df["log_rows"])

    # Weighted trust formula
    df["trust_score"] = (
        0.6 * df["auc_n"] +
        0.3 * (1 - df["brier_n"]) +
        0.1 * df["log_rows_n"]
    ).clip(0, 1)

    # Quality categories
    def label(q):
        if q >= 0.66: return "High"
        if q >= 0.33: return "Medium"
        return "Low"
    df["quality"] = df["trust_score"].apply(label)
    return df

# -------- CLI --------
def parse_args():
    ap = argparse.ArgumentParser(description="Compute Gate-2 model trust per ticker.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="CSV from s82 with per-ticker metrics (AUC, Brier, rows).")
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
    print(dfq[["ticker", "AUC", "Brier", "rows", "trust_score", "quality"]]
          .sort_values("trust_score", ascending=False)
          .to_string(index=False))

if __name__ == "__main__":
    main()