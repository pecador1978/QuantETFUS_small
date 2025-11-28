#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s79_calibrate_gate2_bounds.py — Learn per-ticker normalization ranges for Gate-2
---------------------------------------------------------------------------------
Reads s32 30m parquets, computes robust (q05..q95) ranges per indicator per ticker,
plus global fallbacks, and writes JSON used by s90 to normalize features.

Defaults:
  INPUT_DIR  = /Users/Finance/QuantETFUS_small/data_enriched/30min
  OUTPUT_JSON= /Users/Finance/QuantETFUS_small/config/gate2_norm_bounds.json
"""

from pathlib import Path
import argparse, json, os
import pandas as pd
import numpy as np

# --- Project root resolution ---
SCRIPT_DIR    = Path(__file__).resolve().parent
DEFAULT_ROOT  = SCRIPT_DIR.parents[1]            # repo root (…/QuantETF_LSE or …/QuantETF_NY)
PROJECT_ROOT  = Path(os.environ.get("PROJECT_ROOT", DEFAULT_ROOT))  

DEFAULT_INPUT_DIR   = PROJECT_ROOT / "data_enriched" / "30min"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "config" / "gate2_norm_bounds.json"

FEATURES = [
    "RSI14",
    "ADX14",
    "EMA5_slope",
    "EMA20_slope",
    "EMA44_slope",
    "EMA260_slope",
    "Donchian_position",
    "Volatility_ATR",
    "Trend_alignment",
]

# hard safety floors so ranges never collapse
MIN_SPAN = {
    "RSI14": 5.0,
    "ADX14": 5.0,
    "EMA5_slope": 1e-5,
    "EMA20_slope": 5e-6,
    "EMA44_slope": 5e-6,
    "EMA260_slope": 5e-6,
    "Donchian_position": 0.05,
    "Volatility_ATR": 0.02,
    "Trend_alignment": 1.0,  # boolean-like; keep [0,1]
}

def robust_bounds(s: pd.Series, f: str) -> tuple[float,float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        # sensible defaults if missing
        if f == "RSI14": return (30.0, 70.0)
        if f == "ADX14": return (10.0, 40.0)
        if f.startswith("EMA") and "slope" in f: return (-0.001, 0.001)
        if f == "Donchian_position": return (0.0, 1.0)
        if f == "Volatility_ATR": return (0.0, 2.0)
        if f == "Trend_alignment": return (0.0, 1.0)
    q05 = s.quantile(0.05)
    q95 = s.quantile(0.95)
    # for boolean-like Trend_alignment, force [0,1]
    if f == "Trend_alignment":
        q05, q95 = 0.0, 1.0
    # ensure minimum span
    if (q95 - q05) < MIN_SPAN.get(f, 1e-9):
        mid = (q05 + q95) / 2.0
        half = MIN_SPAN.get(f, 1e-9) / 2.0
        q05, q95 = mid - half, mid + half
    # guard: RSI/ADX sensible caps
    if f == "RSI14":
        q05, q95 = max(0.0, q05), min(100.0, q95)
    if f == "ADX14":
        q05, q95 = max(0.0, q05), q95
    # Donchian in [0,1]
    if f == "Donchian_position":
        q05, q95 = max(0.0, q05), min(1.0, q95)
    return float(q05), float(q95)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibrate per-ticker Gate-2 normalization bounds from history.")
    ap.add_argument("--input_dir",  default=str(DEFAULT_INPUT_DIR), help="Folder with s32 parquets.")
    ap.add_argument("--glob",       default="*.parquet",           help="Glob inside input_dir.")
    ap.add_argument("--output",     default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path.")
    return ap.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    paths = sorted([p for p in in_dir.glob(args.glob) if p.suffix.lower()==".parquet"])
    if not paths:
        raise SystemExit(f"[ERR] No parquets in {in_dir}/{args.glob}")

    # collect global stacks to compute global fallbacks
    global_stacks = {f: [] for f in FEATURES}
    per_ticker: dict[str, dict[str, list[float]]] = {}

    for p in paths:
        try:
            df = pd.read_parquet(p)
            tkr = str(df["ticker"].iloc[0]) if "ticker" in df.columns and len(df) else p.stem.upper()
            # accumulate global stacks
            for f in FEATURES:
                if f in df.columns:
                    global_stacks[f].append(pd.to_numeric(df[f], errors="coerce"))
            # per-ticker bounds
            t_bounds = {}
            for f in FEATURES:
                if f in df.columns:
                    q05, q95 = robust_bounds(df[f], f)
                else:
                    # missing feature → use default
                    q05, q95 = robust_bounds(pd.Series(dtype=float), f)
                t_bounds[f] = [q05, q95]
            per_ticker[tkr] = t_bounds
            print(f"[OK] {tkr}: calibrated")
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")

    # compute global fallbacks
    global_bounds = {}
    for f in FEATURES:
        if global_stacks[f]:
            stack = pd.concat(global_stacks[f], ignore_index=True)
            q05, q95 = robust_bounds(stack, f)
        else:
            q05, q95 = robust_bounds(pd.Series(dtype=float), f)
        global_bounds[f] = [q05, q95]

    out = {
        "schema": "gate2_norm_bounds.v1",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "global": global_bounds,
        "per_ticker": per_ticker
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"✅ Bounds written → {out_path}")

if __name__ == "__main__":
    main()