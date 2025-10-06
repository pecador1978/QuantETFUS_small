#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s80_confidence_gate2.py — Gate-2 confidence indicator (Parquet, multicore, defaults)
------------------------------------------------------------------------------------
- Reads 30m enriched Parquets from s32 (default folder).
- Computes Gate-2 confidence using gate2_weights.json (per-ticker overrides).
- Multicore via ProcessPoolExecutor.
- Writes a single CSV incrementally (low memory).
- Sensible defaults so it can be run with no CLI args.

Defaults:
  --input_dir  /Users/Finance/QuantETFUS_small/data_enriched/30min
  --glob       *.parquet
  --output     /Users/Finance/QuantETFUS_small/data_enriched/gate2_confidence_30m.csv
  --weights    /Users/Finance/QuantETFUS_small/config/gate2_weights.json
  --workers    (cpu_count - 1) or 1 if cpu_count==1
"""

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import multiprocessing as mp
from datetime import datetime

# -------- Defaults (edit here if paths change) --------
PROJECT_ROOT  = Path("/Users/Finance/QuantETFUS_small")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data_enriched" / "30min"
DEFAULT_GLOB      = "*.parquet"
DEFAULT_OUTPUT    = PROJECT_ROOT / "data_enriched" / "gate2_confidence_30m.csv"
DEFAULT_WEIGHTS   = PROJECT_ROOT / "config" / "gate2_weights.json"

# ---------------- Helpers ----------------
def normalize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").clip(lower, upper)
    return (s - lower) / (upper - lower) if upper > lower else s * 0.0

def load_gate2_weights(weights_path: str | Path) -> dict:
    p = Path(weights_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"gate2_weights.json not found at: {p}")
    with open(p, "r") as f:
        return json.load(f)

def compute_confidence_frame(df: pd.DataFrame, ticker: str, weights: dict) -> pd.DataFrame:
    """Return df with confidence_score column (expects s32 Gate-2 fields)."""
    base_w = dict(weights.get("global", {}))
    overrides = weights.get("overrides", {})
    if ticker in overrides:
        base_w.update(overrides[ticker])

    def col_or(name: str, neutral: float) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(neutral, index=df.index, dtype="float64")

    rsi     = normalize_series(col_or("RSI14",             50.0), 30, 70)
    adx     = normalize_series(col_or("ADX14",             20.0), 10, 40)
    ema5    = normalize_series(col_or("EMA5_slope",         0.0), -0.002, 0.002)
    ema20   = normalize_series(col_or("EMA20_slope",        0.0), -0.001, 0.001)
    ema44   = normalize_series(col_or("EMA44_slope",        0.0), -0.001, 0.001)
    ema340  = normalize_series(col_or("EMA340_slope",       0.0), -0.001, 0.001)
    donch   = normalize_series(col_or("Donchian_position",  0.5), 0, 1)
    atr     = normalize_series(col_or("Volatility_ATR",     1.0), 0, 2)
    trend   = col_or("Trend_alignment", 0.0)

    def w(k: str) -> float:
        return float(base_w.get(k, 0.0))

    conf = (
        w("RSI14")               * rsi
        + w("ADX14")             * adx
        + w("EMA5_slope")        * ema5
        + w("EMA20_slope")       * ema20
        + w("EMA44_slope")       * ema44
        + w("EMA340_slope")      * ema340
        + w("Donchian_position") * donch
        + w("Volatility_ATR")    * atr
        + w("Trend_alignment")   * trend
    ).clip(0, 1)

    out = df.copy()
    out["confidence_score"] = conf
    # Ensure ticker column exists and is stable
    if "ticker" not in out.columns:
        out["ticker"] = str(ticker)
    else:
        out["ticker"] = out["ticker"].astype(str).fillna(str(ticker))
    return out

def _process_one_file(path: Path, weights: dict) -> pd.DataFrame:
    """Worker: read parquet, infer ticker if needed, compute confidence, return DataFrame."""
    df = pd.read_parquet(path)
    tkr = df["ticker"].iloc[0] if "ticker" in df.columns and len(df) else path.stem.upper()
    return compute_confidence_frame(df, str(tkr), weights)

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Gate-2 confidence scores (Parquet, multicore).")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Folder with s32 parquets.")
    ap.add_argument("--glob",      default=DEFAULT_GLOB,          help="Glob pattern (default: *.parquet)")
    ap.add_argument("--output",    default=str(DEFAULT_OUTPUT),   help="Output CSV path.")
    ap.add_argument("--weights",   default=str(DEFAULT_WEIGHTS),  help="Path to gate2_weights.json")
    ap.add_argument("--workers",   type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Number of processes (default: cpu_count-1).")
    return ap.parse_args()

# ---------------- Main ----------------
def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in in_dir.glob(args.glob) if p.suffix.lower() == ".parquet"])
    if not paths:
        raise SystemExit(f"[ERR] No parquet files matched: {in_dir}/{args.glob}")

    weights = load_gate2_weights(args.weights)

    # Stream-write: header on first write, then append
    wrote_header = False
    processed = 0
    total = len(paths)
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[INFO] Gate-2 scoring start {start_ts} | files={total} | workers={args.workers}")
    print(f"[INFO] input_dir={in_dir} | output={out_csv}")

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as ex:
        futs = {ex.submit(_process_one_file, p, weights): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                df_part = fut.result()
                # Write chunk
                if not wrote_header:
                    df_part.to_csv(out_csv, index=False, mode="w")
                    wrote_header = True
                else:
                    df_part.to_csv(out_csv, index=False, mode="a", header=False)
                processed += 1
                print(f"[OK] {processed}/{total}: {p.name}")
            except Exception as e:
                processed += 1
                print(f"[WARN] {processed}/{total}: {p.name} → {e}")

    print(f"✅ Gate-2 confidence saved → {out_csv}")

if __name__ == "__main__":
    main()