#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s80_confidence_gate2.py — Gate-2 confidence indicator (Parquet, multicore, calibrated)
-------------------------------------------------------------------------------------
- Reads 30m enriched Parquets from s32 (default folder)
- Uses learned normalization bounds from gate2_norm_bounds.json (s79)
- Uses per-ticker weights from gate2_weights.json
- Computes weighted confidence per bar
- Multicore processing (ProcessPoolExecutor, spawn-safe for macOS)
- Writes merged CSV (only essential columns)

Defaults:
  --input_dir  /Users/Finance/QuantETFUS_small/data_enriched/30min
  --glob       *.parquet
  --output     /Users/Finance/QuantETFUS_small/data_enriched/gate2_confidence_30m.csv
  --weights    /Users/Finance/QuantETFUS_small/config/gate2_weights.json
  --norms      /Users/Finance/QuantETFUS_small/config/gate2_norm_bounds.json
"""

import os, json, argparse, multiprocessing as mp, gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# -------- Defaults --------
PROJECT_ROOT   = Path("/Users/Finance/QuantETFUS_small")
DEFAULT_INPUT  = PROJECT_ROOT / "data_enriched" / "30min"
DEFAULT_GLOB   = "*.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_enriched" / "gate2_confidence_30m.csv"
DEFAULT_WEIGHTS= PROJECT_ROOT / "config" / "gate2_weights.json"
DEFAULT_NORMS  = PROJECT_ROOT / "config" / "gate2_norm_bounds.json"

# Only keep these columns in the final CSV
ESSENTIAL_COLS = [
    "datetime","ticker",
    "RSI14","ADX14",
    "EMA5_slope","EMA20_slope","EMA44_slope","EMA340_slope",
    "Donchian_position","Volatility_ATR","Trend_alignment",
    "confidence_score"
]

# -------- Helpers --------
def normalize_series(s: pd.Series, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").clip(low, high)
    return (s - low) / (high - low) if high > low else s * 0.0

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r") as f:
        return json.load(f)

def get_bounds(norms: dict, ticker: str, key: str) -> tuple[float, float]:
    """Return (low, high) from per_ticker else global else fallback."""
    try:
        per = norms.get("per_ticker", {}).get(ticker, {})
        if key in per:
            return tuple(per[key])
    except Exception:
        pass
    glob = norms.get("global", {}).get(key, None)
    if glob:
        return tuple(glob)
    # Fallback defaults
    defaults = {
        "RSI14": (30,70), "ADX14": (10,40),
        "EMA5_slope":(-0.002,0.002),"EMA20_slope":(-0.001,0.001),
        "EMA44_slope":(-0.001,0.001),"EMA340_slope":(-0.001,0.001),
        "Donchian_position":(0,1),"Volatility_ATR":(0,2),"Trend_alignment":(0,1)
    }
    return defaults.get(key,(0,1))

def compute_confidence(df: pd.DataFrame, ticker: str, weights: dict, norms: dict) -> pd.DataFrame:
    base_w = dict(weights.get("global", {}))
    overrides = weights.get("overrides", {})
    if ticker in overrides:
        base_w.update(overrides[ticker])

    def col_or(name: str, neutral: float) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(neutral, index=df.index, dtype="float64")

    # normalizations via calibrated bounds
    def nfeat(name: str, neutral: float) -> pd.Series:
        low, high = get_bounds(norms, ticker, name)
        return normalize_series(col_or(name, neutral), low, high)

    rsi     = nfeat("RSI14", 50.0)
    adx     = nfeat("ADX14", 20.0)
    ema5    = nfeat("EMA5_slope", 0.0)
    ema20   = nfeat("EMA20_slope", 0.0)
    ema44   = nfeat("EMA44_slope", 0.0)
    ema340  = nfeat("EMA340_slope", 0.0)
    donch   = nfeat("Donchian_position", 0.5)
    atr     = nfeat("Volatility_ATR", 1.0)
    trend   = col_or("Trend_alignment", 0.0)

    def w(k: str) -> float:
        return float(base_w.get(k, 0.0))

    conf = (
        w("RSI14")               * rsi +
        w("ADX14")               * adx +
        w("EMA5_slope")          * ema5 +
        w("EMA20_slope")         * ema20 +
        w("EMA44_slope")         * ema44 +
        w("EMA340_slope")        * ema340 +
        w("Donchian_position")   * donch +
        w("Volatility_ATR")      * atr +
        w("Trend_alignment")     * trend
    ).clip(0, 1)

    out = df.copy()
    out["confidence_score"] = conf
    out["ticker"] = str(ticker)

    # Trim to essential columns early (saves a LOT of memory/IO)
    keep = [c for c in ESSENTIAL_COLS if c in out.columns]
    out = out[keep].copy()
    return out

def _process_one(path: Path, weights: dict, norms: dict) -> pd.DataFrame:
    df = pd.read_parquet(path)
    tkr = str(df["ticker"].iloc[0]) if "ticker" in df.columns and len(df) else path.stem.upper()
    return compute_confidence(df, tkr, weights, norms)

# -------- CLI --------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Gate-2 confidence (multicore, calibrated).")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT), help="Folder with s32 parquets")
    ap.add_argument("--glob",      default=DEFAULT_GLOB,       help="Glob (default: *.parquet)")
    ap.add_argument("--output",    default=str(DEFAULT_OUTPUT),help="Output CSV")
    ap.add_argument("--weights",   default=str(DEFAULT_WEIGHTS),help="Path to gate2_weights.json")
    ap.add_argument("--norms",     default=str(DEFAULT_NORMS), help="Path to gate2_norm_bounds.json")
    ap.add_argument("--workers",   type=int, default=max(1,(os.cpu_count() or 2)-1))
    return ap.parse_args()

# -------- Main --------
def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob(args.glob) if p.suffix.lower()==".parquet"])
    if not files:
        raise SystemExit(f"[ERR] No files found in {in_dir}")

    weights = load_json(Path(args.weights))
    norms   = load_json(Path(args.norms))

    total = len(files)
    print(f"[INFO] Gate-2 scoring start | files={total} | workers={args.workers}")
    print(f"[INFO] Using calibrated bounds from {args.norms}")

    results = []
    # spawn is safer on macOS with pandas
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_process_one, f, weights, norms): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            path = futs[fut]
            try:
                df = fut.result()
                results.append(df)
                gc.collect()  # free memory as we go
                print(f"[OK] {i}/{total}: {path.name}")
            except Exception as e:
                print(f"[WARN] {i}/{total}: {path.name} → {e}")

    if not results:
        raise SystemExit("[ERR] No results produced.")

    merged = pd.concat(results, ignore_index=True)
    # Columns already trimmed in workers; keep guard just in case
    keep = [c for c in ESSENTIAL_COLS if c in merged.columns]
    merged = merged[keep].copy()

    merged.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"✅ Gate-2 confidence saved → {out_csv} | rows={len(merged):,} | tickers={merged['ticker'].nunique()}")

if __name__ == "__main__":
    main()