#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s40_30m_build_bt_inputs.py
From raw 30m CSVs → per-ticker *_bt_30m.csv used by s50_30m backtests.

Changes in this version
-----------------------
- EMA "base" length is configurable (default 340 = ~20 trading days on 30m).
- Logic uses ema_base for the trend filter; legacy column ema275 is still written
  for backward compatibility (as an alias of ema_base).
- Adds explicit aliases: ema_base, ema20d (same value as ema_base).

Reads
-----
- P.DATA_RAW/30min/{TICKER}_30min_raw.csv | {TICKER}_30min.csv

Writes
------
- P.ROOT/backtest_data/30min/{TICKER}_bt_30m.csv
  Columns:
    datetime,ticker,open,high,low,close,
    ema5,ema20,ema44,ema_base,ema20d,ema275,entry_signal
"""

from pathlib import Path
import sys
import os
import argparse
import pandas as pd

# ---- import centralized paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402

pd.set_option("future.no_silent_downcasting", True)

# ---------------- CLI / input dir resolution ----------------
def resolve_in_dir(bucket_arg: str | None) -> Path:
    env_shared = os.environ.get("SHARED_RAW_BASE", "").strip()
    if env_shared:
        cand = Path(env_shared) / "30min"
        if cand.exists():
            return cand

    shared = P.DATA_RAW / "30min"
    if shared.exists():
        return shared

    qs = getattr(P, "QUANTSHARED", None) or (P.ROOT.parent / "QuantShared")
    cand_us = qs / "data_raw_ETF_US" / "30min"
    cand_eu = qs / "data_raw_ETF"    / "30min"
    if cand_us.exists():
        return cand_us
    if cand_eu.exists():
        return cand_eu

    bucket = (
        (bucket_arg or "").strip()
        or os.environ.get("TARGET_BUCKET", "").strip()
        or next((b for b in ("targeted_ETFs_US", "targeted_ETFs")
                 if (P.DATA_RAW / b / "30min").exists()), "")
    )
    if bucket:
        return P.DATA_RAW / bucket / "30min"

    raise SystemExit(
        "[ERR] Could not find 30min input folder.\n"
        f"  Tried project: {shared}\n"
        f"  Tried shared US: {cand_us}\n"
        f"  Tried shared EU: {cand_eu}\n"
        f"  And bucketed fallbacks under {P.DATA_RAW}/<bucket>/30min\n"
        "  (Tip: export SHARED_RAW_BASE=/Users/Finance/QuantShared/data_raw_ETF_US)"
    )

# ---------------- loaders & helpers ----------------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})

    req = ["datetime", "open", "high", "low", "close"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()

def make_entry_signal(g: pd.DataFrame) -> pd.Series:
    """
    Entry when 30m EMAs 'lift' above ema44 under an up-trend vs EMA BASE:
      - Filter: close > ema_base
      - Trigger: (ema5>ema44 & ema20>ema44) crossing from not-both-above → both-above
    """
    above_now  = (g["ema5"] > g["ema44"]) & (g["ema20"] > g["ema44"])
    above_prev = above_now.shift(1).fillna(False).astype(bool)
    cross_lift = above_now & (~above_prev)
    trend_ok   = g["close"] > g["ema_base"]
    return (cross_lift & trend_ok)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", type=str, default=None,
                    help="Fallback bucket under data_raw/ if shared layout isn't found.")
    ap.add_argument("--ema-base-bars", type=int,
                    default=int(os.environ.get("M30_EMA_BARS", "340")),
                    help="EMA base length in 30m bars (default 340 ≈ 20 trading days).")
    args = ap.parse_args()

    IN_DIR = resolve_in_dir(args.bucket)
    OUT_DIR = P.ROOT / "backtest_data" / "30min"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(IN_DIR.glob("*_30min_raw.csv")) + list(IN_DIR.glob("*_30min.csv")))
    if not files:
        raise SystemExit(f"[ERR] No input files in {IN_DIR}")

    print(f"[INFO] Input dir: {IN_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Using EMA base length (30m bars): {args.ema_base_bars}")

    count = 0
    errors = 0

    for f in files:
        try:
            tkr = f.name.replace("_30min_raw.csv", "").replace("_30min.csv", "")
            df = load_csv(f)

            # EMAs
            df["ema5"]    = ema(df["close"], 5)
            df["ema20"]   = ema(df["close"], 20)
            df["ema44"]   = ema(df["close"], 44)
            df["ema_base"] = ema(df["close"], int(args.ema_base_bars))
            df["ema20d"]   = df["ema_base"]            # explicit alias = "20 trading days on 30m"

            # Legacy compatibility: keep writing ema275 (alias to base)
            df["ema275"]  = df["ema_base"]

            # Entry signal (uses ema_base)
            df["entry_signal"] = make_entry_signal(df).astype(bool)

            # Output
            out = df[["datetime","open","high","low","close",
                      "ema5","ema20","ema44","ema_base","ema20d","ema275"]].copy()
            out.insert(1, "ticker", tkr)
            out["entry_signal"] = df["entry_signal"]

            out_path = OUT_DIR / f"{tkr}_bt_30m.csv"
            out.to_csv(out_path, index=False)
            count += 1
        except Exception as e:
            errors += 1
            print(f"[WARN] {f.name}: {e}")

    print(f"[OK] Wrote {count} file(s) to {OUT_DIR}  |  skipped with warnings: {errors}")

if __name__ == "__main__":
    main()