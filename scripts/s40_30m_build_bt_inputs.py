#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s40_30m_build_bt_inputs.py
From raw 30m CSVs → per-ticker *_bt_30m.csv used by s50_30m backtests.

- EMA base length configurable (default 260 ≈ ~20 trading days on 30m)
- Uses ema_base for trend; still writes ema260 as alias for backward-compat
- Writes to project-scoped backtest folder: P.BACKTEST_DATA/30min

Reads
-----
- <shared>/30min/{TICKER}_30min_raw.csv | {TICKER}_30min.csv

Writes
------
- P.BACKTEST_DATA/30min/{TICKER}_bt_30m.csv
  Columns:
    datetime,ticker,open,high,low,close,
    ema5,ema20,ema44,ema_base,ema20d,ema260,entry_signal
"""

from pathlib import Path
import sys, os, argparse
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
    """
    Preferred order:
      1) $SHARED_RAW_BASE/30min (if env provided)
      2) P.SHARED_30M_DIR
      3) P.SHARED_RAW_BASE/30min
      4) (legacy) P.DATA_RAW/<bucket>/30min if explicitly requested
    """
    env_shared = os.environ.get("SHARED_RAW_BASE", "").strip()
    if env_shared:
        cand = Path(env_shared).expanduser().resolve() / "30min"
        if cand.exists():
            return cand

    # Primary project-scoped shared
    if hasattr(P, "SHARED_30M_DIR") and Path(P.SHARED_30M_DIR).exists():
        return Path(P.SHARED_30M_DIR)

    # Fallback to shared raw base
    if hasattr(P, "SHARED_RAW_BASE"):
        cand = Path(P.SHARED_RAW_BASE).expanduser().resolve() / "30min"
        if cand.exists():
            return cand

    # Optional bucket under project-local data_raw (legacy)
    if bucket_arg:
        cand = Path(P.DATA_RAW) / bucket_arg.strip() / "30min"
        if cand.exists():
            return cand

    raise SystemExit(
        "[ERR] Could not find 30min input folder.\n"
        f"  Tried: $SHARED_RAW_BASE/30min, P.SHARED_30M_DIR, P.SHARED_RAW_BASE/30min\n"
        f"  (Tip: export SHARED_RAW_BASE='{P.SHARED_RAW_BASE}' )"
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
                    help="Legacy fallback under project-local data_raw/<bucket>/30min (optional).")
    ap.add_argument("--ema-base-bars", type=int, default=260,
                help="30m EMA base length in bars (default 260 → 20 trading days on 6.5h).")
    args = ap.parse_args()

    IN_DIR = resolve_in_dir(args.bucket)
    OUT_DIR = Path(P.BACKTEST_DATA) / "30min"   # project-scoped backtest path
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(IN_DIR.glob("*_30min_raw.csv")) + list(IN_DIR.glob("*_30min.csv")))
    if not files:
        raise SystemExit(f"[ERR] No input files in {IN_DIR}")

    print(f"[INFO] Input dir:  {IN_DIR}")
    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] EMA base (30m bars): {args.ema_base_bars}")

    count, errors = 0, 0

    for f in files:
        try:
            tkr = f.name.replace("_30min_raw.csv", "").replace("_30min.csv", "")
            df = load_csv(f)

            # EMAs
            df["ema5"]      = ema(df["close"], 5)
            df["ema20"]     = ema(df["close"], 20)
            df["ema44"]     = ema(df["close"], 44)
            df["ema_base"]  = ema(df["close"], int(args.ema_base_bars))
            df["ema20d"]    = df["ema_base"]     # explicit alias = "20 trading days on 30m"
            df["ema260"]    = df["ema_base"]     # legacy compatibility

            # Entry signal
            df["entry_signal"] = make_entry_signal(df).astype(bool)

            # Output
            out = df[["datetime","open","high","low","close",
                      "ema5","ema20","ema44","ema_base","ema20d","ema260"]].copy()
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