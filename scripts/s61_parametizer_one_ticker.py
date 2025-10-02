#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s61_parametizer_one_ticker.py
Runs parameter sweeps for ONE ticker only (parallel).

- Reads ranges from /Users/Finance/QuantETF/config/s60_parameters.json
- Calls backtest_ticker() from s50 directly (in-process)
- Saves ranked results to /Users/Finance/QuantETF/param_results/param_results_{TICKER}_{TS}.csv
- Also saves full unfiltered results to ..._ALL.csv
- Saves full trades for each combo into /Users/Finance/QuantETF/param_results/trades/
"""

import os
import sys
import json
import itertools
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# ---------- optional: nicer progress for joblib ----------
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib  # pip install tqdm_joblib
    TQDM_AVAILABLE = True
except Exception:
    # Fallback: no pretty nested bar, we'll print a simple info line instead
    from tqdm import tqdm

# ---------- joblib ----------
from joblib import Parallel, delayed

# ---------- make sure we can import local modules (s50 / strategy_runtime) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---------- paths ----------
ROOT = Path("/Users/Finance/QuantETF")
PARAM_FILE = ROOT / "config" / "s60_parameters.json"
OUT_DIR = ROOT / "param_results"
TRADES_DIR = OUT_DIR / "trades"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

UNIT_CAPITAL = 10000.0  # per your spec

# ---------- import engine pieces ----------
from scripts.s50_30m_backtest_engine_three_PT import (
    backtest_ticker,
    summarize_trades_per_ticker,
)

def resolve_ticker(cli_ticker: str | None) -> str:
    """Return the ticker to use (CLI overrides strategy_runtime)."""
    if cli_ticker:
        return cli_ticker

    # Try to import strategy_runtime from the same folder
    try:
        import importlib
        sr = importlib.import_module("strategy_runtime")
        sr = importlib.reload(sr)  # ensure we get the latest file you edited
        return getattr(sr, "TICKER")
    except Exception as e:
        raise SystemExit(
            "Ticker not provided and strategy_runtime.TICKER not found.\n"
            "Run with:  --ticker VUAA   (example)\n"
            f"Details: {e}"
        )

def load_param_grid():
    with open(PARAM_FILE, "r") as f:
        cfg = json.load(f)
    # ensure all values are lists for product()
    grid = {k: (v if isinstance(v, list) else [v]) for k, v in cfg.items()}
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return keys, vals

def run_one_combo(combo_id: int, keys: list[str], combo: tuple, ticker: str, ts_global: str) -> dict:
    """Execute one parameter combo: backtest, save trades CSV, return summary row dict."""
    params = dict(zip(keys, combo))
    # keep engine quiet unless grid explicitly sets verbose=True
    params.setdefault("verbose", False)

    trade_file = TRADES_DIR / f"trades_{ticker}_combo{combo_id}_{ts_global}.csv"

    try:
        trades = backtest_ticker(ticker, params)
    except Exception as e:
        # failed combo — keep going
        pd.DataFrame().to_csv(trade_file, index=False)
        row = {
            "combo_id": combo_id,
            "ticker": ticker,
            **params,
            "num_trades": 0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "end_capital": UNIT_CAPITAL,
            "trades_file": trade_file.name,
            "error": str(e),
        }
        return row

    if trades is not None and not trades.empty:
        trades.to_csv(trade_file, index=False)
        cap_df = summarize_trades_per_ticker(trades, unit_capital=UNIT_CAPITAL)
        # should be exactly one row for this ticker
        row_cap = cap_df[cap_df["ticker"] == ticker].iloc[0]
        row = {
            "combo_id": combo_id,
            "ticker": ticker,
            "total_return": float(row_cap["total_return"]),
            **params,
            "num_trades": int(row_cap["num_trades"]),
            "win_rate": float(row_cap["win_rate"]),
            "end_capital": float(row_cap["end_capital"]),
            "trades_file": trade_file.name,
            "error": "",
        }
        return row

    # empty trades
    pd.DataFrame().to_csv(trade_file, index=False)
    row = {
        "combo_id": combo_id,
        "ticker": ticker,
        **params,
        "num_trades": 0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "end_capital": UNIT_CAPITAL,
        "trades_file": trade_file.name,
        "error": "",
    }
    return row

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, help="Override ticker (else strategy_runtime.TICKER)")
    ap.add_argument("--min_trades", type=int, default=5,
                    help="Filter: minimum trades to keep in ranking (default: 5).")
    ap.add_argument("--max_combos", type=int, default=0,
                    help="Optional cap for quick tests (0 = no cap).")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Parallel workers for joblib (default: -1 = all cores).")
    args = ap.parse_args()

    ticker = resolve_ticker(args.ticker)
    keys, vals = load_param_grid()
    combos = list(itertools.product(*vals))
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    print(f"[INFO] Ticker: {ticker}")
    print(f"[INFO] Total combos: {len(combos)}")
    ts_global = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    # -------- run in parallel --------
    iterator = enumerate(combos, start=1)
    run = (delayed(run_one_combo)(combo_id, keys, combo, ticker, ts_global)
           for combo_id, combo in iterator)

    if TQDM_AVAILABLE:
        with tqdm_joblib(tqdm(total=len(combos), desc="Param sweep", unit="combo")):
            results = Parallel(n_jobs=args.n_jobs, backend="loky")(list(run))
    else:
        print("[INFO] tqdm_joblib not available; progress bar will be simpler.")
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(list(run))

    # -------- aggregate & rank --------
    df = pd.DataFrame(results)

    # Optional filter to avoid 1-trade wonders topping the chart
    df_filtered = df[df["num_trades"] >= args.min_trades].copy()

    # Sort from most profitable to least; break ties by more trades, higher win rate
    df_sorted = df_filtered.sort_values(
        by=["total_return", "num_trades", "win_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 1-based rank
    if not df_sorted.empty:
        df_sorted.insert(0, "rank", df_sorted.index + 1)

    # Save ranked summary
    out_file = OUT_DIR / f"param_results_{ticker}_{ts_global}.csv"
    df_sorted.to_csv(out_file, index=False)

    # Also save the full unfiltered results for auditing
    out_file_all = OUT_DIR / f"param_results_{ticker}_{ts_global}_ALL.csv"
    df.to_csv(out_file_all, index=False)

    # Tiny leaderboard in console
    if not df_sorted.empty:
        print("\nTop 10 combos:")
        print(
            df_sorted[[
                "rank", "combo_id", "total_return", "end_capital",
                "num_trades", "win_rate", "trades_file"
            ]].head(10).to_string(index=False)
        )
    else:
        print("\n[WARN] No combos met the min_trades filter; see the _ALL.csv for full details.")

    print(f"\n[OK] Saved ranked summary → {out_file}")
    print(f"[OK] Saved full (unfiltered) results → {out_file_all}")
    print(f"[OK] Saved all trades to → {TRADES_DIR}")

if __name__ == "__main__":
    main()