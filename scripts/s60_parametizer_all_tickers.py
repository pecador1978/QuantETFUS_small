#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s60_parametizer_all_tickers.py — MAIN (parallel route)
Multi-ticker param sweep using the s50 engine (2 targets, no Vermeulen).
Writes per-ticker results, a global leaderboard, all trades, and
s60_final_combos.json (best combo per ticker).

Clone-friendly: picks paths from common.paths.P so it works in QuantETF and QuantETFUS_small.
"""

from __future__ import annotations

import os
import sys
import json
import itertools
from pathlib import Path
from datetime import datetime, timezone

# keep math libs single-threaded to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
from joblib import Parallel, delayed

# ---- progress bars (optional)
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    from tqdm_joblib import tqdm_joblib
    TQDM_AVAILABLE = True
except Exception:
    try:
        from tqdm import tqdm  # minimal fallback
        TQDM_AVAILABLE = True
    except Exception:
        TQDM_AVAILABLE = False

# ---- make project root importable & pull shared paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P, default_etf_sheet  # shared paths + sheet helper

# ---- paths (all dynamic, shared layout)
ROOT = P.ROOT
PARAM_FILE = P.CONFIG_DIR / "s60_parameters.json"
EXCEL_DEFAULT = P.ETF_LIST
OUT_DIR = P.PARAM_RESULTS
TRADES_DIR = P.TRADES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

UNIT_CAPITAL = 10000.0  # fixed capital per ticker


# --------------- helpers ----------------

def _ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def _banner(excel_path: Path, sheet: str, engine_path: Path):
    print("[s60] ===== PARAM SWEEP LAUNCH =====")
    print(f"[s60] ROOT         : {ROOT}")
    print(f"[s60] Excel        : {excel_path} (sheet={sheet})")
    print(f"[s60] Param file   : {PARAM_FILE}")
    print(f"[s60] Engine path  : {engine_path}")
    print(f"[s60] OUT_DIR      : {OUT_DIR}")
    print(f"[s60] TRADES_DIR   : {TRADES_DIR}")


def load_param_grid(param_file: Path):
    if not param_file.exists():
        raise SystemExit(f"[ERR] Missing parameter file: {param_file}")
    with open(param_file, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or not cfg:
        raise SystemExit(f"[ERR] Parameter file must be a non-empty JSON object: {param_file}")
    grid = {k: (v if isinstance(v, list) else [v]) for k, v in cfg.items()}
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    total = 1
    for v in vals:
        total *= max(len(v), 1)
    print(f"[s60] Grid size    : {total} combos across {len(keys)} params")
    return keys, vals


def load_tickers_from_excel(excel_path: Path,
                            sheet_name: str | None = None,
                            ticker_col: str | None = None) -> list[str]:
    """Load tickers from Excel. Auto-detect sheet and column if not provided."""
    sheet_name = sheet_name or default_etf_sheet()
    if not excel_path.exists():
        raise SystemExit(f"[ERR] Excel file not found: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    if ticker_col:
        if ticker_col not in df.columns:
            raise SystemExit(f"[ERR] Column '{ticker_col}' not found in {excel_path}:{sheet_name}. "
                             f"Available: {list(df.columns)}")
        col = ticker_col
    else:
        candidates = [c for c in df.columns if c.lower() in ("ticker", "symbol", "etf", "isin_ticker")]
        if not candidates:
            raise SystemExit(f"[ERR] Could not find a ticker column in {excel_path}:{sheet_name}. "
                             f"Pass --ticker_col or rename one to 'ticker'.")
        col = candidates[0]

    vals = (
        df[col].dropna().astype(str).str.strip()
          .replace({"": None}).dropna().unique().tolist()
    )
    if not vals:
        raise SystemExit(f"[ERR] No tickers found in {excel_path}:{sheet_name} (col={col})")
    return vals


def rank_df(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    df_f = df[df["num_trades"] >= min_trades].copy()
    df_s = df_f.sort_values(by=["total_return", "num_trades", "win_rate"],
                            ascending=[False, False, False]).reset_index(drop=True)
    if not df_s.empty:
        df_s.insert(0, "rank", df_s.index + 1)
    return df_s


def _sample_combos(combos, n, seed):
    """Deterministic sampling of combos (without replacement)."""
    if not n or n <= 0 or n >= len(combos):
        return combos
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(combos), size=n, replace=False)
    return [combos[i] for i in sorted(idx)]


def _build_suffix(ts: str, sample_combos: int, sample_scope: str, seed: int) -> str:
    parts = [ts]
    if sample_combos and sample_combos > 0:
        parts.append(f"S{sample_combos}-{sample_scope}-seed{seed}")
    return "_".join(parts)


def _row_params_dict(row: pd.Series, param_keys: list[str]) -> dict:
    out = {}
    for k in param_keys:
        if k in row:
            v = row[k]
            try:
                if pd.isna(v):
                    out[k] = None
                else:
                    out[k] = v.item() if hasattr(v, "item") else v
            except Exception:
                out[k] = None
        else:
            out[k] = None
    return out


# ---------------- main ----------------

def main():
    import argparse, runpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", type=str, default=str(EXCEL_DEFAULT))
    ap.add_argument("--sheet", type=str, default=None,  # auto: signals / signalsUSD
                    help="Defaults to project-aware sheet (signals / signalsUSD)")
    ap.add_argument("--ticker_col", type=str, default=None)
    ap.add_argument("--engine_path", type=str, default=str(SCRIPT_DIR / "s50_30m_backtest_engine.py"),
                    help="Path to the s50 engine file to import (run_path).")
    ap.add_argument("--min_trades", type=int, default=3)
    ap.add_argument("--max_combos", type=int, default=0,
                    help="Hard cap: take only the first N combos from the full grid (0 = no cap).")
    ap.add_argument("--n_jobs_tickers", type=int, default=1)
    ap.add_argument("--n_jobs_combos", type=int, default=-1)
    ap.add_argument("--top_n_per_ticker", type=int, default=5)
    ap.add_argument("--limit_tickers", type=int, default=0)
    ap.add_argument("--skip_tickers", type=int, default=0,
                    help="Skip the first N tickers before applying --limit_tickers")
    ap.add_argument("--sample_combos", type=int, default=0,
                    help="Randomly sample N combos (0 = use all)")
    ap.add_argument("--sample_scope", choices=["global","per_ticker"], default="global",
                    help="global = sample once; per_ticker = sample independently per ticker")
    ap.add_argument("--seed", type=int, default=42)

    # pass-through engine overrides
    ap.add_argument("--start_date", type=str, default=None)
    ap.add_argument("--end_date", type=str, default=None)
    ap.add_argument("--fees_fixed", type=float, default=3.0)
    ap.add_argument("--slip_bps", type=float, default=0.0)
    ap.add_argument("--cooldown_days", type=int, default=None)
    args = ap.parse_args()

    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        raise SystemExit(f"[ERR] Engine file not found: {engine_path}")

    # load s50 symbols
    _engine_ns = runpy.run_path(str(engine_path))  # executes file, does NOT run main()
    try:
        backtest_ticker = _engine_ns["backtest_ticker"]
        summarize_trades_per_ticker = _engine_ns["summarize_trades_per_ticker"]
    except KeyError as e:
        raise ImportError(
            f"Expected function {e} not found in {engine_path}. "
            "Open that file and confirm the function names are defined at module level."
        )

    # tickers
    excel_path = Path(args.excel_path)
    sheet = args.sheet or default_etf_sheet()
    _banner(excel_path, sheet, engine_path)

    tickers = load_tickers_from_excel(excel_path, sheet, args.ticker_col)
    if args.skip_tickers > 0:
        tickers = tickers[args.skip_tickers:]
    if args.limit_tickers > 0:
        tickers = tickers[: args.limit_tickers]
    print(f"[s60] Tickers      : {len(tickers)} selected")

    # full grid
    param_keys, param_vals = load_param_grid(PARAM_FILE)
    combos_full = list(itertools.product(*param_vals))
    if args.max_combos and args.max_combos > 0:
        combos_full = combos_full[: args.max_combos]

    # sampling strategy
    if args.sample_scope == "global":
        combos_for_all = _sample_combos(combos_full, args.sample_combos, args.seed)
        print(f"[s60] Combos use   : {len(combos_for_all)} (global)")
    else:
        combos_for_all = combos_full
        print(f"[s60] Combos scope : per_ticker (sample {args.sample_combos} each)")

    ts_global = _ts_utc()
    suffix = _build_suffix(ts_global, args.sample_combos, args.sample_scope, args.seed)

    engine_overrides = {
        "verbose": False,
        "fees_fixed": args.fees_fixed,
        "slip_bps": args.slip_bps,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    if args.cooldown_days is not None:
        engine_overrides["cooldown_days"] = args.cooldown_days

    # containers for JSON summary
    json_best_rows: list[tuple[str, pd.Series]] = []

    def run_one_combo_for_ticker(combo_id: int,
                                 keys: list[str],
                                 combo: tuple,
                                 ticker: str,
                                 suffix: str,
                                 trades_subdir: Path,
                                 engine_overrides: dict) -> dict:
        params = dict(zip(keys, combo))
        params.update(engine_overrides)
        params.setdefault("verbose", False)

        trades_subdir.mkdir(parents=True, exist_ok=True)
        trade_file = trades_subdir / f"trades_{ticker}_combo{combo_id}_{suffix}.csv"

        try:
            trades = backtest_ticker(ticker, params)
        except Exception as e:
            pd.DataFrame().to_csv(trade_file, index=False)  # artifact for traceability
            return {
                "combo_id": combo_id, "ticker": ticker, **params,
                "num_trades": 0, "win_rate": 0.0, "total_return": 0.0,
                "end_capital": UNIT_CAPITAL, "trades_file": trade_file.name, "error": str(e),
            }

        if trades is not None and not trades.empty:
            trades.to_csv(trade_file, index=False)
            cap_df = summarize_trades_per_ticker(trades, unit_capital=UNIT_CAPITAL)
            row = cap_df[cap_df["ticker"] == ticker].iloc[0]
            return {
                "combo_id": combo_id, "ticker": ticker, "total_return": float(row["total_return"]),
                **params, "num_trades": int(row["num_trades"]), "win_rate": float(row["win_rate"]),
                "end_capital": float(row["end_capital"]), "trades_file": trade_file.name, "error": "",
            }

        # no trades
        pd.DataFrame().to_csv(trade_file, index=False)
        return {
            "combo_id": combo_id, "ticker": ticker, **params,
            "num_trades": 0, "win_rate": 0.0, "total_return": 0.0,
            "end_capital": UNIT_CAPITAL, "trades_file": trade_file.name, "error": "",
        }

    def sweep_one_ticker(tkr: str):
        # per-ticker combo set (if sampling per ticker)
        if args.sample_scope == "per_ticker":
            salted_seed = args.seed + (abs(hash(tkr)) % 10_000_000)
            combos_local = _sample_combos(combos_full, args.sample_combos, salted_seed)
        else:
            combos_local = combos_for_all

        trades_subdir = TRADES_DIR / tkr
        iterator = enumerate(combos_local, start=1)
        tasks = (
            delayed(run_one_combo_for_ticker)(combo_id, param_keys, combo, tkr, suffix, trades_subdir, engine_overrides)
            for combo_id, combo in iterator
        )

        if TQDM_AVAILABLE:
            with tqdm_joblib(tqdm(total=len(combos_local), desc=f"{tkr} sweep", unit="combo")):
                results = Parallel(n_jobs=args.n_jobs_combos, backend="loky")(list(tasks))
        else:
            results = Parallel(n_jobs=args.n_jobs_combos, backend="loky")(list(tasks))

        df_all = pd.DataFrame(results)
        df_ranked = rank_df(df_all, min_trades=args.min_trades)

        out_ranked = OUT_DIR / f"param_results_{tkr}_{suffix}.csv"
        out_all = OUT_DIR / f"param_results_{tkr}_{suffix}_ALL.csv"
        df_ranked.to_csv(out_ranked, index=False)
        df_all.to_csv(out_all, index=False)

        if not df_ranked.empty:
            print(f"\n[{tkr}] Top 10:")
            print(df_ranked[["rank","combo_id","total_return","end_capital","num_trades","win_rate","trades_file"]]
                  .head(10).to_string(index=False))
            best = df_ranked.iloc[0].copy()
            json_best_rows.append((tkr, best))
        else:
            print(f"\n[{tkr}] [WARN] No combos met the min_trades={args.min_trades} filter.")

        return tkr, df_all, df_ranked

    # ---- outer parallel across tickers
    if TQDM_AVAILABLE:
        with tqdm_joblib(tqdm(total=len(tickers), desc="Tickers", unit="tkr")):
            results_by_ticker = Parallel(n_jobs=args.n_jobs_tickers, backend="loky")(
                delayed(sweep_one_ticker)(tkr) for tkr in tickers
            )
    else:
        results_by_ticker = Parallel(n_jobs=args.n_jobs_tickers, backend="loky")(
            delayed(sweep_one_ticker)(tkr) for tkr in tickers
        )

    # ---- global leaderboard (top N per ticker)
    global_rows = []
    for tkr, _df_all, df_ranked in results_by_ticker:
        if not df_ranked.empty:
            g = df_ranked.copy()
            g["ticker"] = tkr
            global_rows.append(g.head(args.top_n_per_ticker))

    if global_rows:
        global_board = pd.concat(global_rows, ignore_index=True)
        global_board = global_board.sort_values(
            by=["total_return", "num_trades", "win_rate"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        global_file = OUT_DIR / f"param_results_ALL_{suffix}.csv"
        global_board.to_csv(global_file, index=False)
        print(f"\n[OK] Global leaderboard → {global_file}")
    else:
        print("\n[WARN] No ticker produced rows meeting the min_trades filter; global leaderboard skipped.")

    # ---- best combos JSON (one per ticker)
    if json_best_rows:
        final_obj = {
            "created_utc": ts_global,
            "excel_path": str(excel_path),
            "sheet": sheet,
            "param_file": str(PARAM_FILE),
            "seed": int(args.seed),
            "sample_combos": int(args.sample_combos or 0),
            "sample_scope": args.sample_scope,
            "max_combos": int(args.max_combos or 0),
            "min_trades": int(args.min_trades),
            "n_jobs_tickers": int(args.n_jobs_tickers),
            "n_jobs_combos": int(args.n_jobs_combos),
            "suffix": suffix,
            "combos": []
        }

        for tkr, row in json_best_rows:
            params_dict = _row_params_dict(row, param_keys)
            final_obj["combos"].append({
                "ticker": tkr,
                "combo_id": int(row.get("rank", 1)),  # rank 1 row
                "params": params_dict,
                "metrics": {
                    "total_return": float(row.get("total_return", 0.0)),
                    "win_rate": float(row.get("win_rate", 0.0)),
                    "num_trades": int(row.get("num_trades", 0)),
                    "end_capital": float(row.get("end_capital", UNIT_CAPITAL)),
                },
                "trades_file": row.get("trades_file", ""),
                "run_id": suffix
            })

        json_path = OUT_DIR / f"s60_final_combos_{ts_global}.json"
        with open(json_path, "w") as f:
            json.dump(final_obj, f, indent=2)
        print(f"[OK] Final best-combos JSON → {json_path}")
    else:
        print("[WARN] No best rows captured; s60_final_combos.json not written.")

    print(f"[OK] Per-ticker results saved in → {OUT_DIR}")
    print(f"[OK] All trades CSVs saved in → {TRADES_DIR}")


if __name__ == "__main__":
    main()