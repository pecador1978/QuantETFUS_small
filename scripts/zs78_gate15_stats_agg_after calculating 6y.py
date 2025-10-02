#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s78_gate15_stats_agg.py — Build Gate 1.5 historical stats cache (no ML).

Inputs
------
- signals/rule_live_signals_*.csv          (produced by s77)
  Required columns per row:
    ticker, decision, side, setup_fingerprint, m30_close, m30_ema_base
  NOTE: rows without setup_fingerprint are skipped.

- data_enriched/30min/{TICKER}.parquet     (from s32)
  Used to compute entry and future closes by local day.

Environment / CLI
-----------------
- MARKET_TZ env or --market-tz (default Europe/London)
- --horizon-days  Fixed evaluation horizon in local days (default: 10)
- --signals-dir   (default: P.ROOT/signals)
- --out           (default: signals/gate15_stats_cache.parquet)
- --min_rows      Minimum historical rows required per (ticker,fingerprint) to emit stats (default: 8)

Outputs
-------
- signals/gate15_stats_cache.parquet with columns:
    ticker, setup_fingerprint,
    hist_win_rate, avg_return_pct,
    trend_duration_days_min, trend_duration_days_avg, trend_duration_days_max,
    sample_size, horizon_days
"""

from __future__ import annotations
from pathlib import Path
import argparse, sys, os, re
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P

def _market_tz(cli_tz: str | None) -> str:
    if cli_tz: return cli_tz
    tz = os.environ.get("MARKET_TZ","").strip()
    return tz or "Europe/London"

def _read_boards_parquet(ds_dir: Path) -> pd.DataFrame:
    if not ds_dir.exists():
        raise SystemExit(f"[ERR] boards_ds not found: {ds_dir}")
    df = pd.read_parquet(ds_dir)  # reads entire dataset
    # expect columns from s77 + board_ts_utc + board_day
    if "setup_fingerprint" not in df.columns:
        raise SystemExit("[ERR] boards_ds missing setup_fingerprint; rerun s77 with Step 1 changes.")
    # local date for grouping = board_day (string YYYYMMDD → date)
    df["board_day"] = pd.to_datetime(df["board_day"], format="%Y%m%d").dt.date
    return df

def _daily_last_close_by_local(df30: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    local = df30["datetime"].dt.tz_convert(market_tz)
    g = df30.assign(date_local=local.dt.date).groupby("date_local", as_index=False)["close"].last()
    return g.sort_values("date_local").reset_index(drop=True)

def _read_parquet_30m(tkr: str) -> pd.DataFrame | None:
    p = P.DATA_ENRICHED / "30min" / f"{tkr}.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df.columns = [c.strip().lower() for c in df.columns]
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None: dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["datetime","close"]).sort_values("datetime").reset_index(drop=True)

def _entry_future_ret(daily_df: pd.DataFrame, entry_date, horizon_days: int) -> tuple[float,float]:
    r = daily_df[daily_df["date_local"] == entry_date]
    if r.empty: return (np.nan, np.nan)
    entry_close = float(r["close"].iloc[0])
    all_dates = daily_df["date_local"].tolist()
    try: idx = all_dates.index(entry_date)
    except ValueError: return (entry_close, np.nan)
    fut_idx = idx + horizon_days
    if fut_idx >= len(all_dates): return (entry_close, np.nan)
    future_close = float(daily_df.iloc[fut_idx]["close"])
    return (entry_close, future_close)

def main():
    ap = argparse.ArgumentParser(description="Gate 1.5 stats from boards_ds parquet (no CSV glob).")
    ap.add_argument("--boards-ds", type=str, default=str(P.ROOT / "signals" / "boards_ds"))
    ap.add_argument("--market-tz", type=str, default=None)
    ap.add_argument("--horizon-days", type=int, default=10)
    ap.add_argument("--min-rows", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(P.ROOT / "signals" / "gate15_stats_cache.parquet"))
    args = ap.parse_args()

    market_tz = _market_tz(args.market_tz)
    S = _read_boards_parquet(Path(args.boards_ds))
    S = S[S["decision"].isin(["BUY","SELL"])].copy()
    if S.empty: raise SystemExit("[ERR] No BUY/SELL rows in boards_ds.")

    # Use board_day as the local date key
    S["date_local"] = S["board_day"]

    out_rows = []
    for tkr, G in S.groupby("ticker"):
        df30 = _read_parquet_30m(str(tkr))
        if df30 is None: continue
        daily = _daily_last_close_by_local(df30, market_tz)
        if daily.empty: continue
        for fp, df_fp in G.groupby("setup_fingerprint"):
            per_day = df_fp.sort_values("date_local").drop_duplicates("date_local", keep="last")
            if len(per_day) < args.min_rows: continue

            wins, rets, durs = [], [], []
            day_side = per_day[["date_local","side"]].drop_duplicates("date_local", keep="last")
            for _, row in per_day.iterrows():
                entry_date = row["date_local"]
                side = str(row["side"]).upper()
                e, f = _entry_future_ret(daily, entry_date, args.horizon_days)
                if not (np.isfinite(e) and np.isfinite(f)): continue
                r = (f/e - 1.0)
                if side == "SHORT": r = -r
                rets.append(100.0 * r)
                wins.append(1.0 if r > 0.0 else 0.0)
                # crude duration until side change
                after = day_side[day_side["date_local"] > entry_date]
                dur = 1
                last = side
                for _, r2 in after.iterrows():
                    s2 = str(r2["side"]).upper()
                    if s2 != last and s2 in ("LONG","SHORT",""): break
                    dur += 1
                durs.append(dur)

            if not rets: continue
            out_rows.append({
                "ticker": str(tkr).upper(),
                "setup_fingerprint": fp,
                "hist_win_rate": round(float(np.mean(wins))*100.0, 2),
                "avg_return_pct": round(float(np.mean(rets)), 2),
                "trend_duration_days_min": int(np.min(durs)) if durs else 0,
                "trend_duration_days_avg": round(float(np.mean(durs)), 2) if durs else 0.0,
                "trend_duration_days_max": int(np.max(durs)) if durs else 0,
                "sample_size": int(len(rets)),
                "horizon_days": int(args.horizon_days),
            })

    if not out_rows:
        raise SystemExit("[ERR] No aggregates produced from boards_ds.")
    OUT = pd.DataFrame(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OUT.to_parquet(out_path, index=False)
    print(f"[OK] Gate 1.5 stats cache → {out_path} (rows={len(OUT)})")

if __name__ == "__main__":
    main()