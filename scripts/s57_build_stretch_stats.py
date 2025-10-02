#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s57_build_stretch_stats.py — Per-ETF historical stretch & short-horizon stats (ALL history, longs only)

Inputs
------
- Per-ticker 30m parquet from s32:
    P.DATA_ENRICHED/30min/{TICKER}.parquet
  Required columns: datetime (UTC), close, ema20_d, ema44_d

What this computes (per ETF)
----------------------------
1) Stretch vs EMA20/EMA44 until first touch:
   - stretch_p50_vs_ema20, stretch_p80_vs_ema20, stretch_max_vs_ema20, stretch_std_vs_ema20
   - stretch_p50_vs_ema44, stretch_p80_vs_ema44, stretch_max_vs_ema44, stretch_std_vs_ema44
   Definition: from a long-regime entry (see below), walk forward until the FIRST touch of EMA20 (or EMA44).
               Within that window, take max( (close/EMA - 1)*100 ).

2) Time-to-touch (bars, daily bars):
   - median_bars_until_touch_ema20, median_bars_until_touch_ema44

3) Forward returns from entry close for k in {1..5} days:
   - p_drop_next{k}d               (fraction of events where r_{+k} < 0)
   - avg_drop_next{k}d             (mean of r_{+k} < 0 only; NaN if none)
   - avg_gain_next{k}d             (mean of r_{+k} > 0 only; NaN if none)
   - median_abs_move_next{k}d      (median |r_{+k}|)
   - median_return_next{k}d        (median r_{+k})
   Additionally for k=5:
   - ret_p25_next5d, ret_p75_next5d

Long-regime entry proxy (no look-ahead, daily close basis)
----------------------------------------------------------
Enter on the first day a condition flips FALSE -> TRUE:
    (EMA20_d > EMA44_d) AND (Close_d > EMA44_d) AND (slope(EMA20_d) > 0) AND (slope(EMA44_d) > 0)
This approximates your Stage-B/C "bullish regime" without importing all s77 gates.

Outputs
-------
- Parquet table at:
    P.ROOT/signals/prob_models/stretch_stats.parquet
  One row per ETF with: ticker, stage="ALL", window_start, window_end, n_events, metrics above, n_eff_after_pool, shrinkage_weight, rules_fingerprint, generated_at_utc

Notes
-----
- Uses MARKET_TZ (default Europe/London) for daily grouping to avoid look-ahead.
- Safe to run daily/weekly; it rewrites the full file.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os
import numpy as np
import pandas as pd

# ---------- project paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # expects ROOT, DATA_ENRICHED, etc.


# ---------- helpers ----------
def _ts_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _fingerprint_rules() -> str:
    # Tag this computation; replace with hash(rules.json) if you later bind it to exact s77 config.
    return "stretch_v1_long_proxy_allhistory"

def _load_30m(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path.name}: missing columns ['datetime','close']")
    # required daily overlays:
    for c in ("ema20_d", "ema44_d"):
        if c not in df.columns:
            raise ValueError(f"{path.name}: missing required daily overlay '{c}' (run s32)")
    # timestamps
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def _daily_lastbars(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    """
    Collapse 30m to completed local-day last bar snapshots.
    Returns columns: ['datetime','date_local','d_close','d_ema20','d_ema44']
    """
    local = df["datetime"].dt.tz_convert(market_tz)
    g = df.assign(date_local=local.dt.date)
    dlast = (g.groupby("date_local", as_index=False)
               .tail(1)
               .sort_values("date_local")
               .reset_index(drop=True))
    dlast = dlast.rename(columns={"close": "d_close", "ema20_d": "d_ema20", "ema44_d": "d_ema44"})
    return dlast[["datetime", "date_local", "d_close", "d_ema20", "d_ema44"]]

def _collect_events_and_measures(dlast: pd.DataFrame) -> dict:
    """
    Detect long-regime entries; for each, compute:
      - stretch until first EMA20/EMA44 touch (max distance %, two series)
      - bars until first EMA20/EMA44 touch (integers)
      - forward returns r_{+k} for k=1..5 (series)
    Returns dict of lists; caller aggregates.
    """
    out = {
        "stretch20_list": [],
        "stretch44_list": [],
        "bars_to_touch20": [],
        "bars_to_touch44": [],
        "fwd_returns": {k: [] for k in range(1, 6)}
    }
    if len(dlast) < 40:
        return out

    # add daily slopes (no look-ahead)
    dlast = dlast.copy()
    dlast["d_ema20_shift1"] = dlast["d_ema20"].shift(1)
    dlast["d_ema44_shift1"] = dlast["d_ema44"].shift(1)
    dlast["sl20"] = (dlast["d_ema20"] / dlast["d_ema20_shift1"] - 1.0)
    dlast["sl44"] = (dlast["d_ema44"] / dlast["d_ema44_shift1"] - 1.0)

    # long regime condition per completed day
    cond = (
        (dlast["d_ema20"] > dlast["d_ema44"]) &
        (dlast["d_close"]  > dlast["d_ema44"]) &
        (dlast["sl20"] > 0) &
        (dlast["sl44"] > 0)
    )
    enter_idx = np.where((~cond.shift(1, fill_value=False)) & (cond))[0].tolist()
    if not enter_idx:
        return out

    n_rows = len(dlast)
    closes = dlast["d_close"].to_numpy()
    ema20  = dlast["d_ema20"].to_numpy()
    ema44  = dlast["d_ema44"].to_numpy()

    for ix in enter_idx:
        # --- EMA20 stretch & time ---
        j = ix
        max_s20 = 0.0
        while j < n_rows:
            c = closes[j]; e20 = ema20[j]
            if not (np.isfinite(c) and np.isfinite(e20) and e20 > 0):
                break
            max_s20 = max(max_s20, (c / e20 - 1.0) * 100.0)
            if c <= e20:
                out["stretch20_list"].append(max_s20)
                out["bars_to_touch20"].append(j - ix + 1)  # include entry day
                break
            j += 1

        # --- EMA44 stretch & time ---
        j = ix
        max_s44 = 0.0
        while j < n_rows:
            c = closes[j]; e44 = ema44[j]
            if not (np.isfinite(c) and np.isfinite(e44) and e44 > 0):
                break
            max_s44 = max(max_s44, (c / e44 - 1.0) * 100.0)
            if c <= e44:
                out["stretch44_list"].append(max_s44)
                out["bars_to_touch44"].append(j - ix + 1)
                break
            j += 1

        # --- forward returns r_{+k}, k=1..5 (total return from entry close) ---
        base = closes[ix]
        if np.isfinite(base) and base > 0:
            for k in range(1, 6):
                t = ix + k
                if t < n_rows and np.isfinite(closes[t]):
                    out["fwd_returns"][k].append(closes[t] / base - 1.0)
                else:
                    # not enough future data: drop this event for this k horizon
                    pass

    return out

def _safe_percentile(arr: list[float], p: float) -> float:
    return float(np.percentile(arr, p)) if arr else float("nan")

def _safe_mean(arr: list[float]) -> float:
    return float(np.mean(arr)) if arr else float("nan")

def _safe_median(arr: list[float]) -> float:
    return float(np.median(arr)) if arr else float("nan")

def _std(arr: list[float]) -> float:
    return float(np.std(arr, ddof=1)) if len(arr) >= 2 else (0.0 if len(arr) == 1 else float("nan"))

def _aggregate_one_ticker(tkr: str, dlast: pd.DataFrame) -> dict:
    m = _collect_events_and_measures(dlast)
    s20, s44 = m["stretch20_list"], m["stretch44_list"]
    t20, t44 = m["bars_to_touch20"], m["bars_to_touch44"]
    fwd = m["fwd_returns"]  # dict k->list

    # Stretch aggregates
    row = {
        "ticker": tkr,
        "stage": "ALL",
        "window_start": dlast["date_local"].min() if len(dlast) else pd.NaT,
        "window_end":   dlast["date_local"].max() if len(dlast) else pd.NaT,
        "n_events": int(max(len(s20), len(s44), len(fwd[1]))),  # rough count of usable entries
        "stretch_p50_vs_ema20": _safe_percentile(s20, 50),
        "stretch_p80_vs_ema20": _safe_percentile(s20, 80),
        "stretch_max_vs_ema20": max(s20) if s20 else float("nan"),
        "stretch_std_vs_ema20": _std(s20),
        "stretch_p50_vs_ema44": _safe_percentile(s44, 50),
        "stretch_p80_vs_ema44": _safe_percentile(s44, 80),
        "stretch_max_vs_ema44": max(s44) if s44 else float("nan"),
        "stretch_std_vs_ema44": _std(s44),
        "median_bars_until_touch_ema20": float(_safe_median(t20)),
        "median_bars_until_touch_ema44": float(_safe_median(t44)),
    }

    # Forward return aggregates, k = 1..5
    for k in range(1, 6):
        rk = [float(x) for x in fwd[k]]
        neg = [x for x in rk if x < 0]
        pos = [x for x in rk if x > 0]
        row[f"p_drop_next{ k }d"]           = (len(neg) / len(rk)) if len(rk) else float("nan")
        row[f"avg_drop_next{ k }d"]         = _safe_mean(neg)
        row[f"avg_gain_next{ k }d"]         = _safe_mean(pos)
        row[f"median_abs_move_next{ k }d"]  = _safe_median([abs(x) for x in rk])
        row[f"median_return_next{ k }d"]    = _safe_median(rk)

    # 5d percentiles envelope
    r5 = [float(x) for x in fwd[5]]
    row["ret_p25_next5d"] = _safe_percentile(r5, 25)
    row["ret_p75_next5d"] = _safe_percentile(r5, 75)

    # diagnostics / metadata
    nevents = row["n_events"]
    row["n_eff_after_pool"]   = float(nevents)      # placeholder; no pooling yet
    row["shrinkage_weight"]   = 0.0                 # placeholder
    row["rules_fingerprint"]  = _fingerprint_rules()
    row["generated_at_utc"]   = _ts_utc_str()

    return row


def main():
    market_tz = os.environ.get("MARKET_TZ", "Europe/London")
    out_dir = P.ROOT / "signals" / "prob_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stretch_stats.parquet"

    m30_dir = P.DATA_ENRICHED / "30min"
    if not m30_dir.exists():
        raise SystemExit(f"[ERR] {m30_dir} not found. Run s32 first.")
    parqs = sorted(m30_dir.glob("*.parquet"))
    if not parqs:
        raise SystemExit(f"[ERR] No parquet files in {m30_dir}")

    rows = []
    for pq in parqs:
        tkr = pq.stem.upper().split('.', 1)[0]
        try:
            df = _load_30m(pq)
            dlast = _daily_lastbars(df, market_tz)
            row = _aggregate_one_ticker(tkr, dlast)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")

    if not rows:
        raise SystemExit("[ERR] No stats produced.")

    X = pd.DataFrame(rows)
    if 'window_start' in X.columns:
        X['window_start'] = pd.to_datetime(X['window_start'], errors='coerce')
    X.to_parquet(out_path, index=False)
    print(f"[OK] Stretch & short-horizon stats → {out_path}  (rows={len(X)})")
    csv_path = out_path.with_suffix(".csv")
    X.to_csv(csv_path, index=False)
    print(f"[OK] CSV export → {csv_path}")


if __name__ == "__main__":
    main()