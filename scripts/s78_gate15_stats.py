#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s78_gate15_stats.py — Gate-1.5 stats + confidence overlay (+ stretch_regime) for today's signals

What this does
--------------
1) Loads today's operator signals (default: latest s77 operator CSV).
2) LEFT-JOINs per-ETF stretch benchmarks from s57 (stretch_stats.parquet).
3) Computes:
      • stretch_regime: below_p50 / p50_to_p80 / above_p80 / unknown
      • confidence_score_0_100 (transparent, weighted)
      • confidence_bucket: HIGH (≥70) / MED (50–69) / LOW (<50)
4) Writes enriched outputs:
      P.ROOT / signals / analytics / gate15_stats.parquet
      P.ROOT / signals / analytics / gate15_stats.csv (latest + timestamped)

Usage
-----
  python scripts/s78_gate15_stats.py
  python scripts/s78_gate15_stats.py --operator-csv /path/to/operator_today_RULE_all_YYYYMMDD_HHMM.csv
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import sys

import numpy as np
import pandas as pd

# ---------- project paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # expects ROOT, SIGNALS, etc.


def _latest_operator_csv() -> Path | None:
    sigdir = P.SIGNALS_DIR
    pats = sorted(sigdir.glob("operator_today_RULE_all_*.csv"))
    return pats[-1] if pats else None


def _read_operator(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    needed = ["ticker", "decision", "side", "strength_score", "strength20_pct", "strength44_pct"]
    for n in needed:
        if n not in df.columns:
            raise ValueError(f"{path.name}: missing required column '{n}' (check s77 output).")
    return df


def _read_stretch_stats() -> pd.DataFrame | None:
    p = P.SIGNALS_DIR / "prob_models" / "stretch_stats.parquet"
    if not p.exists():
        print(f"[WARN] stretch stats not found: {p}")
        return None
    df = pd.read_parquet(p)
    keep = [
        "ticker", "stage",
        "stretch_p50_vs_ema20", "stretch_p80_vs_ema20",
        "stretch_p50_vs_ema44", "stretch_p80_vs_ema44",
        "median_bars_until_touch_ema20", "median_bars_until_touch_ema44",
        "p_drop_next1d", "p_drop_next3d", "p_drop_next5d",
        "ret_p25_next5d", "ret_p75_next5d",
    ]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    # keep most recent stage per ticker (if multiple)
    df = df.sort_values(["ticker", "stage"]).drop_duplicates(subset=["ticker"], keep="last")
    return df[keep]


# -------- scoring components --------
def _stretch_component(str20: float, p50: float, p80: float) -> float:
    """100 if strength20 < p50; 50 if p50≤strength20<p80; 0 if ≥p80; neutral 50 if missing."""
    if not np.isfinite(str20) or not np.isfinite(p50) or not np.isfinite(p80):
        return 50.0
    if str20 < p50:
        return 100.0
    if str20 < p80:
        return 50.0
    return 0.0


def _adx_component(adx: float) -> float:
    """Linear 0..100 between ADX 10..40; clip; neutral 50 if missing."""
    if not np.isfinite(adx):
        return 50.0
    val = (adx - 10.0) / (40.0 - 10.0) * 100.0
    return float(np.clip(val, 0.0, 100.0))


def _rsi_component(rsi: float) -> float:
    """
    RSI sweet spot 55–65 → 100
    65–70 → 60
    50–55 → 70
    >70 → 20
    <50 → 40
    neutral 60 if missing
    """
    if not np.isfinite(rsi):
        return 60.0
    if 55.0 <= rsi <= 65.0:
        return 100.0
    if 65.0 < rsi <= 70.0:
        return 60.0
    if 50.0 <= rsi < 55.0:
        return 70.0
    if rsi > 70.0:
        return 20.0
    return 40.0


def _donchian_component(width_pct: float) -> float:
    """Linear 0..100 between width 2%..4%; neutral 50 if missing."""
    if not np.isfinite(width_pct):
        return 50.0
    val = (width_pct - 2.0) / (4.0 - 2.0) * 100.0
    return float(np.clip(val, 0.0, 100.0))


def _compute_confidence(df: pd.DataFrame) -> pd.DataFrame:
    str20 = df.get("strength20_pct")
    p50   = df.get("stretch_p50_vs_ema20")
    p80   = df.get("stretch_p80_vs_ema20")
    adx   = df.get("adx14_d") if "adx14_d" in df.columns else pd.Series(np.nan, index=df.index)
    rsi   = df.get("rsi14_d") if "rsi14_d" in df.columns else pd.Series(np.nan, index=df.index)
    dchw  = df.get("donchian_width_pct") if "donchian_width_pct" in df.columns else pd.Series(np.nan, index=df.index)

    stretch_comp  = [_stretch_component(a, b, c) for a, b, c in zip(str20, p50, p80)]
    adx_comp      = [_adx_component(x) for x in adx]
    rsi_comp      = [_rsi_component(x) for x in rsi]
    donchian_comp = [_donchian_component(x) for x in dchw]

    out = df.copy()
    out["conf_stretch"]  = stretch_comp
    out["conf_adx"]      = adx_comp
    out["conf_rsi"]      = rsi_comp
    out["conf_donchian"] = donchian_comp

    out["confidence_score_0_100"] = (
        0.40 * out["conf_stretch"] +
        0.30 * out["conf_adx"] +
        0.20 * out["conf_rsi"] +
        0.10 * out["conf_donchian"]
    ).round(1)

    def _bucket(x: float) -> str:
        if not np.isfinite(x):
            return "MED"
        if x >= 70.0:
            return "HIGH"
        if x >= 50.0:
            return "MED"
        return "LOW"

    out["confidence_bucket"] = out["confidence_score_0_100"].apply(_bucket)
    return out


def _attach_stretch_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    stretch_regime based on today's strength20 vs per-ETF p50/p80 benchmarks:
      - below_p50   : strength20 <  p50
      - p50_to_p80  : p50 <= strength20 < p80
      - above_p80   : strength20 >= p80
      - unknown     : if any inputs missing
    """
    out = df.copy()
    s = out.get("strength20_pct")
    p50 = out.get("stretch_p50_vs_ema20")
    p80 = out.get("stretch_p80_vs_ema20")
    regime = []
    for a, b, c in zip(s, p50, p80):
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
            regime.append("unknown")
        elif a < b:
            regime.append("below_p50")
        elif a < c:
            regime.append("p50_to_p80")
        else:
            regime.append("above_p80")
    out["stretch_regime"] = regime
    return out


def main():
    ap = argparse.ArgumentParser(description="Gate-1.5 stats + confidence overlay (+ stretch_regime) for today's signals")
    ap.add_argument("--operator-csv", type=str, default=None,
                    help="Path to operator_today_RULE_all_*.csv (defaults to latest)")
    args = ap.parse_args()

    # operator file
    if args.operator_csv:
        op_path = Path(args.operator_csv)
        if not op_path.exists():
            raise SystemExit(f"[ERR] operator CSV not found: {op_path}")
    else:
        op_path = _latest_operator_csv()
        if op_path is None:
            raise SystemExit("[ERR] No operator_today_RULE_all_*.csv found in signals/")

    op = _read_operator(op_path)

    # --- normalize decision & side robustly (avoid .str on non-strings) ---
    if "decision" not in op.columns:
        op["decision"] = ""
    op["decision"] = op["decision"].astype("string").str.upper().fillna("")

    if "side" not in op.columns:
        op["side"] = np.where(op["decision"].eq("BUY"), "LONG",
                              np.where(op["decision"].eq("SELL"), "SHORT", ""))
    else:
        op["side"] = op["side"].astype("string").str.upper().fillna("")
        # If it's all blanks after coercion, infer from decision
        if (op["side"] == "").all():
            op["side"] = np.where(op["decision"].eq("BUY"), "LONG",
                                  np.where(op["decision"].eq("SELL"), "SHORT", ""))

    # --- split rows ---
    longs  = op[op["decision"].eq("BUY")  & op["side"].eq("LONG")].copy()
    others = op[~(op["decision"].eq("BUY") & op["side"].eq("LONG"))].copy()

    # --- join stretch stats on longs ---
    stretch = _read_stretch_stats()
    stretch_cols = [
        "stretch_p50_vs_ema20","stretch_p80_vs_ema20",
        "stretch_p50_vs_ema44","stretch_p80_vs_ema44",
        "median_bars_until_touch_ema20","median_bars_until_touch_ema44",
        "p_drop_next1d","p_drop_next3d","p_drop_next5d",
        "ret_p25_next5d","ret_p75_next5d",
    ]
    if stretch is None:
        for c in stretch_cols:
            longs[c] = np.nan
    else:
        longs = longs.merge(stretch, on="ticker", how="left")

    # compute stretch_regime + confidence for longs
    longs = _attach_stretch_regime(longs)
    longs = _compute_confidence(longs)

    # non-longs: fill neutral NaNs for new fields
    if not others.empty:
        for c in stretch_cols + [
            "conf_stretch", "conf_adx", "conf_rsi", "conf_donchian",
            "confidence_score_0_100", "confidence_bucket", "stretch_regime",
        ]:
            others.loc[:, c] = np.nan

    # combine back
    out = pd.concat([longs, others], ignore_index=True)

    # sort: BUYs first by confidence, then others by strength_score
    out["__order"] = np.where(out["decision"] == "BUY", 0, 1)
    sort_cols = ["__order"]
    asc = [True]
    if "confidence_score_0_100" in out.columns:
        sort_cols.append("confidence_score_0_100")
        asc.append(False)
    if "strength_score" in out.columns:
        sort_cols.append("strength_score")
        asc.append(False)

    out = out.sort_values(sort_cols, ascending=asc).drop(columns="__order")

    # outputs
    out_dir = P.SIGNALS_DIR / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    pq_path = out_dir / "gate15_stats.parquet"
    csv_path_ts = out_dir / f"gate15_stats_{ts}.csv"  # timestamped
    csv_path_latest = out_dir / "gate15_stats.csv"    # latest

    out.to_parquet(pq_path, index=False)
    out.to_csv(csv_path_ts, index=False)
    out.to_csv(csv_path_latest, index=False)

    print(f"[OK] Gate-1.5 stats (Parquet) → {pq_path}")
    print(f"[OK] Gate-1.5 stats (CSV latest) → {csv_path_latest}")
    print(f"[OK] Gate-1.5 stats (CSV ts) → {csv_path_ts}")
    if stretch is None:
        print("[WARN] Stretch benchmarks missing — confidence & stretch_regime computed with neutral stretch inputs.")
    else:
        print("[OK] Stretch benchmarks joined; confidence & stretch_regime computed.")


if __name__ == "__main__":
    main()