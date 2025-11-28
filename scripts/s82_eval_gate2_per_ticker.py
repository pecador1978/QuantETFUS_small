#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s82_eval_gate2_per_ticker.py — Per-ticker Gate-2 metrics (AUC, Brier, n_tr, n_va)
-----------------------------------------------------------------------------------
- Auto-detects s80 confidence CSV (project first, then QuantShared)
- Auto-detects 30m parquets (project first, then QuantShared)
- Builds forward-return labels (next H bars) per ticker
- Runs time-series CV per ticker with a simple calibrated logistic pipeline
- Reports per-ticker: auc_mean, auc_std, brier_mean, brier_std, folds, rows, per-fold n_tr/n_va

Output:
  /Users/Finance/QuantETFUS_small/reports/gate2_per_ticker_metrics.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---------- Defaults & discovery ----------
PROJECT_ROOT = Path("/Users/Finance/QuantETFUS_small")
QS_ROOT      = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs" / "QuantShared"

FEATURES = [
    "confidence_score",
    "RSI14","ADX14",
    "EMA5_slope","EMA20_slope","EMA44_slope","EMA260_slope",
    "Donchian_position","Volatility_ATR","Trend_alignment",
]

def find_conf_csv(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path).expanduser()
        if not p.exists():
            raise SystemExit(f"[ERR] --conf_csv not found: {p}")
        return p
    cands = list((PROJECT_ROOT / "data_enriched").glob("gate2_confidence_30m*.csv")) \
          + list((PROJECT_ROOT / "data_enriched").glob("gate2_confidence_30m*.csv.gz"))
    qs_dir = QS_ROOT / "data_enriched"
    if qs_dir.exists():
        cands += list(qs_dir.glob("gate2_confidence_30m*.csv")) \
               + list(qs_dir.glob("gate2_confidence_30m*.csv.gz"))
    if not cands:
        raise SystemExit("[ERR] No gate2_confidence_30m*.csv found in project or QuantShared.")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def find_parquet_dir(user_dir: str | None) -> Path:
    if user_dir:
        d = Path(user_dir).expanduser()
        if not d.exists():
            raise SystemExit(f"[ERR] --parquet_dir not found: {d}")
        return d
    # project first
    d1 = PROJECT_ROOT / "data_enriched" / "30min"
    if d1.exists():
        return d1
    # QuantShared fallback
    d2 = QS_ROOT / "data_enriched" / "30min"
    if d2.exists():
        return d2
    raise SystemExit("[ERR] 30min parquet dir not found in project or QuantShared.")

# ---------- IO ----------
def load_conf(conf_csv: Path) -> pd.DataFrame:
    header = pd.read_csv(conf_csv, nrows=0)
    need = ["datetime","ticker","confidence_score"] + [c for c in FEATURES if c != "confidence_score"]
    usecols = [c for c in need if c in header.columns]
    if len(set(["datetime","ticker","confidence_score"]) - set(usecols)):
        raise SystemExit("[ERR] s80 CSV missing required columns.")
    df = pd.read_csv(conf_csv, usecols=usecols, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime","ticker","confidence_score"]).sort_values(["ticker","datetime"])
    return df

def load_prices(parquet_dir: Path) -> pd.DataFrame:
    paths = sorted(parquet_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"[ERR] No parquets in {parquet_dir}")
    frames = []
    for p in paths:
        try:
            g = pd.read_parquet(p, columns=["datetime","close","ticker"])
        except Exception:
            g = pd.read_parquet(p)
        if "ticker" not in g.columns:
            g["ticker"] = p.stem.upper()
        g["datetime"] = pd.to_datetime(g["datetime"], utc=True, errors="coerce")
        g = g.dropna(subset=["datetime","close"])[["datetime","ticker","close"]].sort_values("datetime")
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(["ticker","datetime"])

def make_labels(prx: pd.DataFrame, horizon: int, thr: float) -> pd.DataFrame:
    df = prx.copy()
    df["close_fwd"] = df.groupby("ticker", sort=False)["close"].shift(-horizon)
    df["fwd_ret"]   = (df["close_fwd"]/df["close"]) - 1.0
    df["y"]         = (df["fwd_ret"] > thr).astype("Int64")
    return df.drop(columns=["close_fwd"])

# ---------- Model ----------
def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(
            solver="lbfgs", penalty="l2", C=1.0, max_iter=1000
        ))
    ])

def eval_one_ticker(df_t: pd.DataFrame, n_splits: int) -> dict:
    # df_t sorted by datetime
    X = df_t[FEATURES].astype(float).to_numpy()
    y = df_t["y"].astype(int).to_numpy()
    idx = np.arange(len(df_t))
    tss = TimeSeriesSplit(n_splits=n_splits)
    aucs, briers, n_tr_list, n_va_list = [], [], [], []
    for tr, va in tss.split(idx):
        model = build_model()
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:,1]
        try:
            auc = roc_auc_score(y[va], p)
        except Exception:
            auc = float("nan")
        try:
            brier = brier_score_loss(y[va], p)
        except Exception:
            brier = float("nan")
        aucs.append(auc); briers.append(brier)
        n_tr_list.append(len(tr)); n_va_list.append(len(va))
    return {
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std":  float(np.nanstd(aucs)),
        "brier_mean": float(np.nanmean(briers)),
        "brier_std":  float(np.nanstd(briers)),
        "folds": n_splits,
        "rows": int(len(df_t)),
        "per_fold_n_tr": ";".join(str(x) for x in n_tr_list),
        "per_fold_n_va": ";".join(str(x) for x in n_va_list),
    }

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-ticker Gate-2 evaluation (AUC, Brier, n_tr, n_va).")
    ap.add_argument("--conf_csv", help="Path to gate2_confidence_30m*.csv (.csv or .csv.gz)")
    ap.add_argument("--parquet_dir", help="Path to data_enriched/30min parquets")
    ap.add_argument("--horizon_bars", type=int, default=17, help="Forward horizon in 30m bars (≈1 trading day)")
    ap.add_argument("--profit_threshold", type=float, default=0.0, help="Return threshold for y=1 (e.g., 0.001 ~ 10bps)")
    ap.add_argument("--min_rows", type=int, default=3000, help="Minimum rows per ticker to evaluate")
    ap.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit folds")
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT / "reports" / "gate2_per_ticker_metrics.csv"),
                    help="Output CSV report path")
    ap.add_argument("--tickers", nargs="*", help="Optional subset of tickers to evaluate")
    return ap.parse_args()

# ---------- Main ----------
def main():
    args = parse_args()
    conf_csv = find_conf_csv(args.conf_csv)
    pq_dir   = find_parquet_dir(args.parquet_dir)

    print(f"[INFO] Using s80 CSV: {conf_csv}")
    print(f"[INFO] Using 30m parquets: {pq_dir}")

    # Load & merge features + prices -> labels
    df_feat = load_conf(conf_csv)
    df_px   = load_prices(pq_dir)
    df = df_feat.merge(df_px, on=["ticker","datetime"], how="inner").sort_values(["ticker","datetime"]).reset_index(drop=True)
    df = make_labels(df, horizon=args.horizon_bars, thr=args.profit_threshold).dropna(subset=["y"])

    # Optional subset
    if args.tickers:
        tickset = set(args.tickers)
        df = df[df["ticker"].isin(tickset)].copy()

    # Filter by min rows
    counts = df.groupby("ticker").size()
    keep = counts[counts >= args.min_rows].index
    df = df[df["ticker"].isin(keep)].copy()
    if df.empty:
        raise SystemExit("[ERR] No tickers meet min_rows; adjust --min_rows or check data.")

    # Evaluate per ticker
    rows = []
    for tkr, g in df.groupby("ticker"):
        g = g.sort_values("datetime").reset_index(drop=True)
        try:
            met = eval_one_ticker(g, n_splits=args.folds)
            met.update({
                "ticker": tkr,
                "horizon_bars": args.horizon_bars,
                "profit_threshold": args.profit_threshold,
            })
            print(f"[OK] {tkr}: AUC={met['auc_mean']:.4f}±{met['auc_std']:.3f}  "
                  f"Brier={met['brier_mean']:.4f}±{met['brier_std']:.3f}  rows={met['rows']}")
            rows.append(met)
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")

    if not rows:
        raise SystemExit("[ERR] No metrics produced.")

    out = Path(args.out_csv).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("auc_mean", ascending=False).to_csv(out, index=False, float_format="%.6f")
    print(f"✅ Per-ticker metrics → {out}")

if __name__ == "__main__":
    main()