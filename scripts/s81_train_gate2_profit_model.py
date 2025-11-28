#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s81_train_gate2_profit_model.py — Train Gate-2 probability-of-profit model
----------------------------------------------------------------------------
Goal
- Learn P(win) that the trend/edge continues profitably over the next N bars,
  given current technical state (Gate-2 features) and context.

Inputs
- Gate-2 per-bar scores (from s80):  data_enriched/gate2_confidence_30m.csv
  Required cols: datetime, ticker, confidence_score, RSI14, ADX14,
                 EMA5_slope, EMA20_slope, EMA44_slope, EMA260_slope,
                 Donchian_position, Volatility_ATR, Trend_alignment
- 30m enriched parquets (from s32) for price: data_enriched/30min/*.parquet
  Required cols: datetime, ticker (added by s32), close

Label
- For each (ticker, datetime): y = 1 if forward_return_pct > profit_threshold
  where forward_return_pct = (close[t+H] / close[t] - 1)
  H is --horizon_bars (default 17 ≈ ~1 trading day on 30m)
  profit_threshold default = 0.0 (you can set e.g. 0.001 for ~10 bps costs)

Model
- Logistic Regression (calibrated by design) inside a sklearn Pipeline.
- Time-aware split using TimeSeriesSplit (no leakage).
- Reports AUC, Brier score, and per-feature coefficients.

Outputs
- models/gate2_model.pkl                  (sklearn Pipeline)
- reports/gate2_model_report.json         (metrics, coefficients, config)
- data_enriched/gate2_model_predictions.csv   (optional, last fold oos preds)

Usage (defaults work out of the box):
  /Users/Finance/QuantETFUS_small/venv/bin/python \
    /Users/Finance/QuantETFUS_small/scripts/s82_train_gate2_profit_model.py
"""

from __future__ import annotations
import os, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib

# ---------- Defaults ----------
PROJECT_ROOT = Path("/Users/Finance/QuantETFUS_small")

DEFAULT_CONF_CSV   = PROJECT_ROOT / "data_enriched" / "gate2_confidence_30m.csv"
DEFAULT_PARQ_DIR   = PROJECT_ROOT / "data_enriched" / "30min"
DEFAULT_MODEL_PKL  = PROJECT_ROOT / "models"        / "gate2_model.pkl"
DEFAULT_REPORT_JSON= PROJECT_ROOT / "reports"       / "gate2_model_report.json"
DEFAULT_OOS_PRED_CSV = PROJECT_ROOT / "data_enriched" / "gate2_model_predictions.csv"

FEATURES = [
    "confidence_score",
    "RSI14", "ADX14",
    "EMA5_slope","EMA20_slope","EMA44_slope","EMA260_slope",
    "Donchian_position","Volatility_ATR","Trend_alignment",
]

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Gate-2 P(win) model from historical 30m bars.")
    ap.add_argument("--conf_csv",   default=str(DEFAULT_CONF_CSV), help="Path to gate2_confidence_30m.csv (from s80)")
    ap.add_argument("--parquet_dir",default=str(DEFAULT_PARQ_DIR), help="Folder of s32 30m parquets (for close prices)")
    ap.add_argument("--horizon_bars", type=int, default=17, help="Forward horizon in 30m bars (≈17 ~ 1d)")
    ap.add_argument("--profit_threshold", type=float, default=0.0, help="Label threshold on forward return (e.g., 0.001 for ~10bps)")
    ap.add_argument("--min_history", type=int, default=500, help="Minimum rows per ticker to keep")
    ap.add_argument("--model_out",  default=str(DEFAULT_MODEL_PKL), help="Output .pkl for trained model")
    ap.add_argument("--report_out", default=str(DEFAULT_REPORT_JSON), help="Output .json with metrics and coefficients")
    ap.add_argument("--oos_pred_out", default=str(DEFAULT_OOS_PRED_CSV), help="Optional CSV with last-fold OOS predictions")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap on rows for faster dev (0=all)")
    return ap.parse_args()

# ---------- Helpers ----------
def load_confidence(conf_csv: Path, usecols: list[str]) -> pd.DataFrame:
    if not conf_csv.exists():
        raise SystemExit(f"[ERR] Missing s80 CSV: {conf_csv}")
    header = pd.read_csv(conf_csv, nrows=0)
    cols = [c for c in usecols if c in header.columns]
    if len(cols) < 3:
        raise SystemExit("[ERR] s80 CSV must include at least: datetime,ticker,confidence_score")
    df = pd.read_csv(conf_csv, usecols=cols, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime","ticker","confidence_score"]).sort_values(["ticker","datetime"])
    return df

def load_prices_from_parquets(parq_dir: Path) -> pd.DataFrame:
    if not parq_dir.exists():
        raise SystemExit(f"[ERR] Parquet dir not found: {parq_dir}")
    paths = sorted([p for p in parq_dir.glob("*.parquet")])
    if not paths:
        raise SystemExit(f"[ERR] No parquets in: {parq_dir}")
    frames = []
    for p in paths:
        try:
            g = pd.read_parquet(p, columns=["datetime","close","ticker"])
        except Exception:
            g = pd.read_parquet(p)
        # normalize
        if "ticker" not in g.columns:
            g["ticker"] = p.stem.upper()
        g["datetime"] = pd.to_datetime(g["datetime"], utc=True, errors="coerce")
        g = g.dropna(subset=["datetime","close"]).sort_values("datetime")
        frames.append(g[["datetime","ticker","close"]])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["ticker","datetime"])

def make_labels(df_prices: pd.DataFrame, horizon: int, profit_thr: float) -> pd.DataFrame:
    df = df_prices.copy()
    df["close_fwd"] = df.groupby("ticker", sort=False)["close"].shift(-horizon)
    df["fwd_ret"]   = (df["close_fwd"] / df["close"]) - 1.0
    df["y"]         = (df["fwd_ret"] > profit_thr).astype("Int64")
    return df.drop(columns=["close_fwd"])

def time_series_cv_splits(df_sorted: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray,np.ndarray]]:
    """Time-based splits across the whole panel (no random shuffles)."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    idx = np.arange(len(df_sorted))
    return list(tss.split(idx))

def train_logistic(X: np.ndarray, y: np.ndarray) -> Pipeline:
    # Features are mostly 0..1, but we standardize anyway for stable coefficients.
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("logit", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=1000,
            n_jobs=None,
            class_weight=None,
        ))
    ])
    pipe.fit(X, y)
    return pipe

# ---------- Main ----------
def main():
    args = parse_args()

    conf_csv  = Path(args.conf_csv)
    parq_dir  = Path(args.parquet_dir)
    model_out = Path(args.model_out)
    report_out= Path(args.report_out)
    oos_pred_out = Path(args.oos_pred_out)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    oos_pred_out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load features (s80 output)
    need = ["datetime","ticker","confidence_score"] + [c for c in FEATURES if c != "confidence_score"]
    df_feat = load_confidence(conf_csv, usecols=list(dict.fromkeys(need)))
    if args.max_rows > 0:
        df_feat = df_feat.tail(args.max_rows).copy()

    # 2) Load prices (for labels)
    df_px = load_prices_from_parquets(parq_dir)

    # 3) Merge on (ticker, datetime)
    df = df_feat.merge(df_px, on=["ticker","datetime"], how="inner")

    # 4) Build labels from forward returns
    df = df.sort_values(["ticker","datetime"]).reset_index(drop=True)
    df = make_labels(df, horizon=args.horizon_bars, profit_thr=args.profit_threshold)

    # 5) Drop rows with missing labels and enforce per-ticker history length
    df = df.dropna(subset=["y"]).copy()
    counts = df.groupby("ticker").size()
    keep_tickers = counts[counts >= args.min_history].index
    df = df[df["ticker"].isin(keep_tickers)].reset_index(drop=True)

    if len(df) < 1000 or len(keep_tickers) < 5:
        raise SystemExit(f"[ERR] Not enough training data after filtering. rows={len(df)} tickers={len(keep_tickers)}")

    # 6) Build X, y
    feat_cols = FEATURES
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] Missing features in s80 CSV: {missing}")
    X_all = df[feat_cols].astype(float).to_numpy()
    y_all = df["y"].astype(int).to_numpy()

    # 7) Time-based CV (no leakage)
    splits = time_series_cv_splits(df, n_splits=5)
    aucs, briers = [], []
    last_fold_preds = None
    last_fold_idx = None

    for k, (tr, va) in enumerate(splits, start=1):
        X_tr, y_tr = X_all[tr], y_all[tr]
        X_va, y_va = X_all[va], y_all[va]

        model = train_logistic(X_tr, y_tr)
        p_va = model.predict_proba(X_va)[:,1]

        try:
            auc  = roc_auc_score(y_va, p_va)
        except Exception:
            auc = float("nan")
        try:
            brier = brier_score_loss(y_va, p_va)
        except Exception:
            brier = float("nan")

        aucs.append(auc); briers.append(brier)
        print(f"[CV{k}] AUC={auc:.4f}  Brier={brier:.4f}  n_tr={len(tr)}  n_va={len(va)}")

        # keep last fold for OOS predictions file
        last_fold_preds = p_va
        last_fold_idx   = va

    # 8) Train final model on all data
    final_model = train_logistic(X_all, y_all)
    joblib.dump(final_model, model_out)
    print(f"[OK] Model saved → {model_out}")

    # 9) Coefficients (feature importances)
    #    Extract from pipeline: scaler + logistic
    scaler = final_model.named_steps["scaler"]
    logit  = final_model.named_steps["logit"]
    coefs  = (logit.coef_[0] / (scaler.scale_ + 1e-12)).tolist()  # approximate raw feature impact
    coef_map = {f: float(c) for f, c in zip(feat_cols, coefs)}

    # 10) Report
    report = {
        "config": {
            "horizon_bars": args.horizon_bars,
            "profit_threshold": args.profit_threshold,
            "min_history": args.min_history,
            "features": feat_cols,
            "rows_used": int(len(df)),
            "tickers_used": int(len(keep_tickers))
        },
        "cv": {
            "folds": len(splits),
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std":  float(np.nanstd(aucs)),
            "brier_mean": float(np.nanmean(briers)),
            "brier_std":  float(np.nanstd(briers)),
            "aucs": [None if np.isnan(x) else float(x) for x in aucs],
            "briers": [None if np.isnan(x) else float(x) for x in briers],
        },
        "coefficients": coef_map,
        "paths": {
            "model_pkl": str(model_out),
        }
    }
    report_out.write_text(json.dumps(report, indent=2))
    print(f"[OK] Report → {report_out}")

    # 11) Optional OOS predictions (last fold)
    if last_fold_preds is not None and last_fold_idx is not None:
        oos = df.iloc[last_fold_idx][["datetime","ticker","close","fwd_ret","y"]].copy()
        oos["p_win"] = last_fold_preds
        oos = oos.sort_values(["ticker","datetime"])
        oos.to_csv(oos_pred_out, index=False, float_format="%.6f")
        print(f"[OK] OOS predictions → {oos_pred_out}  rows={len(oos)}")

if __name__ == "__main__":
    main()