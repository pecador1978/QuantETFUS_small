#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s00_data_integrity_enrichment.py (UTC-aware, robust, JSON-driven)

What it does
------------
- Validates raw OHLCV (dupes, missing bars, outliers)
- Builds an expected UTC-aware schedule and aligns WITHOUT losing real data
- Computes daily TA features from config/ta_features.json (with sane defaults)
- Adds daily regime labels
- Exports:
    <ROOT>/data_clean/prices_clean.parquet
    <ROOT>/data_enriched/prices_enriched.parquet
    <ROOT>/reports/integrity_report_*.csv

Inputs (CSV columns, case-insensitive)
--------------------------------------
datetime, ticker, open, high, low, close, volume
"""

import os, sys, argparse, warnings, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- repo bootstrapping ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.paths import P
from common.debug import log_schema, save_debug

# ---- optional settings import (clone-friendly) ----
try:
    from common.settings import MARKET_TZ as _SETTINGS_TZ  # optional
except Exception:
    _SETTINGS_TZ = None

def _default_market_tz() -> str:
    """Resolve a default market timezone: CLI > ENV > settings > heuristic."""
    if os.environ.get("MARKET_TZ"):
        return os.environ["MARKET_TZ"]
    if _SETTINGS_TZ:
        return _SETTINGS_TZ
    proj_name = Path(getattr(P, "ROOT", PROJECT_ROOT)).name
    return "US/Eastern" if "US" in proj_name.upper() else "Europe/London"

# ---------- Config ----------
REQ_COLS  = ["datetime","ticker","open","high","low","close","volume"]
OUTLIER_Z = 8.0
CFG_PATH  = P.CONFIG_DIR / "ta_features.json"

DEFAULT_CFG = {
    "EMA": [5, 20, 44, 100, 200],
    "SMA": [20, 50, 150],
    "RSI": [14, 21],
    "ATR": [14],
    "ADX": [14],
    "Bollinger": [{"length": 20, "std": 2.0}, {"length": 50, "std": 2.5}],
    "MACD": [{"fast":12, "slow":26, "signal":9}],
    "Stochastic": [{"k":14, "d":3, "smooth":3}],
    "regime": {
        "adx_trend_min": 20.0,
        "adx_sideways_max": 15.0,
        "ema_slope_flat_abs": 0.02,
        "ema_slope_len": 44,
        "ADX_len": 14
    }
}

# ---------- TA helpers ----------
def ema(s, n): return s.ewm(span=int(n), adjust=False).mean()
def sma(s, n): return s.rolling(int(n), min_periods=1).mean()

def rsi(close, length=14):
    length = int(length)
    d = close.diff()
    up   = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    up_e   = up.ewm(alpha=1/length, adjust=False).mean()
    down_e = down.ewm(alpha=1/length, adjust=False).mean()
    rs = up_e / down_e.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _true_range(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(df, length=14, h="high", l="low", c="close"):
    length = int(length)
    tr = _true_range(df[h], df[l], df[c])
    return tr.ewm(alpha=1/length, adjust=False).mean()

def adx(df, length=14, h="high", l="low", c="close"):
    length = int(length)
    high, low, close = df[h], df[l], df[c]
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr   = _true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/length,  adjust=False).mean() / atr_)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr_)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/length, adjust=False).mean()

def bollinger(close, length=20, nstd=2.0):
    length = int(length); nstd = float(nstd)
    ma = close.rolling(length).mean()
    sd = close.rolling(length).std()
    return (ma - nstd*sd, ma, ma + nstd*sd)

def macd(close, fast=12, slow=26, signal=9):
    fast, slow, signal = int(fast), int(slow), int(signal)
    mline   = ema(close, fast) - ema(close, slow)
    msignal = mline.ewm(span=signal, adjust=False).mean()
    mhist   = mline - msignal
    return mline, msignal, mhist

def stochastic(df, k=14, d=3, smooth=3):
    k, d, smooth = int(k), int(d), int(smooth)
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k_s  = k_raw.rolling(smooth).mean()
    d_s  = k_s.rolling(d).mean()
    return k_s, d_s

# ---------- Utilities ----------
def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def read_input_csv(path):
    df = pd.read_csv(path)
    df = normalize_cols(df)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime","ticker"]).copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ticker","datetime"]).reset_index(drop=True)

def expected_index_for_freq(g: pd.DataFrame, freq: str, market_tz: str, rth: str | None):
    """
    Return a UTC-aware DatetimeIndex covering the expected schedule for this ticker.
    DAILY: use ACTUAL bars only (no synthetic business days) so counters are in bars.
    INTRADAY: regular range optionally masked to RTH.
    """
    if g.empty:
        return pd.DatetimeIndex([], tz="UTC")

    dts = pd.to_datetime(g["datetime"], utc=True, errors="coerce").dropna()

    if freq.upper() == "D":
        # use only real trading dates present in the data
        idx = pd.DatetimeIndex(sorted(dts.dt.normalize().unique()))
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx

    # intraday fall-through
    d0 = dts.min().floor(freq)
    d1 = dts.max().ceil(freq)
    idx = pd.date_range(d0, d1, freq=freq, tz="UTC")

    if rth:
        start_s, end_s = [s.strip() for s in rth.split("-")]
        z = idx.tz_convert(market_tz)
        mask = (z.time >= pd.to_datetime(start_s).time()) & (z.time <= pd.to_datetime(end_s).time())
        idx = pd.DatetimeIndex(z[mask].tz_convert("UTC"))

    return idx

def detect_duplicates(g): return g.duplicated(subset=["datetime"], keep="first")
def detect_missing(g, exp_idx):
    have = pd.DatetimeIndex(g["datetime"])
    return exp_idx.difference(have)

def detect_outliers(g):
    lr = np.log(g["close"] / g["close"].shift(1))
    z  = (lr - lr.mean()) / (lr.std(ddof=0) if lr.std(ddof=0) else 1.0)
    return (z.abs() > OUTLIER_Z).fillna(False)

def load_ta_config(path=CFG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CFG

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default=str(P.DATA_RAW / "etf_prices_daily_master.csv"))
    ap.add_argument("--freq",    default="D")
    ap.add_argument("--tz",      default=_default_market_tz(), help="Market timezone (e.g., US/Eastern)")
    ap.add_argument("--rth",     default="", help='Optional RTH window like "09:30-16:00" in market TZ')
    ap.add_argument("--outbase", default=str(P.ROOT))
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--debug_full", action="store_true", default=False)
    args = ap.parse_args()

    warnings.simplefilter("ignore", UserWarning)

    # output dirs
    out_clean_dir = Path(args.outbase) / "data_clean"
    out_enr_dir   = Path(args.outbase) / "data_enriched"
    out_rep_dir   = Path(args.outbase) / "reports"
    out_clean_dir.mkdir(parents=True, exist_ok=True)
    out_enr_dir.mkdir(parents=True, exist_ok=True)
    out_rep_dir.mkdir(parents=True, exist_ok=True)

    # config
    cfg        = load_ta_config(CFG_PATH)
    regime_cfg = cfg.get("regime", DEFAULT_CFG["regime"])

    # load master input
    df = read_input_csv(args.input)

    # accumulators
    rep_rows, clean_blocks, enr_blocks = [], [], []

    for ticker, g in tqdm(df.groupby("ticker", sort=False),
                          total=df["ticker"].nunique(),
                          desc="Processing tickers", unit="ticker"):
        g = g.copy()

        # Integrity
        dup_mask   = detect_duplicates(g); dup_count   = int(dup_mask.sum())
        g          = g.loc[~dup_mask].copy()
        outlier_m  = detect_outliers(g);   outlier_count = int(outlier_m.sum())
        g["outlier_flag"] = outlier_m

        # Build expected schedule (UTC-aware) and align safely
        exp_idx    = expected_index_for_freq(g, args.freq, args.tz, args.rth or None)
        missing_ix = detect_missing(g, exp_idx); missing_count = int(len(missing_ix))

        # SAFE alignment: left-join the schedule to real data
        sched  = pd.DataFrame({"datetime": exp_idx})
        g      = g.sort_values("datetime").drop_duplicates(subset=["datetime"])
        merged = sched.merge(g, on="datetime", how="left")
        merged["ticker"] = ticker

        # minimal fills so TA can compute (no look-ahead beyond ffill)
        for c in ["open","high","low","close","volume"]:
            merged[c] = pd.to_numeric(merged.get(c), errors="coerce")

        merged["close"]  = merged["close"].ffill()
        merged["open"]   = merged["open"].fillna(merged["close"])
        merged["high"]   = merged["high"].fillna(merged[["open","close"]].max(axis=1))
        merged["low"]    = merged["low"].fillna(merged[["open","close"]].min(axis=1))
        merged["volume"] = merged["volume"].fillna(0)

        nnz_close = float(merged["close"].notna().mean())
        if nnz_close < 0.1:
            print(f"[WARN] {ticker}: close coverage low after alignment (nnz={nnz_close:.3f})")

        clean_blocks.append(merged.assign(_source="clean"))

        # ---------- TA enrichment ----------
        ge = merged.copy()
        for c in ["open","high","low","close","volume"]:
            ge[c] = pd.to_numeric(ge[c], errors="coerce")

        for n in cfg.get("EMA", []):      ge[f"ema{int(n)}"] = ema(ge["close"], int(n))
        for n in cfg.get("SMA", []):      ge[f"sma{int(n)}"] = sma(ge["close"], int(n))
        for n in cfg.get("RSI", []):      ge[f"rsi{int(n)}"] = rsi(ge["close"], int(n))
        for n in cfg.get("ATR", []):      ge[f"atr{int(n)}"] = atr(ge, int(n))
        for n in cfg.get("ADX", []):      ge[f"adx{int(n)}"] = adx(ge, int(n))
        for bb in cfg.get("Bollinger", []):
            L = int(bb.get("length", 20)); S = float(bb.get("std", 2.0))
            lo, mid, up = bollinger(ge["close"], L, S)
            ge[f"bb_{L}_{S}_lower"] = lo
            ge[f"bb_{L}_{S}_mid"]   = mid
            ge[f"bb_{L}_{S}_upper"] = up
        for m in cfg.get("MACD", []):
            f = int(m.get("fast",12)); s = int(m.get("slow",26)); sig = int(m.get("signal",9))
            mline, msig, mhist = macd(ge["close"], f, s, sig)
            ge[f"macd_{f}_{s}_{sig}"]      = mline
            ge[f"macd_{f}_{s}_{sig}_sig"]  = msig
            ge[f"macd_{f}_{s}_{sig}_hist"] = mhist
        for st in cfg.get("Stochastic", []):
            k = int(st.get("k",14)); d = int(st.get("d",3)); sm = int(st.get("smooth",3))
            k_s, d_s = stochastic(ge, k, d, sm)
            ge[f"stoch_{k}_{d}_{sm}_k"] = k_s
            ge[f"stoch_{k}_{d}_{sm}_d"] = d_s

        # Regime labels
        ema_len = int(regime_cfg.get("ema_slope_len", 44))
        ema_col = f"ema{ema_len}" if f"ema{ema_len}" in ge.columns else None
        if ema_col:
            ge["ema_slope"] = ge[ema_col] - ge[ema_col].shift(1)
        else:
            tmp = ema(ge["close"], ema_len)
            ge["ema_slope"] = tmp - tmp.shift(1)

        adx_len = int(regime_cfg.get("ADX_len", 14))
        adx_col = f"adx{adx_len}" if f"adx{adx_len}" in ge.columns else (next((c for c in ge.columns if c.startswith("adx")), None))
        if adx_col is None:
            ge["adx_tmp"] = 0.0
            adx_col = "adx_tmp"

        ge["is_trending"] = (ge[adx_col] >= float(regime_cfg.get("adx_trend_min", 20.0))) & (ge["ema_slope"] > 0)
        ge["is_sideways"] = (ge[adx_col] <= float(regime_cfg.get("adx_sideways_max", 15.0))) & (ge["ema_slope"].abs() <= float(regime_cfg.get("ema_slope_flat_abs", 0.02)))

        # ---------- Pullback-to-EMA logic ----------
        pb_cfg = cfg.get("Pullback", {"ema": 20, "trend_fast": 20, "trend_mid": 44, "trend_slow": 100,
                                      "band_pct": 0.005, "rsi_len": 14, "rsi_ceiling": 68})
        ema_pb   = int(pb_cfg.get("ema", 20))
        ema_f    = int(pb_cfg.get("trend_fast", 20))
        ema_m    = int(pb_cfg.get("trend_mid", 44))
        ema_s    = int(pb_cfg.get("trend_slow", 100))
        band_pct = float(pb_cfg.get("band_pct", 0.005))
        rsi_len  = int(pb_cfg.get("rsi_len", 14))
        rsi_max  = float(pb_cfg.get("rsi_ceiling", 68))

        for n in {ema_pb, ema_f, ema_m, ema_s}:
            col = f"ema{n}"
            if col not in ge.columns:
                ge[col] = ema(ge["close"], n)
        rsi_col = f"rsi{rsi_len}"
        if rsi_col not in ge.columns:
            ge[rsi_col] = rsi(ge["close"], rsi_len)

        ge["trend_ok"] = (ge[f"ema{ema_f}"] > ge[f"ema{ema_m}"]) & (ge[f"ema{ema_m}"] > ge[f"ema{ema_s}"])
        ge["dist_ema_pb"] = (ge["close"] - ge[f"ema{ema_pb}"]) / ge[f"ema{ema_pb}"]
        ge["touch_ema_pb"] = (
            ((ge["low"] <= ge[f"ema{ema_pb}"]) & (ge["high"] >= ge[f"ema{ema_pb}"])) |
            (ge["dist_ema_pb"].abs() <= band_pct)
        )
        ge["bounce_up"] = ge["close"] >= ge[f"ema{ema_pb}"]
        ge["pullback_entry"] = ge["trend_ok"] & ge["touch_ema_pb"] & ge["bounce_up"] & (ge[rsi_col] < rsi_max)

        # ---------- EMA20 2-day close trend start & stable age ----------
        if "ema20" not in ge.columns:
            ge["ema20"] = ema(ge["close"], 20)

        cond_above = ge["close"] > ge["ema20"]
        cond_below = ge["close"] < ge["ema20"]
        ge["above_ema20"] = cond_above
        ge["below_ema20"] = cond_below

        # Consecutive closes above EMA20 (bars)
        groups = (~cond_above).cumsum()
        runpos = cond_above.groupby(groups).cumcount() + 1
        ge["above_ema20_streak"] = pd.Series(np.where(cond_above, runpos, 0), index=ge.index)

        ge["pre_decline3"] = ge["close"].diff().rolling(3).sum().shift(2) < 0

        ge["trend_long_start"] = (
            (ge["above_ema20_streak"] == 2) &
            (
                (ge["close"].shift(2) <= ge["ema20"].shift(2)) |
                (ge["pre_decline3"])
            )
        )

        ge["trend_long_age"] = np.where(cond_above, np.maximum(ge["above_ema20_streak"] - 1, 0), 0)
        ge["above_ema20_streak_ge2"] = ge["above_ema20_streak"] >= 2

        # --- Vermeulen-simple counters (Series-only ops; nullable Int64) ---
        s = ge["above_ema20_streak"]
        ge["trend_days_active"] = s.astype("Int64")
        ge["trend_days_since_start"] = (s.where(s >= 2, 1) - 1).astype("Int64")

        ge["trend_signal_label"] = np.where(
            ge["trend_long_start"], "NEW",
            np.where(s >= 2, ge["trend_days_since_start"].astype(str), "0")
        )

        ge["trend_start_today"]  = ge["trend_long_start"].astype(bool)
        ge["trend_active_above"] = ge["above_ema20"].astype(bool)
        ge["trend_active_ge2"]   = ge["above_ema20_streak_ge2"].astype(bool)
        ge["trend_cancelled"]    = ge["below_ema20"].astype(bool)
        ge["pullback_entry"]     = ge["pullback_entry"].astype(bool)

        # ---------- Suffix *_d for daily TA/labels (keep base OHLCV) ----------
        _BASE_KEEP = {"datetime","ticker","open","high","low","close","volume","outlier_flag","_source"}
        cols_to_suffix = [c for c in ge.columns if c not in _BASE_KEEP and not c.endswith("_d")]
        ge = ge.rename(columns={c: f"{c}_d" for c in cols_to_suffix})

        enr_blocks.append(ge.assign(_source="enriched"))

        rep_rows.append({
            "ticker": ticker,
            "dup_removed": dup_count,
            "missing_expected": missing_count,
            "outliers_flagged": outlier_count,
            "first_dt": ge["datetime"].min(),
            "last_dt":  ge["datetime"].max(),
            "freq": args.freq,
            "tz": args.tz,
            "rth": args.rth or ""
        })

    # ---------- Write outputs ----------
    clean_all = pd.concat(clean_blocks, ignore_index=True) if clean_blocks else pd.DataFrame(columns=REQ_COLS)
    enr_all   = pd.concat(enr_blocks,   ignore_index=True) if enr_blocks   else pd.DataFrame()

    clean_path = P.ROOT / "data_clean"    / "prices_clean.parquet"
    enr_path   = P.ROOT / "data_enriched" / "prices_enriched.parquet"
    rep_path   = P.ROOT / "reports"       / f"integrity_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv"

    if not clean_all.empty: clean_all.to_parquet(clean_path, index=False)
    if not enr_all.empty:   enr_all.to_parquet(enr_path,   index=False)
    pd.DataFrame(rep_rows).to_csv(rep_path, index=False)

    print(f"[OK] Clean     → {clean_path}  | rows={len(clean_all):,}")
    print(f"[OK] Enriched  → {enr_path}    | rows={len(enr_all):,}")
    print(f"[OK] Report    → {rep_path}")

    log_schema(enr_all, note="enriched_final")
    if args.debug or args.debug_full:
        save_debug(enr_all, Path(P.ROOT / "data_enriched"), "prices_enriched")

if __name__ == "__main__":
    main()