#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s31_enrich_macro_forex_daily.py
Build a LONG daily macro/FX dataset and add TA features (*_d).
Also merge SPY Vermeulen color labels (one global column per date).

Inputs:
- P.FOREX_DIR:         .../QuantShared/forex          -> *_daily.csv
- P.MARKET_TRENDS_DIR: .../QuantShared/market_trends  -> *_TV_daily.csv (TradingView schema)
- Optional Vermeulen Excel: P.MARKET_TRENDS_DIR / "Macro_all_label_daily_weekly_SPY_vermeulen.xlsx"
  sheet: 'daily_labels' with columns: ['date', 'spy_label' or 'label' or 'color']

Outputs:
- P.DATA_ENRICHED/macro_forex_enriched.parquet   (LONG)
- P.DATA_ENRICHED/macro_forex_enriched.csv       (LONG)
"""

from pathlib import Path
from datetime import datetime, timezone
import sys, os, re, argparse, json
import pandas as pd
import numpy as np

# --- repo bootstrapping ---
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P

# optional debug schema logger
try:
    from common.debug import log_schema  # type: ignore
except Exception:
    def log_schema(df: pd.DataFrame, note: str = ""):
        print(f"[INFO] {note}: shape={df.shape}, cols={len(df.columns)}")

# ---------- TA helpers ----------
def ema(s, n):  return s.ewm(span=int(n), adjust=False).mean()
def sma(s, n):  return s.rolling(int(n), min_periods=1).mean()

def rsi(close, length=14):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    up_e = up.ewm(alpha=1/length, adjust=False).mean()
    dn_e = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = up_e / dn_e.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _true_range(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(df, length=14):
    if {"high","low","close"}.issubset(df.columns):
        tr = _true_range(df["high"], df["low"], df["close"])
    else:
        tr = df["close"].diff().abs()
    return tr.rolling(int(length)).mean()

def adx(df, length=14):
    if not {"high","low","close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff()
    minus_dm = -low.diff()
    plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr   = _true_range(high, low, close)
    atr_ = tr.rolling(int(length)).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/length).mean() / atr_)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length).mean() / atr_)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1/length).mean()

def bollinger(close, length=20, nstd=2.0):
    ma = close.rolling(int(length)).mean()
    sd = close.rolling(int(length)).std()
    return (ma - nstd*sd, ma, ma + nstd*sd)

def macd(close, fast=12, slow=26, signal=9):
    mline   = ema(close, fast) - ema(close, slow)
    msignal = mline.ewm(span=signal, adjust=False).mean()
    mhist   = mline - msignal
    return mline, msignal, mhist

def stochastic(df, k=14, d=3, smooth=3):
    if not {"high","low","close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    low_min  = df["low"].rolling(int(k)).min()
    high_max = df["high"].rolling(int(k)).max()
    k_raw = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k_s  = k_raw.rolling(int(smooth)).mean()
    d_s  = k_s.rolling(int(d)).mean()
    return k_s, d_s

# ---------- utilities ----------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _derive_ticker_from_name(path: Path) -> str:
    """
    Derive a compact symbol from file name.
    Accepts forms: <T>_daily.csv, <T>_TV_daily.csv, <T>_something_daily.csv
    """
    name = path.stem
    name = re.sub(r"_?tv", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_?daily$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"_?tv_?daily$", "", name, flags=re.IGNORECASE)
    return name.upper()

def _pick_datetime_col(df: pd.DataFrame) -> str:
    for c in ("datetime","date","time","timestamp"):
        if c in df.columns:
            return c
    return df.columns[0]

def _pick_close_col(df: pd.DataFrame, dt_col: str) -> str | None:
    for c in ("close","adj_close","price","value","rate"):
        if c in df.columns:
            return c
    # fallback: first numeric column that isn't the datetime
    for c in df.columns:
        if c != dt_col and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def _read_daily_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_cols(df)
    dt_col = _pick_datetime_col(df)
    df["datetime"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["date"] = df["datetime"].dt.floor("D")

    close_col = _pick_close_col(df, dt_col)
    if close_col is None:
        return pd.DataFrame(columns=["date","close"])

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    # add high/low if present (for ADX/stoch/ATR true-range)
    for c in ("high","low"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["date","close"] + [c for c in ("high","low") if c in df.columns]
    out = (df[keep]
           .sort_values("date")
           .drop_duplicates(subset=["date"]))
    return out

def _load_ta_config(path: Path, defaults: dict) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return defaults

# ---------- Vermeulen labels ----------
def _load_vermeulen_labels(xlsx: Path) -> pd.DataFrame | None:
    if not xlsx.exists():
        print(f"[INFO] Vermeulen Excel not found: {xlsx}")
        return None
    try:
        lab = pd.read_excel(xlsx, sheet_name="daily_labels")
    except Exception as e:
        print(f"[WARN] Cannot read Vermeulen sheet: {e}")
        return None
    lab = normalize_cols(lab)
    if "date" not in lab.columns:
        return None
    label_col = next((c for c in ("spy_label","label","color") if c in lab.columns), None)
    if not label_col:
        return None
    out = lab[["date", label_col]].copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.floor("D")
    out = out.rename(columns={label_col: "SPY_vermeulen_color"})
    out["SPY_vermeulen_color"] = out["SPY_vermeulen_color"].astype(str).str.strip().str.lower()
    return out.dropna(subset=["date"])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forex_dir",     type=str, default=str(P.FOREX_DIR),
                    help="Directory with FX *_daily.csv (default: P.FOREX_DIR)")
    ap.add_argument("--macro_dir",     type=str, default=str(P.MARKET_TRENDS_DIR),
                    help="Directory with TV macro *_TV_daily.csv (default: P.MARKET_TRENDS_DIR)")
    ap.add_argument("--vermeulen_xlsx",type=str, default=str(P.MARKET_TRENDS_DIR / "Macro_all_label_daily_weekly_SPY_vermeulen.xlsx"))
    ap.add_argument("--cfg",           type=str, default=str(P.CONFIG_DIR / "ta_features.json"),
                    help="Optional TA config JSON")
    ap.add_argument("--out_parquet",   type=str, default=str(P.DATA_ENRICHED / "macro_forex_enriched.parquet"))
    ap.add_argument("--out_csv",       type=str, default=str(P.DATA_ENRICHED / "macro_forex_enriched.csv"))
    args = ap.parse_args()

    defaults = {
        "EMA":[5,20,44,100,200], "SMA":[20,50,200],
        "RSI":[14,21], "ATR":[14], "ADX":[14],
        "Bollinger":[{"length":20,"std":2.0},{"length":50,"std":2.5}],
        "MACD":[{"fast":12,"slow":26,"signal":9}],
        "Stochastic":[{"k":14,"d":3,"smooth":3}]
    }
    cfg = _load_ta_config(Path(args.cfg), defaults)

    # gather files
    forex_dir = Path(args.forex_dir)
    macro_dir = Path(args.macro_dir)
    files: list[Path] = []
    if forex_dir.exists():
        files += list(forex_dir.glob("*_daily.csv"))
    if macro_dir.exists():
        files += list(macro_dir.glob("*_TV_daily.csv"))
    if not files:
        raise SystemExit("[ERR] No source files found in forex/macro folders.")

    rows = []
    for f in sorted(files):
        try:
            tkr = _derive_ticker_from_name(f)
            df  = _read_daily_file(f)
            if df.empty:
                print(f"[WARN] {f.name}: no usable rows; skip.")
                continue

            g = df.copy()
            # TA enrichment (all with *_d suffix)
            for n in cfg.get("EMA", []): g[f"ema{n}_d"] = ema(g["close"], n)
            for n in cfg.get("SMA", []): g[f"sma{n}_d"] = sma(g["close"], n)
            for n in cfg.get("RSI", []): g[f"rsi{n}_d"] = rsi(g["close"], n)
            for n in cfg.get("ATR", []): g[f"atr{n}_d"] = atr(g, n)
            for n in cfg.get("ADX", []): g[f"adx{n}_d"] = adx(g, n)

            for bb in cfg.get("Bollinger", []):
                length = int(bb.get("length", 20))
                stdv   = float(bb.get("std", 2.0))
                lo, mid, up = bollinger(g["close"], length, stdv)
                g[f"bb_{length}_{stdv}_lower_d"] = lo
                g[f"bb_{length}_{stdv}_mid_d"]   = mid
                g[f"bb_{length}_{stdv}_upper_d"] = up

            for m in cfg.get("MACD", []):
                fast, slow, signal = int(m["fast"]), int(m["slow"]), int(m["signal"])
                ml, ms, mh = macd(g["close"], fast, slow, signal)
                base = f"macd_{fast}_{slow}_{signal}"
                g[f"{base}_d"]      = ml
                g[f"{base}_sig_d"]  = ms
                g[f"{base}_hist_d"] = mh

            for st in cfg.get("Stochastic", []):
                k, d, smooth = int(st["k"]), int(st["d"]), int(st["smooth"])
                k_s, d_s = stochastic(g, k, d, smooth)
                g[f"stoch_{k}_{d}_{smooth}_k_d"] = k_s
                g[f"stoch_{k}_{d}_{smooth}_d_d"] = d_s

            g["ticker"] = tkr
            rows.append(g)
            print(f"[OK] {tkr}: rows={len(g)} from {f.name}")
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")

    if not rows:
        raise SystemExit("[ERR] Nothing to write after processing sources.")

    long_df = (pd.concat(rows, ignore_index=True)
                 .sort_values(["ticker","date"])
                 .reset_index(drop=True))

    # merge Vermeulen (optional)
    labels = _load_vermeulen_labels(Path(args.vermeulen_xlsx))
    if labels is not None and not labels.empty:
        long_df = long_df.merge(labels, on="date", how="left")
        try:
            cov = long_df["SPY_vermeulen_color"].notna().mean()
            print(f"[INFO] Vermeulen labels merged; coverage={cov:.1%}")
        except Exception:
            pass

    # write outputs
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(args.out_parquet, index=False)
    long_df.to_csv(args.out_csv, index=False)

    log_schema(long_df, note="macro_forex_enriched")
    print(f"[OK] Written {len(long_df):,} rows → {args.out_parquet}")

if __name__ == "__main__":
    main()