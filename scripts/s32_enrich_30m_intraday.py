#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s32_enrich_30m_intraday.py — Build per-ticker 30-minute context with daily + macro overlays (project-aware)
"""

from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- project-aware imports ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P

# ---------------- input dir resolution ----------------
def resolve_in_dir(bucket_arg: str | None) -> Path:
    """
    Resolve the input folder for 30m raw CSVs.
    Priority:
      0) P.SHARED_30M_DIR
      1) $DATA_RAW_SHARED/30min or $SHARED_RAW_BASE/30min
      2) P.DATA_RAW/30min
      3) legacy QuantShared/us/eu buckets
      4) bucket under P.DATA_RAW/<bucket>/30min
    """
    shared_30m = getattr(P, "SHARED_30M_DIR", None)
    if shared_30m:
        p = Path(shared_30m)
        if p.exists():
            return p

    for envkey in ("DATA_RAW_SHARED", "SHARED_RAW_BASE"):
        v = os.environ.get(envkey, "").strip()
        if v:
            cand = Path(v).expanduser().resolve() / "30min"
            if cand.exists():
                return cand

    local_flat = Path(P.DATA_RAW) / "30min"
    if local_flat.exists():
        return local_flat

    qs = getattr(P, "QUANTSHARED", P.ROOT.parent / "QuantShared")
    cand_us = qs / "data_raw_ETF_US" / "30min"
    cand_eu = qs / "data_raw_ETF"    / "30min"
    if cand_us.exists():
        return cand_us
    if cand_eu.exists():
        return cand_eu

    bucket = (
        (bucket_arg or "").strip()
        or os.environ.get("TARGET_BUCKET", "").strip()
        or next((b for b in ("targeted_ETFs_US", "targeted_ETFs")
                 if (Path(P.DATA_RAW) / b / "30min").exists()), "")
    )
    if bucket:
        cand = Path(P.DATA_RAW) / bucket / "30min"
        if cand.exists():
            return cand

    raise SystemExit(
        "[ERR] Could not find 30min input folder.\n"
        f"  Tried:\n"
        f"    P.SHARED_30M_DIR → {getattr(P, 'SHARED_30M_DIR', None)}\n"
        f"    $DATA_RAW_SHARED/30min or $SHARED_RAW_BASE/30min\n"
        f"    {local_flat}\n"
        f"    {cand_us}\n"
        f"    {cand_eu}\n"
        f"    bucket under {P.DATA_RAW}\n"
    )

# ---------- helpers ----------
def ema(s, n):  return s.ewm(span=int(n), adjust=False).mean()

def _tr(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Wilder ATR (SMA-seeded RMA) — identical to LSE project logic.
    TR = max(high-low, |high-prevClose|, |low-prevClose|)
    ATR₀ = SMA(TR[0:n]); ATRₜ = (ATRₜ₋₁·(n-1) + TRₜ)/n
    """
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)

    seed = tr.rolling(n, min_periods=n).mean()
    out = pd.Series(index=tr.index, dtype=float)
    si = seed.first_valid_index()
    if si is None:
        return out.rename(f"atr{n}")
    out.loc[si] = seed.loc[si]
    alpha = 1.0 / n
    for i in range(tr.index.get_loc(si) + 1, len(tr)):
        out.iloc[i] = out.iloc[i - 1] * (1 - alpha) + tr.iloc[i] * alpha
    return out.rename(f"atr{n}")

def _to_utc_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt  # guaranteed UTC

def slope_pct(series: pd.Series, lookback: int = 5) -> pd.Series:
    lb = int(lookback)
    base = series.shift(lb).replace(0, np.nan)
    return (series - series.shift(lb)) / base

def donch_pos(close: pd.Series, high: pd.Series, low: pd.Series, length: int = 20) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    rng = (hh - ll).replace(0, np.nan)
    return ((close - ll) / rng).clip(0, 1)

# ---------- IO ----------
def read_30m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    if "datetime" not in df.columns:
        if "time" in df.columns:  df = df.rename(columns={"time":"datetime"})
        elif "date" in df.columns: df = df.rename(columns={"date":"datetime"})
        else:
            raise SystemExit(f"[ERR] {path.name}: no datetime/time/date column found.")
    df["datetime"] = _to_utc_series(df["datetime"])
    for c in ("open","high","low","close"):
        if c not in df.columns:
            raise SystemExit(f"[ERR] {path.name}: missing '{c}' column.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.dropna(subset=["datetime","open","high","low","close"])
              .sort_values("datetime")
              .reset_index(drop=True))

def enrich_30m_only(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    out = df.copy()

    # base length: env override > default 340 (historically EMA340)
    base_len = int(os.environ.get("M30_EMA_BARS", "340"))

    # compatibility fields used by downstream
    out["m30_ema20d"] = ema(out["close"], base_len)
    out["m30_dist_to_ema20d_pct"] = out["close"].div(out["m30_ema20d"]).sub(1.0)
    if "m30_ema260" not in out.columns:
        out["m30_ema260"] = out["m30_ema20d"]
    if "m30_dist_to_ema260_pct" not in out.columns:
        out["m30_dist_to_ema260_pct"] = out["m30_dist_to_ema20d_pct"]

    out["m30_atr14"] = atr(out, 14)
    out["m30_atr14_norm"] = out["m30_atr14"] / out["close"]

    for w in (80, 160):
        hh = out["high"].rolling(w).max()
        ll = out["low"].rolling(w).min()
        out[f"m30_hh{w}_from_close_pct"] = hh.div(out["close"]).sub(1.0)
        out[f"m30_ll{w}_from_close_pct"] = out["close"].div(ll).sub(1.0).mul(-1.0)

    # multi-EMA block used by rules
    out["ema5"]   = ema(out["close"], 5)
    out["ema20"]  = ema(out["close"], 20)
    out["ema44"]  = ema(out["close"], 44)
    out["ema260"] = ema(out["close"], base_len)

    out["EMA5_slope"]   = slope_pct(out["ema5"],   5)
    out["EMA20_slope"]  = slope_pct(out["ema20"],  5)
    out["EMA44_slope"]  = slope_pct(out["ema44"],  5)
    out["EMA260_slope"] = slope_pct(out["ema260"], 5)

    out["Donchian_position"] = donch_pos(out["close"], out["high"], out["low"], 20)

    up   = (out["ema5"] > out["ema20"]) & (out["ema20"] > out["ema44"]) & (out["ema44"] > out["ema260"])
    down = (out["ema5"] < out["ema20"]) & (out["ema20"] < out["ema44"]) & (out["ema44"] < out["ema260"])
    out["Trend_alignment"] = (up | down).astype(float)

    out["Volatility_ATR"] = out.get("m30_atr14_norm", pd.Series(index=out.index, dtype=float))

    local = out["datetime"].dt.tz_convert(market_tz)
    out["m30_entry_hour"] = local.dt.hour.astype("Int64")
    out["m30_weekday"]    = local.dt.weekday.astype("Int64")
    out["date_local"]     = local.dt.normalize()  # midnight local
    return out

def _infer_etf_daily_cols(parquet_path: Path, whitelist: list[str] | None = None) -> list[str]:
    try:
        meta = pd.read_parquet(parquet_path, columns=None)
        cols = [str(c).strip().lower() for c in meta.columns]
        if whitelist:
            wl = [c.strip().lower() for c in whitelist]
            return [c for c in wl if c in cols]
        return [c for c in cols if c.endswith("_d")]
    except Exception:
        return [c.strip().lower() for c in (whitelist or [])]

def load_etf_daily(enriched_parquet: Path, ticker: str, keep_cols: list[str], market_tz: str) -> pd.DataFrame:
    d = pd.read_parquet(enriched_parquet)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d[d["ticker"].astype(str) == str(ticker)].copy()
    if "datetime" not in d.columns:
        raise SystemExit(f"[ERR] {enriched_parquet.name}: missing 'datetime'.")
    d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
    d["date_local"] = d["datetime"].dt.tz_convert(market_tz).dt.normalize()
    keep = ["date_local"] + [c for c in keep_cols if c in d.columns]
    return d[keep].drop_duplicates(subset=["date_local"])

def load_macro_context(macro_parquet: Path, market_tz: str) -> pd.DataFrame:
    if not macro_parquet.exists():
        return pd.DataFrame(columns=["date_local"])
    m = pd.read_parquet(macro_parquet)
    m.columns = [c.strip().lower() for c in m.columns]

    # time column discovery
    dtcand = next((c for c in ["datetime","date","dt","day","timestamp","time"] if c in m.columns), None)
    if not dtcand:
        return pd.DataFrame(columns=["date_local"])

    m["datetime"] = pd.to_datetime(m[dtcand], utc=True, errors="coerce")
    feat_cols = [c for c in m.columns if c not in ("datetime","date","dt","day","timestamp","time","ticker")]

    if "ticker" not in m.columns:
        out = m.copy()
        out["date_local"] = out["datetime"].dt.tz_convert(market_tz).dt.normalize()
        cols = [c for c in out.columns if c not in ("datetime", dtcand)]
        return (out[["date_local"] + [c for c in cols if c != "date_local"]]
                .dropna(subset=["date_local"])
                .drop_duplicates(subset=["date_local"])
                .sort_values("date_local"))

    frames = []
    for f in feat_cols:
        piv = m.pivot_table(index="datetime", columns="ticker", values=f, aggfunc="last")
        piv = piv.rename(columns=lambda t: f"{t}_{f}")
        frames.append(piv)

    out = pd.concat(frames, axis=1).reset_index() if frames else pd.DataFrame(columns=["datetime"])
    out["date_local"] = out["datetime"].dt.tz_convert(market_tz).dt.normalize()
    cols = [c for c in out.columns if c != "datetime"]
    return (out[cols]
            .dropna(subset=["date_local"])
            .drop_duplicates(subset=["date_local"])
            .sort_values("date_local"))

# -------- smart tz default --------
def _default_market_tz() -> str:
    env_tz = os.environ.get("MARKET_TZ", "").strip()
    if env_tz:
        return env_tz
    proj = Path(getattr(P, "ROOT", PROJECT_ROOT)).name.upper()
    sheet = os.environ.get("ETF_SHEET", "").upper()
    if any(k in proj for k in ("NY", "US")) or any(k in sheet for k in ("NY", "USD", "US")):
        return "America/New_York"
    return "Europe/London"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("TARGET_BUCKET",""),
                    help="Optional sub-bucket under data_raw (otherwise auto-resolve).")
    ap.add_argument("--out_dir", default=str(Path(P.DATA_ENRICHED) / "30min"))
    ap.add_argument("--daily_context", default=str(Path(P.DATA_ENRICHED) / "prices_enriched.parquet"))
    ap.add_argument("--macro_context", default=str(Path(P.DATA_ENRICHED) / "macro_forex_enriched.parquet"))
    ap.add_argument("--tz", default="", help="Override market tz (e.g. America/New_York).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = resolve_in_dir(args.bucket)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    mkt_tz = args.tz.strip() or _default_market_tz()

    files = sorted(list(src.glob("*_30min*.csv")))
    if not files:
        raise SystemExit(f"[ERR] No 30m inputs in {src}")

    desired_etf_d = [
        "ema5_d","ema20_d","ema44_d","sma150_d",
        "rsi14_d","atr14_d","atr14_norm_d","adx14_d"
    ]
    # Allow any *_d present; prefer desired list if available
    all_d_cols = _infer_etf_daily_cols(Path(args.daily_context), whitelist=None)
    etf_keep = sorted(set([c.strip().lower() for c in desired_etf_d]).union(all_d_cols))

    macro_ctx = load_macro_context(Path(args.macro_context), mkt_tz)

    catalog = []
    for f in files:
        import re
        t = f.name.replace("_30min_raw.csv","").replace("_30min.csv","")
        t = re.sub(r"\s+\d+$", "", t)
        try:
            df = read_30m(f)
            if df.empty:
                print(f"[WARN] {t}: no valid rows after parsing; skipping.")
                continue

            e  = enrich_30m_only(df, mkt_tz)

            # ---------- merge ETF daily overlays ----------
            try:
                if etf_keep:
                    dctx = load_etf_daily(Path(args.daily_context), t, etf_keep, mkt_tz)
                    if not dctx.empty:
                        # ensure datetime64[ns] for merge_asof
                        e["date_local"]    = pd.to_datetime(e["date_local"])
                        dctx["date_local"] = pd.to_datetime(dctx["date_local"])
                        e = pd.merge_asof(
                                e.sort_values("date_local"),
                                dctx.sort_values("date_local"),
                                on="date_local",
                                direction="backward"
                            ).sort_index()
                    else:
                        print(f"[WARN] {t}: no daily rows in {args.daily_context}; daily overlays will be NaN.")
            except Exception as ex:
                print(f"[WARN] {t}: ETF daily merge skipped → {ex}")

            # ---------- merge Macro/Forex context ----------
            try:
                if not macro_ctx.empty:
                    mctx = macro_ctx.copy()
                    mctx["date_local"] = pd.to_datetime(mctx["date_local"])
                    e = e.merge(mctx, on="date_local", how="left")
            except Exception as ex:
                print(f"[WARN] {t}: Macro daily merge skipped → {ex}")

            # compatibility aliases used by downstream
            if "RSI14" not in e.columns and "rsi14_d" in e.columns:
                e["RSI14"] = e["rsi14_d"]
            if "ADX14" not in e.columns and "adx14_d" in e.columns:
                e["ADX14"] = e["adx14_d"]
            if "Volatility_ATR" not in e.columns and "m30_atr14_norm" in e.columns:
                e["Volatility_ATR"] = e["m30_atr14_norm"]

            e = e.drop(columns=["date_local"])
            e["ticker"] = t
            p = outdir / f"{t}.parquet"
            e.to_parquet(p, index=False)
            catalog.append({"ticker":t, "rows":int(len(e)), "path":str(p)})
            print(f"[OK] {t} → {p} ({len(e)} rows)")
        except Exception as ex:
            print(f"[WARN] {t}: {ex}")

    (outdir / "_catalog.json").write_text(json.dumps(catalog, indent=2))
    print(f"[OK] Catalog → {outdir}/_catalog.json  tickers={len(catalog)}")

    if args.debug and catalog:
        sample_t = catalog[0]["ticker"]
        s = pd.read_parquet(outdir / f"{sample_t}.parquet").tail(300)
        dbg = Path(P.REPORTS_DIR) / "debug_enriched"; dbg.mkdir(parents=True, exist_ok=True)
        sp = dbg / f"s32_{sample_t}_sample.parquet"
        s.to_parquet(sp, index=False)
        print(f"[DEBUG] Saved sample → {sp}")

if __name__ == "__main__":
    main()