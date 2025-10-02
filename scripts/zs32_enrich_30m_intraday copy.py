#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s32_enrich_30m_intraday.py — Build per-ticker 30-minute context with daily + macro overlays (project-aware)

Purpose
- Create cached intraday features used near entry time (e.g., EMA275, distance_to_EMA275, intraday ATR, HH/LL windows).
- Attach SAME-DAY DAILY context (from s30/s31) to each 30m bar (repeated across bars of that day by design).
  * ETF daily (prices_enriched.parquet, *_d cols)
  * Macro/FX (macro_forex_enriched.parquet) — keep ALL features exactly like s67b.

Reads (auto-detected, in this priority)
- P.DATA_RAW/30min
- <QuantShared>/data_raw_ETF_US/30min
- <QuantShared>/data_raw_ETF/30min
- P.DATA_RAW/<TARGET_BUCKET>/30min
- Daily equity context:   P.DATA_ENRICHED/prices_enriched.parquet
- Daily macro/FX context: P.DATA_ENRICHED/macro_forex_enriched.parquet

Writes
- Per-ticker parquet:     P.DATA_ENRICHED/30min/{TICKER}.parquet
- Catalog:                P.DATA_ENRICHED/30min/_catalog.json

Conventions
- Intraday columns: 'm30_*' (e.g., m30_close, m30_ema275, m30_dist_to_ema275_pct, m30_hh80, m30_ll80, m30_entry_hour).
- Daily ETF overlays retain '_d' suffix (lowercase).
- Macro overlays follow s67b naming: '{TICKER}_{feature}' (may or may not end with '_d').
- Session masking & day boundaries use MARKET_TZ (default Europe/London).

Leakage Discipline
- Merge DAILY context by the SAME local day (values available at that day’s open).
"""

import argparse, os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# project-aware
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
      1) Shared flat layout: P.DATA_RAW/30min
      2) QuantShared US flat: <QuantShared>/data_raw_ETF_US/30min
      3) QuantShared EU flat: <QuantShared>/data_raw_ETF/30min
      4) Bucketed fallback:   P.DATA_RAW/<bucket>/30min
         where <bucket> = --bucket | $TARGET_BUCKET | detected defaults
    """
    # 1) project-local flat
    shared = P.DATA_RAW / "30min"
    if shared.exists():
        return shared

    # 2) & 3) QuantShared flat layouts
    # Try explicit P.QUANTSHARED if provided, else derive from project root
    qs = getattr(P, "QUANTSHARED", None)
    if not qs:
        qs = P.ROOT.parent / "QuantShared"
    cand_us = qs / "data_raw_ETF_US" / "30min"
    cand_eu = qs / "data_raw_ETF" / "30min"
    if cand_us.exists():
        return cand_us
    if cand_eu.exists():
        return cand_eu

    # 4) bucketed fallback under project data_raw
    bucket = (
        (bucket_arg or "").strip()
        or os.environ.get("TARGET_BUCKET", "").strip()
        or next(
            (b for b in ("targeted_ETFs_US", "targeted_ETFs") if (P.DATA_RAW / b / "30min").exists()),
            ""
        )
    )
    if bucket:
        return P.DATA_RAW / bucket / "30min"

    raise SystemExit(
        "[ERR] Could not find 30min input folder.\n"
        f"  Tried: {shared}\n"
        f"  And QuantShared flat / bucketed fallbacks under {P.DATA_RAW} and {qs}"
    )

# ---------- helpers ----------
def ema(s, n):  return s.ewm(span=int(n), adjust=False).mean()

def _tr(h,l,c):
    pc = c.shift(1)
    return pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)

def atr(df, n=14): return _tr(df["high"], df["low"], df["close"]).ewm(alpha=1/n, adjust=False).mean()

def _to_utc_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        return dt.dt.tz_localize("UTC")
    return dt.dt.tz_convert("UTC")

# ---------- IO ----------
def read_30m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

    if "datetime" not in df.columns:
        if "time" in df.columns: df = df.rename(columns={"time":"datetime"})
        elif "date" in df.columns: df = df.rename(columns={"date":"datetime"})
        else:
            raise SystemExit(f"[ERR] {path.name}: no datetime/time/date column found.")

    df["datetime"] = _to_utc_series(df["datetime"])
    for c in ("open","high","low","close"):
        if c not in df.columns:
            raise SystemExit(f"[ERR] {path.name}: missing '{c}' column.")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df.dropna(subset=["datetime","open","high","low","close"])
            .sort_values("datetime")
            .reset_index(drop=True))
    return df

def enrich_30m_only(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    out = df.copy()
    out["m30_ema275"] = ema(out["close"], 275)
    out["m30_dist_to_ema275_pct"] = out["close"].div(out["m30_ema275"]).sub(1.0)
    out["m30_atr14"] = atr(out, 14)
    out["m30_atr14_norm"] = out["m30_atr14"] / out["close"]

    for w in (80,160):
        hh = out["high"].rolling(w).max()
        ll = out["low"].rolling(w).min()
        out[f"m30_hh{w}_from_close_pct"] = hh.div(out["close"]).sub(1.0)
        out[f"m30_ll{w}_from_close_pct"] = out["close"].div(ll).sub(1.0).mul(-1.0)

    local = out["datetime"].dt.tz_convert(market_tz)
    out["m30_entry_hour"] = local.dt.hour.astype("Int64")
    out["m30_weekday"]    = local.dt.weekday.astype("Int64")
    out["date_local"]     = local.dt.date  # merge key for daily
    return out

def _infer_etf_daily_cols(parquet_path: Path, whitelist: list[str] | None = None) -> list[str]:
    """
    Return ETF daily columns (lowercased). Prefer whitelisted subset; fallback to all *_d.
    """
    try:
        meta = pd.read_parquet(parquet_path)
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
    dt_utc = _to_utc_series(d["datetime"])
    d["date_local"] = dt_utc.dt.tz_convert(market_tz).dt.date
    keep = ["date_local"] + [c for c in keep_cols if c in d.columns]
    return d[keep].drop_duplicates(subset=["date_local"])

def load_macro_context(macro_parquet: Path, market_tz: str) -> pd.DataFrame:
    """
    Accepts either:
      - wide with a date-like column and NO 'ticker' -> keep all columns
      - long with ['date'|'datetime'...,'ticker', feature_cols] -> pivoted wide per feature
    Returns ['date_local', <ALL macro columns>], matching s67b naming '{TICKER}_{feature}'.
    """
    if not macro_parquet.exists():
        return pd.DataFrame(columns=["date_local"])

    m = pd.read_parquet(macro_parquet)
    m.columns = [c.strip().lower() for c in m.columns]

    # already wide (no 'ticker')?
    if "ticker" not in m.columns:
        dtcand = next((c for c in ["date","datetime","dt","day","timestamp","time"] if c in m.columns), None)
        if not dtcand:
            return pd.DataFrame(columns=["date_local"])
        out = m.copy()
        out["datetime"] = pd.to_datetime(out[dtcand], utc=True, errors="coerce")
        out["date_local"] = out["datetime"].dt.tz_convert(market_tz).dt.date
        cols = [c for c in out.columns if c not in ("datetime", dtcand)]
        return (out[["date_local"] + [c for c in cols if c != "date_local"]]
                .dropna(subset=["date_local"])
                .drop_duplicates(subset=["date_local"])
                .sort_values("date_local"))

    # long -> wide
    dtcand = next((c for c in ["date","datetime","dt","day","timestamp","time"] if c in m.columns), None)
    if not dtcand:
        return pd.DataFrame(columns=["date_local"])
    m["datetime"] = pd.to_datetime(m[dtcand], utc=True, errors="coerce")
    feat_cols = [c for c in m.columns if c not in ("datetime","date","dt","day","timestamp","time","ticker")]

    frames = []
    for f in feat_cols:
        piv = m.pivot_table(index="datetime", columns="ticker", values=f, aggfunc="last")
        piv = piv.rename(columns=lambda t: f"{t}_{f}")  # <-- s67b naming
        frames.append(piv)

    if frames:
        out = pd.concat(frames, axis=1).reset_index()
    else:
        out = pd.DataFrame(columns=["datetime"])

    out["date_local"] = out["datetime"].dt.tz_convert(market_tz).dt.date
    cols = [c for c in out.columns if c != "datetime"]
    return (out[cols]
            .dropna(subset=["date_local"])
            .drop_duplicates(subset=["date_local"])
            .sort_values("date_local"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("TARGET_BUCKET","targeted_ETFs_US"))
    ap.add_argument("--out_dir", default=str(P.DATA_ENRICHED / "30min"))
    ap.add_argument("--daily_context", default=str(P.DATA_ENRICHED / "prices_enriched.parquet"))
    ap.add_argument("--macro_context", default=str(P.DATA_ENRICHED / "macro_forex_enriched.parquet"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = resolve_in_dir(args.bucket)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    mkt_tz = os.environ.get("MARKET_TZ","Europe/London")

    files = sorted(list(src.glob("*_30min*.csv")))
    if not files:
        raise SystemExit(f"[ERR] No 30m inputs in {src}")

    # ETF daily columns: prefer this small set; fallback to all *_d if present (normalize to lowercase)
    desired_etf_d = [
        "ema5_d","ema20_d","ema44_d","sma150_d",
        "rsi14_d","atr14_d","atr14_norm_d","adx14_d"
    ]
    all_d_cols = _infer_etf_daily_cols(Path(args.daily_context), whitelist=None)
    etf_keep = sorted(set([c.strip().lower() for c in desired_etf_d]).union(all_d_cols))

    # Pre-load Macro/FX (ALL columns, no filtering)
    macro_ctx = load_macro_context(Path(args.macro_context), mkt_tz)

    catalog = []
    for f in files:
        t = f.name.replace("_30min_raw.csv","").replace("_30min.csv","")
        try:
            df = read_30m(f)
            if df.empty:
                print(f"[WARN] {t}: no valid rows after parsing; skipping.")
                continue

            e  = enrich_30m_only(df, mkt_tz)

            # ETF daily (same ticker) by local date
            try:
                if etf_keep:
                    dctx = load_etf_daily(Path(args.daily_context), t, etf_keep, mkt_tz)
                    e = e.merge(dctx, on="date_local", how="left")
            except Exception as ex:
                print(f"[WARN] {t}: ETF daily merge skipped → {ex}")

            # Macro/FX daily (ALL columns) by local date
            try:
                if not macro_ctx.empty:
                    e = e.merge(macro_ctx, on="date_local", how="left")
            except Exception as ex:
                print(f"[WARN] {t}: Macro daily merge skipped → {ex}")

            # drop helper key after merges
            e = e.drop(columns=["date_local"])

            # write parquet
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
        dbg = P.REPORTS_DIR / "debug_enriched"; dbg.mkdir(parents=True, exist_ok=True)
        sp = dbg / f"s32_{sample_t}_sample.parquet"
        s.to_parquet(sp, index=False)
        print(f"[DEBUG] Saved sample → {sp}")

if __name__ == "__main__":
    main()