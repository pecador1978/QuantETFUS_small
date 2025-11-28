#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s84_market_traffic_light.py — Daily market regime + rotation “traffic light”
-----------------------------------------------------------------------------

Reads the *daily* master parquet (prices_enriched.parquet) and computes:
- Trend regime from SPY (or fallback) via EMA(20/44) on daily close
- DRI-style rotation mix: defensives vs SPY + optional VIX spot + VX term + UVOL/DVOL
- Status mapping:
    red    = regime is down
    orange = regime up AND rotation mix >= warn threshold
    green  = regime up AND rotation mix <  warn threshold
    orange (neutral) if regime is neutral

Output JSON (default: SIGNALS_SHARED_DIR/market_status.json):
{
  "status": "green|orange|red|unknown",
  "regime": "up|down|neutral",
  "dri_pct": 0.42,
  "bench": "SPY",
  "lookback": 20,
  "ema_fast": 20,
  "ema_slow": 44,
  "warn_pct": 0.5,
  "parts": {"base":true,"vix":true,"term":false,"breadth":true},
  "used_components": ["XLU","XLP","..."],
  "notes": {"vix_pct":1.2,"term_sig":0.35,"breadth_sig":-0.18},
  "updated_utc": "..."
}

Environment (all optional):
  PROJECT_ROOT, PROJECT_SHARED, DATA_ENRICHED_BASE, SIGNALS_SHARED_DIR
  MS_LOOKBACK (20), MS_SMOOTH (5), MS_EMA_FAST (20), MS_EMA_SLOW (44), MS_WARN_PCT (0.5)
  MS_BENCH (SPY)
  MS_COMPONENTS ("XLU,XLP,XLV,XLC,GLD,TLT")
  MS_USE_VIX (1), MS_VIX (VIX), MS_W_VIX (1.0)
  MS_USE_TERM (1), MS_VX1 (VX1), MS_VX2 (VX2), MS_W_TERM (1.5), MS_TERM_SMOOTH (5)
  MS_USE_BREADTH (1), MS_UVOL (UVOL), MS_DVOL (DVOL), MS_W_BREADTH (1.0), MS_BREADTH_SMOOTH (3)
"""

from __future__ import annotations
import os, json
from pathlib import Path
from datetime import timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------- Tunables (env) ----------------
LOOKBACK  = int(os.environ.get("MS_LOOKBACK", 20))
SMOOTH    = int(os.environ.get("MS_SMOOTH", 5))          # (kept for parity; final mix is already smooth-ish)
EMA_FAST  = int(os.environ.get("MS_EMA_FAST", 20))
EMA_SLOW  = int(os.environ.get("MS_EMA_SLOW", 44))
WARN_PCT  = float(os.environ.get("MS_WARN_PCT", 0.5))    # rotation warn threshold in %

DEF_BENCH = os.environ.get("MS_BENCH", "SPY").upper()
DEF_COMPS = os.environ.get("MS_COMPONENTS", "XLU,XLP,XLV,XLC,GLD,TLT")
COMP_LIST = [c.strip().upper() for c in DEF_COMPS.split(",") if c.strip()]

# VIX spot
USE_VIX = os.environ.get("MS_USE_VIX", "1") == "1"
VIX_SYM = os.environ.get("MS_VIX", "VIX").upper()
W_VIX   = float(os.environ.get("MS_W_VIX", 1.0))

# VIX term (VX2/VX1 contango → negative rotation; backwardation (term<0) → risk-on warning)
USE_TERM     = os.environ.get("MS_USE_TERM", "1") == "1"
VX1_SYM      = os.environ.get("MS_VX1", "VX1").upper()
VX2_SYM      = os.environ.get("MS_VX2", "VX2").upper()
W_TERM       = float(os.environ.get("MS_W_TERM", 1.5))
TERM_SMOOTH  = int(os.environ.get("MS_TERM_SMOOTH", 5))

# Breadth (US market up/down volume)
USE_BREADTH     = os.environ.get("MS_USE_BREADTH", "1") == "1"
UVOL_SYM        = os.environ.get("MS_UVOL", "UVOL").upper()
DVOL_SYM        = os.environ.get("MS_DVOL", "DVOL").upper()
W_BREADTH       = float(os.environ.get("MS_W_BREADTH", 1.0))
BREADTH_SMOOTH  = int(os.environ.get("MS_BREADTH_SMOOTH", 3))


# ---------------- Paths ----------------
def _signals_dir() -> Path:
    pr = Path(os.environ.get("PROJECT_ROOT", "."))
    p = Path(os.environ.get("SIGNALS_SHARED_DIR", pr / "signals"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _daily_master_path() -> Path:
    """Resolve data_enriched/prices_enriched.parquet."""
    # Highest-priority: explicit env
    base = os.environ.get("DATA_ENRICHED_BASE", "")
    if base:
        p = Path(base) / "prices_enriched.parquet"
        if p.exists():
            return p
    # Project shared
    pj = os.environ.get("PROJECT_SHARED", "")
    if pj:
        p = Path(pj) / "data_enriched" / "prices_enriched.parquet"
        if p.exists():
            return p
    # Fallback: PROJECT_ROOT
    pr = Path(os.environ.get("PROJECT_ROOT", "."))
    p = pr / "data_enriched" / "prices_enriched.parquet"
    return p


# ---------------- IO helpers ----------------
NUMERIC_CLOSE_CANDIDATES = ("close","Close","adj_close","Adj Close","price_close","last","px_close")

def _read_master_subset(master: Path, tickers: Iterable[str], cols: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Read only the requested tickers from the daily master parquet.
    Works whether the file has 'datetime' (tz-aware) or 'date', and
    avoids asking Arrow for non-existent columns.
    """
    want = list({t.upper() for t in tickers})

    # minimal set we try first
    candidate_close_cols = list(NUMERIC_CLOSE_CANDIDATES)
    preferred_cols = ["ticker", "datetime"] + candidate_close_cols  # don't include 'date' up front

    def try_read(columns: list[str]):
        try:
            return pd.read_parquet(master, columns=columns, filters=[("ticker", "in", want)], engine="pyarrow")
        except Exception:
            # fallback: read without filters but keep columns if possible
            try:
                return pd.read_parquet(master, columns=columns)
            except Exception:
                # last resort: read entire file (could be large, but robust)
                return pd.read_parquet(master)

    df = try_read(preferred_cols)

    # Normalize ticker to string/uppercase for filtering
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "dt", "close"]).astype({"ticker": "string"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"].isin(want)]

    # Build a unified datetime index/column "dt"
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], utc=True, errors="coerce")
    else:
        # try index
        dt = pd.to_datetime(df.index, utc=True, errors="coerce")

    df = df.assign(dt=dt)

    # Find a close-like column
    close_col = None
    for c in candidate_close_cols:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        # pick first numeric column (excluding known non-price fields)
        skip = {"ticker", "dt", "datetime", "date", "__fragment_index", "__batch_index", "__last_in_fragment", "__filename"}
        for c in df.columns:
            if c in skip:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                close_col = c
                break
    if close_col is None:
        return pd.DataFrame(columns=["ticker", "dt", "close"]).astype({"ticker": "string"})

    out = (
        df.loc[:, ["ticker", "dt", close_col]]
          .rename(columns={close_col: "close"})
          .dropna(subset=["dt", "close"])
          .sort_values(["ticker", "dt"])
    )
    return out


def _series_from(df: pd.DataFrame, symbol: str) -> Optional[pd.Series]:
    """Return a daily close series (UTC) for symbol from subset dataframe."""
    s = df.loc[df["ticker"].str.upper()==symbol, ["dt","close"]]
    if s.empty:
        return None
    out = pd.to_numeric(s["close"], errors="coerce")
    out.index = pd.to_datetime(s["dt"], utc=True, errors="coerce")
    out = out.sort_index().dropna()
    # keep a reasonable history
    return out.iloc[-10000:] if len(out) > 10000 else out


# ---------------- Math helpers ----------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n//2)).mean()

def pct_over(s: pd.Series, lbk: int) -> pd.Series:
    return s / s.shift(lbk) - 1.0


# ---------------- Core logic ----------------
def compute_status_from_daily(master_path: Path,
                              bench: str,
                              comps: list[str]) -> dict:
    """
    Implements the TradingView-like logic on *daily data* from the master parquet.
    """

    # Bench fallback chain (so LSE can use CSPX/VUAA if SPY missing)
    bench_fallbacks = [bench, "SPY", "USPY", "CSPX", "VUAA"]
    need = set(comps + bench_fallbacks)
    if USE_VIX:  need.add(VIX_SYM)
    if USE_TERM: need.update([VX1_SYM, VX2_SYM])
    if USE_BREADTH: need.update([UVOL_SYM, DVOL_SYM])

    # Read subset
    if not master_path.exists():
        return {
            "status":"unknown",
            "reason":f"Master parquet not found: {master_path}",
            "updated_utc": pd.Timestamp.utcnow().isoformat()
        }

    sub = _read_master_subset(master_path, need)

    # Resolve an available benchmark
    bench_resolved = None
    for b in bench_fallbacks:
        s = _series_from(sub, b)
        if s is not None and len(s) >= max(EMA_SLOW+LOOKBACK+5, 60):
            bench_resolved = b
            s_bench = s
            break

    if bench_resolved is None:
        return {
            "status":"unknown",
            "reason":f"No usable benchmark among: {bench_fallbacks}",
            "bench": bench,
            "updated_utc": pd.Timestamp.utcnow().isoformat()
        }

    # Trend regime (EMAs)
    e_fast = ema(s_bench, EMA_FAST)
    e_slow = ema(s_bench, EMA_SLOW)
    last, lf, ls = s_bench.iloc[-1], e_fast.iloc[-1], e_slow.iloc[-1]
    if (last > lf) and (last > ls):
        regime = "up"
    elif (last < lf) and (last < ls):
        regime = "down"
    else:
        regime = "neutral"

    # Base defensives-vs-SPY (lookback relative)
    pct_b = pct_over(s_bench, LOOKBACK).iloc[-1]
    rels, used = [], []
    for c in comps:
        s = _series_from(sub, c)
        if s is None or len(s) < LOOKBACK + 2:
            continue
        v = pct_over(s, LOOKBACK).iloc[-1]
        if np.isfinite(v) and np.isfinite(pct_b):
            rels.append(float(v - pct_b))
            used.append(c)
    dri_base = float(np.nanmean(rels)) if rels else np.nan
    base_w   = float(len(rels)) if rels else 0.0

    # VIX spot
    vix_pct, has_vix = None, False
    if USE_VIX:
        s_vix = _series_from(sub, VIX_SYM)
        if s_vix is not None and len(s_vix) >= LOOKBACK + 2:
            v = pct_over(s_vix, LOOKBACK).iloc[-1]
            if np.isfinite(v):
                vix_pct, has_vix = float(v), True

    # VIX term structure (VX2/VX1), smoothed, inverted (backwardation→positive)
    term_sig, has_term = None, False
    if USE_TERM:
        s_v1 = _series_from(sub, VX1_SYM)
        s_v2 = _series_from(sub, VX2_SYM)
        if s_v1 is not None and s_v2 is not None:
            df = pd.concat([s_v1.rename("v1"), s_v2.rename("v2")], axis=1).dropna()
            if not df.empty:
                contango = df["v2"] / df["v1"] - 1.0       # >0 = contango; <0 = backwardation
                ts = ema(contango, TERM_SMOOTH)
                sig = -(ts.iloc[-1])                       # backwardation => +, aligns with “warn”
                if np.isfinite(sig):
                    term_sig, has_term = float(sig), True

    # Breadth (DVOL vs UVOL)
    breadth_sig, has_breadth = None, False
    if USE_BREADTH:
        s_u = _series_from(sub, UVOL_SYM)
        s_d = _series_from(sub, DVOL_SYM)
        if s_u is not None and s_d is not None:
            df = pd.concat([s_u.rename("u"), s_d.rename("d")], axis=1).dropna()
            if not df.empty:
                raw = (df["d"] - df["u"]) / (df["d"] + df["u"])
                bs  = ema(raw, BREADTH_SMOOTH)
                val = bs.iloc[-1]
                if np.isfinite(val):
                    breadth_sig, has_breadth = float(val), True

    # Combine weighted
    total_w, mix = 0.0, 0.0
    if np.isfinite(dri_base):
        mix += base_w * dri_base; total_w += base_w
    if has_vix:
        mix += W_VIX * vix_pct; total_w += W_VIX
    if has_term:
        mix += W_TERM * term_sig; total_w += W_TERM
    if has_breadth:
        mix += W_BREADTH * breadth_sig; total_w += W_BREADTH

    dri = (mix / total_w) if total_w > 0 else np.nan
    dri_pct = float(100.0 * dri) if np.isfinite(dri) else None

    # Status mapping
    warn = WARN_PCT / 100.0
    if regime == "down":
        status = "red"
    elif regime == "up" and (np.isfinite(dri) and (dri >= warn)):
        status = "orange"
    elif regime == "up":
        status = "green"
    else:
        status = "orange"

    out = {
        "status": status,
        "regime": regime,
        "dri_pct": round(dri_pct, 2) if dri_pct is not None else None,
        "bench": bench_resolved,
        "lookback": LOOKBACK,
        "ema_fast": EMA_FAST,
        "ema_slow": EMA_SLOW,
        "warn_pct": WARN_PCT,
        "parts": {
            "base": bool(np.isfinite(dri_base)),
            "vix": bool(has_vix),
            "term": bool(has_term),
            "breadth": bool(has_breadth),
        },
        "used_components": used,
        "notes": {
            "vix_pct": round(100*vix_pct, 2) if vix_pct is not None else None,
            "term_sig": round(100*term_sig, 2) if term_sig is not None else None,
            "breadth_sig": round(100*breadth_sig, 2) if breadth_sig is not None else None,
        },
        "updated_utc": pd.Timestamp.utcnow().replace(tzinfo=timezone.utc).isoformat(),
    }
    return out


# ---------------- CLI ----------------
def main():
    # Resolve paths
    master = _daily_master_path()
    out_json = _signals_dir() / "market_status.json"

    import argparse
    ap = argparse.ArgumentParser(description="Daily market traffic light (TV-like).")
    ap.add_argument("--master", default=str(master), help="Path to prices_enriched.parquet")
    ap.add_argument("--bench", default=DEF_BENCH, help="Benchmark ticker (default: SPY with fallbacks)")
    ap.add_argument("--components", default=",".join(COMP_LIST), help="Comma-separated defensives: XLU,XLP,...")
    ap.add_argument("--output", default=str(out_json), help="Output JSON")
    args = ap.parse_args()

    comps = [c.strip().upper() for c in args.components.split(",") if c.strip()]

    out = compute_status_from_daily(Path(args.master), args.bench.strip().upper(), comps)

    # JSON-safe (avoid numpy types)
    def _to_native(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        raise TypeError(f"{type(o)} not serializable")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=_to_native))

    status = out.get("status","unknown")
    regime = out.get("regime","?")
    print(f"[OK] Market status → {args.output} :: {status} | regime={regime} | DRI={out.get('dri_pct')}% | parts={out.get('parts')}")
    if out.get("status") == "unknown":
        print(f"[WARN] Reason: {out.get('reason','n/a')}")


if __name__ == "__main__":
    main()