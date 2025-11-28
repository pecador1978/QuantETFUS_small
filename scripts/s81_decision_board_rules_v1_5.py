#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s81_decision_board_rules_v1_6_slim.py
Gate-1 (s77) + Gate-1.5 + Daily context + Trend Stage + Trust overlay
Columns kept (per user): Age, Trade, Trend Phase, Trend, Stage, Strength-20/44,
Sentiment, Smart Money, RSI, ADX, P25@5D, P75@5D, Bucket, Trust(merged),
MED touch20, Final Conf, FB↓20, Reason(Long), Reason(Short)

Features
--------
- Stage & Trend chips (interactive filters) + search box
- Reasons rendered inline as chips (single-line)
- Trust merged pill (quality + trust%)
- Final Confidence = CONF(0..100) × trust_score (0..1)
- False-break depth below EMA20 (FB↓20) from latest undercut sequence before a reclaim
- Snapshot fallback for confidence/bucket; safe merge of Gate-1.5 CSVs
- Glossary preserved
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os, json
from typing import Optional, Dict, Any
import json, os
import numpy as np
import pandas as pd

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402

pd.set_option("future.no_silent_downcasting", True)

# ----------------- helpers -----------------
def _wilder_rsi(series: pd.Series, length: int = 14) -> float:
    """Compute Wilder RSI on a price series and return the latest value."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < length + 1:
        return float("nan")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing (EMA with alpha=1/length)
    alpha = 1.0 / float(length)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1])

def load_market_status(path:str|None):
    try:
        if not path: return None
        p = Path(path)
        if not p.exists(): return None
        with p.open("r") as f:
            data = json.load(f)
        # sanity
        if "status" not in data:
            return None
        return data
    except Exception:
        return None

def traffic_light_markup(ms: dict | None) -> str:
    """
    Returns an HTML snippet with 3 lights + label.
    ms['status'] in {'green','orange','red','unknown'}
    """
    status = "unknown"; regime = "—"; dri = None
    if ms:
        status = str(ms.get("status", "unknown")).lower()
        regime = ms.get("regime", "—")
        dri    = ms.get("dri_pct", None)

    # which light is “on”
    on = {"red": 0, "orange": 1, "green": 2}.get(status, -1)

    def dot(cls: str, idx: int) -> str:
        return f'<div class="ms-dot {cls}{" on" if idx == on else ""}"></div>'

    css = """
    <style>
      .ms-traffic { display:flex; align-items:center; gap:.75rem; }
      .ms-lights  { display:flex; gap:.4rem; padding:.28rem .6rem; border-radius:999px; background:#121212; }
      .ms-dot     { width:14px; height:14px; border-radius:50%; opacity:.28;
                    box-shadow: inset 0 0 0 1px rgba(255,255,255,.10); }
      .ms-dot.red    { background:#f32424; }
      .ms-dot.orange { background:#ffb347; }   /* brighter orange */
      .ms-dot.green  { background:#34c759; }
      .ms-dot.on     { opacity:1; box-shadow: 0 0 10px rgba(255,255,255,.25),
                                     inset 0 0 0 1px rgba(255,255,255,.20); }
      /* When overall status is ORANGE, make it pop a bit more */
      .ms-traffic.ms-orange .ms-dot.orange.on {
        box-shadow: 0 0 14px rgba(255,179,71,.70), 0 0 4px rgba(255,179,71,.55),
                    inset 0 0 0 1px rgba(255,255,255,.22);
        transform: translateZ(0); /* crisp glow */
      }
      .ms-traffic.ms-orange .ms-label { color:#ffb347; } /* tint text */
      .ms-label   { font-weight:600; color:#cbd5e1; }
    </style>
    """

    note = f" | DRI {float(dri):.2f}%" if isinstance(dri, (int, float)) else ""
    # add status modifier class (ms-red | ms-orange | ms-green | ms-unknown)
    html = f"""{css}
    <div class="ms-traffic ms-{status}">
      <div class="ms-lights">
        {dot("red",0)}{dot("orange",1)}{dot("green",2)}
      </div>
      <div class="ms-label">Market {status.upper()} ({regime}){note}</div>
    </div>"""
    return html

# Load and render traffic light (default JSON path)
market_status_path = os.path.join(os.environ.get("SIGNALS_SHARED_DIR","signals"), "market_status.json")
traffic_html = traffic_light_markup(load_market_status(market_status_path))

def _load_intraday_rsi_map(dir30: Path, col_prefer: str = "rsi14_m30") -> dict[str, float]:
    """
    Scan 30m enriched parquet files and pull the latest RSI:
    - Prefer precomputed 'rsi14_m30' if present.
    - Otherwise compute RS I(14) from 'close'.
    """
    out: dict[str, float] = {}
    if not dir30.exists():
        print(f"[WARN] 30m dir not found: {dir30}")
        return out
    files = sorted(dir30.glob("*.parquet"))
    for fp in files:
        t = fp.stem.upper()
        try:
            df = pd.read_parquet(fp, columns=["datetime", col_prefer, "close"])
        except Exception:
            # If column pruning fails, read fully
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
        if df is None or df.empty:
            continue
        df = df.sort_values("datetime")
        val = pd.to_numeric(df.get(col_prefer), errors="coerce") if col_prefer in df.columns else None
        rsi_val = float("nan")
        if val is not None and val.notna().any():
            rsi_val = float(val.dropna().iloc[-1])
        else:
            # compute from close if needed
            if "close" in df.columns:
                rsi_val = _wilder_rsi(df["close"], 14)
        if pd.notna(rsi_val):
            out[t] = rsi_val
    return out


def _load_daily_close_history(path: Path, lookback: int = 200) -> dict[str, pd.Series]:
    """Return per-ticker series of recent *daily* closes (last N)."""
    out = {}
    try:
        df = pd.read_parquet(path, columns=["ticker", "datetime", "close"])
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.dropna(subset=["ticker", "close"]).sort_values(["ticker", "datetime"])
        for t, g in df.groupby("ticker"):
            s = pd.to_numeric(g["close"], errors="coerce").dropna().tail(lookback)
            if len(s) >= 15:
                out[t] = s.reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] load close history failed: {e}")
    return out

def _load_intraday_last_close(dir30: Path) -> dict[str, float]:
    """Read latest close from 30m parquet for each ticker (last bar = live proxy for today’s close)."""
    out = {}
    if not dir30.exists():
        print(f"[WARN] 30m dir not found: {dir30}")
        return out
    for fp in dir30.glob("*.parquet"):
        t = fp.stem.upper()
        try:
            df = pd.read_parquet(fp, columns=["datetime", "close"]).sort_values("datetime")
            if not df.empty and pd.notna(df["close"].iloc[-1]):
                out[t] = float(df["close"].iloc[-1])
        except Exception:
            continue
    return out
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def _find_latest_by_label(signals_dir: Path, label: str) -> tuple[Path, Path, str]:
    metas = sorted(signals_dir.glob("rule_live_signals_*.metadata.json"))
    best_meta = None
    best_ts = None
    for mp in metas:
        try:
            obj = json.loads(mp.read_text())
            if str(obj.get("label", "")).strip().lower() != str(label).strip().lower():
                continue
            stem = mp.name.replace(".metadata.json", "")
            ts = stem.split("_", 3)[-1]  # YYYYMMDD_HHMM
            if best_ts is None or ts > best_ts:
                best_ts, best_meta = ts, mp
        except Exception:
            continue
    if not best_meta or not best_ts:
        raise SystemExit(f"[ERR] No metadata found in {signals_dir} for label='{label}'.")
    csv_path = signals_dir / f"rule_live_signals_{best_ts}.csv"
    if not csv_path.exists():
        raise SystemExit(f"[ERR] Paired CSV not found for label='{label}': {csv_path.name}")
    return csv_path, best_meta, best_ts

def _trend_phase(close_d: float, ema20_d: float, ema44_d: float) -> str:
    if np.isfinite(close_d) and np.isfinite(ema20_d) and np.isfinite(ema44_d):
        if (ema20_d > ema44_d) and (close_d >= ema20_d): return "BULLISH"
        if (ema20_d < ema44_d) and (close_d <= ema44_d): return "BEARISH"
        return "NEUTRAL"
    return ""

def _trend_short(close_d: float, ema20_d: float) -> str:
    if np.isfinite(close_d) and np.isfinite(ema20_d):
        return "UP" if float(close_d) >= float(ema20_d) else "DOWN"
    return ""

def _sentiment_risk(close_d: float, ema20_d: float, regime: str,
                    smartmoney: str, adx_val: float, adx_thr: float = 20.0) -> str:
    if not (np.isfinite(close_d) and np.isfinite(ema20_d)):
        return "Risk OFF"
    if float(close_d) < float(ema20_d):
        return "Risk OFF"
    if (regime or "").strip() == "above_p80":
        return "Risk OFF"
    if (smartmoney or "").lower() in ("distribution", "markdown"):
        return "Risk OFF"
    if np.isfinite(adx_val) and float(adx_val) < float(adx_thr):
        return "Risk OFF"
    return "Risk ON"

def _smart_money_wyckoff(close_t: float, close_y: float,
                         vol_t: Optional[float], vol_y: Optional[float],
                         ema20_t: float) -> str:
    if not (np.isfinite(close_t) and np.isfinite(close_y) and np.isfinite(ema20_t)):
        return "Neutral"
    price_change = float(close_t - close_y)
    trend_dir = 1 if close_t > float(ema20_t) else -1
    flat_thresh = abs(close_t) * 0.002  # 0.2%
    is_flat = abs(price_change) < flat_thresh
    vol_ok = np.isfinite(vol_t) and np.isfinite(vol_y)
    vol_rising = (float(vol_t) > float(vol_y)) if vol_ok else None
    if is_flat and vol_ok:
        if not vol_rising: return "Accumulation"
        if vol_rising and trend_dir < 0: return "Distribution"
    if trend_dir > 0 and price_change > 0: return "Markup"
    if trend_dir < 0 and price_change < 0: return "Markdown"
    return "Neutral"

def _trend_stage(row) -> tuple[str, int]:
    tr  = str(row.get("trend","")).upper()
    age = float(row.get("trend_age", np.nan))
    s20 = float(row.get("strength20_pct", np.nan))
    s44 = float(row.get("strength44_pct", np.nan))
    rsi = float(row.get("rsi14_d", np.nan))
    reg = str(row.get("stretch_regime",""))
    sm  = str(row.get("smartmoney","")).lower()
    rc  = bool(row.get("recent_cross_above_ema20_3d", False))

    if tr != "UP" or not np.isfinite(s20) or not np.isfinite(rsi):
        return ("—", 9)

    if ((np.isfinite(age) and age <= 5) or rc) \
       and (0.0 <= s20 <= 2.5) and (np.isfinite(s44) and s44 <= 6.0) \
       and (45 <= rsi <= 65) and reg != "above_p80":
        return ("Emerging 🔥", 0)

    if (-1.8 <= s20 <= 0.8) and (48 <= rsi <= 62) and reg != "above_p80":
        if sm in ("accumulation","neutral"):
            return ("Pullback-20 🔁", 1)
        return ("Pullback-20 🔁", 2)

    if (s20 <= -0.5) and np.isfinite(s44) and (0.0 <= s44 <= 3.5) and (45 <= rsi <= 58):
        if sm in ("accumulation","neutral"):
            return ("Pullback-44 ⚓", 2)
        return ("Pullback-44 ⚓", 3)

    if (0.25 <= s20 <= 5.0) and (np.isfinite(s44) and s44 < 12.0) and (45 <= rsi <= 68) and reg != "above_p80":
        return ("Steady ✅", 4)

    if (s20 > 5.0) or (np.isfinite(s44) and s44 > 12.0) or (rsi > 68) or (reg == "above_p80"):
        return ("Stretched ⚠️", 6)

    if (-2.5 <= s20 <= 1.0) and (40 <= rsi < 48):
        return ("Near-20 🧪", 5)

    return ("—", 9)

# -------- daily enriched loader (from s00) + FB↓20 --------
def _load_daily_enriched_master(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        if not path.exists():
            print(f"[WARN] Daily enriched parquet not found: {path}")
            return out

        df = pd.read_parquet(path)
        if df.empty:
            print("[WARN] Daily enriched parquet is empty.")
            return out

        df.columns = [c.strip().lower() for c in df.columns]
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["datetime"] = dt
        df = df.dropna(subset=["datetime", "ticker"]).copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df.sort_values(["ticker", "datetime"])

        keep = ["close","volume","ema20_d","ema44_d","rsi14_d","adx14_d","trend_days_since_start_d","atr14_d"]
        for k in keep:
            if k not in df.columns:
                df[k] = np.nan

        # latest row per ticker
        last = df.groupby("ticker", as_index=False).tail(1).reset_index(drop=True)
        for _, r in last.iterrows():
            t = str(r["ticker"]).upper()
            out[t] = {
                "close": float(r.get("close", np.nan)) if np.isfinite(r.get("close", np.nan)) else np.nan,
                "volume": float(r.get("volume", np.nan)) if np.isfinite(r.get("volume", np.nan)) else np.nan,
                "ema20_d": float(r.get("ema20_d", np.nan)) if np.isfinite(r.get("ema20_d", np.nan)) else np.nan,
                "ema44_d": float(r.get("ema44_d", np.nan)) if np.isfinite(r.get("ema44_d", np.nan)) else np.nan,
                "rsi14_d": float(r.get("rsi14_d", np.nan)) if np.isfinite(r.get("rsi14_d", np.nan)) else np.nan,
                "adx14_d": float(r.get("adx14_d", np.nan)) if np.isfinite(r.get("adx14_d", np.nan)) else np.nan,
                "trend_days_since_start_d": float(r.get("trend_days_since_start_d", np.nan)),
                "atr14_d": float(r.get("atr14_d", np.nan)) if np.isfinite(r.get("atr14_d", np.nan)) else np.nan,
            }

        # previous day close/vol
        prev = df.groupby("ticker", as_index=False).tail(2).reset_index(drop=True)
        prev = prev.groupby("ticker").nth(-2).reset_index()
        prev["ticker"] = prev["ticker"].astype(str).str.strip().str.upper()
        for _, r in prev.iterrows():
            t = str(r["ticker"]).upper()
            if t in out:
                out[t]["close_y"] = float(r.get("close", np.nan)) if np.isfinite(r.get("close", np.nan)) else np.nan
                out[t]["volume_y"] = float(r.get("volume", np.nan)) if np.isfinite(r.get("volume", np.nan)) else np.nan

        # recent 3 bars reclaim flag
        tri = df.groupby("ticker", as_index=False).tail(3).copy()
        tri["ticker"] = tri["ticker"].astype(str).str.strip().str.upper()
        recent_reclaim = {}
        for t, g in tri.groupby("ticker"):
            g = g.sort_values("datetime")
            crossed = False
            vals = list(zip(g.get("close", []), g.get("ema20_d", [])))
            for i in range(1, len(vals)):
                c_y, e_y = vals[i-1]; c_t, e_t = vals[i]
                if np.isfinite(c_y) and np.isfinite(e_y) and np.isfinite(c_t) and np.isfinite(e_t):
                    if (c_y < e_y) and (c_t >= e_t):
                        crossed = True
            recent_reclaim[t] = crossed
        for t in out.keys():
            out[t]["recent_cross_above_ema20_3d"] = bool(recent_reclaim.get(t, False))

        # ---- False-break depth below EMA20 (FB↓20) before latest reclaim ----
        fb_cap_pct = 20.0  # cap to avoid junk values
        for t, g in df.groupby("ticker"):
            g = g.sort_values("datetime").tail(30)
            if g.empty or t not in out:
                continue

            c_last = float(g["close"].iloc[-1])
            e_last = float(g["ema20_d"].iloc[-1])
            # only if currently reclaimed/above
            if not (np.isfinite(c_last) and np.isfinite(e_last) and c_last > 0 and e_last > 0 and c_last >= e_last):
                continue

            max_depth = None
            seen_below = False
            closes = g["close"].tolist()
            emas20 = g["ema20_d"].tolist()

            for c, e in zip(reversed(closes), reversed(emas20)):
                if not (np.isfinite(c) and np.isfinite(e) and c > 0 and e > 0):
                    if seen_below:
                        break
                    continue

                if c < e:
                    seen_below = True
                    depth_pct = 100.0 * (e - c) / e
                    if depth_pct <= fb_cap_pct:
                        # keep the maximum depth encountered in the last undercut streak
                        if (max_depth is None) or (depth_pct > max_depth):
                            max_depth = depth_pct
                else:
                    if seen_below:
                        break

            if max_depth is not None:
                out[t]["false_break_depth20_pct"] = float(max_depth)

        return out
    except Exception as e:
        print(f"[WARN] Failed reading daily enriched parquet: {e}")
        return out  # always return a dict

# -------- s83: model trust JSON loader --------
def _load_gate2_quality_json(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text())
        return {str(k).upper(): v for k, v in obj.items()}
    except Exception as e:
        print(f"[WARN] Could not read trust JSON: {e}")
        return {}

# ----------------- Gate-1.5 merge helpers -----------------
def _load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] Could not read {path.name}: {e}")
        return None

def _normalize_ticker_col(df: pd.DataFrame, src_name: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    candidates = {"ticker","Ticker","symbol","Symbol"}
    found = next((c for c in df.columns if c in candidates), None)
    if not found:
        print(f"[WARN] {src_name} has no ticker-like column → skipping")
        return None
    if found != "ticker":
        df = df.rename(columns={found:"ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["ticker"])

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    aliases = {
        "ret_p25_next5d": ["ret_p25_next5d","p25_ret_5d","ret25_5d"],
        "ret_p75_next5d": ["ret_p75_next5d","p75_ret_5d","ret75_5d"],
        "confidence_score_0_100": ["confidence_score_0_100","confidence_score","conf_score"],
        "confidence_bucket":      ["confidence_bucket","conf_bucket","bucket"],
        "stretch_regime":         ["stretch_regime","regime"],
        "median_bars_until_touch_ema20": ["median_bars_until_touch_ema20","med_touch20","median_touch_20"],
    }
    rename_map = {}
    for canon, alts in aliases.items():
        for a in alts:
            if a in df.columns:
                rename_map[a] = canon
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _load_gate15_all() -> Optional[pd.DataFrame]:
    base_dir = P.SIGNALS_DIR / "analytics"
    stats = _load_csv_safe(base_dir / "gate15_stats.csv")
    conf  = _load_csv_safe(base_dir / "gate15_calibration_confidence.csv")
    stre  = _load_csv_safe(base_dir / "gate15_calibration_stretch.csv")
    stats = _normalize_ticker_col(stats, "gate15_stats.csv") if stats is not None else None
    conf  = _normalize_ticker_col(conf,  "gate15_calibration_confidence.csv") if conf is not None else None
    stre  = _normalize_ticker_col(stre,  "gate15_calibration_stretch.csv")    if stre is not None else None
    merged = None
    for df_part in (stats, conf, stre):
        if df_part is not None:
            merged = df_part if merged is None else merged.merge(df_part, on="ticker", how="left")
    return _apply_aliases(merged) if merged is not None else None

# [ADD] Gate-2 snapshot fallback for confidence
def _load_gate2_conf_snapshot(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] Gate-2 snapshot not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        need = {"ticker","datetime","confidence_score"}
        if not need.issubset(set(df.columns)):
            print(f"[WARN] Snapshot missing {need} → skipping")
            return None
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        last = (df.sort_values(["ticker","datetime"]).groupby("ticker", as_index=False).tail(1))
        last["confidence_score_0_100"] = (last["confidence_score"].clip(0,1) * 100.0)
        last["confidence_bucket"] = last["confidence_score_0_100"].apply(
            lambda x: "HIGH" if x >= 66 else ("MED" if x >= 33 else "LOW")
        )
        return last[["ticker","confidence_score_0_100","confidence_bucket"]]
    except Exception as e:
        print(f"[WARN] Could not read snapshot: {e}")
        return None

# ----------------- HTML helpers -----------------
def _fmt_pct_smart(x, clip: float | None = None) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    # Heuristic unit detection:
    # - If |v| <= 0.5 → assume FRACTION (e.g., 0.019 = 1.9%) → *100
    # - Else → assume already PERCENT (e.g., 0.9 = 0.9%)
    if abs(v) <= 0.5:
        v *= 100.0
    if clip is not None:
        v = max(-clip, min(clip, v))
    return f"{v:.1f}"

def _fmt_pct_1d(x) -> str:
    try: v = float(x)
    except Exception: return "—"
    if not np.isfinite(v): return "—"
    if abs(v) <= 1.0: v *= 100.0
    return f"{v:.1f}"

def _fmt_xatr1d(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.1f}"

def _reason_chips_inline(reason: str) -> str:
    if not reason:
        return "—"
    chips = []
    for r in [x.strip() for x in str(reason).split(",") if x.strip()]:
        cls = "tag tag-warn" if r == "stretched_vs_ema44" else "tag"
        chips.append(f'<span class="{cls}">{r}</span>')
    return '<div class="reason" style="gap:6px;display:flex;flex-wrap:nowrap;overflow:hidden;text-overflow:ellipsis">' + " ".join(chips) + "</div>"

# ----------------- HTML builder -----------------
def build_html(df: pd.DataFrame, updated_label: str, label: str, adx_thr: float) -> str:
    style = """
<style>
:root{ --bg:#0b1020; --card:#0f172a; --muted:#94a3b8; --txt:#e2e8f0; --line:#1f2937; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial;color:#e2e8f0}
.wrap{max-width:100%;margin:10px;padding:0}
h1{margin:0 0 8px;font-size:22px}
.info{color:var(--muted);font-size:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.search{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:#0b132d;color:#e2e8f0;outline:none;margin-bottom:8px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px}
.filter-group{display:flex;align-items:center;gap:8px;background:#0f172a;border:1px solid var(--line);border-radius:10px;padding:6px 8px}
.filter-group .chips{display:flex;gap:6px;flex-wrap:wrap}
.filter-chip{display:flex;align-items:center;gap:6px;background:#0b132d;border:1px solid #2b344a;border-radius:999px;padding:2px 8px;font-size:12px}
.filter-chip input{accent-color:#93c5fd}
.btn-mini{font-size:12px;padding:2px 8px;border:1px solid #2b344a;background:#0b132d;border-radius:8px;color:#cbd5e1;cursor:pointer}
table{width:100%;border-collapse:collapse;table-layout:auto}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);z-index:2;text-align:left;font-weight:700;cursor:pointer}
tr:hover td{background:#0c1532}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:11px}
.badge-buy{background:rgba(5,150,105,.15);color:#86efac;border:1px solid rgba(5,150,105,.3)}
.badge-sell{background:rgba(220,38,38,.15);color:#fecaca;border:1px solid rgba(220,38,38,.3)}
.badge-wait{background:rgba(71,85,105,.2);color:#cbd5e1;border:1px solid rgba(71,85,105,.35)}
.phase-bullish{color:#86efac;font-weight:700} .phase-bearish{color:#f87171;font-weight:700} .phase-neutral{color:#cbd5e1}
.rsi-hot{color:#ef4444;font-weight:700} .rsi-cold{color:#3b82f6;font-weight:700}
.adx-strong{color:#86efac;font-weight:700} .adx-weak{color:#f87171;font-weight:700} .adx-neutral{color:#cbd5e1}
.trend-up{color:#86efac} .trend-down{color:#fca5a5}
.sent-on{color:#86efac;font-weight:700} .sent-off{color:#fca5a5;font-weight:700}
.sm-acc{color:#60a5fa} .sm-dist{color:#f59e0b} .sm-markup{color:#86efac;font-weight:700} .sm-markdown{color:#ef4444;font-weight:700}
.strg{display:inline-block;padding:2px 8px;border-radius:8px;font-weight:800}
.strg-neutral{background:#0f172a;color:#cbd5e1} .strg-ok{background:#052e1a;color:#86efac}
.strg-warn{background:#3b2a0b;color:#fbbf24} .strg-hot{background:#3b0a0a;color:#fca5a5}
.reason .tag{padding:2px 6px;border-radius:6px;font-size:11px;border:1px solid #2b344a;color:#cbd5e1;background:#101a2f}
.reason .tag-warn{border-color:#8a5500;color:#fbbf24;background:#2a1c05}
.pill{display:inline-block;padding:2px 8px;border-radius:8px;border:1px solid #2b344a;font-size:11px}
.pill-high{background:rgba(22,163,74,.12);color:#86efac;border-color:rgba(22,163,74,.30);font-weight:700}
.pill-med{background:rgba(245,158,11,.12);color:#fbbf24;border-color:rgba(245,158,11,.35);font-weight:700}
.pill-low{background:rgba(220,38,38,.12);color:#fca5a5;border-color:rgba(220,38,38,.35);font-weight:700}
.card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.updated{color:var(--muted);font-size:12px;margin-left:12px;white-space:nowrap}
/* sticky first column (Ticker) */
#tbl{border-collapse:separate;border-spacing:0}
#tbl th:first-child,#tbl td:first-child{position:sticky;left:0;background:var(--card)}
#tbl th:first-child{z-index:6} #tbl td:first-child{z-index:1;border-right:1px solid var(--line)}
</style>
"""
    script = """
<script>
(function(){
  const COL = {
    TICKER:0, AGE:1, TRADE:2,
    PHASE:3, TREND:4, STAGE:5,
    STR20:6, STR44:7, XATR20:8, XATR44:9,
    SENT:10, SM:11, RSI:12, ADX:13,
    P25_5:14, P75_5:15,
    BUCKET:16, TRUSTMERGED:17, MED20:18, FINALCONF:19, FB20:20,
    RLONG:21, RSHORT:22
  };

  const numIdx = new Set([COL.AGE,COL.STR20,COL.STR44,COL.RSI,COL.ADX,COL.P25_5,COL.P75_5,COL.MED20,COL.FINALCONF,COL.FB20]);

  const customOrder = new Map([
    [COL.BUCKET, ["HIGH","MED","LOW"]],
  ]);

  function getCellText(td){ return (td && td.textContent ? td.textContent : '').trim(); }
  function parseNum(s){
    const pct=s.replace(/[,%]/g,'');
    const num=pct.replace(/[^0-9.+-]/g,'');
    const v=parseFloat(num);
    return isNaN(v) ? Number.NEGATIVE_INFINITY : v;
  }

  function sortTable(table, colIndex, asc){
    const tbody=table.tBodies[0];
    const rows=Array.from(tbody.querySelectorAll('tr')).map((r,i)=>({r,i}));
    const order = customOrder.get(colIndex);
    rows.sort((A,B)=>{
      const a=getCellText(A.r.cells[colIndex]);
      const b=getCellText(B.r.cells[colIndex]);
      if(order){
        const ra = order.indexOf(a), rb = order.indexOf(b);
        const va = (ra === -1) ? Number.POSITIVE_INFINITY : ra;
        const vb = (rb === -1) ? Number.POSITIVE_INFINITY : rb;
        let cmp = va - vb; if(!asc) cmp = -cmp; return cmp !== 0 ? cmp : (A.i - B.i);
      }
      let cmp = numIdx.has(colIndex) ? (parseNum(a) - parseNum(b)) : a.toLowerCase().localeCompare(b.toLowerCase());
      if(!asc) cmp = -cmp; return cmp !== 0 ? cmp : (A.i - B.i);
    });
    rows.forEach(o=>tbody.appendChild(o.r));
  }

  function clearIndicators(ths){
    ths.forEach(th=>{ th.removeAttribute('data-asc'); th.setAttribute('aria-sort','none'); const s=th.querySelector('.sort-ind'); if(s) s.textContent='↕'; });
  }

  function stageKey(txt){ txt=(txt||'').trim(); return (!txt||txt==='—')?'__EMPTY__':txt; }
  function trendKey(txt){ txt=(txt||'').trim().toUpperCase(); return (!txt)?'__EMPTY__':txt; }
  function tradeKey(txt){ txt = (txt||'').trim().toUpperCase();  return (!txt) ? '__EMPTY__' : txt; }

  function applyFilters(){
  const tbody=document.getElementById('tbl').tBodies[0];
  const q=(document.getElementById('q')?.value||'').trim().toLowerCase();

  const chosenStage=new Set(Array.from(document.querySelectorAll('input[name="stage-filter"]:checked')).map(i=>i.value));
  const chosenTrend=new Set(Array.from(document.querySelectorAll('input[name="trend-filter"]:checked')).map(i=>i.value));
  const chosenTrade=new Set(Array.from(document.querySelectorAll('input[name="trade-filter"]:checked')).map(i=>i.value)); // NEW

  Array.from(tbody.rows).forEach(tr=>{
    const txt=(tr.getAttribute('data-f')||'').toLowerCase();
    const stageOk = chosenStage.size===0 || chosenStage.has(stageKey(tr.cells[COL.STAGE].textContent));
    const trendOk = chosenTrend.size===0 || chosenTrend.has(trendKey(tr.cells[COL.TREND].textContent));
    const tradeOk = chosenTrade.size===0 || chosenTrade.has(tradeKey(tr.cells[COL.TRADE].innerText)); // NEW
    const searchOk = !q || txt.includes(q);
    tr.style.display = (stageOk && trendOk && tradeOk && searchOk) ? '' : 'none';
  });
}
  window.filterRows = applyFilters;

  function populateChips(){
    const tbody=document.getElementById('tbl').tBodies[0];

    // Stage
    const setStage=new Set();
    Array.from(tbody.rows).forEach(tr=>{ setStage.add(stageKey(tr.cells[COL.STAGE].textContent)); });
    const contStage=document.getElementById('stage-chips');
    contStage.innerHTML='';
    Array.from(setStage).sort().forEach(k=>{
      const label=(k==='__EMPTY__')?'(empty)':k;
      const lab=document.createElement('label');
      lab.className='filter-chip';
      lab.innerHTML='<input type="checkbox" name="stage-filter" value="'+k+'" checked> '+label;
      contStage.appendChild(lab);
    });

    // Trend
    const setTrend=new Set();
    Array.from(tbody.rows).forEach(tr=>{ setTrend.add(trendKey(tr.cells[COL.TREND].textContent)); });
    const contTrend=document.getElementById('trend-chips');
    contTrend.innerHTML='';
    Array.from(setTrend).sort().forEach(k=>{
      if(k==='__EMPTY__') return;
      const lab=document.createElement('label');
      lab.className='filter-chip';
      lab.innerHTML='<input type="checkbox" name="trend-filter" value="'+k+'" checked> '+k;
      contTrend.appendChild(lab);
    });

    // Trade (BUY / DO NOTHING / SELL)
    const setTrade=new Set();
    Array.from(tbody.rows).forEach(tr=>{
    setTrade.add(tradeKey(tr.cells[COL.TRADE].innerText));
    });
    const contTrade=document.getElementById('trade-chips');
    contTrade.innerHTML='';
    Array.from(setTrade).sort().forEach(k=>{
      if(k==='__EMPTY__') return;
      const label = k; // already uppercase
      const lab=document.createElement('label');
      lab.className='filter-chip';
      lab.innerHTML='<input type="checkbox" name="trade-filter" value="'+k+'" checked> '+label;
    contTrade.appendChild(lab);
    });
    
    document.getElementById('trade-all').addEventListener('click',(e)=>{e.preventDefault(); contTrade.querySelectorAll('input').forEach(i=>i.checked=true); applyFilters();});
    document.getElementById('trade-none').addEventListener('click',(e)=>{e.preventDefault(); contTrade.querySelectorAll('input').forEach(i=>i.checked=false); applyFilters();});
    contTrade.addEventListener('change', applyFilters);
    document.getElementById('stage-all').addEventListener('click', (e)=>{e.preventDefault(); contStage.querySelectorAll('input').forEach(i=>i.checked=true); applyFilters();});
    document.getElementById('stage-none').addEventListener('click', (e)=>{e.preventDefault(); contStage.querySelectorAll('input').forEach(i=>i.checked=false); applyFilters();});
    document.getElementById('trend-all').addEventListener('click', (e)=>{e.preventDefault(); contTrend.querySelectorAll('input').forEach(i=>i.checked=true); applyFilters();});
    document.getElementById('trend-none').addEventListener('click', (e)=>{e.preventDefault(); contTrend.querySelectorAll('input').forEach(i=>i.checked=false); applyFilters();});
    contStage.addEventListener('change', applyFilters);
    contTrend.addEventListener('change', applyFilters);
  }

  function init(){
    const table=document.getElementById('tbl');
    const ths=Array.from(table.tHead.rows[0].cells);
    ths.forEach((th, idx)=>{
      const ind=document.createElement('span'); ind.className='sort-ind'; ind.style.marginLeft='6px'; ind.style.opacity='0.7'; ind.textContent='↕';
      th.appendChild(ind); th.setAttribute('role','columnheader'); th.setAttribute('aria-sort','none');
      th.addEventListener('click', ()=>{
        const asc=!(th.dataset.asc==='true'); clearIndicators(ths);
        th.dataset.asc=String(asc); th.setAttribute('aria-sort', asc?'ascending':'descending'); th.querySelector('.sort-ind').textContent=asc?'▲':'▼';
        sortTable(table, idx, asc);
      });
    });

    // default sort by FINAL CONF desc
    const fc = ths[COL.FINALCONF];
    fc.dataset.asc='false'; fc.querySelector('.sort-ind').textContent='▼'; fc.setAttribute('aria-sort','descending');
    sortTable(table, COL.FINALCONF, false);

    populateChips();
    applyFilters();
    const q=document.getElementById('q'); if(q){ q.addEventListener('input', applyFilters); }
  }
  document.addEventListener('DOMContentLoaded', init);
})();
</script>
"""
    def pill(decision: str) -> str:
        d = (decision or "").strip().upper()
        if d == "BUY":  return '<span class="badge badge-buy">CONSIDER</span>'
        if d == "SELL": return '<span class="badge badge-sell">SELL</span>'
        return '<span class="badge badge-wait">DO NOTHING</span>'

    def trend_cls(t: str) -> str:
        return "trend-up" if (t or "").upper() == "UP" else "trend-down"

    def sent_cls(s: str) -> str:
        return "sent-on" if (s or "").upper() == "RISK ON" else "sent-off"

    def sm_cls(s: str) -> str:
        s = (s or "").lower()
        if s == "accumulation": return "sm-acc"
        if s == "distribution": return "sm-dist"
        if s == "markup":       return "sm-markup"
        if s == "markdown":     return "sm-markdown"
        return ""

    def adx_cell(val, thr):
        if val is None or not np.isfinite(val):
            return '<span class="adx-neutral">—</span>';
        v = float(val)
        cls = "adx-strong" if v >= float(thr) else "adx-weak"
        return f'<span class="{cls}">{int(round(v))}</span>'

    def strg_cell(x) -> str:
        if x is None or not np.isfinite(x): return '<span class="strg strg-neutral">—</span>'
        p = float(x); label = f"{p:.2f}%"
        if p <= 0:   cls = "strg-neutral"
        elif p < 5:  cls = "strg-ok"
        elif p < 10: cls = "strg-warn"
        else:        cls = "strg-hot"
        return f'<span class="strg {cls}">{label}</span>'

    def pill_conf_bucket(bucket) -> str:
        b = "" if bucket is None or (isinstance(bucket, float) and np.isnan(bucket)) else str(bucket).strip().upper()
        if b == "HIGH": return '<span class="pill pill-high">HIGH</span>'
        if b == "LOW":  return '<span class="pill pill-low">LOW</span>'
        if b in ("MED","MEDIUM"): return '<span class="pill pill-med">MED</span>'
        return '<span class="pill">—</span>'

    def trust_merged_cell(quality: str, trust_score):
        q = (quality or "").strip().title()
        try:
            pct = int(round(100.0 * float(pd.to_numeric(trust_score, errors="coerce"))))
            pct_txt = f"{pct}"
        except Exception:
            pct_txt = "—"
        base = {"High":"pill-high","Medium":"pill-med","Low":"pill-low"}.get(q, "")
        label = q if q else "—"
        return f'<span class="pill {base}">{label} ({pct_txt}%)</span>'

    def fmt_int(x) -> str:
        try: return str(int(round(float(x))))
        except Exception: return "—"

    head = (
    "<div class=\"wrap\">"
    f"<h1>NY Signals Dashboard"
    "<div class=\"card\">"
      "<div class=\"card-head\">"
        f"{traffic_html}"
        f"<div class=\"updated\">Updated: {updated_label}</div>"
      "</div>"
      "<input id=\"q\" class=\"search\" oninput=\"filterRows()\" "
      "placeholder=\"Search ticker, phase, trend, sentiment, smart money…\"/>"
      "<div class=\"filters\">"
        "<div class=\"filter-group\">"
          "<span style=\"font-size:12px;color:#94a3b8\">Stage:</span>"
          "<div id=\"stage-chips\" class=\"chips\"></div>"
          "<button class=\"btn-mini\" id=\"stage-all\">All</button>"
          "<button class=\"btn-mini\" id=\"stage-none\">None</button>"
        "</div>"
        "<div class=\"filter-group\">"
          "<span style=\"font-size:12px;color:#94a3b8\">Trend:</span>"
          "<div id=\"trend-chips\" class=\"chips\"></div>"
          "<button class=\"btn-mini\" id=\"trend-all\">All</button>"
          "<button class=\"btn-mini\" id=\"trend-none\">None</button>"
        "</div>"
        "<div class=\"filter-group\">"
          "<span style=\"font-size:12px;color:#94a3b8\">Trade:</span>"
          "<div id=\"trade-chips\" class=\"chips\"></div>"
          "<button class=\"btn-mini\" id=\"trade-all\">All</button>"
          "<button class=\"btn-mini\" id=\"trade-none\">None</button>"
        "</div>"
      "</div>"
      "<div style=\"overflow:auto; max-height:78vh; position:relative;\">"
      "<table id=\"tbl\">"
      "<thead><tr>"
      "<th>TICKER</th><th>AGE</th><th>TRADE</th>"
      "<th>TREND PHASE</th><th>TREND</th><th>STAGE</th>"
      "<th>STRENGTH-20</th><th>STRENGTH-44</th>"
      "<th>xATR@20</th><th>xATR@44</th>"
      "<th>SENTIMENT</th><th>SMART MONEY</th>"
      "<th>RSI (live)</th><th>ADX</th>"
      "<th>P25@5D</th><th>P75@5D</th>"
      "<th>BUCKET</th><th>TRUST</th><th>MED touch20</th><th>FINAL CONF</th><th>FB↓20</th>"
      "<th>REASON (LONG)</th><th>REASON (SHORT)</th>"
      "</tr></thead><tbody>"
)

    rows_html = []
    for _, r in df.iterrows():
        data_filter = " ".join([
            str(r.get("ticker", "")),
            str(r.get("decision", "")),
            str(r.get("trend_phase", "")),
            str(r.get("trend", "")),
            str(r.get("trend_stage", "")),
            str(r.get("sentiment", "")),
            str(r.get("smartmoney", "")),
            str(r.get("confidence_bucket", "")),
            str(r.get("stretch_regime", "")),
            str(r.get("model_quality", "")),
            str(r.get("reason_long", "")),
            str(r.get("reason_short", "")),
        ])

        phase = str(r.get("trend_phase","")).upper()
        phase_cls = "phase-neutral"
        if phase == "BULLISH": phase_cls = "phase-bullish"
        elif phase == "BEARISH": phase_cls = "phase-bearish"

        rsi_val = r.get("rsi14_live", r.get("rsi14_d", np.nan))
        rsi_txt = ""; rsi_cls = ""
        if np.isfinite(rsi_val):
            rsi_txt = str(int(round(rsi_val)))
            if rsi_val >= 70: rsi_cls = "rsi-hot"
            elif rsi_val <= 30: rsi_cls = "rsi-cold"

        adx_val = r.get("adx14_d", np.nan)

        rows_html.append(
            f'<tr data-f="{data_filter}">'
            f'<td>{r.get("ticker","")}</td>'
            f'<td>{fmt_int(r.get("trend_age", np.nan))}</td>'
            f'<td>{pill(r.get("decision",""))}</td>'
            f'<td class="{phase_cls}">{phase}</td>'
            f'<td class="{trend_cls(r.get("trend",""))}">{r.get("trend","")}</td>'
            f'<td>{r.get("trend_stage","—")}</td>'
            f'<td>{strg_cell(r.get("strength20_pct", None))}</td>'
            f'<td>{strg_cell(r.get("strength44_pct", None))}</td>'
            f'<td>{_fmt_xatr1d(r.get("xatr_vs_ema20", np.nan))}</td>'
            f'<td>{_fmt_xatr1d(r.get("xatr_vs_ema44", np.nan))}</td>'
            f'<td class="{sent_cls(r.get("sentiment",""))}">{r.get("sentiment","")}</td>'
            f'<td class="{sm_cls(r.get("smartmoney",""))}">{r.get("smartmoney","")}</td>'
            f'<td class="{rsi_cls}">{rsi_txt}</td>'
            f'<td>{adx_cell(adx_val, adx_thr)}</td>'
            f'<td>{_fmt_pct_smart(r.get("ret_p25_next5d", np.nan), clip=25)}</td>'
            f'<td>{_fmt_pct_smart(r.get("ret_p75_next5d", np.nan), clip=25)}</td>'
            f'<td>{pill_conf_bucket(r.get("confidence_bucket",""))}</td>'
            f'<td>{trust_merged_cell(r.get("model_quality",""), r.get("trust_score", np.nan))}</td>'
            f'<td>{fmt_int(r.get("median_bars_until_touch_ema20", np.nan))}</td>'
            f'<td>{fmt_int(r.get("final_confidence_0_100", np.nan))}</td>'
            f'<td>{_fmt_pct_1d(r.get("false_break_depth20_pct", np.nan))}</td>'
            f'<td>{_reason_chips_inline(r.get("reason_long",""))}</td>'
            f'<td>{_reason_chips_inline(r.get("reason_short",""))}</td>'
            f'</tr>'
        )

    tail = "</tbody></table></div></div></div>"

    defs = """
<section class="defs" style="margin:14px 10px;color:#94a3b8">
  <h2 style="font-size:16px;margin:10px 0;color:#cbd5e1">Glossary & Field Guide</h2>
  <div class="grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px">

    <!-- Trend structure -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Trend Phase, Trend & Age</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Trend Phase</b>: Daily <i>regime</i> using Close vs EMA20 &amp; EMA44:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>BULLISH</b>: EMA20 &gt; EMA44 and Close ≥ EMA20.</li>
            <li><b>BEARISH</b>: EMA20 &lt; EMA44 and Close ≤ EMA44.</li>
            <li><b>NEUTRAL</b>: Everything in between (choppy / transition).</li>
          </ul>
        </li>
        <li><b>Trend</b>: Short-term direction from Close vs EMA20:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>UP</b>: Close ≥ EMA20.</li>
            <li><b>DOWN</b>: Close &lt; EMA20.</li>
          </ul>
        </li>
        <li><b>Age</b>: Bars since current daily trend started, adjusted for Gate-1
            confirm days (longs only start counting after <code>confirm_days_long</code>).</li>
      </ul>
    </div>

    <!-- Trend stage -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Trend Stage (UP only)</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Emerging 🔥</b>:
          very early up-trend or fresh EMA20 reclaim with:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Low trend age or recent cross above EMA20,</li>
            <li>Strength-20 modest (&gt;= 0, not stretched),</li>
            <li>Strength-44 shallow,</li>
            <li>RSI in 45–65 zone,</li>
            <li>Not in upper stretch regime.</li>
          </ul>
        </li>
        <li><b>Pullback-20 🔁</b>:
          controlled pullback toward EMA20:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Strength-20 slightly negative to ~flat,</li>
            <li>RSI ~48–62,</li>
            <li>Smart Money ideally Accumulation / Neutral.</li>
          </ul>
        </li>
        <li><b>Pullback-44 ⚓</b>:
          deeper pullback toward EMA44, up-trend still intact.</li>
        <li><b>Steady ✅</b>:
          established up-trend:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Strength-20 positive but not extreme,</li>
            <li>Strength-44 moderate,</li>
            <li>RSI 45–68,</li>
            <li>Not in stretch regime.</li>
          </ul>
        </li>
        <li><b>Stretched ⚠️</b>:
          any of:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Strength-20 &gt; 5%,</li>
            <li>Strength-44 &gt; 12%,</li>
            <li>RSI &gt; 68,</li>
            <li>Regime = above_p80.</li>
          </ul>
        </li>
        <li><b>Near-20 🧪</b>:
          Close close to EMA20 from below, RSI ~40–48: “watch / test” area.</li>
        <li><b>—</b>:
          not in a clean UP trend or no pattern match.</li>
      </ul>
    </div>

    <!-- Strength / RSI / ADX -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Strength, RSI &amp; ADX</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Strength-20</b>:
          % distance Close vs EMA20.
          0% = sitting on EMA20; positive = above; negative = below.</li>
        <li><b>Strength-44</b>:
          medium-term extension vs EMA44 (how far from the “backbone” trend).</li>
        <li><b>RSI (live)</b>:
          RSI(14) recomputed using daily history + latest intraday (30m) close;
          used for stage logic and for display.</li>
        <li><b>ADX</b>:
          ADX(14). Color threshold:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>Strong (green)</b>: ADX ≥ max(adx_min_long, adx_min_short).</li>
            <li><b>Weak (red)</b>: ADX below that level.</li>
          </ul>
        </li>
      </ul>
    </div>

    <!-- Sentiment / Smart Money -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Sentiment &amp; Smart Money</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Sentiment (Risk ON/OFF)</b> combines:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Close vs EMA20,</li>
            <li>Stretch regime (P80 band),</li>
            <li>Smart Money label,</li>
            <li>ADX strength.</li>
          </ul>
          Any major failure → <b>Risk OFF</b>. Only clean alignment → <b>Risk ON</b>.
        </li>
        <li><b>Smart Money (Wyckoff-style)</b>:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>Accumulation</b>:
              flat price + falling volume (quiet buying).</li>
            <li><b>Distribution</b>:
              flat price + rising volume in weak context (quiet selling).</li>
            <li><b>Markup</b>:
              price trending up above EMA20.</li>
            <li><b>Markdown</b>:
              price trending down below EMA20.</li>
            <li><b>Neutral</b>:
              no clear read from last 2 bars.</li>
          </ul>
        </li>
      </ul>
    </div>

    <!-- Gate-1.5 stats -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Gate-1.5: Forward Envelope &amp; Touch Stats</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>P25@5D</b>:
          25th percentile 5-day forward return (conservative / downside-tilted).</li>
        <li><b>P75@5D</b>:
          75th percentile 5-day forward return (optimistic but realistic upside).</li>
        <li><b>Confidence Bucket</b>:
          HIGH / MED / LOW, based on calibration quality of these forward stats.</li>
        <li><b>MED touch20</b>:
          median number of bars until EMA20 is touched after a setup, used as
          rough timing for mean-reversion / pullbacks.</li>
      </ul>
    </div>

    <!-- Trust & Final Confidence -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Trust Overlay &amp; Final Confidence</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Trust pill</b> (High / Medium / Low + %):
          per-ticker model quality from Gate-2 (s83):
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>High</b>:
              stable stats &amp; model behaviour.</li>
            <li><b>Medium</b>:
              usable, some noise / uncertainty.</li>
            <li><b>Low</b>:
              thin or unstable history; size carefully.</li>
          </ul>
          The % inside the pill is <b>trust_score</b> (0–100%).
        </li>
        <li><b>Confidence (0–100)</b>:
          base Gate-2 confidence for this ticker &amp; setup. If missing from
          Gate-1.5 CSVs, a snapshot fallback is used.</li>
        <li><b>Final Conf</b>:
          <code>CONF(0–100) × trust_score</code> (where trust_score is 0–1).  
          This is what the table uses to rank rows.</li>
      </ul>
    </div>

    <!-- Gate-1 rule tags: reasons -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Gate-1 Rule Tags (Reasons)</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>slope_long</b>:
          EMA slopes acceptable for a long setup:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>EMA5_slope ≥ its minimum,</li>
            <li>EMA20_slope ≥ its minimum,</li>
            <li>EMA_base_slope ≥ its minimum (if enabled).</li>
          </ul>
          EMAs are tilting up enough → constructive momentum.
        </li>
        <li><b>slope_long_fail</b>:
          at least one required EMA slope too flat or down → trend lacks
          upward momentum, blocks long entries.</li>

        <li><b>trend_align_long</b>:
          full alignment for longs:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li>Close ≥ EMA20,</li>
            <li>EMA20 ≥ EMA44,</li>
            <li>EMA44 ≥ EMA_base (if rule active).</li>
          </ul>
          Clean bullish stack of moving averages.
        </li>
        <li><b>trend_align_long_fail</b>:
          moving averages misaligned (e.g. EMA20 &lt; EMA44, or price &lt; EMA20),
          structure not strong enough.</li>

        <li><b>trend_gap_small</b>:
          spread between EMAs (e.g. EMA20–EMA44) small and healthy → compressed,
          constructive trend structure.</li>
        <li><b>trend_gap_large</b>:
          EMAs very far apart → late / overextended trend, often combined with
          a Stretched stage.</li>

        <li><b>rsi_cap</b>:
          RSI above the configured cap for longs (e.g. RSI &gt; 65) → too hot
          to initiate new positions.</li>

        <li><b>volatility_cap</b>:
          ATR too high relative to price and rules → volatility risk limit hit.</li>

        <li><b>adx_strong</b>:
          ADX above threshold → trend strong enough to support directional trades.</li>
        <li><b>adx_weak</b>:
          ADX below threshold → trend too weak / noisy; may block longs or shorts.</li>

        <li><b>donchian_breakout_required</b>:
          rules demand a breakout above the Donchian upper band, but price has
          not broken out yet → entry delayed.</li>
        <li><b>donchian_pos_block</b>:
          Donchian position too high for fresh longs (e.g. near top of range);
          rule avoids chasing.</li>
        <li><b>donchian_low_pos_block</b>:
          Donchian position too low for shorts; rule avoids shorting at the bottom.</li>

        <li><b>ignore_gap</b>:
          daily gap exceeds allowed percentage (e.g. &gt; 2%) → setup skipped due
          to gap risk.</li>

        <li><b>strength20_extreme</b>:
          Strength-20 outside allowed band for entries (too extended or too deep).</li>
        <li><b>strength44_extreme</b>:
          Strength-44 indicates overextension of the medium-term trend.</li>

        <li><b>regime_p80_block</b>:
          stretch regime flagged as <code>above_p80</code>; fresh longs disabled
          by risk rules.</li>
      </ul>
    </div>

    <!-- False breaks / trade chips / market status -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">False Breaks, Trade Chips &amp; Market Status</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>FB↓20</b>:
          max % depth <i>below</i> EMA20 during the last undercut sequence before
          the most recent reclaim (30-bar window, capped at 20%).
          Good reference for “where did it shake people out last time?”.</li>
        <li><b>TRADE chip</b>:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>CONSIDER</b> (green) → BUY decision from Gate-1.</li>
            <li><b>SELL</b> (red) → SELL decision.</li>
            <li><b>DO NOTHING</b> (grey) → at least one key rule blocked the setup.</li>
          </ul>
        </li>
        <li><b>Reasons (Long / Short)</b>:
          list of rule tags summarising why a trade is allowed or blocked
          (slope, trend alignment, stretch, volatility, Donchian, etc.).</li>
        <li><b>Market Status</b> (traffic light at the top):
          taken from <code>market_status.json</code>:
          <ul style="margin:4px 0 0 14px;padding:0">
            <li><b>Green</b>: favourable regime, DRI supportive.</li>
            <li><b>Orange</b>: mixed / transitional regime.</li>
            <li><b>Red</b>: stressed / risk-off market environment.</li>
          </ul>
          DRI (%) is an additional breadth / risk indicator backing the light.
        </li>
      </ul>
    </div>

  </div>
</section>
"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
        + style + "</head><body>"
        + head
        + "\n".join(rows_html)
        + tail
        + defs
        + script
        + "</body></html>"
    )

# ----------------- main -----------------
def _load_universe_from_xlsx(path: Path, sheet: str = "signalsNY",
                             preferred_cols=("ticker","symbol","etf","Ticker","Symbol")) -> set[str]:
    try:
        if not path.exists():
            print(f"[WARN] Universe file missing: {path}")
            return set()
        df = pd.read_excel(path, sheet_name=sheet)  # needs openpyxl
        if df is None or df.empty:
            print(f"[WARN] Universe sheet empty: {path.name}:{sheet}")
            return set()
        cols_map = {c.strip().lower(): c for c in df.columns}
        col = next((cols_map[c.lower()] for c in preferred_cols if c.lower() in cols_map), df.columns[0])
        tickers = df[col].astype(str).str.strip().str.upper()
        return {t for t in tickers if t and t not in {"NAN","NONE"}}
    except Exception as e:
        print(f"[WARN] Could not load universe from {path}:{sheet} — {e}")
        return set()

def _serve_local(signals_dir: Path, index_file: str, host: str = "127.0.0.1", port: int = 8080):
    import http.server, socketserver, webbrowser
    os.chdir(str(signals_dir))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/{index_file}"
        print(f"[SERVE] Local server at {url}")
        try: webbrowser.open(url)
        except Exception: pass
        try: httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVE] Stopped by user.")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="s81 v1.6 slim — curated columns + FB↓20 + Trend & Stage filters.")
    ap.add_argument("--label", type=str, default="gate1_v1.0", help="Which s77 label to render.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (debug).")
    ap.add_argument("--signals-dir", type=str, default=str(P.SIGNALS_DIR), help="Signals directory (s77 CSV + metadata).")
    ap.add_argument("--csv", type=str, default="", help="Optional explicit path to rule_live_signals_*.csv to render.")
    ap.add_argument("--universe-sheet", type=str, default="signalsNY", help="Excel sheet in ETF_list.xlsx to filter universe.")
    ap.add_argument("--universe-xlsx", type=str, default="", help="Path to ETF_list.xlsx (override)")
    ap.add_argument("--serve", action="store_true", help="Run a local HTTP server to view the dashboard.")
    ap.add_argument("--port", type=int, default=8080, help="Port for the local server.")
    ap.add_argument("--market-status",
                default=os.environ.get("MARKET_STATUS_JSON",
                                       os.path.join(os.environ.get("SIGNALS_SHARED_DIR", "signals"),
                                                    "market_status.json")),
                help="Path to market_status.json produced by s84.")
    args = ap.parse_args()

    # Resolve universe file + sheet (NEW)
    universe_xlsx = (
        args.universe_xlsx
        or os.environ.get("ETF_LIST_XLSX", "")
        or str(getattr(P, "ETF_LIST", P.CONFIG_DIR / "ETF_list.xlsx"))
    )
    universe_xlsx = Path(universe_xlsx)
    universe_sheet = (args.universe_sheet
                    or os.environ.get("ETF_SHEET")
                    or "signals")
    print(f"[INFO] Universe source: {universe_xlsx} (sheet={universe_sheet})")

    label = args.label.strip()
    signals_dir = Path(args.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    trust_json_path   = P.CONFIG_DIR / "gate2_ticker_quality.json"
    snapshot_csv_path = P.DATA_ENRICHED / "gate2_confidence_snapshot.csv"

    snap_df = _load_gate2_conf_snapshot(snapshot_csv_path)

    # default for safety (used if metadata missing)
    confirm_days_long = 1

    # pick s77 snapshot
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise SystemExit(f"[ERR] --csv path not found: {csv_path}")
        stem = csv_path.stem
        ts = "_".join(stem.split("_")[-2:]) if len(stem.split("_")) >= 3 else ""
        meta_path = csv_path.with_name(f"rule_live_signals_{ts}.metadata.json") if ts else None
        if meta_path and meta_path.exists():
            meta_obj = json.loads(meta_path.read_text())
            adx_thr = float(max(meta_obj.get("adx_min_long", 20.0), meta_obj.get("adx_min_short", 20.0)))
            confirm_days_long = int(meta_obj.get("confirm_days_long", 1))
        else:
            print("[WARN] No metadata found next to the provided --csv; using ADX threshold=20 and confirm_days_long=1.")
            adx_thr = 20.0
    else:
        csv_path, meta_path, _ = _find_latest_by_label(signals_dir, label=label)
        meta_obj = json.loads(meta_path.read_text())
        adx_thr = float(max(meta_obj.get("adx_min_long", 20.0), meta_obj.get("adx_min_short", 20.0)))
        confirm_days_long = int(meta_obj.get("confirm_days_long", 1))

    print(f"[INFO] Using signals CSV: {csv_path.name}")
    if 'meta_path' in locals() and meta_path and meta_path.exists():
        print(f"[INFO] Metadata file     : {meta_path.name}")

    # load s77
    df_sig = pd.read_csv(csv_path)
    need = {"ticker", "decision"}
    if not need.issubset(set(df_sig.columns)):
        raise SystemExit(f"[ERR] {csv_path.name} missing required columns {sorted(need)}.")
    df_sig["ticker"] = df_sig["ticker"].astype(str).str.strip().str.upper()
    df_sig["decision"] = df_sig["decision"].astype(str).str.strip().str.upper()

    # universe filter
    universe = _load_universe_from_xlsx(universe_xlsx, sheet=universe_sheet)
    if universe:
        df_sig = df_sig[df_sig["ticker"].isin(universe)]
    else:
        print("[WARN] Universe is empty; dashboard will include all rows (fallback).")

    if args.limit and args.limit > 0:
        keep = df_sig["ticker"].unique().tolist()[: args.limit]
        df_sig = df_sig[df_sig["ticker"].isin(keep)]

    # Gate-1.5 merge (safe)
    g15 = _load_gate15_all()
    if g15 is not None and not g15.empty:
        forbid = set(df_sig.columns)
        keep_cols = [c for c in g15.columns if (c == "ticker") or (c not in forbid)]
        g15 = g15[keep_cols]
        df = df_sig.merge(g15, on="ticker", how="left")
    else:
        print("[WARN] Gate-1.5 analytics not found — rendering without enriched fields.")
        df = df_sig.copy()

    # snapshot fallback + coalesce confidence/bucket
    def _colidx(df_, prefix: str) -> list[int]:
        idx = []
        for i, c in enumerate(df_.columns):
            if isinstance(c, str) and (c == prefix or c.startswith(prefix + "_")):
                idx.append(i)
        return idx

    conf_idxs = _colidx(df, "confidence_score_0_100")
    if conf_idxs:
        conf_stack = pd.concat([pd.to_numeric(df.iloc[:, i], errors="coerce") for i in conf_idxs], axis=1)
        all_nan = conf_stack.isna().all().all()
    else:
        all_nan = True

    if all_nan and snap_df is not None and not snap_df.empty:
        df = df.merge(snap_df, on="ticker", how="left")
        print("[INFO] Using Gate-2 snapshot for CONF/Bucket fallback.")
        conf_idxs = _colidx(df, "confidence_score_0_100")

    if conf_idxs:
        conf_stack = pd.concat([pd.to_numeric(df.iloc[:, i], errors="coerce") for i in conf_idxs], axis=1)
        df["confidence_score_0_100"] = conf_stack.bfill(axis=1).iloc[:, 0]
        # drop dups except the canonical
        conf_names = [df.columns[i] for i in conf_idxs]
        for c in conf_names:
            if c != "confidence_score_0_100":
                df.drop(columns=c, inplace=True, errors="ignore")

    bucket_idxs = _colidx(df, "confidence_bucket")
    if bucket_idxs:
        buck_stack = pd.concat([df.iloc[:, i].astype(str) for i in bucket_idxs], axis=1)
        df["confidence_bucket"] = buck_stack.bfill(axis=1).iloc[:, 0]
        bucket_names = [df.columns[i] for i in bucket_idxs]
        for c in bucket_names:
            if c != "confidence_bucket":
                df.drop(columns=c, inplace=True, errors="ignore")

    # ---- daily enriched context (from s00) ----
    daily_map = _load_daily_enriched_master(P.DATA_ENRICHED / "prices_enriched.parquet") or {}
    intraday_rsi = _load_intraday_rsi_map(P.DATA_ENRICHED / "30min")
    daily_hist = _load_daily_close_history(P.DATA_ENRICHED / "prices_enriched.parquet")
    last30_map = _load_intraday_last_close(P.DATA_ENRICHED / "30min")


    # ---- s83 model trust ----
    trust_map = _load_gate2_quality_json(trust_json_path)

    # ---- rows ----
    out_rows = []
    for _, row in df.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision", "")).strip().upper()

        dctx = daily_map.get(t, {})
        close_t = dctx.get("close", np.nan)
        close_y = dctx.get("close_y", np.nan)
        ema20_t = dctx.get("ema20_d", np.nan)
        ema44_t = dctx.get("ema44_d", np.nan)
        rsi_daily = row.get("rsi14_d", dctx.get("rsi14_d", np.nan))   # used for stage/logic
        rsi_live  = intraday_rsi.get(t, np.nan)                       # used only for display
        adx14_d = row.get("adx14_d", dctx.get("adx14_d", np.nan))
        vol_t   = dctx.get("volume", np.nan)
        vol_y   = dctx.get("volume_y", np.nan)

        phase = _trend_phase(close_t, ema20_t, ema44_t)
        sm    = _smart_money_wyckoff(close_t, close_y, vol_t, vol_y, ema20_t)
        trn   = _trend_short(close_t, ema20_t)
        sent  = _sentiment_risk(close_t, ema20_t, str(row.get("stretch_regime", "")), sm, adx14_d, adx_thr)

        tinfo = trust_map.get(t, {})
        trust_score = float(pd.to_numeric(tinfo.get("trust_score", np.nan), errors="coerce"))
        model_quality = str(tinfo.get("quality", "")).title() if tinfo else ""

        # baseline = yesterday's daily RSI from parquet
        rsi_daily_y = row.get("rsi14_d", dctx.get("rsi14_d", np.nan))

        # compute live daily RSI = daily history + today's provisional close (latest 30m)
        rsi_daily_live = rsi_daily_y
        s_hist = daily_hist.get(t)
        last30 = last30_map.get(t, np.nan)
        if s_hist is not None and np.isfinite(last30):
            s_live = pd.concat([s_hist, pd.Series([last30])], ignore_index=True)
            rsi_daily_live = _wilder_rsi(s_live, 14)

        # CONF may be missing; final_conf only if both present
        # (use bfill-coalesced "confidence_score_0_100" if available)
        conf_raw = row.get("confidence_score_0_100", np.nan)
        try:
            conf_0_100 = float(pd.to_numeric(conf_raw, errors="coerce"))
        except Exception:
            conf_0_100 = np.nan

        final_conf_0_100 = (
            int(round(conf_0_100 * trust_score))
            if np.isfinite(conf_0_100) and np.isfinite(trust_score) else np.nan
        )

        # --- xATR vs EMA20/EMA44 (signed; + = above EMA, - = below EMA)
        def _xatr(close_v, ema_v, atr_v):
            if np.isfinite(close_v) and np.isfinite(ema_v) and np.isfinite(atr_v) and atr_v > 0:
                return float(np.clip((close_v - ema_v) / atr_v, -10.0, 10.0))
            return np.nan

        # prefer precomputed from s77; else compute from enriched (dctx)
        xatr20 = pd.to_numeric(row.get("xatr_vs_ema20", np.nan), errors="coerce")
        xatr44 = pd.to_numeric(row.get("xatr_vs_ema44", np.nan), errors="coerce")

        # --- make it live using the latest 30-min close if available ---
        if not np.isfinite(xatr20) or not np.isfinite(xatr44):
            live_close = last30_map.get(t, np.nan)
            if np.isfinite(live_close):
                xatr20_live = _xatr(live_close, ema20_t, dctx.get("atr14_d", np.nan))
                xatr44_live = _xatr(live_close, ema44_t, dctx.get("atr14_d", np.nan))
                xatr20 = xatr20_live
                xatr44 = xatr44_live
            else:
                # fallback to yesterday’s close
                if not np.isfinite(xatr20):
                    xatr20 = _xatr(close_t, ema20_t, dctx.get("atr14_d", np.nan))
                if not np.isfinite(xatr44):
                    xatr44 = _xatr(close_t, ema44_t, dctx.get("atr14_d", np.nan))


        # raw age from s00: days since EMA20 trend start
        raw_age = dctx.get("trend_days_since_start_d", np.nan)
        try:
            raw_age_f = float(raw_age)
        except Exception:
            raw_age_f = np.nan

        # Age for entries: start counting only after confirmation
        # e.g. confirm_days_long=2 → first entry day gets Age=1
        if np.isfinite(raw_age_f):
            trend_age_for_entry = max(int(raw_age_f) - (confirm_days_long - 1), 0)
        else:
            trend_age_for_entry = np.nan

        row_dict = {
            "ticker": t,
            "decision": decision,
            "trend_phase": phase,
            "trend": trn,
            "trend_age": trend_age_for_entry,

            "strength20_pct": row.get("strength20_pct", np.nan),
            "strength44_pct": row.get("strength44_pct", np.nan),

            "xatr_vs_ema20": xatr20,
            "xatr_vs_ema44": xatr44,

            "sentiment": sent,
            "smartmoney": sm,
            "rsi14_d": rsi_daily_live,     # use live-daily for logic + display
            "rsi14_y": rsi_daily_y,        # optional: yesterday’s RSI
            "adx14_d": adx14_d,

            "confidence_bucket":      row.get("confidence_bucket", ""),
            "ret_p25_next5d":         row.get("ret_p25_next5d", np.nan),
            "ret_p75_next5d":         row.get("ret_p75_next5d", np.nan),
            "median_bars_until_touch_ema20": row.get("median_bars_until_touch_ema20", np.nan),

            "reason_long":  (row.get("reason_long",  "") if decision == "DO NOTHING" else ""),
            "reason_short": (row.get("reason_short", "") if decision == "DO NOTHING" else ""),
            "recent_cross_above_ema20_3d": bool(dctx.get("recent_cross_above_ema20_3d", False)),

            "model_quality": model_quality,
            "trust_score": trust_score,
            "final_confidence_0_100": final_conf_0_100,

            "false_break_depth20_pct": dctx.get("false_break_depth20_pct", np.nan),
            "stretch_regime": row.get("stretch_regime",""),
        }

        stage_label, stage_rank = _trend_stage({**row, **row_dict})
        row_dict["trend_stage"] = stage_label
        row_dict["_stage_rank"] = stage_rank
        out_rows.append(row_dict)

    out = pd.DataFrame(out_rows)

    # ---- ordering: actionables first; then stage; then final conf; then strength20 ----
    out["__ord"]   = np.where(out["decision"].isin(["BUY","SELL"]), 0, 1)
    out["__stage"] = pd.to_numeric(out.get("_stage_rank", np.nan), errors="coerce").fillna(9)
    out["__fconf"] = pd.to_numeric(out.get("final_confidence_0_100", np.nan), errors="coerce").fillna(-1)
    out["__s20"]   = pd.to_numeric(out.get("strength20_pct", np.nan), errors="coerce").fillna(-1e9)

    out = (out.sort_values(["__ord","__stage","__fconf","__s20"],
                           ascending=[True, True, False, False])
             .drop(columns=["__ord","__stage","__fconf","__s20","_stage_rank"])
             .reset_index(drop=True))

    # ---- render HTML ----
    now_ts = _ts()
    updated_label = f"{now_ts[:4]}-{now_ts[4:6]}-{now_ts[6:8]} {now_ts[9:11]}:{now_ts[11:13]} UTC"
    html = build_html(out, updated_label, label=label, adx_thr=adx_thr)

    # ---- write files ----
    out_ts = signals_dir / f"signals_dashboard_{label}_{now_ts}.html"
    out_latest = signals_dir / f"1_signals_dashboard_latest_{label}.html"
    out_ts.write_text(html, encoding="utf-8")
    out_latest.write_text(html, encoding="utf-8")

    print(f"[OK] Dashboard (timestamped) → {out_ts}")
    print(f"[OK] Dashboard (latest)      → {out_latest}")
    print(f"[INFO] Source (s77 CSV)      → {csv_path.name}")
    if 'meta_path' in locals() and meta_path and meta_path.exists():
        print(f"[INFO] Meta   (s77)          → {meta_path.name}")
    else:
        print("[INFO] Meta   (s77)          → (none / provided via --csv)")
    print(f"[INFO] Daily enriched (s00)  → {P.DATA_ENRICHED / 'prices_enriched.parquet'}")
    print(f"[INFO] Trust JSON (s83)      → {trust_json_path}")

    if args.serve:
        _serve_local(signals_dir, index_file=out_latest.name, host="127.0.0.1", port=int(args.port))

if __name__ == "__main__":
    main()