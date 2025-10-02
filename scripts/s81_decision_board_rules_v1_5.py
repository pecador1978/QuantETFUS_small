#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s81_decision_board_rules_v1_6.py — Gate-1 (s77) decisions + Gate-1.5 analytics + Daily context (from s00)
+ Trend Stage classification (Emerging / Steady / Stretched)

What it shows
-------------
- SOURCE OF TRUTH (s77): decision, strength20_pct, strength44_pct, reason_* (never overridden)
- Gate-1.5 analytics (safe-merged): confidence score/bucket, stretch regime,
  P(up/down) next 1/3/5D, 1D CI bounds, 5D return envelope percentiles (P25/P75),
  median bars to EMA20/44, stretch p50/p80 vs EMA20/44, n_events
- Daily context (from s00 daily enriched master parquet):
  RSI14, ADX14, EMA20/44, derived Trend Phase / Trend (UP/DOWN), Sentiment, Trend AGE (bars)
- NEW: Trend Stage (Emerging / Steady / Stretched) + sort to surface Emerging first

Changes vs previous
-------------------
- Added Trend Stage column and logic (with recent EMA20 reclaim check).
- Fixed glossary and added REGIME / Stage explanations.
- Corrected column order and sorting (stage rank, then confidence, then strength).
- Implemented “recent_cross_above_ema20_3d” inside the daily loader.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os, json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402

pd.set_option("future.no_silent_downcasting", True)

# ----------------- helpers -----------------
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
    """Short-term trend from Close vs EMA20."""
    if np.isfinite(close_d) and np.isfinite(ema20_d):
        return "UP" if float(close_d) >= float(ema20_d) else "DOWN"
    return ""

def _sentiment_risk(close_d: float, ema20_d: float, regime: str,
                    smartmoney: str, adx_val: float, adx_thr: float = 20.0) -> str:
    """
    Risk ON if:
      - Close >= EMA20  (short-term trend up)
      - not 'above_p80' stretch
      - Smart Money not Distribution/Markdown
      - ADX >= threshold (if ADX present)
    Otherwise Risk OFF.
    """
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
    """
    Stage label + rank (lower rank sorts higher).
    Designed so 'Emerging' and active pullbacks bubble to the top.
    """
    tr  = str(row.get("trend","")).upper()
    age = float(row.get("trend_age", np.nan))
    s20 = float(row.get("strength20_pct", np.nan))  # % dist to EMA20
    s44 = float(row.get("strength44_pct", np.nan))  # % dist to EMA44
    rsi = float(row.get("rsi14_d", np.nan))
    reg = str(row.get("stretch_regime",""))
    sm  = str(row.get("smartmoney","")).lower()
    rc  = bool(row.get("recent_cross_above_ema20_3d", False))

    if tr != "UP" or not np.isfinite(s20) or not np.isfinite(rsi):
        return ("—", 9)

    # --- 1) Emerging right after turn or reclaim of 20 ---
    if ((np.isfinite(age) and age <= 5) or rc) \
       and (0.0 <= s20 <= 2.5) and (np.isfinite(s44) and s44 <= 6.0) \
       and (45 <= rsi <= 65) and reg != "above_p80":
        return ("Emerging 🔥", 0)

    # --- 2) Pullback to EMA20 (retest, actionable watch) ---
    if (-1.8 <= s20 <= 0.8) and (48 <= rsi <= 62) and reg != "above_p80":
        if sm in ("accumulation","neutral"):  # soft filter
            return ("Pullback-20 🔁", 1)
        return ("Pullback-20 🔁", 2)

    # --- 3) Pullback toward EMA44 (deeper, still intact) ---
    if (s20 <= -0.5) and np.isfinite(s44) and (0.0 <= s44 <= 3.5) and (45 <= rsi <= 58):
        if sm in ("accumulation","neutral"):
            return ("Pullback-44 ⚓", 2)
        return ("Pullback-44 ⚓", 3)

    # --- 4) Steady continuation ---
    if (0.25 <= s20 <= 5.0) and (np.isfinite(s44) and s44 < 12.0) and (45 <= rsi <= 68) and reg != "above_p80":
        return ("Steady ✅", 4)

    # --- 5) Stretched / late ---
    if (s20 > 5.0) or (np.isfinite(s44) and s44 > 12.0) or (rsi > 68) or (reg == "above_p80"):
        return ("Stretched ⚠️", 6)

    # --- 6) Optional: hovering near 20 but weaker RSI ---
    if (-2.5 <= s20 <= 1.0) and (40 <= rsi < 48):
        return ("Near-20 🧪", 5)

    return ("—", 9)

# -------- daily enriched loader (from s00) --------
def _load_daily_enriched_master(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load latest daily row per ticker from data_enriched/prices_enriched.parquet
    Returns: dict[ticker] = {... selected fields ...} + recent_cross_above_ema20_3d
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        print(f"[WARN] Daily enriched parquet not found: {path}")
        return out

    df = pd.read_parquet(path)
    if df.empty:
        print("[WARN] Daily enriched parquet is empty.")
        return out

    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["datetime"] = dt
    df = df.dropna(subset=["datetime", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df.sort_values(["ticker", "datetime"])

    keep = ["close","volume","ema20_d","ema44_d","rsi14_d","adx14_d","trend_days_since_start_d"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan

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
        }

    # second-to-last for Wyckoff deltas
    prev = df.groupby("ticker", as_index=False).tail(2).reset_index(drop=True)
    prev = prev.groupby("ticker").nth(-2).reset_index()
    prev["ticker"] = prev["ticker"].astype(str).str.strip().str.upper()
    for _, r in prev.iterrows():
        t = str(r["ticker"]).upper()
        if t in out:
            out[t]["close_y"] = float(r.get("close", np.nan)) if np.isfinite(r.get("close", np.nan)) else np.nan
            out[t]["volume_y"] = float(r.get("volume", np.nan)) if np.isfinite(r.get("volume", np.nan)) else np.nan

    # --- 3-bar reclaim of EMA20 ---
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

    return out

def _reason_chips(reason: str) -> str:
    if not reason: return ""
    tags = []
    for r in [x.strip() for x in str(reason).split(",") if x.strip()]:
        cls = "tag tag-warn" if r == "stretched_vs_ema44" else "tag"
        tags.append(f'<span class="{cls}">{r}</span>')
    return '<div class="reason">' + "".join(tags) + "</div>"

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
        "p_up_next1d":   ["p_up_next1d","p_up_1d","p_up1d"],
        "p_down_next1d": ["p_down_next1d","p_down_1d","p_down1d","p_dn_next1d","p_dn_1d","p_dn1d"],
        "p_ci_low_1d":   ["p_ci_low_1d","ci_low_1d","ci1d_low"],
        "p_ci_high_1d":  ["p_ci_high_1d","ci_high_1d","ci1d_high"],
        "p_up_next3d":   ["p_up_next3d","p_up_3d","p_up3d"],
        "p_down_next3d": ["p_down_next3d","p_down_3d","p_down3d","p_dn_next3d","p_dn_3d","p_dn3d"],
        "p_up_next5d":   ["p_up_next5d","p_up_5d","p_up5d"],
        "p_down_next5d": ["p_down_next5d","p_down_5d","p_down5d","p_dn_next5d","p_dn_5d","p_dn5d"],
        "ret_p25_next5d": ["ret_p25_next5d","p25_ret_5d","ret25_5d"],
        "ret_p75_next5d": ["ret_p75_next5d","p75_ret_5d","ret75_5d"],
        "stretch_p50_vs_ema20": ["stretch_p50_vs_ema20","p50_vs_ema20","stretch_p50_ema20"],
        "stretch_p80_vs_ema20": ["stretch_p80_vs_ema20","p80_vs_ema20","stretch_p80_ema20"],
        "stretch_p50_vs_ema44": ["stretch_p50_vs_ema44","p50_vs_ema44","stretch_p50_ema44"],
        "stretch_p80_vs_ema44": ["stretch_p80_vs_ema44","p80_vs_ema44","stretch_p80_ema44"],
        "confidence_score_0_100": ["confidence_score_0_100","confidence_score","conf_score"],
        "confidence_bucket":      ["confidence_bucket","conf_bucket","bucket"],
        "stretch_regime":         ["stretch_regime","regime"],
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
    base_dir = P.ROOT / "signals" / "analytics"
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
    return merged

# ----------------- HTML helpers -----------------
def _fmt_pct_int(x) -> str:
    try: v = float(x)
    except Exception: return "—"
    if not np.isfinite(v): return "—"
    if 0.0 <= v <= 1.0: v *= 100.0
    return f"{int(round(v))}"

def _fmt_pct_1d(x) -> str:
    try: v = float(x)
    except Exception: return "—"
    if not np.isfinite(v): return "—"
    if abs(v) <= 1.0: v *= 100.0
    return f"{v:.1f}"

# ----------------- HTML builder -----------------
def build_html(df: pd.DataFrame, updated_label: str, label: str, adx_thr: float) -> str:
    style = """
<style>
:root{
  --bg:#0b1020; --card:#0f172a; --muted:#94a3b8; --txt:#e2e8f0; --line:#1f2937;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial;color:#e2e8f0}
.wrap{max-width:100%;margin:10px;padding:0}
h1{margin:0 0 8px;font-size:22px}
.info{color:var(--muted);font-size:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.search{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);
  background:#0b132d;color:#e2e8f0;outline:none;margin-bottom:8px}
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

.phase-bullish { color:#86efac; font-weight:700 }  
.phase-bearish { color:#f87171; font-weight:700 }
.phase-neutral { color:#cbd5e1; }

.rsi-hot { color:#ef4444; font-weight:700 }
.rsi-cold { color:#3b82f6; font-weight:700 }
.adx-strong { color:#86efac; font-weight:700 }
.adx-weak   { color:#f87171; font-weight:700 }
.adx-neutral{ color:#cbd5e1; }

.trend-up{color:#86efac}
.trend-down{color:#fca5a5}
.sent-on{color:#86efac;font-weight:700}
.sent-off{color:#fca5a5;font-weight:700}
.sm-acc{color:#60a5fa}
.sm-dist{color:#f59e0b}
.sm-markup{color:#86efac;font-weight:700}
.sm-markdown{color:#ef4444;font-weight:700}

.strg{display:inline-block;padding:2px 8px;border-radius:8px;font-weight:800}
.strg-neutral{background:#0f172a;color:#cbd5e1}
.strg-ok{background:#052e1a;color:#86efac}
.strg-warn{background:#3b2a0b;color:#fbbf24}
.strg-hot{background:#3b0a0a;color:#fca5a5}

.reason{display:flex;gap:6px;flex-wrap:nowrap;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.tag{padding:2px 6px;border-radius:6px;font-size:11px;border:1px solid #2b344a;color:#cbd5e1;background:#101a2f}
.tag-warn{border-color:#8a5500;color:#fbbf24;background:#2a1c05}
/* Gate-1.5 pills */
.pill{
  display:inline-block;
  padding:2px 8px;
  border-radius:8px;
  border:1px solid #2b344a;
  font-size:11px;
}
.pill-high{ /* good / safe */
  background:rgba(22,163,74,.12);
  color:#86efac;
  border-color:rgba(22,163,74,.30);
  font-weight:700;
}
.pill-med{  /* neutral / watch */
  background:rgba(245,158,11,.12);
  color:#fbbf24;
  border-color:rgba(245,158,11,.35);
  font-weight:700;
}
.pill-low{  /* risky */
  background:rgba(220,38,38,.12);
  color:#fca5a5;
  border-color:rgba(220,38,38,.35);
  font-weight:700;
}

/* sticky first column (Ticker) */
#tbl { border-collapse: separate; border-spacing: 0; }
#tbl th:first-child, #tbl td:first-child{ position: sticky; left: 0; background: var(--card); }
#tbl th:first-child{ z-index: 6; }
#tbl td:first-child{ z-index: 1; border-right: 1px solid var(--line); }
</style>
"""

    # --- JS: sorting + Stage filter + search ---
    script = """
<script>
(function(){
  // ---- column indices (0-based) ----
  const COL = {
    TICKER:0, AGE:1, TRADE:2,
    PHASE:3, TREND:4, STAGE:5,
    STR20:6, STR44:7, SENT:8, SM:9, RSI:10, ADX:11,
    P25_5:12, P75_5:13, CONF:14, BUCKET:15, REGIME:16,
    PUP1:17, PDN1:18, CIL1:19, CIH1:20,
    PUP3:21, PDN3:22, PUP5:23, PDN5:24,
    MED20:25, MED44:26, S50_20:27, S80_20:28, S50_44:29, S80_44:30,
    N:31
  };

  const numIdx = new Set([
    COL.AGE, COL.STR20, COL.STR44, COL.RSI, COL.ADX,
    COL.P25_5, COL.P75_5, COL.CONF,
    COL.PUP1, COL.PDN1, COL.CIL1, COL.CIH1,
    COL.PUP3, COL.PDN3, COL.PUP5, COL.PDN5,
    COL.MED20, COL.MED44, COL.S50_20, COL.S80_20, COL.S50_44, COL.S80_44, COL.N
  ]);

  const customOrder = new Map([
    [COL.BUCKET, ["HIGH","MED","LOW"]]
  ]);

  function getCellText(td){ return (td && td.textContent ? td.textContent : '').trim(); }
  function parseNum(s){
    const pct=s.replace(/[,%]/g,'');
    const num=pct.replace(/[^0-9.+-]/g,'');
    const v=parseFloat(num);
    return isNaN(v) ? Number.NEGATIVE_INFINITY : v;
  }

  // ----- sorting -----
  function sortTable(table, colIndex, asc){
    const tbody=table.tBodies[0];
    const rows=Array.from(tbody.querySelectorAll('tr')).map((r,i)=>({r,i}));
    const order = customOrder.get(colIndex);

    rows.sort((A,B)=>{
      const a=getCellText(A.r.cells[colIndex]);
      const b=getCellText(B.r.cells[colIndex]);

      if(order){
        const ra = order.indexOf(a.toUpperCase());
        const rb = order.indexOf(b.toUpperCase());
        const va = (ra === -1) ? Number.POSITIVE_INFINITY : ra;
        const vb = (rb === -1) ? Number.POSITIVE_INFINITY : rb;
        let cmp = va - vb;
        if(!asc) cmp = -cmp;
        return cmp !== 0 ? cmp : (A.i - B.i);
      }

      let cmp;
      if(numIdx.has(colIndex)){
        cmp = parseNum(a) - parseNum(b);
      }else{
        cmp = a.toLowerCase().localeCompare(b.toLowerCase());
      }
      if(!asc) cmp = -cmp;
      return cmp !== 0 ? cmp : (A.i - B.i);
    });

    rows.forEach(o=>tbody.appendChild(o.r));
  }

  function clearIndicators(ths){
    ths.forEach(th=>{
      th.removeAttribute('data-asc');
      th.setAttribute('aria-sort','none');
      const s=th.querySelector('.sort-ind'); if(s) s.textContent='↕';
    });
  }

  // ----- filters -----
  function stageKey(txt){
    txt = (txt||'').trim();
    if(!txt || txt==='—') return '__EMPTY__';
    return txt;
  }

  function applyFilters(){
    const table=document.getElementById('tbl');
    const tbody=table.tBodies[0];
    const q = (document.getElementById('q')?.value || '').trim().toLowerCase();

    const chosen = new Set(
      Array.from(document.querySelectorAll('input[name="stage-filter"]:checked'))
           .map(i=>i.value)
    );

    Array.from(tbody.rows).forEach(tr=>{
      const txt = (tr.getAttribute('data-f')||'').toLowerCase();
      const stageOk = chosen.size===0 || chosen.has(stageKey(getCellText(tr.cells[COL.STAGE])));
      const searchOk = !q || txt.includes(q);
      tr.style.display = (stageOk && searchOk) ? '' : 'none';
    });
  }
  window.filterRows = applyFilters; // keep compatibility with oninput="" in the search box

  function populateStageChips(){
    const table=document.getElementById('tbl');
    const tbody=table.tBodies[0];
    const set=new Set();
    Array.from(tbody.rows).forEach(tr=>{
      set.add(stageKey(getCellText(tr.cells[COL.STAGE])));
    });

    const cont=document.getElementById('stage-chips');
    cont.innerHTML='';
    Array.from(set).sort((a,b)=>a.localeCompare(b)).forEach(k=>{
      const label = (k==='__EMPTY__') ? '(empty)' : k;
      const lab=document.createElement('label');
      lab.className='filter-chip';
      lab.innerHTML = '<input type="checkbox" name="stage-filter" value="'+k+'" checked> '+label;
      cont.appendChild(lab);
    });

    cont.addEventListener('change', applyFilters);
    document.getElementById('stage-all').addEventListener('click', (e)=>{e.preventDefault(); cont.querySelectorAll('input').forEach(i=>i.checked=true); applyFilters();});
    document.getElementById('stage-none').addEventListener('click', (e)=>{e.preventDefault(); cont.querySelectorAll('input').forEach(i=>i.checked=false); applyFilters();});
  }

  // ----- init -----
  function init(){
    const table=document.getElementById('tbl');
    const ths=Array.from(table.tHead.rows[0].cells);

    ths.forEach((th, idx)=>{
      const ind=document.createElement('span');
      ind.className='sort-ind';
      ind.style.marginLeft='6px'; ind.style.opacity='0.7'; ind.textContent='↕';
      th.appendChild(ind);
      th.setAttribute('role','columnheader'); th.setAttribute('aria-sort','none');
      th.addEventListener('click', ()=>{
        const asc=!(th.dataset.asc==='true');
        clearIndicators(ths);
        th.dataset.asc=String(asc);
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        th.querySelector('.sort-ind').textContent=asc?'▲':'▼';
        sortTable(table, idx, asc);
      });
    });

    // default sort by ticker asc
    ths[COL.TICKER].dataset.asc='true';
    ths[COL.TICKER].querySelector('.sort-ind').textContent='▲';
    ths[COL.TICKER].setAttribute('aria-sort','ascending');

    // build Stage chips + hook search
    populateStageChips();
    applyFilters();
    const q = document.getElementById('q');
    if(q){ q.addEventListener('input', applyFilters); }
  }

  document.addEventListener('DOMContentLoaded', init);
})();
</script>
"""

    # --- small helpers for cells (unchanged) ---
    def pill(decision: str) -> str:
        d = (decision or "").strip().upper()
        if d == "BUY":  return '<span class="badge badge-buy">BUY</span>'
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
        import numpy as _np
        if val is None or not _np.isfinite(val):
            return '<span class="adx-neutral">—</span>'
        v = float(val)
        cls = "adx-strong" if v >= float(thr) else "adx-weak"
        return f'<span class="{cls}">{int(round(v))}</span>'

    def strg_cell(x) -> str:
        import numpy as _np
        if x is None or not _np.isfinite(x): return '<span class="strg strg-neutral">—</span>'
        p = float(x); label = f"{p:.2f}%"
        if p <= 0:   cls = "strg-neutral"
        elif p < 5:  cls = "strg-ok"
        elif p < 10: cls = "strg-warn"
        else:        cls = "strg-hot"
        return f'<span class="strg {cls}">{label}</span>'

    def pill_conf_bucket(bucket) -> str:
        import numpy as _np
        b = "" if bucket is None or (isinstance(bucket, float) and _np.isnan(bucket)) else str(bucket).strip().upper()
        if b == "HIGH": return '<span class="pill pill-high">HIGH</span>'
        if b == "LOW":  return '<span class="pill pill-low">LOW</span>'
        if b in ("MED","MEDIUM"): return '<span class="pill pill-med">MED</span>'
        return '<span class="pill pill-med">—</span>'

    def pill_regime(reg) -> str:
        import numpy as _np
        r = "" if reg is None or (isinstance(reg, float) and _np.isnan(reg)) else str(reg).strip()
        if r == "below_p50":  return '<span class="pill pill-high">below_p50</span>'
        if r == "above_p80":  return '<span class="pill pill-low">above_p80</span>'
        if r == "p50_to_p80": return '<span class="pill pill-med">p50_to_p80</span>'
        return '<span class="pill">unknown</span>'

    def fmt_int(x) -> str:
        try: return str(int(round(float(x))))
        except Exception: return "—"

    def _fmt_pct_int(x) -> str:
        import numpy as _np
        try: v = float(x)
        except Exception: return "—"
        if not _np.isfinite(v): return "—"
        if 0.0 <= v <= 1.0: v *= 100.0
        return f"{int(round(v))}"

    def _fmt_pct_1d(x) -> str:
        import numpy as _np
        try: v = float(x)
        except Exception: return "—"
        if not _np.isfinite(v): return "—"
        if abs(v) <= 1.0: v *= 100.0
        return f"{v:.1f}"

    def fmt_pct_int_cell(x) -> str: return _fmt_pct_int(x)
    def fmt_pct_1d_cell(x) -> str:  return _fmt_pct_1d(x)

    def clip_pct(v, lo=-25.0, hi=25.0):
        import numpy as _np
        try:
            f = float(v)
        except Exception:
            return "—"
        if not _np.isfinite(f):
            return "—"
        f = max(lo, min(hi, f))
        return f"{f:.1f}"

    head = (
        "<div class=\"wrap\">"
        f"<h1>Signals Dashboard — Rule Decisions <span style='font-size:14px;color:#94a3b8'>(label: {label})</span></h1>"
        f"<div class=\"info\">Updated: {updated_label}</div>"
        "<div class=\"card\">"
        "<input id=\"q\" class=\"search\" oninput=\"filterRows()\" placeholder=\"Search ticker, phase, trend, sentiment, smart money…\"/>"
        "<div class=\"filters\">"
        "<div class=\"filter-group\">"
        "<span style=\"font-size:12px;color:#94a3b8\">Stage:</span>"
        "<div id=\"stage-chips\" class=\"chips\"></div>"
        "<button class=\"btn-mini\" id=\"stage-all\">All</button>"
        "<button class=\"btn-mini\" id=\"stage-none\">None</button>"
        "</div>"
        "</div>"
        "<div style=\"overflow:auto; max-height:78vh; position:relative;\">"
        "<table id=\"tbl\">"
        "<thead><tr>"
        "<th>TICKER</th><th>AGE</th><th>TRADE</th>"
        "<th>TREND PHASE</th><th>TREND</th><th>STAGE</th>"
        "<th>STRENGTH-20</th><th>STRENGTH-44</th>"
        "<th>SENTIMENT</th><th>SMART MONEY</th>"
        "<th>RSI</th><th>ADX</th>"
        "<th>P25@5D</th><th>P75@5D</th>"
        "<th>CONF</th><th>BUCKET</th><th>REGIME</th>"
        "<th>P↑1D</th><th>P↓1D</th><th>CI↓</th><th>CI↑</th>"
        "<th>P↑3D</th><th>P↓3D</th><th>P↑5D</th><th>P↓5D</th>"
        "<th>MED touch20</th><th>MED touch44</th>"
        "<th>p50@20</th><th>p80@20</th><th>p50@44</th><th>p80@44</th>"
        "<th>n</th>"
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
            str(r.get("reason_long", "")),
            str(r.get("reason_short", "")),
        ])

        phase = str(r.get("trend_phase","")).upper()
        phase_cls = "phase-neutral"
        if phase == "BULLISH": phase_cls = "phase-bullish"
        elif phase == "BEARISH": phase_cls = "phase-bearish"

        rsi_val = r.get("rsi14_d", np.nan)
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
            f'<td class="{sent_cls(r.get("sentiment",""))}">{r.get("sentiment","")}</td>'
            f'<td class="{sm_cls(r.get("smartmoney",""))}">{r.get("smartmoney","")}</td>'
            f'<td class="{rsi_cls}">{rsi_txt}</td>'
            f'<td>{adx_cell(adx_val, adx_thr)}</td>'

            f'<td>{clip_pct(r.get("ret_p25_next5d", np.nan))}</td>'
            f'<td>{clip_pct(r.get("ret_p75_next5d", np.nan))}</td>'

            f'<td>{fmt_int(r.get("confidence_score_0_100", np.nan))}</td>'
            f'<td>{pill_conf_bucket(r.get("confidence_bucket",""))}</td>'
            f'<td>{pill_regime(r.get("stretch_regime",""))}</td>'

            f'<td>{fmt_pct_int_cell(r.get("p_up_next1d", np.nan))}</td>'
            f'<td>{fmt_pct_int_cell(r.get("p_down_next1d", np.nan))}</td>'
            f'<td>{fmt_pct_int_cell(r.get("p_ci_low_1d", np.nan))}</td>'
            f'<td>{fmt_pct_int_cell(r.get("p_ci_high_1d", np.nan))}</td>'

            f'<td>{fmt_pct_int_cell(r.get("p_up_next3d", np.nan))}</td>'
            f'<td>{fmt_pct_int_cell(r.get("p_down_next3d", np.nan))}</td>'

            f'<td>{fmt_pct_int_cell(r.get("p_up_next5d", np.nan))}</td>'
            f'<td>{fmt_pct_int_cell(r.get("p_down_next5d", np.nan))}</td>'

            f'<td>{fmt_int(r.get("median_bars_until_touch_ema20", np.nan))}</td>'
            f'<td>{fmt_int(r.get("median_bars_until_touch_ema44", np.nan))}</td>'

            f'<td>{fmt_pct_1d_cell(r.get("stretch_p50_vs_ema20", np.nan))}</td>'
            f'<td>{fmt_pct_1d_cell(r.get("stretch_p80_vs_ema20", np.nan))}</td>'
            f'<td>{fmt_pct_1d_cell(r.get("stretch_p50_vs_ema44", np.nan))}</td>'
            f'<td>{fmt_pct_1d_cell(r.get("stretch_p80_vs_ema44", np.nan))}</td>'

            f'<td>{fmt_int(r.get("n_events", np.nan))}</td>'

            f'<td>{_reason_chips(r.get("reason_long",""))}</td>'
            f'<td>{_reason_chips(r.get("reason_short",""))}</td>'
            f'</tr>'
        )

    tail = "</tbody></table></div></div></div>"

    # Glossary unchanged (use your existing 'defs' block)
    defs = """
<section class="defs" style="margin:14px 10px;color:#94a3b8">
  <h2 style="font-size:16px;margin:10px 0;color:#cbd5e1">Glossary</h2>

  <div class="grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px">

    <!-- Trend Phase / Trend / Age -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Trend Phase / Trend / Age</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Trend Phase</b>: Daily regime from Close vs EMA20/EMA44 (Bullish/Bearish/Neutral).</li>
        <li><b>Trend</b>: <i>short-term</i>, from Close vs EMA20 — UP if Close ≥ EMA20, else DOWN.</li>
        <li><b>Sentiment</b>: Risk ON only if Trend is UP and not stretched (regime ≠ above_p80), Smart Money ≠ Distribution/Markdown, and ADX ≥ threshold.</li>    
      </ul>
    </div>

    <!-- Strength / RSI / ADX -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Strength / RSI / ADX</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Strength-20 / -44</b>: % distance Close vs EMA20/EMA44 (from Gate-1 s77).</li>
        <li><b>RSI</b>: RSI(14).</li>
        <li><b>ADX</b>: ADX(14) trend strength. Color threshold = max(adx_min_long, adx_min_short) from metadata.</li>
      </ul>
    </div>

    <!-- Wyckoff-style Smart Money -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Wyckoff-style Smart Money</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Accumulation</b>: |ΔPrice| &lt; 0.2% and Volume not rising.</li>
        <li><b>Distribution</b>: |ΔPrice| &lt; 0.2%, Volume rising, Close &lt; EMA20.</li>
        <li><b>Markup</b>: Close &gt; EMA20 and ΔPrice &gt; 0.</li>
        <li><b>Markdown</b>: Close &lt; EMA20 and ΔPrice &lt; 0.</li>
      </ul>
    </div>

    <!-- Gate-1.5 -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Gate-1.5: Confidence, CI, Envelopes</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>CONF</b> / <b>BUCKET</b>: Model confidence score (0–100) and bucket (HIGH/MED/LOW).</li>
        <li><b>CI↓ / CI↑</b>: Calibrated 1-day confidence interval bounds for next-day move (lower/upper).</li>
        <li><b>P25@5D / P75@5D</b>: 5-day forward return percentiles (25th / 75th). Display clipped to ±25%.</li>
        <li><b>P↑/P↓ 1D/3D/5D</b>: Probabilities of up/down over 1/3/5 sessions.</li>
        <li><b>MED touch20/44</b>: Median bars until EMA20/EMA44 touch.</li>
        <li><b>p50/p80 @ 20/44</b>: Stretch percentiles vs EMAs; <b>n</b> = calibration sample size.</li>
      </ul>
    </div>

    <!-- Decision criteria -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Decision Criteria (Reason codes)</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>price_vs_base_long</b>: 30-min close ≤ EMA<small>BASE</small> (+buffer) → LONG gate fails.</li>
        <li><b>price_vs_base_short</b>: 30-min close ≥ EMA<small>BASE</small> (−buffer) → SHORT gate fails.</li>
        <li><b>confirm_days_long&lt;N</b> / <b>confirm_days_short&lt;N</b>: Not enough consecutive daily closes past threshold (today counts).</li>
        <li><b>trend_align_long/short</b>, <b>trend_gap_small</b>: EMA20 vs EMA44 alignment / insufficient gap.</li>
        <li><b>slope_long/short</b>, <b>base_slope_long/short</b>: Daily/30-min EMA slope guards.</li>
        <li><b>rsi_long_guard</b> / <b>rsi_short_guard</b>: RSI14 outside guard bands.</li>
        <li><b>adx_long_guard</b> / <b>adx_short_guard</b>: ADX14 below minimum (or not rising).</li>
        <li><b>donchian_* </b>: Width thin / extreme channel position / breakout not met.</li>
        <li><b>stretched_vs_ema44</b>: Close ≫ EMA44 (~&gt;10%) — caution for LONG.</li>
      </ul>
      <div style="font-size:11px;color:#94a3b8;margin-top:6px">
        <i>EMA<small>BASE</small> = 30-min EMA used by rules (e.g., 340 ≈ daily EMA20). Buffer = <code>buffer_pct</code> × EMA<small>BASE</small>.</i>
      </div>
    </div>

    <!-- Stretch Regime -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Stretch Regime (REGIME)</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>What</b>: Current “stretch” of price vs EMA20, using calibrated per-ticker percentiles.</li>
        <li><b>How</b>: Compare today’s % distance (Close−EMA20)/EMA20 to historical distribution; use p50 (median) and p80 thresholds.</li>
        <li><b>below_p50</b>: Typical stretch (≤ median). Lower mean-reversion risk.</li>
        <li><b>p50_to_p80</b>: Extended but not extreme (median–80th). Monitor / trim OK.</li>
        <li><b>above_p80</b>: Top 20% stretch. Elevated pullback risk; avoid fresh longs / wait for pullback.</li>
        <li style="color:#94a3b8">Tip: See “p50@20 / p80@20” columns for the actual thresholds used.</li>
      </ul>
    </div>

    <!-- Trend Stage -->
    <div class="box" style="background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:10px">
      <h3 style="margin:0 0 6px;font-size:13px;color:#cbd5e1">Trend Stage</h3>
      <ul style="margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4">
        <li><b>Emerging 🔥</b>: Trend just turned UP (AGE ≤ ~5) or 3-bar reclaim of EMA20; not stretched (S20 ≤ 2.5%, S44 ≤ 6%, RSI 45–65, regime ≠ above_p80).</li>
        <li><b>Steady ✅</b>: Healthy continuation (S20 0.5–5%, S44 &lt; 12%, RSI 45–68).</li>
        <li><b>Stretched ⚠️</b>: Late/extended (S20 &gt; 5% or S44 &gt; 12% or RSI &gt; 68 or regime = above_p80).</li>
        <li><i>Near-20 🧪</i>: UP trend hovering near EMA20 (S20 −1% … +0.5%, RSI 40–55) — watchlist for potential bounces.</li>
        <li><i>Pullback-20 🔁</i>: UP trend with S20 &lt; 0 and EMA20 above EMA44 — controlled pullback to the 20 (possible buy-the-dip setups).</li>
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
def _load_universe_from_xlsx(path: Path, sheet: str = "US_small",
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
    ap = argparse.ArgumentParser(description="s81 v1.6 — s77 + Gate-1.5 + Daily enriched + Trend Stage.")
    ap.add_argument("--label", type=str, default="gate1_v1.0", help="Which s77 label to render.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (debug).")
    ap.add_argument("--signals-dir", type=str, default=str(P.ROOT / "signals"), help="Signals directory (s77 CSV + metadata).")
    ap.add_argument("--csv", type=str, default="", help="Optional explicit path to rule_live_signals_*.csv to render.")
    ap.add_argument("--universe-sheet", type=str, default="US_small", help="Excel sheet in ETF_list.xlsx to filter universe.")
    ap.add_argument("--serve", action="store_true", help="Run a local HTTP server to view the dashboard.")
    ap.add_argument("--port", type=int, default=8080, help="Port for the local server.")
    args = ap.parse_args()

    label = args.label.strip()
    signals_dir = Path(args.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

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
        else:
            print("[WARN] No metadata found next to the provided --csv; using ADX threshold=20.")
            adx_thr = 20.0
    else:
        csv_path, meta_path, _ = _find_latest_by_label(signals_dir, label=label)
        meta_obj = json.loads(meta_path.read_text())
        adx_thr = float(max(meta_obj.get("adx_min_long", 20.0),
                            meta_obj.get("adx_min_short", 20.0)))

    print(f"[INFO] Using signals CSV: {csv_path.name}")
    if 'meta_path' in locals() and meta_path and meta_path.exists():
        print(f"[INFO] Metadata file     : {meta_path.name}")

    # ---- load s77 table (source of truth) ----
    df_sig = pd.read_csv(csv_path)
    need = {"ticker", "decision"}
    if not need.issubset(set(df_sig.columns)):
        raise SystemExit(f"[ERR] {csv_path.name} missing required columns {sorted(need)}.")
    df_sig["ticker"] = df_sig["ticker"].astype(str).str.strip().str.upper()
    df_sig["decision"] = df_sig["decision"].astype(str).str.strip().str.upper()

    # ---- universe filter ----
    universe = _load_universe_from_xlsx(P.CONFIG_DIR / "ETF_list.xlsx", sheet=args.universe_sheet)
    if universe:
        df_sig = df_sig[df_sig["ticker"].isin(universe)]
    else:
        print("[WARN] Universe is empty; dashboard will include all rows (fallback).")

    # ---- optional limit (debug) ----
    if args.limit and args.limit > 0:
        keep = df_sig["ticker"].unique().tolist()[: args.limit]
        df_sig = df_sig[df_sig["ticker"].isin(keep)]

    # ---- Gate-1.5 merge (safe) ----
    g15 = _load_gate15_all()
    if g15 is not None and not g15.empty:
        g15 = _apply_aliases(g15)
        # avoid accidental overwrites
        forbid = set(df_sig.columns)
        keep_cols = [c for c in g15.columns if (c == "ticker") or (c not in forbid)]
        g15 = g15[keep_cols]
        df = df_sig.merge(g15, on="ticker", how="left")
    else:
        print("[WARN] Gate-1.5 analytics not found — rendering without enriched fields.")
        df = df_sig.copy()

    # ---- daily enriched context (from s00) ----
    daily_map = _load_daily_enriched_master(P.DATA_ENRICHED / "prices_enriched.parquet")

    # ---- build rows + compute Trend Stage ----
    out_rows = []
    for _, row in df.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision", "")).strip().upper()

        dctx = daily_map.get(t, {})
        close_t = dctx.get("close", np.nan)
        close_y = dctx.get("close_y", np.nan)
        ema20_t = dctx.get("ema20_d", np.nan)
        ema44_t = dctx.get("ema44_d", np.nan)
        rsi14_d = row.get("rsi14_d", dctx.get("rsi14_d", np.nan))
        adx14_d = row.get("adx14_d", dctx.get("adx14_d", np.nan))
        vol_t   = dctx.get("volume", np.nan)
        vol_y   = dctx.get("volume_y", np.nan)

        phase = _trend_phase(close_t, ema20_t, ema44_t)                 # structural (EMA20 vs EMA44)
        sm    = _smart_money_wyckoff(close_t, close_y, vol_t, vol_y, ema20_t)
        trn   = _trend_short(close_t, ema20_t)                          # NEW short-term trend (Close vs EMA20)
        sent  = _sentiment_risk(close_t, ema20_t,
                                str(row.get("stretch_regime", "")),
                                sm,
                                adx14_d,
                                adx_thr)                                 # NEW risk state

        row_dict = {
            "ticker": t,
            "decision": decision,
            "trend_phase": phase,
            "trend": trn,
            "trend_age": dctx.get("trend_days_since_start_d", np.nan),

            # strengths (from s77)
            "strength20_pct": row.get("strength20_pct", np.nan),
            "strength44_pct": row.get("strength44_pct", np.nan),

            "sentiment": sent,
            "smartmoney": sm,
            "rsi14_d": rsi14_d,
            "adx14_d": adx14_d,

            # Gate-1.5 passthrough
            "confidence_score_0_100": row.get("confidence_score_0_100", np.nan),
            "confidence_bucket":      row.get("confidence_bucket", ""),
            "stretch_regime":         row.get("stretch_regime", ""),
            "p_up_next1d":            row.get("p_up_next1d", np.nan),
            "p_down_next1d":          row.get("p_down_next1d", np.nan),
            "p_ci_low_1d":            row.get("p_ci_low_1d", np.nan),
            "p_ci_high_1d":           row.get("p_ci_high_1d", np.nan),
            "p_up_next3d":            row.get("p_up_next3d", np.nan),
            "p_down_next3d":          row.get("p_down_next3d", np.nan),
            "p_up_next5d":            row.get("p_up_next5d", np.nan),
            "p_down_next5d":          row.get("p_down_next5d", np.nan),
            "ret_p25_next5d":         row.get("ret_p25_next5d", np.nan),
            "ret_p75_next5d":         row.get("ret_p75_next5d", np.nan),
            "median_bars_until_touch_ema20": row.get("median_bars_until_touch_ema20", np.nan),
            "median_bars_until_touch_ema44": row.get("median_bars_until_touch_ema44", np.nan),
            "stretch_p50_vs_ema20": row.get("stretch_p50_vs_ema20", np.nan),
            "stretch_p80_vs_ema20": row.get("stretch_p80_vs_ema20", np.nan),
            "stretch_p50_vs_ema44": row.get("stretch_p50_vs_ema44", np.nan),
            "stretch_p80_vs_ema44": row.get("stretch_p80_vs_ema44", np.nan),
            "n_events": row.get("n_events", np.nan),

            # reasons: only when DO NOTHING
            "reason_long":  (row.get("reason_long",  "") if decision == "DO NOTHING" else ""),
            "reason_short": (row.get("reason_short", "") if decision == "DO NOTHING" else ""),
            # recent reclaim flag from daily context (if available)
            "recent_cross_above_ema20_3d": bool(dctx.get("recent_cross_above_ema20_3d", False)),
        }

        # compute stage label/rank
        stage_label, stage_rank = _trend_stage({**row, **row_dict})
        row_dict["trend_stage"] = stage_label
        row_dict["_stage_rank"] = stage_rank

        out_rows.append(row_dict)

    if not out_rows:
        raise SystemExit("[ERR] No rows to render.")

    out = pd.DataFrame(out_rows)

    # ---- ordering: actionables first ----
    out["__ord"]   = np.where(out["decision"].isin(["BUY","SELL"]), 0, 1)
    out["__stage"] = pd.to_numeric(out.get("_stage_rank", np.nan), errors="coerce").fillna(9)
    out["__conf"]  = pd.to_numeric(out.get("confidence_score_0_100", np.nan), errors="coerce").fillna(-1)
    out["__s20"]   = pd.to_numeric(out.get("strength20_pct", np.nan), errors="coerce").fillna(-1e9)

    out = (out.sort_values(["__ord","__stage","__conf","__s20"],
                       ascending=[True, True, False, False])
          .drop(columns=["__ord","__stage","__conf","__s20","_stage_rank"])
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

    # ---- optional local server ----
    if args.serve:
        _serve_local(signals_dir, index_file=out_latest.name, host="127.0.0.1", port=int(args.port))
if __name__ == "__main__":
    main()