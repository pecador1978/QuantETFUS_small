#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s81_decision_board_rules_v1.py — Operator-facing HTML dashboard for Gate-1 (JSON-driven s77, label-aware).

New in this version:
- Fully merges Gate-1.5 analytics:
  * signals/analytics/gate15_stats.csv
  * signals/analytics/gate15_calibration_confidence.csv  (optional)
  * signals/analytics/gate15_calibration_stretch.csv     (optional)
- Adds: confidence score/bucket, stretch regime, next-1/3/5D probabilities, 5D envelopes,
        median bars to EMA20/EMA44, stretch p50/p80 vs EMA20/EMA44, n_events.
- Keeps original styling; new columns are sortable and tolerant to missing data.
- --serve to launch a local server for viewing (127.0.0.1:8080).
- Label-specific output names.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os, json
import pandas as pd
import numpy as np
from typing import Optional

# ---------- project-aware paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # noqa: E402


# ----------------- helpers -----------------
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")


def _market_tz() -> str:
    tz = os.environ.get("MARKET_TZ", "").strip()
    if tz:
        return tz
    try:
        from common.settings import MARKET_TZ as STZ  # type: ignore
        if STZ:
            return str(STZ)
    except Exception:
        pass
    return "Europe/London"


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
        if (ema20_d > ema44_d) and (close_d >= ema20_d):
            return "BULLISH"
        if (ema20_d < ema44_d) and (close_d <= ema44_d):
            return "BEARISH"
        return "NEUTRAL"
    return ""


def _trend(ema20_d: float, ema44_d: float) -> str:
    if np.isfinite(ema20_d) and np.isfinite(ema44_d):
        return "UP" if ema20_d >= ema44_d else "DOWN"
    return ""


def _sentiment(trend: str, close_d: float, ema20_d: float) -> str:
    if trend == "UP" and np.isfinite(close_d) and np.isfinite(ema20_d) and close_d >= ema20_d:
        return "Risk ON"
    return "Risk OFF"


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


def _read_daily_ctx(tkr: str, enriched_dir: Path, mkt_tz: str) -> dict:
    p = enriched_dir / f"{tkr}.parquet"
    if not p.exists(): return {}
    df = pd.read_parquet(p)
    if df.empty or "datetime" not in df.columns: return {}
    df.columns = [c.strip().lower() for c in df.columns]
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None: dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df = df.sort_values("datetime")
    if "volume" not in df.columns: df["volume"] = np.nan
    else: df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    local = df["datetime"].dt.tz_convert(mkt_tz)
    df = df.assign(date_local=local.dt.date)
    day_vol = df.groupby("date_local", as_index=False)["volume"].sum(numeric_only=True).rename(columns={"volume":"day_vol"})
    day_last = df.groupby("date_local", as_index=False).tail(1).reset_index(drop=True)
    daily = pd.merge(day_last, day_vol, on="date_local", how="left")
    if daily.empty: return {}
    def getf(row, c):
        v = row.get(c, np.nan)
        try: return float(v)
        except Exception: return np.nan
    today = daily.iloc[-1]
    yday  = daily.iloc[-2] if len(daily) >= 2 else None
    return {
        "close_t":   getf(today, "close"),
        "ema20_t":   getf(today, "ema20_d"),
        "ema44_t":   getf(today, "ema44_d"),
        "rsi14_t":   getf(today, "rsi14_d"),
        "day_vol_t": getf(today, "day_vol"),
        "close_y":   getf(yday, "close")    if yday is not None else np.nan,
        "day_vol_y": getf(yday, "day_vol")  if yday is not None else np.nan,
    }


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
    # try common candidates
    candidates = {"ticker","Ticker","symbol","Symbol"}
    found = next((c for c in df.columns if c in candidates), None)
    if not found:
        print(f"[WARN] {src_name} has no ticker-like column → skipping")
        return None
    if found != "ticker":
        df = df.rename(columns={found:"ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["ticker"])


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

def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename alternate Gate-1.5 column names to the canonical ones expected by the board.
    Safe to call even if aliases don't exist.
    """
    if df is None or df.empty:
        return df

    aliases = {
        # probabilities (1d/3d/5d)
        "p_up_next1d":   ["p_up_next1d","p_up_1d","p_up1d"],
        "p_down_next1d": ["p_down_next1d","p_down_1d","p_down1d","p_dn_next1d","p_dn_1d","p_dn1d"],
        "p_ci_low_1d":   ["p_ci_low_1d","ci_low_1d","ci1d_low"],
        "p_ci_high_1d":  ["p_ci_high_1d","ci_high_1d","ci1d_high"],

        "p_up_next3d":   ["p_up_next3d","p_up_3d","p_up3d"],
        "p_down_next3d": ["p_down_next3d","p_down_3d","p_down3d","p_dn_next3d","p_dn_3d","p_dn3d"],

        "p_up_next5d":   ["p_up_next5d","p_up_5d","p_up5d"],
        "p_down_next5d": ["p_down_next5d","p_down_5d","p_down5d","p_dn_next5d","p_dn_5d","p_dn5d"],

        # envelopes
        "ret_p25_next5d": ["ret_p25_next5d","p25_ret_5d","ret25_5d"],
        "ret_p75_next5d": ["ret_p75_next5d","p75_ret_5d","ret75_5d"],

        # stretch stats
        "stretch_p50_vs_ema20": ["stretch_p50_vs_ema20","p50_vs_ema20","stretch_p50_ema20"],
        "stretch_p80_vs_ema20": ["stretch_p80_vs_ema20","p80_vs_ema20","stretch_p80_ema20"],
        "stretch_p50_vs_ema44": ["stretch_p50_vs_ema44","p50_vs_ema44","stretch_p50_ema44"],
        "stretch_p80_vs_ema44": ["stretch_p80_vs_ema44","p80_vs_ema44","stretch_p80_ema44"],

        # meta
        "confidence_score_0_100": ["confidence_score_0_100","confidence_score","conf_score"],
        "confidence_bucket":      ["confidence_bucket","conf_bucket","bucket"],
        "stretch_regime":         ["stretch_regime","regime"],
    }

    # build reverse map for renaming
    rename_map = {}
    for canon, alts in aliases.items():
        for a in alts:
            if a in df.columns:
                rename_map[a] = canon
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def _fmt_pct_int(x) -> str:
    """Format probability 0..1 or 0..100 as integer percent (no % sign for sorting consistency)."""
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    if 0.0 <= v <= 1.0:
        v *= 100.0
    return f"{int(round(v))}"


def _fmt_pct_1d(x) -> str:
    """Format decimal return to percent with 1 decimal (0.035 -> 3.5)."""
    try:
        v = float(x)
    except Exception:
        return "—"
    if not np.isfinite(v):
        return "—"
    if abs(v) <= 1.0:
        v *= 100.0
    return f"{v:.1f}"


# ----------------- HTML builder -----------------
def build_html(df: pd.DataFrame, updated_label: str, label: str, adx_thr: float) -> str:
    style = """
<style>
:root{
  --bg:#0b1020; --card:#0f172a; --muted:#94a3b8; --txt:#e2e8f0; --line:#1f2937;
  --pill-buy:#059669; --pill-sell:#dc2626; --pill-wait:#475569;
  --green:#16a34a; --orange:#f59e0b; --red:#ef4444; --neutral:#334155;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial;color:#e2e8f0}
.wrap{max-width:100%;margin:10px;padding:0}
h1{margin:0 0 8px;font-size:22px}
.info{color:var(--muted);font-size:12px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.search{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);
  background:#0b132d;color:#e2e8f0;outline:none;margin-bottom:10px}
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

/* G15 pills */
.pill{display:inline-block;padding:2px 8px;border-radius:8px;border:1px solid #2b344a;font-size:11px}
.pill-high{background:rgba(22,163,74,.12);color:#86efac;border-color:rgba(22,163,74,.3);font-weight:700}
.pill-med {background:rgba(245,158,11,.12);color:#fbbf24;border-color:rgba(245,158,11,.35);font-weight:700}
.pill-low {background:rgba(220,38,38,.12);color:#fca5a5;border-color:rgba(220,38,38,.35);font-weight:700}
.pill-below{color:#86efac;font-weight:700}
.pill-mid  {color:#fbbf24;font-weight:700}
.pill-above{color:#fca5a5;font-weight:700}

/* sticky first column (Ticker) */
#tbl { border-collapse: separate; border-spacing: 0; }
#tbl th:first-child, #tbl td:first-child{
  position: sticky;
  left: 0;
  background: var(--card);
}
#tbl th:first-child{ z-index: 6; }           /* above other headers */
#tbl td:first-child{ z-index: 1; border-right: 1px solid var(--line); }

/* glossary */
.defs{margin-top:14px;color:var(--muted)}
.defs h2{font-size:16px;margin:10px 0}
.defs .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.defs .box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}
.defs h3{margin:0 0 6px;font-size:13px;color:#cbd5e1}
.defs ul{margin:0 0 0 16px;padding:0;font-size:12px;line-height:1.4}
.defs code{background:#0b132d;padding:1px 4px;border-radius:4px;color:#e5e7eb;border:1px solid #1f2937}
</style>
    """
    script = """
<script>
(function(){
  // Update numeric columns: keep in sync with <thead> indexes below.
  function getColType(idx){
    const numIdx = new Set([
      4,5,8,9,               // strength20/44, RSI, ADX
      12,13,14,15,           // P↑1D, P↓1D, CI↓, CI↑
      16,17,18,19,           // P↑3D, P↓3D, P↑5D, P↓5D
      20,21,                 // P25@5D, P75@5D
      22,23,                 // MED touch20/44
      24,25,26,27,           // p50@20, p80@20, p50@44, p80@44
      28                     // n
    ]);
    return numIdx.has(idx) ? 'num' : 'text';
  }

  function parseCell(txt, forcedType){
    const s = (txt || '').trim();
    if (forcedType === 'text') return { val: s.toLowerCase(), type: 'text' };
    if (forcedType === 'num'){
      const pct = s.replace(/[,%]/g,'');
      const num = pct.replace(/[^0-9.+-]/g,'');
      const v = parseFloat(num);
      return { val: isNaN(v) ? Number.NEGATIVE_INFINITY : v, type:'num' };
    }
    const pct = s.replace(/[,%]/g,'');
    if (!isNaN(parseFloat(pct)) && s.indexOf('%')>=0) return {val: parseFloat(pct), type:'num'};
    const num = s.replace(/[^0-9.+-]/g,'');
    if (num && !isNaN(parseFloat(num)) && /[0-9.+-]/.test(s)) return {val: parseFloat(num), type:'num'};
    return {val: s.toLowerCase(), type:'text'};
  }

  function getCellText(td){ return (td && td.textContent ? td.textContent : '').trim(); }

  function sortTable(table, colIndex, asc){
    const forcedType = getColType(colIndex);
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr')).map((r,i)=>({r,i}));
    rows.sort((A,B)=>{
      const a = getCellText(A.r.cells[colIndex]);
      const b = getCellText(B.r.cells[colIndex]);
      const Ap = parseCell(a, forcedType), Bp = parseCell(b, forcedType);
      let cmp = 0;
      if (Ap.type==='num' && Bp.type==='num'){
        cmp = Ap.val - Bp.val;
      } else {
        cmp = String(Ap.val).localeCompare(String(Bp.val));
      }
      if (!asc) cmp = -cmp;
      return cmp !== 0 ? cmp : (A.i - B.i);
    });
    rows.forEach(obj=>tbody.appendChild(obj.r));
  }

  function clearIndicators(ths){
    ths.forEach(th=>{
      th.removeAttribute('data-asc');
      th.setAttribute('aria-sort','none');
      const span = th.querySelector('.sort-ind');
      if (span) span.textContent = '↕';
    });
  }

  function init(){
    const table = document.getElementById('tbl');
    const ths = Array.from(table.tHead.rows[0].cells);
    ths.forEach((th, idx)=>{
      let ind = document.createElement('span');
      ind.className = 'sort-ind';
      ind.style.marginLeft = '6px';
      ind.style.opacity = '0.7';
      ind.textContent = '↕';
      th.appendChild(ind);
      th.setAttribute('role','columnheader');
      th.setAttribute('aria-sort','none');
      th.addEventListener('click', ()=>{
        const asc = !(th.dataset.asc === 'true');
        clearIndicators(ths);
        th.dataset.asc = String(asc);
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        th.querySelector('.sort-ind').textContent = asc ? '▲' : '▼';
        sortTable(table, idx, asc);
      });
    });

    const q = document.getElementById('q');
    if (q){
      q.addEventListener('input', function(){
        const needle = this.value.toLowerCase();
        const rows = table.tBodies[0].querySelectorAll('tr');
        rows.forEach(r=>{
          const txt = (r.getAttribute('data-f') || '').toLowerCase();
          r.style.display = txt.indexOf(needle) >= 0 ? '' : 'none';
        });
      });
    }

    // Default: show as 'Ticker ▲'
    const tickerIdx = 0;
    ths[tickerIdx].dataset.asc = 'true';
    ths[tickerIdx].querySelector('.sort-ind').textContent = '▲';
    ths[tickerIdx].setAttribute('aria-sort','ascending');
  }

  document.addEventListener('DOMContentLoaded', init);
})();
</script>
"""

    def pill(decision: str) -> str:
        d = (decision or "").upper()
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
        if val is None or not np.isfinite(val):
            return '<span class="adx-neutral">—</span>'
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

    # REPLACE around line ~455
    def pill_conf_bucket(bucket) -> str:
        # tolerate NaN/float/None; normalize to UPPER string
        b = "" if bucket is None or (isinstance(bucket, float) and np.isnan(bucket)) else str(bucket).strip().upper()
        if b == "HIGH": return '<span class="pill pill-high">HIGH</span>'
        if b == "LOW":  return '<span class="pill pill-low">LOW</span>'
        if b == "MED" or b == "MEDIUM": return '<span class="pill pill-med">MED</span>'
        # unknown / empty -> neutral MED styling
        return '<span class="pill pill-med">—</span>'


    def pill_regime(reg) -> str:
        r = "" if reg is None or (isinstance(reg, float) and np.isnan(reg)) else str(reg).strip()
        if r == "below_p50":  return '<span class="pill pill-below">below_p50</span>'
        if r == "above_p80":  return '<span class="pill pill-above">above_p80</span>'
        if r == "p50_to_p80": return '<span class="pill pill-mid">p50_to_p80</span>'
        return '<span class="pill">unknown</span>'

    def fmt_int(x) -> str:
        try:
            return str(int(round(float(x))))
        except Exception:
            return "—"

    def fmt_pct_int_cell(x) -> str:
        return _fmt_pct_int(x)

    def fmt_pct_1d_cell(x) -> str:
        return _fmt_pct_1d(x)

    head = (
        "<div class=\"wrap\">"
        f"<h1>Signals Dashboard — Rule Decisions <span style='font-size:14px;color:#94a3b8'>(label: {label})</span></h1>"
        f"<div class=\"info\">Updated: {updated_label}</div>"
        "<div class=\"card\">"
        "<input id=\"q\" class=\"search\" oninput=\"filterRows()\" placeholder=\"Search ticker, phase, trend, sentiment, smart money…\"/>"
        "<div style=\"overflow:auto; max-height:78vh; position:relative;\">"
        "<table id=\"tbl\">"
        "<thead><tr>"
        "<th>TICKER</th><th>TRADE</th>"
        "<th>TREND PHASE</th><th>TREND</th>"
        "<th>STRENGTH-20</th><th>STRENGTH-44</th>"
        "<th>SENTIMENT</th><th>SMART MONEY</th>"
        "<th>RSI</th><th>ADX</th>"
        "<th>CONF</th><th>BUCKET</th><th>REGIME</th>"
        "<th>P↑1D</th><th>P↓1D</th><th>CI↓</th><th>CI↑</th>"
        "<th>P↑3D</th><th>P↓3D</th><th>P↑5D</th><th>P↓5D</th>"
        "<th>P25@5D</th><th>P75@5D</th>"
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
            str(r.get("sentiment", "")),
            str(r.get("smartmoney", "")),
            str(r.get("confidence_bucket", "")),
            str(r.get("stretch_regime", "")),
            str(r.get("reason_long", "")),
            str(r.get("reason_short", "")),
        ])

        # Phase color class
        phase = str(r.get("trend_phase","")).upper()
        phase_cls = "phase-neutral"
        if phase == "BULLISH": phase_cls = "phase-bullish"
        elif phase == "BEARISH": phase_cls = "phase-bearish"

        # RSI value + class
        rsi_val = r.get("rsi14_d", np.nan)
        rsi_txt = ""
        rsi_cls = ""
        if np.isfinite(rsi_val):
            rsi_txt = str(int(round(rsi_val)))
            if rsi_val >= 70: rsi_cls = "rsi-hot"
            elif rsi_val <= 30: rsi_cls = "rsi-cold"

        # ADX value
        adx_val = r.get("adx14_d", np.nan)

        rows_html.append(
            f'<tr data-f="{data_filter}">'
            f'<td>{r.get("ticker","")}</td>'
            f'<td>{pill(r.get("decision",""))}</td>'
            f'<td class="{phase_cls}">{phase}</td>'
            f'<td class="{trend_cls(r.get("trend",""))}">{r.get("trend","")}</td>'
            f'<td>{strg_cell(r.get("strength20_pct", None))}</td>'
            f'<td>{strg_cell(r.get("strength44_pct", None))}</td>'
            f'<td class="{sent_cls(r.get("sentiment",""))}">{r.get("sentiment","")}</td>'
            f'<td class="{sm_cls(r.get("smartmoney",""))}">{r.get("smartmoney","")}</td>'
            f'<td class="{rsi_cls}">{rsi_txt}</td>'
            f'<td>{adx_cell(adx_val, adx_thr)}</td>'
            # Gate-1.5 appended cells
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
            f'<td>{fmt_pct_1d_cell(r.get("ret_p25_next5d", np.nan))}</td>'
            f'<td>{fmt_pct_1d_cell(r.get("ret_p75_next5d", np.nan))}</td>'
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

    defs = """
<section class="defs">
  <h2>Glossary / Rules</h2>
  <div class="grid">
    <div class="box">
      <h3>Trend Phase (daily)</h3>
      <ul>
        <li><b>Bullish</b>: EMA20 &gt; EMA44 and Close ≥ EMA20</li>
        <li><b>Bearish</b>: EMA20 &lt; EMA44 and Close ≤ EMA44</li>
        <li><b>Neutral</b>: Otherwise (transition / flattening)</li>
      </ul>
    </div>
    <div class="box">
      <h3>Trend</h3>
      <ul>
        <li><b>UP</b>: EMA20 ≥ EMA44</li>
        <li><b>DOWN</b>: EMA20 &lt; EMA44</li>
      </ul>
    </div>
    <div class="box">
      <h3>Strength Metrics</h3>
      <ul>
        <li><b>Strength-20</b>: % distance of Close vs EMA20</li>
        <li><b>Strength-44</b>: % distance of Close vs EMA44</li>
      </ul>
    </div>
    <div class="box">
      <h3>Smart Money (Wyckoff-style)</h3>
      <ul>
        <li><b>Accumulation</b>: |ΔPrice| &lt; 0.2% and Volume not rising</li>
        <li><b>Distribution</b>: |ΔPrice| &lt; 0.2%, Volume rising, Close &lt; EMA20</li>
        <li><b>Markup</b>: Close &gt; EMA20 and ΔPrice &gt; 0</li>
        <li><b>Markdown</b>: Close &lt; EMA20 and ΔPrice &lt; 0</li>
        <li><b>Neutral</b>: Otherwise (or if volume missing)</li>
      </ul>
    </div>
    <div class="box">
      <h3>Gate-1.5 Fields</h3>
      <ul>
        <li><b>CONF</b>: 0–100 composite (stretch/ADX/RSI/Donchian weights)</li>
        <li><b>BUCKET</b>: HIGH / MED / LOW (from confidence calibration)</li>
        <li><b>REGIME</b>: below_p50 / p50_to_p80 / above_p80 (stretch vs EMA20 history)</li>
        <li><b>P↑1D / P↑5D</b>: probability of up over next 1/5 days</li>
        <li><b>P25@5D / P75@5D</b>: 5-day forward return envelope (25th/75th pct)</li>
        <li><b>MED touch20/44</b>: median bars until next EMA20/EMA44 touch</li>
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
    """Serve the signals directory on localhost and open the index file."""
    import http.server, socketserver, webbrowser
    os.chdir(str(signals_dir))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/{index_file}"
        print(f"[SERVE] Local server at {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVE] Stopped by user.")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="s81 v1.0 — label-aware dashboard for s77 rule signals, with localhost server.")
    ap.add_argument("--label", type=str, default="gate1_v1.0",
                help="Which s77 label to render (e.g., gate1_v1.0, gate1_strict, ...).")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (debug).")
    ap.add_argument("--signals-dir", type=str, default=str(P.ROOT / "signals"),
                    help="Signals directory containing CSV + metadata from s77.")
    ap.add_argument("--universe-sheet", type=str, default="US_small",
                    help="Excel sheet in ETF_list.xlsx to filter universe.")
    ap.add_argument("--serve", action="store_true", help="Run a local HTTP server to view the dashboard.")
    ap.add_argument("--port", type=int, default=8080, help="Port for the local server.")
    args = ap.parse_args()

    label = args.label.strip()
    signals_dir = Path(args.signals_dir)
    enriched_dir = P.DATA_ENRICHED / "30min"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # Find latest s77 output for this label via metadata
    csv_path, meta_path, ts_sig = _find_latest_by_label(signals_dir, label=label)
    print(f"[INFO] Using signals CSV: {csv_path.name}")
    print(f"[INFO] Metadata file     : {meta_path.name}")
    meta_obj = json.loads(meta_path.read_text())
    adx_thr = float(max(meta_obj.get("adx_min_long", 20.0),
                        meta_obj.get("adx_min_short", 20.0)))

    # Load s77 table
    df_sig = pd.read_csv(csv_path)
    if "ticker" not in df_sig.columns or "decision" not in df_sig.columns:
        raise SystemExit(f"[ERR] {csv_path.name} missing required columns ('ticker','decision').")
    df_sig["ticker"] = df_sig["ticker"].astype(str).str.strip().str.upper()

    # Universe filter
    universe = _load_universe_from_xlsx(P.CONFIG_DIR / "ETF_list.xlsx", sheet=args.universe_sheet)
    if universe:
        df_sig = df_sig[df_sig["ticker"].isin(universe)]
    else:
        print("[WARN] Universe is empty; dashboard will include all rows (fallback).")

    # Optional limit
    if args.limit and args.limit > 0:
        keep = df_sig["ticker"].astype(str).str.upper().unique().tolist()[: args.limit]
        df_sig = df_sig[df_sig["ticker"].isin(keep)]

    mkt_tz = _market_tz()

    # preserve s77 strengths so analytics cannot overwrite with NaN
    if "strength20_pct" in df_sig.columns:
        df_sig = df_sig.rename(columns={"strength20_pct":"strength20_pct_s77"})
    if "strength44_pct" in df_sig.columns:
        df_sig = df_sig.rename(columns={"strength44_pct":"strength44_pct_s77"})

    # ---------------- Gate-1.5 merge ----------------
    g15 = _load_gate15_all()  # if you use _load_gate15_df(), swap here accordingly
    if g15 is not None:
        g15 = _apply_aliases(g15)
        df_sig = df_sig.merge(g15, on="ticker", how="left")

        # Restore strengths if analytics are missing
        if "strength20_pct_s77" in df_sig.columns:
            df_sig["strength20_pct"] = pd.to_numeric(df_sig.get("strength20_pct"), errors="coerce")
            df_sig["strength20_pct"] = df_sig["strength20_pct"].fillna(df_sig["strength20_pct_s77"])
        if "strength44_pct_s77" in df_sig.columns:
            df_sig["strength44_pct"] = pd.to_numeric(df_sig.get("strength44_pct"), errors="coerce")
            df_sig["strength44_pct"] = df_sig["strength44_pct"].fillna(df_sig["strength44_pct_s77"])

        # Normalize categories so render helpers don't choke
        for col in ["confidence_bucket", "stretch_regime"]:
            if col in df_sig.columns:
                df_sig[col] = df_sig[col].astype(str).where(~df_sig[col].isna(), "")
    else:
        print("[WARN] Gate-1.5 analytics not found — rendering without enriched fields.")

    # Build rows (add daily context for phase/sentiment/smart-money if needed)
    out_rows = []
    for _, row in df_sig.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision","")).upper()

        dctx = _read_daily_ctx(t, enriched_dir, mkt_tz)
        close_t = dctx.get("close_t",  np.nan)
        close_y = dctx.get("close_y",  np.nan)
        ema20_t = dctx.get("ema20_t",  np.nan)
        ema44_t = dctx.get("ema44_t",  np.nan)
        # prefer analytics RSI/ADX if present; fallback to parquet context
        rsi14_d = row.get("rsi14_d", np.nan)
        if not np.isfinite(rsi14_d): rsi14_d = dctx.get("rsi14_t", np.nan)
        adx14_d = row.get("adx14_d", np.nan)
        vol_t   = dctx.get("day_vol_t", np.nan)
        vol_y   = dctx.get("day_vol_y", np.nan)

        phase = _trend_phase(close_t, ema20_t, ema44_t)
        trn   = _trend(ema20_t, ema44_t)
        sent  = _sentiment(trn, close_t, ema20_t)
        sm    = _smart_money_wyckoff(close_t, close_y, vol_t, vol_y, ema20_t)

        # strengths: prefer analytics if present; fallback to s77
        str20 = row.get("strength20_pct", np.nan)
        str44 = row.get("strength44_pct", np.nan)

        reason_long  = row.get("reason_long",  "")
        reason_short = row.get("reason_short", "")
        if isinstance(reason_long, float) and np.isnan(reason_long):   reason_long  = ""
        if isinstance(reason_short, float) and np.isnan(reason_short): reason_short = ""

        out_rows.append({
            "ticker": t,
            "decision": decision,
            "trend_phase": phase,
            "trend": trn,
            "strength20_pct": str20,
            "strength44_pct": str44,
            "sentiment": sent,
            "smartmoney": sm,
            "rsi14_d": rsi14_d,
            "adx14_d": adx14_d,
            # Gate-1.5 fields (may be NaN if analytics missing)
            "confidence_score_0_100": row.get("confidence_score_0_100", np.nan),
            "confidence_bucket": row.get("confidence_bucket", ""),
            "stretch_regime": row.get("stretch_regime", ""),
            "p_up_next1d": row.get("p_up_next1d", np.nan),
            "p_down_next1d": row.get("p_down_next1d", np.nan),
            "p_ci_low_1d": row.get("p_ci_low_1d", np.nan),
            "p_ci_high_1d": row.get("p_ci_high_1d", np.nan),
            "p_up_next3d": row.get("p_up_next3d", np.nan),
            "p_down_next3d": row.get("p_down_next3d", np.nan),
            "p_up_next5d": row.get("p_up_next5d", np.nan),
            "p_down_next5d": row.get("p_down_next5d", np.nan),
            "ret_p25_next5d": row.get("ret_p25_next5d", np.nan),
            "ret_p75_next5d": row.get("ret_p75_next5d", np.nan),
            "median_bars_until_touch_ema20": row.get("median_bars_until_touch_ema20", np.nan),
            "median_bars_until_touch_ema44": row.get("median_bars_until_touch_ema44", np.nan),
            "stretch_p50_vs_ema20": row.get("stretch_p50_vs_ema20", np.nan),
            "stretch_p80_vs_ema20": row.get("stretch_p80_vs_ema20", np.nan),
            "stretch_p50_vs_ema44": row.get("stretch_p50_vs_ema44", np.nan),
            "stretch_p80_vs_ema44": row.get("stretch_p80_vs_ema44", np.nan),
            "n_events": row.get("n_events", np.nan),
            "reason_long":  reason_long  if decision == "DO NOTHING" else "",
            "reason_short": reason_short if decision == "DO NOTHING" else "",
        })

    if not out_rows:
        raise SystemExit("[ERR] No rows to render.")

    df = pd.DataFrame(out_rows)
    df["__ord"] = np.where(df["decision"].isin(["BUY","SELL"]), 0, 1)
    # Sort BUY/SELL first; then by confidence score desc; fallback strength20
    conf = pd.to_numeric(df.get("confidence_score_0_100", np.nan), errors="coerce").fillna(-1)
    s20  = pd.to_numeric(df.get("strength20_pct", np.nan), errors="coerce").fillna(-1e9)
    df = (
        df.assign(__conf=conf, __s20=s20)
          .sort_values(["__ord", "__conf", "__s20"], ascending=[True, False, False])
          .drop(columns=["__ord","__conf","__s20"])
          .reset_index(drop=True)
    )

    # Render HTML
    now_ts = _ts()
    updated_label = f"{now_ts[:4]}-{now_ts[4:6]}-{now_ts[6:8]} {now_ts[9:11]}:{now_ts[11:13]} UTC"
    html = build_html(df, updated_label, label=label, adx_thr=adx_thr)

    # Label-specific filenames
    out_ts = signals_dir / f"signals_dashboard_{label}_{now_ts}.html"
    out_latest = signals_dir / f"1_signals_dashboard_latest_{label}.html"
    out_ts.write_text(html, encoding="utf-8")
    out_latest.write_text(html, encoding="utf-8")

    print(f"[OK] Dashboard (timestamped) → {out_ts}")
    print(f"[OK] Dashboard (latest)      → {out_latest}")
    print(f"[INFO] Source (s77 CSV)      → {csv_path.name}")
    print(f"[INFO] Meta   (s77)          → {meta_path.name}")
    if g15 is not None:
        print(f"[OK] Gate-1.5 analytics      → merged")
    print(f"[INFO] Enriched dir (s32)    → {enriched_dir}")

    # Serve locally if requested
    if args.serve:
        index_file = out_latest.name
        _serve_local(signals_dir, index_file=index_file, host="127.0.0.1", port=int(args.port))


if __name__ == "__main__":
    main()