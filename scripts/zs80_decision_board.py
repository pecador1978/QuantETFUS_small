#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s80_decision_board.py — Operator-facing HTML dashboard for Gate-1 (JSON-driven s76).

Inputs
------
- signals/rule_live_signals_*.csv     ← from s76_rule_signals.py
  Must contain at minimum: ticker, decision, strength20_pct, strength44_pct, rsi14_d.
  Optional (preferred): reason_long, reason_short.
  Backward-compat: if only 'reason' exists, we split into long/short buckets heuristically.

- data_enriched/30min/{TICKER}.parquet (from s32) — used to compute DAILY context:
    close (daily proxy from last 30m), ema20_d, ema44_d, rsi14_d, volume (optional)

Outputs
-------
- signals/signals_dashboard_<ts>.html
- signals/1_signals_dashboard_latest.html   (stable filename, always overwritten)

Columns
-------
- Ticker
- Trade         (BUY / SELL / DO NOTHING)
- Trend Phase   (BULLISH / NEUTRAL / BEARISH)  ← daily
- Trend         (UP / DOWN)                     ← EMA20 vs EMA44 (daily)
- Strength-20   (% vs EMA20_d; color-coded)
- Strength-44   (% vs EMA44_d; color-coded)
- Sentiment     (Risk ON / Risk OFF)
- Smart Money   (Accumulation / Distribution / Markup / Markdown / Neutral)
- RSI           (daily, integer)
- Reason (LONG)  ← from s76 (preferred) or derived from legacy 'reason'
- Reason (SHORT) ← from s76 (preferred) or derived from legacy 'reason'
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os
import pandas as pd
import numpy as np

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


def _latest(pattern: Path) -> Path | None:
    files = sorted(pattern.parent.glob(pattern.name))
    return files[-1] if files else None


def _trend_phase(close_d: float, ema20_d: float, ema44_d: float) -> str:
    """
    Daily Trend Phase (simple, non-overlapping):
      - BULLISH:  EMA20 > EMA44 and Close >= EMA20
      - BEARISH:  EMA20 < EMA44 and Close <= EMA44
      - NEUTRAL:  anything else (transition/flattening)
    """
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
    # Risk ON if trend is UP and price is above EMA20; otherwise Risk OFF
    if trend == "UP" and np.isfinite(close_d) and np.isfinite(ema20_d) and close_d >= ema20_d:
        return "Risk ON"
    return "Risk OFF"


def _smart_money_wyckoff(close_t: float, close_y: float,
                         vol_t: float | None, vol_y: float | None,
                         ema20_t: float) -> str:
    """
    Wyckoff-flavored 5-state classification on daily proxy bars:
      - priceChange = close_t - close_y
      - trendDir    =  1 if close_t > ema20_t else -1
      - volRising   =  True if vol_t > vol_y (if volume present for both days)
      - isFlat      =  abs(priceChange) < (close_t * 0.002)  # 0.2% of price

      Rules:
        Accumulation : isFlat and NOT volRising
        Distribution : isFlat and volRising and trendDir < 0
        Markup       : trendDir > 0 and priceChange > 0
        Markdown     : trendDir < 0 and priceChange < 0
        Neutral      : otherwise

      If volume missing, we degrade gracefully:
        - Without volume: only Markup/Markdown/Neutral are emitted (flat=Neutral).
    """
    # Basic guards
    if not (np.isfinite(close_t) and np.isfinite(close_y) and np.isfinite(ema20_t)):
        return "Neutral"

    price_change = float(close_t - close_y)
    trend_dir = 1 if close_t > float(ema20_t) else -1
    flat_thresh = abs(close_t) * 0.002  # 0.2%
    is_flat = abs(price_change) < flat_thresh

    # If we have usable volume both days, use accumulation/distribution logic:
    vol_ok = np.isfinite(vol_t) and np.isfinite(vol_y)
    vol_rising = (float(vol_t) > float(vol_y)) if vol_ok else None

    if is_flat and vol_ok:
        if not vol_rising:
            return "Accumulation"
        if vol_rising and trend_dir < 0:
            return "Distribution"

    # Directional
    if trend_dir > 0 and price_change > 0:
        return "Markup"
    if trend_dir < 0 and price_change < 0:
        return "Markdown"

    return "Neutral"


def _read_s76_latest(signals_dir: Path) -> tuple[pd.DataFrame, str]:
    # pick latest CSV that does NOT have a sibling .metadata.json (those are s77)
    cands = sorted(signals_dir.glob("rule_live_signals_*.csv"))
    pick = None
    for csv in reversed(cands):
        meta = signals_dir / (csv.stem + ".metadata.json")
        if not meta.exists():
            pick = csv
            break
    if pick is None:
        # fallback to absolute latest if none found (keeps previous behavior)
        pick = cands[-1] if cands else None
    if not pick:
        raise SystemExit(f"[ERR] No s76 live file found in {signals_dir}")
    df = pd.read_csv(pick)
    if "ticker" not in df.columns or "decision" not in df.columns:
        raise SystemExit(f"[ERR] {pick.name} missing required columns ('ticker','decision').")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df, pick.name


def _read_daily_ctx(tkr: str, enriched_dir: Path, mkt_tz: str) -> dict:
    """
    Build true daily bars from 30m data:
      - day_close: last close of the local day
      - day_vol  : sum of volumes over the local day
      - ema20_d / ema44_d / rsi14_d: taken from the day's last 30m row
    Returns fields for today (t) and yesterday (y).
    """
    p = enriched_dir / f"{tkr}.parquet"
    if not p.exists():
        return {}

    df = pd.read_parquet(p)
    if df.empty or "datetime" not in df.columns:
        return {}

    # normalize
    df.columns = [c.strip().lower() for c in df.columns]
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df = df.sort_values("datetime")

    # make sure we have numeric volume (may be missing in some sources)
    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # group by local day
    local = df["datetime"].dt.tz_convert(mkt_tz)
    df = df.assign(date_local=local.dt.date)

    # daily sums for volume
    day_vol = df.groupby("date_local", as_index=False)["volume"].sum(numeric_only=True).rename(columns={"volume":"day_vol"})

    # last row per local day for close & daily overlays
    day_last = df.groupby("date_local", as_index=False).tail(1).reset_index(drop=True)

    # merge to get both last-row fields and day volume
    daily = pd.merge(day_last, day_vol, on="date_local", how="left")

    if daily.empty:
        return {}

    def getf(row, c):
        v = row.get(c, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    today = daily.iloc[-1]
    yday  = daily.iloc[-2] if len(daily) >= 2 else None

    out = {
        "close_t":  getf(today, "close"),
        "ema20_t":  getf(today, "ema20_d"),
        "ema44_t":  getf(today, "ema44_d"),
        "rsi14_t":  getf(today, "rsi14_d"),
        "day_vol_t": getf(today, "day_vol"),
        "close_y":  getf(yday, "close")   if yday is not None else np.nan,
        "day_vol_y": getf(yday, "day_vol") if yday is not None else np.nan,
    }
    return out
def _reason_split_from_legacy(legacy: str) -> tuple[str, str]:
    """
    Split a legacy comma-joined 'reason' into (reason_long, reason_short).
    Tokens routed by side:
      LONG  : not_above_ema275_2d, rsi_long_guard, trend_align_long, slope_long
      SHORT : not_below_ema275_2d, rsi_short_guard, trend_align_short, slope_short
    'stretched_vs_ema44' is kept on the LONG side (it blocks chasing longs).
    Unknown tokens are ignored.
    """
    toks = [t.strip() for t in (legacy or "").split(",") if t.strip()]
    long_set  = {"not_above_ema275_2d", "rsi_long_guard", "trend_align_long", "slope_long"}
    short_set = {"not_below_ema275_2d", "rsi_short_guard", "trend_align_short", "slope_short"}

    rl, rs = [], []
    for t in toks:
        if t in long_set:
            if t not in rl: rl.append(t)
        elif t in short_set:
            if t not in rs: rs.append(t)
        elif t == "stretched_vs_ema44":
            if t not in rl: rl.append(t)

    return ",".join(rl), ",".join(rs)

def _reason_chips(reason: str) -> str:
    """Render comma-separated tokens as chips; highlight 'stretched_vs_ema44'."""
    if not reason:
        return ""
    tags = []
    for r in [x.strip() for x in str(reason).split(",") if x.strip()]:
        cls = "tag tag-warn" if r == "stretched_vs_ema44" else "tag"
        tags.append(f'<span class="{cls}">{r}</span>')
    return '<div class="reason">' + "".join(tags) + "</div>"


# ----------------- HTML builder -----------------
def build_html(df: pd.DataFrame, updated_label: str) -> str:
    style = """
<style>
:root{
  --bg:#0b1020; --card:#0f172a; --muted:#94a3b8; --txt:#e2e8f0; --line:#1f2937;
  --pill-buy:#059669; --pill-sell:#dc2626; --pill-wait:#475569;
  --green:#16a34a; --orange:#f59e0b; --red:#ef4444; --neutral:#334155;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);font-family:ui-sans-serif,system-ui,Segoe UI,Helvetica,Arial;color:var(--txt)}
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
.trend-up{color:#86efac}
.trend-down{color:#fca5a5}
.sent-on{color:#86efac;font-weight:700}
.sent-off{color:#fca5a5;font-weight:700}
.sm-acc{color:#86efac}       /* accumulation (neutral-good) */
.sm-dist{color:#f59e0b}      /* distribution (neutral-warn) */
.sm-markup{color:#86efac;font-weight:700}
.sm-markdown{color:#ef4444;font-weight:700}
.strg{display:inline-block;padding:2px 8px;border-radius:8px;font-weight:800}
.strg-neutral{background:#0f172a;color:#cbd5e1}
.strg-ok{background:#052e1a;color:#86efac}
.strg-warn{background:#3b2a0b;color:#fbbf24}
.strg-hot{background:#3b0a0a;color:#fca5a5}
.reason{display:flex;gap:6px;flex-wrap:wrap;white-space:normal}
.tag{padding:2px 6px;border-radius:6px;font-size:11px;border:1px solid #2b344a;color:#cbd5e1;background:#101a2f}
.tag-warn{border-color:#8a5500;color:#fbbf24;background:#2a1c05}

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
// --- client-side search ---
function filterRows(){
  const q = document.getElementById('q').value.toLowerCase();
  const rows = document.querySelectorAll('#tbl tbody tr');
  rows.forEach(r=>{
    const txt = r.getAttribute('data-f').toLowerCase();
    r.style.display = txt.indexOf(q) >= 0 ? '' : 'none';
  });
}

// --- click-to-sort (auto-detect text/number/percent) ---
(function(){
  function parseCell(txt){
    if (txt == null) return {val: txt, type:'text'};
    const s = txt.trim();
    // handle "<span ...>X</span>" by getting textContent in runtime
    const pct = s.replace(/[,%]/g,'');
    if (!isNaN(parseFloat(pct)) && s.indexOf('%')>=0) return {val: parseFloat(pct), type:'num'};
    const num = s.replace(/[^0-9.+-]/g,'');
    if (num && !isNaN(parseFloat(num)) && /[0-9.+-]/.test(s)) return {val: parseFloat(num), type:'num'};
    return {val: s.toLowerCase(), type:'text'};
  }
  function getCellText(td){
    // Prefer innerText/textContent over innerHTML
    return (td.textContent || '').trim();
  }
  function sortTable(table, colIndex, asc){
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr')).filter(r=>r.style.display!=='none');
    rows.sort((a,b)=>{
      const A = parseCell(getCellText(a.cells[colIndex]));
      const B = parseCell(getCellText(b.cells[colIndex]));
      if (A.type==='num' && B.type==='num'){
        return asc ? (A.val - B.val) : (B.val - A.val);
      }
      return asc ? String(A.val).localeCompare(String(B.val)) : String(B.val).localeCompare(String(A.val));
    });
    rows.forEach(r=>tbody.appendChild(r));
  }
  document.addEventListener('click', function(e){
    const th = e.target.closest('th');
    if (!th) return;
    const table = document.getElementById('tbl');
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const asc = !(th.dataset.asc === 'true');
    th.dataset.asc = String(asc);
    sortTable(table, idx, asc);
  });
})();
</script>
    """

    def pill(decision: str) -> str:
        d = (decision or "").upper()
        if d == "BUY":
            return '<span class="badge badge-buy">BUY</span>'
        if d == "SELL":
            return '<span class="badge badge-sell">SELL</span>'
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

    def strg_cell(x) -> str:
        if x is None or not np.isfinite(x):
            return '<span class="strg strg-neutral">—</span>'
        p = float(x)
        label = f"{p:.2f}%"
        if p <= 0:   cls = "strg-neutral"
        elif p < 5:  cls = "strg-ok"
        elif p < 10: cls = "strg-warn"
        else:        cls = "strg-hot"
        return f'<span class="strg {cls}">{label}</span>'

    head = """
<div class="wrap">
  <h1>Signals Dashboard — Rule Decisions</h1>
  <div class="info">Updated: """ + updated_label + """</div>
  <div class="card">
    <input id="q" class="search" oninput="filterRows()" placeholder="Search ticker, phase, trend, sentiment, smart money…"/>
    <div style="overflow:auto; max-height:78vh;">
      <table id="tbl">
        <thead>
          <tr>
            <th>TICKER</th>
            <th>TRADE</th>
            <th>TREND PHASE</th>
            <th>TREND</th>
            <th>STRENGTH-20</th>
            <th>STRENGTH-44</th>
            <th>SENTIMENT</th>
            <th>SMART MONEY</th>
            <th>RSI</th>
            <th>REASON (LONG)</th>
            <th>REASON (SHORT)</th>
          </tr>
        </thead>
        <tbody>
    """

    rows_html = []
    for _, r in df.iterrows():
        data_filter = " ".join([
            str(r.get("ticker", "")),
            str(r.get("decision", "")),
            str(r.get("trend_phase", "")),
            str(r.get("trend", "")),
            str(r.get("sentiment", "")),
            str(r.get("smartmoney", "")),
            str(r.get("reason_long", "")),
            str(r.get("reason_short", "")),
        ])

        rows_html.append(
            f'<tr data-f="{data_filter}">'
            f'<td>{r.get("ticker","")}</td>'
            f'<td>{pill(r.get("decision",""))}</td>'
            f'<td>{r.get("trend_phase","")}</td>'
            f'<td class="{trend_cls(r.get("trend",""))}">{r.get("trend","")}</td>'
            f'<td>{strg_cell(r.get("strength20_pct", None))}</td>'
            f'<td>{strg_cell(r.get("strength44_pct", None))}</td>'
            f'<td class="{sent_cls(r.get("sentiment",""))}">{r.get("sentiment","")}</td>'
            f'<td class="{sm_cls(r.get("smartmoney",""))}">{r.get("smartmoney","")}</td>'
            f'<td>{"" if not np.isfinite(r.get("rsi14_d", np.nan)) else str(int(round(r.get("rsi14_d", 0))))}</td>'
            f'<td>{_reason_chips(r.get("reason_long",""))}</td>'
            f'<td>{_reason_chips(r.get("reason_short",""))}</td>'
            f'</tr>'
        )

    tail = """
        </tbody>
      </table>
    </div>
  </div>
</div>
    """

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
      <h3>Smart Money (Wyckoff-style, daily proxy)</h3>
      <ul>
        <li><b>Accumulation</b>: |ΔPrice| &lt; 0.2% of Close and Volume not rising</li>
        <li><b>Distribution</b>: |ΔPrice| &lt; 0.2% and Volume rising and Close &lt; EMA20</li>
        <li><b>Markup</b>: Close &gt; EMA20 and ΔPrice &gt; 0</li>
        <li><b>Markdown</b>: Close &lt; EMA20 and ΔPrice &lt; 0</li>
        <li><b>Neutral</b>: Otherwise (or if volume missing)</li>
      </ul>
    </div>
    <div class="box">
      <h3>Sentiment</h3>
      <ul>
        <li><b>Risk ON</b>: Trend=UP and Close ≥ EMA20</li>
        <li><b>Risk OFF</b>: Otherwise</li>
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
      <h3>Reason Tokens</h3>
      <ul>
        <li><code>not_above_ema275_2d</code>: last 2 local days not above 30m EMA275 (+buffer) — long gate</li>
        <li><code>not_below_ema275_2d</code>: last 2 local days not below 30m EMA275 — short gate</li>
        <li><code>rsi_long_guard</code> / <code>rsi_short_guard</code>: RSI guard failed</li>
        <li><code>trend_align_long</code> / <code>trend_align_short</code>: EMA20 vs EMA44 alignment failed</li>
        <li><code>slope_long</code> / <code>slope_short</code>: EMA5/EMA20 daily slopes insufficient</li>
        <li><code>stretched_vs_ema44</code>: Close &gt; 10% above EMA44</li>
      </ul>
    </div>
  </div>
</section>
"""

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
        + style +
        "</head><body>" + head + "\n".join(rows_html) + tail + defs + script + "</body></html>"
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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (debug).")
    args = ap.parse_args()

    signals_dir = P.ROOT / "signals"
    enriched_dir = P.DATA_ENRICHED / "30min"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # load latest s76 table
    s76_df, s76_name = _read_s76_latest(signals_dir)

    # filter by Excel universe
    universe = _load_universe_from_xlsx(P.CONFIG_DIR / "ETF_list.xlsx", sheet="US_small")
    if universe:
        s76_df["ticker"] = s76_df["ticker"].astype(str).str.upper()
        s76_df = s76_df[s76_df["ticker"].isin(universe)]
    else:
        print("[WARN] Universe is empty; dashboard will include all rows from s76 (fallback).")

    # optional limit (debug)
    if args.limit and args.limit > 0:
        keep = s76_df["ticker"].astype(str).str.upper().unique().tolist()[: args.limit]
        s76_df = s76_df[s76_df["ticker"].isin(keep)]

    mkt_tz = _market_tz()

    out_rows = []
    for _, row in s76_df.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision","")).upper()

        # DAILY context from s32 (using daily proxy bars from 30m)
        dctx = _read_daily_ctx(t, enriched_dir, mkt_tz)
        close_t = dctx.get("close_t",  np.nan)
        close_y = dctx.get("close_y",  np.nan)
        ema20_t = dctx.get("ema20_t",  np.nan)
        ema44_t = dctx.get("ema44_t",  np.nan)
        rsi14_d = dctx.get("rsi14_t",  row.get("rsi14_d", np.nan))  # prefer s32; fallback s76
        vol_t   = dctx.get("day_vol_t", np.nan)   # <-- daily SUM
        vol_y   = dctx.get("day_vol_y", np.nan)   # <-- daily SUM

        phase = _trend_phase(close_t, ema20_t, ema44_t)
        trn   = _trend(ema20_t, ema44_t)
        sent  = _sentiment(trn, close_t, ema20_t)
        sm    = _smart_money_wyckoff(close_t, close_y, vol_t, vol_y, ema20_t)

        # strengths from s76 (already percentages)
        str20 = row.get("strength20_pct", np.nan)
        str44 = row.get("strength44_pct", np.nan)

        # reasons (prefer explicit long/short columns; else derive from legacy 'reason')
        reason_long  = row.get("reason_long",  "")
        reason_short = row.get("reason_short", "")
        # normalize NaNs to empty strings
        if isinstance(reason_long, float) and np.isnan(reason_long):   reason_long  = ""
        if isinstance(reason_short, float) and np.isnan(reason_short): reason_short = ""

        if (reason_long == "") and (reason_short == ""):
            legacy = row.get("reason", "")
            if isinstance(legacy, float) and np.isnan(legacy): legacy = ""
            rl, rs = _reason_split_from_legacy(legacy)
            reason_long, reason_short = rl, rs

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
            "reason_long":  reason_long  if decision == "DO NOTHING" else "",
            "reason_short": reason_short if decision == "DO NOTHING" else "",
        })

    if not out_rows:
        raise SystemExit("[ERR] No rows to render.")

    df = pd.DataFrame(out_rows)

    # order & sort: BUY/SELL first then by strength-20 desc (fallback to -inf)
    df["__ord"] = np.where(df["decision"].isin(["BUY","SELL"]), 0, 1)
    df["__s20"] = pd.to_numeric(df["strength20_pct"], errors="coerce").fillna(-1e9)
    df = (
        df.sort_values(["__ord", "__s20"], ascending=[True, False])
          .drop(columns=["__ord", "__s20"])
          .reset_index(drop=True)
    )

    # render HTML
    ts = _ts()
    updated_label = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]} UTC"

    html = build_html(df, updated_label)

    out_ts = signals_dir / f"signals_dashboard_{ts}.html"
    out_latest = signals_dir / "1_signals_dashboard_latest.html"
    out_ts.write_text(html, encoding="utf-8")
    out_latest.write_text(html, encoding="utf-8")

    print(f"[OK] Dashboard (timestamped) → {out_ts}")
    print(f"[OK] Dashboard (latest)      → {out_latest}")
    print(f"[INFO] Source (s76)          → {s76_name}")
    print(f"[INFO] Enriched dir (s32)    → {enriched_dir}")

if __name__ == "__main__":
    main()