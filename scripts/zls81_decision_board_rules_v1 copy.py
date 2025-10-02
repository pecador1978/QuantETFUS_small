#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s81_decision_board_rules_v1.py — Operator-facing HTML dashboard for Gate-1 (JSON-driven s77, label-aware).

New features:
- --serve starts a local HTTP server on 127.0.0.1 (default port 8080)
- Label-specific output => distinct URLs:
    signals/1_signals_dashboard_latest_<label>.html  -> http://127.0.0.1:8080/1_signals_dashboard_latest_<label>.html
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
.adx-strong { color:#86efac; font-weight:700 }  /* green */
.adx-weak   { color:#f87171; font-weight:700 }  /* red   */
.adx-neutral{ color:#cbd5e1; }                  /* gray  */
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
function filterRows(){
  const q = document.getElementById('q').value.toLowerCase();
  const rows = document.querySelectorAll('#tbl tbody tr');
  rows.forEach(r=>{
    const txt = r.getAttribute('data-f').toLowerCase();
    r.style.display = txt.indexOf(q) >= 0 ? '' : 'none';
  });
}
(function(){
  function parseCell(txt){
    if (txt == null) return {val: txt, type:'text'};
    const s = txt.trim();
    const pct = s.replace(/[,%]/g,'');
    if (!isNaN(parseFloat(pct)) && s.indexOf('%')>=0) return {val: parseFloat(pct), type:'num'};
    const num = s.replace(/[^0-9.+-]/g,'');
    if (num && !isNaN(parseFloat(num)) && /[0-9.+-]/.test(s)) return {val: parseFloat(num), type:'num'};
    return {val: s.toLowerCase(), type:'text'};
  }
  function getCellText(td){ return (td.textContent || '').trim(); }
  function sortTable(table, colIndex, asc){
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr')).filter(r=>r.style.display!=='none');
    rows.sort((a,b)=>{
      const A = parseCell(getCellText(a.cells[colIndex]));
      const B = parseCell(getCellText(b.cells[colIndex]));
      if (A.type==='num' && B.type==='num'){ return asc ? (A.val - B.val) : (B.val - A.val); }
      return asc ? String(A.val).localeCompare(String(B.val)) : String(B.val).localeCompare(String(A.val));
    });
    rows.forEach(r=>tbody.appendChild(r));
  }
  document.addEventListener('click', function(e){
    const th = e.target.closest('th'); if (!th) return;
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

    head = (
        "<div class=\"wrap\">"
        f"<h1>Signals Dashboard — Rule Decisions <span style='font-size:14px;color:#94a3b8'>(label: {label})</span></h1>"
        f"<div class=\"info\">Updated: {updated_label}</div>"
        "<div class=\"card\">"
        "<input id=\"q\" class=\"search\" oninput=\"filterRows()\" placeholder=\"Search ticker, phase, trend, sentiment, smart money…\"/>"
        "<div style=\"overflow:auto; max-height:78vh;\">"
        "<table id=\"tbl\">"
        "<thead><tr>"
        "<th>TICKER</th><th>TRADE</th><th>TREND PHASE</th><th>TREND</th>"
        "<th>STRENGTH-20</th><th>STRENGTH-44</th><th>SENTIMENT</th><th>SMART MONEY</th>"
        "<th>RSI</th><th>ADX</th><th>REASON (LONG)</th><th>REASON (SHORT)</th>"
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

        # NEW: ADX value
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
            f'<td>{adx_cell(adx_val, adx_thr)}</td>'  # <-- NEW ADX cell
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
      <h3>Donchian (don)</h3>
        <ul>
        <li><b>Upper</b>: highest high of last <code>N</code> days (ex-today)</li>
        <li><b>Lower</b>: lowest low of last <code>N</code> days (ex-today)</li>
        <li><b>Width&nbsp;%</b>: size of channel ((Upper/Lower − 1) × 100)</li>
        <li><b>Location&nbsp;%</b>: close’s spot between Lower (0) and Upper (100)</li>
        <li><b>Caps</b>: cut-offs near channel edges (e.g., 95% / 5%)</li>
        <li><code>donchian_width_thin</code>: channel too narrow</li>
        <li><code>donchian_long_extreme</code>: close too near Upper (above cap)</li>
        <li><code>donchian_short_extreme</code>: close too near Lower (below cap)</li>
        <li><code>donchian_breakout_long</code>: breakout not met (close ≤ Upper × (1+margin))</li>
        <li><code>donchian_breakout_short</code>: breakout not met (close ≥ Lower × (1−margin))</li>
        </ul>
        </li>
      </ul>
    </div>
    <div class="box">
      <h3>ADX / DI</h3>
      <ul>
        <li><b>ADX</b>: trend strength (no direction). Min thresholds: <code>adx_min_long</code>, <code>adx_min_short</code></li>
        <li><b>+DI / −DI</b>: directional movement balance</li>
        <li><b>Tokens</b>:
          <ul>
            <li><code>adx_long_guard</code> / <code>adx_short_guard</code>: ADX below min</li>
            <li><code>adx_long_not_rising</code> / <code>adx_short_not_rising</code>: ΔADX below min when rising required</li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="box">
      <h3>Other Reason Tokens</h3>
      <ul>
        <li><code>rsi_long_guard</code> / <code>rsi_short_guard</code></li>
        <li><code>trend_align_long</code> / <code>trend_align_short</code></li>
        <li><code>slope_long</code> / <code>slope_short</code></li>
        <li><code>base_slope_long</code> / <code>base_slope_short</code></li>
        <li><code>stretched_vs_ema44</code>: Close &gt; 10% above EMA44</li>
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
    + defs          # <-- add this line
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
    # Change working dir so the server roots at signals_dir
    os.chdir(str(signals_dir))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/{index_file}"
        print(f"[SERVE] Local server at {url}")
        try:
            webbrowser.open(url)  # open default browser
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

    # Build rows
    out_rows = []
    for _, row in df_sig.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision","")).upper()

        dctx = _read_daily_ctx(t, enriched_dir, mkt_tz)
        close_t = dctx.get("close_t",  np.nan)
        close_y = dctx.get("close_y",  np.nan)
        ema20_t = dctx.get("ema20_t",  np.nan)
        ema44_t = dctx.get("ema44_t",  np.nan)
        rsi14_d = dctx.get("rsi14_t",  row.get("rsi14_d", np.nan))
        adx14_d = row.get("adx14_d", np.nan)
        vol_t   = dctx.get("day_vol_t", np.nan)
        vol_y   = dctx.get("day_vol_y", np.nan)

        phase = _trend_phase(close_t, ema20_t, ema44_t)
        trn   = _trend(ema20_t, ema44_t)
        sent  = _sentiment(trn, close_t, ema20_t)
        sm    = _smart_money_wyckoff(close_t, close_y, vol_t, vol_y, ema20_t)

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
            "reason_long":  reason_long  if decision == "DO NOTHING" else "",
            "reason_short": reason_short if decision == "DO NOTHING" else "",
        })

    if not out_rows:
        raise SystemExit("[ERR] No rows to render.")

    df = pd.DataFrame(out_rows)
    df["__ord"] = np.where(df["decision"].isin(["BUY","SELL"]), 0, 1)
    df["__s20"] = pd.to_numeric(df["strength20_pct"], errors="coerce").fillna(-1e9)
    df = (
        df.sort_values(["__ord", "__s20"], ascending=[True, False])
          .drop(columns=["__ord", "__s20"])
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
    print(f"[INFO] Enriched dir (s32)    → {enriched_dir}")

    # Serve locally if requested
    if args.serve:
        index_file = out_latest.name  # e.g., 1_signals_dashboard_latest_gate1_relaxed.html
        _serve_local(signals_dir, index_file=index_file, host="127.0.0.1", port=int(args.port))


if __name__ == "__main__":
    main()