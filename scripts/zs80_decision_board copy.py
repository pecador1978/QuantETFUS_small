#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s80_decision_board.py — Operator-facing HTML dashboard for Gate-1 (JSON-driven s76).

Inputs
------
- signals/rule_live_signals_*.csv     ← from s76_rule_signals.py
  (must contain: ticker, decision, strength_score, strength20_pct, strength44_pct,
   rsi14_d, reason_long, reason_short)
- data_enriched/30min/{TICKER}.parquet (from s32) — to compute same-day context:
    close, ema20_d, ema44_d, rsi14_d, volume  (we also read previous close & volume)

Outputs
-------
- signals/signals_dashboard_<ts>.html
- signals/1_signals_dashboard_latest.html   (stable filename, always overwritten)

Columns
-------
- Ticker
- Trade         (BUY / SELL / DO NOTHING)
- Momentum      (BULLISH / ACCUMULATION / DISTRIBUTION / BEARISH)   ← daily timeframe
- Trend         (UP / DOWN)   ← from ema20_d vs ema44_d
- Strength-20   (% vs EMA20_d; color-coded)
- Strength-44   (% vs EMA44_d; color-coded)
- Sentiment     (Risk ON / Risk OFF)
- Smart Money   (Accumulation / Distribution / Markup / Markdown / Neutral)
- RSI           (daily, integer)
- Reason (LONG)  ← s76 reason_long (chips)
- Reason (SHORT) ← s76 reason_short (chips)
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


def _momentum(close_d: float, ema20_d: float, ema44_d: float) -> str:
    """
    Daily timeframe Momentum buckets:
      - BULLISH:       close>ema20 AND ema20>ema44
      - ACCUMULATION:  ema20>ema44 AND close<=ema20
      - DISTRIBUTION:  ema20<=ema44 AND close>ema44
      - BEARISH:       otherwise
    """
    if np.isfinite(close_d) and np.isfinite(ema20_d) and np.isfinite(ema44_d):
        if close_d > ema20_d and ema20_d > ema44_d:
            return "BULLISH"
        if ema20_d > ema44_d and close_d <= ema20_d:
            return "ACCUMULATION"
        if ema20_d <= ema44_d and close_d > ema44_d:
            return "DISTRIBUTION"
        return "BEARISH"
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


def _smart_money_phase(close_now: float, close_prev: float,
                       vol_now: float, vol_prev: float,
                       ema20_d: float) -> str:
    """
    PineScript parity (your snippet):
      priceChange = close - close[1]
      trendDir = close > ema ? 1 : -1
      volRising = volume > volume[1]
      isFlat = abs(priceChange) < (close * 0.002)

      informedMove =
           isFlat and not volRising               ? "Accumulation" :
           isFlat and volRising and trendDir < 0  ? "Distribution" :
           trendDir > 0 and priceChange > 0       ? "Markup" :
           trendDir < 0 and priceChange < 0       ? "Markdown" :
                                                    "Neutral"
    """
    if not (np.isfinite(close_now) and np.isfinite(close_prev) and
            np.isfinite(vol_now) and np.isfinite(vol_prev) and
            np.isfinite(ema20_d)):
        return "Neutral"

    price_change = close_now - close_prev
    trend_dir = 1 if (close_now > ema20_d) else -1
    vol_rising = vol_now > vol_prev
    is_flat = abs(price_change) < (close_now * 0.002)  # 0.2%

    if is_flat and not vol_rising:
        return "Accumulation"
    if is_flat and vol_rising and trend_dir < 0:
        return "Distribution"
    if trend_dir > 0 and price_change > 0:
        return "Markup"
    if trend_dir < 0 and price_change < 0:
        return "Markdown"
    return "Neutral"


def _read_s76_latest(signals_dir: Path) -> tuple[pd.DataFrame, str]:
    latest = _latest(signals_dir / "rule_live_signals_*.csv")
    if not latest:
        raise SystemExit(f"[ERR] No s76 live file found in {signals_dir}")
    df = pd.read_csv(latest)
    if "ticker" not in df.columns or "decision" not in df.columns:
        raise SystemExit(f"[ERR] {latest.name} missing required columns ('ticker','decision').")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df, latest.name


def _read_s32_one(tkr: str, enriched_dir: Path, mkt_tz: str) -> dict:
    p = enriched_dir / f"{tkr}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if df.empty or "datetime" not in df.columns:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    # ensure tz + sort
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df = df.sort_values("datetime")

    # we need last two rows for price/volume delta
    last = df.iloc[-1].copy()
    prev = df.iloc[-2].copy() if len(df) >= 2 else last

    def getf(row, c):
        v = row.get(c, np.nan)
        try:
            return float(v)
        except Exception:
            return np.nan

    out = {
        "close_d": getf(last, "close"),
        "close_prev": getf(prev, "close"),
        "ema20_d": getf(last, "ema20_d"),
        "ema44_d": getf(last, "ema44_d"),
        "rsi14_d": getf(last, "rsi14_d"),
        "vol_d": getf(last, "volume"),
        "vol_prev": getf(prev, "volume"),
    }
    return out


def _reason_chips(reason: str) -> str:
    """Render comma-separated tokens as chips; highlight 'stretched_vs_ema44'."""
    if not reason:
        return ""
    tags = []
    for r in [x.strip() for x in str(reason).split(",") if x.strip()]:
        cls = "tag-warn" if r == "stretched_vs_ema44" else "tag"
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
  background:#0b132d;color:var(--txt);outline:none;margin-bottom:10px}
table{width:100%;border-collapse:collapse;table-layout:auto}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);z-index:2;text-align:left;font-weight:700}
tr:hover td{background:#0c1532}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:700;font-size:11px}
.badge-buy{background:rgba(5,150,105,.15);color:#86efac;border:1px solid rgba(5,150,105,.3)}
.badge-sell{background:rgba(220,38,38,.15);color:#fecaca;border:1px solid rgba(220,38,38,.3)}
.badge-wait{background:rgba(71,85,105,.2);color:#cbd5e1;border:1px solid rgba(71,85,105,.35)}
.mom{font-weight:700}
.trend-up{color:#86efac}
.trend-down{color:#fca5a5}
.sent-on{color:#86efac;font-weight:700}
.sent-off{color:#fca5a5;font-weight:700}
.sm-phase{font-weight:700}
.sm-acc{color:#60a5fa}      /* blue-ish */
.sm-dist{color:#f59e0b}     /* orange */
.sm-markup{color:#22c55e}   /* green */
.sm-markdown{color:#ef4444} /* red */
.sm-neutral{color:#cbd5e1}  /* muted */
.strg{display:inline-block;padding:2px 8px;border-radius:8px;font-weight:800}
.strg-neutral{background:#0f172a;color:#cbd5e1}
.strg-ok{background:#052e1a;color:#86efac}
.strg-warn{background:#3b2a0b;color:#fbbf24}
.strg-hot{background:#3b0a0a;color:#fca5a5}
.reason{display:flex;gap:6px;flex-wrap:wrap;white-space:normal}
.tag{padding:2px 6px;border-radius:6px;font-size:11px;border:1px solid #2b344a;color:#cbd5e1;background:#101a2f}
.tag-warn{border-color:#8a5500;color:#fbbf24;background:#2a1c05}
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
</script>
    """

    # helpers for cells
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

    def sm_phase_cell(phase: str) -> str:
        p = (phase or "").lower()
        cls = "sm-neutral"
        if p == "accumulation": cls = "sm-acc"
        elif p == "distribution": cls = "sm-dist"
        elif p == "markup": cls = "sm-markup"
        elif p == "markdown": cls = "sm-markdown"
        return f'<span class="sm-phase {cls}">{phase}</span>'

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
    <input id="q" class="search" oninput="filterRows()" placeholder="Search ticker, momentum, trend, sentiment, smart money…"/>
    <div style="overflow:auto; max-height:78vh;">
      <table id="tbl">
        <thead>
          <tr>
            <th>TICKER</th>
            <th>TRADE</th>
            <th>MOMENTUM</th>
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
            str(r.get("momentum", "")),
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
            f'<td class="mom">{r.get("momentum","")}</td>'
            f'<td class="{trend_cls(r.get("trend",""))}">{r.get("trend","")}</td>'
            f'<td>{strg_cell(r.get("strength20_pct", None))}</td>'
            f'<td>{strg_cell(r.get("strength44_pct", None))}</td>'
            f'<td class="{sent_cls(r.get("sentiment",""))}">{r.get("sentiment","")}</td>'
            f'<td>{sm_phase_cell(r.get("smartmoney","Neutral"))}</td>'
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

    return "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'/>" + style + "</head><body>" + head + "\n".join(rows_html) + tail + script + "</body></html>"


# ----------------- main -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Limit number of tickers (debug).")
    args = ap.parse_args()

    signals_dir = P.ROOT / "signals"
    enriched_dir = P.DATA_ENRICHED / "30min"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # load latest s76 table (has strengths & reasons)
    s76_df, s76_name = _read_s76_latest(signals_dir)

    # optional limit (debug)
    tickers = s76_df["ticker"].tolist()
    if args.limit and args.limit > 0:
        tickers = tickers[: args.limit]
        s76_df = s76_df[s76_df["ticker"].isin(tickers)]

    mkt_tz = _market_tz()

    out_rows = []
    for _, row in s76_df.sort_values("ticker").iterrows():
        t = str(row["ticker"]).upper()
        decision = str(row.get("decision","")).upper()

        # s32 daily context (+ prev bar for smart money parity)
        dctx = _read_s32_one(t, enriched_dir, mkt_tz)
        close_d     = dctx.get("close_d",  np.nan)
        close_prev  = dctx.get("close_prev", np.nan)
        ema20_d     = dctx.get("ema20_d",  np.nan)
        ema44_d     = dctx.get("ema44_d",  np.nan)
        rsi14_d     = dctx.get("rsi14_d",  row.get("rsi14_d", np.nan))  # prefer s32; fallback s76
        vol_d       = dctx.get("vol_d", np.nan)
        vol_prev    = dctx.get("vol_prev", np.nan)

        # derived context (daily timeframe)
        mom   = _momentum(close_d, ema20_d, ema44_d)
        trn   = _trend(ema20_d, ema44_d)
        sent  = _sentiment(trn, close_d, ema20_d)
        sm    = _smart_money_phase(close_d, close_prev, vol_d, vol_prev, ema20_d)

        # strengths & reasons straight from s76
        str20 = row.get("strength20_pct", np.nan)
        str44 = row.get("strength44_pct", np.nan)
        reason_long  = row.get("reason_long",  "")
        reason_short = row.get("reason_short", "")

        out_rows.append({
            "ticker": t,
            "decision": decision,
            "momentum": mom,
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