#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s76_rule_signals.py — Gate-1 rule-based BUY/SELL signals (no ML, JSON-driven).

Inputs (reads)
-------------
- Enriched 30m parquet per ticker (produced by s32):
    P.DATA_ENRICHED/30min/{TICKER}.parquet
  Expected cols (case-insensitive, subset): datetime, open, high, low, close,
      ema20_d, ema44_d, rsi14_d  (daily overlays computed during s32)

- (Optional) Universe filter (Excel):
    P.CONFIG_DIR/ETF_list.xlsx   (default sheet: "US_small")
  If missing/empty → falls back to processing ALL parquets.

- (Optional) Environment:
    MARKET_TZ  (e.g., "Europe/London") — used for local-day grouping if needed.
    If unset, defaults to "Europe/London".

Outputs (writes)
----------------
- P.ROOT/signals/rule_live_signals_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_top10_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_all_<YYYYMMDD_HHMM>.csv

Columns (primary board)
-----------------------
- decision, side, strength_score (ranking score vs 30m EMA275)
- strength20_pct, strength44_pct (price vs daily EMA20/EMA44 in %; daily last-30m proxy)
- reason_long  (why LONG did not trigger; comma-joined tokens)
- reason_short (why SHORT did not trigger; comma-joined tokens)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, os, argparse, json, re
import pandas as pd
import numpy as np

# ---------- project paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from common.paths import P  # ROOT, DATA_ENRICHED, etc.

# ----- defaults -----
DEFAULT_RULES = {
    "use_ema_buffer": True,
    "buffer_pct": 0.0015,
    "enable_shorts": True,
    "rsi_long_max": 60.0,
    "rsi_short_min": 40.0,
    "require_trend_alignment": True,
    "ema20_gt_ema44_for_long": True,
    "ema20_lt_ema44_for_short": True,
    "ema5_slope_min": 0.001,    # 0.10%
    "ema20_slope_min": 0.0003,  # 0.03%
    "overrides": {}
}

# ---------- ticker normalization ----------
def _norm_ticker(x: str) -> str:
    s = str(x).upper().replace("\u00A0", " ").strip()  # strip NBSP too
    # keep letters, digits, dot, dash, underscore
    return re.sub(r"[^A-Z0-9._-]", "", s)

def _in_universe(stem: str, universe: set[str]) -> bool:
    s = _norm_ticker(stem)              # e.g., 'VUAA.L'
    base = s.split('.', 1)[0]           # 'VUAA'
    base2 = base.replace('-', '').replace('_', '')
    return (s in universe) or (base in universe) or (base2 in universe)

# ---------- Universe from Excel ----------
def _load_universe_from_xlsx(
    path: Path,
    sheet: str = "US_small",
    preferred_cols = ("ticker", "symbol", "etf", "Ticker", "Symbol"),
    verbose: bool = True
) -> set[str]:
    """
    Load allowed tickers from an Excel workbook / sheet.
    Returns NORMALIZED (upper/clean) tickers; empty set on error.
    """
    try:
        if not path.exists():
            print(f"[WARN] Universe file missing: {path}")
            return set()
        df = pd.read_excel(path, sheet_name=sheet)  # needs openpyxl
        if df is None or df.empty:
            print(f"[WARN] Universe sheet empty: {path.name}:{sheet}")
            return set()

        cols_map = {c.strip().lower(): c for c in df.columns}
        chosen = next((cols_map[c.lower()] for c in preferred_cols if c.lower() in cols_map), df.columns[0])

        uni = {_norm_ticker(v) for v in df[chosen].tolist() if pd.notna(v) and str(v).strip()}
        uni -= {"", "NAN", "NONE"}

        if verbose:
            ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"[INFO] Universe: {path.name} (sheet={sheet}, col='{chosen}', mtime={ts}, size={len(uni)})")
            smp = ", ".join(sorted(list(uni))[:12])
            print(f"[INFO] Universe sample: {smp}")
        return uni
    except Exception as e:
        print(f"[WARN] Could not load universe from {path}:{sheet} — {e}")
        return set()

# ----- helpers -----
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

def _yesno(x: bool) -> str:
    return "YES" if bool(x) else "NO"

def _read_rules(path: Path) -> dict:
    if not path.exists():
        return DEFAULT_RULES.copy()
    try:
        cfg = json.loads(path.read_text())
        rules = DEFAULT_RULES.copy()
        rules.update({k: v for k, v in cfg.items() if k in DEFAULT_RULES})
        if "overrides" not in rules or not isinstance(rules["overrides"], dict):
            rules["overrides"] = {}
        return rules
    except Exception as e:
        print(f"[WARN] Could not parse {path.name}: {e}. Using defaults.")
        return DEFAULT_RULES.copy()

def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path.name}: missing required columns (datetime, close)")
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "m30_ema275" not in df.columns:
        df = df.sort_values("datetime").copy()
        df["m30_ema275"] = df["close"].ewm(span=275, adjust=False).mean()
    return (df
            .dropna(subset=["datetime","close"])
            .sort_values("datetime")
            .reset_index(drop=True))

def _latest_daily_lastbars(df: pd.DataFrame, market_tz: str, n_days: int = 3) -> pd.DataFrame:
    local = df["datetime"].dt.tz_convert(market_tz)
    g = df.assign(date_local=local.dt.date).groupby("date_local", as_index=False).tail(1)
    return g.tail(n_days).reset_index(drop=True)

def _daily_relative_slope(today: float, yday: float) -> float | None:
    if np.isfinite(today) and np.isfinite(yday) and yday != 0:
        return float(today / yday - 1.0)
    return None

def _strength_long(m30_close, ema275, rsi14d, min_clip=-0.10, max_clip=0.10):
    above = (m30_close / ema275 - 1.0) if (pd.notna(m30_close) and pd.notna(ema275) and ema275 > 0) else 0.0
    above = float(np.clip(above, min_clip, max_clip))
    rsi_term = 0.01 * ((float(rsi14d) - 50.0) if pd.notna(rsi14d) else 0.0)
    return 2.0 * above + rsi_term

def _strength_short(m30_close, ema275, rsi14d, min_clip=-0.10, max_clip=0.10):
    below = (ema275 / m30_close - 1.0) if (pd.notna(m30_close) and pd.notna(ema275) and m30_close > 0) else 0.0
    below = float(np.clip(below, min_clip, max_clip))
    rsi_term = 0.01 * ((50.0 - float(rsi14d)) if pd.notna(rsi14d) else 0.0)
    return 2.0 * below + rsi_term

# ----- main -----
def main():
    ap = argparse.ArgumentParser(description="Gate-1 rule signals from s32 30m parquet (no ML).")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N tickers (debug).")
    ap.add_argument("--rules-config", type=str, default=str(P.CONFIG_DIR / "gate1_rules.json"),
                    help="JSON file with thresholds/switches.")
    # legacy flags still accepted but JSON rules take precedence when present:
    ap.add_argument("--use-ema-buffer", action="store_true", default=None,
                    help="Use EMA275 +/- buffer (override JSON).")
    ap.add_argument("--buffer-pct", type=float, default=None,
                    help="Buffer percent relative to EMA275 (override JSON).")
    ap.add_argument("--enable-shorts", action="store_true", default=None,
                    help="Allow short signals (override JSON).")
    ap.add_argument("--start-date", type=str, default=None,
                    help="YYYY-MM-DD; ignore signals before this local date.")
    args = ap.parse_args()

    # load rules
    rules_path = Path(args.rules_config)
    R = _read_rules(rules_path)
    if args.use_ema_buffer is not None: R["use_ema_buffer"] = bool(args.use_ema_buffer)
    if args.buffer_pct is not None:     R["buffer_pct"]     = float(args.buffer_pct)
    if args.enable_shorts is not None:  R["enable_shorts"]  = bool(args.enable_shorts)

    mkt_tz = _market_tz()
    m30_dir = P.DATA_ENRICHED / "30min"
    if not m30_dir.exists():
        raise SystemExit(f"[ERR] {m30_dir} not found. Run s32 first.")

    # list parquet files
    parqs = sorted(m30_dir.glob("*.parquet"))
    if not parqs:
        raise SystemExit(f"[ERR] No parquet files in {m30_dir}.")

    # ---------- APPLY EXCEL UNIVERSE ----------
    uni_path = P.CONFIG_DIR / "ETF_list.xlsx"
    universe = _load_universe_from_xlsx(uni_path, sheet="US_small", verbose=True)
    if universe:
        before = len(parqs)
        parqs = [p for p in parqs if _in_universe(p.stem, universe)]
        dropped = before - len(parqs)
        if dropped:
            print(f"[INFO] Universe filter: {before} → {len(parqs)} parquets (dropped {dropped})")
    else:
        print("[WARN] Universe is empty; processing ALL files (fallback).")

    if args.limit:
        parqs = parqs[: args.limit]

    rows = []
    for pq in parqs:
        tkr = pq.stem
        try:
            df = _read_parquet(pq)
            if df.empty:
                continue

            # last 30m bar
            last = df.iloc[-1]
            ema275     = float(last.get("m30_ema275", np.nan))
            m30_close  = float(last.get("close", np.nan))
            rsi14_d    = float(last.get("rsi14_d", np.nan)) if pd.notna(last.get("rsi14_d", np.nan)) else np.nan

            # daily overlays across last local days
            dlast = _latest_daily_lastbars(df, mkt_tz, n_days=3)
            if len(dlast) >= 2:
                today  = dlast.iloc[-1]
                yday   = dlast.iloc[-2]
                ema5_t,  ema5_y  = float(today.get("ema5_d", np.nan)),  float(yday.get("ema5_d", np.nan))
                ema20_t, ema20_y = float(today.get("ema20_d", np.nan)), float(yday.get("ema20_d", np.nan))
                ema44_t           = float(today.get("ema44_d", np.nan))
                # slopes (relative)
                slope5  = _daily_relative_slope(ema5_t,  ema5_y)
                slope20 = _daily_relative_slope(ema20_t, ema20_y)
            else:
                ema5_t = ema20_t = ema44_t = np.nan
                slope5 = slope20 = None

            # start-date gate
            if args.start_date:
                last_local_date = last["datetime"].tz_convert(mkt_tz).date()
                if last_local_date < pd.to_datetime(args.start_date).date():
                    ema275 = np.nan  # force no-entry

            # EMA275 buffer lines
            if R["use_ema_buffer"] and pd.notna(ema275):
                buf = ema275 * float(R["buffer_pct"])
                above_line = ema275 + buf
                below_line = ema275 - buf
            else:
                above_line = ema275
                below_line = ema275

            # prior two daily closes from 30m stream (yday & two-days-ago)
            if len(dlast) >= 3:
                day2_close = float(dlast.iloc[-3]["close"])
                day1_close = float(dlast.iloc[-2]["close"])
            elif len(dlast) == 2:
                day2_close = np.nan
                day1_close = float(dlast.iloc[-2]["close"])
            else:
                day1_close = day2_close = np.nan

            # overrides
            ov = (R.get("overrides") or {}).get(_norm_ticker(tkr), {})
            ema5_slope_min  = float(ov.get("ema5_slope_min",  R["ema5_slope_min"]))
            ema20_slope_min = float(ov.get("ema20_slope_min", R["ema20_slope_min"]))

            # core gates
            long_above_ema275 = (pd.notna(day1_close) and pd.notna(day2_close) and pd.notna(above_line) and
                                 (day1_close > above_line) and (day2_close > above_line))
            short_below_ema275 = (pd.notna(day1_close) and pd.notna(day2_close) and pd.notna(below_line) and
                                  (day1_close < below_line) and (day2_close < below_line))

            rsi_ok_long  = (pd.notna(rsi14_d) and (rsi14_d < float(R["rsi_long_max"])))
            rsi_ok_short = (pd.notna(rsi14_d) and (rsi14_d > float(R["rsi_short_min"])))

            trend_long_ok  = True
            trend_short_ok = True
            if R.get("require_trend_alignment", True):
                if R.get("ema20_gt_ema44_for_long", True):
                    trend_long_ok = (pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t > ema44_t))
                if R.get("ema20_lt_ema44_for_short", True):
                    trend_short_ok = (pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t < ema44_t))

            # slope filters (relative)
            if slope5 is None or slope20 is None:
                slope_long_ok = False
                slope_short_ok = False
            else:
                slope_long_ok  = (slope5  >= ema5_slope_min)  and (slope20 >= ema20_slope_min)
                slope_short_ok = (slope5  <= -ema5_slope_min) and (slope20 <= -ema20_slope_min)

            # combine
            long_ok = long_above_ema275 and rsi_ok_long and trend_long_ok and slope_long_ok
            short_ok = False
            if bool(R.get("enable_shorts", True)):
                short_ok = short_below_ema275 and rsi_ok_short and trend_short_ok and slope_short_ok

            # ranking strengths vs 30m EMA275 (keep as-is)
            s_long  = _strength_long(m30_close, ema275, rsi14_d)
            s_short = _strength_short(m30_close, ema275, rsi14_d)

            # DAILY display strengths (use today's last-30m close)
            daily_close_t = float(today.get("close", np.nan)) if 'today' in locals() else np.nan
            strength20_pct = ((daily_close_t / ema20_t - 1.0) * 100.0
                              if (np.isfinite(daily_close_t) and np.isfinite(ema20_t) and ema20_t > 0)
                              else np.nan)
            strength44_pct = ((daily_close_t / ema44_t - 1.0) * 100.0
                              if (np.isfinite(daily_close_t) and np.isfinite(ema44_t) and ema44_t > 0)
                              else np.nan)

            # decision
            decision = "DO NOTHING"
            decision_side = ""
            strength = 0.0
            if long_ok and (not short_ok or s_long >= s_short):
                decision = "BUY";  decision_side = "LONG";  strength = s_long
            elif short_ok:
                decision = "SELL"; decision_side = "SHORT"; strength = s_short

            # reasons (split per side)
            long_reasons = []
            if not long_above_ema275: long_reasons.append("not_above_ema275_2d")
            if not rsi_ok_long:       long_reasons.append("rsi_long_guard")
            if not trend_long_ok:     long_reasons.append("trend_align_long")
            if not slope_long_ok:     long_reasons.append("slope_long")
            if np.isfinite(strength44_pct) and strength44_pct > 10.0:
                long_reasons.append("stretched_vs_ema44")

            short_reasons = []
            if bool(R.get("enable_shorts", True)):
                if not short_below_ema275: short_reasons.append("not_below_ema275_2d")
                if not rsi_ok_short:       short_reasons.append("rsi_short_guard")
                if not trend_short_ok:     short_reasons.append("trend_align_short")
                if not slope_short_ok:     short_reasons.append("slope_short")

            reason_long  = ",".join(dict.fromkeys(long_reasons))
            reason_short = ",".join(dict.fromkeys(short_reasons))

            rows.append({
                "ticker": _norm_ticker(tkr).split('.', 1)[0],
                "decision": decision,
                "side": decision_side,
                "strength_score": round(float(strength), 6),
                "m30_close": m30_close,
                "m30_ema275": ema275,
                "day1_close": day1_close,
                "day2_close": day2_close,
                "rsi14_d": rsi14_d,
                "strength20_pct": strength20_pct,
                "strength44_pct": strength44_pct,
                "reason_long":  reason_long   if decision == "DO NOTHING" else "",
                "reason_short": reason_short  if decision == "DO NOTHING" else "",
                "long_above_ema275": _yesno(long_above_ema275),
                "short_below_ema275": _yesno(short_below_ema275),
                "trend_long_ok": _yesno(trend_long_ok),
                "trend_short_ok": _yesno(trend_short_ok),
                "slope_long_ok": _yesno(slope_long_ok),
                "slope_short_ok": _yesno(slope_short_ok),
            })
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")

    if not rows:
        raise SystemExit("[ERR] No rows to decide.]".replace("]", ""))  # tidy message

    X = pd.DataFrame(rows)

    # ----- Outputs -----
    out_dir = P.ROOT / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    board_csv = out_dir / f"rule_live_signals_{ts}.csv"
    (X.sort_values(["decision", "strength_score"], ascending=[True, False])
       .to_csv(board_csv, index=False))
    print(f"[OK] Rule live signals (full board) → {board_csv}")

    op = X[X["decision"].isin(["BUY","SELL"])].copy().sort_values("strength_score", ascending=False)
    short_csv = out_dir / f"operator_today_RULE_{ts}.csv"
    op[["ticker","decision","side","strength_score"]].to_csv(short_csv, index=False)
    print(f"[OK] Operator shortlist (BUY/SELL) → {short_csv}  (rows={len(op)})")

    top10 = op.head(10).copy()
    top10_csv = out_dir / f"operator_today_RULE_top10_{ts}.csv"
    top10.to_csv(top10_csv, index=False)
    print(f"[OK] Operator Top-10 (RULE) → {top10_csv}")

    X["decision_order"] = np.where(X["decision"].isin(["BUY","SELL"]), 0, 1)
    all_csv = out_dir / f"operator_today_RULE_all_{ts}.csv"
    (X.sort_values(["decision_order","strength_score"], ascending=[True, False])
       .drop(columns=["decision_order"])
       .to_csv(all_csv, index=False))
    print(f"[OK] Operator ALL (sorted) → {all_csv}")

    print("\n[Decision counts]")
    print(X["decision"].value_counts(dropna=False).to_string())

    if len(top10):
        print("\n[Top-10]")
        print(top10[["ticker","decision","side","strength_score"]].to_string(index=False))

if __name__ == "__main__":
    main()