#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s77_rules_signals_v1.py — Gate-1 (RELAXED) rule-based BUY/SELL signals (no ML, JSON-driven).

Inputs (reads)
--------------
- Enriched 30m parquet per ticker (produced by s32):
    P.DATA_ENRICHED/30min/{TICKER}.parquet
  Expected core cols (subset): datetime, open, high, low, close
  Expected daily overlays from s32: ema5_d, ema20_d, ema44_d, rsi14_d
  Optional daily overlays (if present): adx14_d
  (High/low are used to build daily Donchian from 30m bars.)

- Rules configuration (JSON):
    P.CONFIG_DIR/gate1_v1_rules.json
  Keys include: ema_base_len, confirm_days_long/short, buffer_pct,
      rsi_long_max / rsi_short_min, trend alignment flags,
      slope mins, use_adx + thresholds, use_donchian + parameters,
      and per-ticker "overrides".

- (Optional) Universe filter (Excel):
    P.CONFIG_DIR/ETF_list.xlsx   (default sheet: "US_small")
  If missing/empty → falls back to processing ALL parquets.

- (Optional) Environment:
    MARKET_TZ  (e.g., "Europe/London") — used for local-day grouping
    (daily last-bar proxy). If unset, defaults to "Europe/London".

Outputs (writes)
----------------
- P.ROOT/signals/rule_live_signals_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_top10_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/operator_today_RULE_all_<YYYYMMDD_HHMM>.csv
- P.ROOT/signals/rule_live_signals_<YYYYMMDD_HHMM>.metadata.json  (exact rule set used)

Columns (primary board)
-----------------------
- decision, side, strength_score (ranking score vs 30m EMA base)
- strength20_pct, strength44_pct (price vs daily EMA20/EMA44 in %; daily last-30m proxy)
- reason_long  (why LONG did not trigger; comma-joined tokens; includes ADX/Donchian reasons if enabled)
- reason_short (why SHORT did not trigger; comma-joined tokens; includes ADX/Donchian reasons if enabled)
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
from common.paths import P  # ROOT, DATA_ENRICHED, CONFIG_DIR, etc.

# ----- defaults (Gate-1_v1.0) -----
DEFAULT_RULES = {
    # label / timeframe (informational)
    "label": "gate1_v1.0",
    "timeframe": "30m",

    # EMA base & confirmation logic
    "ema_base_len": 275,
    "confirm_days_long": 1,     # relaxed: 1 day confirmation
    "confirm_days_short": 1,
    "allow_today_open_as_confirm": True,  # treated as "today's day counts" using latest 30m bar

    # buffer vs EMA base
    "use_ema_buffer": True,
    "buffer_pct": 0.0010,       # 0.10%

    # enable shorts
    "enable_shorts": True,

    # RSI guards (daily RSI column expected from enrich step)
    "rsi_long_max": 65.0,
    "rsi_short_min": 40.0,

    # Trend alignment (daily EMAs)
    "require_trend_alignment": True,
    "price_gt_ema_base_for_long": True,
    "price_lt_ema_base_for_short": True,
    "ema20_gt_ema44_for_long": True,
    "ema20_lt_ema44_for_short": True,

    # Slope filters (relative, per day)
    "ema5_slope_min": 0.001,    # 0.10% / day
    "ema20_slope_min": 0.0003,  # 0.03% / day
    "ema_base_slope_min": 0.0,  # ≥0 required for long; ≤0 for short
    
    # Donchian breakout (optional hard requirement; defaults keep today's behavior)
    "donchian_require_breakout_long": False,
    "donchian_require_breakout_short": False,

    # Optional per-ticker overrides
    "overrides": {},

    # Liquidity filter (not enforced here; for s81 if desired)
    "liquidity_filter": {
        "min_aum_m": 0,
        "min_avg_dollar_vol_m": 0.0
    }
}


def _pickf(row: pd.Series, names: list[str]) -> float:
    for n in names:
        if n in row:
            try:
                v = float(row.get(n, np.nan))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return np.nan

# ---------- ticker normalization ----------
def _norm_ticker(x: str) -> str:
    s = str(x).upper().replace("\u00A0", " ").strip()  # strip NBSP too
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

def _asof_date_or_none(market_tz: str):
    s = os.environ.get("ASOF_DATE", "").strip()
    if not s:
        return None
    try:
        # interpret as local date in the market TZ
        return pd.to_datetime(s).tz_localize(market_tz).date()
    except Exception:
        print(f"[WARN] Invalid ASOF_DATE='{s}', ignoring.")
        return None
    
def _yesno(x: bool) -> str:
    return "YES" if bool(x) else "NO"

def _read_rules(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] Rules JSON not found: {path}. Using defaults.")
        return DEFAULT_RULES.copy()
    try:
        cfg = json.loads(path.read_text())
        rules = DEFAULT_RULES.copy()
        rules.update(cfg)  # <-- allow keys not listed in DEFAULT_RULES
        # ensure overrides dict exists
        if "overrides" not in rules or not isinstance(rules["overrides"], dict):
            rules["overrides"] = {}
        return rules
    except Exception as e:
        print(f"[WARN] Could not parse {path.name}: {e}. Using defaults.")
        return DEFAULT_RULES.copy()

def _read_parquet(path: Path, ema_base_len: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{path.name}: missing required columns (datetime, close)")
    dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    df["datetime"] = dt
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Dynamic EMA base column (e.g., m30_ema275)
    ema_col = f"m30_ema{int(ema_base_len)}"
    if ema_col not in df.columns:
        df = df.sort_values("datetime").copy()
        df[ema_col] = df["close"].ewm(span=int(ema_base_len), adjust=False).mean()

    return (df
            .dropna(subset=["datetime", "close"])
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

def _strength_long(m30_close, ema_base, rsi14d, min_clip=-0.10, max_clip=0.10):
    above = (m30_close / ema_base - 1.0) if (pd.notna(m30_close) and pd.notna(ema_base) and ema_base > 0) else 0.0
    above = float(np.clip(above, min_clip, max_clip))
    rsi_term = 0.01 * ((float(rsi14d) - 50.0) if pd.notna(rsi14d) else 0.0)
    return 2.0 * above + rsi_term

def _strength_short(m30_close, ema_base, rsi14d, min_clip=-0.10, max_clip=0.10):
    below = (ema_base / m30_close - 1.0) if (pd.notna(m30_close) and pd.notna(ema_base) and m30_close > 0) else 0.0
    below = float(np.clip(below, min_clip, max_clip))
    rsi_term = 0.01 * ((50.0 - float(rsi14d)) if pd.notna(rsi14d) else 0.0)
    return 2.0 * below + rsi_term

def _ema_base_slope_rel(df: pd.DataFrame, ema_col: str, market_tz: str) -> float | None:
    """Relative slope of EMA base between yesterday's last 30m bar and today's last 30m bar."""
    dlast = _latest_daily_lastbars(df, market_tz, n_days=3)
    if len(dlast) < 2:
        return None
    today = float(dlast.iloc[-1].get(ema_col, np.nan))
    yday  = float(dlast.iloc[-2].get(ema_col, np.nan))
    return _daily_relative_slope(today, yday)
def _daily_hilo(df: pd.DataFrame, market_tz: str) -> pd.DataFrame:
    """Return per-local-day high/low/close DataFrame sorted by date_local."""
    local = df["datetime"].dt.tz_convert(market_tz)
    g = df.assign(date_local=local.dt.date)
    by = g.groupby("date_local", as_index=False)
    dd = by.agg(
        day_high=("high","max"),
        day_low =("low","min"),
        day_close=("close","last")
    ).sort_values("date_local").reset_index(drop=True)
    return dd

def _prev_daily_closes_from_30m(df: pd.DataFrame, market_tz: str, n: int = 2) -> list[float]:
    """
    Return the last n completed local-day closes using the 30m stream.
    Always excludes the current local calendar day to avoid partial sessions.
    """
    local = df["datetime"].dt.tz_convert(market_tz)
    dd = (df.assign(date_local=local.dt.date)
            .groupby("date_local", as_index=False)["close"].last()
            .sort_values("date_local"))
    today_local = _asof_date_or_none(market_tz) or pd.Timestamp.now(tz=market_tz).date()
    dd = dd[dd["date_local"] < today_local]  # drop today's incomplete day
    vals = dd["close"].tail(n).tolist()
    # ensure length n; most-recent first
    vals = (vals if len(vals) >= n else [np.nan] * (n - len(vals)) + vals)
    return vals[::-1]
# ---- Confidence→probability calibration (loaded once) ----
BIN_EDGES = [0, 40, 60, 80, 90, 101]
BIN_LABELS = ["0-40", "40-60", "60-80", "80-90", "90-100"]
_CALIB_NEXT1D = None

def _score_bin(x: float) -> str:
    try:
        i = np.digitize([float(x)], BIN_EDGES, right=False)[0] - 1
        return BIN_LABELS[max(0, min(i, len(BIN_LABELS)-1))]
    except Exception:
        return "unknown"

def _load_next1d_calib():
    """Load calibration table produced by s78_gate15_calibrate_next1d.py."""
    global _CALIB_NEXT1D
    if _CALIB_NEXT1D is not None:
        return _CALIB_NEXT1D
    path = P.ROOT / "signals" / "analytics" / "gate15_calib_next1d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # two lookup dicts: (side, score_bin) then (side, label_bin)
    by_score = {(str(r.side), str(r.score_bin)): (float(r.p_hat), float(r.p_lo), float(r.p_hi), int(r.n))
                for r in df[df["level"]=="side+score_bin"].itertuples()}
    by_label = {(str(r.side), str(r.score_bin)): (float(r.p_hat), float(r.p_lo), float(r.p_hi), int(r.n))
                for r in df[df["level"]=="side+label"].itertuples()}
    _CALIB_NEXT1D = {"by_score": by_score, "by_label": by_label}
    return _CALIB_NEXT1D

def _lookup_p_next1d(side: str, score: float, label: str, nmin: int = 50):
    """
    Hierarchical back-off:
      1) (side, score_bin) if n>=nmin
      2) (side, label)     if n>=nmin
      3) global neutral prior 0.50 (no CI)
    Returns (p, lo, hi, used_level)
    """
    C = _load_next1d_calib()
    if not C:
        return (0.50, np.nan, np.nan, "prior")
    sb = _score_bin(score)
    key = (str(side or ""), sb)
    hit = C["by_score"].get(key)
    if hit and hit[3] >= nmin:
        return (hit[0], hit[1], hit[2], f"side+score_bin[{sb}]")

    key2 = (str(side or ""), str(label or ""))
    hit2 = C["by_label"].get(key2)
    if hit2 and hit2[3] >= nmin:
        return (hit2[0], hit2[1], hit2[2], f"side+label[{label}]")

    return (0.50, np.nan, np.nan, "prior")
# --- Gate 1.5: confidence score (continuous/weighted) ---
def _gate15_confidence_label(score: float, cfg: dict) -> str:
    th = (cfg.get("label_thresholds") or {"low": 60, "high": 80})
    if score >= float(th.get("high", 80)): return "High"
    if score >= float(th.get("low", 60)):  return "Medium"
    return "Low"

def _lin(a: float, b: float, x: float) -> float:
    """
    Linear ramp: full credit at 'a', zero at 'b'. Clipped to [0,1].
    Use a<b for long-type windows (cap→soft), or a>b for short-type (floor→soft).
    """
    if a == b:
        return 0.0
    t = (x - b) / (a - b)
    return float(np.clip(t, 0.0, 1.0))

def _pct_to_points(ok_frac: float, weight: float) -> float:
    return float(max(0.0, min(1.0, ok_frac)) * float(weight))

def _gate15_confidence_score(row, params, R_conf):
    """
    Continuous 0–100 confidence with JSON weights.
    `row`    = row_calc prepared in s77 (booleans/metrics),
    `params` = effective rule params already computed,
    `R_conf` = R.get("confidence", {...}) from rules JSON.
    """
    W = (R_conf.get("weights") or {})
    w_price   = float(W.get("core_price_vs_base", 20.0))
    w_20v44   = float(W.get("core_20v44_trend",   20.0))
    w_confirm = float(W.get("core_confirm_days",  15.0))
    w_buffer  = float(W.get("core_buffer_respected", 5.0))
    w_rsi     = float(W.get("rsi", 10.0))
    w_slopes  = float(W.get("slopes_pack", 15.0))
    w_adx     = float(W.get("adx", 10.0))
    w_dc      = float(W.get("donchian", 5.0))

    score = 0.0

    # Core (binary but heavy)
    score += _pct_to_points(1.0 if row.get("price_vs_ema_base_ok") else 0.0, w_price)
    score += _pct_to_points(1.0 if row.get("ema20_vs_ema44_ok")   else 0.0, w_20v44)
    score += _pct_to_points(1.0 if row.get("confirm_days_ok")     else 0.0, w_confirm)
    if params.get("use_ema_buffer", False):
        score += _pct_to_points(1.0 if row.get("ema_buffer_ok")   else 0.0, w_buffer)

    # RSI (soft window)
    rsi_conf = 0.0
    rsi = row.get("rsi_value")
    rsi_cfg = (R_conf.get("rsi") or {})
    if rsi is not None and np.isfinite(rsi):
        if row.get("is_long", False):
            cap  = float(rsi_cfg.get("long_cap",  70.0))
            soft = float(rsi_cfg.get("long_soft", 75.0))
            rsi_conf = _lin(cap, soft, float(rsi))     # 1→0 as RSI goes cap→soft
        elif row.get("is_short", False):
            floor = float(rsi_cfg.get("short_floor", 40.0))
            soft  = float(rsi_cfg.get("short_soft",  35.0))
            rsi_conf = _lin(floor, soft, float(rsi))   # 1→0 as RSI goes floor→soft
    score += _pct_to_points(rsi_conf, w_rsi)

    # Slopes pack (ema5, ema20, base) with partial credit 0–1
    def slope_frac(ok_flag, val, min_req):
        if ok_flag:
            return 1.0
        if (val is None) or (min_req is None) or float(min_req) == 0.0:
            return 0.0
        ratio = float(val) / float(min_req)
        if ratio >= 1.0: return 1.0
        if ratio >= 0.8: return 0.5
        return 0.0

    s5  = slope_frac(row.get("ema5_slope_ok"),        row.get("ema5_slope"),        params.get("ema5_slope_min"))
    s20 = slope_frac(row.get("ema20_slope_ok"),       row.get("ema20_slope"),       params.get("ema20_slope_min"))
    sb  = slope_frac(row.get("ema_base_slope_ok"),    row.get("ema_base_slope"),    params.get("ema_base_slope_min"))
    score += _pct_to_points((s5 + s20 + sb) / 3.0, w_slopes)

    # ADX (split level 80% + rising 20%)
    adx_pts = 0.0
    if params.get("use_adx", False):
        level_pts = 0.8 * w_adx
        rise_pts  = 0.2 * w_adx
        if row.get("adx_ok"): adx_pts += level_pts
        if params.get("adx_require_rising", False):
            if row.get("adx_rising_ok"): adx_pts += rise_pts
        else:
            adx_pts += rise_pts  # neutral full credit if not required
    score += adx_pts

    # Donchian (width 40%, pos 40%, breakout 20%)
    dc_pts = 0.0
    if params.get("use_donchian", False):
        w_w, w_p, w_b = 0.4 * w_dc, 0.4 * w_dc, 0.2 * w_dc
        if row.get("donchian_width_ok"):    dc_pts += w_w
        if row.get("donchian_pos_ok"):      dc_pts += w_p
        if row.get("donchian_breakout_ok"): dc_pts += w_b
    score += dc_pts

    return round(float(np.clip(score, 0.0, 100.0)), 1)
def _donchian_from_daily(dd: pd.DataFrame, length: int) -> tuple[float,float,float,float]:
    """
    Build Donchian channel from the previous N days (exclude today).
    Returns (upper, lower, width_pct, pos) for today's close.
    pos in [0,1]: (close - lower)/(upper - lower)
    """
    if dd is None or len(dd) < (length + 1):
        return (np.nan, np.nan, np.nan, np.nan)
    prev = dd.iloc[-(length+1):-1]          # last N days excluding today
    today = dd.iloc[-1]
    upper = float(prev["day_high"].max())
    lower = float(prev["day_low"].min())
    if not (np.isfinite(upper) and np.isfinite(lower) and upper > lower > 0):
        return (np.nan, np.nan, np.nan, np.nan)
    width_pct = (upper / lower - 1.0) * 100.0
    pos = (float(today["day_close"]) - lower) / (upper - lower)
    return (upper, lower, width_pct, pos)

# ----- main -----
def main():
    ap = argparse.ArgumentParser(description="s77 v1.0 — Gate-1 rule signals from s32 30m parquet (no ML).")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N tickers (debug).")
    ap.add_argument("--rules-config", type=str,
                    default=str(P.CONFIG_DIR / "gate1_v1_rules.json"),
                    help="Path to rules JSON (e.g., config/gate1_v1_rules.json)")
    # legacy flags (optional overrides)
    ap.add_argument("--use-ema-buffer", action="store_true", default=None,
                    help="Use EMA base +/- buffer (override JSON).")
    ap.add_argument("--buffer-pct", type=float, default=None,
                    help="Buffer percent relative to EMA base (override JSON).")
    ap.add_argument("--enable-shorts", action="store_true", default=None,
                    help="Allow short signals (override JSON).")
    ap.add_argument("--start-date", type=str, default=None,
                    help="YYYY-MM-DD; ignore signals before this local date.")

    # >>> NEW FLAGS (Step 1)
    ap.add_argument("--append-parquet", action="store_true",
                    help="Append this board to signals/boards_ds/ Parquet dataset.")
    ap.add_argument("--no-operator-exports", action="store_true",
                    help="Skip writing operator CSVs (shortlist/top10/all).")
    ap.add_argument("--no-board-csv", action="store_true",
                    help="Skip writing the per-run rule_live_signals_*.csv file.")

    args = ap.parse_args()

    # load rules
    rules_path = Path(args.rules_config)
    R = _read_rules(rules_path)

    # CLI overrides (if provided)
    if args.use_ema_buffer is not None: R["use_ema_buffer"] = bool(args.use_ema_buffer)
    if args.buffer_pct is not None:     R["buffer_pct"]     = float(args.buffer_pct)
    if args.enable_shorts is not None:  R["enable_shorts"]  = bool(args.enable_shorts)

    ema_base_len = int(R.get("ema_base_len", 275))
    ema_col = f"m30_ema{ema_base_len}"

    mkt_tz = _market_tz()
    m30_dir = P.DATA_ENRICHED / "30min"
    if not m30_dir.exists():
        raise SystemExit(f"[ERR] {m30_dir} not found. Run s32 first.")

    # list parquet files
    parqs = sorted(m30_dir.glob("*.parquet"))
    if not parqs:
        raise SystemExit(f"[ERR] No parquet files in {m30_dir}.")

    # ---------- APPLY EXCEL UNIVERSE (QuantShared + signalsUSD) ----------

    uni_path = P.ETF_LIST                            # /Users/Finance/QuantShared/ETF_list.xlsx (or QETF_LIST)
    sheet    = os.environ.get("ETF_SHEET", "signalsUSD")
    universe = _load_universe_from_xlsx(uni_path, sheet=sheet, verbose=True)
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
            df = _read_parquet(pq, ema_base_len=ema_base_len)
            if df.empty:
                continue
            asof = _asof_date_or_none(mkt_tz)
            if asof is not None:
                _local = df["datetime"].dt.tz_convert(mkt_tz)
                df = df.assign(_date_local=_local.dt.date)
                df = df[df["_date_local"] <= asof].drop(columns=["_date_local"])
                if df.empty or len(df) < 10:
                    continue
            # last 30m bar
            last = df.iloc[-1]
            ema_base  = float(last.get(ema_col, np.nan))
            m30_close = float(last.get("close", np.nan))
            rsi14_d   = float(last.get("rsi14_d", np.nan)) if pd.notna(last.get("rsi14_d", np.nan)) else np.nan

            # --- daily overlays across last local days (for trend & slopes) ---
            dlast = _latest_daily_lastbars(df, mkt_tz, n_days=4)

            # defaults (so names always exist)
            ema5_t = ema5_y = ema20_t = ema20_y = ema44_t = np.nan
            adx_t = adx_y = adx_delta = np.nan
            slope5 = slope20 = None
            daily_close_t = np.nan
            rsi14_d = _pickf(last, ["rsi14_d", "rsi14"])  # fallback from latest row

            if len(dlast) >= 2:
                today = dlast.iloc[-1]
                yday  = dlast.iloc[-2]

                # Daily EMAs (accept both *_d and plain)
                ema5_t  = _pickf(today, ["ema5_d", "ema5"])
                ema5_y  = _pickf(yday,  ["ema5_d", "ema5"])
                ema20_t = _pickf(today, ["ema20_d", "ema20"])
                ema20_y = _pickf(yday,  ["ema20_d", "ema20"])
                ema44_t = _pickf(today, ["ema44_d", "ema44"])

                # RSI (prefer today's daily; fallback kept above)
                rsi14_t = _pickf(today, ["rsi14_d", "rsi14"])
                if np.isfinite(rsi14_t):
                    rsi14_d = rsi14_t

                # ADX (optional)
                adx_t = _pickf(today, ["adx14_d", "adx14"])
                adx_y = _pickf(yday,  ["adx14_d", "adx14"])
                adx_delta = (adx_t - adx_y) if (np.isfinite(adx_t) and np.isfinite(adx_y)) else np.nan

                # Slopes (relative day-over-day)
                slope5  = _daily_relative_slope(ema5_t,  ema5_y)
                slope20 = _daily_relative_slope(ema20_t, ema20_y)

                # Use today's last-30m as the daily close proxy
                daily_close_t = float(today.get("close", np.nan))

            # Donchian (from daily highs/lows built off 30m)
            dc_len   = int((R.get("donchian_len") or 20))
            use_dc   = bool(R.get("use_donchian", True))
            dc_upper = dc_lower = dc_width_pct = dc_pos = np.nan
            if use_dc:
                dd = _daily_hilo(df, mkt_tz)
                dc_upper, dc_lower, dc_width_pct, dc_pos = _donchian_from_daily(dd, dc_len)

            # start-date gate
            if args.start_date:
                last_local_date = last["datetime"].tz_convert(mkt_tz).date()
                if last_local_date < pd.to_datetime(args.start_date).date():
                    ema_base = np.nan  # force no-entry

            # EMA base buffer lines
            if R.get("use_ema_buffer", True) and pd.notna(ema_base):
                buf = ema_base * float(R.get("buffer_pct", 0.0))
                above_line = ema_base + buf
                below_line = ema_base - buf
            else:
                above_line = ema_base
                below_line = ema_base

            # collect last few daily closes from 30m stream
            # (day_k_close: k=1 -> yesterday, k=2 -> two days ago, etc.)
            closes_by_day = _prev_daily_closes_from_30m(df, mkt_tz, n=4)  # yesterday, 2d ago, ...
            day1_close = closes_by_day[0] if len(closes_by_day) > 0 else np.nan  # yesterday
            day2_close = closes_by_day[1] if len(closes_by_day) > 1 else np.nan  # 2 sessions ago

            # per-ticker overrides
            ov = (R.get("overrides") or {}).get(_norm_ticker(tkr), {})
            ema5_slope_min      = float(ov.get("ema5_slope_min",  R["ema5_slope_min"]))
            ema20_slope_min     = float(ov.get("ema20_slope_min", R["ema20_slope_min"]))
            ema_base_slope_min  = float(ov.get("ema_base_slope_min", R.get("ema_base_slope_min", 0.0)))

            # ----- RELAXED CONFIRMATION LOGIC -----
            def _count_days_above(line: float) -> int:
                if not np.isfinite(line):
                    return 0
                cnt = 0
                for dc in closes_by_day:  # yesterday, two-days-ago, ...
                    if np.isfinite(dc) and dc > line:
                        cnt += 1
                    else:
                        break  # require consecutive days (most recent backwards)
                # allow today's (intraday) to count as a confirm day if enabled
                if R.get("allow_today_open_as_confirm", True):
                    if np.isfinite(m30_close) and m30_close > line:
                        cnt = max(cnt, 1)  # at least 1 day counted
                return cnt

            def _count_days_below(line: float) -> int:
                if not np.isfinite(line):
                    return 0
                cnt = 0
                for dc in closes_by_day:
                    if np.isfinite(dc) and dc < line:
                        cnt += 1
                    else:
                        break
                if R.get("allow_today_open_as_confirm", True):
                    if np.isfinite(m30_close) and m30_close < line:
                        cnt = max(cnt, 1)
                return cnt

            need_long = int(R.get("confirm_days_long", 1))
            need_short = int(R.get("confirm_days_short", 1))

            days_above = _count_days_above(above_line)
            days_below = _count_days_below(below_line)

            long_confirm_ok  = (days_above >= need_long)
            short_confirm_ok = (days_below >= need_short)

            # explicit price vs EMA base flags (if required)
            price_gt_base_ok = True
            price_lt_base_ok = True
            if R.get("price_gt_ema_base_for_long", True):
                price_gt_base_ok = (np.isfinite(m30_close) and np.isfinite(above_line) and (m30_close > above_line))
            if R.get("price_lt_ema_base_for_short", True):
                price_lt_base_ok = (np.isfinite(m30_close) and np.isfinite(below_line) and (m30_close < below_line))

            # RSI guards
            rsi_ok_long  = (pd.notna(rsi14_d) and (rsi14_d <= float(R["rsi_long_max"])))
            rsi_ok_short = (pd.notna(rsi14_d) and (rsi14_d >= float(R["rsi_short_min"])))

            # trend alignment
            trend_long_ok  = True
            trend_short_ok = True
            if R.get("require_trend_alignment", True):
                if R.get("ema20_gt_ema44_for_long", True):
                    trend_long_ok = (pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t > ema44_t))
                if R.get("ema20_lt_ema44_for_short", True):
                    trend_short_ok = (pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t < ema44_t))

            # --- NEW: minimum percent gap between EMA20 and EMA44 ---
            gap_long_block = gap_short_block = False
            ema_gap_req = float(R.get("ema20_min_gap_pct", 0.0))
            if ema_gap_req > 0 and np.isfinite(ema20_t) and np.isfinite(ema44_t) and np.isfinite(m30_close) and m30_close > 0:
                gap_pct = abs(ema20_t/ema44_t - 1.0) * 100.0
                if gap_pct < ema_gap_req:
                    if R.get("ema20_gt_ema44_for_long", True):
                        trend_long_ok = False; gap_long_block = True
                    if R.get("ema20_lt_ema44_for_short", True):
                        trend_short_ok = False; gap_short_block = True

            # slope filters (daily EMAs)
            if (slope5 is None) or (slope20 is None):
                slope_long_ok = False
                slope_short_ok = False
            else:
                slope_long_ok  = (slope5  >= ema5_slope_min)  and (slope20 >= ema20_slope_min)
                slope_short_ok = (slope5  <= -ema5_slope_min) and (slope20 <= -ema20_slope_min)

            # ---------- ADX guards ----------
            use_adx          = bool(R.get("use_adx", True))
            adx_min_long     = float(R.get("adx_min_long", 20.0))
            adx_min_short    = float(R.get("adx_min_short", 20.0))
            adx_req_rising   = bool(R.get("adx_require_rising", False))
            adx_min_delta    = float(R.get("adx_min_delta", 0.0))

            # allow per-ticker override
            ov = (R.get("overrides") or {}).get(_norm_ticker(tkr), {})
            adx_min_long  = float(ov.get("adx_min_long",  adx_min_long))
            adx_min_short = float(ov.get("adx_min_short", adx_min_short))

            adx_long_ok  = True
            adx_short_ok = True
            if use_adx:
                adx_long_ok  = (np.isfinite(adx_t) and adx_t >= adx_min_long)
                adx_short_ok = (np.isfinite(adx_t) and adx_t >= adx_min_short)
                if adx_req_rising:
                    adx_long_ok  = adx_long_ok  and (np.isfinite(adx_delta) and adx_delta >= adx_min_delta)
                    adx_short_ok = adx_short_ok and (np.isfinite(adx_delta) and adx_delta >= adx_min_delta)

            # ---------- Donchian guards ----------
            use_dc = bool(R.get("use_donchian", True))
            dc_min_width_pct          = float(R.get("donchian_min_width_pct", 4.0))
            dc_avoid_long_if_pos_gt   = float(R.get("donchian_avoid_long_if_pos_gt", 0.90))
            dc_avoid_short_if_pos_lt  = float(R.get("donchian_avoid_short_if_pos_lt", 0.10))

            dc_long_ok  = True
            dc_short_ok = True
            if use_dc and np.isfinite(dc_width_pct) and np.isfinite(dc_pos):
                # width filter
                if dc_width_pct < dc_min_width_pct:
                    dc_long_ok = dc_short_ok = False
                # stretch extremes
                if dc_pos > dc_avoid_long_if_pos_gt:
                    dc_long_ok = False
                if dc_pos < dc_avoid_short_if_pos_lt:
                    dc_short_ok = False
            # --- NEW: Donchian breakout margin (percent) ---
            dc_margin = float(R.get("donchian_breakout_margin_pct", 0.0)) / 100.0

            if use_dc:
                thr_long  = (1.0 + dc_margin)
                thr_short = (1.0 - dc_margin)
                dc_breakout_long  = (np.isfinite(dc_upper) and np.isfinite(daily_close_t) and daily_close_t > (dc_upper * thr_long))
                dc_breakout_short = (np.isfinite(dc_lower) and np.isfinite(daily_close_t) and daily_close_t < (dc_lower * thr_short))

                if R.get("donchian_require_breakout_long", False):
                    dc_long_ok = dc_long_ok and dc_breakout_long
                else:
                    dc_breakout_long = True

                if R.get("donchian_require_breakout_short", False):
                    dc_short_ok = dc_short_ok and dc_breakout_short
                else:
                    dc_breakout_short = True
            else:
                dc_breakout_long = dc_breakout_short = True

            # EMA base slope filter (using 30m EMA base day-to-day)
            base_slope_rel = _ema_base_slope_rel(df, ema_col=ema_col, market_tz=mkt_tz)
            base_slope_long_ok  = (base_slope_rel is not None and base_slope_rel >= ema_base_slope_min)
            base_slope_short_ok = (base_slope_rel is not None and base_slope_rel <= -ema_base_slope_min)

            # combine (wire ADX + Donchian)
            long_ok = (long_confirm_ok and price_gt_base_ok and rsi_ok_long and
                    trend_long_ok and slope_long_ok and base_slope_long_ok and
                    (not use_adx or adx_long_ok) and
                    (not use_dc or dc_long_ok))

            short_ok = False
            if bool(R.get("enable_shorts", True)):
                short_ok = (short_confirm_ok and price_lt_base_ok and rsi_ok_short and
                            trend_short_ok and slope_short_ok and base_slope_short_ok and
                            (not use_adx or adx_short_ok) and
                            (not use_dc or dc_short_ok))

            # ranking strengths vs 30m EMA base (keep)
            s_long  = _strength_long(m30_close, ema_base, rsi14_d)
            s_short = _strength_short(m30_close, ema_base, rsi14_d)

            # DAILY display strengths (use today's last-30m close captured above)
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

            # reasons
            long_reasons = []
            short_reasons = []
            if not long_confirm_ok:     long_reasons.append(f"confirm_days_long<{need_long}")
            if not price_gt_base_ok:    long_reasons.append("price_vs_base_long")
            if not rsi_ok_long:         long_reasons.append("rsi_long_guard")
            if not trend_long_ok:
                long_reasons.append("trend_align_long")
                if gap_long_block:
                    long_reasons.append("trend_gap_small")
            if not trend_short_ok:
                short_reasons.append("trend_align_short")
                if gap_short_block:
                    short_reasons.append("trend_gap_small")
            if not slope_long_ok:       long_reasons.append("slope_long")
            if not base_slope_long_ok:  long_reasons.append("base_slope_long")
            if np.isfinite(strength44_pct) and strength44_pct > 10.0:
                long_reasons.append("stretched_vs_ema44")

            # ADX / Donchian reasons (long)
            if use_adx and not adx_long_ok:
                if not np.isfinite(adx_t) or adx_t < adx_min_long:
                    long_reasons.append("adx_long_guard")
                elif adx_req_rising and (not np.isfinite(adx_delta) or adx_delta < adx_min_delta):
                    long_reasons.append("adx_long_not_rising")

            if use_dc:
                if np.isfinite(dc_width_pct) and dc_width_pct < dc_min_width_pct:
                    long_reasons.append("donchian_width_thin")
                if np.isfinite(dc_pos) and dc_pos > dc_avoid_long_if_pos_gt:
                    long_reasons.append("donchian_long_extreme")
            if use_dc and R.get("donchian_require_breakout_long", False) and not dc_breakout_long:
                long_reasons.append("donchian_breakout_long")

            short_reasons = []
            if bool(R.get("enable_shorts", True)):
                if not short_confirm_ok:    short_reasons.append(f"confirm_days_short<{need_short}")
                if not price_lt_base_ok:    short_reasons.append("price_vs_base_short")
                if not rsi_ok_short:        short_reasons.append("rsi_short_guard")
                if not trend_short_ok:      short_reasons.append("trend_align_short")
                if not slope_short_ok:      short_reasons.append("slope_short")
                if not base_slope_short_ok: short_reasons.append("base_slope_short")

            reason_long  = ",".join(dict.fromkeys(long_reasons))
            if use_adx and not adx_short_ok:
                if not np.isfinite(adx_t) or adx_t < adx_min_short:
                    short_reasons.append("adx_short_guard")
                elif adx_req_rising and (not np.isfinite(adx_delta) or adx_delta < adx_min_delta):
                    short_reasons.append("adx_short_not_rising")

            if use_dc:
                if np.isfinite(dc_width_pct) and dc_width_pct < dc_min_width_pct:
                    short_reasons.append("donchian_width_thin")
                if np.isfinite(dc_pos) and dc_pos < dc_avoid_short_if_pos_lt:
                    short_reasons.append("donchian_short_extreme")

            if use_dc and R.get("donchian_require_breakout_short", False) and not dc_breakout_short:
                short_reasons.append("donchian_breakout_short")
            reason_short = ",".join(dict.fromkeys(short_reasons))
            
            # --- Gate 1.5: setup fingerprint (for historical grouping) ---
            # decide which side we are evaluating
            _fp_side = "long" if decision == "BUY" else ("short" if decision == "SELL" else "none")

            # Core checks
            _fp_base      = 1 if (np.isfinite(m30_close) and np.isfinite(ema_base) and
                                ((m30_close > (ema_base + (ema_base*float(R.get("buffer_pct",0.0))))) if _fp_side=="long"
                                else (m30_close < (ema_base - (ema_base*float(R.get("buffer_pct",0.0))))))) else 0
            _fp_20v44     = 1 if ((pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t > ema44_t)) if _fp_side=="long"
                                else (pd.notna(ema20_t) and pd.notna(ema44_t) and (ema20_t < ema44_t))) else 0
            _fp_confirm   = 1 if ((long_confirm_ok if _fp_side=="long" else short_confirm_ok)) else 0
            _fp_buffer    = 1 if R.get("use_ema_buffer", True) else 0

            # Guards
            _fp_rsi       = 1 if ((rsi_ok_long if _fp_side=="long" else rsi_ok_short)) else 0

            # Slopes packed as 3 digits: ema5, ema20, base (1/0 each, per side)
            _fp_s5  = 1 if ((slope_long_ok if _fp_side=="long" else slope_short_ok)) else 0
            _fp_s20 = _fp_s5
            _fp_sb  = 1 if ((base_slope_long_ok if _fp_side=="long" else base_slope_short_ok)) else 0

            # ADX
            if bool(R.get("use_adx", True)):
                _fp_adx_ok = 1 if ((adx_long_ok if _fp_side=="long" else adx_short_ok)) else 0
                _fp_adx_r  = 1 if (bool(R.get("adx_require_rising", False)) and np.isfinite(adx_delta) and
                                adx_delta >= float(R.get("adx_min_delta",0.0))) else 0
                _fp_adx = f"{_fp_adx_ok}{_fp_adx_r}"
            else:
                _fp_adx = "x"

            # Donchian
            if bool(R.get("use_donchian", True)):
                _fp_d_w = 1 if (np.isfinite(dc_width_pct) and dc_width_pct >= float(R.get("donchian_min_width_pct",4.0))) else 0
                if _fp_side == "long":
                    _fp_d_p = 1 if (np.isfinite(dc_pos) and dc_pos <= float(R.get("donchian_avoid_long_if_pos_gt",0.90))) else 0
                    _fp_d_b = 1 if (dc_breakout_long) else 0
                else:
                    _fp_d_p = 1 if (np.isfinite(dc_pos) and dc_pos >= float(R.get("donchian_avoid_short_if_pos_lt",0.10))) else 0
                    _fp_d_b = 1 if (dc_breakout_short) else 0
                _fp_don = f"{_fp_d_w}{_fp_d_p}{_fp_d_b}"
            else:
                _fp_don = "x"

            setup_fingerprint = "|".join([
                _fp_side,
                f"base={_fp_base}",
                f"20>44={_fp_20v44}",
                f"confirm={_fp_confirm}",
                f"buffer={_fp_buffer}",
                f"rsi={_fp_rsi}",
                f"slopes={_fp_s5}{_fp_s20}{_fp_sb}",
                f"adx={_fp_adx}",
                f"don={_fp_don}",
            ])


            # ---- Gate 1.5 confidence (per decided side) ----
            is_long  = (decision_side == "LONG")
            is_short = (decision_side == "SHORT")

            row_calc = {
                "is_long": is_long,
                "is_short": is_short,
                "price_vs_ema_base_ok": (price_gt_base_ok if is_long else price_lt_base_ok),
                "ema20_vs_ema44_ok": (trend_long_ok if is_long else trend_short_ok),
                "confirm_days_ok": (long_confirm_ok if is_long else short_confirm_ok),
                "ema_buffer_ok": (
                    (np.isfinite(m30_close) and np.isfinite(above_line) and m30_close > above_line) if is_long
                    else (np.isfinite(m30_close) and np.isfinite(below_line) and m30_close < below_line)
                ),
                "rsi_value": rsi14_d,
                "rsi_guard_ok": (rsi_ok_long if is_long else rsi_ok_short),

                # slopes + values
                "ema5_slope_ok":   (slope_long_ok if is_long else slope_short_ok),
                "ema20_slope_ok":  (slope_long_ok if is_long else slope_short_ok),
                "ema_base_slope_ok": (base_slope_long_ok if is_long else base_slope_short_ok),
                "ema5_slope":  slope5,
                "ema20_slope": slope20,
                "ema_base_slope": base_slope_rel,

                # ADX
                "use_adx": use_adx,
                "adx_value": adx_t,
                "adx_ok": (adx_long_ok if is_long else adx_short_ok),
                "adx_rising_ok": (np.isfinite(adx_delta) and adx_delta >= float(R.get("adx_min_delta", 0.0))) if bool(R.get("adx_require_rising", False)) else True,

                # Donchian
                "use_donchian": use_dc,
                "donchian_width_pct": dc_width_pct,
                "donchian_pos": dc_pos,
                "donchian_width_ok": (dc_width_pct >= float(R.get("donchian_min_width_pct", 4.0))) if (use_dc and np.isfinite(dc_width_pct)) else True,
                "donchian_pos_ok": (
                    (dc_pos <= float(R.get("donchian_avoid_long_if_pos_gt", 0.90))) if is_long
                    else (dc_pos >= float(R.get("donchian_avoid_short_if_pos_lt", 0.10)))
                ) if (use_dc and np.isfinite(dc_pos)) else True,
                "donchian_breakout_ok": (dc_breakout_long if is_long else dc_breakout_short),
            }

            params_effective = {
                "use_ema_buffer": bool(R.get("use_ema_buffer", True)),
                "rsi_long_max": float(R.get("rsi_long_max", 65.0)),
                "rsi_short_min": float(R.get("rsi_short_min", 40.0)),
                "ema5_slope_min": float(ema5_slope_min),
                "ema20_slope_min": float(ema20_slope_min),
                "ema_base_slope_min": float(ema_base_slope_min),

                "use_adx": use_adx,
                "adx_require_rising": bool(R.get("adx_require_rising", False)),
                "adx_min_long": float(adx_min_long),
                "adx_min_short": float(adx_min_short),

                "use_donchian": use_dc,
                "donchian_min_width_pct": float(R.get("donchian_min_width_pct", 4.0)),
                "donchian_avoid_long_if_pos_gt": float(R.get("donchian_avoid_long_if_pos_gt", 0.90)),
                "donchian_avoid_short_if_pos_lt": float(R.get("donchian_avoid_short_if_pos_lt", 0.10)),
            }

            R_conf = (R.get("confidence") or {})
            confidence_score = _gate15_confidence_score(row_calc, params_effective, R_conf)
            confidence_label = _gate15_confidence_label(confidence_score, R_conf)

            # probability that next completed session moves in the signal's direction
            p_hat, p_lo, p_hi, p_used = _lookup_p_next1d(decision_side, confidence_score, confidence_label)
            if decision_side == "SHORT":
                # for shorts, report both for clarity
                p_down_next1d = p_hat
                p_up_next1d   = 1.0 - p_hat
            else:
                p_up_next1d   = p_hat
                p_down_next1d = 1.0 - p_hat

            rows.append({
                "ticker": _norm_ticker(tkr).split('.', 1)[0],
                "prev_close_d": day1_close,
                "prev2_close_d": day2_close,
                "decision": decision,
                "side": decision_side,
                "strength_score": round(float(strength), 6),
                "m30_close": m30_close,
                "m30_ema_base": ema_base,
                "ema_base_len": ema_base_len,
                "rsi14_d": rsi14_d,
                "strength20_pct": strength20_pct,
                "strength44_pct": strength44_pct,
                "adx14_d": float(adx_t) if np.isfinite(adx_t) else np.nan,          # today's daily ADX
                "adx14_prev_d": float(adx_y) if np.isfinite(adx_y) else np.nan,      # yesterday's daily ADX
                "adx14_delta": float(adx_delta) if np.isfinite(adx_delta) else np.nan, # today - yesterday
                "adx_long_ok": _yesno(adx_long_ok) if use_adx else "N/A",
                "adx_short_ok": _yesno(adx_short_ok) if use_adx else "N/A",
                "donchian_pos_pct": (float(dc_pos) * 100.0) if np.isfinite(dc_pos) else np.nan,   # 0 = at lower band, 100 = at upper band
                "donchian_width_pct": float(dc_width_pct) if np.isfinite(dc_width_pct) else np.nan,
                "reason_long":  reason_long   if decision == "DO NOTHING" else "",
                "reason_short": reason_short  if decision == "DO NOTHING" else "",
                "confirm_days_long_needed": int(need_long),
                "confirm_days_short_needed": int(need_short),
                "confidence_score": confidence_score,
                "confidence_label": confidence_label,
                "setup_fingerprint": setup_fingerprint,
                "days_above_counted": int(days_above),
                "days_below_counted": int(days_below),
                "trend_long_ok": _yesno(trend_long_ok),
                "trend_short_ok": _yesno(trend_short_ok),
                "slope_long_ok": _yesno(slope_long_ok),
                "slope_short_ok": _yesno(slope_short_ok),
                "p_up_next1d": float(p_up_next1d),
                "p_down_next1d": float(p_down_next1d),
                "p_ci_low_1d": float(p_lo) if np.isfinite(p_lo) else np.nan,
                "p_ci_high_1d": float(p_hi) if np.isfinite(p_hi) else np.nan,
                "calib_bucket_used": p_used,
            })
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")

    if not rows:
        raise SystemExit("[ERR] No rows to decide.")

    X = pd.DataFrame(rows)

    # ----- Outputs -----
    out_dir = P.ROOT / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts()

    # (1) Optional single Parquet dataset append
    if args.append_parquet:
        try:
            X_copy = X.copy()
            X_copy["board_ts_utc"] = ts
            asof = _asof_date_or_none(_market_tz())
            board_day_str = (asof.strftime("%Y%m%d") if asof else ts.split("_")[0])  # YYYYMMDD
            X_copy["board_day"] = board_day_str
            ds_dir = out_dir / "boards_ds"
            ds_dir.mkdir(parents=True, exist_ok=True)
            X_copy.to_parquet(ds_dir, index=False, engine="pyarrow", partition_cols=["board_day"])
            print(f"[OK] Appended to {ds_dir} (partition={X_copy['board_day'].iloc[0]})")
        except Exception as e:
            print(f"[WARN] Could not append to boards_ds: {e}")

    # (2) Per-run board CSV (unless suppressed)
    board_csv = out_dir / f"rule_live_signals_{ts}.csv"
    if not args.no_board_csv:
        (X.sort_values(["decision","strength_score"], ascending=[True, False])
        .to_csv(board_csv, index=False))
        print(f"[OK] Rule live signals (full board) → {board_csv}")

    # (3) OPERATOR exports (SUBSTEP 3): only if not suppressed
    top10 = pd.DataFrame()  # default empty, so later 'if len(top10):' is safe

    if not args.no_operator_exports:
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

    # ----- Metadata (so s81 can reuse the exact rules) -----
    meta = {
        "generated_at_utc": ts,
        "rules_path": str(Path(args.rules_config).resolve()),
        "label": str(R.get("label", "gate1_relaxed")),
        "ema_base_len": int(R.get("ema_base_len", 275)),
        "confirm_days_long": int(R.get("confirm_days_long", 1)),
        "confirm_days_short": int(R.get("confirm_days_short", 1)),
        "allow_today_open_as_confirm": bool(R.get("allow_today_open_as_confirm", True)),
        "buffer_pct": float(R.get("buffer_pct", 0.0)),
        "enable_shorts": bool(R.get("enable_shorts", True)),
        "require_trend_alignment": bool(R.get("require_trend_alignment", True)),
        # NEW for traceability
        "ema20_min_gap_pct": float(R.get("ema20_min_gap_pct", 0.0)),
        "use_adx": bool(R.get("use_adx", True)),
        "adx_min_long": float(R.get("adx_min_long", 20.0)),
        "adx_min_short": float(R.get("adx_min_short", 20.0)),
        "adx_require_rising": bool(R.get("adx_require_rising", False)),
        "adx_min_delta": float(R.get("adx_min_delta", 0.0)),
        "use_donchian": bool(R.get("use_donchian", True)),
        "donchian_len": int(R.get("donchian_len", 20)),
        "donchian_min_width_pct": float(R.get("donchian_min_width_pct", 4.0)),
        "donchian_breakout_margin_pct": float(R.get("donchian_breakout_margin_pct", 0.0)),
        "donchian_require_breakout_long": bool(R.get("donchian_require_breakout_long", False)),
        "donchian_require_breakout_short": bool(R.get("donchian_require_breakout_short", False)),
    }
    meta_path = out_dir / f"rule_live_signals_{ts}.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Metadata → {meta_path}")

    print("\n[Decision counts]")
    print(X["decision"].value_counts(dropna=False).to_string())

    if len(top10):
        print("\n[Top-10]")
        print(top10[["ticker","decision","side","strength_score"]].to_string(index=False))

if __name__ == "__main__":
    main()