#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s50_30m_backtest_engine.py — clone-friendly, NO Vermeulen, T1/T2 only (fixed 50/50)

Adds parametric filters:
- streak_mode: none | prev1 | 1of2 | 2of2
- daily_mode : none | tight | price_above_ema20 | loose
"""

from __future__ import annotations
import os, argparse, warnings, sys
from datetime import datetime, timezone, time as dt_time, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

# ---------- bootstrap project import path ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- project-aware paths ----------
from common.paths import P  # type: ignore

# ---------- clone-friendly detection ----------
def _resolve_bucket() -> str:
    env_b = os.environ.get("TARGET_BUCKET", "").strip()
    if env_b:
        return env_b
    try:
        from common.settings import TARGET_BUCKET as SBUCKET  # type: ignore
        if SBUCKET:
            return str(SBUCKET)
    except Exception:
        pass
    if (P.DATA_RAW / "targeted_ETFs_US" / "30min").exists() or (P.DATA_RAW / "targeted_ETFs_US" / "daily").exists():
        return "targeted_ETFs_US"
    return "targeted_ETFs"

def _resolve_session(bucket: str) -> tuple[str, str, str]:
    tz = os.environ.get("MARKET_TZ", "").strip()
    op = os.environ.get("MARKET_OPEN", "").strip()
    cl = os.environ.get("MARKET_CLOSE", "").strip()
    if tz and op and cl:
        return tz, op, cl
    try:
        from common.settings import MARKET_TZ as STZ, MARKET_OPEN as SOPEN, MARKET_CLOSE as SCLOSE  # type: ignore
        if all([STZ, SOPEN, SCLOSE]):
            return str(STZ), str(SOPEN), str(SCLOSE)
    except Exception:
        pass
    repo_us = PROJECT_ROOT.name.endswith("US") or bucket == "targeted_ETFs_US"
    if repo_us:
        return "Europe/London", "08:00", "16:30"
    return "Europe/Madrid", "09:00", "17:30"

def _resolve_inputs() -> tuple[Path, Path, str]:
    shared_30 = P.DATA_RAW / "30min"
    shared_1d = P.DATA_RAW / "daily"
    if shared_30.exists() and shared_1d.exists():
        return shared_30, shared_1d, "(shared)"

    # NEW: look in QuantShared flat layouts
    qs = getattr(P, "QUANTSHARED", None) or (P.ROOT.parent / "QuantShared")
    qs_30 = qs / "data_raw_ETF_US" / "30min"
    qs_1d = qs / "data_raw_ETF_US" / "daily"
    if qs_30.exists() and qs_1d.exists():
        return qs_30, qs_1d, "(QuantShared US)"

    # fallback: bucket under project
    bucket = _resolve_bucket()
    b30 = P.DATA_RAW / bucket / "30min"
    b1d = P.DATA_RAW / bucket / "daily"
    if b30.exists() and b1d.exists():
        return b30, b1d, bucket

    raise SystemExit(
        f"[ERR] Could not find inputs.\n"
        f"  Tried shared: {shared_30} & {shared_1d}\n"
        f"  Tried QuantShared US: {qs_30} & {qs_1d}\n"
        f"  Tried bucket: {b30} & {b1d} (bucket='{bucket}')"
    )

TARGET_BUCKET = _resolve_bucket()
MARKET_TZ, MARKET_OPEN, MARKET_CLOSE = _resolve_session(TARGET_BUCKET)

ROOT = P.ROOT
IN30, IN1D, IN_TAG = _resolve_inputs()
OUT  = ROOT / "backtest_results_30m"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[s50] MARKET_TZ={MARKET_TZ} | MARKET_OPEN={MARKET_OPEN} | MARKET_CLOSE={MARKET_CLOSE}")
print(f"[s50] Inputs: {IN_TAG}  30m={IN30}  daily={IN1D}")
print(f"[s50] Results → {OUT}")

# ---------------- helpers ----------------
def _filter_session_local(df_utc: pd.DataFrame) -> pd.DataFrame:
    if df_utc.empty:
        return df_utc
    df = df_utc.copy()
    local = df["datetime"].dt.tz_convert(MARKET_TZ)
    t = local.dt.time
    hh_o, mm_o = map(int, str(MARKET_OPEN).split(":"))
    hh_c, mm_c = map(int, str(MARKET_CLOSE).split(":"))
    mask = (t >= dt_time(hh_o, mm_o)) & (t <= dt_time(hh_c, mm_c))
    return df.loc[mask].reset_index(drop=True)

def _load_30min_any(ticker: str) -> pd.DataFrame:
    f_raw = IN30 / f"{ticker}_30min_raw.csv"
    f_alt = IN30 / f"{ticker}_30min.csv"
    f = f_raw if f_raw.exists() else f_alt
    if not f.exists():
        raise FileNotFoundError(f"30m file not found for {ticker} ({f_raw} / {f_alt})")
    df = pd.read_csv(f)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    dtcol = next((c for c in df.columns if "date" in c or "time" in c), None)
    if not dtcol:
        raise ValueError(f"No datetime column in {f}")
    df["datetime"] = pd.to_datetime(df[dtcol], utc=True, errors="coerce")
    for c in ("open","high","low","close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime","open","high","low","close"])
    df = _filter_session_local(df)
    df["date"] = df["datetime"].dt.tz_convert(MARKET_TZ).dt.date
    return df.sort_values("datetime").reset_index(drop=True)

def _load_daily_any(ticker: str) -> pd.DataFrame:
    f_raw = IN1D / f"{ticker}_daily_raw.csv"
    f_alt = IN1D / f"{ticker}_daily.csv"
    f = f_raw if f_raw.exists() else f_alt
    if not f.exists():
        raise FileNotFoundError(f"Daily file not found for {ticker} ({f_raw} / {f_alt})")
    dfd = pd.read_csv(f)
    dfd.columns = [c.strip().lower().replace(" ", "") for c in dfd.columns]
    date_col = "date" if "date" in dfd.columns else "datetime"
    dfd["datetime"] = pd.to_datetime(dfd[date_col], utc=True, errors="coerce").dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    dfd["date"] = dfd["datetime"].dt.date
    for c in ("open","high","low","close"):
        if c in dfd.columns:
            dfd[c] = pd.to_numeric(dfd[c], errors="coerce")
    return dfd.dropna(subset=["datetime","open","high","low","close"]).sort_values("datetime").reset_index(drop=True)

def _atr_manual(dfd: pd.DataFrame, n: int = 14) -> pd.Series:
    d = dfd.copy()
    d["H-L"]  = d["high"] - d["low"]
    d["H-PC"] = (d["high"] - d["close"].shift()).abs()
    d["L-PC"] = (d["low"] - d["close"].shift()).abs()
    d["TR"]   = d[["H-L","H-PC","L-PC"]].max(axis=1)
    return d["TR"].rolling(window=n).mean()

def apply_costs(px: float, side: str, slip_bps: float) -> float:
    if pd.isna(px) or px <= 0:
        return px
    adj = (slip_bps / 10000.0)
    return px * (1.0 + adj) if side == "buy" else px * (1.0 - adj)

def _apply_window_ts(df: pd.DataFrame, ts_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df[ts_col] >= pd.to_datetime(start, utc=True, errors="coerce")]
    if end:
        df = df[df[ts_col] <= pd.to_datetime(end,   utc=True, errors="coerce")]
    return df

def _apply_window_date(df: pd.DataFrame, date_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df[date_col] >= pd.to_datetime(start).date()]
    if end:
        df = df[df[date_col] <= pd.to_datetime(end).date()]
    return df

# ---------------- core backtest ----------------
def backtest_ticker(ticker: str, params: dict) -> pd.DataFrame:
    verbose = bool(params.get("verbose", False))
    if not verbose:
        warnings.filterwarnings("ignore", category=FutureWarning)

    # NEW: 30m EMA base length (bars)
    base_len = int(params.get("m30_ema_bars", 340))

    df30 = _load_30min_any(ticker)
    dfd  = _load_daily_any(ticker)

    start = params.get("start_date"); end = params.get("end_date")
    if start or end:
        df30 = _apply_window_ts(df30, "datetime", start, end)
        dfd  = _apply_window_date(dfd, "date", start, end)

    if df30.empty or dfd.empty:
        return pd.DataFrame(columns=["ticker","entry_dt","exit_dt","entry_px","exit_px","qty","ret_pct","fee_eur","leg"])

    # === DAILY enrich (NO Vermeulen) ===
    atr_len = int(params.get("atr_length", 14))
    dfd["ATR"] = _atr_manual(dfd, atr_len)
    dfd["ATR_sma20"] = dfd["ATR"].rolling(20).mean()
    dfd["atr_normalized"] = dfd["ATR"] / dfd["close"]
    dfd["atr_surge"] = dfd["ATR"] > dfd["ATR_sma20"]

    dfd["ema5"]  = EMAIndicator(dfd["close"], window=5).ema_indicator()
    dfd["ema20"] = EMAIndicator(dfd["close"], window=20).ema_indicator()
    dfd["ema44"] = EMAIndicator(dfd["close"], window=44).ema_indicator()
    dfd["ema5_slope_pct"]  = (dfd["ema5"]  / dfd["ema5"].shift(1))  - 1
    dfd["ema20_slope_pct"] = (dfd["ema20"] / dfd["ema20"].shift(1)) - 1

    min_ema5_slope  = float(params.get("ema5_slope_min", 0.0015))
    min_ema20_slope = float(params.get("ema20_slope_min", 0.0005))
    dfd["ema5_slope_up"]   = dfd["ema5_slope_pct"]  > min_ema5_slope
    dfd["ema20_slope_up"]  = dfd["ema20_slope_pct"] > min_ema20_slope
    dfd["ema5_crossed_up"] = (dfd["ema5"] > dfd["ema20"]) & (dfd["ema5"].shift(1) <= dfd["ema20"].shift(1))
    dfd["momentum_breakout"] = dfd["ema5_crossed_up"] & dfd["ema5_slope_up"] & dfd["ema20_slope_up"]

    # RSI gate (build once on daily; 30m uses membership)
    dfd["RSI"] = RSIIndicator(dfd["close"], window=14).rsi()
    good_days = set(dfd.loc[dfd["RSI"] < float(params.get("rsi_entry_threshold", 60.0)), "date"])

    # Pre-compute daily EMA44 slope per date and merge later to 30m
    ema44_slope_daily = dfd[["date","ema44"]].copy()
    ema44_slope_daily["ema44_slope"] = ema44_slope_daily["ema44"].diff()

    # === 30m enrich ===
    df30["EMA_BASE"] = EMAIndicator(df30["close"], window=base_len).ema_indicator()
    df30["EMA275"]   = df30["EMA_BASE"]  # legacy alias for compatibility
    buffer = float(params.get("entry_buffer_pct", 0.0)) / 100.0
    df30["date"] = df30["datetime"].dt.tz_convert(MARKET_TZ).dt.date

    # Bring daily features needed on 30m (NO daily_ok yet)
    daily_cols_for_30m = ["date","ema5","ema20","ema44","momentum_breakout","ATR","atr_normalized","atr_surge"]
    df30 = df30.merge(dfd[daily_cols_for_30m], on="date", how="left")
    df30 = df30.merge(ema44_slope_daily[["date","ema44_slope"]], on="date", how="left")

    # ---------- STREAK MODES (none | prev1 | 1of2 | 2of2) ----------
    streak_mode = str(params.get("streak_mode", "2of2")).lower()

    # last bar per day, relative to EMA275 (with buffer)
    lastbars = df30.sort_values("datetime").groupby("date", as_index=False).tail(1).copy()
    lastbars["day_close_above"] = lastbars["close"] > lastbars["EMA_BASE"] * (1.0 + buffer)

    day_status = lastbars[["date","day_close_above"]].sort_values("date").reset_index(drop=True)
    day_status["prev1_above"] = day_status["day_close_above"].shift(1).fillna(False)
    day_status["prev2_above"] = day_status["day_close_above"].shift(2).fillna(False)
    day_status["one_of_two"]  = day_status["prev1_above"] | day_status["prev2_above"]
    day_status["two_of_two"]  = day_status["prev1_above"] & day_status["prev2_above"]

    df30 = df30.merge(
        day_status[["date","prev1_above","prev2_above","one_of_two","two_of_two"]],
        on="date", how="left"
    )

    if streak_mode == "none":
        df30["streak_ok"] = True
    elif streak_mode == "prev1":
        df30["streak_ok"] = df30["prev1_above"].fillna(False)
    elif streak_mode == "1of2":
        df30["streak_ok"] = df30["one_of_two"].fillna(False)
    elif streak_mode == "2of2":
        df30["streak_ok"] = df30["two_of_two"].fillna(False)
    else:
        df30["streak_ok"] = df30["two_of_two"].fillna(False)

    # ---------- DAILY MODES (none | tight | price_above_ema20 | loose) ----------
    price_above_ema20 = df30["close"] > df30["ema20"]
    price_above_ema44 = df30["close"] > df30["ema44"]
    ema44_not_down    = (df30["ema44_slope"] >= -0.0001)

    daily_mode = str(params.get("daily_mode", "tight")).lower()
    if daily_mode == "none":
        df30["daily_ok"] = True
    elif daily_mode == "price_above_ema20":
        df30["daily_ok"] = price_above_ema20.fillna(False)
    elif daily_mode == "loose":
        # allow price > EMA20 OR EMA44 not falling
        df30["daily_ok"] = (price_above_ema20 | ema44_not_down).fillna(False)
    else:
        # "tight" default: price > EMA20 AND price > EMA44 AND EMA44 not falling
        df30["daily_ok"] = (price_above_ema20 & price_above_ema44 & ema44_not_down).fillna(False)

    # other filters (as before)
    df30["ema_alignment_ok"] = (df30["ema5"] > df30["ema20"]) & (df30["ema20"] > df30["ema44"])
    df30["rsi_ok"] = df30["date"].isin(good_days)

    # ATR filter
    atr_mode = str(params.get("atr_mode", "none")).lower()
    if atr_mode == "surge":
        df30["atr_ok"] = df30["atr_surge"].fillna(False)
    elif atr_mode == "band":
        mn = float(params.get("atr_min_norm", 0.0)); mx = float(params.get("atr_max_norm", 1.0))
        df30["atr_ok"] = df30["atr_normalized"].between(mn, mx, inclusive="both")
    else:
        df30["atr_ok"] = True

    # momentum toggle
    if bool(params.get("momentum_breakout_enabled", False)):
        df30["momentum_ok"] = df30["momentum_breakout"].eq(True)
    else:
        df30["momentum_ok"] = True

    # master entry condition
    df30["entry_ok"] = (
        df30["streak_ok"]
        & df30["rsi_ok"]
        & df30["momentum_ok"]
        & df30["daily_ok"]
        & df30["ema_alignment_ok"]
        & df30["atr_ok"]
    )

    # one signal per day
    df30 = df30.sort_values("datetime")
    df30["first_signal_today"] = df30.groupby("date")["entry_ok"].transform(lambda s: s.cumsum().eq(1))
    df30["entry_ok"] = df30["entry_ok"] & df30["first_signal_today"]

    # === backtest loop (T1/T2, fixed 50/50) ===
    cooldown_days = int(params.get("cooldown_days", 0))
    next_allowed_date = None

    t1_pct = float(params.get("t1_pct", 3.0)) / 100.0
    t2_pct = float(params.get("t2_pct", 10.0)) / 100.0
    w1, w2 = 0.50, 0.50  # fixed split

    drop_exit_pct = float(params.get("drop_exit_pct", 2.0)) / 100.0
    fees_fixed = float(params.get("fees_fixed", 3.0))
    slip_bps   = float(params.get("slip_bps", 0.0))

    trades = []
    current = None

    for _, row in df30.iterrows():
        d = row["date"]

        # open
        if current is None:
            if bool(row["entry_ok"]) and (next_allowed_date is None or d >= next_allowed_date):
                entry_raw = float(row["close"])
                entry_adj = apply_costs(entry_raw, "buy", slip_bps)
                current = {
                    "EntryTime": row["datetime"],
                    "EntryPxRaw": entry_raw,
                    "EntryPxAdj": entry_adj,
                    "T1_done": False,
                    "T2_done": False,
                    "T1_date": None,
                    "remaining_qty": 1.0,
                    "stop_level": None,
                    "entry_fee_pending": True
                }
            continue

        # manage
        high = float(row["high"]); low = float(row["low"])
        epx_raw = float(current["EntryPxRaw"]); epx_adj = float(current["EntryPxAdj"])

        t1p = epx_raw * (1.0 + t1_pct)
        t2p = epx_raw * (1.0 + t2_pct)
        dp  = epx_raw * (1.0 - drop_exit_pct)

        # T1: can be same day
        if (not current["T1_done"]) and high >= t1p:
            qty_leg = min(w1, current["remaining_qty"])
            if qty_leg > 0:
                exit_adj = apply_costs(t1p, "sell", slip_bps)
                ret = (exit_adj / epx_adj) - 1.0
                fee_eur = fees_fixed * qty_leg
                if current["entry_fee_pending"]:
                    fee_eur += fees_fixed * qty_leg
                    current["entry_fee_pending"] = False
                trades.append({
                    "ticker": ticker, "entry_dt": current["EntryTime"], "exit_dt": row["datetime"],
                    "entry_px": epx_adj, "exit_px": exit_adj, "qty": qty_leg,
                    "ret_pct": ret, "fee_eur": fee_eur, "leg": "T1"
                })
                current["remaining_qty"] -= qty_leg
            current["T1_done"] = True
            current["T1_date"] = row["datetime"].date()
            current["stop_level"] = epx_raw  # breakeven stop after T1
            continue

        # T2: ANY day AFTER T1 (not same day)
        if current["T1_done"] and (not current["T2_done"]) and (row["datetime"].date() > current["T1_date"]) and high >= t2p:
            qty_leg = min(w2, current["remaining_qty"])
            if qty_leg > 0:
                exit_adj = apply_costs(t2p, "sell", slip_bps)
                ret = (exit_adj / epx_adj) - 1.0
                fee_eur = fees_fixed * qty_leg
                if current["entry_fee_pending"]:
                    fee_eur += fees_fixed * qty_leg
                    current["entry_fee_pending"] = False
                trades.append({
                    "ticker": ticker, "entry_dt": current["EntryTime"], "exit_dt": row["datetime"],
                    "entry_px": epx_adj, "exit_px": exit_adj, "qty": qty_leg,
                    "ret_pct": ret, "fee_eur": fee_eur, "leg": "T2"
                })
                current["remaining_qty"] -= qty_leg
            current["T2_done"] = True
            current["stop_level"] = t1p  # trail stop up to T1 target after T2
            if current["remaining_qty"] <= 1e-12:
                current = None
                if cooldown_days > 0:
                    next_allowed_date = row["datetime"].date() + timedelta(days=cooldown_days)
            continue

        # protective stop / drop-exit for any residual
        if current is not None:
            stop_raw = current["stop_level"]
            stop_candidate = max(dp, stop_raw) if stop_raw is not None else dp
            if low <= stop_candidate:
                qty_leg = current["remaining_qty"]
                if qty_leg > 0:
                    exit_adj = apply_costs(stop_candidate, "sell", slip_bps)
                    ret = (exit_adj / epx_adj) - 1.0
                    fee_eur = fees_fixed * qty_leg
                    if current["entry_fee_pending"]:
                        fee_eur += fees_fixed * qty_leg
                        current["entry_fee_pending"] = False
                    trades.append({
                        "ticker": ticker, "entry_dt": current["EntryTime"], "exit_dt": row["datetime"],
                        "entry_px": epx_adj, "exit_px": exit_adj, "qty": qty_leg,
                        "ret_pct": ret, "fee_eur": fee_eur,
                        "leg": "StopExit" if stop_raw is not None else "DropExit"
                    })
                current = None
                if cooldown_days > 0:
                    next_allowed_date = row["datetime"].date() + timedelta(days=cooldown_days)
                continue

    return pd.DataFrame(trades)

# ---------------- summaries ----------------
def summarize_fills(trades: pd.DataFrame):
    if trades is None or trades.empty:
        agg = pd.DataFrame([{"tickers":0,"fills":0,"win_rate":0.0,"avg_ret":0.0,"median_ret":0.0,"avg_bars":0.0,"gross_ret":0.0}])
        by_tkr = pd.DataFrame(columns=["ticker","fills","mean","median"])
        return agg, by_tkr
    win = (trades["ret_pct"] > 0).mean()
    avg = trades["ret_pct"].mean()
    med = trades["ret_pct"].median()
    bars = max(0.0, (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds().mean() / (30*60))
    gross = (trades["ret_pct"] * trades["qty"]).sum()
    by_tkr = trades.groupby("ticker", as_index=False).apply(
        lambda g: pd.Series({
            "fills": len(g),
            "mean": (g["ret_pct"] * g["qty"]).sum() / max(g["qty"].sum(), 1e-12),
            "median": g["ret_pct"].median()
        }), include_groups=False
    ).reset_index(drop=True)
    agg = pd.DataFrame([{
        "tickers": trades["ticker"].nunique(),"fills": len(trades),
        "win_rate": win,"avg_ret": avg,"median_ret": med,"avg_bars": bars,"gross_ret": gross
    }])
    return agg, by_tkr

def summarize_trades_per_ticker(trades: pd.DataFrame, unit_capital: float) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["ticker","initial_capital","win_rate","total_return","num_trades","end_capital"])
    tr = trades.copy()
    tr["trade_id"] = tr["ticker"].astype(str) + "|" + tr["entry_dt"].astype(str)
    trade_rows = tr.groupby(["ticker","trade_id"], as_index=False).apply(
        lambda g: pd.Series({
            "entry_dt": g["entry_dt"].iloc[0],
            "exit_dt": g["exit_dt"].max(),
            "ret_trade": (g["ret_pct"] * g["qty"]).sum()
                         - (g["fee_eur"].sum() / unit_capital if "fee_eur" in g else 0.0)
        }), include_groups=False
    ).reset_index(drop=True)
    out = []
    for tkr, g in trade_rows.sort_values("exit_dt").groupby("ticker"):
        cap = unit_capital
        wins = (g["ret_trade"] > 0).sum()
        for _, r in g.iterrows():
            cap *= (1.0 + r["ret_trade"])
        out.append({
            "ticker": tkr,"initial_capital": unit_capital,"win_rate": wins / max(len(g), 1),
            "total_return": (cap / unit_capital) - 1.0,"num_trades": len(g),"end_capital": cap
        })
    return pd.DataFrame(out)

def equity_curve(trades: pd.DataFrame, unit_capital: float) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date","equity"])
    tr = trades.sort_values("exit_dt").copy()
    fees = tr["fee_eur"] if "fee_eur" in tr.columns else 0.0
    tr["pnl"] = (tr["ret_pct"] * tr["qty"] * unit_capital) - fees
    tr["equity"] = unit_capital + tr["pnl"].cumsum()
    return tr[["exit_dt","equity"]].rename(columns={"exit_dt":"date"})

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # Entries / exits
    ap.add_argument("--t1_pct", type=float, default=3.0)
    ap.add_argument("--t2_pct", type=float, default=10.0)
    ap.add_argument("--drop_exit_pct", type=float, default=2.0)
    ap.add_argument("--cooldown_days", type=int, default=0)
    ap.add_argument("--start_date", type=str, default=None)
    ap.add_argument("--end_date", type=str, default=None)
    # Filters
    ap.add_argument("--rsi_entry_threshold", type=float, default=60.0)
    ap.add_argument("--entry_buffer_pct", type=float, default=0.0)
    ap.add_argument("--ema5_slope_min", type=float, default=0.0015)
    ap.add_argument("--ema20_slope_min", type=float, default=0.0005)
    ap.add_argument("--momentum_breakout_enabled", action="store_true", default=False)
    ap.add_argument("--atr_length", type=int, default=14)
    ap.add_argument("--atr_mode", type=str, default="none", choices=["none","surge","band"])
    ap.add_argument("--atr_min_norm", type=float, default=0.0)
    ap.add_argument("--atr_max_norm", type=float, default=1.0)
    ap.add_argument("--streak_mode", type=str, default="2of2", choices=["none","prev1","1of2","2of2"])
    ap.add_argument("--daily_mode", type=str, default="tight", choices=["none","tight","price_above_ema20","loose"])
    # Costs/capital
    ap.add_argument("--fees_fixed", type=float, default=3.0)
    ap.add_argument("--slip_bps", type=float, default=0.0)
    ap.add_argument("--unit_capital", type=float, default=10000.0)
    ap.add_argument(
    "--m30_ema_bars",
    type=int,
    default=int(os.environ.get("M30_EMA_BARS", "340")),
    help="30m EMA base length in bars (default 340 ≈ 20 trading days).",
    )
    # Verbosity
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    params = dict(
        t1_pct=args.t1_pct, t2_pct=args.t2_pct,
        drop_exit_pct=args.drop_exit_pct,
        rsi_entry_threshold=args.rsi_entry_threshold,
        entry_buffer_pct=args.entry_buffer_pct,
        ema5_slope_min=args.ema5_slope_min,
        ema20_slope_min=args.ema20_slope_min,
        momentum_breakout_enabled=args.momentum_breakout_enabled,
        atr_length=args.atr_length, atr_mode=args.atr_mode,
        atr_min_norm=args.atr_min_norm, atr_max_norm=args.atr_max_norm,
        streak_mode=args.streak_mode, daily_mode=args.daily_mode,
        fees_fixed=args.fees_fixed, slip_bps=args.slip_bps,
        cooldown_days=args.cooldown_days, start_date=args.start_date, end_date=args.end_date,
        verbose=args.verbose,
        m30_ema_bars=args.m30_ema_bars,
    )
    print(f"[s50] Using 30m EMA base: {args.m30_ema_bars} bars")
    print(f"[s50] Scanning 30m inputs in: {IN30}")
    files = sorted(list(IN30.glob("*_30min_raw.csv")) + list(IN30.glob("*_30min.csv")))
    print(f"[s50] Found {len(files)} file(s)")
    if not files:
        raise SystemExit(f"No 30m inputs in {IN30} — expected *_30min_raw.csv or *_30min.csv")

    all_trades=[]
    for f in files:
        tkr = f.name.replace("_30min_raw.csv","").replace("_30min.csv","")
        try:
            tr = backtest_ticker(tkr, params)
            if tr is not None and not tr.empty:
                tr["ticker"] = tkr
                all_trades.append(tr)
        except Exception as e:
            print(f"[WARN] {tkr}: {e}")

    if not all_trades:
        raise SystemExit("No trades generated.")

    trades = pd.concat(all_trades, ignore_index=True).sort_values("exit_dt").reset_index(drop=True)

    # summaries & outputs
    agg, by_tkr = summarize_fills(trades)
    curve = equity_curve(trades, unit_capital=args.unit_capital)
    cap_by_tkr = summarize_trades_per_ticker(trades, unit_capital=args.unit_capital)

    prefix = "s50"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    trades.to_csv(OUT / f"{prefix}_trades_{ts}.csv", index=False)
    curve.to_csv(OUT / f"{prefix}_equity_curve_{ts}.csv", index=False)
    agg.to_csv(OUT / f"{prefix}_summary_{ts}.csv", index=False)
    by_tkr.to_csv(OUT / f"{prefix}_summary_by_ticker_{ts}.csv", index=False)
    cap_by_tkr.to_csv(OUT / f"{prefix}_summary_capital_by_ticker_{ts}.csv", index=False)
    print(f"[OK] {prefix} Trades   → {OUT}/{prefix}_trades_{ts}.csv")
    print(f"[OK] {prefix} Curve    → {OUT}/{prefix}_equity_curve_{ts}.csv")
    print(f"[OK] {prefix} Summary  → {OUT}/{prefix}_summary_{ts}.csv")
    print(f"[OK] {prefix} ByTicker → {OUT}/{prefix}_summary_by_ticker_{ts}.csv")
    print(f"[OK] {prefix} Cap/Tkr  → {OUT}/{prefix}_summary_capital_by_ticker_{ts}.csv")

if __name__ == "__main__":
    main()