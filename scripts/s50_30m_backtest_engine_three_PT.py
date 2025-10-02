#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s50_30m_backtest_engine.py  — clone-friendly

- Auto-detects TARGET_BUCKET and market session from env/settings/repo layout.
- Works out of the box for both EU and US clones.
"""

import os
import argparse
import warnings
from datetime import datetime, timezone, time as dt_time, timedelta
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

# ---------- bootstrap project import path ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # .../scripts -> project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- project-aware paths ----------
from common.paths import P  # dynamic project paths

# ---------- clone-friendly detection ----------
def _resolve_bucket() -> str:
    """
    Choose TARGET_BUCKET in this order: ENV -> settings -> autodetect US dir -> EU default.
    """
    # 1) ENV
    env_b = os.environ.get("TARGET_BUCKET", "").strip()
    if env_b:
        return env_b

    # 2) settings if available
    try:
        from common.settings import TARGET_BUCKET as SBUCKET  # type: ignore
        if SBUCKET:
            return str(SBUCKET)
    except Exception:
        pass

    # 3) autodetect by folder presence (prefer US if it exists)
    if (P.DATA_RAW / "targeted_ETFs_US" / "30min").exists() or (P.DATA_RAW / "targeted_ETFs_US" / "daily").exists():
        return "targeted_ETFs_US"

    # 4) fallback EU
    return "targeted_ETFs"


def _resolve_session(bucket: str) -> tuple[str, str, str]:
    """
    Decide MARKET_TZ / MARKET_OPEN / MARKET_CLOSE:
    Order: ENV -> settings -> auto by repo name/bucket -> EU default.
    """
    # 1) ENV always wins
    env_tz   = os.environ.get("MARKET_TZ", "").strip()
    env_open = os.environ.get("MARKET_OPEN", "").strip()
    env_close= os.environ.get("MARKET_CLOSE", "").strip()
    if env_tz and env_open and env_close:
        return env_tz, env_open, env_close

    # 2) settings if available (and complete)
    try:
        from common.settings import MARKET_TZ as STZ, MARKET_OPEN as SOPEN, MARKET_CLOSE as SCLOSE  # type: ignore
        if all([STZ, SOPEN, SCLOSE]):
            return str(STZ), str(SOPEN), str(SCLOSE)
    except Exception:
        pass

    # 3) auto by repo name or chosen bucket
    repo_us = PROJECT_ROOT.name.endswith("US") or bucket == "targeted_ETFs_US"
    if repo_us:
        # London session (LSE ETFs)
        return "Europe/London", "08:00", "16:30"

    # 4) EU default
    return "Europe/Madrid", "09:00", "17:30"


TARGET_BUCKET = _resolve_bucket()
MARKET_TZ, MARKET_OPEN, MARKET_CLOSE = _resolve_session(TARGET_BUCKET)

# ---------- local util for Vermeulen labels ----------
sys.path.append(str(SCRIPT_DIR))
from trend_utils import add_vermeulen_trend  # noqa: E402

ROOT = P.ROOT
IN30 = P.DATA_RAW / TARGET_BUCKET / "30min"
IN1D = P.DATA_RAW / TARGET_BUCKET / "daily"
OUT  = ROOT / "backtest_results_30m"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[s50] TARGET_BUCKET={TARGET_BUCKET} | MARKET_TZ={MARKET_TZ} "
      f"| MARKET_OPEN={MARKET_OPEN} | MARKET_CLOSE={MARKET_CLOSE}")
print(f"[s50] IN30={IN30}")
print(f"[s50] IN1D={IN1D}")

# ---------------- helpers ----------------

def _filter_session_local(df_utc: pd.DataFrame) -> pd.DataFrame:
    """
    Keep bars whose local time (MARKET_TZ) is within [MARKET_OPEN, MARKET_CLOSE].
    Assumes df_utc['datetime'] is tz-aware UTC.
    """
    if df_utc.empty:
        return df_utc

    df = df_utc.copy()
    local = df["datetime"].dt.tz_convert(MARKET_TZ)
    t = local.dt.time

    hh_o, mm_o = map(int, str(MARKET_OPEN).split(":"))
    hh_c, mm_c = map(int, str(MARKET_CLOSE).split(":"))

    mask = (t >= dt_time(hh_o, mm_o)) & (t <= dt_time(hh_c, mm_c))
    return df.loc[mask].reset_index(drop=True)

def risk_metrics_from_trades(trades: pd.DataFrame, unit_capital: float) -> dict:
    # Placeholder — keep if you use downstream
    if trades is None or trades.empty:
        return {
            "cagr": 0.0, "max_dd": 0.0, "calmar": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "trades_per_year": 0.0, "total_return": 0.0
        }
    return {}

def _load_30min_any(ticker: str) -> pd.DataFrame:
    f_raw = IN30 / f"{ticker}_30min_raw.csv"
    f_alt = IN30 / f"{ticker}_30min.csv"
    f     = f_raw if f_raw.exists() else f_alt
    if not f.exists():
        raise FileNotFoundError(f"30m file not found for {ticker} ({f_raw} / {f_alt})")
    df = pd.read_csv(f)
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    dtcol = next((c for c in df.columns if "date" in c or "time" in c), None)
    if not dtcol:
        raise ValueError(f"No datetime column in {f}")
    df["datetime"] = pd.to_datetime(df[dtcol], utc=True, errors="coerce")  # tz-aware UTC
    df = _filter_session_local(df)
    for c in ("open","high","low","close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("datetime").reset_index(drop=True)

def _load_daily_any(ticker: str) -> pd.DataFrame:
    f_raw = IN1D / f"{ticker}_daily_raw.csv"
    f_alt = IN1D / f"{ticker}_daily.csv"
    f     = f_raw if f_raw.exists() else f_alt
    if not f.exists():
        raise FileNotFoundError(f"Daily file not found for {ticker} ({f_raw} / {f_alt})")
    dfd = pd.read_csv(f)
    dfd.columns = [c.strip().lower().replace(" ", "") for c in dfd.columns]
    date_col = "date" if "date" in dfd.columns else "datetime"
    dfd["datetime"] = (
        pd.to_datetime(dfd[date_col], utc=True, errors="coerce")
          .dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    )
    dfd["date"] = dfd["datetime"].dt.date
    for c in ("open","high","low","close"):
        if c in dfd.columns:
            dfd[c] = pd.to_numeric(dfd[c], errors="coerce")
    return dfd.sort_values("datetime").reset_index(drop=True)

def _atr_manual(dfd: pd.DataFrame, n: int = 14) -> pd.Series:
    dfd = dfd.copy()
    dfd["H-L"]  = dfd["high"] - dfd["low"]
    dfd["H-PC"] = (dfd["high"] - dfd["close"].shift()).abs()
    dfd["L-PC"] = (dfd["low"] - dfd["close"].shift()).abs()
    dfd["TR"]   = dfd[["H-L","H-PC","L-PC"]].max(axis=1)
    return dfd["TR"].rolling(window=n).mean()

def apply_costs(px: float, side: str, slip_bps: float) -> float:
    if pd.isna(px) or px <= 0:
        return px
    adj = (slip_bps / 10000.0)
    return px * (1.0 + adj) if side == "buy" else px * (1.0 - adj)

def _apply_window_ts(df: pd.DataFrame, ts_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df[ts_col] >= pd.to_datetime(start, utc=True, errors="coerce")]
    if end:
        df = df[df[ts_col] <= pd.to_datetime(end, utc=True, errors="coerce")]
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
    def log(msg: str):
        if verbose:
            print(msg)

    if not verbose:
        warnings.filterwarnings("ignore", category=FutureWarning)

    df30 = _load_30min_any(ticker)
    dfd  = _load_daily_any(ticker)
    start = params.get("start_date")
    end   = params.get("end_date")

    if start or end:
        df30 = _apply_window_ts(df30, "datetime", start, end)
        dfd  = _apply_window_date(dfd, "date", start, end)

    if df30.empty or dfd.empty:
        return pd.DataFrame(columns=[
            "ticker","entry_dt","exit_dt","entry_px","exit_px","qty","ret_pct","fee_eur","leg"
        ])

    # === DAILY enrich ===
    dfd = add_vermeulen_trend(dfd)

    # ATR (configurable length)
    atr_len = int(params.get("atr_length", 14))
    dfd["ATR"] = _atr_manual(dfd, atr_len)
    dfd["ATR_sma20"] = dfd["ATR"].rolling(20).mean()
    dfd["atr_normalized"] = dfd["ATR"] / dfd["close"]
    dfd["atr_surge"] = dfd["ATR"] > dfd["ATR_sma20"]

    # EMAs & momentum
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

    ema44_slope = dfd["ema44"] - dfd["ema44"].shift(1)
    price_trend_ok = (dfd["close"] > dfd["ema20"]) & (dfd["close"] > dfd["ema44"])
    ema44_ok = ema44_slope >= -0.0001
    dfd["daily_trend_ok"] = price_trend_ok & ema44_ok
    dfd["daily_ok"] = dfd["daily_trend_ok"]

    dfd["RSI"] = RSIIndicator(dfd["close"], window=14).rsi()
    good_days = set(dfd.loc[dfd["RSI"] < float(params.get("rsi_entry_threshold", 60.0)), "date"])

    # === 30m enrich ===
    df30["EMA275"] = EMAIndicator(df30["close"], window=275).ema_indicator()
    buffer = float(params.get("entry_buffer_pct", 0.0)) / 100.0
    df30["above"] = df30["close"] > df30["EMA275"] * (1.0 + buffer)
    df30["date"] = df30["datetime"].dt.tz_convert(MARKET_TZ).dt.date

    lastbars = (
        df30.sort_values("datetime")
            .groupby("date", as_index=False)
            .tail(1)
            .copy()
    )
    lastbars["day_close_above"] = lastbars["close"] > lastbars["EMA275"] * (1.0 + buffer)

    day_status = lastbars[["date", "day_close_above"]].sort_values("date").reset_index(drop=True)
    day_status["prev1_above"] = day_status["day_close_above"].shift(1).fillna(False)
    day_status["prev2_above"] = day_status["day_close_above"].shift(2).fillna(False)
    day_status["prev2days_ok"] = day_status["prev1_above"] & day_status["prev2_above"]

    df30 = df30.merge(day_status[["date", "prev2days_ok"]], on="date", how="left")
    df30["streak_ok"] = df30["prev2days_ok"].fillna(False)

    dcols = ["date","vermeulen_trend","ema5","ema20","ema44","momentum_breakout","daily_ok",
             "ATR","atr_normalized","atr_surge"]
    df30 = df30.merge(dfd[dcols], on="date", how="left")

    df30["ema_alignment_ok"] = (df30["ema5"] > df30["ema20"]) & (df30["ema20"] > df30["ema44"])
    df30["vermeulen_ok"] = df30["vermeulen_trend"].isin(["green","purple"])
    df30["rsi_ok"] = df30["date"].isin(good_days)

    # ATR filter modes
    atr_mode = str(params.get("atr_mode", "none")).lower()
    if atr_mode == "surge":
        df30["atr_ok"] = df30["atr_surge"].fillna(False)
    elif atr_mode == "band":
        mn = float(params.get("atr_min_norm", 0.0))
        mx = float(params.get("atr_max_norm", 1.0))
        df30["atr_ok"] = df30["atr_normalized"].between(mn, mx, inclusive="both")
    else:
        df30["atr_ok"] = True

    df30["momentum_ok"] = True
    if bool(params.get("momentum_breakout_enabled", False)):
        df30["momentum_ok"] = df30["momentum_breakout"].eq(True)

    # master entry condition
    df30["all_filters_ok"] = (
        df30["streak_ok"] &
        df30["rsi_ok"] &
        df30["momentum_ok"] &
        df30["daily_ok"] &
        df30["ema_alignment_ok"] &
        df30["vermeulen_ok"] &
        df30["atr_ok"]
    )

    # one signal per day
    df30["signal_raw"] = df30["all_filters_ok"]
    df30["first_signal_today"] = (
        df30.sort_values("datetime")
            .groupby("date")["signal_raw"]
            .transform(lambda s: s.cumsum().eq(1))
    )
    df30["entry_ok"] = df30["signal_raw"] & df30["first_signal_today"]

    if verbose:
        filter_counts = {
            "streak_ok": int(df30["streak_ok"].sum()),
            "rsi_ok": int(df30["rsi_ok"].sum()),
            "momentum_ok": int(df30["momentum_ok"].sum()),
            "daily_ok": int(df30["daily_ok"].sum()),
            "ema_alignment_ok": int(df30["ema_alignment_ok"].sum()),
            "vermeulen_ok": int(df30["vermeulen_ok"].sum()),
            "atr_ok": int(df30["atr_ok"].sum()),
            "all_filters_ok": int(df30["all_filters_ok"].sum()),
            "first_signal_today": int(df30["first_signal_today"].sum()),
            "entry_ok": int(df30["entry_ok"].sum()),
        }
        print(f"[DEBUG {ticker}] " + " ".join(f"{k}={v}" for k,v in filter_counts.items()))

    # === backtest loop ===
    cooldown_days = int(params.get("cooldown_days", 0))
    next_allowed_date = None

    t1_pct = float(params.get("t1_pct", 3.0)) / 100.0
    t2_pct = float(params.get("t2_pct", 5.0)) / 100.0
    t3_pct = float(params.get("t3_pct", 8.0)) / 100.0

    w1 = float(params.get("w1", 0.50))
    w2 = float(params.get("w2", 0.25))
    w3 = float(params.get("w3", 0.25))

    drop_exit_pct = float(params.get("drop_exit_pct", 2.0)) / 100.0
    fees_fixed = float(params.get("fees_fixed", 3.0))
    slip_bps   = float(params.get("slip_bps", 0.0))

    trades = []
    current = None

    for _, row in df30.iterrows():
        d = row["date"]

        if current is None:
            if bool(row["entry_ok"]) and (next_allowed_date is None or d >= next_allowed_date):
                entry_raw = float(row["close"])
                entry_adj = apply_costs(entry_raw, "buy", slip_bps)
                current = {
                    "Ticker": ticker,
                    "EntryTime": row["datetime"],
                    "EntryPxRaw": entry_raw,
                    "EntryPxAdj": entry_adj,
                    "T1_done": False,
                    "T2_done": False,
                    "T1_date": None,
                    "T2_date": None,
                    "remaining_qty": 1.0,
                    "stop_level": None,
                    "entry_fee_pending": True
                }
            continue

        # we have an open position
        high = float(row["high"])
        low  = float(row["low"])

        epx_raw = float(current["EntryPxRaw"])
        epx_adj = float(current["EntryPxAdj"])

        t1p = epx_raw * (1.0 + t1_pct)
        t2p = epx_raw * (1.0 + t2_pct)
        t3p = epx_raw * (1.0 + t3_pct)
        dp  = epx_raw * (1.0 - drop_exit_pct)

        # T1 (same day allowed)
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
                    "ticker": ticker,
                    "entry_dt": current["EntryTime"],
                    "exit_dt": row["datetime"],
                    "entry_px": epx_adj,
                    "exit_px": exit_adj,
                    "qty": qty_leg,
                    "ret_pct": ret,
                    "fee_eur": fee_eur,
                    "leg": "T1"
                })
                current["remaining_qty"] -= qty_leg
            current["T1_done"] = True
            current["T1_date"] = row["datetime"].date()
            current["stop_level"] = epx_raw
            continue

        # T2 (next day or later)
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
                    "ticker": ticker,
                    "entry_dt": current["EntryTime"],
                    "exit_dt": row["datetime"],
                    "entry_px": epx_adj,
                    "exit_px": exit_adj,
                    "qty": qty_leg,
                    "ret_pct": ret,
                    "fee_eur": fee_eur,
                    "leg": "T2"
                })
                current["remaining_qty"] -= qty_leg
            current["T2_done"] = True
            current["T2_date"] = row["datetime"].date()
            current["stop_level"] = t1p
            continue

        # T3 (next day or later) → exit all remaining
        if current["T2_done"] and (row["datetime"].date() > current["T2_date"]) and high >= t3p:
            qty_leg = current["remaining_qty"]
            if qty_leg > 0:
                exit_adj = apply_costs(t3p, "sell", slip_bps)
                ret = (exit_adj / epx_adj) - 1.0
                fee_eur = fees_fixed * qty_leg
                if current["entry_fee_pending"]:
                    fee_eur += fees_fixed * qty_leg
                    current["entry_fee_pending"] = False
                trades.append({
                    "ticker": ticker,
                    "entry_dt": current["EntryTime"],
                    "exit_dt": row["datetime"],
                    "entry_px": epx_adj,
                    "exit_px": exit_adj,
                    "qty": qty_leg,
                    "ret_pct": ret,
                    "fee_eur": fee_eur,
                    "leg": "T3"
                })
            current = None
            if cooldown_days > 0:
                next_allowed_date = row["datetime"].date() + timedelta(days=cooldown_days)
            continue

        # stop / drop-exit
        stop_raw = current["stop_level"]
        stop_candidate = dp if stop_raw is None else max(dp, stop_raw)
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
                    "ticker": ticker,
                    "entry_dt": current["EntryTime"],
                    "exit_dt": row["datetime"],
                    "entry_px": epx_adj,
                    "exit_px": exit_adj,
                    "qty": qty_leg,
                    "ret_pct": ret,
                    "fee_eur": fee_eur,
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
        agg = pd.DataFrame([{
            "tickers": 0, "fills": 0, "win_rate": 0.0, "avg_ret": 0.0,
            "median_ret": 0.0, "avg_bars": 0.0, "gross_ret": 0.0
        }])
        by_tkr = pd.DataFrame(columns=["ticker","fills","mean","median"])
        return agg, by_tkr

    win = (trades["ret_pct"] > 0).mean()
    avg = trades["ret_pct"].mean()
    med = trades["ret_pct"].median()
    bars = max(0.0, (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds().mean() / (30 * 60))
    gross = (trades["ret_pct"] * trades["qty"]).sum()

    by_tkr = trades.groupby("ticker", as_index=False).apply(
        lambda g: pd.Series({
            "fills": len(g),
            "mean": (g["ret_pct"] * g["qty"]).sum() / max(g["qty"].sum(), 1e-12),
            "median": g["ret_pct"].median()
        }),
        include_groups=False
    ).reset_index(drop=True)

    agg = pd.DataFrame([{
        "tickers": trades["ticker"].nunique(),
        "fills": len(trades),
        "win_rate": win,
        "avg_ret": avg,
        "median_ret": med,
        "avg_bars": bars,
        "gross_ret": gross
    }])
    return agg, by_tkr


def summarize_trades_per_ticker(trades: pd.DataFrame, unit_capital: float) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=[
            "ticker","initial_capital","win_rate","total_return","num_trades","end_capital"
        ])

    tr = trades.copy()
    tr["trade_id"] = tr["ticker"].astype(str) + "|" + tr["entry_dt"].astype(str)

    trade_rows = tr.groupby(["ticker","trade_id"], as_index=False).apply(
        lambda g: pd.Series({
            "entry_dt": g["entry_dt"].iloc[0],
            "exit_dt": g["exit_dt"].max(),
            "ret_trade": (g["ret_pct"] * g["qty"]).sum()
                         - (g["fee_eur"].sum() / unit_capital if "fee_eur" in g else 0.0)
        }),
        include_groups=False
    ).reset_index(drop=True)

    out = []
    for tkr, g in trade_rows.sort_values("exit_dt").groupby("ticker"):
        cap = unit_capital
        wins = 0
        for _, r in g.iterrows():
            if r["ret_trade"] > 0:
                wins += 1
            cap *= (1.0 + r["ret_trade"])
        out.append({
            "ticker": tkr,
            "initial_capital": unit_capital,
            "win_rate": wins / max(len(g), 1),
            "total_return": (cap / unit_capital) - 1.0,
            "num_trades": len(g),
            "end_capital": cap
        })
    return pd.DataFrame(out)


def equity_curve(trades: pd.DataFrame, unit_capital: float) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date","equity"])

    tr = trades.sort_values("exit_dt").copy()
    fees = tr["fee_eur"] if "fee_eur" in tr.columns else 0.0
    tr["pnl"] = (tr["ret_pct"] * tr["qty"] * unit_capital) - fees
    tr["equity"] = unit_capital + tr["pnl"].cumsum()
    return tr[["exit_dt","equity"]].rename(columns={"exit_dt": "date"})

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    # Entries / exits
    ap.add_argument("--t1_pct", type=float, default=3.0)
    ap.add_argument("--t2_pct", type=float, default=5.0)
    ap.add_argument("--t3_pct", type=float, default=8.0)
    ap.add_argument("--w1", type=float, default=0.50)
    ap.add_argument("--w2", type=float, default=0.25)
    ap.add_argument("--w3", type=float, default=0.25)
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
    # Costs/capital
    ap.add_argument("--fees_fixed", type=float, default=3.0)
    ap.add_argument("--slip_bps", type=float, default=0.0)
    ap.add_argument("--unit_capital", type=float, default=10000.0)
    # Verbosity
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    params = dict(
        t1_pct=args.t1_pct, t2_pct=args.t2_pct, t3_pct=args.t3_pct,
        w1=args.w1, w2=args.w2, w3=args.w3,
        drop_exit_pct=args.drop_exit_pct,
        rsi_entry_threshold=args.rsi_entry_threshold,
        entry_buffer_pct=args.entry_buffer_pct,
        ema5_slope_min=args.ema5_slope_min,
        ema20_slope_min=args.ema20_slope_min,
        momentum_breakout_enabled=args.momentum_breakout_enabled,
        atr_length=args.atr_length,
        atr_mode=args.atr_mode,
        atr_min_norm=args.atr_min_norm,
        atr_max_norm=args.atr_max_norm,
        fees_fixed=args.fees_fixed,
        slip_bps=args.slip_bps,
        cooldown_days=args.cooldown_days,
        verbose=args.verbose,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"[s50] Scanning 30m inputs: {IN30}")
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

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    trades.to_csv(OUT / f"trades_{ts}.csv", index=False)
    curve.to_csv(OUT / f"equity_curve_{ts}.csv", index=False)
    agg.to_csv(OUT / f"summary_{ts}.csv", index=False)
    by_tkr.to_csv(OUT / f"summary_by_ticker_{ts}.csv", index=False)
    cap_by_tkr.to_csv(OUT / f"summary_capital_by_ticker_{ts}.csv", index=False)

    print(f"[OK] Trades   → {OUT}/trades_{ts}.csv")
    print(f"[OK] Curve    → {OUT}/equity_curve_{ts}.csv")
    print(f"[OK] Summary  → {OUT}/summary_{ts}.csv")
    print(f"[OK] ByTicker → {OUT}/summary_by_ticker_{ts}.csv")
    print(f"[OK] Cap/Tkr  → {OUT}/summary_capital_by_ticker_{ts}.csv")

if __name__ == "__main__":
    main()