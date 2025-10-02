import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

def add_vermeulen_trend(dfd: pd.DataFrame, slope_tol=0.001):
    """
    Adds a 'vermeulen_trend' column to the daily dataframe (dfd),
    using EMA5, EMA20, EMA44 and Vermeulen-style trend state logic.
    """

    # --- Compute EMAs and slope ---
    dfd = dfd.copy()
    dfd["EMA5"]  = EMAIndicator(dfd["close"], window=5).ema_indicator()
    dfd["EMA20"] = EMAIndicator(dfd["close"], window=20).ema_indicator()
    dfd["EMA44"] = EMAIndicator(dfd["close"], window=44).ema_indicator()
    dfd["EMA44slope"] = dfd["EMA44"].diff()
    dfd["EMA20slope"] = dfd["EMA20"].diff()
    dfd["EMA5slope"]  = dfd["EMA5"].diff()

    trend = []
    prev_state = "red"

    for i in range(len(dfd)):
        if i < 12:
            trend.append(prev_state)
            continue

        row = dfd.iloc[i]
        row_prev = dfd.iloc[i - 1]
        row_prev11 = dfd.iloc[i - 11]

        ema5, ema20, ema44 = row["EMA5"], row["EMA20"], row["EMA44"]
        ema5_prev, ema20_prev = row_prev["EMA5"], row_prev["EMA20"]
        ema5_prev11, ema20_prev11 = row_prev11["EMA5"], row_prev11["EMA20"]
        ema44_slope = row["EMA44slope"]
        ema20_slope = row["EMA20slope"]
        ema5_slope  = row["EMA5slope"]

        close, high, low = row["close"], row["high"], row["low"]

        ema44_up = ema44_slope > slope_tol
        ema44_flat = abs(ema44_slope) <= slope_tol
        ema20_up = ema20_slope > 0

        # === GREEN logic ===
        isGreen = (
            (ema5 > ema20 and ema20 > ema44 and close > ema44 and (ema44_up or ema44_flat)) or
            (prev_state == "green" and ema5 > ema20 and ema5 > ema44 and ema20_up) or
            (prev_state == "yellow" and ema5 > ema20 and ema5 > ema44 and ema20_up)
        )

        # === YELLOW (transition from green) ===
        isYellow = (
            prev_state == "green" and (
                ema5 < ema20 and ema20 < ema44 and ema5_slope < 0 and ema20_slope < 0 or
                high > ema20 and close < ema44
            )
        )

        # === PURPLE (transition from red) ===
        crossUpNow = ema5 > ema20 and ema5_prev <= ema20_prev
        crossUpPrev = ema5_prev > ema20_prev11
        crossOverEMA44 = ema5 > ema44 and ema5_prev <= ema44
        isPurple = (
            prev_state == "red" and
            (crossUpNow or crossUpPrev or crossOverEMA44) and
            ema5 > ema44 and low > ema44
        )

        # === RED fallback ===
        isAllowedRed = not isGreen and not isYellow and not isPurple and prev_state != "purple"

        # === State decision ===
        if prev_state == "purple":
            new_state = "green"
        elif isYellow:
            new_state = "yellow"
        elif isPurple:
            new_state = "purple"
        elif isGreen:
            new_state = "green"
        elif isAllowedRed:
            new_state = "red"
        else:
            new_state = prev_state

        trend.append(new_state)
        prev_state = new_state

    dfd["vermeulen_trend"] = trend
    return dfd