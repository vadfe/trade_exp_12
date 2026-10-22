from __future__ import annotations

import numpy as np
import pandas as pd

RELATIVE_FEATURE_DEFINITIONS = {
    "log_return": "Логарифм отношения close к предыдущему close.",
    "body_atr": "Тело (close − open), делённое на текущий известный ATR14.",
    "upper_wick_atr": "Верхняя тень, делённая на ATR14.",
    "lower_wick_atr": "Нижняя тень, делённая на ATR14.",
    "range_atr": "Диапазон (high − low), делённый на ATR14.",
    "relative_volume": "Объём / медиана объёмов предыдущих 20 завершённых баров.",
    "volume_spike_10": "Объём / средний объём предыдущих 10 завершённых баров.",
    "trend_slope_10": "Линейный наклон close за 10 баров, делённый на ATR14.",
    "trend_slope_30": "Линейный наклон close за 30 баров, делённый на ATR14.",
    "position_in_range_20": "Положение close между min low и max high последних 20 баров.",
    "atr_ratio": "Отношение ATR7 к ATR30.",
    "roc_5": "Относительное изменение close за 5 баров.",
    "roc_15": "Относительное изменение close за 15 баров.",
    "consecutive_direction": "Длина текущей серии направлений тел свечей, со знаком.",
    "avg_body_atr_5": "Среднее body_atr за 5 баров.",
    "avg_upper_wick_5": "Средняя верхняя тень в ATR за 5 баров.",
    "avg_lower_wick_5": "Средняя нижняя тень в ATR за 5 баров.",
    "avg_range_5": "Средний диапазон в ATR за 5 баров.",
    "body_deceleration": "Модуль body_atr / средний модуль body_atr за 5 баров.",
    "wick_ratio": "Верхняя тень / сумма верхней и нижней теней.",
    "avg_return_5": "Средняя лог-доходность за 5 баров.",
    "avg_return_10": "Средняя лог-доходность за 10 баров.",
}


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Vectorised rolling linear regression slope (fast, no scipy.apply)."""
    y = series.values.astype(np.float64)
    n = len(y)
    result = np.full(n, np.nan)
    # x = 0, 1, ..., w-1; precompute sums
    x = np.arange(window, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x * x).sum()
    denom = window * sum_x2 - sum_x * sum_x
    if denom == 0:
        return pd.Series(result, index=series.index)
    # cumulative sums for vectorised computation
    cum_y = np.nancumsum(y)
    cum_xy = np.nancumsum(np.arange(n, dtype=np.float64) * y)
    for i in range(window - 1, n):
        j = i - window + 1
        s_y = cum_y[i] - (cum_y[j - 1] if j > 0 else 0)
        # sum of (local_x * y) = sum of ((global_idx - j) * y)
        s_xy = cum_xy[i] - (cum_xy[j - 1] if j > 0 else 0) - j * s_y
        result[i] = (window * s_xy - sum_x * s_y) / denom
    return pd.Series(result, index=series.index)


def _consecutive_direction(body: pd.Series) -> pd.Series:
    """Count consecutive same-direction bars. Positive = up streak, negative = down."""
    sign = np.sign(body.values)
    result = np.zeros(len(sign))
    for i in range(len(sign)):
        if i == 0 or sign[i] == 0:
            result[i] = sign[i]
        elif sign[i] == sign[i - 1]:
            result[i] = result[i - 1] + sign[i]
        else:
            result[i] = sign[i]
    return pd.Series(result, index=body.index)


def add_relative_features(frame: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    result = frame.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_window, min_periods=atr_window).mean()
    prior_volume_median = result["volume"].shift(1).rolling(20, min_periods=20).median()

    # --- existing bar-level features ---
    result["log_return"] = np.log(result["close"] / previous_close)
    result["atr"] = atr
    result["body_atr"] = (result["close"] - result["open"]) / atr
    result["upper_wick_atr"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / atr
    result["lower_wick_atr"] = (
        result[["open", "close"]].min(axis=1) - result["low"]
    ) / atr
    result["range_atr"] = (result["high"] - result["low"]) / atr
    result["relative_volume"] = result["volume"] / prior_volume_median

    # --- volume spike (current / mean of last 10) ---
    vol_mean_10 = result["volume"].shift(1).rolling(10, min_periods=10).mean()
    result["volume_spike_10"] = result["volume"] / vol_mean_10

    # --- trend: slope of close over N bars, normalised by ATR ---
    result["trend_slope_10"] = _rolling_slope(result["close"], 10) / atr
    result["trend_slope_30"] = _rolling_slope(result["close"], 30) / atr

    # --- position in range: where is close in last-20 high/low corridor ---
    rolling_high_20 = result["high"].rolling(20, min_periods=20).max()
    rolling_low_20 = result["low"].rolling(20, min_periods=20).min()
    corridor = rolling_high_20 - rolling_low_20
    result["position_in_range_20"] = (result["close"] - rolling_low_20) / corridor.clip(lower=1e-10)

    # --- volatility compression/expansion: short ATR / long ATR ---
    atr_7 = true_range.rolling(7, min_periods=7).mean()
    atr_30 = true_range.rolling(30, min_periods=30).mean()
    result["atr_ratio"] = atr_7 / atr_30

    # --- momentum (rate of change) ---
    result["roc_5"] = result["close"].pct_change(5)
    result["roc_15"] = result["close"].pct_change(15)

    # --- consecutive bar direction (streak) ---
    body = result["close"] - result["open"]
    result["consecutive_direction"] = _consecutive_direction(body)

    # --- bar pattern averages (rolling means of bar characteristics) ---
    # Average body size over 5 bars — shows if bars are getting bigger or smaller
    result["avg_body_atr_5"] = result["body_atr"].rolling(5, min_periods=5).mean()
    # Average upper wick over 5 bars — long wicks = rejection
    result["avg_upper_wick_5"] = result["upper_wick_atr"].rolling(5, min_periods=5).mean()
    # Average lower wick over 5 bars
    result["avg_lower_wick_5"] = result["lower_wick_atr"].rolling(5, min_periods=5).mean()
    # Average range over 5 bars — narrowing = consolidation
    result["avg_range_5"] = result["range_atr"].rolling(5, min_periods=5).mean()

    # --- deceleration: current body vs average of last 5 ---
    # < 1 means bars are shrinking = movement slowing down
    result["body_deceleration"] = (
        result["body_atr"].abs() / result["body_atr"].abs().rolling(5, min_periods=5).mean()
    )

    # --- wick ratio: upper / (upper + lower) ---
    # Near 1 = strong rejection from above, near 0 = rejection from below
    total_wick = result["upper_wick_atr"] + result["lower_wick_atr"]
    result["wick_ratio"] = result["upper_wick_atr"] / total_wick.clip(lower=1e-10)

    # --- average log_return over 5 and 10 bars (smoothed direction) ---
    result["avg_return_5"] = result["log_return"].rolling(5, min_periods=5).mean()
    result["avg_return_10"] = result["log_return"].rolling(10, min_periods=10).mean()

    # feature_ready: all core features must be present
    result["feature_ready"] = result[
        [
            "log_return", "atr",
            "body_atr", "upper_wick_atr", "lower_wick_atr", "range_atr",
            "relative_volume", "volume_spike_10",
            "trend_slope_10", "trend_slope_30",
            "position_in_range_20", "atr_ratio",
            "roc_5", "roc_15",
            "avg_body_atr_5", "avg_upper_wick_5", "avg_lower_wick_5", "avg_range_5",
            "body_deceleration", "wick_ratio",
            "avg_return_5", "avg_return_10",
        ]
    ].notna().all(axis=1)
    return result
