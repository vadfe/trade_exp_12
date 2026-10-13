from __future__ import annotations

import numpy as np
import pandas as pd


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
    result["feature_ready"] = result[
        [
            "log_return",
            "atr",
            "body_atr",
            "upper_wick_atr",
            "lower_wick_atr",
            "range_atr",
            "relative_volume",
        ]
    ].notna().all(axis=1)
    return result
