from __future__ import annotations

import math

import pandas as pd


def aggregate_candles(frame: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, int]:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(frame) <= max_points:
        return frame.reset_index(drop=True), 1
    group_size = math.ceil(len(frame) / max_points)
    groups = pd.Series(range(len(frame)), index=frame.index) // group_size
    aggregated = frame.groupby(groups, sort=True).agg(
        open_time=("open_time", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        turnover=("turnover", "sum"),
    )
    return aggregated.reset_index(drop=True), group_size


def aggregate_features(frame: pd.DataFrame, max_points: int) -> tuple[pd.DataFrame, int]:
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(frame) <= max_points:
        return frame.reset_index(drop=True), 1
    group_size = math.ceil(len(frame) / max_points)
    groups = pd.Series(range(len(frame)), index=frame.index) // group_size
    value_columns = [
        "log_return",
        "body_atr",
        "upper_wick_atr",
        "lower_wick_atr",
        "range_atr",
        "relative_volume",
    ]
    aggregation = {column: (column, "mean") for column in value_columns}
    aggregation["open_time"] = ("open_time", "last")
    aggregation["atr_percent"] = ("atr_percent", "mean")
    result = frame.groupby(groups, sort=True).agg(**aggregation)
    return result.reset_index(drop=True), group_size
