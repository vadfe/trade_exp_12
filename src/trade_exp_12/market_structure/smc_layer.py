from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_exp_12.market_structure.causal import analyze_causal_smc

os.environ.setdefault("SMC_CREDIT", "0")
from smartmoneyconcepts import smc


@dataclass(frozen=True)
class SMCAnalysis:
    features: pd.DataFrame
    swings: list[dict[str, object]]
    structure: list[dict[str, object]]
    fair_value_gaps: list[dict[str, object]]
    order_blocks: list[dict[str, object]]
    liquidity: list[dict[str, object]]


def analyze_smc(
    frame: pd.DataFrame, swing_length: int = 20, liquidity_range: float = 0.01
) -> SMCAnalysis:
    return SMCAnalysis(**analyze_causal_smc(frame, swing_length, liquidity_range))


def analyze_smc_legacy(
    frame: pd.DataFrame, swing_length: int = 20, liquidity_range: float = 0.01
) -> SMCAnalysis:
    if swing_length < 2:
        raise ValueError("swing_length must be at least 2")
    source = frame.reset_index(drop=True).copy()
    ohlcv = source[["open", "high", "low", "close", "volume"]]
    swing_frame = smc.swing_highs_lows(ohlcv, swing_length=swing_length)
    fvg_frame = smc.fvg(ohlcv, join_consecutive=True)
    structure_frame = smc.bos_choch(ohlcv, swing_frame, close_break=True)
    ob_frame = smc.ob(ohlcv, swing_frame, close_mitigation=False)
    liquidity_frame = smc.liquidity(
        ohlcv, swing_frame, range_percent=liquidity_range
    )
    count = len(source)
    times = source["open_time"]

    swings: list[dict[str, object]] = []
    for index in swing_frame.index[swing_frame["HighLow"].notna()]:
        known_index = index + swing_length
        if known_index >= count:
            continue
        direction = int(swing_frame.at[index, "HighLow"])
        swings.append(
            {
                "direction": direction,
                "kind": "resistance" if direction == 1 else "support",
                "pivot_index": int(index),
                "known_index": int(known_index),
                "pivot_time": times.iloc[index].isoformat(),
                "known_at": times.iloc[known_index].isoformat(),
                "level": float(swing_frame.at[index, "Level"]),
            }
        )

    structure: list[dict[str, object]] = []
    for index in structure_frame.index[structure_frame["Level"].notna()]:
        broken_index = _valid_index(structure_frame.at[index, "BrokenIndex"], count)
        if broken_index is None:
            continue
        event_type = "BOS" if pd.notna(structure_frame.at[index, "BOS"]) else "CHOCH"
        direction = structure_frame.at[index, event_type]
        structure.append(
            {
                "type": event_type,
                "direction": int(direction),
                "source_index": int(index),
                "known_index": broken_index,
                "source_time": times.iloc[index].isoformat(),
                "known_at": times.iloc[broken_index].isoformat(),
                "level": float(structure_frame.at[index, "Level"]),
            }
        )

    fair_value_gaps: list[dict[str, object]] = []
    for index in fvg_frame.index[fvg_frame["FVG"].notna()]:
        known_index = index + 1
        if known_index >= count:
            continue
        end_index = _valid_index(fvg_frame.at[index, "MitigatedIndex"], count)
        if end_index is None or end_index <= known_index:
            end_index = count - 1
        fair_value_gaps.append(
            {
                "direction": int(fvg_frame.at[index, "FVG"]),
                "source_index": int(index),
                "known_index": int(known_index),
                "end_index": int(end_index),
                "source_time": times.iloc[index].isoformat(),
                "known_at": times.iloc[known_index].isoformat(),
                "end_time": times.iloc[end_index].isoformat(),
                "top": float(fvg_frame.at[index, "Top"]),
                "bottom": float(fvg_frame.at[index, "Bottom"]),
            }
        )

    order_blocks: list[dict[str, object]] = []
    for index in ob_frame.index[ob_frame["OB"].notna()]:
        direction = int(ob_frame.at[index, "OB"])
        known_index = _order_block_known_index(index, direction, structure)
        if known_index is None:
            continue
        end_index = _valid_index(ob_frame.at[index, "MitigatedIndex"], count)
        if end_index is None or end_index <= known_index:
            end_index = count - 1
        order_blocks.append(
            {
                "direction": direction,
                "source_index": int(index),
                "known_index": int(known_index),
                "end_index": int(end_index),
                "source_time": times.iloc[index].isoformat(),
                "known_at": times.iloc[known_index].isoformat(),
                "end_time": times.iloc[end_index].isoformat(),
                "top": float(ob_frame.at[index, "Top"]),
                "bottom": float(ob_frame.at[index, "Bottom"]),
                "volume": float(ob_frame.at[index, "OBVolume"]),
                "strength": float(ob_frame.at[index, "Percentage"]),
            }
        )

    liquidity: list[dict[str, object]] = []
    for index in liquidity_frame.index[liquidity_frame["Liquidity"].notna()]:
        end_pivot = _valid_index(liquidity_frame.at[index, "End"], count)
        if end_pivot is None:
            continue
        known_index = end_pivot + swing_length
        if known_index >= count:
            continue
        swept_index = _valid_index(liquidity_frame.at[index, "Swept"], count)
        if swept_index is None or swept_index <= known_index:
            swept_index = count - 1
        liquidity.append(
            {
                "direction": int(liquidity_frame.at[index, "Liquidity"]),
                "source_index": int(index),
                "known_index": int(known_index),
                "end_index": int(swept_index),
                "source_time": times.iloc[index].isoformat(),
                "known_at": times.iloc[known_index].isoformat(),
                "end_time": times.iloc[swept_index].isoformat(),
                "level": float(liquidity_frame.at[index, "Level"]),
                "swept": swept_index < count - 1,
            }
        )

    features = _feature_frame(
        source, swings, structure, fair_value_gaps, order_blocks, liquidity
    )
    return SMCAnalysis(
        features=features,
        swings=swings,
        structure=structure,
        fair_value_gaps=fair_value_gaps,
        order_blocks=order_blocks,
        liquidity=liquidity,
    )


def _valid_index(value: object, count: int) -> int | None:
    if pd.isna(value):
        return None
    index = int(value)
    return index if 0 < index < count else None


def _order_block_known_index(
    source_index: int, direction: int, structure: list[dict[str, object]]
) -> int | None:
    candidates = [
        int(event["known_index"])
        for event in structure
        if int(event["known_index"]) > source_index
        and int(event["direction"]) == direction
    ]
    return min(candidates) if candidates else None


def _feature_frame(
    source: pd.DataFrame,
    swings: list[dict[str, object]],
    structure: list[dict[str, object]],
    fvgs: list[dict[str, object]],
    order_blocks: list[dict[str, object]],
    liquidity: list[dict[str, object]],
) -> pd.DataFrame:
    result = source.copy()
    count = len(result)
    result["smc_swing_high_level"] = np.nan
    result["smc_swing_low_level"] = np.nan
    result["smc_bos"] = 0
    result["smc_choch"] = 0
    result["smc_fvg_signal"] = 0
    result["smc_ob_signal"] = 0
    result["smc_liquidity_signal"] = 0

    for event in swings:
        column = (
            "smc_swing_high_level"
            if int(event["direction"]) == 1
            else "smc_swing_low_level"
        )
        result.at[int(event["known_index"]), column] = float(event["level"])
    result["smc_swing_high_level"] = result["smc_swing_high_level"].ffill()
    result["smc_swing_low_level"] = result["smc_swing_low_level"].ffill()
    atr = result["atr"].replace(0, np.nan)
    result["smc_resistance_distance_atr"] = (
        result["smc_swing_high_level"] - result["close"]
    ) / atr
    result["smc_support_distance_atr"] = (
        result["close"] - result["smc_swing_low_level"]
    ) / atr

    for event in structure:
        column = "smc_bos" if event["type"] == "BOS" else "smc_choch"
        result.at[int(event["known_index"]), column] = int(event["direction"])
    for events, signal_column in (
        (fvgs, "smc_fvg_signal"),
        (order_blocks, "smc_ob_signal"),
        (liquidity, "smc_liquidity_signal"),
    ):
        for event in events:
            result.at[int(event["known_index"]), signal_column] = int(event["direction"])

    for events, prefix in ((fvgs, "fvg"), (order_blocks, "ob")):
        for direction, label in ((1, "bullish"), (-1, "bearish")):
            difference = np.zeros(count + 1, dtype=np.int32)
            for event in events:
                if int(event["direction"]) != direction:
                    continue
                start = int(event["known_index"])
                end = min(int(event["end_index"]), count - 1)
                difference[start] += 1
                difference[end + 1] -= 1
            result[f"smc_active_{label}_{prefix}_count"] = np.cumsum(difference[:-1])
    return result
