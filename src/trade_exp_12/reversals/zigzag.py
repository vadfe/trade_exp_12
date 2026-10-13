from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Reversal:
    direction: str
    pivot_type: str
    pivot_time: str
    known_at: str
    pivot_price: float
    confirmation_price: float
    bars_to_confirm: int
    threshold_atr: float


def detect_atr_reversals(frame: pd.DataFrame, atr_multiplier: float = 3.0) -> list[Reversal]:
    if atr_multiplier <= 0:
        raise ValueError("atr_multiplier must be positive")
    required = {"open_time", "high", "low", "close", "atr"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    valid = frame[frame["atr"].notna() & (frame["atr"] > 0)].reset_index(drop=True)
    if valid.empty:
        return []

    highs = valid["high"].to_numpy()
    lows = valid["low"].to_numpy()
    closes = valid["close"].to_numpy()
    atrs = valid["atr"].to_numpy()
    direction = 0
    high_index = low_index = 0
    high_price = float(highs[0])
    low_price = float(lows[0])
    reversals: list[Reversal] = []

    for index in range(1, len(valid)):
        high = float(highs[index])
        low = float(lows[index])
        close = float(closes[index])
        threshold = atr_multiplier * float(atrs[index])

        if high > high_price:
            high_price, high_index = high, index
        if low < low_price:
            low_price, low_index = low, index

        if direction == 0:
            rise = close - low_price
            fall = high_price - close
            if rise >= threshold and rise >= fall:
                reversals.append(
                    _reversal(
                        valid,
                        "up",
                        "low",
                        low_index,
                        index,
                        low_price,
                        close,
                        atr_multiplier,
                    )
                )
                direction = 1
                high_price, high_index = high, index
            elif fall >= threshold:
                reversals.append(
                    _reversal(
                        valid,
                        "down",
                        "high",
                        high_index,
                        index,
                        high_price,
                        close,
                        atr_multiplier,
                    )
                )
                direction = -1
                low_price, low_index = low, index
        elif direction == 1 and high_price - close >= threshold:
            reversals.append(
                _reversal(
                    valid,
                    "down",
                    "high",
                    high_index,
                    index,
                    high_price,
                    close,
                    atr_multiplier,
                )
            )
            direction = -1
            low_price, low_index = low, index
        elif direction == -1 and close - low_price >= threshold:
            reversals.append(
                _reversal(valid, "up", "low", low_index, index, low_price, close, atr_multiplier)
            )
            direction = 1
            high_price, high_index = high, index

    return reversals


def _reversal(
    frame: pd.DataFrame,
    direction: str,
    pivot_type: str,
    pivot_index: int,
    known_index: int,
    pivot_price: float,
    confirmation_price: float,
    atr_multiplier: float,
) -> Reversal:
    return Reversal(
        direction=direction,
        pivot_type=pivot_type,
        pivot_time=frame.iloc[pivot_index]["open_time"].isoformat(),
        known_at=frame.iloc[known_index]["open_time"].isoformat(),
        pivot_price=pivot_price,
        confirmation_price=confirmation_price,
        bars_to_confirm=known_index - pivot_index,
        threshold_atr=atr_multiplier,
    )


def reversals_as_dicts(reversals: list[Reversal]) -> list[dict[str, object]]:
    return [asdict(reversal) for reversal in reversals]
