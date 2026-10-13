from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trade_exp_12.data.bybit import interval_milliseconds


@dataclass(frozen=True)
class Gap:
    previous_open_time: str
    next_open_time: str
    missing_candles: int


@dataclass(frozen=True)
class QualityReport:
    rows: int
    duplicates: int
    out_of_order: int
    gaps: int
    invalid_ohlc: int
    invalid_volume: int
    gap_details: tuple[Gap, ...] = ()
    invalid_times: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(
            [self.duplicates, self.out_of_order, self.invalid_ohlc, self.invalid_volume]
        )


def inspect_candles(frame: pd.DataFrame, interval: str) -> QualityReport:
    if frame.empty:
        return QualityReport(0, 0, 0, 0, 0, 0)
    times = frame["open_time"]
    differences = times.diff().dt.total_seconds().mul(1000)
    expected = interval_milliseconds(interval)
    gap_indices = differences.index[differences > expected]
    gap_details = tuple(
        Gap(
            previous_open_time=times.iloc[position - 1].isoformat(),
            next_open_time=times.iloc[position].isoformat(),
            missing_candles=int(differences.iloc[position] // expected) - 1,
        )
        for position in gap_indices
    )
    invalid_mask = (
        (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame["high"])
        | (frame["volume"] < 0)
        | (frame["turnover"] < 0)
    )
    invalid_ohlc = int(
        (
            (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame["high"])
        ).sum()
    )
    return QualityReport(
        rows=len(frame),
        duplicates=int(times.duplicated().sum()),
        out_of_order=int((differences.iloc[1:] <= 0).sum()),
        gaps=len(gap_details),
        invalid_ohlc=invalid_ohlc,
        invalid_volume=int(((frame["volume"] < 0) | (frame["turnover"] < 0)).sum()),
        gap_details=gap_details,
        invalid_times=tuple(value.isoformat() for value in times[invalid_mask]),
    )
