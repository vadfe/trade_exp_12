import pandas as pd

from trade_exp_12.data.moex import TIMEFRAME_TO_ISS, parse_candles

# ISS columns order: open, close, high, low, value, volume, begin, end
ROWS = [
    [100.0, 101.0, 102.5, 99.5, 1_000_000.0, 5000,
     "2026-10-01 10:00:00", "2026-10-01 11:00:00"],
    [101.0, 100.5, 101.5, 100.0, 900_000.0, 4500,
     "2026-10-01 11:00:00", "2026-10-01 12:00:00"],
]


def test_parse_candles_maps_columns():
    frame = parse_candles(ROWS)
    assert list(frame.columns) == [
        "open_time", "open", "high", "low", "close", "volume", "turnover"]
    assert len(frame) == 2
    assert frame.iloc[0]["open"] == 100.0
    assert frame.iloc[0]["close"] == 101.0
    assert frame.iloc[0]["high"] == 102.5
    assert frame.iloc[0]["low"] == 99.5
    assert frame.iloc[0]["volume"] == 5000
    assert frame.iloc[0]["turnover"] == 1_000_000.0
    assert frame.iloc[0]["open_time"] == pd.Timestamp("2026-10-01 10:00:00", tz="UTC")


def test_parse_candles_empty():
    frame = parse_candles([])
    assert frame.empty


def test_parse_candles_dedupes_and_sorts():
    rows = ROWS[::-1] + [ROWS[0]]
    frame = parse_candles(rows)
    assert len(frame) == 2
    assert frame.iloc[0]["open_time"] < frame.iloc[1]["open_time"]


def test_timeframe_map_has_hour():
    assert TIMEFRAME_TO_ISS["60"] == 60
