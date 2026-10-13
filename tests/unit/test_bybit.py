import pandas as pd

from trade_exp_12.data.bybit import interval_milliseconds, parse_klines


def test_parse_klines_sorts_and_converts_values() -> None:
    frame = parse_klines(
        [
            ["1700000900000", "2", "4", "1", "3", "20", "50"],
            ["1700000000000", "1", "3", "0.5", "2", "10", "20"],
        ]
    )

    assert frame["open_time"].is_monotonic_increasing
    assert frame.iloc[0]["open"] == 1
    assert pd.api.types.is_float_dtype(frame["close"])


def test_interval_milliseconds() -> None:
    assert interval_milliseconds("15") == 900_000
    assert interval_milliseconds("60") == 3_600_000
