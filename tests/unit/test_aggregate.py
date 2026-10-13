import pandas as pd

from trade_exp_12.data.aggregate import aggregate_candles


def test_aggregate_candles_preserves_ohlcv_semantics() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC"),
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "low": [9.0, 8.0, 11.0, 10.0, 13.0, 12.0],
            "close": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "turnover": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    result, group_size = aggregate_candles(frame, max_points=2)

    assert group_size == 3
    assert len(result) == 2
    assert result.iloc[0]["open"] == 10
    assert result.iloc[0]["high"] == 14
    assert result.iloc[0]["low"] == 8
    assert result.iloc[0]["close"] == 13
    assert result.iloc[0]["volume"] == 6
