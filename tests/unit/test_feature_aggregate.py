import pandas as pd

from trade_exp_12.data.aggregate import aggregate_features


def test_feature_aggregation_uses_bucket_means() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
            "log_return": [1.0, 3.0, 5.0, 7.0],
            "body_atr": [1.0, 1.0, 3.0, 3.0],
            "upper_wick_atr": [0.1] * 4,
            "lower_wick_atr": [0.2] * 4,
            "range_atr": [2.0] * 4,
            "relative_volume": [1.0, 2.0, 3.0, 4.0],
            "atr_percent": [0.5, 0.7, 0.9, 1.1],
        }
    )

    result, group_size = aggregate_features(frame, max_points=2)

    assert group_size == 2
    assert result["log_return"].tolist() == [2.0, 6.0]
    assert result["relative_volume"].tolist() == [1.5, 3.5]
    assert result.iloc[0]["open_time"] == frame.iloc[1]["open_time"]
