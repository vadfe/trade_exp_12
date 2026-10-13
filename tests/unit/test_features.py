import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from trade_exp_12.features.relative import add_relative_features


def candles(rows: int = 80) -> pd.DataFrame:
    close = pd.Series(np.linspace(100, 120, rows))
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.7,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
            "turnover": np.linspace(100_000, 240_000, rows),
        }
    )


def test_relative_features_become_ready_after_warmup() -> None:
    result = add_relative_features(candles())

    assert not result.iloc[0]["feature_ready"]
    assert result.iloc[-1]["feature_ready"]
    assert result.iloc[-1]["range_atr"] > 0


def test_relative_features_are_causal() -> None:
    complete = add_relative_features(candles())
    prefix = add_relative_features(candles().iloc[:50])
    columns = [
        "log_return",
        "atr",
        "body_atr",
        "upper_wick_atr",
        "lower_wick_atr",
        "range_atr",
        "relative_volume",
        "feature_ready",
    ]

    assert_frame_equal(complete.loc[:49, columns], prefix.loc[:, columns])
