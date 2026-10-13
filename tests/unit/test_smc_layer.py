import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from trade_exp_12.features.relative import add_relative_features
from trade_exp_12.market_structure.smc_layer import analyze_smc


def market_frame(rows: int = 320) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100 + np.sin(index / 8) * 8 + index * 0.02
    open_price = close + np.sin(index / 3) * 0.4
    return add_relative_features(
        pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2025-01-01", periods=rows, freq="15min", tz="UTC"
                ),
                "open": open_price,
                "high": np.maximum(open_price, close) + 0.8,
                "low": np.minimum(open_price, close) - 0.8,
                "close": close,
                "volume": 1000 + np.cos(index / 5) * 200,
                "turnover": close * (1000 + np.cos(index / 5) * 200),
            }
        )
    )


def test_smc_events_have_causal_known_at() -> None:
    analysis = analyze_smc(market_frame(), swing_length=10)

    assert analysis.swings
    assert all(event["known_index"] > event["pivot_index"] for event in analysis.swings)
    assert all(event["known_index"] > event["source_index"] for event in analysis.structure)
    assert all(event["known_index"] > event["source_index"] for event in analysis.fair_value_gaps)
    assert analysis.features.filter(like="smc_").shape[1] >= 10


def test_confirmed_smc_features_do_not_change_when_future_is_appended() -> None:
    complete = market_frame()
    prefix = complete.iloc[:240].copy()
    full_analysis = analyze_smc(complete, swing_length=10).features
    prefix_analysis = analyze_smc(prefix, swing_length=10).features
    columns = [
        "smc_swing_high_level",
        "smc_swing_low_level",
        "smc_bos",
        "smc_choch",
        "smc_resistance_distance_atr",
        "smc_support_distance_atr",
    ]

    for column in columns:
        assert_series_equal(
            full_analysis.loc[:219, column],
            prefix_analysis.loc[:219, column],
            check_names=False,
        )
