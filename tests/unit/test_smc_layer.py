import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from trade_exp_12.features.relative import add_relative_features
from trade_exp_12.market_structure.smc_layer import analyze_smc
from trade_exp_12.model.dataset import FEATURE_COLUMNS


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


def test_causal_structure_and_zone_lifecycle():
    source = pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=7, freq="h", tz="UTC"),
        "open": [11, 12, 14, 13, 12, 16, 16],
        "high": [12, 13, 15, 14, 14, 17, 17],
        "low": [10, 11, 8, 10, 11, 15, 7],
        "close": [11, 12, 13, 12, 13, 16, 7.5],
        "volume": [100] * 7, "atr": [1.0] * 7, "feature_ready": [True] * 7,
    })
    result = analyze_smc(source, swing_length=2)
    assert result.features.loc[5, "smc_bos"] == 1
    assert result.features.loc[6, "smc_choch"] == -1
    assert result.features.loc[5, "smc_active_bullish_ob_count"] == 1
    assert result.features.loc[6, "smc_active_bullish_ob_count"] == 0
    assert result.features.loc[5, "smc_fvg_signal"] == 1
    assert result.features.loc[6, "smc_active_bullish_fvg_count"] == 0
    assert all(event["known_index"] == event["pivot_index"] + 2 for event in result.swings)


def test_all_model_inputs_on_random_raw_prefixes():
    for seed in [17, 42, 73]:
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(rng.normal(0, 0.01, 360).cumsum())
        open_price = np.r_[close[0], close[:-1]]
        raw = pd.DataFrame({
            "open_time": pd.date_range("2025-01-01", periods=360, freq="h", tz="UTC"),
            "open": open_price, "close": close,
            "high": np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.01, 360)),
            "low": np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.01, 360)),
            "volume": rng.integers(100, 10000, 360),
            "turnover": rng.integers(10000, 100000, 360),
        })
        complete = analyze_smc(add_relative_features(raw)).features
        for count in [100, 150, 200, 250, 300]:
            prefix = analyze_smc(add_relative_features(raw.iloc[:count])).features
            for column in [*FEATURE_COLUMNS, "feature_ready"]:
                assert_series_equal(
                    complete[column].iloc[:count], prefix[column], check_names=False,
                    obj=f"{column}, seed {seed}, prefix {count}",
                )


def test_all_model_inputs_are_prefix_invariant_up_to_last_visible_bar():
    source = market_frame()
    complete = analyze_smc(source, swing_length=10).features
    for count in [80, 100, 150, 200, 240, 300]:
        prefix = analyze_smc(source.iloc[:count], swing_length=10).features
        for column in FEATURE_COLUMNS:
            assert_series_equal(
                complete[column].iloc[:count], prefix[column], check_names=False,
                obj=f"{column} at prefix {count}",
            )
