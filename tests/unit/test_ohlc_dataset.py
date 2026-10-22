import numpy as np
import pandas as pd
import pytest
import torch

from trade_exp_12.model.dataset import FEATURE_COLUMNS
from trade_exp_12.model.dataset_v2 import OHLCFeatureScaler, OHLCForecastDataset


def sample_frame(rows=600):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    close = 100 + np.arange(rows) * 0.01
    frame["open_time"] = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    frame["open"], frame["close"] = close - 0.1, close
    frame["high"], frame["low"] = close + 0.2, close - 0.2
    frame["feature_ready"] = True
    return frame


def test_known_context_and_future_ohlc_are_separate():
    source = sample_frame()
    dataset = OHLCForecastDataset(source)
    inputs, target, anchor = dataset[0]
    assert inputs.shape == (61, 33)
    assert target.shape == (3, 4)
    assert anchor.item() == source.iloc[60]["close"]
    np.testing.assert_allclose(target.numpy(), source.iloc[61:64][["open", "high", "low", "close"]])
    changed = source.copy()
    changed.loc[61:, FEATURE_COLUMNS] = 1e6
    changed.loc[61:, ["open", "high", "low", "close"]] += 50
    changed.loc[61:63, "feature_ready"] = False
    changed_inputs, changed_target, changed_anchor = OHLCForecastDataset(changed)[0]
    torch.testing.assert_close(inputs, changed_inputs)
    torch.testing.assert_close(anchor, changed_anchor)
    assert not torch.equal(target, changed_target)


def test_unready_rows_are_not_removed_or_stitched():
    source = sample_frame()
    source.loc[20, "feature_ready"] = False
    dataset = OHLCForecastDataset(source)
    assert dataset.preview()["source_start_row"] == 21
    assert dataset.preview()["context_times"][0] == source.iloc[21].open_time.isoformat()
    assert not ((dataset.starts <= 20) & (dataset.starts + 61 > 20)).any()


def test_partitions_and_scaler_never_use_evaluation_rows():
    source = sample_frame()
    dataset = OHLCForecastDataset(source)
    parts = dataset.partitions()
    train_end = dataset.starts[parts["train"][-1]] + 64
    val_start = dataset.starts[parts["validation"][0]]
    val_end = dataset.starts[parts["validation"][-1]] + 64
    test_start = dataset.starts[parts["evaluation"][0]]
    assert train_end <= val_start < val_end <= test_start
    scaler = OHLCFeatureScaler.fit(dataset, parts["train"])
    source.loc[420:, FEATURE_COLUMNS] += 10000
    other = OHLCFeatureScaler.fit(OHLCForecastDataset(source), parts["train"])
    np.testing.assert_array_equal(scaler.mean, other.mean)
    np.testing.assert_array_equal(scaler.scale, other.scale)
    assert np.isfinite(scaler.transform(dataset[0][0].numpy())).all()
    assert scaler.state()["feature_columns"] == FEATURE_COLUMNS
    with pytest.raises(ValueError, match="только на train"):
        OHLCFeatureScaler.fit(dataset, parts["evaluation"])


def test_invalid_prices_times_and_short_history_are_rejected():
    with pytest.raises(ValueError, match="Недостаточно"):
        OHLCForecastDataset(sample_frame(63))
    source = sample_frame()
    source.loc[1, "open_time"] = source.loc[0, "open_time"]
    with pytest.raises(ValueError, match="порядок"):
        OHLCForecastDataset(source)
    source = sample_frame()
    source.loc[100, "high"] = 0
    with pytest.raises(ValueError, match="OHLC"):
        OHLCForecastDataset(source)
