import sys

import numpy as np
import pandas as pd
import pytest
import torch

from trade_exp_12.model.dataset import FEATURE_COLUMNS
from trade_exp_12.model.dataset_v2 import OHLCFeatureScaler, OHLCForecastDataset
from trade_exp_12.model.run_ohlc_forecast import main, ohlc_metrics, train_ohlc
from trade_exp_12.model.transformer_v2 import MarketOHLCForecaster
from trade_exp_12.registry import Registry


def test_model_accepts_only_known_context_and_produces_valid_candles():
    model = MarketOHLCForecaster(d_model=8, layers=1, d_ff=16, dropout=0).eval()
    inputs = torch.randn(2, 61, 33)
    with torch.no_grad():
        output = model(inputs, torch.tensor([100.0, 200.0]))
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all() and (output > 0).all()
    assert (output[:, :, 1] >= output[:, :, [0, 3]].max(dim=-1).values).all()
    assert (output[:, :, 2] <= output[:, :, [0, 3]].min(dim=-1).values).all()
    with pytest.raises(ValueError, match="только известный"):
        model(torch.randn(2, 64, 33), torch.tensor([100.0, 200.0]))


def test_metrics_use_all_ohlc_and_known_anchor():
    actual = np.array([
        [[100, 103, 99, 102], [102, 104, 101, 103], [103, 105, 102, 104]]], dtype=float)
    anchor = np.array([100.0])
    last = np.array([[99, 101, 98, 100]], dtype=float)
    report = ohlc_metrics(actual, actual, anchor, last)
    assert report["overall"]["model"] == 0
    assert report["overall"]["flat"] > 0
    assert len(report["per_component"]) == 4
    assert len(report["per_horizon"]) == 3
    flat = np.full_like(actual, 100)
    report = ohlc_metrics(flat, actual, anchor, last)
    assert report["overall"]["model"] == report["overall"]["flat"]


def test_direct_training_learns_predictable_ohlc_and_checkpoint(tmp_path):
    torch.set_num_threads(1)
    count = 600
    close = 100 * np.exp(np.arange(count) * 0.001)
    frame = pd.DataFrame(np.zeros((count, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    frame["open_time"] = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    frame["open"], frame["close"] = close * 0.9995, close
    frame["high"], frame["low"] = close * 1.001, close * 0.9985
    frame["feature_ready"] = True
    report = train_ohlc(frame, tmp_path / "RUN", {"symbol": "SBER", "timeframe": "60"},
                        epochs=30, d_model=8, layers=1, batch_size=64)
    assert report["model_type"] == "ohlc_forecast"
    assert report["status"] == "historical_development_not_qualified"
    assert report["metrics"]["overall"]["model"] < report["metrics"]["overall"]["flat"]
    assert report["metrics"]["overall"]["model_close"] < report["metrics"]["overall"]["flat_close"]
    source = OHLCForecastDataset(frame)
    sample = report["examples"][0]
    x, _, anchor = source[sample["dataset_index"]]
    scaler = OHLCFeatureScaler(np.array(report["scaler"]["mean"]),
                               np.array(report["scaler"]["scale"]))
    model = MarketOHLCForecaster(**report["architecture"]).eval()
    model.load_state_dict(torch.load(tmp_path / "RUN" / "model.pt", weights_only=True,
                                     map_location="cpu"))
    with torch.no_grad():
        predicted = model(torch.from_numpy(scaler.transform(x.numpy()))[None], anchor[None])
    np.testing.assert_allclose(predicted.numpy()[0], sample["predicted"], rtol=1e-6)
    split = report["split"]
    assert split["train"]["last_target_row"] < split["validation"]["first_source_row"]
    assert split["validation"]["last_target_row"] < split["evaluation"]["first_source_row"]


def test_cli_marks_changed_input_failed_before_training(tmp_path, monkeypatch):
    source = tmp_path / "changed.parquet"
    source.write_bytes(b"changed")
    registry = Registry(tmp_path / "registry.db")
    registry.initialize()
    job = registry.create_job("ohlc_forecast", "SBER", "60")
    output = tmp_path / "RUN"
    monkeypatch.setattr(sys, "argv", [
        "runner", "--parquet", str(source), "--output-dir", str(output),
        "--symbol", "SBER", "--timeframe", "60", "--raw-version", "raw",
        "--processed-version", "processed", "--expected-checksum", "incorrect",
        "--data-audit", "{}", "--db", str(registry.path), "--job-id", str(job),
    ])
    with pytest.raises(ValueError, match="изменилась"):
        main()
    assert registry.get_job(job)["status"] == "failed"
    assert not output.exists()
