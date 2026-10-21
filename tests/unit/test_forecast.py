import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trade_exp_12.model.dataset import FEATURE_COLUMNS
from trade_exp_12.model.dataset_v2 import prepare_pretrain_dataset
from trade_exp_12.model.run_forecast_eval import (
    _load_model,
    _predict_tails,
    _verdict,
    forecast_metrics,
    validate_report,
)
from trade_exp_12.model.trainer_v2 import pretrain


def test_forecast_metrics_and_constant_baselines():
    actual = np.array([[0.01, -0.02], [0.02, 0.01]])
    result = forecast_metrics(actual, actual, np.zeros(2))
    assert result["overall"]["price_mae"] == 0
    assert result["overall"]["flat_mae"] > 0
    assert _verdict(result["overall"]) == "better"
    zero = forecast_metrics(np.zeros((2, 2)), actual, np.zeros(2))
    assert zero["overall"]["price_mae"] == zero["overall"]["flat_mae"]
    assert _verdict(zero["overall"]) == "none"
    assert zero["overall"]["corr"] == 0
    with pytest.raises(ValueError):
        forecast_metrics(np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
    with pytest.raises(ValueError):
        forecast_metrics(actual * np.nan, actual, np.zeros(2))


def test_old_model_rejected():
    with pytest.raises(ValueError, match="Старая модель"):
        validate_report({}, "SBER", "60")


def test_train_save_load_and_forecast(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "shares" / "SBER" / "60" / "fixture.parquet"
    path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    n = 240
    frame = pd.DataFrame(rng.normal(0, 0.001, (n, len(FEATURE_COLUMNS))),
                         columns=FEATURE_COLUMNS)
    frame["open_time"] = pd.date_range("2020-01-01", periods=n, freq="h")
    frame["close"] = 100 * np.exp(frame["log_return"].cumsum())
    frame["feature_ready"] = True
    frame.to_parquet(path)
    dataset, info = prepare_pretrain_dataset([path], window_size=16)
    np.testing.assert_allclose(dataset.windows[0, 0].numpy(),
                               frame[FEATURE_COLUMNS].iloc[0].values, rtol=1e-6)
    output = tmp_path / "run"
    pretrain(dataset, info, output, epochs=1, batch_size=32, d_model=8,
             n_encoder_layers=1, n_decoder_layers=1, d_ff=16,
             dropout=0, mask_mode="tail", forecast_bars=2)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    split = report["split"]
    assert split["train_end"] - 1 + info.window_size <= split["val_start"]
    assert split["val_end"] - 1 + info.window_size <= split["test_start"]
    with path.open("rb") as source:
        report["dataset_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    report_path.write_text(json.dumps(report))
    validate_report(report, "SBER", "60")
    with pytest.raises(ValueError, match="совпадать"):
        validate_report(report, "GAZP", "60")
    model = _load_model(str(output / "encoder.pt"), report, torch.device("cpu"))
    windows = dataset.windows[-3:].numpy()
    first, _ = _predict_tails(model, windows, torch.device("cpu"))
    changed = windows.copy()
    changed[:, -2:] = 100
    second, _ = _predict_tails(model, changed, torch.device("cpu"))
    np.testing.assert_allclose(first, second)
    target = output / "forecast_SBER_60.json"
    subprocess.run([
        sys.executable, "-m", "trade_exp_12.model.run_forecast_eval",
        "--encoder-path", str(output / "encoder.pt"),
        "--pretrain-report", str(report_path), "--target-parquet", str(path),
        "--symbol", "SBER", "--timeframe", "60", "--output", str(target),
    ], check=True, timeout=120)
    result = json.loads(target.read_text())
    assert result["run_metadata"]["eval_windows"] > 0
    assert len(result["example"]["actual_path"]) == 3
    assert len(result["metrics"]["per_horizon"]) == 2
    assert Path(target).is_file()
