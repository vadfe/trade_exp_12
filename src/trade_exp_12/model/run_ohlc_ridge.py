"""Simple Ridge diagnostic on the same 61x33 -> 3-OHLC task.

Answers one question: does a linear model beat the Transformer (and the flat
baseline) on identical inputs/targets/splits? If not, the feature set lacks
forecastable information at this horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from trade_exp_12.model.dataset_v2 import OHLCFeatureScaler, OHLCForecastDataset
from trade_exp_12.model.run_ohlc_forecast import ohlc_metrics


def _windows(dataset: OHLCForecastDataset, features: np.ndarray,
             indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, targets, anchors = [], [], []
    for index in indices:
        start = int(dataset.starts[index])
        end = start + dataset.context_size
        rows.append(features[start:end].ravel())
        targets.append(dataset.prices[end:end + dataset.horizon])
        anchors.append(dataset.prices[end - 1, 3])
    return np.asarray(rows), np.asarray(targets), np.asarray(anchors)


def run_ridge(frame: pd.DataFrame, output_dir: Path, metadata: dict) -> dict:
    dataset = OHLCForecastDataset(frame)
    parts = dataset.partitions()
    scaler = OHLCFeatureScaler.fit(dataset, parts["train"])
    features = np.full_like(dataset.features, np.nan)
    finite = np.isfinite(dataset.features).all(axis=1)
    features[finite] = scaler.transform(dataset.features[finite])

    x_train, _, _ = _windows(dataset, features, parts["train"])
    _, y_train, anchor_train = _windows(dataset, features, parts["train"])
    target_train = np.log(y_train / anchor_train[:, None, None]).reshape(len(x_train), -1)

    model = Ridge(alpha=1.0)
    model.fit(x_train, target_train)

    eval_indices = parts["evaluation"][::dataset.horizon]
    x_eval, y_eval, anchor_eval = _windows(dataset, features, eval_indices)
    predicted = anchor_eval[:, None, None] * np.exp(
        model.predict(x_eval).reshape(-1, dataset.horizon, 4))
    invalid = int(
        ((predicted[:, :, 1] < predicted[:, :, [0, 3]].max(axis=2))
         | (predicted[:, :, 2] > predicted[:, :, [0, 3]].min(axis=2))).sum())

    last_rows = dataset.starts[eval_indices] + dataset.context_size - 1
    metrics = ohlc_metrics(
        predicted, y_eval, anchor_eval, dataset.prices[last_rows])
    report = {
        "model_type": "ridge_diagnostic", "metadata": metadata,
        "train_examples": len(x_train), "evaluation_windows": len(eval_indices),
        "invalid_candles": invalid, "metrics": metrics,
        "status": "diagnostic_not_qualified",
        "limitations": [
            "Диагностика «есть ли вообще сигнал в признаках», а не кандидат в сигналы.",
            "Те же входы, цели, разбиения и метрики, что у Transformer-отчёта.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary = output_dir / "report.json.tmp"
    temporary.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_dir / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-checksum", required=True)
    args = parser.parse_args()

    with open(args.parquet, "rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != args.expected_checksum:
            raise ValueError("Файл признаков не совпадает с контрольной суммой")
        source.seek(0)
        frame = pd.read_parquet(source)

    report = run_ridge(frame, Path(args.output_dir), {"processed_checksum": checksum})
    overall = report["metrics"]["overall"]
    print(f"windows: train {report['train_examples']}, "
          f"evaluation {report['evaluation_windows']}, "
          f"invalid candles {report['invalid_candles']}")
    for name in ("model", "flat", "repeat", "model_close", "flat_close",
                 "improvement_vs_best_pct"):
        print(f"{name}: {overall[name]:.4f}")
    for row in report["metrics"]["per_horizon"]:
        print(row)


if __name__ == "__main__":
    main()
