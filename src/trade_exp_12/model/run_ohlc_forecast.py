from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from trade_exp_12.model.dataset import FEATURE_COLUMNS
from trade_exp_12.model.dataset_v2 import OHLC_COLUMNS, OHLCFeatureScaler, OHLCForecastDataset
from trade_exp_12.model.transformer_v2 import MarketOHLCForecaster


def ohlc_metrics(predicted: np.ndarray, actual: np.ndarray, anchor: np.ndarray,
                 last_bar: np.ndarray) -> dict:
    if predicted.ndim != 3 or predicted.shape != actual.shape or predicted.shape[-1] != 4:
        raise ValueError("Нужны одинаковые массивы [N, H, 4] OHLC")
    if (not len(predicted) or anchor.shape != (len(predicted),)
            or last_bar.shape != (len(predicted), 4)):
        raise ValueError("Пустые или несовместимые данные оценки")
    if (not all(np.isfinite(a).all() for a in (predicted, actual, anchor, last_bar))
            or any((a <= 0).any() for a in (predicted, actual, anchor, last_bar))):
        raise ValueError("Нечисловая оценка OHLC")
    errors = {
        "model": np.abs(predicted - actual) / anchor[:, None, None] * 100,
        "flat": np.abs(anchor[:, None, None] - actual) / anchor[:, None, None] * 100,
        "repeat": np.abs(last_bar[:, None, :] - actual) / anchor[:, None, None] * 100,
    }
    overall = {name: float(value.mean()) for name, value in errors.items()}
    overall.update({f"{name}_close": float(value[:, :, 3].mean())
                    for name, value in errors.items()})
    best = min(overall["flat"], overall["repeat"])
    overall["improvement_vs_best_pct"] = 100 * (1 - overall["model"] / best) if best > 0 else None
    return {
        "overall": overall,
        "per_horizon": [
            {"horizon": h + 1, **{name: float(value[:, h].mean())
                                 for name, value in errors.items()},
             **{f"{name}_close": float(value[:, h, 3].mean())
                for name, value in errors.items()}}
            for h in range(predicted.shape[1])
        ],
        "per_component": [
            {"component": column, **{name: float(value[:, :, c].mean())
                                     for name, value in errors.items()}}
            for c, column in enumerate(OHLC_COLUMNS)
        ],
    }


class _NormalizedWindows(Dataset):
    def __init__(self, dataset, features, indices):
        self.dataset, self.features, self.indices = dataset, features, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = int(self.dataset.starts[self.indices[index]])
        end = start + self.dataset.context_size
        return (
            torch.from_numpy(self.features[start:end].copy()),
            torch.from_numpy(self.dataset.prices[end:end + self.dataset.horizon].copy()),
            torch.tensor(self.dataset.prices[end - 1, 3]),
        )


def _predictions(model, loader, device):
    model.eval()
    predicted, actual, anchors = [], [], []
    with torch.no_grad():
        for x, y, anchor in loader:
            output = model(x.to(device), anchor.to(device)).cpu().numpy()
            if not np.isfinite(output).all():
                raise ValueError("Модель выдала нечисловые цены")
            predicted.append(output)
            actual.append(y.numpy())
            anchors.append(anchor.numpy())
    return np.concatenate(predicted), np.concatenate(actual), np.concatenate(anchors)


def train_ohlc(frame: pd.DataFrame, output_dir: Path, metadata: dict,
               epochs: int = 50, seed: int = 42, batch_size: int = 256,
               d_model: int = 64, layers: int = 4, progress=None) -> dict:
    if epochs < 1:
        raise ValueError("Нужно хотя бы одно прохождение обучения")
    dataset = OHLCForecastDataset(frame)
    parts = dataset.partitions()
    if any(not len(indices) for indices in parts.values()):
        raise ValueError("Недостаточно истории для раздельных train/validation/evaluation")
    scaler = OHLCFeatureScaler.fit(dataset, parts["train"])
    features = np.full_like(dataset.features, np.nan)
    finite = np.isfinite(dataset.features).all(axis=1)
    features[finite] = scaler.transform(dataset.features[finite])
    train_ends = dataset.starts[parts["train"]] + dataset.context_size
    train_targets = dataset.prices[train_ends[:, None] + np.arange(dataset.horizon)]
    train_anchors = dataset.prices[train_ends - 1, 3]
    target_scale = max(float(np.median(np.abs(np.log(
        train_targets / train_anchors[:, None, None])))), 1e-4)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = {
        "n_features": len(FEATURE_COLUMNS), "context_size": 61, "horizon": 3,
        "d_model": d_model, "n_heads": 4, "layers": layers, "d_ff": d_model * 4,
        "dropout": 0.1, "target_scale": target_scale,
    }
    model = MarketOHLCForecaster(**architecture).to(device)
    train_loader = DataLoader(_NormalizedWindows(dataset, features, parts["train"]),
                              batch_size=batch_size, shuffle=True)
    validation = DataLoader(_NormalizedWindows(dataset, features, parts["validation"]),
                            batch_size=batch_size)
    evaluation_indices = parts["evaluation"][::dataset.horizon]
    evaluation = DataLoader(_NormalizedWindows(dataset, features, evaluation_indices),
                            batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    output_dir.mkdir(parents=True, exist_ok=False)
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, samples = 0.0, 0
        for x, y, anchor in train_loader:
            x, y, anchor = x.to(device), y.to(device), anchor.to(device)
            predicted = model(x, anchor)
            loss = (torch.abs(predicted - y) / anchor[:, None, None] * 100).mean()
            if not torch.isfinite(loss):
                raise ValueError("Нечисловая ошибка обучения OHLC")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(x)
            samples += len(x)
        pred, actual, anchors = _predictions(model, validation, device)
        val = float((np.abs(pred - actual) / anchors[:, None, None] * 100).mean())
        history.append({"epoch": epoch, "train_mae_pct": total / samples,
                        "validation_mae_pct": val})
        if val < best:
            best = val
            torch.save(model.state_dict(), output_dir / "model.pt")
        if progress is not None:
            progress(epoch, epochs, val)
    model.load_state_dict(torch.load(
        output_dir / "model.pt", weights_only=True, map_location=device))
    predicted, actual, anchors = _predictions(model, evaluation, device)
    last_rows = dataset.starts[evaluation_indices] + dataset.context_size - 1
    last_bars = dataset.prices[last_rows]
    metrics = ohlc_metrics(predicted, actual, anchors, last_bars)
    examples = []
    for i, index in enumerate(evaluation_indices):
        start = int(dataset.starts[index])
        end = start + dataset.context_size
        examples.append({
            "dataset_index": int(index), "source_start_row": start,
            "anchor_close": float(anchors[i]),
            "anchor_time": dataset.frame.open_time.iloc[end - 1].isoformat(),
            "context_available_at": (dataset.frame.available_at.iloc[end - 1].isoformat()
                                     if "available_at" in dataset.frame else None),
            "times": [t.isoformat() for t in dataset.frame.open_time.iloc[end:end + 3]],
            "predicted": predicted[i].tolist(), "actual": actual[i].tolist(),
            "last_bar": last_bars[i].tolist(),
        })
    split = {}
    for name, indices in parts.items():
        first = int(dataset.starts[indices[0]])
        last = int(dataset.starts[indices[-1]]) + dataset.context_size + dataset.horizon - 1
        split[name] = {
            "examples": len(indices), "first_source_row": first, "last_target_row": last,
            "first_time": dataset.frame.open_time.iloc[first].isoformat(),
            "last_time": dataset.frame.open_time.iloc[last].isoformat(),
        }
    report = {
        "schema_version": 3, "model_type": "ohlc_forecast", "metadata": metadata,
        "architecture": architecture, "seed": seed, "epochs": history,
        "best_validation_mae_pct": best, "scaler": scaler.state(),
        "split": split,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
        "evaluation_windows": len(evaluation_indices), "metrics": metrics,
        "examples": examples, "status": "historical_development_not_qualified",
        "limitations": [
            "Это ретроспективная проверка, а не независимое подтверждение сигнала.",
            "Критерии перехода к сигналам ещё не согласованы; автоматического допуска нет.",
            "Торговые издержки и исполнение этим отчётом не оцениваются.",
            "Цели — следующие три торговых бара. Из-за задержки ISS первый из них "
            "может уже идти к моменту получения данных; вход по прошедшему открытию недопустим.",
        ],
    }
    temporary = output_dir / "report.json.tmp"
    temporary.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    os.replace(temporary, output_dir / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in (
            "parquet", "output-dir", "symbol", "timeframe", "raw-version", "processed-version"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-checksum", required=True)
    parser.add_argument("--data-audit", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    args = parser.parse_args()

    def update(status, value, message):
        with sqlite3.connect(args.db) as connection:
            connection.execute(
                "UPDATE jobs SET status=?, progress=?, message=?, updated_at=? WHERE id=?",
                (status, value, message, datetime.now(UTC).isoformat(), args.job_id))

    try:
        update("running", 0.05, "Подготовка известного контекста и OHLC-целей")
        with open(args.parquet, "rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
            if checksum != args.expected_checksum:
                raise ValueError("Версия файла признаков изменилась перед обучением")
            source.seek(0)
            frame = pd.read_parquet(source)
        report_metadata = {
            "symbol": args.symbol, "timeframe": args.timeframe, "job_id": args.job_id,
            "raw_version": args.raw_version, "processed_version": args.processed_version,
            "feature_set": "relative_smc_v3", "processed_checksum": checksum,
            "data_audit": json.loads(args.data_audit),
        }
        train_ohlc(
            frame, Path(args.output_dir), report_metadata,
            epochs=args.epochs, seed=args.seed,
            progress=lambda epoch, total, val: update(
                "running", 0.1 + 0.85 * epoch / total,
                f"OHLC {epoch}/{total}: validation MAE {val:.4f}%"))
        update("completed", 1, "Прямой OHLC-прогноз проверен на отложенной части. Откройте отчёт.")
    except Exception as exc:
        update("failed", 1, str(exc))
        raise


if __name__ == "__main__":
    main()
