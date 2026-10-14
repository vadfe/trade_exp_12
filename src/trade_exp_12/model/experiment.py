"""Experiment runner — wires dataset assembly, training, and result storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trade_exp_12.model.dataset import prepare_datasets
from trade_exp_12.model.trainer import TrainResult, train


def run_experiment(
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_root: Path,
    window_size: int = 64,
    atr_multiplier: float = 3.0,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    progress_callback=None,
) -> TrainResult:
    """End-to-end experiment: features -> labels -> windows -> train -> report."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    output_dir = output_root / symbol / timeframe / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_info = prepare_datasets(
        frame, window_size=window_size, atr_multiplier=atr_multiplier
    )

    result = train(
        train_ds,
        val_ds,
        test_ds,
        split_info,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        progress_callback=progress_callback,
    )
    return result


def list_experiments(output_root: Path) -> list[dict]:
    """List all completed experiments with their reports."""
    experiments = []
    if not output_root.exists():
        return experiments
    for symbol_dir in sorted(output_root.iterdir()):
        if not symbol_dir.is_dir():
            continue
        for tf_dir in sorted(symbol_dir.iterdir()):
            if not tf_dir.is_dir():
                continue
            for run_dir in sorted(tf_dir.iterdir()):
                report_path = run_dir / "report.json"
                if not report_path.exists():
                    continue
                with open(report_path) as f:
                    report = json.load(f)
                experiments.append({
                    "symbol": symbol_dir.name,
                    "timeframe": tf_dir.name,
                    "run_id": run_dir.name,
                    "test_accuracy": report.get("test_accuracy"),
                    "test_per_class": report.get("test_per_class"),
                    "split": report.get("split"),
                    "hyperparameters": report.get("hyperparameters"),
                    "epochs": report.get("epochs"),
                    "test_confusion": report.get("test_confusion"),
                    "path": str(run_dir),
                })
    return experiments
