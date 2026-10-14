"""Training loop for MarketTransformer."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trade_exp_12.model.dataset import (
    LABEL_NAMES,
    NUM_CLASSES,
    PatchDataset,
    SplitInfo,
)
from trade_exp_12.model.transformer import MarketTransformer


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_per_class: dict[str, float]
    elapsed_seconds: float


@dataclass
class TrainResult:
    split: SplitInfo
    epochs: list[EpochMetrics]
    test_accuracy: float
    test_per_class: dict[str, float]
    test_confusion: list[list[int]]
    model_path: str


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float], np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for windows, labels in loader:
            windows, labels = windows.to(device), labels.to(device)
            logits = model(windows)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy(), strict=False):
                confusion[true, pred] += 1
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        class_total = confusion[i].sum()
        per_class[name] = float(confusion[i, i] / max(class_total, 1))

    # F1-score for reversal classes (skip hold=0)
    f1_sum = 0.0
    f1_count = 0
    for i in range(1, NUM_CLASSES):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_sum += f1
        f1_count += 1
    reversal_f1 = f1_sum / max(f1_count, 1)

    return avg_loss, accuracy, per_class, confusion, reversal_f1


def compute_class_weights(dataset: PatchDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    labels = dataset.labels.numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.from_numpy(weights)


def train(
    train_ds: PatchDataset,
    val_ds: PatchDataset,
    test_ds: PatchDataset,
    split_info: SplitInfo,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    atr_multiplier: float = 3.0,
    progress_callback=None,
) -> TrainResult:
    """Train the MarketTransformer and return metrics.

    Args:
        progress_callback: optional callable(epoch, total_epochs, metrics_dict)
            for real-time progress updates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = split_info.feature_count
    window_size = split_info.window_size

    model = MarketTransformer(
        n_features=n_features,
        num_classes=NUM_CLASSES,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=window_size,
    ).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    history: list[EpochMetrics] = []
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += len(labels)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc, val_per_class, _, val_f1 = _evaluate(
            model, val_loader, criterion, device,
        )
        elapsed = time.time() - t0

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=round(train_loss, 5),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 5),
            val_accuracy=round(val_acc, 4),
            val_per_class={k: round(v, 4) for k, v in val_per_class.items()},
            elapsed_seconds=round(elapsed, 2),
        )
        history.append(metrics)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if progress_callback:
            progress_callback(epoch, epochs, asdict(metrics))

    # Load best model for test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_loss, test_acc, test_per_class, test_confusion, test_f1 = _evaluate(
        model, test_loader, criterion, device,
    )

    result = TrainResult(
        split=split_info,
        epochs=history,
        test_accuracy=round(test_acc, 4),
        test_per_class={k: round(v, 4) for k, v in test_per_class.items()},
        test_confusion=test_confusion.tolist(),
        model_path=str(output_dir / "best_model.pt"),
    )

    # Save report
    report = {
        "split": asdict(split_info),
        "test_accuracy": result.test_accuracy,
        "test_per_class": result.test_per_class,
        "test_confusion": result.test_confusion,
        "epochs": [asdict(e) for e in history],
        "model_path": result.model_path,
        "hyperparameters": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "window_size": window_size,
            "atr_multiplier": atr_multiplier,
        },
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return result
