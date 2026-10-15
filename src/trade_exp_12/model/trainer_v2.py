"""Training loops for MAE pretraining and finetuning (v2)."""

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
from trade_exp_12.model.dataset_v2 import PretrainDataset, PretrainInfo
from trade_exp_12.model.transformer_v2 import MarketMAE

# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

@dataclass
class PretrainEpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    elapsed_seconds: float


@dataclass
class PretrainResult:
    info: PretrainInfo
    epochs: list[PretrainEpochMetrics]
    model_path: str


def pretrain(
    dataset: PretrainDataset,
    info: PretrainInfo,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    mask_ratio: float = 0.4,
    mask_mode: str = "random",
    forecast_bars: int = 8,
    progress_callback=None,
) -> PretrainResult:
    """MAE pretraining: mask timesteps and reconstruct features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 90/10 train/val split (random, not temporal — pretraining is unsupervised)
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MarketMAE(
        n_features=info.feature_count,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=info.window_size,
        mask_ratio=mask_ratio,
        mask_mode=mask_mode,
        forecast_bars=forecast_bars,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            reconstruction, target, mask = model.pretrain_forward(batch)
            # MSE only on masked positions
            mask_f = mask.unsqueeze(-1).expand_as(target).float()
            loss = (((reconstruction - target) ** 2) * mask_f).sum()
            loss = loss / mask_f.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                reconstruction, target, mask = model.pretrain_forward(batch)
                mask_f = mask.unsqueeze(-1).expand_as(target).float()
                loss = (((reconstruction - target) ** 2) * mask_f).sum()
                loss = loss / mask_f.sum().clamp(min=1)
                val_total += loss.item()
                val_n += 1
        avg_val = val_total / max(val_n, 1)

        elapsed = time.time() - t0
        metrics = PretrainEpochMetrics(
            epoch=epoch, train_loss=round(avg_train, 5),
            val_loss=round(avg_val, 5), elapsed_seconds=round(elapsed, 2),
        )
        epoch_metrics.append(metrics)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_dir / "encoder.pt")

        if progress_callback:
            progress_callback(epoch, epochs, asdict(metrics))

    # Save report
    report = {
        "type": "pretrain",
        "info": asdict(info),
        "epochs": [asdict(m) for m in epoch_metrics],
        "best_val_loss": round(best_val_loss, 5),
        "model_path": str(output_dir / "encoder.pt"),
        "hyperparameters": {
            "d_model": d_model, "n_heads": n_heads,
            "n_encoder_layers": n_encoder_layers,
            "n_decoder_layers": n_decoder_layers,
            "d_ff": d_ff, "dropout": dropout,
            "lr": lr, "batch_size": batch_size,
            "epochs": epochs, "mask_ratio": mask_ratio,
            "mask_mode": mask_mode, "forecast_bars": forecast_bars,
            "window_size": info.window_size,
        },
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return PretrainResult(
        info=info, epochs=epoch_metrics,
        model_path=str(output_dir / "encoder.pt"),
    )


# ---------------------------------------------------------------------------
# Finetuning
# ---------------------------------------------------------------------------

@dataclass
class FinetuneEpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_per_class: dict[str, float]
    elapsed_seconds: float


@dataclass
class FinetuneResult:
    split: SplitInfo
    epochs: list[FinetuneEpochMetrics]
    test_accuracy: float
    test_per_class: dict[str, float]
    test_confusion: list[list[int]]
    model_path: str


def _evaluate_ft(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float], np.ndarray, float]:
    """Evaluate finetuned model. Returns (loss, acc, per_class, confusion, reversal_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.classify(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    total = len(labels)
    accuracy = (preds == labels).sum() / max(total, 1)

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        mask = labels == i
        if mask.sum() > 0:
            per_class[name] = round(float((preds[mask] == i).sum() / mask.sum()), 4)
        else:
            per_class[name] = 0.0

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for pred_i, true_i in zip(preds, labels, strict=False):
        confusion[true_i, pred_i] += 1

    # F1 for reversal classes
    rev_tp = sum(confusion[i, i] for i in range(1, NUM_CLASSES))
    rev_fp = sum(
        confusion[j, i] for i in range(1, NUM_CLASSES)
        for j in range(NUM_CLASSES) if j != i
    )
    rev_fn = sum(
        confusion[i, j] for i in range(1, NUM_CLASSES)
        for j in range(NUM_CLASSES) if j != i
    )
    prec = rev_tp / max(rev_tp + rev_fp, 1)
    rec = rev_tp / max(rev_tp + rev_fn, 1)
    reversal_f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    avg_loss = total_loss / max(total, 1)
    return avg_loss, round(float(accuracy), 4), per_class, confusion.tolist(), round(reversal_f1, 4)


def compute_class_weights(dataset: PatchDataset) -> torch.Tensor:
    """Sqrt-inverse-frequency class weights."""
    labels = dataset.labels.numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = np.sqrt(1.0 / counts)
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.from_numpy(weights)


def finetune(
    encoder_path: str | Path,
    train_ds: PatchDataset,
    val_ds: PatchDataset,
    test_ds: PatchDataset,
    split_info: SplitInfo,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 3e-4,
    d_model: int = 64,
    n_heads: int = 4,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 2,
    d_ff: int = 256,
    dropout: float = 0.1,
    mask_ratio: float = 0.4,
    atr_multiplier: float = 3.0,
    freeze_encoder: bool = True,
    progress_callback=None,
) -> FinetuneResult:
    """Finetune pretrained encoder for reversal classification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window_size = split_info.window_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MarketMAE(
        n_features=split_info.feature_count,
        d_model=d_model, n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff, dropout=dropout,
        max_len=window_size, mask_ratio=mask_ratio,
    ).to(device)

    # Load pretrained encoder weights
    state = torch.load(encoder_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)

    # Freeze encoder: keep last 2 encoder layers + head trainable
    if freeze_encoder:
        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head always
        for param in model.head.parameters():
            param.requires_grad = True
        # Unfreeze last 2 encoder layers for adaptation
        n_layers = len(model.encoder.layers)
        for i in range(max(0, n_layers - 2), n_layers):
            for param in model.encoder.layers[i].parameters():
                param.requires_grad = True
        # Unfreeze cls_token (important for classification)
        model.cls_token.requires_grad = True

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only optimize unfrozen params
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    epoch_metrics = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model.classify(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)

        val_loss, val_acc, val_per_class, _, val_f1 = _evaluate_ft(
            model, val_loader, criterion, device,
        )

        elapsed = time.time() - t0
        metrics = FinetuneEpochMetrics(
            epoch=epoch,
            train_loss=round(total_loss / max(total, 1), 5),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 5),
            val_accuracy=round(val_acc, 4),
            val_per_class=val_per_class,
            elapsed_seconds=round(elapsed, 2),
        )
        epoch_metrics.append(metrics)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if progress_callback:
            progress_callback(epoch, epochs, asdict(metrics))

    # Test with best model
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", weights_only=True)
    )
    test_loss, test_acc, test_per_class, test_confusion, test_f1 = _evaluate_ft(
        model, test_loader, criterion, device,
    )

    report = {
        "type": "finetune",
        "split": asdict(split_info),
        "test_accuracy": test_acc,
        "test_per_class": test_per_class,
        "test_confusion": test_confusion,
        "epochs": [asdict(m) for m in epoch_metrics],
        "model_path": str(output_dir / "best_model.pt"),
        "encoder_path": str(encoder_path),
        "hyperparameters": {
            "d_model": d_model, "n_heads": n_heads,
            "n_encoder_layers": n_encoder_layers,
            "n_decoder_layers": n_decoder_layers,
            "d_ff": d_ff, "dropout": dropout,
            "lr": lr, "batch_size": batch_size,
            "epochs": epochs, "mask_ratio": mask_ratio,
            "window_size": window_size,
            "atr_multiplier": atr_multiplier,
            "freeze_encoder": freeze_encoder,
        },
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return FinetuneResult(
        split=split_info, epochs=epoch_metrics,
        test_accuracy=test_acc, test_per_class=test_per_class,
        test_confusion=test_confusion,
        model_path=str(output_dir / "best_model.pt"),
    )
