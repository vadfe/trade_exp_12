"""Dataset assembly for MAE pretraining and finetuning (v2).

Pretraining: windows from ALL pairs, no labels, just features.
Finetuning:  windows from one pair with reversal labels (same as v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trade_exp_12.model.dataset import (
    FEATURE_COLUMNS,
    PatchDataset,
    SplitInfo,
    build_windows,
)


class PretrainDataset(Dataset):
    """Unlabeled windows for MAE pretraining."""

    def __init__(self, windows: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


@dataclass(frozen=True)
class PretrainInfo:
    total_windows: int
    n_pairs: int
    feature_count: int
    window_size: int
    pairs: list[str]


def _load_pair_windows(
    parquet_path: str | Path,
    window_size: int,
) -> np.ndarray | None:
    """Load a parquet, extract feature windows (no labels)."""
    p = Path(parquet_path)
    if not p.exists():
        return None
    frame = pd.read_parquet(p)
    # Add reversal labels to get feature_ready column
    # (ATR computation populates feature_ready)
    ready = frame[frame["feature_ready"]].reset_index(drop=True)

    features = ready[FEATURE_COLUMNS].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Build windows without labels (use dummy labels)
    dummy_labels = np.zeros(len(features), dtype=np.int64)
    windows, _ = build_windows(features, dummy_labels, window_size)
    return windows


def prepare_pretrain_dataset(
    parquet_paths: list[str | Path],
    window_size: int = 64,
) -> tuple[PretrainDataset, PretrainInfo]:
    """Build pretraining dataset from multiple pairs.

    Each parquet is loaded, features extracted, windows built,
    then all concatenated into one big dataset.
    """
    all_windows = []
    pair_names = []

    for path in parquet_paths:
        p = Path(path)
        # Extract pair name from path: .../BRETTUSDT/15/xxx.parquet
        pair_name = f"{p.parent.parent.name}/{p.parent.name}"
        windows = _load_pair_windows(p, window_size)
        if windows is None:
            continue
        pair_names.append(pair_name)
        all_windows.append(windows)

    combined = np.concatenate(all_windows, axis=0)
    # Shuffle to mix pairs
    perm = np.arange(len(combined))
    combined = combined[perm]

    info = PretrainInfo(
        total_windows=len(combined),
        n_pairs=len(parquet_paths),
        feature_count=len(FEATURE_COLUMNS),
        window_size=window_size,
        pairs=pair_names,
    )
    return PretrainDataset(combined), info


def prepare_finetune_datasets(
    frame: pd.DataFrame,
    window_size: int = 64,
    atr_multiplier: float = 3.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[PatchDataset, PatchDataset, PatchDataset, SplitInfo]:
    """Same as v1 prepare_datasets — for finetuning classification head."""
    from trade_exp_12.model.dataset import prepare_datasets
    return prepare_datasets(
        frame, window_size=window_size,
        atr_multiplier=atr_multiplier,
        train_ratio=train_ratio, val_ratio=val_ratio,
    )
