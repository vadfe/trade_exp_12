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


OHLC_COLUMNS = ["open", "high", "low", "close"]


class OHLCForecastDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, context_size: int = 61, horizon: int = 3):
        if context_size < 1 or horizon < 1:
            raise ValueError("Размер контекста и горизонт должны быть положительными")
        required = {"open_time", "feature_ready", *FEATURE_COLUMNS, *OHLC_COLUMNS}
        if not required.issubset(frame.columns):
            raise ValueError(f"Нет полей: {sorted(required - set(frame.columns))}")
        times = frame["open_time"]
        if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError("Нарушен порядок времени исходных баров")
        self.frame = frame.reset_index(drop=True).copy()
        self.context_size, self.horizon = context_size, horizon
        self.features = self.frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self.prices = self.frame[OHLC_COLUMNS].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(self.prices).all() or (self.prices <= 0).any()
            or (self.prices[:, 1] < self.prices[:, [0, 3]].max(axis=1)).any()
            or (self.prices[:, 2] > self.prices[:, [0, 3]].min(axis=1)).any()
        ):
            raise ValueError("Невалидные исходные OHLC")
        count = len(frame) - context_size - horizon + 1
        if count < 1:
            raise ValueError("Недостаточно баров для контекста и будущих целей")
        ready = (
            self.frame["feature_ready"].eq(True).to_numpy()
            & np.isfinite(self.features).all(axis=1))
        valid = np.convolve(ready.astype(int), np.ones(context_size, dtype=int), mode="valid")
        self.starts = np.flatnonzero(valid[:count] == context_size)
        if not len(self.starts):
            raise ValueError("Нет примеров с полностью готовым известным контекстом")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        end = start + self.context_size
        return (
            torch.from_numpy(self.features[start:end].copy()),
            torch.from_numpy(self.prices[end:end + self.horizon].copy()),
            torch.tensor(self.prices[end - 1, 3]),
        )

    def partitions(self) -> dict[str, np.ndarray]:
        first, second = int(len(self.frame) * 0.7), int(len(self.frame) * 0.85)
        ends = self.starts + self.context_size + self.horizon
        return {
            "train": np.flatnonzero(ends <= first),
            "validation": np.flatnonzero((self.starts >= first) & (ends <= second)),
            "evaluation": np.flatnonzero(self.starts >= second),
        }

    def preview(self, index: int = 0) -> dict:
        start = int(self.starts[index])
        end = start + self.context_size
        total = end + self.horizon
        return {
            "index": index, "examples": len(self), "source_start_row": start,
            "excluded_examples": len(self.frame) - self.context_size - self.horizon + 1 - len(self),
            "context_size": self.context_size, "horizon": self.horizon,
            "feature_columns": list(FEATURE_COLUMNS), "ohlc_columns": list(OHLC_COLUMNS),
            "input_features": self.features[start:end].tolist(),
            "context_ohlc": self.prices[start:end].tolist(),
            "target_ohlc": self.prices[end:total].tolist(),
            "anchor_close": float(self.prices[end - 1, 3]),
            "context_available_at": (self.frame.available_at.iloc[end - 1].isoformat()
                                     if "available_at" in self.frame else None),
            "context_times": [value.isoformat() for value in self.frame.open_time.iloc[start:end]],
            "target_times": [value.isoformat() for value in self.frame.open_time.iloc[end:total]],
            "partition_sizes": {key: len(value) for key, value in self.partitions().items()},
            "normalization": "Значения до нормализации; scaler обучается только на train",
        }


@dataclass(frozen=True)
class OHLCFeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, dataset: OHLCForecastDataset, train_indices: np.ndarray):
        if not len(train_indices):
            raise ValueError("Нет обучающих примеров для нормализации")
        if not np.isin(train_indices, dataset.partitions()["train"]).all():
            raise ValueError("Scaler разрешено обучать только на train")
        mask = np.zeros(len(dataset.frame), dtype=bool)
        for index in train_indices:
            start = int(dataset.starts[index])
            mask[start:start + dataset.context_size] = True
        values = dataset.features[mask].astype(np.float64)
        mean, scale = values.mean(axis=0), values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("Несовместимые или нечисловые входы нормализации")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def state(self) -> dict:
        return {"feature_columns": list(FEATURE_COLUMNS),
                "mean": self.mean.tolist(), "scale": self.scale.tolist()}


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
