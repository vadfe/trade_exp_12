"""Assemble model-ready dataset from processed features + auto-reversal labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trade_exp_12.reversals.zigzag import detect_atr_reversals

# Feature columns used as model input (relative + SMC, all causal)
FEATURE_COLUMNS = [
    "log_return",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "range_atr",
    "relative_volume",
    "smc_resistance_distance_atr",
    "smc_support_distance_atr",
    "smc_bos",
    "smc_choch",
    "smc_fvg_signal",
    "smc_ob_signal",
    "smc_liquidity_signal",
    "smc_active_bullish_fvg_count",
    "smc_active_bearish_fvg_count",
    "smc_active_bullish_ob_count",
    "smc_active_bearish_ob_count",
]

LABEL_MAP = {"hold": 0, "reversal_up": 1, "reversal_down": 2}
LABEL_NAMES = ["hold", "reversal_up", "reversal_down"]
NUM_CLASSES = len(LABEL_MAP)


@dataclass(frozen=True)
class SplitInfo:
    train_size: int
    val_size: int
    test_size: int
    feature_count: int
    window_size: int
    label_counts: dict[str, int]


def add_reversal_labels(
    frame: pd.DataFrame, atr_multiplier: float = 3.0
) -> pd.DataFrame:
    """Add a ``label`` column based on ATR zigzag reversal detection.

    Labels:
        hold (0) — no reversal at this bar
        reversal_up (1) — confirmed price minimum (reversal upward)
        reversal_down (2) — confirmed price maximum (reversal downward)

    The label is placed on the pivot bar (the extremum), not on the
    confirmation bar.  Confirmation is used only to decide whether
    the extremum is a real reversal.
    """
    result = frame.copy()
    reversals = detect_atr_reversals(result, atr_multiplier)
    labels = np.full(len(result), LABEL_MAP["hold"], dtype=np.int64)

    times = result["open_time"]
    for reversal in reversals:
        pivot_ts = pd.Timestamp(reversal.pivot_time)
        idx_arr = times[times == pivot_ts].index
        if len(idx_arr) > 0:
            label_key = (
                "reversal_up" if reversal.direction == "up"
                else "reversal_down"
            )
            labels[idx_arr[0]] = LABEL_MAP[label_key]

    result["label"] = labels
    return result


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of shape ``[N, window_size, n_features]``.

    The label for each window is the label of the *last* bar in the window.
    Only windows where ``feature_ready`` is True for all bars are included.
    """
    n_rows, n_features = features.shape
    if n_rows < window_size:
        return np.empty((0, window_size, n_features)), np.empty(0, dtype=np.int64)
    count = n_rows - window_size + 1
    windows = np.lib.stride_tricks.sliding_window_view(features, window_size, axis=0)
    # sliding_window_view gives [count, n_features, window_size], transpose last two
    windows = windows.transpose(0, 2, 1).copy()
    window_labels = labels[window_size - 1 :].copy()
    assert len(windows) == count == len(window_labels)
    return windows, window_labels


def temporal_split(
    total: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[range, range, range]:
    """Split indices chronologically into train/val/test."""
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return range(0, train_end), range(train_end, val_end), range(val_end, total)


class PatchDataset(Dataset):
    """PyTorch dataset wrapping pre-built windows."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.labels[index]


def prepare_datasets(
    frame: pd.DataFrame,
    window_size: int = 64,
    atr_multiplier: float = 3.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[PatchDataset, PatchDataset, PatchDataset, SplitInfo]:
    """Full pipeline: frame -> labeled windows -> train/val/test datasets."""
    labeled = add_reversal_labels(frame, atr_multiplier)
    # Keep only feature-ready rows
    ready = labeled[labeled["feature_ready"]].reset_index(drop=True)
    features = ready[FEATURE_COLUMNS].values.astype(np.float32)
    labels_arr = ready["label"].values.astype(np.int64)

    # Replace NaN/inf with 0 in features
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    windows, window_labels = build_windows(features, labels_arr, window_size)
    train_range, val_range, test_range = temporal_split(
        len(windows), train_ratio, val_ratio
    )

    train_ds = PatchDataset(windows[train_range.start : train_range.stop],
                            window_labels[train_range.start : train_range.stop])
    val_ds = PatchDataset(windows[val_range.start : val_range.stop],
                          window_labels[val_range.start : val_range.stop])
    test_ds = PatchDataset(windows[test_range.start : test_range.stop],
                           window_labels[test_range.start : test_range.stop])

    unique, counts = np.unique(window_labels, return_counts=True)
    label_counts = {LABEL_NAMES[k]: int(v) for k, v in zip(unique, counts, strict=False)}

    info = SplitInfo(
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        feature_count=len(FEATURE_COLUMNS),
        window_size=window_size,
        label_counts=label_counts,
    )
    return train_ds, val_ds, test_ds, info
