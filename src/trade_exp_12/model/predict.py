"""Load a trained model and produce predictions over processed data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trade_exp_12.model.dataset import (
    FEATURE_COLUMNS,
    LABEL_NAMES,
    NUM_CLASSES,
    add_reversal_labels,
    build_windows,
)
from trade_exp_12.model.transformer import MarketTransformer


def load_model(experiment_dir: Path) -> tuple[MarketTransformer, dict]:
    """Load the best model and its report from an experiment directory."""
    report_path = experiment_dir / "report.json"
    model_path = experiment_dir / "best_model.pt"
    with open(report_path) as f:
        report = json.load(f)
    hp = report["hyperparameters"]
    model = MarketTransformer(
        n_features=report["split"]["feature_count"],
        num_classes=NUM_CLASSES,
        d_model=hp["d_model"],
        n_heads=hp.get("n_heads", 4),
        n_layers=hp["n_layers"],
        d_ff=hp.get("d_ff", hp["d_model"] * 4),
        dropout=hp.get("dropout", 0.1),
        max_len=hp["window_size"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.eval()
    return model, report


def predict_on_frame(
    model: MarketTransformer,
    frame: pd.DataFrame,
    window_size: int,
    atr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Run inference and return a DataFrame with predictions aligned to bars.

    Returns a DataFrame with columns:
        open_time, close, label_true, label_pred, prob_up, prob_down, prob_undefined, correct
    """
    labeled = add_reversal_labels(frame, atr_multiplier)
    ready = labeled[labeled["feature_ready"]].reset_index(drop=True)

    features = ready[FEATURE_COLUMNS].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels_arr = ready["label"].values.astype(np.int64)

    windows, window_labels = build_windows(features, labels_arr, window_size)
    if len(windows) == 0:
        return pd.DataFrame()

    # Run inference in batches
    all_probs = []
    all_preds = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float()
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).numpy()
            preds = logits.argmax(dim=1).numpy()
            all_probs.append(probs)
            all_preds.append(preds)

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)

    # Align to the last bar of each window
    # The first window ends at index (window_size - 1), so predictions
    # correspond to ready.iloc[window_size-1 : window_size-1+len(windows)]
    offset = window_size - 1
    result = pd.DataFrame({
        "open_time": ready["open_time"].iloc[offset : offset + len(windows)].values,
        "close": ready["close"].iloc[offset : offset + len(windows)].values,
        "label_true": window_labels,
        "label_pred": preds,
        "label_true_name": [LABEL_NAMES[i] for i in window_labels],
        "label_pred_name": [LABEL_NAMES[i] for i in preds],
        "prob_up": probs[:, 0],
        "prob_down": probs[:, 1],
        "prob_undefined": probs[:, 2] if probs.shape[1] > 2 else 0.0,
        "correct": (window_labels == preds).astype(int),
    })
    return result
