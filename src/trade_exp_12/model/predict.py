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
        open_time, open, high, low, close, label_true, label_pred,
        prob_up, prob_down, correct
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
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)

    # Align to the last bar of each window
    offset = window_size - 1
    sl = slice(offset, offset + len(windows))
    result = pd.DataFrame({
        "open_time": ready["open_time"].iloc[sl].values,
        "open": ready["open"].iloc[sl].values,
        "high": ready["high"].iloc[sl].values,
        "low": ready["low"].iloc[sl].values,
        "close": ready["close"].iloc[sl].values,
        "label_true": window_labels,
        "label_pred": preds,
        "label_true_name": [LABEL_NAMES[i] for i in window_labels],
        "label_pred_name": [LABEL_NAMES[i] for i in preds],
        "prob_up": probs[:, 0],
        "prob_down": probs[:, 1],
        "correct": (window_labels == preds).astype(int),
    })
    return result


def compute_signals(predictions: pd.DataFrame) -> dict:
    """Aggregate reversal predictions into chart-ready signals and PnL.

    Signals are generated only when the model predicts reversal_up or
    reversal_down. Between signals, position is held (long after reversal_up,
    short after reversal_down).  PnL is computed from these position changes.
    """
    if predictions.empty:
        return {"segments": [], "signals": [], "pnl": [], "stats": {}}

    df = predictions.copy()
    total = len(df)

    # True reversal bars (not hold)
    true_reversals = df[df["label_true_name"] != "hold"]
    pred_reversals = df[df["label_pred_name"] != "hold"]

    # Signals = only bars where model predicts a reversal
    signals = []
    for _, row in pred_reversals.iterrows():
        signals.append({
            "time": _ts(row["open_time"]),
            "direction": row["label_pred_name"],
            "close": float(row["close"]),
            "correct": bool(row["correct"]),
        })

    # True reversal markers for chart overlay
    true_markers = []
    for _, row in true_reversals.iterrows():
        true_markers.append({
            "time": _ts(row["open_time"]),
            "direction": row["label_true_name"],
            "close": float(row["close"]),
        })

    # PnL: position changes only at predicted reversals
    # long after reversal_up, short after reversal_down, flat before first signal
    closes = df["close"].values.astype(float)
    position = np.zeros(len(df))
    current_pos = 0.0  # flat until first signal
    for i in range(len(df)):
        pred = df.iloc[i]["label_pred_name"]
        if pred == "reversal_up":
            current_pos = 1.0
        elif pred == "reversal_down":
            current_pos = -1.0
        position[i] = current_pos

    returns = np.diff(closes) / np.maximum(closes[:-1], 1e-12)
    strategy_returns = position[:-1] * returns
    cumulative = np.cumsum(strategy_returns)
    cumulative = np.insert(cumulative, 0, 0.0)

    bh_returns = (closes - closes[0]) / max(closes[0], 1e-12)

    # Downsample PnL for chart (max ~500 points)
    step = max(1, len(df) // 500)
    pnl_points = []
    for idx in range(0, len(df), step):
        pnl_points.append({
            "time": _ts(df.iloc[idx]["open_time"]),
            "strategy": round(float(cumulative[idx]) * 100, 2),
            "buyhold": round(float(bh_returns[idx]) * 100, 2),
        })
    if len(df) > 0 and (len(df) - 1) % step != 0:
        pnl_points.append({
            "time": _ts(df.iloc[-1]["open_time"]),
            "strategy": round(float(cumulative[-1]) * 100, 2),
            "buyhold": round(float(bh_returns[-1]) * 100, 2),
        })

    final_pnl = round(float(cumulative[-1]) * 100, 2)
    final_bh = round(float(bh_returns[-1]) * 100, 2)

    # Detection metrics on reversal bars
    tp = int((pred_reversals["correct"] == 1).sum())
    fp = len(pred_reversals) - tp
    fn = len(true_reversals) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    # Win rate: how many predicted signals were correct
    win_rate = tp / max(len(pred_reversals), 1)

    stats = {
        "total_bars": total,
        "true_reversals": len(true_reversals),
        "pred_reversals": len(pred_reversals),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "win_rate": round(win_rate, 4),
        "strategy_pnl_pct": final_pnl,
        "buyhold_pnl_pct": final_bh,
        "excess_pnl_pct": round(final_pnl - final_bh, 2),
    }

    return {
        "signals": signals,
        "true_markers": true_markers,
        "pnl": pnl_points,
        "stats": stats,
    }


def _ts(val) -> str:
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)
