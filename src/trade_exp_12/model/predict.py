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
    """Aggregate predictions into chart-ready segments and PnL.

    Returns a dict with:
        segments: list of {start, end, direction, bars, correct_pct}
        signals: list of {time, direction, close} — only at transitions
        pnl: cumulative PnL series
        stats: summary statistics
    """
    if predictions.empty:
        return {"segments": [], "signals": [], "pnl": [], "stats": {}}

    df = predictions.copy()
    total = len(df)
    correct = int(df["correct"].sum())

    # Find signal transitions: where pred changes from up<->down
    df["prev_pred"] = df["label_pred_name"].shift(1)
    df["is_signal"] = df["label_pred_name"] != df["prev_pred"]
    # Always mark first row as a signal
    df.iloc[0, df.columns.get_loc("is_signal")] = True

    # Build segments (contiguous runs of same prediction)
    segments = []
    signals = []
    seg_start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or df.iloc[i]["is_signal"]:
            seg = df.iloc[seg_start:i]
            direction = seg.iloc[0]["label_pred_name"]
            seg_correct = int(seg["correct"].sum())
            seg_dict = {
                "start": _ts(seg.iloc[0]["open_time"]),
                "end": _ts(seg.iloc[-1]["open_time"]),
                "direction": direction,
                "bars": len(seg),
                "correct_pct": round(seg_correct / len(seg), 3) if len(seg) > 0 else 0,
            }
            segments.append(seg_dict)
            signals.append({
                "time": _ts(seg.iloc[0]["open_time"]),
                "direction": direction,
                "close": float(seg.iloc[0]["close"]),
            })
            if i < len(df):
                seg_start = i

    # Simple PnL: long when UP, short when DOWN, position changes at signals
    # Returns are computed bar-by-bar on close prices
    closes = df["close"].values.astype(float)
    returns = np.diff(closes) / closes[:-1]
    position = np.where(df["label_pred_name"].values[:-1] == "up", 1.0, -1.0)
    strategy_returns = position * returns
    cumulative = np.cumsum(strategy_returns)
    cumulative = np.insert(cumulative, 0, 0.0)  # start at 0

    # Buy & hold for comparison
    bh_returns = (closes - closes[0]) / closes[0]

    # Downsample PnL for chart (max ~500 points)
    step = max(1, len(df) // 500)
    pnl_points = []
    for idx in range(0, len(df), step):
        pnl_points.append({
            "time": _ts(df.iloc[idx]["open_time"]),
            "strategy": round(float(cumulative[idx]) * 100, 2),
            "buyhold": round(float(bh_returns[idx]) * 100, 2),
        })
    # Always include last point
    if len(df) > 0 and (len(df) - 1) % step != 0:
        pnl_points.append({
            "time": _ts(df.iloc[-1]["open_time"]),
            "strategy": round(float(cumulative[-1]) * 100, 2),
            "buyhold": round(float(bh_returns[-1]) * 100, 2),
        })

    final_pnl = round(float(cumulative[-1]) * 100, 2)
    final_bh = round(float(bh_returns[-1]) * 100, 2)
    n_trades = len(signals)

    # Win rate on segments
    seg_wins = sum(1 for s in segments if s["correct_pct"] > 0.5)

    stats = {
        "total_bars": total,
        "correct": correct,
        "accuracy": round(correct / max(total, 1), 4),
        "strategy_pnl_pct": final_pnl,
        "buyhold_pnl_pct": final_bh,
        "excess_pnl_pct": round(final_pnl - final_bh, 2),
        "n_signals": n_trades,
        "n_segments": len(segments),
        "segment_win_rate": round(seg_wins / max(len(segments), 1), 3),
    }

    return {
        "segments": segments,
        "signals": signals,
        "pnl": pnl_points,
        "stats": stats,
    }


def _ts(val) -> str:
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)
