"""Evaluate the MAE tail-forecast: does the model actually predict the last
``forecast_bars`` of a window?

For every window the model predicts the masked tail feature vectors.  From
the predicted ``log_return`` we reconstruct a price path and compare it with
what really happened:

* direction accuracy — did the predicted bar move the right way;
* MAE of predicted log_return vs a naive baseline (next bar = last seen bar);
* per-horizon breakdown (bar +1 .. +forecast_bars);
* the most recent window drawn out as predicted-vs-actual close prices.

Usage::

    python -m trade_exp_12.model.run_forecast_eval \
        --encoder-path data/pretrain/RUN_ID/encoder.pt \
        --pretrain-report data/pretrain/RUN_ID/report.json \
        --target-parquet path/to/share.parquet \
        --symbol SBER --timeframe 60 \
        --output data/pretrain/RUN_ID/forecast_SBER_60.json \
        [--eval-windows 3000] [--db DB --job-id ID]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trade_exp_12.model.dataset import (
    FEATURE_COLUMNS,
    build_windows,
)
from trade_exp_12.model.transformer_v2 import MarketMAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

LR_IDX = FEATURE_COLUMNS.index("log_return")


def _update_job(db_path: str | None, job_id: int | None,
                status: str, progress: float, message: str) -> None:
    if db_path is None or job_id is None:
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status=?, progress=?, message=?, updated_at=? WHERE id=?",
            (status, progress, message, datetime.now(UTC).isoformat(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def _load_model(encoder_path: str, report: dict, device: torch.device) -> MarketMAE:
    hp = report["hyperparameters"]
    model = MarketMAE(
        n_features=len(FEATURE_COLUMNS),
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        max_len=hp["window_size"],
        n_encoder_layers=hp["n_encoder_layers"],
        n_decoder_layers=hp["n_decoder_layers"],
        d_ff=hp["d_ff"],
        mask_mode="tail",
        forecast_bars=hp.get("forecast_bars", 8),
    )
    state = torch.load(encoder_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _predict_tails(model: MarketMAE, windows: np.ndarray,
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_lr, actual_lr) arrays of shape [N, forecast_bars]."""
    preds, actuals = [], []
    fb = model.forecast_bars
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            batch = torch.from_numpy(windows[i:i + 256]).float().to(device)
            recon, target, mask = model.pretrain_forward(batch)
            # masked positions == last fb bars in tail mode
            p = recon[:, -fb:, LR_IDX].cpu().numpy()
            t = target[:, -fb:, LR_IDX].cpu().numpy()
            m = mask[:, -fb:].cpu().numpy()
            preds.append(np.where(m, p, np.nan))
            actuals.append(np.where(m, t, np.nan))
    return np.concatenate(preds), np.concatenate(actuals)


def forecast_metrics(pred: np.ndarray, actual: np.ndarray,
                     last_lr: np.ndarray) -> dict:
    """Aggregate forecast quality.

    ``last_lr``: log_return of the last visible bar per window — the naive
    persistence baseline ("tomorrow repeats today").
    """
    if pred.ndim != 2 or pred.shape != actual.shape or not pred.size:
        raise ValueError("Forecast arrays must be nonempty and have matching shapes")
    if not all(np.isfinite(x).all() for x in (pred, actual, last_lr)):
        raise ValueError("Forecast contains nonfinite values")
    price_pred = np.expm1(np.cumsum(pred, axis=1)) * 100
    price_actual = np.expm1(np.cumsum(actual, axis=1)) * 100
    if not np.isfinite(price_pred).all():
        raise ValueError("Predicted price path overflow")
    valid = np.isfinite(pred)
    horizons = pred.shape[1]
    rows = []
    for h in range(horizons):
        m = valid[:, h]
        if not m.any():
            continue
        p, a = pred[m, h], actual[m, h]
        naive = last_lr[m]
        rows.append({
            "horizon": h + 1,
            "n": int(m.sum()),
            "price_mae": float(np.abs(price_pred[:, h] - price_actual[:, h]).mean()),
            "flat_mae": float(np.abs(price_actual[:, h]).mean()),
            "direction_acc": round(float((np.sign(p) == np.sign(a)).mean()), 4),
            "mae_model": round(float(np.abs(p - a).mean()), 6),
            "mae_naive": round(float(np.abs(naive - a).mean()), 6),
            "mae_zero": round(float(np.abs(a).mean()), 6),
            "corr": _correlation(p, a),
            "bias": round(float((p - a).mean()), 6),
        })
    allm = valid
    p_all, a_all = pred[allm], actual[allm]
    naive_all = np.broadcast_to(last_lr[:, None], pred.shape)[allm]
    return {
        "per_horizon": rows,
        "overall": {
            "n_forecasts": int(allm.sum()),
            "direction_acc": round(float((np.sign(p_all) == np.sign(a_all)).mean()), 4),
            "mae_model": round(float(np.abs(p_all - a_all).mean()), 6),
            "mae_naive": round(float(np.abs(naive_all - a_all).mean()), 6),
            "mae_zero": round(float(np.abs(a_all).mean()), 6),
            "corr": _correlation(p_all, a_all),
            "price_mae": float(np.abs(price_pred - price_actual).mean()),
            "flat_mae": float(np.abs(price_actual).mean()),
        },
    }


def _correlation(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.size < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return 0.0
    return round(float(np.corrcoef(pred, actual)[0, 1]), 4)


def validate_report(report: dict, symbol: str, timeframe: str) -> None:
    if report.get("forecast_protocol") != 2:
        raise ValueError("Старая модель: требуется переобучение без утечки будущих баров")
    if report["hyperparameters"].get("mask_mode") != "tail":
        raise ValueError("Для проверки нужна модель с tail-маскированием")
    if report["info"]["pairs"] != [f"{symbol}/{timeframe}"]:
        raise ValueError("Инструмент проверки должен совпадать с инструментом обучения")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--pretrain-report", required=True)
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-windows", type=int, default=3000)
    parser.add_argument("--atr-multiplier", type=float, default=3.0)
    parser.add_argument("--db", default=None)
    parser.add_argument("--job-id", type=int, default=None)
    args = parser.parse_args()

    _update_job(args.db, args.job_id, "running", 0.1, "Loading data")
    frame = pd.read_parquet(args.target_parquet)
    ready = frame[frame["feature_ready"]].reset_index(drop=True)

    with open(args.pretrain_report) as f:
        report = json.load(f)
    validate_report(report, args.symbol, args.timeframe)
    with open(args.target_parquet, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != report.get("dataset_sha256"):
        raise ValueError("Данные изменились после обучения. Переобучите модель на этой версии")
    window_size = report["hyperparameters"]["window_size"]
    forecast_bars = report["hyperparameters"].get("forecast_bars", 8)

    feats = ready[FEATURE_COLUMNS].values.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    dummy = np.zeros(len(feats), dtype=np.int64)
    windows, _ = build_windows(feats, dummy, window_size)
    if len(windows) != report["split"]["total_windows"]:
        raise ValueError("Window count differs from training history")
    windows = windows[report["split"]["test_start"]::forecast_bars]
    windows = windows[:max(1, args.eval_windows)]
    if not len(windows):
        raise ValueError("Нет окон в отложенной истории")
    log.info("Evaluating tail forecast on %d windows (W=%d, F=%d)",
             len(windows), window_size, forecast_bars)

    _update_job(args.db, args.job_id, "running", 0.4, "Running model")
    device = torch.device("cpu")
    model = _load_model(args.encoder_path, report, device)
    pred, actual = _predict_tails(model, windows, device)

    # naive baseline = log_return of last visible bar (position W-F-1)
    last_lr = windows[:, window_size - forecast_bars - 1, LR_IDX]

    _update_job(args.db, args.job_id, "running", 0.8, "Computing metrics")
    metrics = forecast_metrics(pred, actual, last_lr)

    # most recent window as a price-path example
    last_start = report["split"]["test_start"] + (len(windows) - 1) * forecast_bars
    anchor_idx = last_start + window_size - forecast_bars - 1
    anchor_close = float(ready["close"].iloc[anchor_idx])
    tail_times = [
        str(t) for t in ready["open_time"].iloc[anchor_idx + 1:anchor_idx + 1 + forecast_bars]
    ]
    act_lr = actual[-1]
    pr_lr = pred[-1]
    act_path, pr_path, naive_path = [anchor_close], [anchor_close], [anchor_close]
    for h in range(forecast_bars):
        a = 0.0 if np.isnan(act_lr[h]) else act_lr[h]
        p = 0.0 if np.isnan(pr_lr[h]) else pr_lr[h]
        act_path.append(act_path[-1] * float(np.exp(a)))
        pr_path.append(pr_path[-1] * float(np.exp(p)))
        naive_path.append(naive_path[-1] * float(np.exp(last_lr[-1])))

    out = {
        "schema_version": 1,
        "run_metadata": {
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "window_size": window_size,
            "forecast_bars": forecast_bars,
            "eval_windows": len(windows),
        },
        "metrics": metrics,
        "example": {
            "times": tail_times,
            "anchor_close": anchor_close,
            "actual_path": [round(v, 4) for v in act_path],
            "predicted_path": [round(v, 4) for v in pr_path],
            "naive_path": [round(v, 4) for v in naive_path],
        },
        "verdict": _verdict(metrics["overall"]),
    }
    Path(args.output).write_text(json.dumps(out, allow_nan=False), encoding="utf-8")
    log.info("Forecast eval saved: %s (dir_acc=%.3f)",
             args.output, metrics["overall"]["direction_acc"])
    _update_job(args.db, args.job_id, "completed", 1.0, "Проверка прогноза готова")


def _verdict(overall: dict) -> str:
    return "better" if overall["price_mae"] < overall["flat_mae"] else "none"


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--db" in sys.argv and "--job-id" in sys.argv:
            _update_job(
                sys.argv[sys.argv.index("--db") + 1],
                int(sys.argv[sys.argv.index("--job-id") + 1]), "failed", 1.0, str(exc),
            )
        raise
