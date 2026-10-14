"""Subprocess entry-point for finetuning pretrained encoder (v2).

Usage::

    python -m trade_exp_12.model.run_finetune \
        --encoder-path data/pretrain/RUN_ID/encoder.pt \
        --parquet path/to/processed.parquet \
        --symbol BRETTUSDT --timeframe 15 \
        --output-dir data/experiments_v2/BRETTUSDT/15/RUN_ID \
        --db data/registry/trade_exp_12.db \
        --job-id 42 \
        [--window-size 64] [--atr-multiplier 3.0] [--epochs 30] \
        [--d-model 64] [--n-encoder-layers 4] [--freeze-encoder]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trade_exp_12.model.dataset_v2 import prepare_finetune_datasets
from trade_exp_12.model.trainer_v2 import finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def _update_job(db_path: str, job_id: int, status: str, progress: float, message: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status=?, progress=?, message=?, updated_at=? WHERE id=?",
            (status, progress, message, datetime.now(UTC).isoformat(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--atr-multiplier", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    args = parser.parse_args()

    freeze = args.freeze_encoder and not args.unfreeze_encoder

    _update_job(args.db, args.job_id, "running", 0.05, "Loading data for finetuning")
    log.info("Loading %s", args.parquet)
    frame = pd.read_parquet(args.parquet)
    log.info("Loaded %d rows", len(frame))

    _update_job(args.db, args.job_id, "running", 0.1, "Preparing datasets")
    train_ds, val_ds, test_ds, split_info = prepare_finetune_datasets(
        frame,
        window_size=args.window_size,
        atr_multiplier=args.atr_multiplier,
    )

    def progress(epoch: int, total: int, metrics: dict) -> None:
        pct = 0.1 + 0.85 * (epoch / total)
        msg = f"Finetune epoch {epoch}/{total} — val_acc={metrics['val_accuracy']:.3f}"
        _update_job(args.db, args.job_id, "running", pct, msg)
        log.info("%s %sm: %s", args.symbol, args.timeframe, msg)

    try:
        result = finetune(
            encoder_path=args.encoder_path,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            split_info=split_info,
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=4,
            n_encoder_layers=args.n_encoder_layers,
            n_decoder_layers=args.n_decoder_layers,
            d_ff=args.d_ff,
            dropout=0.1,
            mask_ratio=args.mask_ratio,
            atr_multiplier=args.atr_multiplier,
            freeze_encoder=freeze,
            progress_callback=progress,
        )
        _update_job(
            args.db, args.job_id, "completed", 1.0,
            f"Test accuracy: {result.test_accuracy:.1%}",
        )
        log.info("Completed: test_acc=%.4f", result.test_accuracy)
    except Exception as exc:
        log.exception("Finetuning failed")
        _update_job(args.db, args.job_id, "failed", 1.0, str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
