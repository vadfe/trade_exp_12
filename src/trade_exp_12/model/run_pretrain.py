"""Subprocess entry-point for MAE pretraining (v2).

Usage::

    python -m trade_exp_12.model.run_pretrain \
        --parquets path1.parquet path2.parquet ... \
        --output-dir data/pretrain/RUN_ID \
        --db data/registry/trade_exp_12.db \
        --job-id 42 \
        [--window-size 64] [--epochs 50] [--mask-ratio 0.4] \
        [--d-model 64] [--n-encoder-layers 4] [--n-decoder-layers 2]
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime

from trade_exp_12.model.dataset_v2 import prepare_pretrain_dataset
from trade_exp_12.model.trainer_v2 import pretrain

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
    parser.add_argument("--parquets", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-encoder-layers", type=int, default=4)
    parser.add_argument("--n-decoder-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    from pathlib import Path

    _update_job(args.db, args.job_id, "running", 0.05, "Loading data for pretraining")
    log.info("Loading %d parquets", len(args.parquets))

    dataset, info = prepare_pretrain_dataset(
        args.parquets, window_size=args.window_size,
    )
    log.info("Prepared %d windows from %d pairs", info.total_windows, info.n_pairs)
    msg = f"{info.total_windows} windows from {info.n_pairs} pairs"
    _update_job(args.db, args.job_id, "running", 0.1, msg)

    def progress(epoch: int, total: int, metrics: dict) -> None:
        pct = 0.1 + 0.85 * (epoch / total)
        msg = f"Pretrain epoch {epoch}/{total} — val_loss={metrics['val_loss']:.5f}"
        _update_job(args.db, args.job_id, "running", pct, msg)
        log.info(msg)

    try:
        result = pretrain(
            dataset, info,
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
            progress_callback=progress,
        )
        _update_job(
            args.db, args.job_id, "completed", 1.0,
            f"Pretrain done — best val_loss={result.epochs[-1].val_loss:.5f}",
        )
        log.info("Pretraining completed: %s", result.model_path)
    except Exception as exc:
        log.exception("Pretraining failed")
        _update_job(args.db, args.job_id, "failed", 1.0, str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
