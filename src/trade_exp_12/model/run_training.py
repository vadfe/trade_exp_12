"""Standalone training entry-point — invoked as a subprocess by the web server.

Usage::

    python -m trade_exp_12.model.run_training \\
        --parquet path/to/processed.parquet \\
        --symbol BRETTUSDT --timeframe 15 \\
        --output-root data/experiments \\
        --db data/registry/trade_exp_12.db \\
        --job-id 42 \\
        [--window-size 64] [--atr-multiplier 3.0] [--epochs 30] \\
        [--d-model 64] [--n-layers 4] [--lr 0.001]

Writes progress to the SQLite job table and emits log lines to stderr.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trade_exp_12.model.experiment import run_experiment

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
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--atr-multiplier", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    _update_job(args.db, args.job_id, "running", 0.05, "Loading data in subprocess")
    log.info("Loading %s", args.parquet)
    frame = pd.read_parquet(args.parquet)
    log.info("Loaded %d rows", len(frame))

    _update_job(args.db, args.job_id, "running", 0.1, "Preparing dataset")

    def progress(epoch: int, total: int, metrics: dict) -> None:
        pct = 0.1 + 0.85 * (epoch / total)
        msg = f"Epoch {epoch}/{total} — val_acc={metrics['val_accuracy']:.3f}"
        _update_job(args.db, args.job_id, "running", pct, msg)
        log.info("%s %sm: %s", args.symbol, args.timeframe, msg)

    try:
        result = run_experiment(
            frame,
            args.symbol,
            args.timeframe,
            Path(args.output_root),
            window_size=args.window_size,
            atr_multiplier=args.atr_multiplier,
            epochs=args.epochs,
            d_model=args.d_model,
            n_layers=args.n_layers,
            lr=args.lr,
            progress_callback=progress,
        )
        _update_job(
            args.db, args.job_id, "completed", 1.0,
            f"Test accuracy: {result.test_accuracy:.1%}",
        )
        log.info("Completed: test_acc=%.4f", result.test_accuracy)
    except Exception as exc:
        log.exception("Training failed")
        _update_job(args.db, args.job_id, "failed", 1.0, str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
