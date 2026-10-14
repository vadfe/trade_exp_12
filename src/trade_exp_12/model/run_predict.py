"""Standalone prediction entry-point — invoked as a subprocess by the web server.

Usage::

    python -m trade_exp_12.model.run_predict \\
        --experiment-dir data/experiments/BRETTUSDT/15/20261014T... \\
        --parquet data/processed/spot/NEARUSDT/15/xxx.parquet \\
        --target-symbol NEARUSDT --target-timeframe 15

Writes the prediction result to a JSON file in the experiment directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from trade_exp_12.model.predict import compute_signals, load_model, predict_on_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--target-symbol", required=True)
    parser.add_argument("--target-timeframe", required=True)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    output_path = exp_dir / f"predictions_{args.target_symbol}_{args.target_timeframe}.json"

    log.info("Loading model from %s", exp_dir)
    model, report = load_model(exp_dir)
    hp = report["hyperparameters"]

    log.info("Loading data from %s", args.parquet)
    frame = pd.read_parquet(args.parquet)
    log.info("Loaded %d rows", len(frame))

    atr_m = hp.get("atr_multiplier", 3.0)
    log.info("Running inference (window=%d, atr=%.1f)...", hp["window_size"], atr_m)
    predictions = predict_on_frame(
        model, frame,
        window_size=hp["window_size"],
        atr_multiplier=atr_m,
    )
    log.info("Predictions: %d bars", len(predictions))

    log.info("Computing signals and PnL...")
    result = compute_signals(predictions)
    result["model_symbol"] = report.get("hyperparameters", {}).get("symbol", "?")
    result["target_symbol"] = args.target_symbol
    result["target_timeframe"] = args.target_timeframe

    with open(output_path, "w") as f:
        json.dump(result, f)
    log.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
