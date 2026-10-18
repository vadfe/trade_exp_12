"""Assign every bar of a target pair to an embedding cluster.

Fits KMeans on embeddings sampled from ALL parquets (consistent with the
embeddings analysis), then encodes every window of the target parquet and
assigns each bar the nearest cluster.  Output feeds the cluster-map chart.

Usage::

    python -m trade_exp_12.model.run_cluster_map \
        --encoder-path data/pretrain/RUN_ID/encoder.pt \
        --pretrain-report data/pretrain/RUN_ID/report.json \
        --parquets all1.parquet all2.parquet ... \
        --target-parquet path/to/pair.parquet \
        --output data/pretrain/RUN_ID/clustermap_BRETTUSDT_15.json \
        [--window-size 64] [--atr-multiplier 3.0] \
        [--n-clusters 8] [--fit-windows 20000] \
        [--db data/registry/trade_exp_12.db --job-id 42]
"""

from __future__ import annotations

import argparse
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
    add_reversal_labels,
    build_windows,
)
from trade_exp_12.model.transformer_v2 import MarketMAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


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


def _features(frame: pd.DataFrame, atr_multiplier: float) -> pd.DataFrame:
    labeled = add_reversal_labels(frame, atr_multiplier)
    return labeled[labeled["feature_ready"]].reset_index(drop=True)


def _windows(frame: pd.DataFrame, window_size: int) -> np.ndarray:
    feats = frame[FEATURE_COLUMNS].values.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    dummy = np.zeros(len(feats), dtype=np.int64)
    windows, _ = build_windows(feats, dummy, window_size)
    return windows


def _encode(model: MarketMAE, windows: np.ndarray, device: torch.device) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), 256):
            batch = torch.from_numpy(windows[i:i + 256]).float().to(device)
            out.append(model.encode(batch).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--pretrain-report", required=True)
    parser.add_argument("--parquets", required=True, nargs="+")
    parser.add_argument("--target-parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--atr-multiplier", type=float, default=3.0)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--fit-windows", type=int, default=20000)
    parser.add_argument("--db", default=None)
    parser.add_argument("--job-id", type=int, default=None)
    args = parser.parse_args()

    _update_job(args.db, args.job_id, "running", 0.05, "Loading encoder")

    with open(args.pretrain_report) as f:
        pt_report = json.load(f)
    hp = pt_report["hyperparameters"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketMAE(
        n_features=pt_report["info"]["feature_count"],
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        n_encoder_layers=hp["n_encoder_layers"],
        n_decoder_layers=hp["n_decoder_layers"],
        d_ff=hp["d_ff"],
        dropout=0.0,
        max_len=hp["window_size"],
        mask_ratio=0.0,
    ).to(device)
    state = torch.load(args.encoder_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    log.info("Loaded encoder: d_model=%d", hp["d_model"])

    # --- Step 1: sample windows across all parquets for KMeans fitting ---
    _update_job(args.db, args.job_id, "running", 0.15, "Sampling windows for KMeans")
    rng = np.random.default_rng(42)
    fit_windows = []
    per_pair = max(1000, args.fit_windows // max(len(args.parquets), 1))
    for pq_path in args.parquets:
        pq = Path(pq_path)
        if not pq.exists():
            continue
        frame = _features(pd.read_parquet(pq), args.atr_multiplier)
        w = _windows(frame, args.window_size)
        if len(w) > per_pair:
            w = w[rng.choice(len(w), per_pair, replace=False)]
        fit_windows.append(w)
        log.info("%s: sampled %d windows", pq.parent.parent.name, len(w))
    fit_windows = np.concatenate(fit_windows)

    _update_job(args.db, args.job_id, "running", 0.3, "Encoding fit windows")
    fit_emb = _encode(model, fit_windows, device)

    _update_job(args.db, args.job_id, "running", 0.5, "Fitting KMeans")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    km.fit(fit_emb)
    log.info("KMeans fitted on %d embeddings", len(fit_emb))

    # --- Step 2: encode the target pair fully, assign clusters ---
    _update_job(args.db, args.job_id, "running", 0.6, "Encoding target pair")
    tframe = pd.read_parquet(args.target_parquet)
    tframe = _features(tframe, args.atr_multiplier)
    twindows = _windows(tframe, args.window_size)
    temb = _encode(model, twindows, device)
    tclusters = km.predict(temb)
    log.info("Target: %d bars assigned to clusters", len(tclusters))

    # Cluster of bar = cluster of window whose LAST bar is this bar.
    # Window i covers rows [i, i+window_size-1]; last-bar index = i + window_size - 1.
    ws = args.window_size
    n_bars = len(twindows)
    bars = []
    for i in range(n_bars):
        row = tframe.iloc[i + ws - 1]
        bars.append({
            "open_time": str(row["open_time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "cluster": int(tclusters[i]),
            "label": int(row["label"]),
        })

    # Per-cluster stats on the target pair
    counts = np.bincount(tclusters, minlength=args.n_clusters)
    rev_up = np.bincount(
        tclusters[tframe["label"].values[ws - 1: ws - 1 + n_bars] == 1],
        minlength=args.n_clusters,
    )
    rev_dn = np.bincount(
        tclusters[tframe["label"].values[ws - 1: ws - 1 + n_bars] == 2],
        minlength=args.n_clusters,
    )
    stats = [
        {
            "cluster": c,
            "total": int(counts[c]),
            "reversal_up": int(rev_up[c]),
            "reversal_down": int(rev_dn[c]),
            "pct": round(counts[c] / max(n_bars, 1) * 100, 1),
        }
        for c in range(args.n_clusters)
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "n_clusters": args.n_clusters,
            "n_bars": n_bars,
            "cluster_stats": stats,
            "bars": bars,
        }, f)
    log.info("Saved %d bars to %s", n_bars, out)

    _update_job(args.db, args.job_id, "completed", 1.0,
                f"Cluster map: {n_bars} bars, {args.n_clusters} clusters")


if __name__ == "__main__":
    main()
