"""Generate embeddings from pretrained encoder and analyze them.

Produces a JSON with embeddings, t-SNE 2D projection, KMeans clusters,
and reversal labels for visualization.

Usage::

    python -m trade_exp_12.model.run_embeddings \
        --encoder-path data/pretrain/RUN_ID/encoder.pt \
        --pretrain-report data/pretrain/RUN_ID/report.json \
        --parquets path1.parquet path2.parquet ... \
        --output data/pretrain/RUN_ID/embeddings.json \
        [--window-size 64] [--atr-multiplier 3.0] \
        [--n-clusters 8] [--max-windows 20000] \
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", required=True)
    parser.add_argument("--pretrain-report", required=True)
    parser.add_argument("--parquets", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--atr-multiplier", type=float, default=3.0)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=20000)
    parser.add_argument("--db", default=None)
    parser.add_argument("--job-id", type=int, default=None)
    args = parser.parse_args()

    _update_job(args.db, args.job_id, "running", 0.05, "Loading encoder")

    # Load pretrain report for architecture params
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
    log.info("Loaded encoder: d_model=%d, layers=%d", hp["d_model"], hp["n_encoder_layers"])

    # Load and prepare data from all parquets
    _update_job(args.db, args.job_id, "running", 0.1, "Loading data")
    all_windows = []
    all_labels = []
    all_pair_ids = []
    pair_names = []

    for pq_path in args.parquets:
        frame = pd.read_parquet(pq_path)
        # Extract pair name from path
        parts = Path(pq_path).parts
        pair_name = "unknown"
        for i, part in enumerate(parts):
            if part in ("spot", "futures", "linear"):
                if i + 1 < len(parts):
                    pair_name = parts[i + 1]
                break
        pair_names.append(pair_name)
        pair_id = len(pair_names) - 1

        frame = add_reversal_labels(frame, args.atr_multiplier)
        ready = frame[frame["feature_ready"]].copy()
        features = ready[FEATURE_COLUMNS].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        labels = ready["label"].values.astype(np.int64)

        windows, window_labels = build_windows(features, labels, args.window_size)
        all_windows.append(windows)
        all_labels.append(window_labels)
        all_pair_ids.append(np.full(len(windows), pair_id, dtype=np.int32))
        log.info("%s: %d windows", pair_name, len(windows))

    windows = np.concatenate(all_windows)
    labels = np.concatenate(all_labels)
    pair_ids = np.concatenate(all_pair_ids)
    log.info("Total: %d windows", len(windows))

    # Subsample if too many
    if len(windows) > args.max_windows:
        rng = np.random.default_rng(42)
        # Stratified: keep all reversals, subsample hold
        rev_mask = labels > 0
        rev_idx = np.where(rev_mask)[0]
        hold_idx = np.where(~rev_mask)[0]
        n_hold = args.max_windows - len(rev_idx)
        if n_hold > 0 and n_hold < len(hold_idx):
            hold_idx = rng.choice(hold_idx, size=n_hold, replace=False)
        idx = np.sort(np.concatenate([rev_idx, hold_idx]))
        windows = windows[idx]
        labels = labels[idx]
        pair_ids = pair_ids[idx]
        log.info("Subsampled to %d windows (%d reversals kept)", len(windows), rev_mask.sum())

    # Generate embeddings
    _update_job(args.db, args.job_id, "running", 0.3, "Computing embeddings")
    embeddings = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i:i + batch_size]).float().to(device)
            emb = model.encode(batch)  # [B, d_model]
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings)  # [N, d_model]
    log.info("Embeddings shape: %s", embeddings.shape)

    # t-SNE
    _update_job(args.db, args.job_id, "running", 0.5, "Computing t-SNE")
    from sklearn.manifold import TSNE
    log.info("Running t-SNE on %d points...", len(embeddings))
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                random_state=42, n_iter=1000)
    coords_2d = tsne.fit_transform(embeddings)
    log.info("t-SNE done")

    # KMeans clustering
    _update_job(args.db, args.job_id, "running", 0.8, "Clustering")
    from sklearn.cluster import KMeans
    n_clusters = min(args.n_clusters, len(embeddings))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(embeddings)

    # Analyze clusters: what fraction of each cluster is reversal?
    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        total = int(mask.sum())
        rev_up = int((labels[mask] == 1).sum())
        rev_down = int((labels[mask] == 2).sum())
        hold = total - rev_up - rev_down
        cluster_stats.append({
            "cluster": c,
            "total": total,
            "hold": hold,
            "reversal_up": rev_up,
            "reversal_down": rev_down,
            "reversal_pct": round((rev_up + rev_down) / max(total, 1) * 100, 2),
        })
    cluster_stats.sort(key=lambda x: x["reversal_pct"], reverse=True)

    # Save results
    _update_job(args.db, args.job_id, "running", 0.9, "Saving results")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_names = ["hold", "reversal_up", "reversal_down"]
    result = {
        "n_points": len(embeddings),
        "n_features": int(embeddings.shape[1]),
        "n_clusters": n_clusters,
        "pair_names": pair_names,
        "cluster_stats": cluster_stats,
        "points": {
            "x": [round(float(v), 4) for v in coords_2d[:, 0]],
            "y": [round(float(v), 4) for v in coords_2d[:, 1]],
            "label": [label_names[int(lbl)] for lbl in labels],
            "cluster": [int(c) for c in cluster_ids],
            "pair": [pair_names[int(p)] for p in pair_ids],
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f)
    log.info("Saved %d points to %s", len(embeddings), output_path)

    _update_job(args.db, args.job_id, "completed", 1.0,
                f"Embeddings: {len(embeddings)} points, {n_clusters} clusters")
    log.info("Done")


if __name__ == "__main__":
    main()
