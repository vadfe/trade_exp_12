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


def segment_lengths(cluster_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a cluster sequence into (start_cluster, length) runs."""
    n = len(cluster_ids)
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    starts = np.concatenate([[0], np.where(np.diff(cluster_ids) != 0)[0] + 1])
    ends = np.concatenate([starts[1:], [n]])
    return cluster_ids[starts], ends - starts


def transition_counts(cluster_ids: np.ndarray, n_clusters: int) -> np.ndarray:
    """Off-diagonal transition counts between consecutive bars."""
    trans = np.zeros((n_clusters, n_clusters), dtype=int)
    for a, b in zip(cluster_ids[:-1], cluster_ids[1:], strict=False):
        if a != b:
            trans[a, b] += 1
    return trans


def boundary_reversal_stats(
    labels: np.ndarray, cluster_ids: np.ndarray, window: int = 3
) -> dict:
    """Share of reversal labels within ``window`` bars of a cluster boundary."""
    n = len(cluster_ids)
    seg_cl, seg_len = segment_lengths(cluster_ids)
    boundaries = np.where(np.diff(cluster_ids) != 0)[0]
    rev_idx = np.where(labels > 0)[0]
    near = np.zeros(len(rev_idx), dtype=bool)
    if len(boundaries) and len(rev_idx):
        pos = np.searchsorted(boundaries, rev_idx)
        for side in (pos - 1, pos):
            ok = (side >= 0) & (side < len(boundaries))
            near[ok] |= np.abs(boundaries[side[ok]] - rev_idx[ok]) <= window
    return {
        "reversals_total": int(len(rev_idx)),
        "reversals_near_boundary": int(near.sum()),
        "pct": round(float(near.mean() * 100), 1) if len(rev_idx) else 0.0,
        "baseline_pct": round(min(len(boundaries) * (2 * window + 1) / max(n, 1), 1.0) * 100, 1),
        "n_segments": int(len(seg_len)),
        "flicker_pct": round(float((seg_len == 1).mean() * 100), 1) if len(seg_len) else 0.0,
        "mean_run_all": round(float(seg_len.mean()), 1) if len(seg_len) else 0.0,
        "n_clusters_seen": int(len(np.unique(seg_cl))) if len(seg_cl) else 0,
    }


def cluster_feature_stats(
    frame: pd.DataFrame,
    cluster_ids: np.ndarray,
    bar_idx: np.ndarray,
    n_clusters: int,
) -> list[dict]:
    """Per-cluster market-behavior profile from causal features of each bar."""
    n_bars = len(cluster_ids)
    lab = frame["label"].values[bar_idx]
    seg_cl, seg_len = segment_lengths(cluster_ids)
    global_rev = max(float((lab > 0).mean()), 1e-9)

    def col(name: str) -> np.ndarray:
        if name in frame:
            return frame[name].values[bar_idx].astype(float)
        return np.zeros(n_bars)

    cols = {name: col(name) for name in (
        "range_atr", "volume", "relative_volume", "avg_return_10",
        "position_in_range_20", "trend_slope_10", "atr_ratio", "volume_spike_10",
    )}

    stats = []
    for c in range(n_clusters):
        m = cluster_ids == c
        if not m.any():
            stats.append({"cluster": c, "total": 0})
            continue
        c_lens = seg_len[seg_cl == c]
        stats.append({
            "cluster": c,
            "total": int(m.sum()),
            "pct": round(float(m.mean() * 100), 1),
            "reversal_up": int((lab[m] == 1).sum()),
            "reversal_down": int((lab[m] == 2).sum()),
            "reversal_rate": round(float((lab[m] > 0).mean() * 100), 2),
            "global_reversal_lift": round(float((lab[m] > 0).mean() / global_rev), 2),
            "avg_range_atr": round(float(np.nanmean(cols["range_atr"][m])), 3),
            "avg_volume": round(float(np.nanmean(cols["volume"][m])), 0),
            "avg_rel_volume": round(float(np.nanmean(cols["relative_volume"][m])), 3),
            "avg_ret10_bp": round(float(np.nanmean(cols["avg_return_10"][m])) * 1e4, 1),
            "avg_abs_ret10_bp": round(
                float(np.nanmean(np.abs(cols["avg_return_10"][m]))) * 1e4, 1),
            "avg_pos_in_range": round(float(np.nanmean(cols["position_in_range_20"][m])), 3),
            "avg_trend_slope": round(float(np.nanmean(cols["trend_slope_10"][m])), 4),
            "avg_atr_ratio": round(float(np.nanmean(cols["atr_ratio"][m])), 3),
            "avg_vol_spike": round(float(np.nanmean(cols["volume_spike_10"][m])), 3),
            "run_mean": round(float(c_lens.mean()), 1),
            "run_median": float(np.median(c_lens)),
            "run_max": int(c_lens.max()),
            "n_segments": int(len(c_lens)),
        })
    return stats


def add_interpretations(stats: list[dict]) -> None:
    """Attach heuristic state labels derived only from measured extrema."""
    populated = [s for s in stats if s["total"]]
    if not populated:
        return
    quiet = min(populated, key=lambda s: s["avg_range_atr"])["cluster"]
    volatile = max(populated, key=lambda s: s["avg_range_atr"])["cluster"]
    active = max(populated, key=lambda s: s["avg_rel_volume"])["cluster"]
    directional = max(populated, key=lambda s: abs(s["avg_ret10_bp"]))["cluster"]
    persistent = max(populated, key=lambda s: s["run_mean"])["cluster"]
    flickery = min(populated, key=lambda s: s["run_mean"])["cluster"]
    for s in populated:
        labels = []
        c = s["cluster"]
        if c == quiet:
            labels.append("quiet / low-volatility")
        if c == volatile:
            labels.append("volatility expansion")
        if c == active:
            labels.append("high relative volume")
        if c == directional:
            direction = "upward" if s["avg_ret10_bp"] >= 0 else "downward"
            labels.append(f"{direction} directional")
        if c == persistent:
            labels.append("most persistent")
        if c == flickery:
            labels.append("transition/noise-prone")
        s["interpretation"] = ", ".join(labels) or "intermediate state"


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
    bar_idx = np.arange(ws - 1, ws - 1 + n_bars)
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

    # ---- Numeric state analysis -------------------------------------------
    nc = args.n_clusters
    stats = cluster_feature_stats(tframe, tclusters, bar_idx, nc)
    add_interpretations(stats)

    trans = transition_counts(tclusters, nc)
    transitions = [
        {"from": a, "to": b, "count": int(trans[a, b])}
        for a in range(nc) for b in range(nc)
        if trans[a, b] > 0
    ]
    transitions.sort(key=lambda t: t["count"], reverse=True)

    boundary_stats = boundary_reversal_stats(
        tframe["label"].values[bar_idx], tclusters
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "schema_version": 2,
            "n_clusters": nc,
            "n_bars": n_bars,
            "run_metadata": {
                "window_size": args.window_size,
                "atr_multiplier": args.atr_multiplier,
                "n_clusters": nc,
                "fit_windows": args.fit_windows,
                "feature_count": pt_report["info"]["feature_count"],
                "d_model": hp["d_model"],
                "mask_mode": hp.get("mask_mode"),
                "forecast_bars": hp.get("forecast_bars"),
                "best_val_loss": pt_report.get("best_val_loss"),
            },
            "cluster_stats": stats,
            "transitions": transitions[:20],
            "boundary_stats": boundary_stats,
            "bars": bars,
        }, f)
    log.info("Saved %d bars to %s", n_bars, out)

    _update_job(args.db, args.job_id, "completed", 1.0,
                f"Cluster map: {n_bars} bars, {args.n_clusters} clusters")


if __name__ == "__main__":
    main()
