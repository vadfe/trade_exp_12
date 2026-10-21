import numpy as np
import pandas as pd

from trade_exp_12.model.run_cluster_map import (
    add_interpretations,
    boundary_reversal_stats,
    cluster_feature_stats,
    oos_signal_stats,
    segment_lengths,
    transition_counts,
)


def test_tail_forecast_does_not_see_future():
    import torch

    from trade_exp_12.model.transformer_v2 import MarketMAE

    torch.manual_seed(7)
    model = MarketMAE(
        n_features=3, d_model=8, n_encoder_layers=1, n_decoder_layers=1,
        d_ff=16, dropout=0.0, mask_mode="tail", forecast_bars=2,
    ).eval()
    original = torch.randn(2, 8, 3)
    changed = original.clone()
    changed[:, -2:] = torch.randn(2, 2, 3) * 100
    with torch.no_grad():
        first, _, mask = model.pretrain_forward(original)
        second, _, _ = model.pretrain_forward(changed)
    assert mask[:, -2:].all()
    torch.testing.assert_close(first, second)


def test_segment_lengths_runs():
    cl = np.array([0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    seg_cl, seg_len = segment_lengths(cl)
    assert seg_cl.tolist() == [0, 1, 0, 2]
    assert seg_len.tolist() == [3, 2, 1, 4]


def test_segment_lengths_empty():
    seg_cl, seg_len = segment_lengths(np.array([], dtype=int))
    assert len(seg_cl) == 0 and len(seg_len) == 0


def test_transition_counts_off_diagonal_only():
    cl = np.array([0, 0, 1, 1, 0, 1])
    trans = transition_counts(cl, 3)
    assert trans[0, 1] == 2  # switches at 1->2 and 4->5
    assert trans[1, 0] == 1
    assert trans[1, 1] == 0  # self-transitions not counted
    assert trans.sum() == 3


def test_boundary_reversal_stats_counts_near():
    cl = np.array([0] * 10 + [1] * 10)
    labels = np.zeros(20, dtype=int)
    labels[9] = 1   # at boundary (index 9 -> switch at 9->10)
    labels[12] = 2  # 2 bars after boundary
    labels[18] = 1  # far from boundary
    stats = boundary_reversal_stats(labels, cl, window=3)
    assert stats["reversals_total"] == 3
    assert stats["reversals_near_boundary"] == 2
    assert stats["flicker_pct"] == 0.0
    assert stats["mean_run_all"] == 10.0


def test_boundary_reversal_stats_no_reversals():
    stats = boundary_reversal_stats(np.zeros(10, dtype=int), np.zeros(10, dtype=int), window=3)
    assert stats["reversals_total"] == 0
    assert stats["pct"] == 0.0


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "label": np.zeros(n, dtype=int),
        "range_atr": np.linspace(0.5, 2.0, n),
        "volume": np.full(n, 100.0),
        "relative_volume": np.linspace(0.5, 1.5, n),
        "avg_return_10": np.linspace(-0.001, 0.001, n),
        "position_in_range_20": np.linspace(0, 1, n),
        "trend_slope_10": np.linspace(-0.01, 0.01, n),
        "atr_ratio": np.linspace(0.8, 1.2, n),
        "volume_spike_10": np.linspace(0.5, 2.0, n),
    })


def test_cluster_feature_stats_profiles_and_empty():
    frame = _frame(20)
    clusters = np.array([0] * 10 + [1] * 10)
    bar_idx = np.arange(20)
    stats = cluster_feature_stats(frame, clusters, bar_idx, n_clusters=3)
    assert len(stats) == 3
    assert stats[2]["total"] == 0  # empty cluster handled
    assert stats[0]["total"] == 10
    # cluster 0 has the lower range_atr half
    assert stats[0]["avg_range_atr"] < stats[1]["avg_range_atr"]
    assert stats[0]["run_mean"] == 10.0


def test_add_interpretations_labels_extrema():
    stats = [
        {"cluster": 0, "total": 10, "avg_range_atr": 0.5, "avg_rel_volume": 1.0,
         "avg_ret10_bp": 1.0, "run_mean": 50.0},
        {"cluster": 1, "total": 10, "avg_range_atr": 2.0, "avg_rel_volume": 3.0,
         "avg_ret10_bp": -20.0, "run_mean": 5.0},
    ]
    add_interpretations(stats)
    assert "quiet" in stats[0]["interpretation"]
    assert "most persistent" in stats[0]["interpretation"]
    assert "volatility expansion" in stats[1]["interpretation"]
    assert "high relative volume" in stats[1]["interpretation"]
    assert "downward directional" in stats[1]["interpretation"]
    assert "transition/noise-prone" in stats[1]["interpretation"]


def test_oos_signal_stats_last_segment():
    # 100 bars: first 80 quiet cluster 0, last 20 mixed — cluster 9 has all reversals
    bars = (
        [{"cluster": 0, "label": 0, "open_time": f"2026-01-01T{i:02d}"} for i in range(80)]
        + [{"cluster": 9 if i % 2 else 0, "label": 1 if i % 2 else 0,
            "open_time": f"2026-01-01T{i}"} for i in range(80, 100)]
    )
    res = oos_signal_stats(bars, test_frac=0.2)
    assert res["test_bars"] == 20
    by_cl = {r["cluster"]: r for r in res["rows"]}
    assert by_cl[9]["bars"] == 10
    assert by_cl[9]["reversals"] == 10
    assert by_cl[9]["rate"] == 100.0
    assert by_cl[9]["lift"] > 1.5
    assert by_cl[0]["lift"] == 0.0


def test_oos_signal_stats_empty():
    res = oos_signal_stats([], test_frac=0.2)
    assert res["rows"] == []


def test_add_interpretations_empty():
    stats = [{"cluster": 0, "total": 0}]
    add_interpretations(stats)
    assert "interpretation" not in stats[0]
