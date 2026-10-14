"""Tests for model dataset assembly and transformer forward pass."""

import numpy as np
import pandas as pd
import torch

from trade_exp_12.features.relative import add_relative_features
from trade_exp_12.model.dataset import (
    FEATURE_COLUMNS,
    NUM_CLASSES,
    add_reversal_labels,
    build_windows,
    prepare_datasets,
    temporal_split,
)
from trade_exp_12.model.transformer import MarketTransformer


def _make_frame(n: int = 200) -> pd.DataFrame:
    """Create a synthetic processed dataframe with all required columns."""
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    openp = close + rng.randn(n) * 0.1
    volume = rng.uniform(1000, 5000, n)

    frame = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": openp,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "turnover": volume * close,
    })
    # Use real feature pipeline
    frame = add_relative_features(frame)
    # SMC features (synthetic)
    frame["smc_swing_high_level"] = high
    frame["smc_swing_low_level"] = low
    frame["smc_resistance_distance_atr"] = rng.randn(n) * 0.5
    frame["smc_support_distance_atr"] = rng.randn(n) * 0.5
    frame["smc_bos"] = 0
    frame["smc_choch"] = 0
    frame["smc_fvg_signal"] = 0
    frame["smc_ob_signal"] = 0
    frame["smc_liquidity_signal"] = 0
    frame["smc_active_bullish_fvg_count"] = 0
    frame["smc_active_bearish_fvg_count"] = 0
    frame["smc_active_bullish_ob_count"] = 0
    frame["smc_active_bearish_ob_count"] = 0
    return frame


def test_add_reversal_labels():
    frame = _make_frame(300)
    labeled = add_reversal_labels(frame, atr_multiplier=1.5)
    assert "label" in labeled.columns
    unique_labels = set(labeled["label"].unique())
    assert unique_labels.issubset({0, 1, 2})
    assert len(labeled) == 300


def test_build_windows():
    n_features = 5
    n_rows = 100
    features = np.random.randn(n_rows, n_features).astype(np.float32)
    labels = np.random.randint(0, 3, n_rows).astype(np.int64)
    window_size = 16

    windows, window_labels = build_windows(features, labels, window_size)
    expected_count = n_rows - window_size + 1
    assert windows.shape == (expected_count, window_size, n_features)
    assert window_labels.shape == (expected_count,)
    # Label corresponds to last bar in window
    assert window_labels[0] == labels[window_size - 1]
    assert window_labels[-1] == labels[n_rows - 1]


def test_temporal_split():
    train_r, val_r, test_r = temporal_split(1000)
    assert train_r.start == 0
    assert train_r.stop == 700
    assert val_r.start == 700
    assert val_r.stop == 850
    assert test_r.start == 850
    assert test_r.stop == 1000


def test_prepare_datasets():
    frame = _make_frame(500)
    train_ds, val_ds, test_ds, info = prepare_datasets(
        frame, window_size=32, atr_multiplier=1.5
    )
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    assert info.feature_count == len(FEATURE_COLUMNS)
    assert info.window_size == 32
    # Check shapes
    x, y = train_ds[0]
    assert x.shape == (32, len(FEATURE_COLUMNS))
    assert y.shape == ()
    assert y.dtype == torch.long


def test_transformer_forward():
    n_features = len(FEATURE_COLUMNS)
    model = MarketTransformer(
        n_features=n_features, num_classes=NUM_CLASSES,
        d_model=32, n_heads=4, n_layers=2, d_ff=64,
    )
    batch = torch.randn(4, 32, n_features)
    logits = model(batch)
    assert logits.shape == (4, NUM_CLASSES)

    emb = model.embedding(batch)
    assert emb.shape == (4, 32)  # d_model


def test_transformer_backward():
    n_features = len(FEATURE_COLUMNS)
    model = MarketTransformer(
        n_features=n_features, num_classes=NUM_CLASSES,
        d_model=32, n_heads=4, n_layers=2, d_ff=64,
    )
    batch = torch.randn(4, 32, n_features)
    labels = torch.randint(0, NUM_CLASSES, (4,))
    logits = model(batch)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    # Check gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            break
