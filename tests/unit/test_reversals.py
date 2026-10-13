import pandas as pd

from trade_exp_12.reversals.zigzag import detect_atr_reversals


def test_atr_zigzag_tracks_pivot_and_confirmation_separately() -> None:
    closes = [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0]
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=len(closes), freq="1h", tz="UTC"),
            "high": [value + 0.2 for value in closes],
            "low": [value - 0.2 for value in closes],
            "close": closes,
            "atr": [1.0] * len(closes),
        }
    )

    reversals = detect_atr_reversals(frame, atr_multiplier=1.5)

    assert len(reversals) >= 3
    assert [reversal.direction for reversal in reversals[:3]] == ["down", "up", "down"]
    assert [reversal.pivot_type for reversal in reversals[:3]] == ["high", "low", "high"]
    assert all(reversal.pivot_time < reversal.known_at for reversal in reversals[:3])
    assert all(reversal.bars_to_confirm > 0 for reversal in reversals[:3])


def test_atr_zigzag_rejects_invalid_multiplier() -> None:
    frame = pd.DataFrame(columns=["open_time", "high", "low", "close", "atr"])

    try:
        detect_atr_reversals(frame, atr_multiplier=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected ValueError")
