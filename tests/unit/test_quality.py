import pandas as pd

from trade_exp_12.data.quality import inspect_candles


def frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(times, utc=True),
            "open": [1.0] * len(times),
            "high": [2.0] * len(times),
            "low": [0.5] * len(times),
            "close": [1.5] * len(times),
            "volume": [10.0] * len(times),
            "turnover": [15.0] * len(times),
        }
    )


def test_quality_report_detects_gap() -> None:
    report = inspect_candles(
        frame(["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"]), "15"
    )

    assert report.valid
    assert report.gaps == 1
    assert report.gap_details[0].missing_candles == 1


def test_quality_report_detects_invalid_ohlc() -> None:
    candles = frame(["2026-01-01T00:00:00Z"])
    candles.loc[0, "low"] = 1.7

    assert not inspect_candles(candles, "15").valid
