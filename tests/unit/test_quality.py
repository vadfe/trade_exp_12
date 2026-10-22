import pandas as pd

from trade_exp_12.data.quality import audit_moex_history, inspect_candles


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


def test_quality_rejects_nonfinite_and_nonpositive_prices():
    for value in [float("nan"), float("inf"), 0.0, -1.0]:
        candles = frame(["2026-01-01T00:00:00Z"])
        candles.loc[0, "close"] = value
        assert inspect_candles(candles, "15").invalid_ohlc == 1
    candles = frame(["2026-01-01T00:00:00Z"])
    candles.loc[0, "volume"] = float("nan")
    assert inspect_candles(candles, "15").invalid_volume == 1


def test_history_audit_reports_missing_dates_not_calendar_guesses():
    hourly = frame(["2025-01-01T07:00Z", "2025-01-03T07:00Z"])
    daily = frame(["2025-01-01T00:00Z", "2025-01-02T00:00Z", "2025-01-03T00:00Z"])
    report = audit_moex_history(hourly, daily)
    assert report["missing_trading_dates"] == ["2025-01-02"]
    assert report["status"] == "needs_review"
    assert report["ohlc_disagreements"] == []
    assert len(report["reference_candles"]) == 3


def test_history_audit_keeps_ohlc_disagreements_visible():
    hourly = frame(["2025-01-01T07:00Z", "2025-01-02T07:00Z"])
    daily = frame(["2025-01-01T00:00Z", "2025-01-02T00:00Z"])
    daily.loc[1, "close"] = 1.7
    report = audit_moex_history(hourly, daily)
    assert report["ohlc_disagreements"][0]["date"] == "2025-01-02"
    assert report["ohlc_disagreements"][0]["hourly"]["close"] == 1.5
    assert report["ohlc_disagreements"][0]["daily"]["close"] == 1.7
    assert report["status"] == "needs_review"


def test_history_audit_links_known_pause_but_does_not_approve_it():
    hourly = frame(["2024-02-13T10:00Z", "2024-02-13T12:00Z"])
    daily = frame(["2024-02-13T00:00Z"])
    report = audit_moex_history(hourly, daily)
    assert len(report["intraday_gaps"]) == 1
    assert "https://www.moex.com/n67529" in report["intraday_gaps"][0]["halt_sources"]
    assert report["unreferenced_intraday_gaps"] == 0
    assert report["status"] == "needs_review"
