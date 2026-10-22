import httpx
import pandas as pd
import pytest

from trade_exp_12.config import MoexConfig
from trade_exp_12.data.moex import (
    TIMEFRAME_TO_ISS,
    MoexClient,
    parse_candles,
    validate_time_metadata,
)

# ISS columns order: open, close, high, low, value, volume, begin, end
ROWS = [
    [100.0, 101.0, 102.5, 99.5, 1_000_000.0, 5000,
     "2026-10-01 10:00:00", "2026-10-01 11:00:00"],
    [101.0, 100.5, 101.5, 100.0, 900_000.0, 4500,
     "2026-10-01 11:00:00", "2026-10-01 12:00:00"],
]


def test_parse_candles_maps_columns():
    frame = parse_candles(ROWS)
    assert list(frame.columns) == [
        "open_time", "open", "high", "low", "close", "volume", "turnover",
        "source_end_time", "available_at", "moex_time_schema"]
    assert len(frame) == 2
    assert frame.iloc[0]["open"] == 100.0
    assert frame.iloc[0]["close"] == 101.0
    assert frame.iloc[0]["high"] == 102.5
    assert frame.iloc[0]["low"] == 99.5
    assert frame.iloc[0]["volume"] == 5000
    assert frame.iloc[0]["turnover"] == 1_000_000.0
    assert frame.iloc[0]["open_time"] == pd.Timestamp("2026-10-01 07:00:00", tz="UTC")
    assert frame.iloc[0]["source_end_time"] == pd.Timestamp("2026-10-01 08:00:00", tz="UTC")
    assert frame.iloc[0]["available_at"] > frame.iloc[0]["source_end_time"]


def test_invalid_volume_is_not_silently_replaced_with_zero():
    row = list(ROWS[0])
    row[5] = "broken"
    with pytest.raises(ValueError):
        parse_candles([row])
    row[5] = None
    assert pd.isna(parse_candles([row]).iloc[0]["volume"])


def test_parse_candles_empty():
    frame = parse_candles([])
    assert frame.empty


def test_parse_candles_dedupes_and_sorts():
    rows = ROWS[::-1] + [ROWS[0]]
    frame = parse_candles(rows)
    assert len(frame) == 2
    assert frame.iloc[0]["open_time"] < frame.iloc[1]["open_time"]


def test_timeframe_map_has_hour():
    assert TIMEFRAME_TO_ISS["60"] == 60


@pytest.mark.asyncio
async def test_download_filters_unfinished_and_delayed_candles():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"candles": {"data": ROWS}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="https://iss.moex.com/iss"
    ) as client:
        downloader = MoexClient(MoexConfig(), client)
        start = int(pd.Timestamp("2026-09-30 22:00:00Z").timestamp() * 1000)
        cutoff = int(pd.Timestamp("2026-10-01 08:15:01Z").timestamp() * 1000)
        frame = await downloader.download("SBER", "60", start, cutoff)
        assert len(frame) == 1
        assert frame.iloc[0]["open_time"] == pd.Timestamp("2026-10-01 07:00:00Z")
        validate_time_metadata(frame, "60")
        empty = await downloader.download("SBER", "60", start, cutoff - 1000)
        assert empty.empty
        assert requests[0].url.params["from"] == "2026-10-01"


def test_metadata_rejects_legacy_and_tampered_availability():
    with pytest.raises(ValueError, match="Старая схема"):
        validate_time_metadata(parse_candles(ROWS), "60")
    frame = parse_candles(ROWS)
    frame["downloaded_at"] = pd.Timestamp("2026-10-02 00:00:00Z")
    validate_time_metadata(frame, "60")
    frame.loc[0, "available_at"] = frame.loc[0, "open_time"]
    with pytest.raises(ValueError, match="метаданные"):
        validate_time_metadata(frame, "60")


def test_reported_partial_end_does_not_close_the_full_hour():
    rows = [[100, 101, 102, 99, 1000, 10, "2026-10-01 10:00:00", "2026-10-01 10:05:00"]]
    frame = parse_candles(rows)
    assert frame.iloc[0]["available_at"] == pd.Timestamp("2026-10-01 08:15:00Z")


def test_daily_end_is_not_available_before_next_midnight():
    rows = [[100, 101, 102, 99, 1000, 10, "2026-10-01 00:00:00", "2026-10-01 18:40:00"]]
    frame = parse_candles(rows, "D")
    assert frame.iloc[0]["available_at"] == pd.Timestamp("2026-10-01 21:15:00Z")
